"""
train.py – REINFORCE agent training on LotoFootEnv.

Usage:
    uv run python train.py
    uv run python train.py --val-ratio 0.2 --test-ratio 0.2 --k-grids 30 --episodes 8000
    uv run python train.py --help
"""

from __future__ import annotations

import argparse
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from env import LotoFootEnv, N_OUTCOMES, N_FEATURES, COST_PER_GRID

EXCEL_FILE = (
    "loto-foot-8_grilles-2026-01-31_au_2026-03-04_grille-joueur_top50"
    "_rang-2026-02-19_au_2026-03-21_train.xlsx"
)

# Prize ranks in the dataset — ordered from best to worst
RANG_COLS = ["rang1", "rang2", "rang3", "rang4"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _outcome(score: str) -> int:
    h, a = map(int, score.split("-"))
    return 0 if h > a else (1 if h == a else 2)



def _player_consensus(df: pd.DataFrame, gdf: pd.DataFrame) -> list[float]:
    """
    For each match in gdf, compute the fraction of top players picking each outcome.

    Returns a flat list of n_matches * 3 values interleaved as:
        [p1_m0, pN_m0, p2_m0,  p1_m1, pN_m1, p2_m1, ...]

    Player columns are identified by the presence of a '{player} Résultat' column.
    Each player's prediction is binary (0/1 per outcome); the fraction is the mean.
    """
    player_ids = [col[: -len(" Résultat")]
                  for col in df.columns if col.endswith(" Résultat")]
    if not player_ids:
        return []

    cols1 = [f"{p} 1" for p in player_ids if f"{p} 1" in gdf.columns]
    colsN = [f"{p} N" for p in player_ids if f"{p} N" in gdf.columns]
    cols2 = [f"{p} 2" for p in player_ids if f"{p} 2" in gdf.columns]

    p1 = gdf[cols1].values.mean(axis=1) if cols1 else np.zeros(len(gdf))
    pN = gdf[colsN].values.mean(axis=1) if colsN else np.zeros(len(gdf))
    p2 = gdf[cols2].values.mean(axis=1) if cols2 else np.zeros(len(gdf))

    result: list[float] = []
    for i in range(len(gdf)):
        result.extend([float(p1[i]), float(pN[i]), float(p2[i])])
    return result


def load_data(
    path: str = EXCEL_FILE,
    loto_type: str | None = None,
    extra_features: list[str] | None = None,
) -> list[dict]:
    """
    extra_features : optional list of feature groups to append to each grid.
        Supported values:
          "player_consensus" — 3 extra features per match: fraction of top-50
                               players who selected outcome 1 / N / 2.
    """
    """
    Load all complete rounds from the Excel file.

    Parameters
    ----------
    loto_type : e.g. 'loto-foot-8'. If None, loads all types.
    """
    df = pd.read_excel(path)
    if loto_type is not None:
        df = df[df["loto_type_name"] == loto_type]

    grids = []
    for gid, gdf in df.groupby("grid_index"):
        gdf = gdf.sort_values("match_index").reset_index(drop=True)
        if gdf["Score"].isna().any():
            continue  # skip incomplete rounds

        n_matches = len(gdf)
        ltype = str(gdf["loto_type_name"].iloc[0])

        features_raw = []
        outcomes     = []
        match_info   = []
        for _, row in gdf.iterrows():
            features_raw.extend([
                row["Cote 1"], row["Cote 2"], row["Cote 3"],
                row["Rep 1"],  row["Rep 2"],  row["Rep 3"],
            ])
            outcomes.append(_outcome(row["Score"]))
            match_info.append({
                "home":  str(row.get("home_team", "?")),
                "away":  str(row.get("away_team", "?")),
                "score": str(row.get("Score", "")),
            })

        # Build prize dict: {n_correct → prize_amount}, skip zero-prize ranks
        # rang1 → n_matches correct, rang2 → n_matches-1 correct, etc.
        prizes: dict[int, float] = {}
        for i, col in enumerate(RANG_COLS):
            if col not in gdf.columns:
                break
            val = float(gdf[col].iloc[0])
            if val > 0:
                prizes[n_matches - i] = val

        grid: dict = {
            "grid_index":   gid,
            "date":         gdf["date"].iloc[0],
            "loto_type":    ltype,
            "n_matches":    n_matches,
            "features_raw": features_raw,
            "outcomes":     np.array(outcomes, dtype=np.int32),
            "prizes":       prizes,
            "match_info":   match_info,
        }
        if extra_features:
            if "player_consensus" in extra_features:
                grid["extra_features_raw"] = _player_consensus(df, gdf)
        grids.append(grid)

    grids.sort(key=lambda x: x["date"])

    return grids


def split_data(
    grids: list[dict],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,  # unused — split is always chronological, not random
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Chronological split: oldest rounds → train, most recent → val → test.

    Grids are assumed to be pre-sorted by date (load_data guarantees this).
    Chronological split correctly simulates deployment: the agent is always
    evaluated on rounds it could not have seen during training, matching how
    a real bettor would use the model.
    """
    n = len(grids)
    n_test  = max(1, round(n * test_ratio))
    n_val   = max(1, round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    return (
        grids[:n_train],
        grids[n_train : n_train + n_val],
        grids[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# REINFORCE agent  (PyTorch policy, system-bet / binary mask)
# ---------------------------------------------------------------------------

class _PolicyNet(nn.Module):
    """Linear or one-hidden-layer MLP policy network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.net[0].bias)
            nn.init.normal_(self.net[2].weight, std=math.sqrt(2.0 / hidden_dim))
            nn.init.zeros_(self.net[2].bias)
        else:
            self.net = nn.Linear(obs_dim, action_dim)
            nn.init.normal_(self.net.weight, std=0.01)
            nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent:
    """
    REINFORCE policy agent backed by PyTorch autograd.

    Supports linear (hidden_dim=0) and one-hidden-layer MLP (hidden_dim>0).
    Action: (n_matches, 3) independent Bernoulli — system bet with k_max cap.
    Updated via REINFORCE + entropy regularisation using Adam.
    """

    def __init__(
        self,
        n_matches: int,
        k_max: int = 20,
        lr: float = 0.005,
        entropy_coef: float = 0.05,
        seed: int = 42,
        hidden_dim: int = 0,
        obs_dim: int | None = None,   # override when extra features are used
    ):
        self.n_matches    = n_matches
        self.k_max        = k_max
        self.obs_dim      = obs_dim if obs_dim is not None else n_matches * N_FEATURES
        self.action_dim   = n_matches * N_OUTCOMES
        self.lr           = lr
        self.entropy_coef = entropy_coef
        self.hidden_dim   = hidden_dim

        torch.manual_seed(seed)
        self.policy    = _PolicyNet(self.obs_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self._rng      = np.random.default_rng(seed)
        self._reward_baseline = -1.0  # EMA baseline: -1 = break-even per combo (cost only)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def get_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return (n_matches, 3) sigmoid probabilities for an observation."""
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            logits = self.policy(x)
            probs  = torch.sigmoid(logits).numpy()
        return probs.reshape(self.n_matches, N_OUTCOMES)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Return a flat (n_matches * 3) float32 binary mask.

        Training: sample each outcome independently via Bernoulli(sigmoid(logit)).
        Deterministic: threshold at 0.5.
        Ensures at least 1 outcome per match, then prunes to k_max combos.
        """
        probs = self.get_probs(obs)

        if deterministic:
            mask = (probs > 0.5).astype(np.float32)
        else:
            rand = self._rng.random((self.n_matches, N_OUTCOMES))
            mask = (rand < probs).astype(np.float32)

        for m in range(self.n_matches):
            if not mask[m].any():
                mask[m, int(probs[m].argmax())] = 1.0

        while math.prod(int(mask[m].sum()) for m in range(self.n_matches)) > self.k_max:
            best_m, best_o, best_val = -1, -1, np.inf
            for m in range(self.n_matches):
                sel = [o for o in range(N_OUTCOMES) if mask[m, o] > 0.5]
                if len(sel) <= 1:
                    continue
                for o in sel:
                    if probs[m, o] < best_val:
                        best_val, best_m, best_o = probs[m, o], m, o
            if best_m == -1:
                break
            mask[best_m, best_o] = 0.0

        return mask.flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        per_combo_rewards: np.ndarray,  # (n_combos,) from info dict
        selections: np.ndarray,         # (n_matches, 3) bool from info dict
    ) -> None:
        """REINFORCE update via autograd: log-prob gradient + entropy bonus."""
        raw_reward = float(per_combo_rewards.mean())
        advantage  = raw_reward - self._reward_baseline
        self._reward_baseline += 0.05 * (raw_reward - self._reward_baseline)

        x      = torch.tensor(obs, dtype=torch.float32)
        logits = self.policy(x)
        probs  = torch.sigmoid(logits).reshape(self.n_matches, N_OUTCOMES)

        sel_t = torch.tensor(selections, dtype=torch.float32)
        # Bernoulli log-prob: b*log(p) + (1-b)*log(1-p)
        log_prob = (
            sel_t * torch.log(probs.clamp(1e-8, 1 - 1e-8))
            + (1 - sel_t) * torch.log((1 - probs).clamp(1e-8, 1 - 1e-8))
        ).sum()

        # Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)
        p_c     = probs.clamp(1e-7, 1 - 1e-7)
        entropy = -(p_c * torch.log(p_c) + (1 - p_c) * torch.log(1 - p_c)).sum()

        n    = max(len(per_combo_rewards), 1)
        loss = -(advantage * log_prob + self.entropy_coef * entropy) / n

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save weights and hyperparameters to a .npz file."""
        data: dict = {
            "n_matches":    np.int32(self.n_matches),
            "k_max":        np.int32(self.k_max),
            "lr":           np.float32(self.lr),
            "entropy_coef": np.float32(self.entropy_coef),
            "hidden_dim":   np.int32(self.hidden_dim),
        }
        for k, v in self.policy.state_dict().items():
            data[f"p_{k}"] = v.numpy()
        np.savez(path, **data)
        print(f"Agent saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "REINFORCEAgent":
        """Load agent from a .npz file (new torch format or legacy numpy format)."""
        data = np.load(path)
        keys = set(data.keys())

        # ── New torch format ─────────────────────────────────────────────
        if "hidden_dim" in keys and any(k.startswith("p_") for k in keys):
            agent = cls(
                n_matches=int(data["n_matches"]),
                k_max=int(data["k_max"]),
                lr=float(data["lr"]),
                entropy_coef=float(data["entropy_coef"]),
                hidden_dim=int(data["hidden_dim"]),
            )
            sd = {k[2:]: torch.tensor(data[k]) for k in keys if k.startswith("p_")}
            agent.policy.load_state_dict(sd)
            return agent

        # ── Legacy numpy format (linear: W/b, mlp: W1/b1/W2/b2) ─────────
        k_max = int(data["k_max"]) if "k_max" in keys else 20
        if "W1" in keys:
            hidden_dim = int(data["hidden_dim"]) if "hidden_dim" in keys else int(data["W1"].shape[0])
            agent = cls(
                n_matches=int(data["n_matches"]), k_max=k_max,
                lr=float(data["lr"]), entropy_coef=float(data["entropy_coef"]),
                hidden_dim=hidden_dim,
            )
            sd = agent.policy.state_dict()
            sd["net.0.weight"] = torch.tensor(data["W1"])
            sd["net.0.bias"]   = torch.tensor(data["b1"])
            sd["net.2.weight"] = torch.tensor(data["W2"])
            sd["net.2.bias"]   = torch.tensor(data["b2"])
        else:
            agent = cls(
                n_matches=int(data["n_matches"]), k_max=k_max,
                lr=float(data["lr"]), entropy_coef=float(data["entropy_coef"]),
                hidden_dim=0,
            )
            sd = agent.policy.state_dict()
            sd["net.weight"] = torch.tensor(data["W"])
            sd["net.bias"]   = torch.tensor(data["b"])
        agent.policy.load_state_dict(sd)
        return agent


class MLPREINFORCEAgent(REINFORCEAgent):
    """Backward-compatible alias: MLP agent with explicit hidden_dim argument."""

    def __init__(
        self,
        n_matches: int,
        hidden_dim: int = 64,
        k_max: int = 20,
        lr: float = 0.005,
        entropy_coef: float = 0.05,
        seed: int = 42,
        obs_dim: int | None = None,
    ):
        super().__init__(
            n_matches=n_matches, k_max=k_max, lr=lr,
            entropy_coef=entropy_coef, seed=seed, hidden_dim=hidden_dim,
            obs_dim=obs_dim,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _collect_results(env: LotoFootEnv, get_action) -> dict:
    """Run one full pass over env, collecting stats."""
    env.restart_eval()
    n_rounds = len(env.grids)
    n_matches = env.n_matches
    total_grids = 0
    total_cost = total_earnings = 0.0
    hits: dict[str, int] = {}   # aggregated across all rounds
    rounds_with_hit = 0

    while not env.eval_done():
        obs, _ = env.reset()
        action = get_action(obs)
        _, _, _, _, info = env.step(action)

        total_grids    += info["n_combos"]
        total_cost     += info["total_cost"]
        total_earnings += info["total_earnings"]
        round_total_hits = 0
        for rang, count in info["hits"].items():
            hits[rang] = hits.get(rang, 0) + count
            round_total_hits += count
        if round_total_hits > 0:
            rounds_with_hit += 1

    total_hits = sum(hits.values())
    return {
        "n_matches": n_matches,
        "n_rounds": n_rounds,
        "total_grids": total_grids,
        "total_cost": total_cost,
        "total_earnings": total_earnings,
        "net": total_earnings - total_cost,
        "hits": hits,
        "total_hits": total_hits,
        "rounds_with_hit": rounds_with_hit,
        "grid_hit_rate": total_hits / max(total_grids, 1),
        "round_hit_rate": rounds_with_hit / max(n_rounds, 1),
    }


def _print_results(label: str, r: dict) -> None:
    n = r["n_matches"]
    k = r["total_grids"] // r["n_rounds"] if r["n_rounds"] > 0 else 0
    cost_per_round = r["total_cost"] / r["n_rounds"]
    earn_per_round = r["total_earnings"] / r["n_rounds"]
    net_per_round  = r["net"] / r["n_rounds"]

    # Build per-rank lines dynamically
    rank_lines = []
    for i, (rang, count) in enumerate(sorted(r["hits"].items())):
        correct = n - i
        rank_lines.append(
            f"  {rang.capitalize()} winning grids : {count}  ({correct}/{n} correct)"
        )

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Grid type            : loto-foot-{n}")
    print(f"  Rounds played        : {r['n_rounds']}")
    print(f"  Grids/round (avg)    : {k}  (${cost_per_round:.2f}/round)")
    print(f"  Total grids played   : {r['total_grids']}")
    print()
    for line in rank_lines:
        print(line)
    print(f"  Grid hit rate        : {r['grid_hit_rate']*100:.3f}%"
          f"  ({r['total_hits']}/{r['total_grids']} grids paid)")
    print(f"  Round hit rate       : {r['round_hit_rate']*100:.1f}%"
          f"  ({r['rounds_with_hit']}/{r['n_rounds']} rounds had ≥1 payout)")
    print()
    print(f"  Avg earnings/round   : ${earn_per_round:.2f}")
    print(f"  Avg cost/round       : ${cost_per_round:.2f}")
    print(f"  Avg net/round        : ${net_per_round:+.2f}")
    print(f"  Total net P&L        : ${r['net']:+.2f}")
    print(f"{'─'*50}")


def evaluate(agent: REINFORCEAgent, env: LotoFootEnv, label: str) -> None:
    r = _collect_results(env, lambda obs: agent.act(obs, deterministic=False))
    _print_results(label, r)


def random_baseline(env: LotoFootEnv) -> None:
    rng = np.random.default_rng(0)
    r = _collect_results(
        env,
        lambda obs: rng.uniform(0, 1, size=env.action_dim).astype(np.float32),
    )
    _print_results("RANDOM BASELINE", r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Loto Foot RL strategy trainer")
    parser.add_argument("--val-ratio",  type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of data for test (default: 0.2)")
    parser.add_argument("--k-grids",    type=int, default=20,
                        help="Max grids (combos) per round, max 50 (default: 20)")
    parser.add_argument("--episodes",   type=int, default=6000,
                        help="Training episodes (default: 6000)")
    parser.add_argument("--lr",           type=float, default=0.005,
                        help="Learning rate (default: 0.005)")
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy regularisation coefficient (default: 0.05)")
    parser.add_argument("--loto-type", type=str, default=None,
                        help="Filter by grid type, e.g. loto-foot-8 (default: auto-detect)")
    parser.add_argument("--save", type=str, default=None, metavar="PATH",
                        help="Save trained agent to this .npz file after training")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--extra-features", type=str, default="",
                        help="Comma-separated extra feature groups to load. "
                             "Supported: player_consensus")
    args = parser.parse_args()

    k = min(max(args.k_grids, 1), 50)
    extra_features = [f.strip() for f in args.extra_features.split(",") if f.strip()]

    # ---- Data ---------------------------------------------------------------
    all_grids = load_data(loto_type=args.loto_type, extra_features=extra_features or None)
    if not all_grids:
        raise ValueError(f"No complete grids found (loto_type={args.loto_type!r})")

    train_grids, val_grids, test_grids = split_data(
        all_grids, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    n_matches = all_grids[0]["n_matches"]
    ltype = all_grids[0]["loto_type"]
    print(f"Type={ltype}  Rounds total={len(all_grids)}  "
          f"train={len(train_grids)}  val={len(val_grids)}  test={len(test_grids)}")
    print(f"k_max={k}  episodes={args.episodes}  lr={args.lr}  entropy={args.entropy_coef}")

    # ---- Environments -------------------------------------------------------
    train_env      = LotoFootEnv(train_grids, k_max=k, mode="train")
    eval_train_env = LotoFootEnv(train_grids, k_max=k, mode="eval")
    eval_val_env   = LotoFootEnv(val_grids,   k_max=k, mode="eval")
    eval_test_env  = LotoFootEnv(test_grids,  k_max=k, mode="eval")

    # ---- Agent --------------------------------------------------------------
    agent = REINFORCEAgent(n_matches=n_matches, k_max=k, lr=args.lr,
                           entropy_coef=args.entropy_coef, seed=args.seed,
                           obs_dim=train_env.obs_dim)

    # ---- Training loop ------------------------------------------------------
    reward_window: list[float] = []
    obs, _ = train_env.reset(seed=args.seed)

    for ep in range(1, args.episodes + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, _, info = train_env.step(action)

        agent.update(obs, action, info["per_combo_rewards"], info["selections"])
        reward_window.append(reward)

        obs, _ = train_env.reset()

        if ep % 1000 == 0:
            avg = float(np.mean(reward_window[-1000:]))
            print(f"  ep {ep:6d} | avg reward/grid: {avg:+.4f}")

    # ---- Evaluation ---------------------------------------------------------
    evaluate(agent, eval_train_env, f"TRAIN  ({len(train_grids)} rounds)")
    evaluate(agent, eval_val_env,   f"VAL    ({len(val_grids)} rounds)")
    evaluate(agent, eval_test_env,  f"TEST   ({len(test_grids)} rounds)")
    random_baseline(eval_test_env)

    # ---- Save ---------------------------------------------------------------
    if args.save:
        agent.save(args.save)


if __name__ == "__main__":
    main()
