"""
train.py – REINFORCE agent training on LotoFootEnv.

Usage:
    uv run python train.py
    uv run python train.py --val-ratio 0.2 --test-ratio 0.2 --k-grids 30 --episodes 8000
    uv run python train.py --help
"""

from __future__ import annotations

import argparse
import os
import re
import numpy as np
import pandas as pd

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


def parse_cutoff_date(path: str) -> str | None:
    """
    Extract the training cutoff date from the filename.

    The convention is: the last occurrence of 'au_YYYY-MM-DD' in the filename
    gives the maximum allowed grid date for training.

    Example:
        loto-foot-8_..._rang-2026-02-19_au_2026-03-21_train.xlsx
        → cutoff = '2026-03-21'
    """
    basename = os.path.basename(path)
    matches = re.findall(r'au_(\d{4}-\d{2}-\d{2})', basename)
    return matches[-1] if matches else None


def load_data(
    path: str = EXCEL_FILE,
    loto_type: str | None = None,
    cutoff_date: str | None = "auto",
) -> list[dict]:
    """
    Load all complete rounds from the Excel file.

    Parameters
    ----------
    loto_type    : e.g. 'loto-foot-8'. If None, loads all types.
    cutoff_date  : Only include grids with date ≤ cutoff_date.
                   'auto' (default) parses the cutoff from the filename.
                   None disables filtering.
    """
    if cutoff_date == "auto":
        cutoff_date = parse_cutoff_date(path)

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

        grids.append({
            "grid_index":  gid,
            "date":        gdf["date"].iloc[0],
            "loto_type":   ltype,
            "n_matches":   n_matches,
            "features_raw": features_raw,
            "outcomes":    np.array(outcomes, dtype=np.int32),
            "prizes":      prizes,
            "match_info":  match_info,
        })

    grids.sort(key=lambda x: x["date"])

    if cutoff_date is not None:
        before = len(grids)
        grids = [g for g in grids if g["date"] <= cutoff_date]
        print(f"Cutoff date: {cutoff_date}  ({before} → {len(grids)} rounds kept)")

    return grids


def split_data(
    grids: list[dict],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Randomly split grids into train / val / test sets.

    Returns (train_grids, val_grids, test_grids).
    """
    rng = np.random.default_rng(seed)
    n = len(grids)
    idx = rng.permutation(n)
    n_test = max(1, round(n * test_ratio))
    n_val  = max(1, round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    train_idx = sorted(idx[:n_train])
    val_idx   = sorted(idx[n_train : n_train + n_val])
    test_idx  = sorted(idx[n_train + n_val :])
    return (
        [grids[i] for i in train_idx],
        [grids[i] for i in val_idx],
        [grids[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------
# REINFORCE agent  (linear policy, numpy-only, system-bet / binary mask)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class REINFORCEAgent:
    """
    Linear policy: logits = W @ obs + b  →  (n_matches, 3) independent Bernoulli.
    Each outcome is independently selected (sigmoid output) to form a system bet.
    Updated via REINFORCE with a mean-reward baseline + entropy regularisation.

    Works for any grid size (7, 8, 12, 15 matches).
    """

    def __init__(
        self,
        n_matches: int,
        k_max: int = 20,
        lr: float = 0.005,
        entropy_coef: float = 0.05,
        seed: int = 42,
    ):
        self.n_matches = n_matches
        self.k_max = k_max
        self.obs_dim = n_matches * N_FEATURES
        self.action_dim = n_matches * N_OUTCOMES
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((self.action_dim, self.obs_dim)).astype(np.float32) * 0.01
        self.b = np.zeros(self.action_dim, dtype=np.float32)
        self.lr = lr
        self.entropy_coef = entropy_coef
        self._rng = rng

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Return a flat (n_matches * 3) float32 binary mask.

        Training: sample each outcome independently via Bernoulli(sigmoid(logit)).
        Deterministic: threshold at 0.5.
        Ensures at least 1 outcome per match, then prunes to k_max combos.
        """
        logits = (self.W @ obs + self.b).astype(np.float64)  # (action_dim,)
        probs = _sigmoid(logits).reshape(self.n_matches, N_OUTCOMES)  # (n_matches, 3)

        if deterministic:
            mask = (probs > 0.5).astype(np.float32)
        else:
            rand = self._rng.random((self.n_matches, N_OUTCOMES))
            mask = (rand < probs).astype(np.float32)

        # Ensure at least 1 per match
        for m in range(self.n_matches):
            if not mask[m].any():
                mask[m, int(probs[m].argmax())] = 1.0

        # Prune to k_max: while product > k_max, remove the selected outcome
        # with the lowest probability in the match that has multiple selections.
        import math
        def n_combos(m):
            return math.prod(int(mask[i].sum()) for i in range(self.n_matches))

        while n_combos(mask) > self.k_max:
            best_match = -1
            best_val = np.inf
            best_outcome = -1
            for m in range(self.n_matches):
                sel_outcomes = [o for o in range(N_OUTCOMES) if mask[m, o] > 0.5]
                if len(sel_outcomes) <= 1:
                    continue
                for o in sel_outcomes:
                    if probs[m, o] < best_val:
                        best_val = probs[m, o]
                        best_match = m
                        best_outcome = o
            if best_match == -1:
                break
            mask[best_match, best_outcome] = 0.0

        return mask.flatten().astype(np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        per_combo_rewards: np.ndarray,  # (n_combos,) from info dict
        selections: np.ndarray,         # (n_matches, 3) bool from info dict
    ) -> None:
        """REINFORCE update over all system-bet combos from the episode."""
        baseline = float(per_combo_rewards.mean())

        logits = (self.W @ obs + self.b).astype(np.float64)
        probs = _sigmoid(logits).reshape(self.n_matches, N_OUTCOMES)  # (n_matches, 3)

        grad_W = np.zeros_like(self.W, dtype=np.float64)
        grad_b = np.zeros_like(self.b, dtype=np.float64)

        # Average advantage across all combos in this episode
        advantage = float(per_combo_rewards.mean()) - baseline  # = 0, so use per-combo
        # REINFORCE: for each outcome (m, o), the log-prob gradient is:
        # (b - sigma) * advantage, where b = selections[m, o]
        # We average advantage over per_combo_rewards vs baseline.
        avg_advantage = float(per_combo_rewards.mean() - baseline)
        # Better: use mean of advantages directly
        advantages = per_combo_rewards - baseline  # (n_combos,)
        mean_advantage = float(advantages.mean()) if len(advantages) > 0 else 0.0

        for m in range(self.n_matches):
            for o in range(N_OUTCOMES):
                b_mo = float(selections[m, o])
                sigma = probs[m, o]
                grad_logit = (b_mo - sigma) * mean_advantage
                s_idx = m * N_OUTCOMES + o
                grad_W[s_idx] += grad_logit * obs
                grad_b[s_idx] += grad_logit

        # Entropy regularisation for Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
        # dH/d_logit = (log(1-p) - log(p)) * p * (1-p)  ... simplifies to:
        # dH/d_sigma = log(1-p) - log(p), dH/d_logit = dH/d_sigma * sigma*(1-sigma)
        if self.entropy_coef > 0:
            for m in range(self.n_matches):
                for o in range(N_OUTCOMES):
                    p = probs[m, o]
                    p = np.clip(p, 1e-7, 1 - 1e-7)
                    entropy_grad_logit = (np.log(1 - p) - np.log(p)) * p * (1 - p)
                    s_idx = m * N_OUTCOMES + o
                    grad_W[s_idx] += self.entropy_coef * entropy_grad_logit * obs
                    grad_b[s_idx] += self.entropy_coef * entropy_grad_logit

        n = max(len(per_combo_rewards), 1)
        self.W += (self.lr * grad_W / n).astype(np.float32)
        self.b += (self.lr * grad_b / n).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent weights and hyperparameters to a .npz file."""
        np.savez(
            path,
            W=self.W,
            b=self.b,
            n_matches=np.int32(self.n_matches),
            k_max=np.int32(self.k_max),
            lr=np.float32(self.lr),
            entropy_coef=np.float32(self.entropy_coef),
        )
        print(f"Agent saved → {path}")

    @classmethod
    def load(cls, path: str) -> "REINFORCEAgent":
        """Load a linear or MLP agent from a .npz file (dispatches to subclass)."""
        data = np.load(path)
        if "W1" in data:
            return MLPREINFORCEAgent.load(path)
        k_max = int(data["k_max"]) if "k_max" in data else 20
        agent = cls(
            n_matches=int(data["n_matches"]),
            k_max=k_max,
            lr=float(data["lr"]),
            entropy_coef=float(data["entropy_coef"]),
        )
        agent.W = data["W"]
        agent.b = data["b"]
        return agent


class MLPREINFORCEAgent(REINFORCEAgent):
    """
    One-hidden-layer MLP policy: obs → ReLU(W1@obs+b1) → W2@h+b2 → logits.
    Inherits act(), update(), save(), load() interface from REINFORCEAgent.
    """

    def __init__(
        self,
        n_matches: int,
        hidden_dim: int = 64,
        k_max: int = 20,
        lr: float = 0.005,
        entropy_coef: float = 0.05,
        seed: int = 42,
    ):
        # Initialise base class but override W/b with MLP weights
        super().__init__(n_matches=n_matches, k_max=k_max,
                         lr=lr, entropy_coef=entropy_coef, seed=seed)
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / self.obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (rng.standard_normal((hidden_dim, self.obs_dim)) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((self.action_dim, hidden_dim)) * scale2).astype(np.float32)
        self.b2 = np.zeros(self.action_dim, dtype=np.float32)
        # Remove linear weights from base (not used)
        del self.W, self.b

    def _forward(self, obs: np.ndarray):
        """Forward pass; returns (logits, hidden_pre, hidden)."""
        h_pre = self.W1 @ obs.astype(np.float64) + self.b1
        h = np.maximum(0.0, h_pre)               # ReLU
        logits = self.W2 @ h + self.b2
        return logits, h_pre, h

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        logits, _, _ = self._forward(obs)
        probs = _sigmoid(logits).reshape(self.n_matches, N_OUTCOMES)

        if deterministic:
            mask = (probs > 0.5).astype(np.float32)
        else:
            rand = self._rng.random((self.n_matches, N_OUTCOMES))
            mask = (rand < probs).astype(np.float32)

        for m in range(self.n_matches):
            if not mask[m].any():
                mask[m, int(probs[m].argmax())] = 1.0

        import math
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

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        per_combo_rewards: np.ndarray,
        selections: np.ndarray,
    ) -> None:
        baseline = float(per_combo_rewards.mean())
        mean_advantage = float(per_combo_rewards.mean() - baseline)  # = 0; use raw mean
        mean_advantage = float(per_combo_rewards.mean())             # use mean reward as signal

        logits, h_pre, h = self._forward(obs)
        probs = _sigmoid(logits).reshape(self.n_matches, N_OUTCOMES)

        # Gradient of Bernoulli log-prob w.r.t. logits: (b - sigma) * advantage
        grad_logits = np.zeros(self.action_dim, dtype=np.float64)
        advantages = per_combo_rewards - baseline
        adv = float(advantages.mean())

        for m in range(self.n_matches):
            for o in range(N_OUTCOMES):
                b_mo = float(selections[m, o])
                sigma = probs[m, o]
                grad_logits[m * N_OUTCOMES + o] = (b_mo - sigma) * adv

        # Entropy bonus
        if self.entropy_coef > 0:
            for m in range(self.n_matches):
                for o in range(N_OUTCOMES):
                    p = np.clip(probs[m, o], 1e-7, 1 - 1e-7)
                    grad_logits[m * N_OUTCOMES + o] += (
                        self.entropy_coef * (np.log(1 - p) - np.log(p)) * p * (1 - p)
                    )

        # Backprop through W2
        grad_W2 = np.outer(grad_logits, h)
        grad_b2 = grad_logits.copy()
        grad_h = self.W2.T @ grad_logits

        # Backprop through ReLU + W1
        grad_h_pre = grad_h * (h_pre > 0)
        obs64 = obs.astype(np.float64)
        grad_W1 = np.outer(grad_h_pre, obs64)
        grad_b1 = grad_h_pre.copy()

        n = max(len(per_combo_rewards), 1)
        self.W2 += (self.lr * grad_W2 / n).astype(np.float32)
        self.b2 += (self.lr * grad_b2 / n).astype(np.float32)
        self.W1 += (self.lr * grad_W1 / n).astype(np.float32)
        self.b1 += (self.lr * grad_b1 / n).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            n_matches=np.int32(self.n_matches),
            hidden_dim=np.int32(self.hidden_dim),
            k_max=np.int32(self.k_max),
            lr=np.float32(self.lr),
            entropy_coef=np.float32(self.entropy_coef),
        )
        print(f"MLP agent saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MLPREINFORCEAgent":
        data = np.load(path)
        agent = cls(
            n_matches=int(data["n_matches"]),
            hidden_dim=int(data["hidden_dim"]),
            k_max=int(data["k_max"]) if "k_max" in data else 20,
            lr=float(data["lr"]),
            entropy_coef=float(data["entropy_coef"]),
        )
        agent.W1 = data["W1"]
        agent.b1 = data["b1"]
        agent.W2 = data["W2"]
        agent.b2 = data["b2"]
        return agent


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
    parser.add_argument("--cutoff-date", type=str, default="auto",
                        help="Max grid date for training (default: parsed from filename, e.g. 2026-03-21)")
    parser.add_argument("--save", type=str, default=None, metavar="PATH",
                        help="Save trained agent to this .npz file after training")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    k = min(max(args.k_grids, 1), 50)

    # ---- Data ---------------------------------------------------------------
    all_grids = load_data(loto_type=args.loto_type, cutoff_date=args.cutoff_date)
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
                           entropy_coef=args.entropy_coef, seed=args.seed)

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
