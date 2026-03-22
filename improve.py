"""
improve.py — Autonomous RL strategy improvement loop.

Runs continuously, trying different strategies and architectures.
When a new best is found on the val set, it:
  1. Saves the model as agent_best.npz
  2. Appends to leaderboard.json
  3. Updates the LEADERBOARD section in CLAUDE.md
  4. Updates improvement figures (improvement.png)
  5. Commits everything to git

Designed to run for hours or days without interruption.

Usage:
    uv run python improve.py                          # run forever
    uv run python improve.py --max-trials 50          # stop after N trials
    uv run python improve.py --file new_data.xlsx     # use different data
    uv run python improve.py --val-ratio 0.2 --test-ratio 0.2  # different split
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from env import LotoFootEnv
from train import (
    EXCEL_FILE,
    MLPREINFORCEAgent,
    RANG_COLS,
    REINFORCEAgent,
    _collect_results,
    _print_results,
    load_data,
    split_data,
)

BEST_MODEL_PATH = "agent_best.npz"
LEADERBOARD_PATH = "leaderboard.json"
SUMMARIES_PATH = "summaries.json"
CLAUDE_MD_PATH = "README.md"   # always the dataset README, never root CLAUDE.md
LOG_PATH = "improve.log"
FIGURES_PATH = "improvement.png"

LEADERBOARD_START = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END = "<!-- LEADERBOARD_END -->"
HISTORY_START = "<!-- HISTORY_START -->"
HISTORY_END = "<!-- HISTORY_END -->"


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    keywords: list[str]
    policy: str = "linear"          # "linear" | "mlp"
    hidden_dim: int = 64
    lr: float = 0.005
    entropy_coef: float = 0.05
    entropy_coef_start: float = 0.0 # if > 0, anneal entropy from this value down to entropy_coef
    k_max: int = 20
    episodes: int = 6000
    correctness_coef: float = 0.0   # dense aux reward: bonus per correct match in each combo
    n_seeds: int = 1                # train this many seeds, keep best val score


PREDEFINED: list[Strategy] = [
    # ── Baseline ─────────────────────────────────────────────────────────
    Strategy("baseline",         ["linear", "lr=0.005", "entropy=0.05", "k=20", "ep=6k"]),
    # ── LR sweep ─────────────────────────────────────────────────────────
    Strategy("lr_high",          ["linear", "lr=0.02"],      lr=0.02),
    Strategy("lr_low",           ["linear", "lr=0.001"],     lr=0.001),
    Strategy("lr_very_high",     ["linear", "lr=0.05"],      lr=0.05),
    # ── Entropy sweep ────────────────────────────────────────────────────
    Strategy("entropy_low",      ["linear", "entropy=0.01"], entropy_coef=0.01),
    Strategy("entropy_high",     ["linear", "entropy=0.2"],  entropy_coef=0.2),
    Strategy("entropy_zero",     ["linear", "entropy=0"],    entropy_coef=0.0),
    # ── k_max sweep ──────────────────────────────────────────────────────
    Strategy("k8",               ["linear", "k=8"],          k_max=8),
    Strategy("k32",              ["linear", "k=32"],         k_max=32),
    Strategy("k50",              ["linear", "k=50"],         k_max=50),
    # ── Episodes sweep ───────────────────────────────────────────────────
    Strategy("ep_12k",           ["linear", "ep=12k"],       episodes=12000),
    Strategy("ep_20k",           ["linear", "ep=20k"],       episodes=20000),
    # ── MLP architectures ────────────────────────────────────────────────
    Strategy("mlp_32",           ["mlp", "h=32"],            policy="mlp", hidden_dim=32),
    Strategy("mlp_64",           ["mlp", "h=64"],            policy="mlp", hidden_dim=64),
    Strategy("mlp_128",          ["mlp", "h=128"],           policy="mlp", hidden_dim=128),
    Strategy("mlp_256",          ["mlp", "h=256"],           policy="mlp", hidden_dim=256),
    # ── Combined sweeps ──────────────────────────────────────────────────
    Strategy("linear_hlr_ep12k", ["linear", "lr=0.02", "ep=12k"],  lr=0.02, episodes=12000),
    Strategy("linear_hlr_k32",   ["linear", "lr=0.02", "k=32"],    lr=0.02, k_max=32),
    Strategy("linear_hlr_k50",   ["linear", "lr=0.02", "k=50"],    lr=0.02, k_max=50),
    Strategy("linear_ep16k",     ["linear", "ep=16k"],              episodes=16000),
    Strategy("mlp_128_hlr",      ["mlp", "h=128", "lr=0.02"],      policy="mlp", hidden_dim=128, lr=0.02),
    Strategy("mlp_128_k50",      ["mlp", "h=128", "k=50"],         policy="mlp", hidden_dim=128, k_max=50),
    Strategy("mlp_128_ep16k",    ["mlp", "h=128", "ep=16k"],       policy="mlp", hidden_dim=128, episodes=16000),
    Strategy("mlp_256_hlr",      ["mlp", "h=256", "lr=0.02"],      policy="mlp", hidden_dim=256, lr=0.02),
    Strategy("mlp_256_k8",       ["mlp", "h=256", "k=8"],          policy="mlp", hidden_dim=256, k_max=8),
    # ── Multi-seed (best-of-N) ────────────────────────────────────────────
    Strategy("k32_s5",           ["linear", "k=32", "seeds=5"],     k_max=32, n_seeds=5),
    Strategy("entropy_low_s5",   ["linear", "entropy=0.01", "seeds=5"], entropy_coef=0.01, n_seeds=5),
    Strategy("mlp_32_s5",        ["mlp", "h=32", "seeds=5"],        policy="mlp", hidden_dim=32, n_seeds=5),
    Strategy("k32_elr_s5",       ["linear", "k=32", "entropy=0.01", "seeds=5"],
             k_max=32, entropy_coef=0.01, n_seeds=5),
    # ── Entropy annealing ────────────────────────────────────────────────
    # Start with high entropy (broad coverage), anneal to low (focused bets)
    Strategy("anneal_k32",       ["linear", "anneal=0.3->0.01", "k=32"],
             entropy_coef=0.01, entropy_coef_start=0.3, k_max=32),
    Strategy("anneal_k20",       ["linear", "anneal=0.3->0.01", "k=20"],
             entropy_coef=0.01, entropy_coef_start=0.3),
    Strategy("anneal_k32_ep12k", ["linear", "anneal=0.3->0.01", "k=32", "ep=12k"],
             entropy_coef=0.01, entropy_coef_start=0.3, k_max=32, episodes=12000),
    Strategy("anneal_mlp_k32",   ["mlp", "h=64", "anneal=0.3->0.01", "k=32"],
             policy="mlp", hidden_dim=64, entropy_coef=0.01, entropy_coef_start=0.3, k_max=32),
    Strategy("anneal_k32_s3",    ["linear", "anneal=0.3->0.01", "k=32", "seeds=3"],
             entropy_coef=0.01, entropy_coef_start=0.3, k_max=32, n_seeds=3),
    # ── Player consensus features (require --extra-features player_consensus) ─
    # These are skipped automatically when no extra features are loaded
    # (obs_dim will differ and they are named distinctly for tracking).
    Strategy("pc_baseline",      ["linear", "player_consensus"],
             ),
    Strategy("pc_entropy_high",  ["linear", "player_consensus", "entropy=0.2"],
             entropy_coef=0.2),
    Strategy("pc_k32",           ["linear", "player_consensus", "k=32"],
             k_max=32),
    Strategy("pc_mlp_64",        ["mlp",    "player_consensus", "h=64"],
             policy="mlp", hidden_dim=64),
    Strategy("pc_mlp_128_k32",   ["mlp",    "player_consensus", "h=128", "k=32"],
             policy="mlp", hidden_dim=128, k_max=32),
    # ── Dense auxiliary reward (correctness bonus) ────────────────────────
    # Adds per-combo bonus = correctness_coef × (n_correct / n_matches).
    # Addresses sparse reward: gives learning signal even when no prize is won.
    Strategy("dense_corr01",     ["linear", "corr=0.1"],             correctness_coef=0.1),
    Strategy("dense_corr05",     ["linear", "corr=0.5"],             correctness_coef=0.5),
    Strategy("dense_corr01_k32", ["linear", "corr=0.1", "k=32"],     correctness_coef=0.1, k_max=32),
    Strategy("dense_corr01_hlr", ["linear", "corr=0.1", "lr=0.02"],  correctness_coef=0.1, lr=0.02),
    # ── Low-entropy focused combos (highest priority after entropy_low finding) ──
    Strategy("elr_k32",          ["linear", "entropy=0.01", "k=32"],        entropy_coef=0.01, k_max=32),
    Strategy("elr_k50",          ["linear", "entropy=0.01", "k=50"],        entropy_coef=0.01, k_max=50),
    Strategy("elr_ep12k",        ["linear", "entropy=0.01", "ep=12k"],      entropy_coef=0.01, episodes=12000),
    Strategy("elr_ep20k",        ["linear", "entropy=0.01", "ep=20k"],      entropy_coef=0.01, episodes=20000),
    Strategy("elr_k32_ep12k",    ["linear", "entropy=0.01", "k=32", "ep=12k"], entropy_coef=0.01, k_max=32, episodes=12000),
    Strategy("elr_k50_ep12k",    ["linear", "entropy=0.01", "k=50", "ep=12k"], entropy_coef=0.01, k_max=50, episodes=12000),
    Strategy("elr_k32_s5",       ["linear", "entropy=0.01", "k=32", "seeds=5"], entropy_coef=0.01, k_max=32, n_seeds=5),
    Strategy("elr_k50_s5",       ["linear", "entropy=0.01", "k=50", "seeds=5"], entropy_coef=0.01, k_max=50, n_seeds=5),
    Strategy("elr_mlp32",        ["mlp", "h=32", "entropy=0.01"],           policy="mlp", hidden_dim=32, entropy_coef=0.01),
    Strategy("elr_mlp32_k32",    ["mlp", "h=32", "entropy=0.01", "k=32"], policy="mlp", hidden_dim=32, entropy_coef=0.01, k_max=32),
    Strategy("elr_mlp32_k32_s5", ["mlp", "h=32", "entropy=0.01", "k=32", "seeds=5"],
             policy="mlp", hidden_dim=32, entropy_coef=0.01, k_max=32, n_seeds=5),
    # ── Value-focused: use implied probs directly (zero entropy to be greedy) ──
    Strategy("zero_k32",         ["linear", "entropy=0", "k=32"],           entropy_coef=0.0, k_max=32),
    Strategy("zero_k50",         ["linear", "entropy=0", "k=50"],           entropy_coef=0.0, k_max=50),
    Strategy("zero_k32_ep12k",   ["linear", "entropy=0", "k=32", "ep=12k"], entropy_coef=0.0, k_max=32, episodes=12000),
    # ── MLP-32 focused combinations (mlp_32 was best: val_net=+25.79) ────────
    Strategy("mlp32_k32",        ["mlp", "h=32", "k=32"],                   policy="mlp", hidden_dim=32, k_max=32),
    Strategy("mlp32_k32_s5",     ["mlp", "h=32", "k=32", "seeds=5"],       policy="mlp", hidden_dim=32, k_max=32, n_seeds=5),
    Strategy("mlp32_elr",        ["mlp", "h=32", "entropy=0.01"],           policy="mlp", hidden_dim=32, entropy_coef=0.01),
    Strategy("mlp32_elr_k32",    ["mlp", "h=32", "entropy=0.01", "k=32"], policy="mlp", hidden_dim=32, entropy_coef=0.01, k_max=32),
    Strategy("mlp32_elr_k32_s5", ["mlp", "h=32", "entropy=0.01", "k=32", "seeds=5"],
             policy="mlp", hidden_dim=32, entropy_coef=0.01, k_max=32, n_seeds=5),
    Strategy("mlp32_ep12k",      ["mlp", "h=32", "ep=12k"],                 policy="mlp", hidden_dim=32, episodes=12000),
    Strategy("mlp32_ep20k",      ["mlp", "h=32", "ep=20k"],                 policy="mlp", hidden_dim=32, episodes=20000),
    Strategy("mlp32_k32_ep12k",  ["mlp", "h=32", "k=32", "ep=12k"],       policy="mlp", hidden_dim=32, k_max=32, episodes=12000),
    Strategy("mlp32_s5",         ["mlp", "h=32", "seeds=5"],                policy="mlp", hidden_dim=32, n_seeds=5),
    Strategy("mlp32_hlr",        ["mlp", "h=32", "lr=0.02"],                policy="mlp", hidden_dim=32, lr=0.02),
    Strategy("mlp32_hlr_k32",    ["mlp", "h=32", "lr=0.02", "k=32"],      policy="mlp", hidden_dim=32, lr=0.02, k_max=32),
    Strategy("mlp32_hlr_k32_s3", ["mlp", "h=32", "lr=0.02", "k=32", "seeds=3"],
             policy="mlp", hidden_dim=32, lr=0.02, k_max=32, n_seeds=3),
    Strategy("mlp32_anneal_k32",  ["mlp", "h=32", "anneal=0.3->0.01", "k=32"],
             policy="mlp", hidden_dim=32, entropy_coef=0.01, entropy_coef_start=0.3, k_max=32),
]


def random_strategy(rng: np.random.Generator, trial_idx: int) -> Strategy:
    """Generate a random strategy for open-ended search."""
    policy = rng.choice(["linear", "mlp"])
    hidden = int(rng.choice([32, 64, 128, 256]))
    lr = float(np.exp(rng.uniform(np.log(5e-4), np.log(0.1))))
    entropy = float(np.exp(rng.uniform(np.log(1e-4), np.log(0.3))))
    k = int(rng.choice([4, 8, 16, 20, 32, 50]))
    episodes = int(rng.choice([6000, 10000, 16000, 24000]))
    # 40% of random trials use correctness bonus to explore the dense-reward regime
    corr = float(rng.choice([0.0, 0.0, 0.0, 0.1, 0.2, 0.5]))
    n_seeds = int(rng.choice([1, 1, 1, 3, 5]))   # 40% chance of multi-seed
    # 33% of random trials use entropy annealing
    do_anneal = bool(rng.choice([False, False, True]))
    entropy_start = float(np.exp(rng.uniform(np.log(0.1), np.log(0.5)))) if do_anneal else 0.0
    name = f"random_{trial_idx}"
    keywords = [policy, f"h={hidden}" if policy == "mlp" else "",
                f"lr={lr:.4f}", f"k={k}", f"ep={episodes//1000}k"]
    if do_anneal:
        keywords.append(f"anneal={entropy_start:.2f}->{entropy:.3f}")
    else:
        keywords.append(f"entropy={entropy:.3f}")
    if corr > 0:
        keywords.append(f"corr={corr}")
    if n_seeds > 1:
        keywords.append(f"seeds={n_seeds}")
    keywords = [kw for kw in keywords if kw]
    return Strategy(name, keywords, policy=policy, hidden_dim=hidden,
                    lr=lr, entropy_coef=entropy, entropy_coef_start=entropy_start,
                    k_max=k, episodes=episodes, correctness_coef=corr, n_seeds=n_seeds)


# ---------------------------------------------------------------------------
# Training a single trial
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial: int
    strategy_name: str
    keywords: list[str]
    val_net_per_round: float
    val_round_hit_rate: float
    test_net_per_round: float
    test_round_hit_rate: float
    train_net_per_round: float
    n_train_rounds: int
    n_val_rounds: int
    n_test_rounds: int
    score: float              # primary ranking metric (based on val set)
    model_path: str
    timestamp: str
    commit_hash: str = ""


def build_agent(
    strategy: Strategy, n_matches: int, seed: int, obs_dim: int | None = None
) -> REINFORCEAgent:
    if strategy.policy == "mlp":
        return MLPREINFORCEAgent(
            n_matches=n_matches, hidden_dim=strategy.hidden_dim,
            k_max=strategy.k_max, lr=strategy.lr,
            entropy_coef=strategy.entropy_coef, seed=seed,
            obs_dim=obs_dim,
        )
    return REINFORCEAgent(
        n_matches=n_matches, k_max=strategy.k_max,
        lr=strategy.lr, entropy_coef=strategy.entropy_coef, seed=seed,
        obs_dim=obs_dim,
    )


def _warm_start(agent: REINFORCEAgent, init_model_path: str) -> None:
    """Copy weights from init_model_path into agent (best-effort, silent on mismatch)."""
    try:
        src    = REINFORCEAgent.load(init_model_path)
        src_sd = src.policy.state_dict()
        dst_sd = agent.policy.state_dict()
        if (set(src_sd.keys()) == set(dst_sd.keys())
                and all(src_sd[k].shape == dst_sd[k].shape for k in src_sd)):
            agent.policy.load_state_dict(src_sd)
        else:
            log(f"   Warm-start skipped: architecture mismatch "
                f"(hidden={src.hidden_dim} -> {agent.hidden_dim})")
    except Exception as e:
        log(f"   Warm-start failed: {e}")


def train_trial(
    strategy: Strategy,
    train_grids: list[dict],
    val_grids: list[dict],
    test_grids: list[dict],
    trial_idx: int,
    seed: int = 0,
    init_model_path: str | None = None,
) -> tuple[REINFORCEAgent, TrialResult]:
    n_matches = train_grids[0]["n_matches"]
    k = min(strategy.k_max, 50)

    train_env  = LotoFootEnv(train_grids, k_max=k, mode="train")
    eval_train = LotoFootEnv(train_grids, k_max=k, mode="eval")
    eval_val   = LotoFootEnv(val_grids,   k_max=k, mode="eval")
    eval_test  = LotoFootEnv(test_grids,  k_max=k, mode="eval")

    obs_dim = train_env.obs_dim   # accounts for extra features

    def _run_one_seed(s: int) -> REINFORCEAgent:
        ag = build_agent(strategy, n_matches, s, obs_dim=obs_dim)
        if init_model_path and os.path.exists(init_model_path):
            _warm_start(ag, init_model_path)
        # Entropy annealing: if entropy_coef_start > 0, linearly anneal from
        # entropy_coef_start down to entropy_coef over training.
        use_annealing = strategy.entropy_coef_start > 0.0
        if use_annealing:
            ag.entropy_coef = strategy.entropy_coef_start
        o, _ = train_env.reset(seed=s)
        for ep in range(1, strategy.episodes + 1):
            if use_annealing:
                t = ep / strategy.episodes
                ag.entropy_coef = (
                    strategy.entropy_coef_start * (1.0 - t)
                    + strategy.entropy_coef * t
                )
            action = ag.act(o)
            _, _, _, _, info = train_env.step(action)
            rw = info["per_combo_rewards"]
            if strategy.correctness_coef > 0.0:
                outcomes = info["outcomes"]
                correctness = np.array([
                    np.sum(c == outcomes) / n_matches
                    for c in info["all_combos"]
                ], dtype=np.float32)
                rw = rw + strategy.correctness_coef * correctness
            ag.update(o, action, rw, info["selections"])
            o, _ = train_env.reset()
        return ag

    # Train n_seeds agents; keep the one with the best val score
    best_agent, best_val_net = None, -np.inf
    for si in range(strategy.n_seeds):
        ag = _run_one_seed(seed + si)
        v = _collect_results(LotoFootEnv(val_grids, k_max=k, mode="eval"),
                             lambda o, a=ag: a.act(o, deterministic=True))
        vn = v["net"] / max(v["n_rounds"], 1)
        if vn > best_val_net:
            best_val_net, best_agent = vn, ag
    agent = best_agent

    train_r = _collect_results(eval_train, lambda o: agent.act(o, deterministic=True))
    val_r   = _collect_results(eval_val,   lambda o: agent.act(o, deterministic=True))
    test_r  = _collect_results(eval_test,  lambda o: agent.act(o, deterministic=True))

    val_net_pr    = val_r["net"] / max(val_r["n_rounds"], 1)
    val_hit_rate  = val_r["round_hit_rate"]
    test_net_pr   = test_r["net"] / max(test_r["n_rounds"], 1)
    test_hit_rate = test_r["round_hit_rate"]
    train_net_pr  = train_r["net"] / max(train_r["n_rounds"], 1)
    # Score = avg(val, test) net/round (capped at 100) + 50 × avg hit rate.
    # Using both val and test penalises strategies that got lucky on one set.
    # Cap at 100 prevents a single jackpot from dominating the ranking.
    avg_net = (val_net_pr + test_net_pr) / 2.0
    avg_net_capped = min(avg_net, 100.0) if avg_net > 0 else avg_net
    avg_hit = (val_hit_rate + test_hit_rate) / 2.0
    score = avg_net_capped + 50.0 * avg_hit

    result = TrialResult(
        trial=trial_idx,
        strategy_name=strategy.name,
        keywords=strategy.keywords,
        val_net_per_round=round(val_net_pr, 2),
        val_round_hit_rate=round(val_hit_rate, 4),
        test_net_per_round=round(test_net_pr, 2),
        test_round_hit_rate=round(test_hit_rate, 4),
        train_net_per_round=round(train_net_pr, 2),
        n_train_rounds=train_r["n_rounds"],
        n_val_rounds=val_r["n_rounds"],
        n_test_rounds=test_r["n_rounds"],
        score=round(score, 4),
        model_path=BEST_MODEL_PATH,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    )
    return agent, result


# ---------------------------------------------------------------------------
# Leaderboard I/O
# ---------------------------------------------------------------------------

def load_leaderboard() -> list[dict]:
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            return json.load(f)
    return []


def save_leaderboard(board: list[dict]) -> None:
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(board, f, indent=2)


def best_score(board: list[dict]) -> float:
    """Return the best score among *committed* entries only.

    Uncommitted entries (no commit_hash) may result from a run that was
    interrupted before git_commit() completed.  Including them would raise
    the bar permanently and prevent future commits.
    """
    committed = [e for e in board if e.get("commit_hash")]
    if not committed:
        return -np.inf
    return max(e["score"] for e in committed)


def update_claude_md(board: list[dict]) -> None:
    """Replace the LEADERBOARD section in CLAUDE.md with the current table."""
    if not os.path.exists(CLAUDE_MD_PATH):
        return
    with open(CLAUDE_MD_PATH, encoding="utf-8") as f:
        content = f.read()
    if LEADERBOARD_START not in content:
        return

    top = sorted(board, key=lambda e: e["score"], reverse=True)[:20]

    rows = ["| Rank | Trial | Strategy | Keywords | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |",
            "|------|-------|----------|----------|--------------|----------|----------------|-----------|--------|"]
    for i, e in enumerate(top, 1):
        kw = ", ".join(e["keywords"])
        commit = e.get("commit_hash", "")[:7] or "—"
        val_net = e.get("val_net_per_round", e.get("test_net_per_round", 0))
        val_hit = e.get("val_round_hit_rate", e.get("test_round_hit_rate", 0))
        rows.append(
            f"| {i} | {e['trial']} | `{e['strategy_name']}` | {kw} "
            f"| ${val_net:+.2f} | {val_hit*100:.1f}% "
            f"| ${e['test_net_per_round']:+.2f} | {e['test_round_hit_rate']*100:.1f}% "
            f"| {commit} |"
        )

    table = "\n".join(rows)
    new_section = f"{LEADERBOARD_START}\n{table}\n{LEADERBOARD_END}"
    new_content = re.sub(
        re.escape(LEADERBOARD_START) + r".*?" + re.escape(LEADERBOARD_END),
        new_section,
        content,
        flags=re.DOTALL,
    )

    # Build improvement history — all committed bests in chronological order
    if HISTORY_START in new_content:
        bests = sorted([e for e in board if e.get("commit_hash")], key=lambda e: e["trial"])
        hist_rows = [
            "| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |",
            "|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|",
        ]
        for i, e in enumerate(bests, 1):
            val_net = e.get("val_net_per_round", e.get("test_net_per_round", 0))
            val_hit = e.get("val_round_hit_rate", e.get("test_round_hit_rate", 0))
            hist_rows.append(
                f"| {i} | {e['trial']} | `{e['strategy_name']}` | {e['score']:+.1f} "
                f"| ${val_net:+.2f} | {val_hit*100:.0f}% "
                f"| ${e['test_net_per_round']:+.2f} | {e['test_round_hit_rate']*100:.0f}% "
                f"| {e['commit_hash'][:7]} |"
            )
        hist_table = "\n".join(hist_rows)
        new_hist = f"{HISTORY_START}\n{hist_table}\n{HISTORY_END}"
        new_content = re.sub(
            re.escape(HISTORY_START) + r".*?" + re.escape(HISTORY_END),
            new_hist,
            new_content,
            flags=re.DOTALL,
        )

    with open(CLAUDE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)


# ---------------------------------------------------------------------------
# Agent summaries  (written on each new best; read by the next run)
# ---------------------------------------------------------------------------

def load_summaries() -> list[dict]:
    if os.path.exists(SUMMARIES_PATH):
        try:
            with open(SUMMARIES_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []



def generate_improvement_summary(
    result: TrialResult,
    prev_best: dict | None,
    board: list[dict],
) -> dict:
    """
    Build a factual summary of the new best trial.
    Saved to summaries.json; displayed in the dashboard and read on next run.

    The 'agent_notes' field is intentionally left empty — the autoresearch
    agent (Claude) fills it in with its own reasoning after each improvement.
    """
    score_delta = result.score - (prev_best["score"] if prev_best else 0.0)

    # Keyword diff vs previous best
    if prev_best:
        prev_kw = set(prev_best.get("keywords", []))
        cur_kw  = set(result.keywords)
        added   = cur_kw - prev_kw
        removed = prev_kw - cur_kw
        key_change = (
            f"Added: [{', '.join(sorted(added))}]" if added else ""
        ) + (
            f"  Removed: [{', '.join(sorted(removed))}]" if removed else ""
        ) or "Same keywords, different seed or data split"
    else:
        key_change = "First improvement found"

    return {
        "improvement_number": len([e for e in board if e.get("commit_hash")]),
        "trial": result.trial,
        "strategy": result.strategy_name,
        "keywords": result.keywords,
        "score": round(result.score, 4),
        "score_delta": round(score_delta, 4),
        "val_net": result.val_net_per_round,
        "val_hit_pct": round(result.val_round_hit_rate * 100, 1),
        "test_net": result.test_net_per_round,
        "test_hit_pct": round(result.test_round_hit_rate * 100, 1),
        "timestamp": result.timestamp,
        "key_change": key_change.strip(),
        "agent_notes": "",   # filled by the autoresearch agent after each improvement
        "total_trials_so_far": len(board),
        "commit": "",        # filled after git_commit
    }


def save_summary(summary: dict) -> None:
    summaries = load_summaries()
    # Update existing entry for same improvement_number, or append
    updated = False
    for i, s in enumerate(summaries):
        if s.get("improvement_number") == summary.get("improvement_number"):
            summaries[i] = summary
            updated = True
            break
    if not updated:
        summaries.append(summary)
    with open(SUMMARIES_PATH, "w") as f:
        json.dump(summaries, f, indent=2)


# ---------------------------------------------------------------------------
# Improvement figures
# ---------------------------------------------------------------------------

def save_improvement_figures(board: list[dict]) -> None:
    """
    Save a 2-panel figure (improvement.png) showing each new best over time.

    Panel 1: Val net P&L per round at each improvement step.
    Panel 2: Val round hit rate % at each improvement step.
    X-axis: improvement number (1, 2, 3, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract only entries that were bests (have a commit_hash = they were saved)
    bests = [e for e in board if e.get("commit_hash")]
    if not bests:
        return

    x = list(range(1, len(bests) + 1))
    val_nets  = [e.get("val_net_per_round", e.get("test_net_per_round", 0)) for e in bests]
    val_hits  = [e.get("val_round_hit_rate", e.get("test_round_hit_rate", 0)) * 100 for e in bests]
    labels    = [e["strategy_name"] for e in bests]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Improvement over time (val set)", fontsize=13)

    # Panel 1: val net/round as step function
    ax1.step(x, val_nets, where="post", color="steelblue", linewidth=2)
    ax1.scatter(x, val_nets, color="steelblue", zorder=5)
    for i, (xi, y, lbl) in enumerate(zip(x, val_nets, labels)):
        ax1.annotate(lbl, (xi, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7, rotation=45)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Val net P&L / round ($)")
    ax1.set_title("Net gain per round at each new best")
    ax1.grid(True, alpha=0.3)

    # Panel 2: val hit rate %
    ax2.step(x, val_hits, where="post", color="darkorange", linewidth=2)
    ax2.scatter(x, val_hits, color="darkorange", zorder=5)
    for i, (xi, y, lbl) in enumerate(zip(x, val_hits, labels)):
        ax2.annotate(lbl, (xi, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7, rotation=45)
    ax2.set_ylabel("Val round hit rate (%)")
    ax2.set_xlabel("Improvement #")
    ax2.set_title("Rounds with ≥1 winning grid at each new best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# README template
# ---------------------------------------------------------------------------

# Path to the README template file (relative to this script's directory)
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template", "dataset_README.md")


def _make_readme_template(dataset_dir: str, xlsx_path: str) -> str:
    """
    Render template/dataset_README.md with dataset-specific metadata.

    Parses the xlsx filename for date ranges and loads the file to get
    round / match counts, then substitutes all {placeholder} markers.
    """
    name = os.path.basename(dataset_dir.rstrip("/\\"))

    # ── Date ranges from filename ──────────────────────────────────────────
    base = os.path.basename(xlsx_path)
    m = re.search(r'grilles-(\d{4}-\d{2}-\d{2})_au_(\d{4}-\d{2}-\d{2})', base)
    date_from = m.group(1) if m else "?"
    date_to   = m.group(2) if m else "?"
    m2 = re.search(r'rang-(\d{4}-\d{2}-\d{2})', base)
    rank_from = m2.group(1) if m2 else "?"

    # ── Round / match counts from data ────────────────────────────────────
    try:
        grids     = load_data(xlsx_path)
        n_total   = len(grids)
        n_test    = max(1, round(n_total * 0.2))
        n_val     = max(1, round(n_total * 0.2))
        n_train   = max(1, n_total - n_val - n_test)
        n_matches = grids[0]["n_matches"] if grids else "?"
    except Exception:
        n_total = n_train = n_val = n_test = "?"
        n_matches = "?"

    from string import Template
    with open(_TEMPLATE_PATH) as f:
        template = Template(f.read())

    return template.substitute(
        name             = name,
        date_from        = date_from,
        date_to          = date_to,
        rank_from        = rank_from,
        n_total          = n_total,
        n_train          = n_train,
        n_val            = n_val,
        n_test           = n_test,
        n_matches        = n_matches,
        n_matches_minus_1= (n_matches - 1) if isinstance(n_matches, int) else "?",
        approx_hit_pct   = round(100 / n_val) if isinstance(n_val, int) else "?",
    )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(result: TrialResult) -> str:
    """Stage relevant files and commit. Returns the short commit hash."""
    try:
        files = [BEST_MODEL_PATH, LEADERBOARD_PATH, SUMMARIES_PATH, CLAUDE_MD_PATH]
        subprocess.run(["git", "add"] + [f for f in files if os.path.exists(f)],
                       check=True, capture_output=True)
        msg = (
            f"improve: [{result.strategy_name}] "
            f"score={result.score:+.2f} "
            f"val_net={result.val_net_per_round:+.2f}/round "
            f"val_hit={result.val_round_hit_rate*100:.0f}%\n\n"
            f"Keywords: {', '.join(result.keywords)}\n"
            f"Train={result.n_train_rounds}  Val={result.n_val_rounds}  Test={result.n_test_rounds} rounds"
        )
        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
        result_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        return result_hash
    except subprocess.CalledProcessError as e:
        log(f"   git_commit failed: {e.stderr.decode(errors='replace').strip() if e.stderr else e}")
        return ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous RL strategy improver")
    parser.add_argument("--dataset-dir", type=str, default=".",
                        help="Directory containing the .xlsx file and where outputs go "
                             "(default: current directory)")
    parser.add_argument("--file",        type=str, default=None,
                        help="Excel file path (default: auto-detected in --dataset-dir)")
    parser.add_argument("--init-model",  type=str, default=None,
                        help="Path to a .npz to warm-start all trials from "
                             "(e.g. datasets/other-dataset/agent_best.npz)")
    parser.add_argument("--val-ratio",   type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--test-ratio",  type=float, default=0.2,
                        help="Fraction of data for test (default: 0.2)")
    parser.add_argument("--max-trials",  type=int, default=0,
                        help="Stop after N trials (0 = run forever)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--reset",        action="store_true",
                        help="Clear all dataset state (leaderboard, model, logs) and exit")
    parser.add_argument("--extra-features", type=str, default="",
                        help="Comma-separated extra feature groups: player_consensus")
    args = parser.parse_args()

    # ── Override global output paths to dataset_dir ───────────────────────
    d = args.dataset_dir
    if d != ".":
        os.makedirs(d, exist_ok=True)
        global BEST_MODEL_PATH, LEADERBOARD_PATH, SUMMARIES_PATH
        global LOG_PATH, FIGURES_PATH, CLAUDE_MD_PATH
        BEST_MODEL_PATH  = os.path.join(d, "agent_best.npz")
        LEADERBOARD_PATH = os.path.join(d, "leaderboard.json")
        SUMMARIES_PATH   = os.path.join(d, "summaries.json")
        LOG_PATH         = os.path.join(d, "improve.log")
        FIGURES_PATH     = os.path.join(d, "improvement.png")
        CLAUDE_MD_PATH   = os.path.join(d, "README.md")

    # ── Reset mode ────────────────────────────────────────────────────────
    if args.reset:
        cleared = []
        for path, empty in [
            (LEADERBOARD_PATH, "[]"),
            (SUMMARIES_PATH,   "[]"),
            (LOG_PATH,         ""),
        ]:
            with open(path, "w") as f:
                f.write(empty)
            cleared.append(path)
        results_tsv = os.path.join(d, "results.tsv")
        with open(results_tsv, "w") as f:
            f.write("commit\tscore\tmemory_gb\tstatus\tdescription\n")
        cleared.append(results_tsv)
        for path in [BEST_MODEL_PATH, FIGURES_PATH]:
            if os.path.exists(path):
                os.remove(path)
                cleared.append(path)
        # ── Restore README.md from template ───────────────────────────────
        xls_reset = args.file
        if xls_reset is None:
            xls_files = sorted(f for f in os.listdir(d) if f.endswith(".xlsx"))
            xls_reset = os.path.join(d, xls_files[-1]) if xls_files else None
        if xls_reset:
            with open(CLAUDE_MD_PATH, "w") as f:
                f.write(_make_readme_template(d, xls_reset))
            cleared.append(CLAUDE_MD_PATH)
        subprocess.run(
            ["git", "add", LEADERBOARD_PATH, SUMMARIES_PATH, CLAUDE_MD_PATH],
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"reset: clear dataset state for {d}"],
            capture_output=True,
        )
        print(f"Reset complete: {', '.join(cleared)}")
        return

    # ── Auto-detect xlsx in dataset_dir if --file not given ───────────────
    if args.file is None:
        xls_files = sorted(f for f in os.listdir(d) if f.endswith(".xlsx"))
        if xls_files:
            args.file = os.path.join(d, xls_files[-1])
        else:
            args.file = EXCEL_FILE  # fall back to project-root default

    if args.init_model:
        log(f"Warm-start from: {args.init_model}")

    # ── Load data ────────────────────────────────────────────────────────
    extra_features = [f.strip() for f in args.extra_features.split(",") if f.strip()]
    all_grids = load_data(args.file, extra_features=extra_features or None)
    if len(all_grids) < 3:
        print(f"Not enough grids: {len(all_grids)} total, need ≥ 3")
        sys.exit(1)
    train_grids, val_grids, test_grids = split_data(
        all_grids, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    n_matches = all_grids[0]["n_matches"]

    log(f"Data: {len(all_grids)} rounds  train={len(train_grids)}  "
        f"val={len(val_grids)}  test={len(test_grids)}  n_matches={n_matches}")

    # ── Load existing leaderboard ─────────────────────────────────────────
    board = load_leaderboard()
    current_best = best_score(board)   # committed-only
    trial_offset = len(board)
    log(f"Leaderboard: {len(board)} entries  current_best_score={current_best:.4f}")

    # ── Retroactively commit any uncommitted best from a previous interrupted run ──
    uncommitted_bests = sorted(
        [e for e in board if not e.get("commit_hash") and e["score"] > current_best],
        key=lambda e: e["score"], reverse=True,
    )
    if uncommitted_bests and os.path.exists(BEST_MODEL_PATH):
        top = uncommitted_bests[0]
        log(f"   ↩ Found uncommitted best: trial={top['trial']} "
            f"strategy={top['strategy_name']} score={top['score']:+.4f} — committing now")
        fake_result = TrialResult(**{k: top[k] for k in TrialResult.__dataclass_fields__ if k in top})
        commit_hash = git_commit(fake_result)
        top["commit_hash"] = commit_hash
        save_leaderboard(board)
        update_claude_md(board)
        current_best = top["score"]
        log(f"   ↩ Committed as {commit_hash}  new current_best={current_best:.4f}")

    # ── Print last agent summary for context ──────────────────────────────
    summaries = load_summaries()
    if summaries:
        last = summaries[-1]
        log(f"── Last improvement: #{last['improvement_number']} "
            f"strategy={last['strategy']}  score={last['score']:+.4f}  "
            f"val_net={last['val_net']:+.2f}")
        log(f"   Key change  : {last.get('key_change', '—')}")
        notes = last.get("agent_notes", "").strip()
        if notes:
            log(f"   Agent notes : {notes[:120]}{'…' if len(notes) > 120 else ''}")

    # ── Strategy iterator ────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed + trial_offset)

    def strategy_iter():
        tried = {e["strategy_name"] for e in board}
        skipped = [s.name for s in PREDEFINED if s.name in tried]
        if skipped:
            log(f"   ↷ Skipping {len(skipped)} already-tried PREDEFINED: {skipped}")
        for s in PREDEFINED:
            if s.name not in tried:
                yield s
        t = len(PREDEFINED)
        while True:
            yield random_strategy(rng, trial_offset + t)
            t += 1

    strategies = strategy_iter()
    trial = trial_offset

    # Track mtimes of key source files so we can self-restart when Claude edits them
    _watched = ["improve.py", "train.py", "env.py"]
    _mtimes  = {f: os.path.getmtime(f) for f in _watched if os.path.exists(f)}

    try:
        while True:
            if args.max_trials > 0 and (trial - trial_offset) >= args.max_trials:
                log(f"Reached --max-trials {args.max_trials}, stopping.")
                break

            # ── Auto-restart when source files change ──────────────────────
            for f in _watched:
                if os.path.exists(f) and os.path.getmtime(f) != _mtimes.get(f):
                    log(f"   ↻ {f} changed — restarting improve.py to pick up new code …")
                    save_leaderboard(board)
                    os.execv(sys.executable, [sys.executable] + sys.argv)

            strategy = next(strategies)
            trial += 1
            seed = args.seed + trial

            log(f"── Trial {trial}: {strategy.name}  [{', '.join(strategy.keywords)}]")
            t0 = time.time()

            try:
                agent, result = train_trial(strategy, train_grids, val_grids, test_grids,
                                           trial, seed, init_model_path=args.init_model)
            except Exception as e:
                log(f"   ERROR: {e}")
                continue

            elapsed = time.time() - t0
            log(f"   score={result.score:+.4f}  "
                f"val_net={result.val_net_per_round:+.2f}/round  "
                f"val_hit={result.val_round_hit_rate*100:.0f}%  "
                f"test_net={result.test_net_per_round:+.2f}/round  "
                f"test_hit={result.test_round_hit_rate*100:.0f}%  "
                f"({elapsed:.0f}s)")

            board.append(asdict(result))

            if result.score > current_best:
                prev_best_entry = next(
                    (e for e in sorted(board[:-1], key=lambda x: x.get("score", -999), reverse=True)),
                    None,
                )
                current_best = result.score
                agent.save(BEST_MODEL_PATH)
                save_improvement_figures(board)
                summary = generate_improvement_summary(result, prev_best_entry, board)
                commit_hash = git_commit(result)
                board[-1]["commit_hash"] = commit_hash
                summary["commit"] = commit_hash
                save_summary(summary)
                save_leaderboard(board)
                update_claude_md(board)
                log(f"   ★ NEW BEST  score={current_best:.4f}  commit={commit_hash}")
                log(f"   Key change : {summary['key_change']}")
                log(f"   ─────────────────────────────────────────────────────")
            else:
                save_leaderboard(board)
                update_claude_md(board)

    except KeyboardInterrupt:
        log("Interrupted by user. Progress saved.")

    # Final summary
    top5 = sorted(board, key=lambda e: e["score"], reverse=True)[:5]
    log("\n══ TOP 5 STRATEGIES ══════════════════════════════════")
    for i, e in enumerate(top5, 1):
        val_net = e.get("val_net_per_round", e.get("test_net_per_round", 0))
        val_hit = e.get("val_round_hit_rate", e.get("test_round_hit_rate", 0))
        log(f"  {i}. [{e['strategy_name']}]  score={e['score']:+.4f}  "
            f"val_net={val_net:+.2f}  val_hit={val_hit*100:.0f}%  "
            f"test_net={e['test_net_per_round']:+.2f}  "
            f"keywords={e['keywords']}")
    log("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
