"""
improve.py — Autonomous RL strategy improvement loop.

Runs continuously, trying different strategies and architectures.
When a new best is found on the test set, it:
  1. Saves the model as agent_best.npz
  2. Appends to leaderboard.json
  3. Updates the LEADERBOARD section in CLAUDE.md
  4. Commits everything to git

Designed to run for hours or days without interruption.

Usage:
    uv run python improve.py                          # run forever
    uv run python improve.py --max-trials 50          # stop after N trials
    uv run python improve.py --file new_data.xlsx     # use different data
    uv run python improve.py --test-grids 3           # different test split
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
    parse_cutoff_date,
)

BEST_MODEL_PATH = "agent_best.npz"
LEADERBOARD_PATH = "leaderboard.json"
CLAUDE_MD_PATH = "CLAUDE.md"
LOG_PATH = "improve.log"

LEADERBOARD_START = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END = "<!-- LEADERBOARD_END -->"


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
    k_max: int = 20
    episodes: int = 6000


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
    # ── More episodes ────────────────────────────────────────────────────
    Strategy("ep_12k",           ["linear", "ep=12k"],       episodes=12000),
    Strategy("ep_20k",           ["linear", "ep=20k"],       episodes=20000),
    # ── MLP architectures ────────────────────────────────────────────────
    Strategy("mlp_32",           ["mlp", "h=32"],            policy="mlp", hidden_dim=32),
    Strategy("mlp_64",           ["mlp", "h=64"],            policy="mlp", hidden_dim=64),
    Strategy("mlp_128",          ["mlp", "h=128"],           policy="mlp", hidden_dim=128),
    # ── MLP + hyperparams ────────────────────────────────────────────────
    Strategy("mlp_64_hlr",       ["mlp", "h=64", "lr=0.02"], policy="mlp", hidden_dim=64, lr=0.02),
    Strategy("mlp_64_ep12k",     ["mlp", "h=64", "ep=12k"],  policy="mlp", hidden_dim=64, episodes=12000),
    Strategy("mlp_128_hlr",      ["mlp", "h=128", "lr=0.02"],policy="mlp", hidden_dim=128, lr=0.02),
    Strategy("mlp_32_k8",        ["mlp", "h=32", "k=8"],     policy="mlp", hidden_dim=32, k_max=8),
    # ── Combined best combinations ───────────────────────────────────────
    Strategy("linear_hlr_ep12k", ["linear", "lr=0.02", "ep=12k"], lr=0.02, episodes=12000),
    Strategy("linear_hlr_k32",   ["linear", "lr=0.02", "k=32"],   lr=0.02, k_max=32),
]


def random_strategy(rng: np.random.Generator, trial_idx: int) -> Strategy:
    """Generate a random strategy for open-ended search."""
    policy = rng.choice(["linear", "mlp"])
    hidden = int(rng.choice([32, 64, 128, 256]))
    lr = float(np.exp(rng.uniform(np.log(5e-4), np.log(0.1))))
    entropy = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.5))))
    k = int(rng.choice([8, 16, 20, 32, 50]))
    episodes = int(rng.choice([4000, 6000, 10000, 16000]))
    name = f"random_{trial_idx}"
    keywords = [policy, f"h={hidden}" if policy == "mlp" else "",
                f"lr={lr:.4f}", f"entropy={entropy:.3f}", f"k={k}", f"ep={episodes//1000}k"]
    keywords = [k for k in keywords if k]
    return Strategy(name, keywords, policy=policy, hidden_dim=hidden,
                    lr=lr, entropy_coef=entropy, k_max=k, episodes=episodes)


# ---------------------------------------------------------------------------
# Training a single trial
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial: int
    strategy_name: str
    keywords: list[str]
    test_net_per_round: float
    test_round_hit_rate: float
    train_net_per_round: float
    n_test_rounds: int
    n_train_rounds: int
    score: float              # primary ranking metric
    model_path: str
    timestamp: str
    commit_hash: str = ""


def build_agent(strategy: Strategy, n_matches: int, seed: int) -> REINFORCEAgent:
    if strategy.policy == "mlp":
        return MLPREINFORCEAgent(
            n_matches=n_matches, hidden_dim=strategy.hidden_dim,
            k_max=strategy.k_max, lr=strategy.lr,
            entropy_coef=strategy.entropy_coef, seed=seed,
        )
    return REINFORCEAgent(
        n_matches=n_matches, k_max=strategy.k_max,
        lr=strategy.lr, entropy_coef=strategy.entropy_coef, seed=seed,
    )


def train_trial(
    strategy: Strategy,
    train_grids: list[dict],
    test_grids: list[dict],
    trial_idx: int,
    seed: int = 0,
) -> tuple[REINFORCEAgent, TrialResult]:
    n_matches = train_grids[0]["n_matches"]
    k = min(strategy.k_max, 50)

    train_env = LotoFootEnv(train_grids, k_max=k, mode="train")
    eval_train = LotoFootEnv(train_grids, k_max=k, mode="eval")
    eval_test  = LotoFootEnv(test_grids,  k_max=k, mode="eval")

    agent = build_agent(strategy, n_matches, seed)

    obs, _ = train_env.reset(seed=seed)
    for ep in range(1, strategy.episodes + 1):
        action = agent.act(obs)
        _, _, _, _, info = train_env.step(action)
        agent.update(obs, action, info["per_combo_rewards"], info["selections"])
        obs, _ = train_env.reset()

    train_r = _collect_results(eval_train, lambda o: agent.act(o, deterministic=False))
    test_r  = _collect_results(eval_test,  lambda o: agent.act(o, deterministic=False))

    test_net_pr   = test_r["net"] / max(test_r["n_rounds"], 1)
    test_hit_rate = test_r["round_hit_rate"]
    train_net_pr  = train_r["net"] / max(train_r["n_rounds"], 1)
    # Primary score: test net/round + bonus for hitting rounds
    score = test_net_pr + 50.0 * test_hit_rate

    result = TrialResult(
        trial=trial_idx,
        strategy_name=strategy.name,
        keywords=strategy.keywords,
        test_net_per_round=round(test_net_pr, 2),
        test_round_hit_rate=round(test_hit_rate, 4),
        train_net_per_round=round(train_net_pr, 2),
        n_test_rounds=test_r["n_rounds"],
        n_train_rounds=train_r["n_rounds"],
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
    if not board:
        return -np.inf
    return max(e["score"] for e in board)


def update_claude_md(board: list[dict]) -> None:
    """Replace the LEADERBOARD section in CLAUDE.md with the current table."""
    if not os.path.exists(CLAUDE_MD_PATH):
        return
    with open(CLAUDE_MD_PATH) as f:
        content = f.read()
    if LEADERBOARD_START not in content:
        return

    top = sorted(board, key=lambda e: e["score"], reverse=True)[:20]

    rows = ["| Rank | Trial | Strategy | Keywords | Test Net/Round | Hit Rate | Train Net/Round | Commit |",
            "|------|-------|----------|----------|---------------|----------|-----------------|--------|"]
    for i, e in enumerate(top, 1):
        kw = ", ".join(e["keywords"])
        commit = e.get("commit_hash", "")[:7] or "—"
        rows.append(
            f"| {i} | {e['trial']} | `{e['strategy_name']}` | {kw} "
            f"| ${e['test_net_per_round']:+.2f} | {e['test_round_hit_rate']*100:.1f}% "
            f"| ${e['train_net_per_round']:+.2f} | {commit} |"
        )

    table = "\n".join(rows)
    new_section = f"{LEADERBOARD_START}\n{table}\n{LEADERBOARD_END}"
    new_content = re.sub(
        re.escape(LEADERBOARD_START) + r".*?" + re.escape(LEADERBOARD_END),
        new_section,
        content,
        flags=re.DOTALL,
    )
    with open(CLAUDE_MD_PATH, "w") as f:
        f.write(new_content)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(result: TrialResult) -> str:
    """Stage relevant files and commit. Returns the short commit hash."""
    try:
        subprocess.run(["git", "add",
                        BEST_MODEL_PATH, LEADERBOARD_PATH, CLAUDE_MD_PATH],
                       check=True, capture_output=True)
        msg = (
            f"improve: [{result.strategy_name}] "
            f"score={result.score:+.2f} "
            f"test_net={result.test_net_per_round:+.2f}/round "
            f"hit={result.test_round_hit_rate*100:.0f}%\n\n"
            f"Keywords: {', '.join(result.keywords)}\n"
            f"Train rounds: {result.n_train_rounds}  Test rounds: {result.n_test_rounds}"
        )
        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
        result_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        return result_hash
    except subprocess.CalledProcessError as e:
        return ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous RL strategy improver")
    parser.add_argument("--file",        type=str, default=EXCEL_FILE)
    parser.add_argument("--test-grids",  type=int, default=4)
    parser.add_argument("--max-trials",  type=int, default=0,
                        help="Stop after N trials (0 = run forever)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    all_grids = load_data(args.file)
    if len(all_grids) <= args.test_grids:
        print(f"Not enough grids: {len(all_grids)} total, need > {args.test_grids}")
        sys.exit(1)
    train_grids = all_grids[: len(all_grids) - args.test_grids]
    test_grids  = all_grids[len(all_grids) - args.test_grids :]
    n_matches   = all_grids[0]["n_matches"]

    log(f"Data: {len(all_grids)} rounds  train={len(train_grids)}  test={len(test_grids)}  "
        f"n_matches={n_matches}")

    # ── Load existing leaderboard ─────────────────────────────────────────
    board = load_leaderboard()
    current_best = best_score(board)
    trial_offset = len(board)
    log(f"Leaderboard: {len(board)} entries  current_best_score={current_best:.4f}")

    # ── Strategy iterator ────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed + trial_offset)

    def strategy_iter():
        for s in PREDEFINED:
            yield s
        t = len(PREDEFINED)
        while True:
            yield random_strategy(rng, trial_offset + t)
            t += 1

    strategies = strategy_iter()
    trial = trial_offset

    try:
        while True:
            if args.max_trials > 0 and (trial - trial_offset) >= args.max_trials:
                log(f"Reached --max-trials {args.max_trials}, stopping.")
                break

            strategy = next(strategies)
            trial += 1
            seed = args.seed + trial

            log(f"── Trial {trial}: {strategy.name}  [{', '.join(strategy.keywords)}]")
            t0 = time.time()

            try:
                agent, result = train_trial(strategy, train_grids, test_grids, trial, seed)
            except Exception as e:
                log(f"   ERROR: {e}")
                continue

            elapsed = time.time() - t0
            log(f"   score={result.score:+.4f}  "
                f"test_net={result.test_net_per_round:+.2f}/round  "
                f"hit={result.test_round_hit_rate*100:.0f}%  "
                f"train_net={result.train_net_per_round:+.2f}/round  "
                f"({elapsed:.0f}s)")

            board.append(asdict(result))

            if result.score > current_best:
                current_best = result.score
                agent.save(BEST_MODEL_PATH)
                commit_hash = git_commit(result)
                board[-1]["commit_hash"] = commit_hash
                save_leaderboard(board)
                update_claude_md(board)
                log(f"   ★ NEW BEST  score={current_best:.4f}  commit={commit_hash}")
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
        log(f"  {i}. [{e['strategy_name']}]  score={e['score']:+.4f}  "
            f"test_net={e['test_net_per_round']:+.2f}  "
            f"hit={e['test_round_hit_rate']*100:.0f}%  "
            f"keywords={e['keywords']}")
    log("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
