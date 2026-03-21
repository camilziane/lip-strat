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
    parse_cutoff_date,
    split_data,
)

BEST_MODEL_PATH = "agent_best.npz"
LEADERBOARD_PATH = "leaderboard.json"
SUMMARIES_PATH = "summaries.json"
CLAUDE_MD_PATH = "CLAUDE.md"
LOG_PATH = "improve.log"
FIGURES_PATH = "improvement.png"

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


def _warm_start(agent: REINFORCEAgent, init_model_path: str) -> None:
    """Copy weights from init_model_path into agent (best-effort, silent on mismatch)."""
    try:
        src = REINFORCEAgent.load(init_model_path)
        if isinstance(agent, MLPREINFORCEAgent) and isinstance(src, MLPREINFORCEAgent):
            if agent.W1.shape == src.W1.shape and agent.W2.shape == src.W2.shape:
                agent.W1[:] = src.W1; agent.b1[:] = src.b1
                agent.W2[:] = src.W2; agent.b2[:] = src.b2
                return
        elif not isinstance(agent, MLPREINFORCEAgent) and not isinstance(src, MLPREINFORCEAgent):
            if agent.W.shape == src.W.shape:
                agent.W[:] = src.W; agent.b[:] = src.b
                return
        log(f"   Warm-start skipped: architecture mismatch "
            f"({type(src).__name__} → {type(agent).__name__})")
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

    agent = build_agent(strategy, n_matches, seed)
    if init_model_path and os.path.exists(init_model_path):
        _warm_start(agent, init_model_path)

    obs, _ = train_env.reset(seed=seed)
    for ep in range(1, strategy.episodes + 1):
        action = agent.act(obs)
        _, _, _, _, info = train_env.step(action)
        agent.update(obs, action, info["per_combo_rewards"], info["selections"])
        obs, _ = train_env.reset()

    train_r = _collect_results(eval_train, lambda o: agent.act(o, deterministic=False))
    val_r   = _collect_results(eval_val,   lambda o: agent.act(o, deterministic=False))
    test_r  = _collect_results(eval_test,  lambda o: agent.act(o, deterministic=False))

    val_net_pr    = val_r["net"] / max(val_r["n_rounds"], 1)
    val_hit_rate  = val_r["round_hit_rate"]
    test_net_pr   = test_r["net"] / max(test_r["n_rounds"], 1)
    test_hit_rate = test_r["round_hit_rate"]
    train_net_pr  = train_r["net"] / max(train_r["n_rounds"], 1)
    # Primary score: val net/round + bonus for hitting rounds (val-based selection)
    score = val_net_pr + 50.0 * val_hit_rate

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
    with open(CLAUDE_MD_PATH, "w") as f:
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


def _kw_map(keywords: list[str]) -> dict[str, str]:
    """Parse ['lr=0.02', 'mlp', 'h=64'] → {'lr': '0.02', 'mlp': '', 'h': '64'}."""
    out: dict[str, str] = {}
    for kw in keywords:
        if "=" in kw:
            k, v = kw.split("=", 1)
            out[k] = v
        else:
            out[kw] = ""
    return out


def generate_improvement_summary(
    result: TrialResult,
    prev_best: dict | None,
    board: list[dict],
) -> dict:
    """
    Auto-generate a structured summary of what worked and what to try next.
    Saved to summaries.json; read by the next agent run for context.
    """
    kw = _kw_map(result.keywords)
    policy = "mlp" if "mlp" in kw else "linear"
    lr = float(kw.get("lr", 0.005))
    entropy = float(kw.get("entropy", 0.05)) if "entropy" in kw else 0.05
    h = int(kw.get("h", 0))
    ep_str = kw.get("ep", "6k")
    ep = int(ep_str.replace("k", "")) * 1000 if "k" in ep_str else int(ep_str)
    k_val = int(kw.get("k", 20))

    score_delta = result.score - (prev_best["score"] if prev_best else 0.0)

    # Key change from previous best
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

    # What worked analysis
    insights: list[str] = []
    if policy == "mlp":
        insights.append(f"Non-linear MLP policy (h={h}) beat the linear baseline")
    if lr >= 0.02:
        insights.append(f"Higher learning rate ({lr}) helped — faster adaptation to prize structure")
    elif lr <= 0.001:
        insights.append(f"Low learning rate ({lr}) — slow but stable convergence")
    if ep >= 12000:
        insights.append(f"More episodes ({ep_str}) gave better convergence")
    if k_val >= 32:
        insights.append(f"Wider system bet (k={k_val}) covered more combinations")
    if entropy <= 0.01:
        insights.append("Near-zero entropy — policy exploits learned patterns aggressively")
    if not insights:
        insights.append(f"Strategy [{result.strategy_name}] found a better combination of features")

    # What to try next (hypotheses for the next agent)
    tried_strategies = {e["strategy_name"] for e in board}
    hypotheses: list[str] = []

    if policy == "linear":
        hypotheses.append(
            f"Try MLP policy (h=64 or h=128) with same lr={lr:.4f} — "
            "non-linear features may extract more signal from odds + crowd %"
        )
    else:
        if h < 128:
            hypotheses.append(
                f"Try larger MLP h={h * 2} — with only 10 features/match, "
                "more capacity may help"
            )
        hypotheses.append(
            "Try a 2-hidden-layer MLP: add second layer after ReLU "
            "(needs architecture change in train.py)"
        )

    if ep < 12000:
        hypotheses.append(
            f"Try ep=12k or ep=20k with current winning config "
            f"[{', '.join(result.keywords)}] — more training may refine the policy"
        )

    if lr < 0.02 and "lr_high" not in tried_strategies:
        hypotheses.append(f"Try lr=0.02 — higher LR sometimes escapes flat regions faster")
    elif lr >= 0.02 and entropy > 0.05:
        hypotheses.append("Try entropy=0.01 — reduce exploration once a good LR is found")

    if k_val <= 20:
        hypotheses.append(
            "Try k=32 — wider system bet coverage increases chances of hitting rang2 "
            "at the cost of more grids per round"
        )

    # Reward shaping idea
    hypotheses.append(
        "Try reward shaping: scale reward by 1/implied_prob of the correct outcome "
        "to encourage backing value bets (upsets)"
    )

    hypotheses = list(dict.fromkeys(hypotheses))[:4]  # deduplicate, keep 4

    # Strategies not yet tried from PREDEFINED
    all_predefined = [s.name for s in PREDEFINED]
    not_tried = [s for s in all_predefined if s not in tried_strategies][:5]

    return {
        "improvement_number": len([e for e in board if e.get("commit_hash")]),
        "trial": result.trial,
        "strategy": result.strategy_name,
        "keywords": result.keywords,
        "policy": policy,
        "score": round(result.score, 4),
        "score_delta": round(score_delta, 4),
        "val_net": result.val_net_per_round,
        "val_hit_pct": round(result.val_round_hit_rate * 100, 1),
        "test_net": result.test_net_per_round,
        "test_hit_pct": round(result.test_round_hit_rate * 100, 1),
        "timestamp": result.timestamp,
        "key_change": key_change.strip(),
        "what_worked": insights,
        "hypotheses_for_next_agent": hypotheses,
        "predefined_not_yet_tried": not_tried,
        "total_trials_so_far": len(board),
        "commit": "",  # filled after git_commit
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
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(result: TrialResult) -> str:
    """Stage relevant files and commit. Returns the short commit hash."""
    try:
        files = [BEST_MODEL_PATH, LEADERBOARD_PATH, SUMMARIES_PATH, CLAUDE_MD_PATH, FIGURES_PATH]
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
    all_grids = load_data(args.file)
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
    current_best = best_score(board)
    trial_offset = len(board)
    log(f"Leaderboard: {len(board)} entries  current_best_score={current_best:.4f}")

    # ── Print last agent summary for context ──────────────────────────────
    summaries = load_summaries()
    if summaries:
        last = summaries[-1]
        log(f"── Last improvement: #{last['improvement_number']} "
            f"strategy={last['strategy']}  score={last['score']:+.4f}  "
            f"val_net={last['val_net']:+.2f}")
        log(f"   What worked : {'; '.join(last.get('what_worked', []))}")
        for h in last.get("hypotheses_for_next_agent", []):
            log(f"   → {h}")

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
                log(f"   Next ideas : {summary['hypotheses_for_next_agent'][0] if summary['hypotheses_for_next_agent'] else '—'}")
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
