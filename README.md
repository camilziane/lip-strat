# lfp-strat

Reinforcement learning agent for Loto Foot betting. Trains a REINFORCE policy to pick system bets that maximise net profit.

## Quick start

```bash
uv sync
uv run python dashboard.py    # open http://localhost:8765
```

Use the dashboard to launch training on a dataset via the **Commencer l'entraînement** modal.

## Real-time improvement with Claude Code

Two loops run in parallel: `improve.py` tries strategies continuously, and Claude reads the results and edits the code itself.

**Step 1 — start the loops in the background** (type in Claude Code's prompt)

```
! uv run python improve.py --dataset-dir datasets/loto-foot-8-2026-Q1 &
! uv run python dashboard.py --no-browser &
```

**Step 2 — have Claude monitor and improve on a schedule** (type in Claude Code's prompt)

```
/loop 15m read the leaderboard in datasets/loto-foot-8-2026-Q1/README.md, understand what's working and what isn't, then add promising new strategies to PREDEFINED in improve.py or modify the reward function / features in train.py or env.py. Commit any changes.
```

- **Inner loop**: `improve.py` tries hyperparameter/architecture variants (a few minutes per trial)
- **Outer loop**: Claude reads results every 15 min and edits code (new features, reward shaping, architectures), which `improve.py` picks up automatically
- **Dashboard**: `http://localhost:8765` — live charts, log, and predictions

## Autonomous research loop (autoresearch)

A deeper mode where Claude acts as an autonomous researcher: it forms hypotheses, modifies the architecture or algorithm, runs experiments, and keeps or reverts changes based on results — all without human input.

### Setup

```
# 1. Propose a run tag based on today's date, e.g. mar22
git checkout -b autoresearch/mar22

# 2. Create results.tsv (untracked — never commit it)
```

`results.tsv` columns (tab-separated):

```
commit	score	memory_gb	status	description
```

- `commit` — 7-char git hash
- `score` — `val_net_per_round + 50 × val_round_hit_rate` (higher is better); use `0.000000` for crashes
- `memory_gb` — leave `0.0` (no GPU constraint here)
- `status` — `keep`, `discard`, or `crash`
- `description` — short description of what was tried

### Files

| File | Role |
|------|------|
| `train.py` | **Modify freely** — agent architecture, optimizer, features, reward shaping |
| `improve.py` | **Modify freely** — PREDEFINED strategies, random search space |
| `env.py` | **Read-only** — environment, reward computation, observation space |
| `datasets/` | **Read-only** — raw data |

### Experiment loop

```
uv run python improve.py --dataset-dir datasets/loto-foot-8-2026-Q1 --max-trials 5 > run.log 2>&1
```

Extract the result:

```bash
grep "score=" run.log | tail -3
# or read the leaderboard directly:
cat datasets/loto-foot-8-2026-Q1/leaderboard.json | python3 -c \
  "import json,sys; b=json.load(sys.stdin); e=max(b,key=lambda x:x['score']); print(e['score'], e['strategy_name'])"
```

**Loop logic:**
- First run: baseline (run as-is, record result)
- Each subsequent run: modify `train.py` or `improve.py`, `git commit`, run, record
- If `score` improved → keep the commit, advance the branch
- If `score` equal or worse → `git reset --hard HEAD~1`, log `discard`
- If crash → fix if trivial, otherwise log `crash` and move on
- **Never stop** — keep iterating until manually interrupted

### Directions to explore

| Direction | Where | Why |
|-----------|-------|-----|
| Richer observation features | `train.py` `load_data()` | More signal from odds + crowd % |
| Reward shaping | `train.py` `update()` | Scale advantage by `1/implied_prob` to back upsets |
| Deeper MLP (2 hidden layers) | `train.py` `_PolicyNet` | More expressive policy |
| Value baseline network | `train.py` | Lower REINFORCE variance |
| LR schedule (cosine decay) | `train.py` `REINFORCEAgent` | Better convergence |
| Match-order augmentation | `train.py` training loop | More training diversity from limited data |
| Ensemble (5 seeds, threshold) | `improve.py` `train_trial()` | Reduce single-seed variance |
| Optimistic k strategy | `improve.py` PREDEFINED | Higher k on rounds where model is confident |

### The metric

```
score = val_net_per_round + 50 × val_round_hit_rate
```

Higher is better. The `×50` weight means one rang2 hit on a val round (+~$150/round) outweighs many marginal cost savings. Focus on hit rate, not just net.

---

## Datasets

Each dataset lives in its own subfolder under `datasets/`:

```
datasets/
  loto-foot-7-2026-Q1/
  loto-foot-8-2026-Q1/
  loto-foot-8-2026-Q2/
```

Each folder contains the `.xlsx` file, a `README.md` with the per-dataset leaderboard (auto-updated on each new best), and the best model (`agent_best.npz`).

## Manual commands

```bash
# Continuous improvement
uv run python improve.py --dataset-dir datasets/loto-foot-8-2026-Q1
uv run python improve.py --dataset-dir datasets/loto-foot-8-2026-Q1 --max-trials 50
uv run python improve.py --dataset-dir datasets/loto-foot-8-2026-Q2 \
                          --init-model datasets/loto-foot-8-2026-Q1/agent_best.npz

# Single training run
uv run python train.py --episodes 10000 --lr 0.02 --entropy-coef 0.1

# Inference
uv run python predict.py --model datasets/loto-foot-8-2026-Q1/agent_best.npz \
                          --file datasets/loto-foot-8-2026-Q1/<file>.xlsx
```
