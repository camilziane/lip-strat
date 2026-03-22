# lfp-strat

Reinforcement learning agent for Loto Foot betting. Trains a REINFORCE policy (PyTorch) to pick system bets that maximise net profit.

## Quick start

```bash
uv sync
uv run python dashboard.py    # open http://localhost:8765
```

Launch training on a dataset from the **Commencer l'entraînement** modal, or directly:

```bash
uv run python improve.py --dataset-dir datasets/loto-foot-8
```

## Datasets

Each dataset lives in `datasets/<name>/` and contains:

- the `.xlsx` data file
- `README.md` — leaderboard, what has been tried, what to explore next (auto-updated)
- `agent_best.npz` — best model found so far

```
datasets/
  loto-foot-7-2026/
  loto-foot-8/
  loto-foot-8/
```

## Autonomous research with Claude Code

Two ways to run Claude autonomously:

### /loop — Claude improves code while improve.py runs

```
! uv run python improve.py --dataset-dir datasets/loto-foot-8 &
! uv run python dashboard.py --no-browser &
```

```
/loop 15m read datasets/loto-foot-8/README.md, understand what's working, then improve train.py or improve.py. Commit changes.
```

To clear all state and start fresh:

```bash
uv run python improve.py --dataset-dir datasets/loto-foot-8 --reset
```

### autoresearch — fully autonomous experiment loop

```
! git checkout -b autoresearch/mar22
```

```
Read CLAUDE.md and datasets/loto-foot-8/README.md for full context, then start the autoresearch loop:
1. Create results.tsv with header: commit	score	memory_gb	status	description
2. Run baseline: uv run python improve.py --dataset-dir datasets/loto-foot-8 --max-trials 5 > run.log 2>&1
3. Loop forever: hypothesis → modify train.py or improve.py → git commit → run → record → keep or revert. Never stop.
```
