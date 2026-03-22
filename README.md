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
