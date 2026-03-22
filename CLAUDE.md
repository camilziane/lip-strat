# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Maximise profit for a Loto Foot betting agent. Each dataset lives in `datasets/<name>/` — read its `README.md` for the current leaderboard, what has been tried, and what to explore next.

## Commands

```bash
uv sync                          # install dependencies

# Continuous improvement (the main command)
uv run python improve.py --dataset-dir datasets/<name>
uv run python improve.py --dataset-dir datasets/<name> --max-trials 30
uv run python improve.py --dataset-dir datasets/<name> \
    --init-model datasets/<other>/agent_best.npz   # warm-start from another dataset

# Live monitoring dashboard → http://localhost:8765
uv run python dashboard.py

# Single training run
uv run python train.py --episodes 10000 --lr 0.02 --entropy-coef 0.1

# Inference on new data
uv run python predict.py --model datasets/<name>/agent_best.npz \
                          --file datasets/<name>/<file>.xlsx
```

## Architecture

```
Excel → load_data() → list[grid_dict]   (sorted chronologically)
                           │
              ┌────────────┴────────────┐
         LotoFootEnv               REINFORCEAgent
         (env.py)                  (train.py)
              │                         │
         observation (n×10)        _PolicyNet (PyTorch)
              │                    logits → sigmoid → Bernoulli mask
         step(binary_mask)              │
              └── reward = (earnings − n_combos) / n_combos ───┘
```

**Observation** — `n_matches × 10` features per match:
`implied_p1, implied_pN, implied_p2, rep1, repN, rep2, margin, log_spread, value1, value2`

**Action** — `n_matches × 3` binary mask (system bet).
Total grids played = product of selections per match, capped at `k_max`.

**Reward** — mean per-grid net profit: `(total_earnings − n_combos × $1) / n_combos`

**Policy** — `_PolicyNet`: linear (`hidden_dim=0`) or one-hidden-layer MLP (`hidden_dim>0`), Bernoulli sigmoid outputs, trained with REINFORCE + entropy bonus + Adam + EMA baseline.

**Data split** — chronological: oldest rounds → train, most recent → val → test.

**Persistence** — `.npz` files store weights + hyperparams. `REINFORCEAgent.load()` handles both new (torch state dict) and legacy (numpy W/b) formats.

## improve.py — the autonomous loop

- Tries `PREDEFINED` strategies first, then infinite random search.
- **Improvement criterion**: `score = val_net_per_round + 50 × val_round_hit_rate`
- On new best: saves `agent_best.npz`, updates the dataset `README.md` leaderboard, commits.
- Restartable: reads `leaderboard.json` on startup to resume from where it left off.
- All outputs (model, leaderboard, log, figures) go into the dataset folder.

## Autoresearch loop

To run Claude as an autonomous researcher (modifies code, runs experiments, keeps or reverts):

```bash
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
```

Then give Claude this prompt:

```
Read CLAUDE.md and datasets/<name>/README.md for full context, then start the autoresearch loop:
1. Create results.tsv with header: commit	score	memory_gb	status	description
2. Run baseline: uv run python improve.py --dataset-dir datasets/<name> --max-trials 5 > run.log 2>&1
3. Loop forever: hypothesis → modify train.py or improve.py → git commit → run → record → keep or revert.
   Keep if score improves, git reset --hard HEAD~1 otherwise. Never stop.
```

**Rules:**
- Modify `train.py` and `improve.py` freely
- Do not modify `env.py` (environment contract) or `dashboard.py`
- `results.tsv` stays untracked (never commit it)
- Never stop or ask for confirmation
