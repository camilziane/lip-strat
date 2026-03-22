# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Maximise test profit for a Loto Foot betting agent. The loop is simple:

1. [ ] Run `uv run python improve.py` — it tries strategies continuously, commits new bests automatically.
2. [ ] When you resume, the leaderboard is already updated. Read it, understand what helped, and push further.
3. [ ] **Never break the loop** — if you change architecture or training code, verify it runs before committing.

## Commands

```bash
uv sync                                     # install dependencies

# Continuous improvement (the main command)
uv run python improve.py                             # run forever
uv run python improve.py --max-trials 30             # stop after N trials
uv run python improve.py --val-ratio 0.2 --test-ratio 0.2  # custom split

# Live monitoring dashboard (open http://localhost:8765)
uv run python dashboard.py

# Single training run
uv run python train.py --save agent.npz
uv run python train.py --episodes 10000 --lr 0.02 --entropy-coef 0.1 --save agent.npz

# Inference on new data
uv run python predict.py --model agent_best.npz --file new_data.xlsx
uv run python predict.py --model agent_best.npz --file new_data.xlsx --round 42 --include-played
```

## Data

Excel file (auto-detected by filename pattern):
`loto-foot-8_grilles-..._au_YYYY-MM-DD_train.xlsx`

- [ ] The last `au_YYYY-MM-DD` in the filename is the training cutoff date (auto-parsed).
- [ ] Each grid = one round; rows = matches within that round.
- [ ] 12 complete rounds currently (8 train / 4 test with default `--test-grids 4`).
- [ ] Key columns: `Cote 1/2/3` (odds home/draw/away), `Rep 1/2/3` (crowd %), `Score`, `rang1`–`rang4` (prizes).
- [ ] `rang1` = prize for 0 errors, `rang2` = 1 error, etc. (same value repeated across all match rows).

## Architecture

```
Excel → load_data() → list[grid_dict]
                           │
              ┌────────────┴────────────┐
         LotoFootEnv               REINFORCEAgent / MLPREINFORCEAgent
         (env.py)                  (train.py)
              │                         │
         observation (n×10)        logits → sigmoid → Bernoulli mask
              │                         │
         step(binary_mask)         update() via REINFORCE + entropy bonus
              └──── reward = (earnings − n_combos) / n_combos ───┘
```

**Observation** (n_matches × 10 features per match):
`implied_p1, implied_pN, implied_p2, rep1, repN, rep2, margin, log_spread, value1, value2`

**Action** (n_matches × 3 binary): per-match multi-selection (system bet).
Total grids = product of selections per match (capped at `k_max`).

**Reward**: mean per-grid net profit = `(total_earnings − n_combos × $1) / n_combos`

**Agents**:

- `REINFORCEAgent` — linear policy `W @ obs + b`, Bernoulli sigmoid outputs
- `MLPREINFORCEAgent` — one hidden layer + ReLU, same Bernoulli output head

**Persistence**: `.npz` files contain weights + all hyperparams. `REINFORCEAgent.load()` auto-dispatches to MLP if `W1` key is present.

## improve.py — the autonomous loop

- Tries `PREDEFINED` strategies first (22 variants), then infinite random search.
- **Improvement criterion**: `score = test_net_per_round + 50 × test_round_hit_rate`
- On new best: saves `agent_best.npz`, commits `agent_best.npz + leaderboard.json + CLAUDE.md`.
- Logs everything to `improve.log`.
- Restartable: reads `leaderboard.json` on startup to resume from where it left off.

## Directions to explore (for the next agent)

When `improve.py` has exhausted `PREDEFINED`, consider adding to `PREDEFINED` in `improve.py`:

| Direction                   | What to try                                                 | Why it might help           |
| --------------------------- | ----------------------------------------------------------- | --------------------------- |
| **Richer features**   | Add home/away form, league encoding, head-to-head           | More signal for the policy  |
| **Reward shaping**    | Scale reward by `1/implied_prob` of the correct outcome   | Encourage backing upsets    |
| **Deeper MLP**        | 2 hidden layers, batch norm                                 | More expressive policy      |
| **Value baseline**    | Separate value network for advantage estimation             | Lower REINFORCE variance    |
| **Coverage loss**     | Penalise selecting the same outcome for all matches         | Force diverse grids         |
| **Ensemble**          | Average predictions from 5 seeds, threshold ensemble logits | Reduce single-seed variance |
| **LR schedule**       | Cosine decay or step decay                                  | Better convergence          |
| **Data augmentation** | Permute match order within a round                          | More training diversity     |

## Improvement guidelines

- The **primary bottleneck is data size** (12 rounds). Once the real dataset arrives, retrain from scratch.
- Watch `train_net_per_round` vs `test_netd çèktrbdhvcfsw_per_round` gap — large gap = overfitting; increase `entropy_coef` or reduce `episodes`.
- Rang1 hits are extremely rare (prob ≈ (1/3)^8 ≈ 0.015%). Focus on rang2 hit rate as a short-term signal.
- `k_max` controls cost vs coverage tradeoff. Higher k = more coverage but more cost per round.
- Score formula weighs round hit rate heavily (`×50`) because even one rang2 hit on 4 test rounds is significant.

## Leaderboard

Auto-updated by `improve.py` on every new best. Sorted by `score = test_net_per_round + 50 × test_round_hit_rate`.

<!-- LEADERBOARD_START -->
| Rank | Trial | Strategy | Keywords | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|------|-------|----------|----------|--------------|----------|----------------|-----------|--------|
| 1 | 13 | `mlp_32` | mlp, h=32 | $+14.02 | 11.8% | $-14.47 | 0.0% | — |
| 2 | 2 | `lr_high` | linear, lr=0.02 | $+12.51 | 11.8% | $-18.00 | 0.0% | — |
| 3 | 37 | `random_36` | mlp, h=128, lr=0.0391, entropy=0.016, k=20, ep=4k | $-7.29 | 5.9% | $-12.00 | 0.0% | — |
| 4 | 6 | `entropy_high` | linear, entropy=0.2 | $-10.44 | 11.8% | $-17.88 | 0.0% | — |
| 5 | 8 | `k8` | linear, k=8 | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 6 | 19 | `mlp_32_k8` | mlp, h=32, k=8 | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 7 | 25 | `best234_clone` | mlp, h=256, lr=0.0006, entropy=0.271, k=8, ep=16k | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 8 | 28 | `mlp_256_k8` | mlp, h=256, k=8 | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 9 | 34 | `random_33` | linear, lr=0.0051, entropy=0.208, k=8, ep=10k | $-6.00 | 0.0% | $-6.12 | 0.0% | — |
| 10 | 36 | `random_35` | mlp, h=32, lr=0.0054, entropy=0.010, k=8, ep=16k | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 11 | 44 | `random_43` | mlp, h=64, lr=0.0412, entropy=0.078, k=8, ep=6k | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 12 | 49 | `random_48` | linear, lr=0.0061, entropy=0.034, k=8, ep=16k | $-6.00 | 0.0% | $-6.24 | 0.0% | — |
| 13 | 41 | `random_40` | mlp, h=32, lr=0.0187, entropy=0.102, k=16, ep=16k | $-9.00 | 0.0% | $-2.04 | 5.9% | — |
| 14 | 47 | `random_46` | mlp, h=256, lr=0.0169, entropy=0.080, k=16, ep=16k | $-9.00 | 0.0% | $-9.00 | 0.0% | — |
| 15 | 48 | `random_47` | mlp, h=64, lr=0.0102, entropy=0.002, k=16, ep=4k | $-9.00 | 0.0% | $-9.00 | 0.0% | — |
| 16 | 51 | `random_50` | mlp, h=32, lr=0.0051, entropy=0.004, k=16, ep=6k | $-9.41 | 0.0% | $-9.18 | 0.0% | — |
| 17 | 12 | `ep_20k` | linear, ep=20k | $-12.69 | 5.9% | $-16.12 | 0.0% | — |
| 18 | 30 | `mlp_128_k50` | mlp, h=128, k=50 | $-12.00 | 0.0% | $-12.00 | 0.0% | — |
| 19 | 38 | `random_37` | linear, lr=0.0007, entropy=0.171, k=16, ep=10k | $-12.06 | 0.0% | $-13.12 | 0.0% | — |
| 20 | 18 | `mlp_128_hlr` | mlp, h=128, lr=0.02 | $-15.12 | 5.9% | $-0.54 | 11.8% | — |
<!-- LEADERBOARD_END -->

## Improvement History

Chronological record of every new best (each row = a git commit).

<!-- HISTORY_START -->
| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|
| 1 | 1 | `baseline` | -17.9 | $-17.88 | 0% | $-17.76 | 0% | b4475c0 |
<!-- HISTORY_END -->
