# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Maximise test profit for a Loto Foot betting agent. The loop is simple:
1. Run `uv run python improve.py` — it tries strategies continuously, commits new bests automatically.
2. When you resume, the leaderboard is already updated. Read it, understand what helped, and push further.
3. **Never break the loop** — if you change architecture or training code, verify it runs before committing.

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

- The last `au_YYYY-MM-DD` in the filename is the training cutoff date (auto-parsed).
- Each grid = one round; rows = matches within that round.
- 12 complete rounds currently (8 train / 4 test with default `--test-grids 4`).
- Key columns: `Cote 1/2/3` (odds home/draw/away), `Rep 1/2/3` (crowd %), `Score`, `rang1`–`rang4` (prizes).
- `rang1` = prize for 0 errors, `rang2` = 1 error, etc. (same value repeated across all match rows).

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

| Direction | What to try | Why it might help |
|-----------|-------------|-------------------|
| **Richer features** | Add home/away form, league encoding, head-to-head | More signal for the policy |
| **Reward shaping** | Scale reward by `1/implied_prob` of the correct outcome | Encourage backing upsets |
| **Deeper MLP** | 2 hidden layers, batch norm | More expressive policy |
| **Value baseline** | Separate value network for advantage estimation | Lower REINFORCE variance |
| **Coverage loss** | Penalise selecting the same outcome for all matches | Force diverse grids |
| **Ensemble** | Average predictions from 5 seeds, threshold ensemble logits | Reduce single-seed variance |
| **LR schedule** | Cosine decay or step decay | Better convergence |
| **Data augmentation** | Permute match order within a round | More training diversity |

## Improvement guidelines

- The **primary bottleneck is data size** (12 rounds). Once the real dataset arrives, retrain from scratch.
- Watch `train_net_per_round` vs `test_net_per_round` gap — large gap = overfitting; increase `entropy_coef` or reduce `episodes`.
- Rang1 hits are extremely rare (prob ≈ (1/3)^8 ≈ 0.015%). Focus on rang2 hit rate as a short-term signal.
- `k_max` controls cost vs coverage tradeoff. Higher k = more coverage but more cost per round.
- Score formula weighs round hit rate heavily (`×50`) because even one rang2 hit on 4 test rounds is significant.

## Leaderboard

Auto-updated by `improve.py` on every new best. Sorted by `score = test_net_per_round + 50 × test_round_hit_rate`.

<!-- LEADERBOARD_START -->
| Rank | Trial | Strategy | Keywords | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|------|-------|----------|----------|--------------|----------|----------------|-----------|--------|
| 1 | 2 | `lr_high` | linear, lr=0.02 | $+122.50 | 25.0% | $+122.50 | 25.0% | e4a8913 |
| 2 | 33 | `random_32` | linear, lr=0.0283, entropy=0.056, k=20, ep=6k | $-5.00 | 0.0% | $-12.00 | 0.0% | — |
| 3 | 34 | `random_33` | linear, lr=0.0210, entropy=0.245, k=32, ep=4k | $-5.00 | 0.0% | $-24.00 | 0.0% | — |
| 4 | 28 | `random_27` | linear, lr=0.0117, entropy=0.024, k=8, ep=10k | $-6.00 | 0.0% | $-8.00 | 0.0% | — |
| 5 | 38 | `random_37` | mlp, h=128, lr=0.0022, entropy=0.002, k=8, ep=6k | $-6.00 | 0.0% | $-8.00 | 0.0% | — |
| 6 | 12 | `entropy_high` | linear, entropy=0.2 | $-7.00 | 0.0% | $-7.00 | 0.0% | — |
| 7 | 14 | `k8` | linear, k=8 | $-7.00 | 0.0% | $+12.40 | 50.0% | — |
| 8 | 15 | `k32` | linear, k=32 | $-8.00 | 0.0% | $-9.50 | 0.0% | — |
| 9 | 18 | `ep_20k` | linear, ep=20k | $-8.00 | 0.0% | $-18.00 | 0.0% | — |
| 10 | 22 | `mlp_64_hlr` | mlp, h=64, lr=0.02 | $-8.00 | 0.0% | $+42.85 | 50.0% | — |
| 11 | 25 | `mlp_32_k8` | mlp, h=32, k=8 | $-8.00 | 0.0% | $-8.00 | 0.0% | — |
| 12 | 29 | `random_28` | mlp, h=128, lr=0.0484, entropy=0.039, k=8, ep=10k | $-8.00 | 0.0% | $-7.00 | 0.0% | — |
| 13 | 35 | `random_34` | linear, lr=0.0032, entropy=0.371, k=32, ep=4k | $-8.00 | 0.0% | $-4.00 | 0.0% | — |
| 14 | 37 | `random_36` | linear, lr=0.0047, entropy=0.002, k=8, ep=10k | $-8.00 | 0.0% | $-7.00 | 0.0% | — |
| 15 | 36 | `random_35` | mlp, h=64, lr=0.0023, entropy=0.204, k=16, ep=16k | $-9.00 | 0.0% | $-16.00 | 0.0% | — |
| 16 | 5 | `lr_high` | linear, lr=0.02 | $-10.00 | 0.0% | $-12.00 | 0.0% | — |
| 17 | 11 | `entropy_low` | linear, entropy=0.01 | $-10.00 | 0.0% | $-14.00 | 0.0% | — |
| 18 | 39 | `random_38` | linear, lr=0.0942, entropy=0.001, k=20, ep=6k | $-10.00 | 0.0% | $-12.00 | 0.0% | — |
| 19 | 7 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-10.50 | 0.0% | $-14.00 | 0.0% | — |
| 20 | 4 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-11.00 | 0.0% | $-6.00 | 0.0% | — |
<!-- LEADERBOARD_END -->
