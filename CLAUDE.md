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
| 1 | 41 | `random_40` | linear, lr=0.0954, entropy=0.354, k=20, ep=16k | $+1452.00 | 50.0% | $-8.00 | 0.0% | 661d6b3 |
| 2 | 2 | `lr_high` | linear, lr=0.02 | $+122.50 | 25.0% | $+122.50 | 25.0% | e4a8913 |
| 3 | 116 | `random_115` | linear, lr=0.0792, entropy=0.005, k=8, ep=16k | $+72.00 | 50.0% | $-6.00 | 0.0% | — |
| 4 | 138 | `random_137` | linear, lr=0.0607, entropy=0.059, k=20, ep=4k | $+60.70 | 50.0% | $+5.40 | 50.0% | — |
| 5 | 227 | `random_226` | mlp, h=128, lr=0.0301, entropy=0.068, k=50, ep=6k | $+60.00 | 50.0% | $-30.00 | 0.0% | — |
| 6 | 120 | `random_119` | mlp, h=128, lr=0.0451, entropy=0.052, k=20, ep=16k | $+30.30 | 100.0% | $-18.00 | 0.0% | — |
| 7 | 109 | `random_108` | linear, lr=0.0467, entropy=0.006, k=32, ep=10k | $+52.00 | 50.0% | $-10.00 | 0.0% | — |
| 8 | 136 | `random_135` | linear, lr=0.0252, entropy=0.288, k=50, ep=16k | $+15.30 | 100.0% | $-18.00 | 0.0% | — |
| 9 | 203 | `random_202` | mlp, h=32, lr=0.0007, entropy=0.001, k=8, ep=6k | $+35.00 | 50.0% | $-8.00 | 0.0% | — |
| 10 | 200 | `random_199` | mlp, h=32, lr=0.0377, entropy=0.001, k=32, ep=4k | $+26.00 | 50.0% | $-8.00 | 0.0% | — |
| 11 | 91 | `random_90` | mlp, h=64, lr=0.0010, entropy=0.033, k=16, ep=4k | $+24.00 | 50.0% | $-14.00 | 0.0% | — |
| 12 | 56 | `random_55` | linear, lr=0.0027, entropy=0.010, k=20, ep=6k | $-6.70 | 50.0% | $-16.00 | 0.0% | — |
| 13 | 144 | `random_143` | mlp, h=32, lr=0.0035, entropy=0.003, k=16, ep=6k | $-8.70 | 50.0% | $-9.00 | 0.0% | — |
| 14 | 57 | `random_56` | linear, lr=0.0328, entropy=0.168, k=32, ep=4k | $-3.00 | 0.0% | $-32.00 | 0.0% | — |
| 15 | 121 | `random_120` | linear, lr=0.0018, entropy=0.102, k=20, ep=16k | $-3.00 | 0.0% | $-14.00 | 0.0% | — |
| 16 | 48 | `random_47` | linear, lr=0.0028, entropy=0.011, k=8, ep=16k | $-3.50 | 0.0% | $-4.00 | 0.0% | — |
| 17 | 178 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-4.00 | 0.0% | $-15.00 | 0.0% | — |
| 18 | 78 | `random_77` | mlp, h=64, lr=0.0275, entropy=0.002, k=20, ep=4k | $-4.50 | 0.0% | $-12.00 | 0.0% | — |
| 19 | 33 | `random_32` | linear, lr=0.0283, entropy=0.056, k=20, ep=6k | $-5.00 | 0.0% | $-12.00 | 0.0% | — |
| 20 | 34 | `random_33` | linear, lr=0.0210, entropy=0.245, k=32, ep=4k | $-5.00 | 0.0% | $-24.00 | 0.0% | — |
<!-- LEADERBOARD_END -->
