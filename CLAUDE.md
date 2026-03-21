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
uv run python improve.py                    # run forever
uv run python improve.py --max-trials 30   # stop after N trials
uv run python improve.py --test-grids 4   # hold out last 4 rounds

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
| Rank | Trial | Strategy | Keywords | Test Net/Round | Hit Rate | Train Net/Round | Commit |
|------|-------|----------|----------|---------------|----------|-----------------|--------|
| 1 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-15.00 | 0.0% | $-14.50 | — |
<!-- LEADERBOARD_END -->
