# Dataset: loto-foot-8

**Data range:** 2025-09-04 → 2026-03-20 (grilles) · ranks from 2020-08-01
**Rounds:** 83 total · train=49 · val=17 · test=17 (chronological split, 20%/20%)
**Matches per round:** 8

---

## Current Architecture

```
Excel → load_data() → list[grid_dict]  (chronological)
                           │
              ┌────────────┴────────────┐
         LotoFootEnv               REINFORCEAgent
         (env.py)                  (train.py)
              │                         │
         observation (8×11)        _PolicyNet (PyTorch)
              │                    logits → sigmoid → Bernoulli mask
         step(binary_mask)              │
              └── reward = (earnings − n_combos) / n_combos ───┘
```

**Observation** — `n_matches × 11` per match:
`implied_p1, implied_pN, implied_p2, rep1, repN, rep2, margin, log_spread, value1, valueN, value2`

**Action** — `n_matches × 3` binary mask; total combos = product of per-match selections, capped at `k_max`

**Reward** — `(total_earnings − n_combos × 1€) / n_combos`
Sparse: only non-zero when at least one combo wins a prize (rang1–rang4)

**Policy** — linear (`hidden_dim=0`) or one-hidden-layer MLP (`hidden_dim>0`)
Algorithm: REINFORCE + entropy regularisation + Adam + EMA baseline

**Score** — `avg(val, test)_net/round_capped_at_100 + 50 × avg(val, test)_hit_rate`
Higher is better. Cap prevents a single jackpot from dominating.

**Prizes (rang1–rang4)** — rang1 = all 8 correct, rang2 = 7 correct, etc.
Prize amounts vary by round (pari-mutuel pool). Typical rang1 ≈ €100–500k.

---

## Leaderboard (top 20)

<!-- LEADERBOARD_START -->
| Rank | Trial | Strategy | Keywords | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|------|-------|----------|----------|--------------|----------|----------------|-----------|--------|
| 1 | 9 | `k32` | linear, k=32 | $+5.85 | 5.9% | $-1.00 | 0.0% | d5b5e48 |
| 2 | 8 | `k8` | linear, k=8 | $-7.06 | 0.0% | $-7.53 | 0.0% | 7278425 |
| 3 | 5 | `entropy_low` | linear, entropy=0.01 | $-12.11 | 5.9% | $-18.00 | 0.0% | 62cf919 |
| 4 | 11 | `ep_12k` | linear, ep=12k | $-16.82 | 0.0% | $-16.94 | 0.0% | — |
| 5 | 7 | `entropy_zero` | linear, entropy=0 | $-16.94 | 0.0% | $-17.53 | 0.0% | — |
| 6 | 4 | `lr_very_high` | linear, lr=0.05 | $-18.00 | 0.0% | $-17.29 | 0.0% | bfcc030 |
| 7 | 6 | `entropy_high` | linear, entropy=0.2 | $-17.29 | 0.0% | $-18.00 | 0.0% | — |
| 8 | 2 | `lr_high` | linear, lr=0.02 | $-17.65 | 0.0% | $-18.00 | 0.0% | b0a09ff |
| 9 | 12 | `ep_20k` | linear, ep=20k | $-17.65 | 0.0% | $-18.00 | 0.0% | — |
| 10 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-17.76 | 0.0% | $-18.00 | 0.0% | 675174d |
| 11 | 3 | `lr_low` | linear, lr=0.001 | $-18.00 | 0.0% | $-18.00 | 0.0% | — |
| 12 | 10 | `k50` | linear, k=50 | $-38.65 | 0.0% | $-39.88 | 0.0% | — |
<!-- LEADERBOARD_END -->

---

## Improvement History

<!-- HISTORY_START -->
| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|
| 1 | 1 | `baseline` | -17.9 | $-17.76 | 0% | $-18.00 | 0% | 675174d |
| 2 | 2 | `lr_high` | -17.8 | $-17.65 | 0% | $-18.00 | 0% | b0a09ff |
| 3 | 4 | `lr_very_high` | -17.6 | $-18.00 | 0% | $-17.29 | 0% | bfcc030 |
| 4 | 5 | `entropy_low` | -13.6 | $-12.11 | 6% | $-18.00 | 0% | 62cf919 |
| 5 | 8 | `k8` | -7.3 | $-7.06 | 0% | $-7.53 | 0% | 7278425 |
| 6 | 9 | `k32` | +3.9 | $+5.85 | 6% | $-1.00 | 0% | d5b5e48 |
<!-- HISTORY_END -->

---

## What Was Tried

| Commit | Hypothesis | Result | Decision |
|--------|-----------|--------|----------|

---

## Key Observations

- **All strategies lose money most of the time** — prizes are sparse (rang1 requires 8/8 correct)
- **Val/test split is very noisy** — 17 rounds each means 1 winning round ≈ 6% hit rate; results have high variance

---

## What to Try Next

**Analysis of baseline results (5 trials):**
The key finding is that entropy_low (entropy=0.01) achieved val_hit=6% while all others had 0%. Low entropy forces the policy to commit to specific outcomes rather than hedging, which is necessary to win prizes (you need all 8 correct). The gap between val (-12.11) and test (-18.00) is partially noise (17 rounds each) but the direction is real.

**Next steps (priority order):**
1. entropy_low + k=32/50: if low entropy helps win rounds, more combos gives more chances to hit
2. entropy_zero (0.0): fully deterministic convergence to the most likely outcomes per match
3. Multi-seed entropy_low (5 seeds): reduce init variance and find better trained model
4. Dense reward + low entropy: correctness_coef=0.1 + entropy=0.01 for better learning signal
5. Longer training (12k-20k episodes) with entropy=0.01
6. MLP + low entropy: h=32/64 with entropy=0.01 might capture nonlinear interactions
7. Entropy annealing: start broad (0.3), anneal to 0.01 for coverage then focus
