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
| 1 | 22 | `mlp_128_k50` | mlp, h=128, k=50 | $+93.98 | 5.9% | $-27.00 | 0.0% | 19490eb |
| 2 | 13 | `mlp_32` | mlp, h=32 | $+25.79 | 5.9% | $-16.00 | 0.0% | ed4e24d |
| 3 | 9 | `k32` | linear, k=32 | $+5.85 | 5.9% | $-1.00 | 0.0% | d5b5e48 |
| 4 | 24 | `mlp_256_hlr` | mlp, h=256, lr=0.02 | $-15.88 | 0.0% | $+2.28 | 11.8% | — |
| 5 | 25 | `mlp_256_k8` | mlp, h=256, k=8 | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 6 | 8 | `k8` | linear, k=8 | $-7.06 | 0.0% | $-7.53 | 0.0% | 7278425 |
| 7 | 17 | `linear_hlr_ep12k` | linear, lr=0.02, ep=12k | $-7.50 | 5.9% | $-18.00 | 0.0% | — |
| 8 | 5 | `entropy_low` | linear, entropy=0.01 | $-12.11 | 5.9% | $-18.00 | 0.0% | 62cf919 |
| 9 | 27 | `entropy_low_s5` | linear, entropy=0.01, seeds=5 | $-12.11 | 5.9% | $-18.00 | 0.0% | — |
| 10 | 14 | `mlp_64` | mlp, h=64 | $-16.00 | 0.0% | $-16.00 | 0.0% | — |
| 11 | 11 | `ep_12k` | linear, ep=12k | $-16.82 | 0.0% | $-16.94 | 0.0% | — |
| 12 | 7 | `entropy_zero` | linear, entropy=0 | $-16.94 | 0.0% | $-17.53 | 0.0% | — |
| 13 | 4 | `lr_very_high` | linear, lr=0.05 | $-18.00 | 0.0% | $-17.29 | 0.0% | bfcc030 |
| 14 | 6 | `entropy_high` | linear, entropy=0.2 | $-17.29 | 0.0% | $-18.00 | 0.0% | — |
| 15 | 20 | `linear_ep16k` | linear, ep=16k | $-17.53 | 0.0% | $-17.76 | 0.0% | — |
| 16 | 2 | `lr_high` | linear, lr=0.02 | $-17.65 | 0.0% | $-18.00 | 0.0% | b0a09ff |
| 17 | 12 | `ep_20k` | linear, ep=20k | $-17.65 | 0.0% | $-18.00 | 0.0% | — |
| 18 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-17.76 | 0.0% | $-18.00 | 0.0% | 675174d |
| 19 | 3 | `lr_low` | linear, lr=0.001 | $-18.00 | 0.0% | $-18.00 | 0.0% | — |
| 20 | 15 | `mlp_128` | mlp, h=128 | $-18.00 | 0.0% | $-18.00 | 0.0% | — |
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
| 7 | 13 | `mlp_32` | +6.4 | $+25.79 | 6% | $-16.00 | 0% | ed4e24d |
| 8 | 22 | `mlp_128_k50` | +35.0 | $+93.98 | 6% | $-27.00 | 0% | 19490eb |
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

**Batch 2 analysis (trials 6-15):**
Major finding: mlp_32 (h=32 MLP, default params) achieved val_net=+25.79/round - genuinely profitable on val! score=+6.36. Also k32 (linear, k=32) got +3.90 with test_net=-1.00. k50 was very bad (-39.26): too many combos, too expensive. The MLP nonlinearity captures feature interactions that linear policy cannot. The k=32 sweep shows more combos helps hit winning rounds without being too expensive (unlike k=50 which costs too much when no prize).

**Next steps (priority order):**
1. mlp32 + k=32: combine the two best findings
2. mlp32 + k=32 + multi-seed (5): reduce init variance
3. mlp32 + entropy=0.01 + k=32: combine low entropy with MLP and more combos
4. mlp32 + longer training (12k-20k episodes): let the MLP converge better
5. mlp32 + lr sweep: find optimal learning rate for h=32 MLP
6. entropy annealing with MLP: start broad, focus on best outcomes
7. Dense reward with mlp32: correctness bonus to address sparse rewards
