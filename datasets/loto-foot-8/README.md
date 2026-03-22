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
         observation (8×10)        _PolicyNet (PyTorch)
              │                    logits → sigmoid → Bernoulli mask
         step(binary_mask)              │
              └── reward = (earnings − n_combos) / n_combos ───┘
```

**Observation** — `n_matches × 10` per match:
`implied_p1, implied_pN, implied_p2, rep1, repN, rep2, margin, log_spread, value1, value2`

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
| 1 | 11 | `entropy_high` | linear, entropy=0.2 | $-17.53 | 0.0% | $+9.74 | 6.0% | 0513d7a |
| 2 | 23 | `k8` | linear, k=8 | $-6.00 | 0.0% | $-6.00 | 0.0% | — |
| 3 | 38 | `k8` | linear, k=8 | $-6.12 | 0.0% | $-6.00 | 0.0% | — |
| 4 | 13 | `k8` | linear, k=8 | $-6.35 | 0.0% | $-6.47 | 0.0% | — |
| 5 | 42 | `ep_20k` | linear, ep=20k | $-5.62 | 6.0% | $-15.88 | 0.0% | — |
| 6 | 17 | `lr_high` | linear, lr=0.02 | $-7.50 | 6.0% | $-18.00 | 0.0% | — |
| 7 | 28 | `mlp_32` | mlp, h=32 | $-12.00 | 0.0% | $-12.00 | 0.0% | — |
| 8 | 12 | `entropy_zero` | linear, entropy=0 | $-11.76 | 6.0% | $-17.29 | 0.0% | — |
| 9 | 20 | `entropy_low` | linear, entropy=0.01 | $-17.29 | 0.0% | $-11.85 | 6.0% | — |
| 10 | 5 | `entropy_low` | linear, entropy=0.01 | $-12.11 | 6.0% | $-18.00 | 0.0% | bbdcac2 |
<!-- LEADERBOARD_END -->

---

## Improvement History

<!-- HISTORY_START -->
| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|
| 1 | 1 | `baseline` | -17.88 | $-17.76 | 0% | $-18.00 | 0% | f31f09b |
| 2 | 2 | `lr_high` | -17.82 | $-17.65 | 0% | $-18.00 | 0% | fc5d7a6 |
| 3 | 4 | `lr_very_high` | -17.65 | $-18.00 | 0% | $-17.29 | 0% | 928810f |
| 4 | 5 | `entropy_low` | -13.59 | $-12.11 | 6% | $-18.00 | 0% | bbdcac2 |
| 5 | 11 | `entropy_high` | -2.42 | $-17.53 | 0% | $+9.74 | 6% | 0513d7a |
<!-- HISTORY_END -->

---

## What Was Tried

| Commit | Hypothesis | Result | Decision |
|--------|-----------|--------|----------|
| bbdcac2 | Baseline PREDEFINED sweep (5 trials) | Best: entropy_low −13.59; all strategies lose on test at k=20 | Keep |
| 767dc53 | Multi-seed best-of-N (n_seeds=5): reduce init-seed variance | Multi-seed strategies never reached in short --max-trials runs (placed too late in PREDEFINED); inconclusive | Kept code, needs longer run |
| 0513d7a | Continued PREDEFINED sweep | entropy_high (entropy=0.2) made money on test (+9.74/round); suggests high-entropy/random coverage helps | Keep |

---

## Key Observations

- **All strategies lose money most of the time** — prizes are sparse (rang1 requires 8/8 correct)
- **k=8 is the most consistent** — loses ~6€/round on both val and test, no surprise wins but no big losses
- **High entropy (0.2) is the current best** — more random coverage occasionally hits prizes; test_hit=6% (1/17 rounds won)
- **Val/test split is very noisy** — 17 rounds each means 1 winning round = 6% hit rate; results have high variance
- **Multi-seed strategies not yet properly tested** — they sit late in PREDEFINED and need longer runs to reach

---

## What to Try Next

1. **entropy_high + k=32** — combine high entropy coverage with more combinations per round
2. **Run multi-seed strategies** — run `--max-trials 30` to reach k32_s5, entropy_low_s5 etc.
3. **Rethink reward signal** — current reward is sparse; shaped reward (partial correctness bonus) could help learning
4. **Ask user for more features** — current obs uses only odds + crowd %. Historical hit rates or team form could add signal
5. **Value-based approach** — instead of RL, fit a model to predict P(prize | selections) directly from odds
