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
| 1 | 4 | `lr_very_high` | linear, lr=0.05 | $-18.00 | 0.0% | $-17.29 | 0.0% | bfcc030 |
| 2 | 2 | `lr_high` | linear, lr=0.02 | $-17.65 | 0.0% | $-18.00 | 0.0% | b0a09ff |
| 3 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-17.76 | 0.0% | $-18.00 | 0.0% | 675174d |
| 4 | 3 | `lr_low` | linear, lr=0.001 | $-18.00 | 0.0% | $-18.00 | 0.0% | — |
<!-- LEADERBOARD_END -->

---

## Improvement History

<!-- HISTORY_START -->
| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|
| 1 | 1 | `baseline` | -17.9 | $-17.76 | 0% | $-18.00 | 0% | 675174d |
| 2 | 2 | `lr_high` | -17.8 | $-17.65 | 0% | $-18.00 | 0% | b0a09ff |
| 3 | 4 | `lr_very_high` | -17.6 | $-18.00 | 0% | $-17.29 | 0% | bfcc030 |
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

1. **Entropy sweep** — try entropy_coef ∈ {0.0, 0.01, 0.05, 0.2} to balance exploration vs exploitation
2. **k_max sweep** — try k=8 (consistent loss) vs k=32/50 (more coverage, higher hit chance)
3. **Multi-seed (n_seeds=5)** — reduce init-seed variance by keeping the best of 5 seeds
4. **Dense reward** — add correctness_coef bonus to address sparse reward signal
5. **Player consensus features** — run with `--extra-features player_consensus` to add top-50 player picks
