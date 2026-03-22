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
| 1 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-17.76 | 0.0% | $-18.00 | 0.0% | 20656d5 |
<!-- LEADERBOARD_END -->

---

## Improvement History

<!-- HISTORY_START -->
| # | Trial | Strategy | Score | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|---|-------|----------|-------|--------------|----------|----------------|-----------|--------|
| 1 | 1 | `baseline` | -17.9 | $-17.76 | 0% | $-18.00 | 0% | 20656d5 |
<!-- HISTORY_END -->

---

## What Was Tried

| Commit | Hypothesis | Result | Decision |
|--------|-----------|--------|----------|

---

## Key Observations

- **All strategies lose money most of the time** — prizes are sparse (rang1 requires 8/8 correct)
- **k=8 is the most consistent** — loses ~6€/round on both val and test
- **Val/test split is very noisy** — 17 rounds each means 1 winning round = 6% hit rate; results have high variance

---

## What to Try Next

1. **entropy_high + k=32** — combine high entropy coverage with more combinations per round
2. **Run multi-seed strategies** — run `--max-trials 30` to reach k32_s5, entropy_low_s5 etc.
3. **Rethink reward signal** — current reward is sparse; shaped reward (partial correctness bonus) could help learning
4. **Ask user for more features** — current obs uses only odds + crowd %. Historical hit rates or team form could add signal
5. **Value-based approach** — instead of RL, fit a model to predict P(prize | selections) directly from odds
