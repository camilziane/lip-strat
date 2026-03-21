# Dataset: loto-foot-8 — 2026 Q1

## Data
- **File**: `loto-foot-8_grilles-2026-01-31_au_2026-03-04_grille-joueur_top50_rang-2026-02-19_au_2026-03-21_train.xlsx`
- **Grid type**: loto-foot-8 (8 matches per round)
- **Date range**: 2026-01-31 → 2026-03-21
- **Total rounds**: 12
- **Default split**: 8 train / 2 val / 2 test (random, seed=42)

## Prize structure
- rang1 (8/8 correct): ~$2 000–$5 000
- rang2 (7/8 correct): ~$100–$400
- rang3/rang4: smaller prizes for some rounds

## Notes
<!-- Add dataset-specific notes here — observations about team strengths, league patterns, etc. -->

## Key learnings
<!-- Auto-populated by improve.py on each new best. Do not edit between the markers. -->

<!-- LEADERBOARD_START -->
| Rank | Trial | Strategy | Keywords | Val Net/Round | Val Hit% | Test Net/Round | Test Hit% | Commit |
|------|-------|----------|----------|--------------|----------|----------------|-----------|--------|
| 1 | 9 | `k8` | linear, k=8 | $-8.00 | 0.0% | $+49.85 | 50.0% | 0174921 |
| 2 | 5 | `lr_very_high` | linear, lr=0.05 | $-10.00 | 0.0% | $-12.00 | 0.0% | 7732256 |
| 3 | 7 | `entropy_high` | linear, entropy=0.2 | $-10.50 | 0.0% | $-14.00 | 0.0% | — |
| 4 | 4 | `lr_low` | linear, lr=0.001 | $-11.00 | 0.0% | $-6.00 | 0.0% | cb9baf2 |
| 5 | 2 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-12.00 | 0.0% | $-12.00 | 0.0% | 61a9605 |
| 6 | 3 | `lr_high` | linear, lr=0.02 | $-14.00 | 0.0% | $-14.00 | 0.0% | — |
| 7 | 8 | `entropy_zero` | linear, entropy=0 | $-15.00 | 0.0% | $-17.00 | 0.0% | — |
| 8 | 1 | `baseline` | linear, lr=0.005, entropy=0.05, k=20, ep=6k | $-16.00 | 0.0% | $-16.00 | 0.0% | 8140806 |
| 9 | 6 | `entropy_low` | linear, entropy=0.01 | $-16.00 | 0.0% | $-16.00 | 0.0% | — |
| 10 | 11 | `k50` | linear, k=50 | $-16.00 | 0.0% | $-30.00 | 0.0% | — |
| 11 | 12 | `ep_12k` | linear, ep=12k | $-16.00 | 0.0% | $-7.00 | 0.0% | — |
| 12 | 10 | `k32` | linear, k=32 | $-21.00 | 0.0% | $+44.85 | 50.0% | — |
<!-- LEADERBOARD_END -->
