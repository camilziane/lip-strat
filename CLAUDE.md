# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Maximise profit for a Loto Foot betting agent. Read `datasets/<name>/README.md` for the current leaderboard, what has been tried, and what to explore next.

## Commands

```bash
uv sync

uv run python improve.py --dataset-dir datasets/<name>             # run forever
uv run python improve.py --dataset-dir datasets/<name> --max-trials 30
uv run python improve.py --dataset-dir datasets/<name> \
    --init-model datasets/<other>/agent_best.npz                   # warm-start

uv run python dashboard.py                                         # http://localhost:8765
uv run python train.py --episodes 10000 --lr 0.02 --entropy-coef 0.1
uv run python predict.py --model datasets/<name>/agent_best.npz \
                          --file datasets/<name>/<file>.xlsx
```

## Architecture

```
Excel → load_data() → list[grid_dict]  (chronological)
                           │
              ┌────────────┴────────────┐
         LotoFootEnv               REINFORCEAgent
         (env.py)                  (train.py)
              │                         │
         observation (n×10)        _PolicyNet (PyTorch)
              │                    logits → sigmoid → Bernoulli mask
         step(binary_mask)              │
              └── reward = (earnings − n_combos) / n_combos ───┘
```

- **Observation** — `n_matches × 10`: `implied_p1, implied_pN, implied_p2, rep1, repN, rep2, margin, log_spread, value1, value2`
- **Action** — `n_matches × 3` binary mask; total grids = product of selections per match, capped at `k_max`
- **Reward** — `(total_earnings − n_combos × $1) / n_combos`
- **Policy** — linear (`hidden_dim=0`) or MLP (`hidden_dim>0`), REINFORCE + entropy + Adam + EMA baseline
- **Split** — chronological: oldest → train, most recent → val → test

## improve.py

- Tries `PREDEFINED` strategies first, then infinite random search
- Score: `avg(val,test)_net_capped_at_100 + 50 × avg(val,test)_hit_rate` (higher is better)
- On new best: saves `agent_best.npz`, updates `datasets/<name>/README.md`, commits
- Restartable from `leaderboard.json`; all outputs stay inside the dataset folder

## Autoresearch

Run on a dedicated branch. Modify only `train.py` and `improve.py` — never `env.py` or `dashboard.py`. `results.tsv` lives in `datasets/<name>/results.tsv` (untracked).

```bash
git checkout -b autoresearch/YYYY-MMdd
```

Loop:
1. Baseline: `uv run python improve.py --dataset-dir datasets/<name> --max-trials 5 > run.log 2>&1`
2. Read result: `grep "score=" run.log | tail -3`
3. Form hypothesis → edit code → `git commit` → run → record in `datasets/<name>/results.tsv`
4. Score improved → keep; otherwise `git reset --hard HEAD~1`
5. Never stop or ask for confirmation
