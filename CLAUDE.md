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
uv run python improve.py --dataset-dir datasets/<name> --reset     # clear all state & commit

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

### Startup (once per session)

Before the first run, create two files in the dataset dir:

**`datasets/<name>/results.tsv`** — experiment log (untracked, never reset):
```
commit	score	memory_gb	status	description
```

**`datasets/<name>/README.md`** — living document (tracked). Initialise with:
- Current architecture snapshot (policy type, features, reward formula, split)
- Data range (parse from xlsx filename)
- Empty leaderboard and history sections (with HTML comment markers)
- What to try next section

The README must be kept up to date throughout the session:
- After every new best: `improve.py` auto-updates the leaderboard table
- After every hypothesis run: add a note to "What was tried" even if it failed
- After any architecture change: update the Architecture section

### Loop

1. Baseline: `uv run python improve.py --dataset-dir datasets/<name> --max-trials 5 > datasets/<name>/run.log 2>&1`
2. Read result: `grep "score=" datasets/<name>/run.log | tail -5`
3. Form hypothesis → edit code → `git commit` → run → record in `results.tsv`
4. Score improved → keep; otherwise `git reset --hard HEAD~1`
5. Never stop or ask for confirmation

### Mindset — you have full creative freedom

You are not limited to parameter sweeps. At any point you may:

- **Redesign the reward function** — e.g. penalise variance, reward partial matches, use a shaped proxy signal
- **Add new features** to the observation — derived statistics, historical hit rates, value bets ratios. If you need raw columns that are not currently exposed by `env.py`, **ask the user** to add them (describe exactly what you need and why it would help)
- **Replace the RL algorithm** — e.g. switch from REINFORCE to PPO, Actor-Critic, evolutionary strategies, or a pure supervised approach if labelled data is available
- **Change the action space** — e.g. output a probability distribution over k-subsets rather than independent Bernoulli masks
- **Request more data** — ask the user if a larger or more recent xlsx file is available, or if additional context (team rankings, weather, odds history) can be exported

When you hit a wall (val/test gap, reward always -18, no signal from random search), step back and rethink the whole approach rather than tweaking hyperparameters. Write your reasoning in README.md before implementing.
