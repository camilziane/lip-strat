"""
env.py – Gym-compatible environment for Loto Foot betting.

Supports loto-foot-7, loto-foot-8, loto-foot-12, loto-foot-15.

Design
------
Each *episode* corresponds to one round (grid_index).  It is a single-step
episode:
    1. reset()  → observation of the N matches
    2. step(action) → reward, terminated=True, info

Observation space  (Box, n_matches × 11)
    Per match (11 features):
        implied_p1, implied_pN, implied_p2   – normalised from bookmaker odds
        rep1, repN, rep2                     – crowd % / 100
        margin                               – bookmaker overround (vig)
        log_spread                           – log(max_cote / min_cote)
        value1, valueN, value2               – crowd vs implied divergence for all 3 outcomes

Action space  (Box, n_matches × 3, in [0, 1])
    Binary selection mask: values > 0.5 mean "select this outcome".
    The environment expands all combinations of selected outcomes per match.

Prize structure  (stored per grid dict)
    prizes: dict[int, float]  →  {n_correct: prize_amount}
    e.g. loto-foot-8 : {8: rang1, 7: rang2}
         loto-foot-15: {15: rang1, 14: rang2, 13: rang3, 12: rang4}

Reward
    Per-grid net profit:  reward = (total_earnings - n_combos * 1.0) / max(n_combos, 1)
"""

from __future__ import annotations

import itertools

import numpy as np
import gymnasium as gym
from gymnasium import spaces


N_OUTCOMES = 3       # 0=home(1), 1=draw(N), 2=away(2)
N_FEATURES = 11      # per match
COST_PER_GRID = 1.0

# Supported grid sizes and how many ranks they use
VALID_N_MATCHES = {7, 8, 12, 15}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_obs(grid: dict) -> np.ndarray:
    """
    Build the observation vector from a grid dict.

    Base features (N_FEATURES = 11 per match), built from 'features_raw':
        implied_p1, implied_pN, implied_p2   – normalised from bookmaker odds
        rep1, repN, rep2                     – crowd % / 100
        margin                               – bookmaker overround (vig)
        log_spread                           – log(max_cote / min_cote)
        value1, valueN, value2               – crowd vs implied divergence

    Extra features (optional, appended after the base block):
        player_consensus  →  3 features per match: fraction of top-50 players
                             who selected outcome 1 / N / 2.  Stored in grid
                             dict under 'extra_features_raw' by load_data().

    Total obs shape: (n_matches × (11 + n_extra),) flattened.
    """
    n = grid["n_matches"]
    raw = np.array(grid["features_raw"], dtype=np.float32).reshape(n, 6)
    obs = np.zeros((n, N_FEATURES), dtype=np.float32)

    for m in range(n):
        c1, cn, c2, r1, rn, r2 = raw[m]

        inv = np.array([1.0 / c1, 1.0 / cn, 1.0 / c2])
        margin = float(inv.sum() - 1.0)
        p = inv / inv.sum()

        crowd = np.array([r1, rn, r2]) / 100.0

        log_spread = float(np.log(max(c1, cn, c2) / min(c1, cn, c2)))
        value1 = float(crowd[0] - p[0])
        valueN = float(crowd[1] - p[1])
        value2 = float(crowd[2] - p[2])

        obs[m] = [p[0], p[1], p[2], crowd[0], crowd[1], crowd[2],
                  margin, log_spread, value1, valueN, value2]

    extra_raw = grid.get("extra_features_raw")
    if extra_raw:
        n_extra = len(extra_raw) // n
        extra = np.array(extra_raw, dtype=np.float32).reshape(n, n_extra)
        obs = np.concatenate([obs, extra], axis=1)

    return obs.flatten()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LotoFootEnv(gym.Env):
    """
    Parameters
    ----------
    grids       : list of grid dicts (from load_data in train.py).
                  All grids must have the same n_matches.
    k_max       : maximum grids budget (1–50); excess combos are pruned
    mode        : 'train' (random sampling) | 'eval' (sequential, one pass)
    render_mode : 'human' to print match details
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grids: list[dict],
        k_max: int = 20,
        mode: str = "train",
        render_mode: str | None = None,
    ):
        super().__init__()
        assert grids, "grids list must not be empty"
        assert 1 <= k_max <= 50, "k_max must be in [1, 50]"
        assert mode in ("train", "eval")

        n_matches_set = {g["n_matches"] for g in grids}
        assert len(n_matches_set) == 1, (
            f"All grids must have the same n_matches, got {n_matches_set}. "
            "Filter by loto_type before creating the env."
        )

        self.grids = grids
        self.k_max = k_max
        self.mode = mode
        self.render_mode = render_mode

        self.n_matches: int = grids[0]["n_matches"]
        self.obs_dim: int = len(make_obs(grids[0]))   # accounts for any extra features
        self.action_dim: int = self.n_matches * N_OUTCOMES

        self._eval_idx: int = 0
        self._current_grid: dict | None = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.mode == "train":
            self._current_grid = self.grids[
                self.np_random.integers(len(self.grids))
            ]
        else:
            if self._eval_idx >= len(self.grids):
                raise StopIteration("All eval grids exhausted — call env.restart_eval()")
            self._current_grid = self.grids[self._eval_idx]
            self._eval_idx += 1

        obs = make_obs(self._current_grid)
        info = {
            "grid_index": self._current_grid["grid_index"],
            "date": self._current_grid["date"],
            "n_matches": self.n_matches,
            "prizes": self._current_grid["prizes"],
        }
        return obs, info

    def step(self, action: np.ndarray):
        assert self._current_grid is not None, "Call reset() before step()"

        grid = self._current_grid
        # Reshape action to (n_matches, 3) and threshold at 0.5
        action_vals = np.array(action, dtype=np.float32).reshape(self.n_matches, N_OUTCOMES)
        selections = action_vals > 0.5   # (n_matches, 3) bool

        # Ensure at least 1 outcome selected per match (fallback to argmax)
        for m in range(self.n_matches):
            if not selections[m].any():
                selections[m, int(action_vals[m].argmax())] = True

        # Generate all combinations of selected outcomes per match
        selected_indices = [
            [o for o in range(N_OUTCOMES) if selections[m, o]]
            for m in range(self.n_matches)
        ]
        all_combos = list(itertools.product(*selected_indices))
        n_combos = len(all_combos)

        # Prune to k_max by removing least-confident extra selection
        # Strategy: while over budget, find the match with multiple selections
        # that has the lowest action value among its selected outcomes, remove it.
        while n_combos > self.k_max:
            best_match = -1
            best_val = np.inf
            best_outcome = -1
            for m in range(self.n_matches):
                sel_outcomes = [o for o in range(N_OUTCOMES) if selections[m, o]]
                if len(sel_outcomes) <= 1:
                    continue  # can't remove, would leave match empty
                for o in sel_outcomes:
                    if action_vals[m, o] < best_val:
                        best_val = action_vals[m, o]
                        best_match = m
                        best_outcome = o
            if best_match == -1:
                break  # can't prune further
            selections[best_match, best_outcome] = False
            selected_indices = [
                [o for o in range(N_OUTCOMES) if selections[m, o]]
                for m in range(self.n_matches)
            ]
            all_combos = list(itertools.product(*selected_indices))
            n_combos = len(all_combos)

        outcomes = grid["outcomes"]                  # (n_matches,)
        prizes: dict[int, float] = grid["prizes"]   # {n_correct: amount}

        hits: dict[str, int] = {f"rang{i+1}": 0 for i in range(len(prizes))}
        total_earnings = 0.0
        per_combo_rewards = np.full(n_combos, -COST_PER_GRID, dtype=np.float32)
        sorted_prize_keys = sorted(prizes.keys(), reverse=True)

        all_combos_arrays = [np.array(c, dtype=np.int32) for c in all_combos]

        for i, combo_arr in enumerate(all_combos_arrays):
            correct = int(np.sum(combo_arr == outcomes))
            if correct in prizes:
                prize = prizes[correct]
                per_combo_rewards[i] = prize - COST_PER_GRID
                total_earnings += prize
                rank_idx = sorted_prize_keys.index(correct) + 1
                hits[f"rang{rank_idx}"] += 1

        total_cost = n_combos * COST_PER_GRID
        net = total_earnings - total_cost
        reward = net / max(n_combos, 1)

        info = {
            "grid_index": grid["grid_index"],
            "date": grid["date"],
            "n_matches": self.n_matches,
            "selections": selections,               # (n_matches, 3) bool ndarray
            "n_combos": n_combos,
            "all_combos": all_combos_arrays,        # list of (n_matches,) int32 arrays
            "per_combo_rewards": per_combo_rewards, # (n_combos,) float32
            "hits": hits,                           # {'rang1': n, 'rang2': n, ...}
            "total_earnings": total_earnings,
            "total_cost": total_cost,
            "net": net,
            "outcomes": outcomes,
        }

        next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        return next_obs, float(reward), True, False, info

    def render(self) -> None:
        if self.render_mode != "human" or self._current_grid is None:
            return
        grid = self._current_grid
        label = ["1 (home)", "N (draw)", "2 (away)"]
        prizes_str = "  ".join(
            f"rang{i+1}=${v:.0f}"
            for i, (k, v) in enumerate(
                sorted(grid["prizes"].items(), reverse=True)
            )
        )
        print(f"\nGrid {grid['grid_index']}  {grid['date']}  {prizes_str}")
        raw = np.array(grid["features_raw"]).reshape(self.n_matches, 6)
        for m in range(self.n_matches):
            c1, cn, c2, r1, rn, r2 = raw[m]
            print(
                f"  m{m+1:2d}: cotes=[{c1:.2f},{cn:.2f},{c2:.2f}]"
                f"  crowd=[{r1:.0f}%,{rn:.0f}%,{r2:.0f}%]"
                f"  actual={label[int(grid['outcomes'][m])]}"
            )

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def restart_eval(self) -> None:
        self._eval_idx = 0

    def eval_done(self) -> bool:
        return self._eval_idx >= len(self.grids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _logits_to_probs(self, action: np.ndarray) -> np.ndarray:
        """Utility: convert logits (n_matches*3,) to softmax probs (n_matches, 3)."""
        logits = action.reshape(self.n_matches, N_OUTCOMES).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
