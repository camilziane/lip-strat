"""
predict.py – Real-life inference: load a trained agent and output the system-bet
selections for each upcoming round.

Rounds that still have no Score (upcoming matches) are treated as inference
targets.  Rounds with full scores are skipped (already played).

Usage:
    uv run python predict.py --model agent.npz --file new_data.xlsx
    uv run python predict.py --model agent.npz --file new_data.xlsx --k-grids 30
    uv run python predict.py --model agent.npz --file new_data.xlsx --round 42
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from env import make_obs, N_OUTCOMES, N_FEATURES, COST_PER_GRID
from train import REINFORCEAgent, RANG_COLS

OUTCOME_LABEL = {0: "1 (home)", 1: "N (draw)", 2: "2 (away)"}
OUTCOME_SHORT  = {0: "1", 1: "N", 2: "2"}


# ---------------------------------------------------------------------------
# Data loading  (no Score required — features only)
# ---------------------------------------------------------------------------

def load_rounds(path: str, loto_type: str | None = None) -> list[dict]:
    """
    Load rounds from an Excel file.
    Returns ALL rounds (complete and incomplete); caller decides which to use.
    """
    df = pd.read_excel(path)
    if loto_type:
        df = df[df["loto_type_name"] == loto_type]

    rounds = []
    for gid, gdf in df.groupby("grid_index"):
        gdf = gdf.sort_values("match_index").reset_index(drop=True)
        n_matches = len(gdf)

        features_raw = []
        for _, row in gdf.iterrows():
            features_raw.extend([
                row["Cote 1"], row["Cote 2"], row["Cote 3"],
                row["Rep 1"],  row["Rep 2"],  row["Rep 3"],
            ])

        # Prizes (may be 0 / absent for upcoming rounds)
        prizes: dict[int, float] = {}
        for i, col in enumerate(RANG_COLS):
            if col not in gdf.columns:
                break
            val = float(gdf[col].iloc[0])
            if val > 0:
                prizes[n_matches - i] = val

        has_scores = not gdf["Score"].isna().any()

        rounds.append({
            "grid_index": gid,
            "date": gdf["date"].iloc[0],
            "loto_type": str(gdf["loto_type_name"].iloc[0]),
            "n_matches": n_matches,
            "features_raw": features_raw,
            "match_info": [
                {
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "cote1": row["Cote 1"],
                    "coteN": row["Cote 2"],
                    "cote2": row["Cote 3"],
                    "rep1":  row["Rep 1"],
                    "repN":  row["Rep 2"],
                    "rep2":  row["Rep 3"],
                    "score": row["Score"] if has_scores else None,
                }
                for _, row in gdf.iterrows()
            ],
            "prizes": prizes,
            "has_scores": has_scores,
        })

    rounds.sort(key=lambda x: x["date"])
    return rounds


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def agent_probs(agent: REINFORCEAgent, round_: dict) -> np.ndarray:
    """Return (n_matches, 3) probability matrix for a round (sigmoid, not softmax)."""
    obs = make_obs(round_)
    return agent.get_probs(obs).astype(np.float32)


def print_round_prediction(
    round_: dict,
    selections: np.ndarray,   # (n_matches, 3) bool
    probs: np.ndarray,        # (n_matches, 3) float
) -> None:
    n = round_["n_matches"]
    prizes_str = "  ".join(
        f"rang{i+1}=${v:.0f}"
        for i, (_, v) in enumerate(sorted(round_["prizes"].items(), reverse=True))
    ) if round_["prizes"] else "prizes unknown"

    # Compute n_combos from selections
    import math
    counts = [int(selections[m].sum()) for m in range(n)]
    n_combos = math.prod(counts) if counts else 1

    print(f"\n{'═'*70}")
    print(f"  Grid {round_['grid_index']}  |  {round_['date']}  |  {round_['loto_type']}")
    print(f"  {prizes_str}")
    print(f"{'═'*70}")

    # Header
    col_w = 22
    print(f"  {'':>{col_w}}   {'1':^7} {'N':^7} {'2':^7}   "
          f"{'Odds (1/N/2)':^16}  {'Crowd%':^14}")
    print(f"  {'─'*col_w}   {'─'*7} {'─'*7} {'─'*7}   {'─'*16}  {'─'*14}")

    combo_parts = []
    for m, mi in enumerate(round_["match_info"]):
        label = f"{mi['home'][:10]} v {mi['away'][:9]}"
        c1 = "  ✓  " if selections[m, 0] else "     "
        cn = "  ✓  " if selections[m, 1] else "     "
        c2 = "  ✓  " if selections[m, 2] else "     "
        odds = f"{mi['cote1']:.2f}/{mi['coteN']:.2f}/{mi['cote2']:.2f}"
        crowd = f"{mi['rep1']:.0f}/{mi['repN']:.0f}/{mi['rep2']:.0f}%"
        score = f"  [{mi['score']}]" if mi["score"] else ""
        print(f"  {m+1:2d} {label:>{col_w}}  [{c1}][{cn}][{c2}]"
              f"  {odds:^16}  {crowd:^14}{score}")
        combo_parts.append(str(counts[m]))

    # Footer
    formula = " × ".join(combo_parts) + f" = {n_combos}"
    print(f"\n  {'─'*66}")
    print(f"  Selections : {formula} grid(s)")
    print(f"  Total cost : ${n_combos:.2f}")
    print(f"{'═'*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loto Foot inference — output system-bet selections from a trained agent"
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help=".npz file saved by train.py --save")
    parser.add_argument("--file", required=True, metavar="PATH",
                        help="Excel file with upcoming (or all) rounds")
    parser.add_argument("--k-grids", type=int, default=None,
                        help="Max grids (combos) per round (default: use value from model)")
    parser.add_argument("--round", type=int, default=None, dest="round_id",
                        help="Predict only for this grid_index (default: all upcoming)")
    parser.add_argument("--loto-type", type=str, default=None,
                        help="Filter by grid type, e.g. loto-foot-8")
    parser.add_argument("--include-played", action="store_true",
                        help="Also show predictions for already-completed rounds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- Load agent ---------------------------------------------------------
    agent = REINFORCEAgent.load(args.model)
    # Override k_max if user specifies --k-grids
    if args.k_grids is not None:
        agent.k_max = min(max(args.k_grids, 1), 50)
    k = agent.k_max
    print(f"Loaded agent: loto-foot-{agent.n_matches}  "
          f"(obs_dim={agent.obs_dim}, action_dim={agent.action_dim}, k_max={k})")

    # ---- Load rounds --------------------------------------------------------
    rounds = load_rounds(args.file, loto_type=args.loto_type)
    if not rounds:
        print("No rounds found in the file.")
        return

    # Filter to matching grid size
    rounds = [r for r in rounds if r["n_matches"] == agent.n_matches]
    if not rounds:
        print(f"No rounds with n_matches={agent.n_matches} found. "
              f"Use --loto-type to filter.")
        return

    # Filter by round id if requested
    if args.round_id is not None:
        rounds = [r for r in rounds if r["grid_index"] == args.round_id]
        if not rounds:
            print(f"Round {args.round_id} not found.")
            return

    # By default only predict on upcoming (no scores yet)
    if not args.include_played:
        upcoming = [r for r in rounds if not r["has_scores"]]
        if not upcoming:
            print("No upcoming rounds found (all have scores). "
                  "Use --include-played to predict on completed rounds.")
            return
        rounds = upcoming

    print(f"\nPredicting {len(rounds)} round(s)  |  k_max={k} grids/round")

    total_combos = 0
    for round_ in rounds:
        obs = make_obs(round_)
        mask = agent.act(obs, deterministic=True)                   # flat (n*3,) binary
        selections = mask.reshape(round_["n_matches"], N_OUTCOMES).astype(bool)
        probs = agent_probs(agent, round_)
        print_round_prediction(round_, selections, probs)

        import math
        counts = [int(selections[m].sum()) for m in range(round_["n_matches"])]
        total_combos += math.prod(counts) if counts else 1

    print(f"\n{'─'*60}")
    print(f"  {len(rounds)} round(s)  |  {total_combos} total grid(s)  "
          f"=  ${total_combos:.2f} total cost")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
