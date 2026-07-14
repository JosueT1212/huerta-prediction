"""
One-off: compute mean kg for the first `skip_first_weeks` positional weeks
of each season, per greenhouse, from the historical Kg Excel files.

These weeks are excluded from CNN-RNN training (see CLAUDE.md §8) — the
model was never trained to predict them, so live inference falls back to
this historical mean instead. Re-run manually only if the Kg Excel data
changes.

Usage: python scripts/compute_historical_means.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Models"))

from cnn_rnn_yield import load_production  # noqa: E402

RESULTS_DIR = ROOT / "Models" / "results"
N_POSITIONS = 3


def historical_means_by_position(df: pd.DataFrame, n_positions: int = N_POSITIONS) -> dict[int, float]:
    df = df.copy()
    df["pos"] = df.groupby("temporada").cumcount()
    subset = df[df["pos"] < n_positions]
    return subset.groupby("pos")["kg_reales"].mean().round(2).to_dict()


def main():
    for inv in (3, 4):
        df = load_production(inv)
        means = historical_means_by_position(df)
        out_path = RESULTS_DIR / f"historical_mean_inv{inv}.json"
        with open(out_path, "w") as f:
            json.dump({str(k): float(v) for k, v in means.items()}, f, indent=2)
        print(f"  Inv{inv} → {out_path}: {means}")


if __name__ == "__main__":
    main()
