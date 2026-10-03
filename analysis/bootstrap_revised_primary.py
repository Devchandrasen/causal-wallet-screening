"""Paired bootstrap for the reconstructed seed-42 primary comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    frame = frame.loc[(frame["learner"] == "XGBoost") & (frame["seed"] == 42)]
    wide_score = frame.pivot(index="address", columns="features", values="score")
    wide_threshold = frame.groupby("features", observed=True)["threshold"].first()
    label = frame.drop_duplicates("address").set_index("address").loc[wide_score.index, "label"].to_numpy()
    activity = wide_score["activity"].to_numpy()
    augmented = wide_score["activity_sna"].to_numpy()
    rng = np.random.default_rng(args.seed)
    rows = []
    for draw in range(args.resamples):
        index = rng.integers(0, len(label), len(label))
        y = label[index]
        activity_score = activity[index]
        augmented_score = augmented[index]
        rows.append(
            {
                "draw": draw,
                "delta_ap": average_precision_score(y, augmented_score) - average_precision_score(y, activity_score),
                "delta_f1": f1_score(y, augmented_score >= wide_threshold["activity_sna"])
                - f1_score(y, activity_score >= wide_threshold["activity"]),
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    for metric in ("delta_ap", "delta_f1"):
        values = output[metric].to_numpy()
        print(metric, float(values.mean()), np.quantile(values, [0.025, 0.975]).tolist(), int((values > 0).sum()))


if __name__ == "__main__":
    main()
