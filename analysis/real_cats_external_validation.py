"""Out-of-period Ethereum profile transfer on Real-CATS and Sup-CATS.

This is deliberately reported as an external transfer check, not as a causal
fixed-horizon replication. The released profile tables summarize address
histories. Absolute timestamps and lifetime are excluded from the predictors.
"""

from __future__ import annotations

import argparse
import json
import time
import subprocess
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from reconstruct_wallet_experiments import precision_at_fraction
from audit_utils import environment, sha256, config_hash, save_json, utc_now, prediction_diagnostics


ACTIVITY = [
    "transaction_number",
    "payment_transactions",
    "receipt_transactions",
    "activity_d",
    "activity_w",
    "activity_time",
    "total_received_ETH",
    "total_sent_ETH",
]

RELATION = [
    "from_contract_wd_internal",
    "from_contract_wd_normal",
    "from_contract_wod_internal",
    "from_contract_wod_normal",
    "from_EOA_wd_internal",
    "from_EOA_wd_normal",
    "from_EOA_wod_internal",
    "from_EOA_wod_normal",
    "to_contract_wd_internal",
    "to_contract_wd_normal",
    "to_contract_wod_internal",
    "to_contract_wod_normal",
    "to_EOA_wd_internal",
    "to_EOA_wd_normal",
    "to_EOA_wod_internal",
    "to_EOA_wod_normal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = values.clip(lower=0.0)
    return np.log1p(values.to_numpy(dtype=np.float32))


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    budget = max(1, int(np.ceil(0.01 * len(y_true))))
    top = np.argsort(-probability, kind="stable")[:budget]
    return {
        "ap": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "p_at_1pct": float(y_true[top].mean()),
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benign = pd.read_csv(args.data_dir / "BE.tsv", sep="\t", low_memory=False)
    criminal = pd.read_csv(args.data_dir / "CE.tsv", sep="\t", low_memory=False)
    benign["target"] = 0
    criminal["target"] = 1
    audit_rows = []
    for name, frame in [("BE", benign), ("CE", criminal)]:
        active = pd.to_numeric(frame["transaction_number"], errors="coerce").fillna(0) > 0
        audit_rows.append({"cohort": name, "raw": len(frame), "inactive_removed": int((~active).sum()), "active": int(active.sum()), "duplicate_addresses_active": int(frame.loc[active, "address"].str.lower().duplicated().sum())})
    source = pd.concat([benign, criminal], ignore_index=True)
    source["address"] = source.address.str.strip().str.lower()
    source = source[pd.to_numeric(source["transaction_number"], errors="coerce").fillna(0) > 0].copy()
    assert source.address.is_unique, "Conflicting or repeated source identities require explicit resolution"

    external = pd.read_csv(args.data_dir / "Sup-CATS.tsv", sep="\t", low_memory=False)
    external["address"] = external.address.str.strip().str.lower()
    assert set(external.label.str.lower().unique()) == {"phish", "benign"}
    external["target"] = external["label"].astype(str).str.lower().eq("phish").astype(np.int8)
    for label in ("benign", "phish"):
        frame = external[external.label.str.lower() == label]
        active = pd.to_numeric(frame["transaction_number"], errors="coerce").fillna(0) > 0
        audit_rows.append({"cohort": f"Sup-{label}", "raw": len(frame), "inactive_removed": int((~active).sum()), "active": int(active.sum()), "duplicate_addresses_active": int(frame.loc[active, "address"].duplicated().sum())})
    external = external[pd.to_numeric(external["transaction_number"], errors="coerce").fillna(0) > 0].copy()
    duplicate_rows = external[external.address.duplicated(keep=False)]
    if len(duplicate_rows):
        assert duplicate_rows.groupby('address').target.nunique().max() == 1
        # Every repeated profile must be identical, not merely share a label.
        assert duplicate_rows.drop_duplicates().address.is_unique
    duplicate_rows.to_csv(args.output_dir / "external_duplicate_rows.csv", index=False)
    external = external.drop_duplicates("address", keep="first")
    overlap = external[external.address.isin(source.address)].merge(source[["address", "target"]], on="address", suffixes=("_external", "_source"))
    overlap.to_csv(args.output_dir / "source_external_overlap.csv", index=False)
    external = external[~external.address.isin(source.address)].copy()
    assert external.address.is_unique and not set(source.address).intersection(external.address)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "filter_audit.csv", index=False)
    source[["address", "target"]].to_parquet(args.output_dir / "source_cohort.parquet", index=False)
    external[["address", "target"]].to_parquet(args.output_dir / "external_cohort.parquet", index=False)

    train_index, validation_index = train_test_split(
        np.arange(len(source)),
        test_size=0.2,
        random_state=2026,
        stratify=source["target"].to_numpy(),
    )
    y = source["target"].to_numpy(dtype=np.int8)
    y_external = external["target"].to_numpy(dtype=np.int8)

    rows: list[dict[str, object]] = []
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    all_predictions, all_validation_predictions = [], []
    def archive(model, feature_name, learner, seed, columns, validation_probability, probability, threshold, started):
        classes = model.classes_
        assert np.array_equal(classes, [0, 1])
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        top_count = max(1, int(np.ceil(.01 * len(external))))
        top = np.argsort(-probability, kind="stable")[:top_count]
        result = {"learner": learner, "seed": seed, "features": feature_name,
                  **metrics(y_external, probability, threshold), **prediction_diagnostics(y_external, probability),
                  "top_count": top_count, "top_illicit": int(y_external[top].sum()),
                  "external_prevalence": float(y_external.mean()),
                  "reversed_score_ap_diagnostic_only": float(average_precision_score(y_external, 1-probability)),
                  "run_timestamp": utc_now(), "elapsed_seconds": time.perf_counter()-started,
                  "configuration_hash": config_hash({"model": model.get_params(), "features": columns, "seed": seed, "split_seed": 2026})}
        rows.append(result)
        for frame, score, collection in [(external[["address", "target"]].copy(), probability, all_predictions),
                                          (source.iloc[validation_index][["address", "target"]].copy(), validation_probability, all_validation_predictions)]:
            frame["learner"], frame["seed"], frame["features"] = learner, seed, feature_name
            frame["score"], frame["threshold"] = score, threshold
            collection.append(frame)
        joblib.dump({"model": model, "columns": columns, "threshold": threshold, "positive_class": 1}, models_dir / f"{learner}_{feature_name}_seed{seed}.joblib")
        print(f"{learner} {feature_name} seed{seed} AP={result['ap']:.6f} P1={result['p_at_1pct']:.6f}", flush=True)
    for feature_name, columns in (("activity", ACTIVITY), ("activity_relation", ACTIVITY + RELATION)):
        x = numeric_frame(source, columns)
        x_external = numeric_frame(external, columns)
        scale = float((y[train_index] == 0).sum() / max(1, (y[train_index] == 1).sum()))
        for seed in (13, 42, 97):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=2,
                scale_pos_weight=scale,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x[train_index], y[train_index])
            validation_probability = model.predict_proba(x[validation_index])[:, 1]
            threshold = choose_threshold(y[validation_index], validation_probability)
            probability = model.predict_proba(x_external)[:, 1]
            archive(model, feature_name, "XGBoost", seed, columns, validation_probability, probability, threshold, started)

        for seed in (13, 42, 97):
          started = time.perf_counter()
          mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=10,
                random_state=seed,
            ),
        )
          mlp.fit(x[train_index], y[train_index])
          validation_probability = mlp.predict_proba(x[validation_index])[:, 1]
          threshold = choose_threshold(y[validation_index], validation_probability)
          probability = mlp.predict_proba(x_external)[:, 1]
          archive(mlp, feature_name, "MLP", seed, columns, validation_probability, probability, threshold, started)

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "real_cats_external_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).rename(columns={"target": "label"}).to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    pd.concat(all_validation_predictions, ignore_index=True).rename(columns={"target": "label"}).to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    source[["address", "target"]].assign(split=np.where(np.isin(np.arange(len(source)), train_index), "train", "validation")).to_csv(args.output_dir / "source_split.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "Real-CATS Ethereum profiles (BE.tsv and CE.tsv)",
                "external_test": "Sup-CATS Ethereum deployment cohort",
                "source_rows": int(len(source)),
                "source_benign": int((y==0).sum()), "source_criminal": int((y==1).sum()),
                "external_rows": int(len(external)),
                "external_benign": int((y_external==0).sum()), "external_criminal": int((y_external==1).sum()),
                "exact_duplicate_rows_removed": len(duplicate_rows) - duplicate_rows.address.nunique(),
                "source_external_overlap_removed": len(overlap),
                "overlap_external_class_counts": overlap.target_external.value_counts().to_dict(),
                "external_positive_rate": float(y_external.mean()),
                "split_seed": 2026,
                "activity_features": ACTIVITY,
                "relation_features": RELATION,
                "important_scope_note": "Dataset-provided profile histories; not a fixed-horizon causal replication.",
                "repository_commit": subprocess.check_output(["git", "-C", str(args.data_dir), "rev-parse", "HEAD"], text=True).strip(),
                "raw_sha256": {f: sha256(args.data_dir / f) for f in ["BE.tsv", "CE.tsv", "Sup-CATS.tsv"]},
                "script_sha256": sha256(Path(__file__)), "environment": environment(),
                "train_count": len(train_index), "validation_count": len(validation_index),
                "checks": {"labels": "BE=0 CE=1 Sup benign=0 phish=1", "positive_probability_column": "classes_ == [0,1], column1", "top1": "stable descending scores, ceil(0.01N)", "normalization": "source training only", "orientation": "original source-learned scores; inverted AP diagnostic is not used to choose test score direction"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
