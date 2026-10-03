from __future__ import annotations

import argparse
import json
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from audit_utils import config_hash, environment, save_json, sha256, utc_now


HORIZON = 4
SEEDS = (13, 42, 97)
ACTIVITY = [
    "unique_transactions",
    "incoming_transactions",
    "outgoing_transactions",
    "active_steps",
    "address_event_count",
    "event_rate",
    "incoming_event_fraction",
    "last_event_lag",
]
SNA = [
    "in_counterparties",
    "out_counterparties",
    "total_counterparties",
    "in_pair_events",
    "out_pair_events",
    "in_repeat_fraction",
    "out_repeat_fraction",
    "reciprocal_counterparties",
    "reciprocal_fraction",
    "max_in_concentration",
    "max_out_concentration",
    "counterparty_diversity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--include-rf", action="store_true")
    return parser.parse_args()


def precision_at_fraction(y_true: np.ndarray, score: np.ndarray, fraction: float = 0.01) -> float:
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-score, kind="mergesort")[:k]
    return float(np.mean(y_true[order]))


def select_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    candidates = np.unique(score[order])
    if len(candidates) > 4000:
        candidates = np.quantile(score, np.linspace(0.0, 1.0, 4001))
    # Same candidate grid and first-maximum tie rule as the earlier loop.
    # Cumulative counts avoid recomputing 4,001 full confusion matrices.
    sorted_y = np.asarray(y_true)[order]
    prefix = np.r_[0, np.cumsum(sorted_y)]
    insertion = np.searchsorted(score[order], candidates, side="left")
    tp = prefix[-1] - prefix[insertion]
    predicted = len(score) - insertion
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted > 0)
    recall = tp / max(1, prefix[-1])
    values = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(len(tp), dtype=float), where=precision + recall > 0)
    return float(candidates[int(np.argmax(values))])


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score >= threshold
    return {
        "ap": float(average_precision_score(y_true, score)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "brier": float(brier_score_loss(y_true, score)),
        "p_at_1pct": precision_at_fraction(y_true, score),
        "threshold": float(threshold),
    }


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx_time = pd.read_csv(
        data_dir / "txs_features.csv",
        usecols=["txId", "Time step"],
        dtype={"txId": "int64", "Time step": "int16"},
    ).drop_duplicates("txId")
    labels = pd.read_csv(
        data_dir / "wallets_classes.csv",
        dtype={"address": "string", "class": "int8"},
    )
    labels = labels.loc[labels["class"].isin([1, 2])].copy()
    labels["label"] = (labels["class"] == 1).astype("int8")
    labels = labels.drop(columns="class").drop_duplicates("address")

    inputs = pd.read_csv(
        data_dir / "AddrTx_edgelist.csv",
        dtype={"input_address": "string", "txId": "int64"},
    ).drop_duplicates()
    outputs = pd.read_csv(
        data_dir / "TxAddr_edgelist.csv",
        dtype={"txId": "int64", "output_address": "string"},
    ).drop_duplicates()
    inputs = inputs.merge(tx_time, on="txId", how="inner", validate="many_to_one")
    outputs = outputs.merge(tx_time, on="txId", how="inner", validate="many_to_one")
    return tx_time, labels, inputs, outputs


def build_wallet_index(labels: pd.DataFrame, inputs: pd.DataFrame, outputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outgoing = inputs.rename(columns={"input_address": "address"}).copy()
    outgoing["direction"] = "out"
    incoming = outputs.rename(columns={"output_address": "address"}).copy()
    incoming["direction"] = "in"
    events = pd.concat([outgoing, incoming], ignore_index=True)
    events = events.drop_duplicates(["address", "txId", "Time step", "direction"])
    labeled_events = events.merge(labels, on="address", how="inner", validate="many_to_one")
    first = labeled_events.groupby("address", observed=True)["Time step"].min().rename("first_step")
    wallets = labels.merge(first, on="address", how="inner", validate="one_to_one")
    wallets["cutoff"] = wallets["first_step"] + HORIZON - 1
    wallets = wallets.loc[wallets["cutoff"] <= 49].copy()
    wallets["split"] = np.select(
        [wallets["first_step"] <= 29, wallets["first_step"].between(30, 36), wallets["first_step"].between(37, 45)],
        ["train", "validation", "test"],
        default="excluded",
    )
    wallets = wallets.loc[wallets["split"] != "excluded"].copy()
    return wallets, events


def activity_features(wallets: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    visible = events.merge(wallets[["address", "first_step", "cutoff"]], on="address", how="inner")
    visible = visible.loc[visible["Time step"].le(visible["cutoff"])].copy()

    overall = visible.groupby("address", observed=True).agg(
        unique_transactions=("txId", "nunique"),
        active_steps=("Time step", "nunique"),
        address_event_count=("txId", "size"),
        last_event=("Time step", "max"),
    )
    directional = (
        visible.groupby(["address", "direction"], observed=True)["txId"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={"in": "incoming_transactions", "out": "outgoing_transactions"})
    )
    for column in ("incoming_transactions", "outgoing_transactions"):
        if column not in directional:
            directional[column] = 0
    frame = wallets.set_index("address").join(overall).join(directional)
    frame["event_rate"] = frame["address_event_count"] / HORIZON
    frame["incoming_event_fraction"] = frame["incoming_transactions"] / frame["unique_transactions"].clip(lower=1)
    frame["last_event_lag"] = frame["cutoff"] - frame["last_event"]
    return frame.reset_index()


def pair_features(wallets: pd.DataFrame, inputs: pd.DataFrame, outputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = inputs[["txId", "Time step", "input_address"]].merge(
        outputs[["txId", "output_address"]], on="txId", how="inner", validate="many_to_many"
    )
    pairs = pairs.loc[pairs["input_address"] != pairs["output_address"]]
    pairs = pairs.drop_duplicates(["txId", "Time step", "input_address", "output_address"])

    index = wallets[["address", "cutoff"]]
    outgoing = pairs.merge(index, left_on="input_address", right_on="address", how="inner")
    outgoing = outgoing.loc[outgoing["Time step"] <= outgoing["cutoff"], ["address", "output_address", "txId", "Time step"]]
    outgoing = outgoing.rename(columns={"output_address": "counterparty"})
    outgoing["direction"] = "out"

    incoming = pairs.merge(index, left_on="output_address", right_on="address", how="inner")
    incoming = incoming.loc[incoming["Time step"] <= incoming["cutoff"], ["address", "input_address", "txId", "Time step"]]
    incoming = incoming.rename(columns={"input_address": "counterparty"})
    incoming["direction"] = "in"
    incident = pd.concat([outgoing, incoming], ignore_index=True)

    counts = incident.groupby(["address", "direction", "counterparty"], observed=True).size().rename("n").reset_index()
    summary = counts.groupby(["address", "direction"], observed=True).agg(
        counterparties=("counterparty", "nunique"),
        pair_events=("n", "sum"),
        maximum=("n", "max"),
    ).reset_index()
    wide = summary.pivot(index="address", columns="direction").fillna(0)
    wide.columns = [f"{direction}_{measure}" for measure, direction in wide.columns]

    result = wallets[["address"]].set_index("address").join(wide)
    for direction in ("in", "out"):
        result[f"{direction}_counterparties"] = result.get(f"{direction}_counterparties", 0)
        result[f"{direction}_pair_events"] = result.get(f"{direction}_pair_events", 0)
        maximum = result.get(f"{direction}_maximum", 0)
        result[f"{direction}_repeat_fraction"] = 1.0 - (
            result[f"{direction}_counterparties"] / result[f"{direction}_pair_events"].clip(lower=1)
        )
        result[f"max_{direction}_concentration"] = maximum / result[f"{direction}_pair_events"].clip(lower=1)

    total_distinct = incident.groupby("address", observed=True)["counterparty"].nunique().rename("total_counterparties")
    result = result.join(total_distinct)
    result["total_counterparties"] = result["total_counterparties"].fillna(0)

    incoming_sets = incoming.groupby("address", observed=True)["counterparty"].agg(set)
    outgoing_sets = outgoing.groupby("address", observed=True)["counterparty"].agg(set)
    reciprocal = pd.DataFrame({"incoming": incoming_sets, "outgoing": outgoing_sets}).apply(
        lambda row: len((row["incoming"] if isinstance(row["incoming"], set) else set()) & (row["outgoing"] if isinstance(row["outgoing"], set) else set())),
        axis=1,
    ).rename("reciprocal_counterparties")
    result = result.join(reciprocal)
    result["reciprocal_counterparties"] = result["reciprocal_counterparties"].fillna(0)
    result["reciprocal_fraction"] = result["reciprocal_counterparties"] / result["total_counterparties"].clip(lower=1)
    result["counterparty_diversity"] = result["total_counterparties"] / (
        result["in_pair_events"] + result["out_pair_events"]
    ).clip(lower=1)
    result = result.reset_index()
    return result[["address", *SNA]], incident


def fit_models(samples: pd.DataFrame, output_dir: Path, include_rf: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    train = samples["split"] == "train"
    validation = samples["split"] == "validation"
    test = samples["split"] == "test"
    y_train = samples.loc[train, "label"].to_numpy()
    y_validation = samples.loc[validation, "label"].to_numpy()
    y_test = samples.loc[test, "label"].to_numpy()
    scale = float((y_train == 0).sum() / (y_train == 1).sum())
    validation_predictions = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def archive(model, learner_name, seed, feature_name, columns, val_score, test_score, threshold, scaler=None):
        key = f"{learner_name}_{feature_name}_seed{seed}"
        joblib.dump({"model": model, "scaler": scaler, "columns": columns, "threshold": threshold}, model_dir / f"{key}.joblib")
        v = samples.loc[validation, ["address", "label", "first_step", "cutoff"]].copy()
        v["learner"], v["seed"], v["features"], v["score"], v["threshold"] = learner_name, seed, feature_name, val_score, threshold
        validation_predictions.append(v)

    for seed in SEEDS:
        for feature_name, columns in (("activity", ACTIVITY), ("activity_sna", ACTIVITY + SNA)):
            model = XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=2,
                scale_pos_weight=scale,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
            model.fit(samples.loc[train, columns], y_train)
            validation_score = model.predict_proba(samples.loc[validation, columns])[:, 1]
            threshold = select_f1_threshold(y_validation, validation_score)
            test_score = model.predict_proba(samples.loc[test, columns])[:, 1]
            metrics = evaluate(y_test, test_score, threshold)
            rows.append({"learner": "XGBoost", "seed": seed, "features": feature_name, **metrics})
            prediction = samples.loc[test, ["address", "label", "first_step", "cutoff"]].copy()
            prediction["learner"] = "XGBoost"
            prediction["seed"] = seed
            prediction["features"] = feature_name
            prediction["score"] = test_score
            prediction["threshold"] = threshold
            predictions.append(prediction)
            archive(model, "XGBoost", seed, feature_name, columns, validation_score, test_score, threshold)
            print(f"XGBoost {feature_name} seed={seed} AP={metrics['ap']:.6f}", flush=True)

    for learner_name, learner in (
        (
            "HGB",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=42,
                class_weight="balanced",
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ):
        if learner_name == "RandomForest" and not include_rf:
            continue
        for feature_name, columns in (("activity", ACTIVITY), ("activity_sna", ACTIVITY + SNA)):
            learner.fit(samples.loc[train, columns], y_train)
            validation_score = learner.predict_proba(samples.loc[validation, columns])[:, 1]
            threshold = select_f1_threshold(y_validation, validation_score)
            test_score = learner.predict_proba(samples.loc[test, columns])[:, 1]
            rows.append({"learner": learner_name, "seed": 42, "features": feature_name, **evaluate(y_test, test_score, threshold)})
            prediction = samples.loc[test, ["address", "label", "first_step", "cutoff"]].copy()
            prediction["learner"] = learner_name
            prediction["seed"] = 42
            prediction["features"] = feature_name
            prediction["score"] = test_score
            prediction["threshold"] = threshold
            predictions.append(prediction)
            archive(learner, learner_name, 42, feature_name, columns, validation_score, test_score, threshold)

    for feature_name, columns in (("activity", ACTIVITY), ("activity_sna", ACTIVITY + SNA)):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(samples.loc[train, columns])
        x_validation = scaler.transform(samples.loc[validation, columns])
        x_test = scaler.transform(samples.loc[test, columns])
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        mlp.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        validation_score = mlp.predict_proba(x_validation)[:, 1]
        threshold = select_f1_threshold(y_validation, validation_score)
        test_score = mlp.predict_proba(x_test)[:, 1]
        rows.append({"learner": "MLP", "seed": 42, "features": feature_name, **evaluate(y_test, test_score, threshold)})
        prediction = samples.loc[test, ["address", "label", "first_step", "cutoff"]].copy()
        prediction["learner"] = "MLP"
        prediction["seed"] = 42
        prediction["features"] = feature_name
        prediction["score"] = test_score
        prediction["threshold"] = threshold
        predictions.append(prediction)
        archive(mlp, "MLP", 42, feature_name, columns, validation_score, test_score, threshold, scaler)
    pd.concat(validation_predictions, ignore_index=True).to_parquet(output_dir / "validation_predictions.parquet", index=False)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def temporal_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[(predictions["learner"] == "XGBoost") & (predictions["seed"] == 42)].copy()
    rows: list[dict[str, object]] = []
    for (feature_name, cutoff), group in selected.groupby(["features", "cutoff"], observed=True):
        y = group["label"].to_numpy()
        score = group["score"].to_numpy()
        threshold = float(group["threshold"].iloc[0])
        metrics = evaluate(y, score, threshold)
        rows.append(
            {
                "features": feature_name,
                "cutoff": int(cutoff),
                "n": int(len(group)),
                "illicit": int(y.sum()),
                "prevalence": float(y.mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def recency_weighting(samples: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    train = samples["split"] == "train"
    validation = samples["split"] == "validation"
    test = samples["split"] == "test"
    y_train = samples.loc[train, "label"].to_numpy()
    y_validation = samples.loc[validation, "label"].to_numpy()
    y_test = samples.loc[test, "label"].to_numpy()
    scale = float((y_train == 0).sum() / (y_train == 1).sum())
    rows: list[dict[str, float]] = []
    predictions = []
    validation_predictions = []
    for decay in (0.0, 0.03, 0.06, 0.10, 0.20):
        weight = np.exp(decay * (samples.loc[train, "first_step"].to_numpy() - 29.0))
        model = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=2,
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(samples.loc[train, ACTIVITY + SNA], y_train, sample_weight=weight)
        validation_score = model.predict_proba(samples.loc[validation, ACTIVITY + SNA])[:, 1]
        threshold = select_f1_threshold(y_validation, validation_score)
        test_score = model.predict_proba(samples.loc[test, ACTIVITY + SNA])[:, 1]
        rows.append(
            {
                "decay": decay,
                "validation_ap": float(average_precision_score(y_validation, validation_score)),
                **evaluate(y_test, test_score, threshold),
            }
        )
        for mask, score, collection in [(test, test_score, predictions), (validation, validation_score, validation_predictions)]:
            frame = samples.loc[mask, ["address", "label", "first_step", "cutoff"]].copy()
            frame["score"], frame["threshold"], frame["decay"] = score, threshold, decay
            collection.append(frame)
    pd.concat(predictions).to_parquet(output_dir / "recency_test_predictions.parquet", index=False)
    pd.concat(validation_predictions).to_parquet(output_dir / "recency_validation_predictions.parquet", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    global HORIZON
    args = parse_args()
    HORIZON = args.horizon
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, labels, inputs, outputs = load_raw(args.data_dir)
    wallets, events = build_wallet_index(labels, inputs, outputs)
    activity = activity_features(wallets, events)
    sna, incident = pair_features(wallets, inputs, outputs)
    samples = activity.merge(sna, on="address", how="left", validate="one_to_one")
    samples[SNA] = samples[SNA].fillna(0.0)

    visible_events = events.merge(wallets[["address", "cutoff"]], on="address", how="inner", validate="many_to_one")
    visible_events = visible_events.loc[visible_events["Time step"] <= visible_events["cutoff"]].drop(columns="cutoff")
    assert samples["address"].is_unique
    assert (incident.merge(wallets[["address", "cutoff"]], on="address")["Time step"] <= incident.merge(wallets[["address", "cutoff"]], on="address")["cutoff"]).all()
    # Export reconstructed inputs before learner runs so downstream runs can begin.
    samples.to_parquet(args.output_dir / "wallet_samples.parquet", index=False)
    incident.to_parquet(args.output_dir / "visible_incident_pairs.parquet", index=False)
    visible_events.to_parquet(args.output_dir / "visible_address_events.parquet", index=False)
    print(f"Raw reconstruction complete: {len(samples)} wallets; horizon={HORIZON}", flush=True)

    cohort = samples.groupby("split", observed=True)["label"].agg(["size", "sum", "mean"]).reset_index()
    metrics, predictions = fit_models(samples, args.output_dir, args.include_rf)
    temporal = temporal_breakdown(predictions)
    recency = recency_weighting(samples, args.output_dir)

    samples.to_parquet(args.output_dir / "wallet_samples.parquet", index=False)
    incident.to_parquet(args.output_dir / "visible_incident_pairs.parquet", index=False)
    cohort.to_csv(args.output_dir / "cohort_counts.csv", index=False)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    temporal.to_csv(args.output_dir / "temporal_breakdown.csv", index=False)
    recency.to_csv(args.output_dir / "recency_weighting.csv", index=False)
    manifest = {
        "horizon": HORIZON,
        "seeds": list(SEEDS),
        "activity_features": ACTIVITY,
        "sna_features": SNA,
        "n_samples": int(len(samples)),
        "n_incident_pair_events": int(len(incident)),
        "environment": environment(), "run_timestamp": utc_now(), "elapsed_seconds": time.perf_counter() - start,
        "configuration_hash": config_hash({"horizon": HORIZON, "seeds": SEEDS, "activity": ACTIVITY, "sna": SNA, "threshold_grid": 4001}),
        "raw_sha256": {f: sha256(args.data_dir / f) for f in ["txs_features.csv", "wallets_classes.csv", "AddrTx_edgelist.csv", "TxAddr_edgelist.csv"]},
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(cohort.to_string(index=False))
    print(metrics.to_string(index=False))
    print(recency.to_string(index=False))


if __name__ == "__main__":
    main()
