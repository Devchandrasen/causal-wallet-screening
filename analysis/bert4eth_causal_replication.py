"""Rebuild a separate timestamped Ethereum experiment from author-released CSVs.

No author embeddings, augmentation, test-selected settings or raw-data publication.
Run prepare before tabular/neural; the locally frozen protocol is retained with hashes.
"""
import argparse
import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from audit_utils import environment, save_json, sha256, utc_now
from reconstruct_wallet_experiments import ACTIVITY, SNA, evaluate, select_f1_threshold
from train_prefix_consistent_transformer import build_views, address_activity

SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL = SCRIPT_DIR / "bert4eth_external_protocol.json"


def timestamp(value):
    return int(pd.Timestamp(value).timestamp())


def load_transactions(path, labels, positive):
    frames, record = [], {"archive": path.name, "sha256": sha256(path), "files": []}
    roots = set()
    with zipfile.ZipFile(path) as archive:
        names = [x for x in archive.namelist() if x.endswith(".csv") and not x.startswith("__MACOSX/")]
        if len(names) != 2:
            raise ValueError(f"Expected two directional CSVs, got {names}")
        for direction in ["out", "in"]:
            name = next(x for x in names if f"_{direction}" in x)
            raw_rows, missing, selected, min_time, max_time = 0, 0, 0, None, None
            with archive.open(name) as stream:
                chunks = pd.read_csv(stream, header=None, usecols=[0, 5, 6, 11],
                                     names=["txId", "nonce", "block_hash", "block_number", "transaction_index", "from", "to", "value", "gas", "gas_price", "input", "timestamp", "max_fee_per_gas", "max_priority_fee_per_gas", "transaction_type"],
                                     dtype={"txId": "string", "from": "string", "to": "string", "timestamp": "int64"},
                                     chunksize=300_000)
                for chunk in chunks:
                    raw_rows += len(chunk)
                    min_time = min(min_time if min_time is not None else chunk.timestamp.min(), chunk.timestamp.min())
                    max_time = max(max_time if max_time is not None else chunk.timestamp.max(), chunk.timestamp.max())
                    missing += int(chunk[["from", "to"]].isna().any(axis=1).sum())
                    chunk = chunk.dropna(subset=["from", "to"]).copy()
                    chunk["from"], chunk["to"] = chunk["from"].str.lower(), chunk["to"].str.lower()
                    root_column = "from" if direction == "out" else "to"
                    root = chunk[root_column]
                    if direction == "out":
                        if positive:
                            keep = root.isin(labels)
                        else:
                            keep = ~root.isin(labels) & root.str.slice(2, 10).map(lambda x: int(x, 16) % 20 == 0)
                        roots.update(root[keep].tolist())
                    else:
                        keep = root.isin(roots)
                    selected += int(keep.sum())
                    frame = chunk.loc[keep].copy()
                    frame["address"] = frame[root_column]
                    frame["counterparty"] = frame["to" if direction == "out" else "from"]
                    frame["direction"] = direction
                    frame["label"] = int(positive)
                    frames.append(frame[["address", "counterparty", "direction", "txId", "timestamp", "label"]])
            record["files"].append({"name": name, "rows": raw_rows, "missing_endpoints": missing,
                                    "selected_event_rows": selected, "min_timestamp": int(min_time), "max_timestamp": int(max_time)})
            print(f"read {name}: raw={raw_rows} selected={selected} roots={len(roots)}", flush=True)
    return pd.concat(frames, ignore_index=True), record


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "PREPARED.json").exists():
        raise ValueError("Use a new directory rather than overwrite a prepared experiment")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    save_json(args.output / "PROTOCOL_FREEZE.json", {"timestamp": utc_now(), "protocol_sha256": sha256(PROTOCOL),
                "adapter_sha256": sha256(Path(__file__)), "protocol": protocol,
                "test_prediction_inspection": "None in this corpus before this freeze"})
    labels_path = args.data / "phisher_account.txt"
    labels = {x.lower().strip() for x in labels_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    benign, benign_record = load_transactions(args.data / "BERT4ETH_normal.zip", labels, False)
    phishing, phishing_record = load_transactions(args.data / "BERT4ETH_phish.zip", labels, True)
    events = pd.concat([benign, phishing], ignore_index=True)
    original_rows = len(events)
    assert events.groupby("address").label.nunique().max() == 1
    events = events.drop_duplicates(["address", "txId", "direction"]).copy()
    first = events.groupby("address", sort=True).agg(first_timestamp=("timestamp", "min"), label=("label", "first")).reset_index()
    first["split"] = "excluded"
    for split, key in [("train", "train_first_timestamp"), ("validation", "validation_first_timestamp"), ("test", "test_first_timestamp")]:
        low, high = [timestamp(x) for x in protocol[key]]
        first.loc[first.first_timestamp.between(low, high), "split"] = split
    end = min(timestamp(protocol["collection_end_timestamp"]),
              max(x["max_timestamp"] for x in benign_record["files"]),
              max(x["max_timestamp"] for x in phishing_record["files"]))
    first["cutoff_timestamp"] = first.first_timestamp + 56 * 86400
    first = first.loc[first.split.ne("excluded") & first.cutoff_timestamp.le(end)].reset_index(drop=True)
    first["first_step"], first["cutoff"] = 1, 4
    visible = events.merge(first[["address", "first_timestamp", "cutoff_timestamp"]], on="address", how="inner", validate="many_to_one")
    visible = visible.loc[visible.timestamp.lt(visible.cutoff_timestamp)].copy()
    visible["Time step"] = ((visible.timestamp - visible.first_timestamp) // (14 * 86400) + 1).astype("int16")
    assert visible["Time step"].between(1, 4).all()
    assert first.address.is_unique and len(first) == visible.address.nunique()
    assert first.groupby("split").label.nunique().eq(2).all()
    for left, right in [("train", "validation"), ("validation", "test")]:
        assert first.loc[first.split.eq(left), "cutoff_timestamp"].max() < first.loc[first.split.eq(right), "first_timestamp"].min()
    incident = visible.loc[visible.address.ne(visible.counterparty), ["address", "counterparty", "direction", "txId", "Time step"]].copy()
    address_events = visible[["address", "txId", "Time step", "direction"]].copy()
    views = {}
    for name, steps in [("full", 4), ("prefix", 2)]:
        pair_rows = incident.loc[incident["Time step"].le(steps)]
        pairs, wallet = build_views(first, pair_rows, steps)
        activity = address_activity(first, address_events.loc[address_events["Time step"].le(steps)], steps)
        wallet = activity.merge(wallet[["address", *SNA]], on="address", validate="one_to_one")
        views[name] = (pairs, wallet)
    samples = first.merge(views["full"][1][["address", *ACTIVITY, *SNA]], on="address", validate="one_to_one")
    assert np.isfinite(samples[ACTIVITY + SNA].to_numpy()).all()
    cache = args.output / "neural_views_v2"
    cache.mkdir(exist_ok=True)
    for name, (pairs, wallet) in views.items():
        pairs.to_parquet(cache / f"{name}_pairs.parquet", index=False)
        wallet.to_parquet(cache / f"{name}_wallet.parquet", index=False)
    samples.to_parquet(args.output / "wallet_samples.parquet", index=False)
    address_events.to_parquet(args.output / "visible_address_events.parquet", index=False)
    incident.to_parquet(args.output / "visible_incident_pairs.parquet", index=False)
    # Adding future records cannot alter the view, since selection is based on each wallet's own first-time cutoff.
    future = visible.copy()
    future["timestamp"] = future["cutoff_timestamp"] + 86400
    mutated = pd.concat([visible, future], ignore_index=True)
    pd.testing.assert_frame_equal(visible.reset_index(drop=True), mutated.loc[mutated.timestamp.lt(mutated.cutoff_timestamp)].reset_index(drop=True))
    counts = first.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts.to_csv(args.output / "cohort_counts.csv")
    save_json(args.output / "PREPARED.json", {"timestamp": utc_now(), "protocol_sha256": sha256(PROTOCOL),
        "adapter_sha256": sha256(Path(__file__)), "source_records": [benign_record, phishing_record],
        "label_sha256": sha256(labels_path), "label_list_unique": len(labels),
        "selected_roots_before_cohort": int(events.address.nunique()), "deduplicated_rows_removed": original_rows - len(events),
        "collection_end_timestamp_used": end, "samples": len(samples), "visible_events": len(visible),
        "self_event_rows_excluded_from_pairs": len(visible) - len(incident), "causal_assertions": "passed",
        "cohort_counts": counts.reset_index().to_dict(orient="records"), "environment": environment(),
        "sample_sha256": sha256(args.output / "wallet_samples.parquet"), "limitations": protocol["limitations"]})
    print(counts.to_string(), flush=True)


def tabular(args):
    rows, validation_rows, test_rows = [], [], []
    samples = pd.read_parquet(args.output / "wallet_samples.parquet")
    train, validation, test = [samples.split.eq(x) for x in ["train", "validation", "test"]]
    y = samples.label.to_numpy()
    scale = float((y[train] == 0).sum() / (y[train] == 1).sum())
    folder = args.output / "tabular"
    folder.mkdir(exist_ok=True)
    for seed in [13, 42, 97]:
        for feature_name, columns in [("activity", ACTIVITY), ("activity_sna", ACTIVITY + SNA)]:
            for name in ["XGBoost", "MLP"]:
                scaler = None
                if name == "XGBoost":
                    model = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=.9,
                        colsample_bytree=.9, min_child_weight=2, reg_lambda=2, scale_pos_weight=scale,
                        random_state=seed, n_jobs=-1, tree_method="hist")
                    inputs = samples[columns]
                    model.fit(inputs.loc[train], y[train])
                else:
                    scaler = StandardScaler().fit(samples.loc[train, columns])
                    inputs = pd.DataFrame(scaler.transform(samples[columns]), index=samples.index)
                    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
                        batch_size=1024, learning_rate_init=1e-3, max_iter=100, early_stopping=True,
                        validation_fraction=.1, n_iter_no_change=10, random_state=seed)
                    model.fit(inputs.loc[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
                v_score, t_score = [model.predict_proba(inputs.loc[mask])[:, 1] for mask in [validation, test]]
                threshold = select_f1_threshold(y[validation], v_score)
                result = {"learner": name, "features": feature_name, "seed": seed, **evaluate(y[test], t_score, threshold)}
                rows.append(result)
                key = f"{name}_{feature_name}_seed{seed}"
                joblib.dump({"model": model, "scaler": scaler, "columns": columns, "threshold": threshold}, folder / f"{key}.joblib")
                for mask, score, destination in [(validation, v_score, validation_rows), (test, t_score, test_rows)]:
                    pred = samples.loc[mask, ["address", "label", "first_timestamp", "cutoff_timestamp"]].copy()
                    pred["learner"], pred["features"], pred["seed"] = name, feature_name, seed
                    pred["score"], pred["threshold"] = score, threshold
                    destination.append(pred)
                print(f"{name} {feature_name} seed={seed} AP={result['ap']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.concat(validation_rows).to_parquet(folder / "validation_predictions.parquet", index=False)
    pd.concat(test_rows).to_parquet(folder / "test_predictions.parquet", index=False)
    save_json(folder / "manifest.json", {"timestamp": utc_now(), "adapter_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(PROTOCOL), "input_sha256": sha256(args.output / "wallet_samples.parquet"),
        "environment": environment(), "settings": "Unchanged locked primary XGBoost and MLP settings; every seed reported"})


def neural(args):
    for variant in json.loads(PROTOCOL.read_text())["neural_variants"]:
        for seed in [13, 42, 97]:
            destination = args.output / "neural" / f"{variant}_seed{seed}"
            if (destination / "manifest.json").exists():
                continue
            subprocess.run([sys.executable, "-u", str(SCRIPT_DIR / "train_prefix_consistent_transformer.py"),
                "--results-dir", str(args.output), "--output-dir", str(destination), "--variant", variant,
                "--seed", str(seed)], check=True)
            manifest = json.loads((destination / "manifest.json").read_text())
            manifest.update({"external_protocol_sha256": sha256(PROTOCOL), "adapter_sha256": sha256(Path(__file__)),
                "historical_note": "New independent BERT4ETH corpus for this analysis. Settings frozen before test predictions; retrospective labels and corpus sampling remain."})
            save_json(destination / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "tabular", "neural"])
    parser.add_argument("--data", type=Path, default=Path("tmp/external_candidate_20261003"))
    parser.add_argument("--output", type=Path, default=Path("analysis/external_bert4eth_20261003"))
    args = parser.parse_args()
    {"prepare": prepare, "tabular": tabular, "neural": neural}[args.phase](args)
