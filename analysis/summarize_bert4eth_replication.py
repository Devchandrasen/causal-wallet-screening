"""Independently verify saved external predictions and generate aggregate tables."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from audit_utils import sha256, save_json, utc_now


def independent_metrics(prediction, score_column="score", threshold_column="threshold"):
    y, score = prediction.label.to_numpy(), prediction[score_column].to_numpy()
    threshold = float(prediction[threshold_column].iloc[0])
    alerts = np.argsort(-score, kind="stable")[:int(np.ceil(.01 * len(score)))]
    return {"ap": average_precision_score(y, score), "f1": f1_score(y, score >= threshold, zero_division=0),
            "precision": precision_score(y, score >= threshold, zero_division=0),
            "recall": recall_score(y, score >= threshold, zero_division=0), "roc_auc": roc_auc_score(y, score),
            "brier": brier_score_loss(y, score), "p_at_1pct": float(y[alerts].mean()), "threshold": threshold}


def verify_threshold(prediction, score_column="score", threshold_column="threshold"):
    y, score = prediction.label.to_numpy(), prediction[score_column].to_numpy()
    candidates = np.unique(score)
    if len(candidates) > 4000:
        candidates = np.quantile(score, np.linspace(0, 1, 4001))
    values = []
    # Direct boolean confusion counts, independent of trainer's sorted cumulative search.
    for chunk in np.array_split(candidates, max(1, int(np.ceil(len(candidates) / 64)))):
        positive = score[:, None] >= chunk[None, :]
        tp = (positive & y.astype(bool)[:, None]).sum(axis=0)
        denom = positive.sum(axis=0) + y.sum()
        values.extend(np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0))
    selected = float(candidates[int(np.argmax(values))])
    assert selected == float(prediction[threshold_column].iloc[0]), (selected, prediction[threshold_column].iloc[0])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("analysis/external_bert4eth_20261003"))
    parser.add_argument("--bundle", type=Path, default=Path("revision_bundle"))
    args = parser.parse_args()
    checks, all_rows, summaries = [], [], []
    root = args.root
    prepared = json.loads((root / "PREPARED.json").read_text())
    frozen = json.loads((root / "PROTOCOL_FREEZE.json").read_text())
    assert prepared["protocol_sha256"] == frozen["protocol_sha256"] == sha256(Path("analysis/bert4eth_external_protocol.json"))
    assert sha256(Path("analysis/bert4eth_causal_replication.py")) == prepared["adapter_sha256"]
    tabular = pd.read_csv(root / "tabular/metrics.csv")
    tab_val = pd.read_parquet(root / "tabular/validation_predictions.parquet")
    tab_test = pd.read_parquet(root / "tabular/test_predictions.parquet")
    for (model, features, seed), group in tab_test.groupby(["learner", "features", "seed"]):
        validation = tab_val.loc[tab_val.learner.eq(model) & tab_val.features.eq(features) & tab_val.seed.eq(seed)]
        assert not set(validation.address) & set(group.address)
        result = independent_metrics(group)
        threshold = verify_threshold(validation)
        expected = tabular.loc[tabular.learner.eq(model) & tabular.features.eq(features) & tabular.seed.eq(seed)].iloc[0]
        for key, value in result.items():
            np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-7 if key == "brier" else 1e-10)
        row = {"model": model, "features": features, "seed": int(seed), "view": "full", **result}
        all_rows.append(row)
        checks.append({"model": model, "features": features, "seed": int(seed), "view": "full", "metrics": "passed", "threshold": threshold})
    neural = []
    variants = ["tet_activity_sna", "dual_tet_activity_sna", "cons_tet_activity_sna", "pc_tet_activity_sna"]
    for variant in variants:
        for seed in [13, 42, 97]:
            directory = root / "neural" / f"{variant}_seed{seed}"
            expected = pd.read_csv(directory / "metrics.csv").iloc[0]
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["external_protocol_sha256"] == frozen["protocol_sha256"]
            assert manifest["script_sha256"] == sha256(Path("analysis/train_prefix_consistent_transformer.py"))
            validation = pd.read_parquet(directory / "validation_predictions.parquet")
            test = pd.read_parquet(directory / "test_predictions.parquet")
            assert not set(validation.address) & set(test.address)
            assert test.address.is_unique and validation.address.is_unique
            for view, prefix in [("full", ""), ("prefix", "prefix_")]:
                result = independent_metrics(test, prefix + "score", prefix + "threshold")
                threshold = verify_threshold(validation, prefix + "score", prefix + "threshold")
                for key, value in result.items():
                    np.testing.assert_allclose(value, expected[prefix + key], rtol=0, atol=1e-7 if key == "brier" else 1e-10)
                all_rows.append({"model": variant, "features": "activity_sna", "seed": seed, "view": view, **result})
                checks.append({"model": variant, "seed": seed, "view": view, "metrics": "passed", "threshold": threshold})
            gap = float(np.abs(test.score - test.prefix_score).mean())
            np.testing.assert_allclose(gap, expected.mean_full_prefix_gap, rtol=0, atol=1e-10)
            neural.append({"model": variant, "seed": seed, "ap": expected.ap, "f1": expected.f1,
                           "p_at_1pct": expected.p_at_1pct, "prefix_ap": expected.prefix_ap, "gap": gap})
    pd.DataFrame(all_rows).to_csv(root / "MASTER_EXTERNAL_RESULTS.csv", index=False)
    bootstrap = []
    for model in ["XGBoost", "MLP"]:
        for seed in [13, 42, 97]:
            scores = tab_test.loc[tab_test.learner.eq(model) & tab_test.seed.eq(seed)]
            scores = scores.pivot(index=["address", "label"], columns="features", values="score").reset_index()
            y = scores.label.to_numpy()
            rng = np.random.default_rng(2026)
            differences = []
            for _ in range(500):
                idx = rng.integers(0, len(y), len(y))
                differences.append(average_precision_score(y[idx], scores.activity_sna.to_numpy()[idx]) -
                                   average_precision_score(y[idx], scores.activity.to_numpy()[idx]))
            bootstrap.append({"model": model, "seed": seed, "n": len(y), "replicates": 500,
                "ap_difference": average_precision_score(y, scores.activity_sna) - average_precision_score(y, scores.activity),
                "lower": float(np.quantile(differences, .025)), "upper": float(np.quantile(differences, .975))})
    pd.DataFrame(bootstrap).to_csv(root / "paired_bootstrap.csv", index=False)
    def summarise(group):
        return {key: float(group[key].mean()) for key in group.select_dtypes(include=[np.number]).columns} | {
            key + "_sd": float(group[key].std(ddof=1)) for key in group.select_dtypes(include=[np.number]).columns}
    for (model, features), group in tabular.groupby(["learner", "features"], sort=False):
        summaries.append({"model": model, "features": features, **summarise(group)})
    neural_frame = pd.DataFrame(neural)
    for model, group in neural_frame.groupby("model", sort=False):
        summaries.append({"model": model, "features": "activity_sna", **summarise(group)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "SUMMARY_EXTERNAL_RESULTS.csv", index=False)
    names = {"tet_activity_sna": "TET-4", "dual_tet_activity_sna": "Dual-horizon TET", "cons_tet_activity_sna": "Consistency TET", "pc_tet_activity_sna": "PC-TET"}
    def cell(row, name):
        return f"{row[name]:.4f} $\\pm$ {row[name + '_sd']:.4f}"
    tab_rows, objective_rows = [], []
    for row in summaries:
        if row["model"] in names:
            objective_rows.append(names[row["model"]] + " & " + " & ".join(cell(row, name) for name in ["ap", "prefix_ap", "gap"]))
        else:
            tab_rows.append(row["model"] + " & " + ("Activity" if row["features"] == "activity" else "Activity + SNA") + " & " +
                            " & ".join(cell(row, name) for name in ["ap", "f1", "p_at_1pct"]))
    (args.bundle / "generated/bert4eth_tabular_rows.tex").write_text(" \\\\\n".join(tab_rows), encoding="utf-8")
    (args.bundle / "generated/bert4eth_objective_rows.tex").write_text(" \\\\\n".join(objective_rows), encoding="utf-8")
    save_json(root / "EXTERNAL_VERIFICATION.json", {"timestamp": utc_now(), "protocol_sha256": frozen["protocol_sha256"],
                "metric_and_threshold_rows_verified": len(checks), "checks": checks, "bootstrap_rows": bootstrap,
                "numerical_tolerance": "1e-10, except Brier 1e-7 for float32/float64 promotion in the combined tabular predictions",
                "data_cutoff_and_disjointness_checks": "passed", "test_alert_count": int(np.ceil(.01 * prepared["cohort_counts"][0]["0"] + .01 * prepared["cohort_counts"][0]["1"]))})
    print(summary[["model", "features", "ap", "f1", "p_at_1pct", "prefix_ap", "gap"]].to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))


if __name__ == "__main__":
    main()
