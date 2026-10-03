from __future__ import annotations

import argparse
import copy
import json
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from reconstruct_wallet_experiments import ACTIVITY, SNA, precision_at_fraction, select_f1_threshold
from audit_utils import environment, sha256, config_hash, utc_now


PAIR_COLUMNS = [
    "log_in_count",
    "log_out_count",
    "reciprocal",
    "relative_first",
    "relative_last",
    "in_share",
    "out_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_counterparty_table(samples: pd.DataFrame, incident: pd.DataFrame) -> pd.DataFrame:
    counts = (
        incident.groupby(["address", "counterparty", "direction"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ("in", "out"):
        if column not in counts:
            counts[column] = 0
    timing = incident.groupby(["address", "counterparty"], observed=True)["Time step"].agg(["min", "max"]).reset_index()
    frame = counts.merge(timing, on=["address", "counterparty"], validate="one_to_one")
    frame = frame.merge(
        samples[["address", "first_step", "in_pair_events", "out_pair_events"]],
        on="address",
        how="left",
        validate="many_to_one",
    )
    frame["log_in_count"] = np.log1p(frame["in"].astype("float32"))
    frame["log_out_count"] = np.log1p(frame["out"].astype("float32"))
    frame["reciprocal"] = ((frame["in"] > 0) & (frame["out"] > 0)).astype("float32")
    frame["relative_first"] = ((frame["min"] - frame["first_step"]) / 3.0).astype("float32")
    frame["relative_last"] = ((frame["max"] - frame["first_step"]) / 3.0).astype("float32")
    frame["in_share"] = (frame["in"] / frame["in_pair_events"].clip(lower=1)).astype("float32")
    frame["out_share"] = (frame["out"] / frame["out_pair_events"].clip(lower=1)).astype("float32")
    return frame[["address", "counterparty", *PAIR_COLUMNS]]


class CausalEgoSAGE(nn.Module):
    def __init__(self, wallet_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(wallet_dim + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, wallet_x: torch.Tensor, pair_x: torch.Tensor, pair_wallet: torch.Tensor) -> torch.Tensor:
        message = self.message(pair_x)
        n_wallets = wallet_x.shape[0]
        summed = torch.zeros((n_wallets, message.shape[1]), device=message.device, dtype=message.dtype)
        summed.index_add_(0, pair_wallet, message)
        count = torch.bincount(pair_wallet, minlength=n_wallets).clamp(min=1).to(message.dtype).unsqueeze(1)
        mean_pool = summed / count

        max_pool = torch.full_like(summed, -torch.inf)
        index = pair_wallet.unsqueeze(1).expand_as(message)
        max_pool.scatter_reduce_(0, index, message, reduce="amax", include_self=True)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.classifier(torch.cat([wallet_x, mean_pool, max_pool], dim=1)).squeeze(1)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score >= threshold
    return {
        "ap": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "p_at_1pct": precision_at_fraction(y, score),
        "threshold": float(threshold),
    }


def train_variant(
    samples: pd.DataFrame,
    pair_table: pd.DataFrame,
    wallet_columns: list[str],
    seed: int,
    epochs: int,
    patience: int,
    output_dir: Path,
    name: str,
) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = samples.reset_index(drop=True).copy()
    address_index = pd.Series(np.arange(len(samples), dtype=np.int64), index=samples["address"])
    pair_table = pair_table.loc[pair_table["address"].isin(address_index.index)].copy()
    pair_table["wallet_index"] = pair_table["address"].map(address_index).astype("int64")

    train_mask_np = samples["split"].eq("train").to_numpy()
    validation_mask_np = samples["split"].eq("validation").to_numpy()
    test_mask_np = samples["split"].eq("test").to_numpy()

    wallet_scaler = StandardScaler().fit(samples.loc[train_mask_np, wallet_columns])
    wallet_np = wallet_scaler.transform(samples[wallet_columns]).astype("float32")
    train_addresses = set(samples.loc[train_mask_np, "address"])
    pair_scaler = StandardScaler().fit(pair_table.loc[pair_table["address"].isin(train_addresses), PAIR_COLUMNS])
    pair_np = pair_scaler.transform(pair_table[PAIR_COLUMNS]).astype("float32")

    wallet_x = torch.from_numpy(wallet_np).to(device)
    pair_x = torch.from_numpy(pair_np).to(device)
    pair_wallet = torch.from_numpy(pair_table["wallet_index"].to_numpy("int64")).to(device)
    labels = torch.from_numpy(samples["label"].to_numpy("float32")).to(device)
    train_mask = torch.from_numpy(train_mask_np).to(device)

    positives = float(samples.loc[train_mask_np, "label"].sum())
    negatives = float(train_mask_np.sum() - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    model = CausalEgoSAGE(len(wallet_columns), len(PAIR_COLUMNS)).to(device)
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_ap = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    y_validation = samples.loc[validation_mask_np, "label"].to_numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(wallet_x, pair_x, pair_wallet)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(wallet_x, pair_x, pair_wallet)).detach().cpu().numpy()
        validation_ap = float(average_precision_score(y_validation, score[validation_mask_np]))
        history.append({"epoch": float(epoch), "loss": float(loss.item()), "validation_ap": validation_ap})
        print(f"seed={seed} columns={len(wallet_columns)} epoch={epoch} loss={loss.item():.6f} val_ap={validation_ap:.6f}", flush=True)
        if validation_ap > best_ap + 1e-5:
            best_ap = validation_ap
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(wallet_x, pair_x, pair_wallet)).detach().cpu().numpy()
    threshold = select_f1_threshold(y_validation, score[validation_mask_np])
    y_test = samples.loc[test_mask_np, "label"].to_numpy()
    result = metrics(y_test, score[test_mask_np], threshold)
    result["validation_ap"] = best_ap
    result["epochs"] = float(len(history))
    prediction = samples.loc[test_mask_np, ["address", "label", "first_step", "cutoff"]].copy()
    prediction["score"] = score[test_mask_np]
    prediction["threshold"] = threshold
    torch.save(best_state, output_dir / f"{name}_model.pt")
    joblib.dump({"wallet_scaler": wallet_scaler, "pair_scaler": pair_scaler}, output_dir / f"{name}_preprocessing.joblib")
    validation_prediction = samples.loc[validation_mask_np, ["address", "label", "first_step", "cutoff"]].copy()
    validation_prediction["score"], validation_prediction["threshold"] = score[validation_mask_np], threshold
    validation_prediction.to_parquet(output_dir / f"{name}_validation_predictions.parquet", index=False)
    result["parameter_count"] = sum(p.numel() for p in model.parameters())
    result["elapsed_seconds"] = time.perf_counter()-started
    result["run_timestamp"] = utc_now()
    return result, prediction, history


def main() -> None:
    torch.set_num_threads(4)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(args.results_dir / "wallet_samples.parquet")
    incident = pd.read_parquet(args.results_dir / "visible_incident_pairs.parquet")
    pair_table = build_counterparty_table(samples, incident)
    pair_table.to_parquet(args.output_dir / "counterparty_snapshots.parquet", index=False)

    rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    for name, columns in (("ego_activity", ACTIVITY), ("ego_activity_sna", ACTIVITY + SNA)):
        result, prediction, history = train_variant(
            samples,
            pair_table,
            columns,
            args.seed,
            args.epochs,
            args.patience,
            args.output_dir,
            name,
        )
        rows.append({"model": "CausalEgoSAGE", "variant": name, "seed": args.seed, **result})
        prediction["model"] = "CausalEgoSAGE"
        prediction["variant"] = name
        prediction["seed"] = args.seed
        all_predictions.append(prediction)
        history_frame = pd.DataFrame(history)
        history_frame["variant"] = name
        history_frame["seed"] = args.seed
        all_history.append(history_frame)

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.output_dir / "ego_model_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(args.output_dir / "ego_test_predictions.parquet", index=False)
    pd.concat(all_history, ignore_index=True).to_csv(args.output_dir / "ego_training_history.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "pair_features": PAIR_COLUMNS,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "n_counterparty_snapshots": int(len(pair_table)),
                "environment": environment(), "script_sha256": sha256(Path(__file__)),
                "configuration_hash": config_hash(vars(args)),
                "lr": .001, "weight_decay": .0001, "dropout": .15,
                "batch": "full local-ego tensor; no cross-wallet message passing", "initialization": "PyTorch defaults",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
