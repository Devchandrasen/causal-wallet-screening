"""Train a cutoff-safe temporal ego transformer for wallet screening.

PC-TET uses identifier-free counterparty tokens, a Transformer encoder, and a
prefix-consistency objective. All full and prefix inputs are reconstructed from
events visible by the corresponding cutoff.
"""

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
from torch.utils.data import DataLoader, TensorDataset

from reconstruct_wallet_experiments import ACTIVITY, precision_at_fraction, select_f1_threshold
from audit_utils import config_hash, environment, prediction_diagnostics, sha256, utc_now


PAIR_COLUMNS = [
    "log_in_count",
    "log_out_count",
    "reciprocal",
    "relative_first",
    "relative_last",
    "in_share",
    "out_share",
]

PAIR_ACTIVITY = [
    "unique_transactions",
    "incoming_transactions",
    "outgoing_transactions",
    "active_steps",
    "pair_event_count",
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
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--variant", choices=["tet_activity", "tet_activity_sna", "dual_tet_activity_sna", "cons_tet_activity_sna", "pc_tet_activity_sna"], required=True)
    return parser.parse_args()


def build_views(
    samples: pd.DataFrame,
    incident: pd.DataFrame,
    horizon_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per wallet-counterparty and one row per wallet."""
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
    pairs = counts.merge(timing, on=["address", "counterparty"], validate="one_to_one")

    direction_totals = incident.groupby(["address", "direction"], observed=True).size().unstack(fill_value=0)
    for column in ("in", "out"):
        if column not in direction_totals:
            direction_totals[column] = 0
    direction_totals = direction_totals.rename(columns={"in": "in_pair_events", "out": "out_pair_events"}).reset_index()
    pairs = pairs.merge(direction_totals, on="address", how="left", validate="many_to_one")
    first_step = samples.set_index("address")["first_step"]
    pairs["first_step"] = pairs["address"].map(first_step)
    denominator = float(max(1, horizon_steps - 1))
    pairs["log_in_count"] = np.log1p(pairs["in"].astype("float32"))
    pairs["log_out_count"] = np.log1p(pairs["out"].astype("float32"))
    pairs["reciprocal"] = ((pairs["in"] > 0) & (pairs["out"] > 0)).astype("float32")
    pairs["relative_first"] = ((pairs["min"] - pairs["first_step"]) / denominator).astype("float32")
    pairs["relative_last"] = ((pairs["max"] - pairs["first_step"]) / denominator).astype("float32")
    pairs["in_share"] = (pairs["in"] / pairs["in_pair_events"].clip(lower=1)).astype("float32")
    pairs["out_share"] = (pairs["out"] / pairs["out_pair_events"].clip(lower=1)).astype("float32")
    pairs["pair_total"] = pairs["in"] + pairs["out"]

    wallet = samples[["address", "first_step"]].copy().set_index("address")
    grouped = incident.groupby("address", observed=True)
    wallet["unique_transactions"] = grouped["txId"].nunique()
    wallet["active_steps"] = grouped["Time step"].nunique()
    wallet["pair_event_count"] = grouped.size()
    wallet["last_event"] = grouped["Time step"].max()
    wallet["total_counterparties"] = grouped["counterparty"].nunique()

    for direction, prefix in (("in", "incoming"), ("out", "outgoing")):
        subset = incident.loc[incident["direction"].eq(direction)]
        by_address = subset.groupby("address", observed=True)
        wallet[f"{prefix}_transactions"] = by_address["txId"].nunique()
        wallet[f"{direction}_counterparties"] = by_address["counterparty"].nunique()
        wallet[f"{direction}_pair_events"] = by_address.size()

    wallet = wallet.fillna(0)
    cutoff = wallet["first_step"] + horizon_steps - 1
    wallet["event_rate"] = wallet["pair_event_count"] / float(horizon_steps)
    wallet["incoming_event_fraction"] = wallet["in_pair_events"] / wallet["pair_event_count"].clip(lower=1)
    wallet["last_event_lag"] = cutoff - wallet["last_event"]
    wallet["in_repeat_fraction"] = 1.0 - wallet["in_counterparties"] / wallet["in_pair_events"].clip(lower=1)
    wallet["out_repeat_fraction"] = 1.0 - wallet["out_counterparties"] / wallet["out_pair_events"].clip(lower=1)

    reciprocal = pairs.loc[pairs["reciprocal"].eq(1)].groupby("address", observed=True).size()
    wallet["reciprocal_counterparties"] = reciprocal
    wallet["reciprocal_counterparties"] = wallet["reciprocal_counterparties"].fillna(0)
    wallet["reciprocal_fraction"] = wallet["reciprocal_counterparties"] / wallet["total_counterparties"].clip(lower=1)

    for direction, output in (("in", "max_in_concentration"), ("out", "max_out_concentration")):
        maximum = pairs.groupby("address", observed=True)[direction].max()
        wallet[output] = maximum / wallet[f"{direction}_pair_events"].clip(lower=1)
    wallet["counterparty_diversity"] = wallet["total_counterparties"] / wallet["pair_event_count"].clip(lower=1)
    wallet = wallet.fillna(0).reset_index()
    return pairs, wallet


def address_activity(samples, events, horizon):
    """Same eight predictors as the tabular baseline, from address-tx events.

    Pair projection may multiply transaction incidences, so it must not define
    transaction activity. Prefix activity is rebuilt before any aggregation.
    """
    visible = events.merge(samples[["address", "first_step"]], on="address", validate="many_to_one")
    visible = visible[visible["Time step"] <= visible["first_step"] + horizon - 1]
    result = samples[["address", "first_step"]].set_index("address")
    grouped = visible.groupby("address", observed=True)
    result["unique_transactions"] = grouped["txId"].nunique()
    result["active_steps"] = grouped["Time step"].nunique()
    result["address_event_count"] = grouped.size()
    result["last_event"] = grouped["Time step"].max()
    for direction, prefix in [("in", "incoming"), ("out", "outgoing")]:
        result[f"{prefix}_transactions"] = visible[visible.direction == direction].groupby("address")["txId"].nunique()
    result = result.fillna(0)
    result["event_rate"] = result.address_event_count / horizon
    result["incoming_event_fraction"] = result.incoming_transactions / result.unique_transactions.clip(lower=1)
    result["last_event_lag"] = result.first_step + horizon - 1 - result.last_event
    return result.reset_index()[["address", *ACTIVITY]]


def token_tensor(
    samples: pd.DataFrame,
    pairs: pd.DataFrame,
    max_tokens: int,
    scaler: StandardScaler | None,
    train_addresses: set[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    pairs = pairs.sort_values(
        ["address", "pair_total", "relative_first", "counterparty"],
        ascending=[True, False, True, True],
        kind="stable",
    ).copy()
    pairs["rank"] = pairs.groupby("address", observed=True).cumcount()
    pairs = pairs.loc[pairs["rank"] < max_tokens]
    if scaler is None:
        scaler = StandardScaler().fit(pairs.loc[pairs["address"].isin(train_addresses), PAIR_COLUMNS])
    values = scaler.transform(pairs[PAIR_COLUMNS]).astype("float32")
    address_index = pd.Series(np.arange(len(samples), dtype=np.int64), index=samples["address"])
    wallet_index = pairs["address"].map(address_index).to_numpy(dtype=np.int64)
    rank = pairs["rank"].to_numpy(dtype=np.int64)
    tokens = np.zeros((len(samples), max_tokens, len(PAIR_COLUMNS)), dtype=np.float32)
    padding = np.ones((len(samples), max_tokens), dtype=bool)
    tokens[wallet_index, rank] = values
    padding[wallet_index, rank] = False
    return tokens, padding, scaler


class TemporalEgoTransformer(nn.Module):
    def __init__(self, wallet_dim: int, token_dim: int) -> None:
        super().__init__()
        model_dim = 64
        self.token_projection = nn.Sequential(nn.Linear(token_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim + wallet_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, wallet: torch.Tensor, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        token_state = self.token_projection(tokens)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        state = torch.cat([cls, token_state], dim=1)
        cls_padding = torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        state = self.encoder(state, src_key_padding_mask=torch.cat([cls_padding, padding], dim=1))
        return self.classifier(torch.cat([wallet, state[:, 0]], dim=1)).squeeze(1)


def score_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
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


def predict(
    model: nn.Module,
    wallet: torch.Tensor,
    tokens: torch.Tensor,
    padding: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(indices)), batch_size=batch_size, shuffle=False)
    offset = 0
    with torch.no_grad():
        for (batch_index,) in loader:
            batch_index = batch_index.long()
            probability = torch.sigmoid(
                model(
                    wallet[batch_index].to(device),
                    tokens[batch_index].to(device),
                    padding[batch_index].to(device),
                )
            ).cpu().numpy()
            output[offset : offset + len(probability)] = probability
            offset += len(probability)
    return output


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = pd.read_parquet(args.results_dir / "wallet_samples.parquet").reset_index(drop=True)
    cache = args.results_dir / "neural_views_v2"
    cache.mkdir(exist_ok=True)
    if not (cache / "prefix_wallet.parquet").exists():
        incident = pd.read_parquet(args.results_dir / "visible_incident_pairs.parquet")
        incident = incident.merge(samples[["address", "first_step"]], on="address", how="left", validate="many_to_one")
        assert (incident["Time step"] <= incident.first_step + 3).all()
        full_incident = incident.drop(columns="first_step")
        prefix_incident = incident.loc[incident["Time step"] <= incident["first_step"] + 1].drop(columns="first_step")
        full_pairs, full_wallet = build_views(samples, full_incident, 4)
        prefix_pairs, prefix_wallet = build_views(samples, prefix_incident, 2)
        events = pd.read_parquet(args.results_dir / "visible_address_events.parquet")
        full_activity = address_activity(samples, events, 4)
        prefix_activity = address_activity(samples, events, 2)
        full_wallet = full_activity.merge(full_wallet[["address", *SNA]], on="address", validate="one_to_one")
        prefix_wallet = prefix_activity.merge(prefix_wallet[["address", *SNA]], on="address", validate="one_to_one")
        np.testing.assert_allclose(full_wallet[ACTIVITY].to_numpy(), samples[ACTIVITY].to_numpy())
        for name, frame in [("full_pairs", full_pairs), ("prefix_pairs", prefix_pairs), ("full_wallet", full_wallet), ("prefix_wallet", prefix_wallet)]:
            frame.to_parquet(cache / f"{name}.parquet", index=False)
    full_pairs, prefix_pairs, full_wallet, prefix_wallet = [pd.read_parquet(cache / f"{name}.parquet") for name in ["full_pairs", "prefix_pairs", "full_wallet", "prefix_wallet"]]

    train_mask = samples["split"].eq("train").to_numpy()
    validation_mask = samples["split"].eq("validation").to_numpy()
    test_mask = samples["split"].eq("test").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)
    train_addresses = set(samples.loc[train_mask, "address"])

    full_tokens_np, full_padding_np, token_scaler = token_tensor(
        samples, full_pairs, args.max_tokens, None, train_addresses
    )
    prefix_tokens_np, prefix_padding_np, _ = token_tensor(
        samples, prefix_pairs, args.max_tokens, token_scaler, train_addresses
    )

    wallet_columns = ACTIVITY if args.variant == "tet_activity" else ACTIVITY + SNA
    wallet_scaler = StandardScaler().fit(full_wallet.loc[train_mask, wallet_columns])
    full_wallet_np = wallet_scaler.transform(full_wallet[wallet_columns]).astype("float32")
    prefix_wallet_np = wallet_scaler.transform(prefix_wallet[wallet_columns]).astype("float32")

    full_wallet_tensor = torch.from_numpy(full_wallet_np)
    prefix_wallet_tensor = torch.from_numpy(prefix_wallet_np)
    full_tokens = torch.from_numpy(full_tokens_np)
    prefix_tokens = torch.from_numpy(prefix_tokens_np)
    full_padding = torch.from_numpy(full_padding_np)
    prefix_padding = torch.from_numpy(prefix_padding_np)
    labels = torch.from_numpy(samples["label"].to_numpy(dtype=np.float32))

    positives = float(samples.loc[train_mask, "label"].sum())
    negatives = float(train_mask.sum() - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    model = TemporalEgoTransformer(len(wallet_columns), len(PAIR_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    best_ap = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    y_validation = samples.loc[validation_mask, "label"].to_numpy()
    prefix_weight, consistency_weight = {
        "tet_activity": (0., 0.), "tet_activity_sna": (0., 0.),
        "dual_tet_activity_sna": (.5, 0.), "cons_tet_activity_sna": (0., .2),
        "pc_tet_activity_sna": (.5, .2),
    }[args.variant]
    use_prefix = prefix_weight > 0 or consistency_weight > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        seen = 0
        for (batch_index,) in train_loader:
            batch_index = batch_index.long()
            y_batch = labels[batch_index].to(device)
            optimizer.zero_grad(set_to_none=True)
            full_logits = model(
                full_wallet_tensor[batch_index].to(device, non_blocking=True),
                full_tokens[batch_index].to(device, non_blocking=True),
                full_padding[batch_index].to(device, non_blocking=True),
            )
            loss = loss_fn(full_logits, y_batch)
            if use_prefix:
                prefix_logits = model(
                    prefix_wallet_tensor[batch_index].to(device, non_blocking=True),
                    prefix_tokens[batch_index].to(device, non_blocking=True),
                    prefix_padding[batch_index].to(device, non_blocking=True),
                )
                prefix_loss = loss_fn(prefix_logits, y_batch)
                consistency = torch.mean((torch.sigmoid(full_logits) - torch.sigmoid(prefix_logits)) ** 2)
                loss = loss + prefix_weight * prefix_loss + consistency_weight * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            cumulative_loss += float(loss.item()) * len(batch_index)
            seen += len(batch_index)

        validation_score = predict(
            model,
            full_wallet_tensor,
            full_tokens,
            full_padding,
            validation_indices,
            args.batch_size * 2,
            device,
        )
        validation_ap = float(average_precision_score(y_validation, validation_score))
        epoch_loss = cumulative_loss / max(1, seen)
        history.append({"epoch": epoch, "loss": epoch_loss, "validation_ap": validation_ap})
        print(
            f"variant={args.variant} seed={args.seed} epoch={epoch} loss={epoch_loss:.6f} val_ap={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_ap + 1e-5:
            best_ap = validation_ap
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_score = predict(
        model, full_wallet_tensor, full_tokens, full_padding, validation_indices, args.batch_size * 2, device
    )
    threshold = select_f1_threshold(y_validation, validation_score)
    test_score = predict(model, full_wallet_tensor, full_tokens, full_padding, test_indices, args.batch_size * 2, device)
    prefix_validation_score = predict(
        model, prefix_wallet_tensor, prefix_tokens, prefix_padding, validation_indices, args.batch_size * 2, device
    )
    prefix_threshold = select_f1_threshold(y_validation, prefix_validation_score)
    prefix_test_score = predict(
        model, prefix_wallet_tensor, prefix_tokens, prefix_padding, test_indices, args.batch_size * 2, device
    )
    y_test = samples.loc[test_mask, "label"].to_numpy()
    prefix_result = score_metrics(y_test, prefix_test_score, prefix_threshold)
    result = {
        "model": {"pc_tet_activity_sna": "PC-TET", "dual_tet_activity_sna": "Dual-horizon TET", "cons_tet_activity_sna": "Consistency TET"}.get(args.variant, "TET"),
        "variant": args.variant,
        "seed": args.seed,
        **score_metrics(y_test, test_score, threshold),
        **{f"prefix_{key}": value for key, value in prefix_result.items()},
        "mean_full_prefix_gap": float(np.mean(np.abs(test_score - prefix_test_score))),
        "validation_ap": best_ap,
        "epochs": len(history),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
        "max_tokens": args.max_tokens,
        **prediction_diagnostics(y_test, test_score),
        **{f"prefix_{k}": v for k, v in prediction_diagnostics(y_test, prefix_test_score).items()},
        "run_timestamp": utc_now(),
    }
    pd.DataFrame([result]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    prediction = samples.loc[test_mask, ["address", "label", "first_step", "cutoff"]].copy()
    prediction["score"] = test_score
    prediction["prefix_score"] = prefix_test_score
    prediction["threshold"] = threshold
    prediction["prefix_threshold"] = prefix_threshold
    prediction.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    validation_prediction = samples.loc[validation_mask, ["address", "label", "first_step", "cutoff"]].copy()
    validation_prediction["score"], validation_prediction["prefix_score"] = validation_score, prefix_validation_score
    validation_prediction["threshold"], validation_prediction["prefix_threshold"] = threshold, prefix_threshold
    validation_prediction.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    joblib.dump({"token_scaler": token_scaler, "wallet_scaler": wallet_scaler, "wallet_columns": wallet_columns}, args.output_dir / "preprocessing.joblib")
    torch.save(best_state, args.output_dir / "model_state.pt")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "seed": args.seed,
                "full_horizon_steps": 4,
                "prefix_steps": 2,
                "max_counterparty_tokens": args.max_tokens,
                "token_features": PAIR_COLUMNS,
                "wallet_features": wallet_columns,
                "prefix_objective": use_prefix,
                "loss": f"BCE(full) + {prefix_weight}*BCE(prefix) + {consistency_weight}*MSE(probabilities)",
                "device": str(device),
                "environment": environment(), "run_timestamp": result["run_timestamp"],
                "parameter_count": result["parameter_count"], "elapsed_seconds": result["elapsed_seconds"],
                "configuration_hash": config_hash(vars(args)), "script_sha256": sha256(Path(__file__)),
                "input_sha256": sha256(args.results_dir / "wallet_samples.parquet"),
                "optimizer": {"name": "AdamW", "lr": 8e-4, "weight_decay": 1e-4},
                "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
                "initialization": "PyTorch defaults; CLS normal(0,0.02). Cloned Transformer layers share initial values but have separate parameters.",
                "token_order": "descending pair count, ascending first interaction, lexical counterparty tie break; no positional embedding",
                "masking": "zero padded tokens excluded as attention keys; CLS always unmasked",
                "class_weight": negatives / positives,
                "historical_note": "Aligned address-event activity supersedes pair-event activity used in the earlier pilot. The holdout was previously examined.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
