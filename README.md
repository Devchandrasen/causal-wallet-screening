# Causal wallet screening

Research code for **Causal Wallet Screening with Counterparty Features and Prefix-Consistent Temporal Learning**, by Ashish Pandey, Chandrasen Pandey and Stuti Pandey.

The experiments reconstruct a wallet's activity and counterparty distribution from events available by its decision cutoff. The primary Bitcoin task uses four Elliptic++ time steps. The temporal model compares separately reconstructed full and prefix views. Classical, local relational and two-hop TGAT comparisons, temporal-shift controls and Ethereum profile transfer are included.

## Public deposit contents

`analysis/` contains the executable reconstruction, learner, campaign and verification scripts, pinned configuration and environment requirements. `results/MASTER_RESULTS.csv` records all 108 revised Bitcoin/profile-transfer metric rows. Independent metric, validation-threshold and source-provenance checks accompany it.

This code-and-summary deposit does **not** include raw datasets, reviewer correspondence, submission documents, large fitted checkpoints, or per-address predictions. The complete fitted-run supplement is separate. A DOI has not been registered. Public visibility is not an assertion that a reuse license has been granted: the authors have not yet selected a code license. Source datasets remain subject to their own terms.

## Input sources

- [Elliptic++](https://github.com/git-disl/EllipticPlusPlus), commit `08fe6aded83afb97bf5a79a71130f542ca783c2e`. Obtain the linked actor files `txs_features.csv`, `wallets_classes.csv`, `AddrTx_edgelist.csv`, and `TxAddr_edgelist.csv`. The repository checkout also supplies the transaction-class and transaction-edge files for linked controls.
- [Real-CATS](https://github.com/sjdseu/Real-CATS), commit `99623a1e588500d84379d58f17ec38e2d1816ed3`, with `BE.tsv`, `CE.tsv`, and `Sup-CATS.tsv`.
- [BERT4ETH](https://github.com/git-disl/BERT4ETH), commit `9513ea4c5f8b7a411960b8a5f0a0a9c836d86647`, is the source for the separate timestamped Ethereum replication protocol. Its authors link the phishing and normal transaction archives in their README. The protocol is frozen locally before model fitting; execution status and results are reported separately, not inferred from this protocol file.

## Reproduce the primary campaign

Use a separate Python environment. The recorded versions are Python 3.12.10, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, XGBoost 3.2.0 and PyTorch 2.11.0+cu128. Install `analysis/requirements_analysis.txt`, choosing an official PyTorch wheel appropriate for your hardware. GPU results can have residual nondeterminism.

```text
python analysis/run_locked_pipeline.py --elliptic-data PATH_TO_ACTOR_FILES --elliptic-repo PATH_TO_ELLIPTIC_CHECKOUT --real-cats PATH_TO_REAL_CATS --output analysis/new_rerun --bundle generated_publication
```

Use a fresh output directory. `--resume` is for continuing the same unchanged configuration, not for comparing changed settings. This command fits models and generates publication tables/figures, but does not submit or publish anything.

The full fitted supplement permits verification without fitting:

```text
python analysis/summarize_locked_results.py --root analysis/locked_20261002 --bundle generated_publication
python analysis/verify_thresholds.py --root analysis/locked_20261002
```

These verification commands require that supplement's prediction and derived-view files, which are not part of this lightweight Git repository. The public aggregate checks do not substitute for the missing predictions.

## Interpretation

The Bitcoin chronological holdout was examined in the earlier study. These revised results are audited historical evaluations, not preregistered confirmation. Counterparty features improve the primary tabular ranking, but temporal deterioration persists. The consistency term contributes little additional stability beyond prefix supervision in the four-way ablation. Real-CATS/Sup-CATS completed-profile transfer performs poorly at the top alert budget and is not an eight-week causal replication.

Addresses are pseudonymous ledger identifiers, not verified people. Model outputs are screening scores, not evidence of identity or criminal conduct. Metrics and uncertainty are conditional on dataset sampling and retrospective labels.
