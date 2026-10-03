"""Shared provenance utilities for the locked revision runs."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def environment() -> dict:
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def prediction_diagnostics(y, score) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    bins = np.minimum((score * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(float(score[bins == b].mean()) - float(y[bins == b].mean()))
              for b in range(10) if np.any(bins == b))
    return {'score_mean': float(score.mean()), 'score_sd': float(score.std()),
            'score_p05': float(np.quantile(score, .05)), 'score_p95': float(np.quantile(score, .95)),
            'positive_score_mean': float(score[y == 1].mean()),
            'negative_score_mean': float(score[y == 0].mean()), 'ece_10bins': float(ece)}
