from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class ScoreResult:
    timestamps: pd.DatetimeIndex
    scores: np.ndarray           # (N,) sequence-level score
    threshold: float
    is_anomaly: np.ndarray       # (N,) bool
    method: str


@torch.no_grad()
def reconstruction_scores(
    model: torch.nn.Module,
    X: np.ndarray,          # (N, L, 1) normalized
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Sequence-level reconstruction error (MSE over time).
    Returns: (N,)
    """
    model.eval()
    loss_fn = nn.MSELoss(reduction="none")

    scores = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        recon = model(xb)
        # per-sample MSE: mean over (L,1)
        mse = loss_fn(recon, xb).mean(dim=(1, 2))
        scores.append(mse.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def percentile_threshold(scores: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(scores, q))


def build_score_dataframe(
    t0: np.ndarray,
    scores: np.ndarray,
    seq_len: int,
    sampling: str,
) -> pd.DataFrame:
    """
    score'u sequence başlangıç timestamp'ine yazarız (t0).
    İstersen daha sonra score'u sequence ortasına da yazabiliriz.
    """
    ts = pd.to_datetime(t0, unit="ns")
    df = pd.DataFrame({"timestamp": ts, "anomaly_score": scores})
    df = df.sort_values("timestamp")
    return df
