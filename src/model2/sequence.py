from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class SeqData:
    X: np.ndarray            # (N, L, 1)
    t0: np.ndarray           # (N,) int64 ns
    scaler: Dict[str, float] # {"mean":..., "std":...}


def load_pred_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # timestamp tekil olmalı
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    s = pd.Series(df["predicted_power"].values, index=df["timestamp"])
    return s.astype("float32")


def make_sequences(s: pd.Series, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    values = s.values.astype(np.float32)
    times = s.index.view("int64")  # ns

    X_list = []
    t0_list = []

    for start in range(0, len(values) - seq_len + 1, stride):
        end = start + seq_len
        X_list.append(values[start:end])
        t0_list.append(times[start])

    if len(X_list) == 0:
        X = np.empty((0, seq_len, 1), dtype=np.float32)
        t0 = np.empty((0,), dtype=np.int64)
        return X, t0

    X = np.stack(X_list).astype(np.float32)  # (N, L)
    t0 = np.array(t0_list, dtype=np.int64)

    # (N,L) -> (N,L,1)
    return X[..., None], t0


def zscore_fit(x: np.ndarray) -> Dict[str, float]:
    return {"mean": float(x.mean()), "std": float(x.std() + 1e-6)}


def zscore_apply(x: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
    return (x - scaler["mean"]) / (scaler["std"] + 1e-6)


def build_train_test_sequences(
    csv_path: str,
    seq_len: int,
    stride: int,
    train_ratio: float,
    normalize: str = "zscore",
) -> Tuple[SeqData, SeqData]:
    """
    csv -> time-based split -> sequences
    normalize scaler only from TRAIN.
    """
    s = load_pred_series(csv_path)

    n = len(s)
    n_train = int(n * float(train_ratio))

    s_train = s.iloc[:n_train]
    s_test = s.iloc[n_train:]

    Xtr, t0tr = make_sequences(s_train, seq_len, stride)
    Xte, t0te = make_sequences(s_test, seq_len, stride)

    if normalize == "zscore":
        scaler = zscore_fit(Xtr.reshape(-1))
        Xtr = zscore_apply(Xtr, scaler)
        Xte = zscore_apply(Xte, scaler)
    else:
        scaler = {"mean": 0.0, "std": 1.0}

    train_data = SeqData(X=Xtr, t0=t0tr, scaler=scaler)
    test_data = SeqData(X=Xte, t0=t0te, scaler=scaler)
    return train_data, test_data
