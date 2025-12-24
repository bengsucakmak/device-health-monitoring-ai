from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    sampling: str
    drop_negative: bool = True
    fillna_method: str = "ffill"
    clip_max_watt: float = 5000.0


def preprocess_df(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    - resample (sampling)
    - negative değerleri temizle
    - fillna (ffill -> bfill)
    - aşırı uçları clip et
    """
    df = df.copy()

    # 1) Resample
    df = df.resample(cfg.sampling).mean()

    # 2) Negatifleri düzelt
    if cfg.drop_negative:
        df["aggregate"] = df["aggregate"].clip(lower=0)
        df["appliance"] = df["appliance"].clip(lower=0)

    # 3) Fill NaN
    if cfg.fillna_method == "ffill":
        df = df.ffill().bfill()
    elif cfg.fillna_method == "bfill":
        df = df.bfill().ffill()
    else:
        df = df.dropna()

    # 4) Clip extreme
    df["aggregate"] = df["aggregate"].clip(upper=cfg.clip_max_watt)

    return df


def time_split(df: pd.DataFrame, train: float, val: float, test: float):
    """
    Zaman sıralı split (shuffle yok).
    """
    assert abs(train + val + test - 1.0) < 1e-6, "split oranları 1 etmeli"

    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    return df_train, df_val, df_test


def make_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: aggregate windows, y: appliance windows, t0: her window başlangıç timestamp (int64 ns)
    """
    agg = df["aggregate"].to_numpy(dtype=np.float32)
    app = df["appliance"].to_numpy(dtype=np.float32)
    t_index = df.index.view("int64")  # ns

    X_list, y_list, t0_list = [], [], []

    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        X_list.append(agg[start:end])
        y_list.append(app[start:end])
        t0_list.append(t_index[start])

    X = np.stack(X_list) if X_list else np.empty((0, window_size), dtype=np.float32)
    y = np.stack(y_list) if y_list else np.empty((0, window_size), dtype=np.float32)
    t0 = np.array(t0_list, dtype=np.int64)

    return X, y, t0
