from __future__ import annotations
import numpy as np
import pandas as pd

def health_series_from_scores(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Returns df with columns: timestamp, anomaly_score, health
    health = 100 when score<=threshold; decays as score rises.
    """
    s = scores_df.sort_values("timestamp").copy()
    eps = 1e-12
    ratio = (s["anomaly_score"].to_numpy(dtype=np.float32) + eps) / float(threshold + eps)
    health = 100.0 - np.clip((ratio - 1.0) * 60.0, 0.0, 100.0)
    s["health"] = health
    return s[["timestamp", "anomaly_score", "health"]]

def health_delta_last_hours(health_df: pd.DataFrame, hours: int = 24) -> float:
    """
    Delta = last_health - health_at_(now-hours). Positive is improvement.
    """
    hdf = health_df.sort_values("timestamp")
    now = pd.Timestamp(hdf["timestamp"].max())
    t0 = now - pd.Timedelta(hours=int(hours))
    before = hdf[hdf["timestamp"] <= t0]
    if len(before) == 0:
        return float(hdf["health"].iloc[-1] - hdf["health"].iloc[0])
    return float(hdf["health"].iloc[-1] - before["health"].iloc[-1])
