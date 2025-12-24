from __future__ import annotations
import pandas as pd

def hourly_event_profile(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Returns hourly anomaly-point profile (simple + explainable).
    Columns: hour, anomaly_points
    """
    s = scores_df.copy()
    s["is_anom"] = s["anomaly_score"] > threshold
    s["hour"] = pd.to_datetime(s["timestamp"]).dt.hour
    prof = s.groupby("hour")["is_anom"].sum().reset_index()
    prof = prof.rename(columns={"is_anom": "anomaly_points"})
    return prof
