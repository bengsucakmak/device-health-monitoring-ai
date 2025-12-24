from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AlertSummary:
    last_anomaly_ts: pd.Timestamp | None
    repeats_24h: int                 # "event" count in last 24h
    consecutive_windows: int         # trailing consecutive anomaly windows
    anomaly_points_24h: int          # raw points over threshold (debug/extra)
    severity: str                    # "none" | "soft" | "critical"
    message_title: str
    message_body: str


def _median_step_minutes(ts: pd.Series) -> float:
    if len(ts) < 3:
        return 60.0
    diffs = ts.sort_values().diff().dropna().dt.total_seconds().values / 60.0
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 60.0
    return float(np.median(diffs))


def _trailing_consecutive(flags: np.ndarray) -> int:
    # Count consecutive True values from the end.
    if len(flags) == 0:
        return 0
    c = 0
    for v in flags[::-1]:
        if v:
            c += 1
        else:
            break
    return c


def _event_count(ts: pd.Series, is_anom: pd.Series, gap_minutes: float) -> int:
    # Count anomaly "events": consecutive anomaly points are one event;
    # a new event starts if time gap > gap_minutes.
    df = pd.DataFrame({"ts": ts, "is_anom": is_anom}).sort_values("ts")
    df = df[df["is_anom"]].copy()
    if len(df) == 0:
        return 0

    df["dt_min"] = df["ts"].diff().dt.total_seconds() / 60.0
    # New event if first row OR gap too large
    df["new_event"] = (df["dt_min"].isna()) | (df["dt_min"] > gap_minutes)
    return int(df["new_event"].sum())


def build_alert_summary(
    scores_df: pd.DataFrame,
    threshold: float,
    now: pd.Timestamp | None = None,
    lookback_hours: int = 24,
) -> AlertSummary:
    """
    scores_df: must include ['timestamp','anomaly_score'].
    threshold: anomaly threshold.
    lookback_hours: window for "repeats".
    """
    s = scores_df.copy()
    s = s.sort_values("timestamp")
    s["is_anom"] = s["anomaly_score"] > threshold

    if now is None:
        now = pd.Timestamp(s["timestamp"].max())

    # last anomaly time (global)
    last_anom_ts = None
    if s["is_anom"].any():
        last_anom_ts = pd.Timestamp(s.loc[s["is_anom"], "timestamp"].max())

    # 24h slice
    start_24 = now - pd.Timedelta(hours=int(lookback_hours))
    s24 = s[s["timestamp"] >= start_24].copy()

    step_min = _median_step_minutes(s["timestamp"])
    # event gap: if sampling is 1 min, consecutive points are one event.
    # allow some missing points → 2.5x median step is a good heuristic.
    gap_min = max(5.0, 2.5 * step_min)

    repeats_24h = _event_count(s24["timestamp"], s24["is_anom"], gap_minutes=gap_min)
    points_24h = int(s24["is_anom"].sum())
    consecutive = _trailing_consecutive(s["is_anom"].to_numpy(dtype=bool))

    # Severity rules (simple + B2C-friendly)
    # - critical: long streak OR too many repeats in 24h
    # - soft: any repeat OR short streak
    severity = "none"
    if consecutive >= 6 or repeats_24h >= 3:
        severity = "critical"
    elif consecutive >= 2 or repeats_24h >= 1:
        severity = "soft"

    if severity == "none":
        title = "Şimdilik sorun yok"
        body = "Anomali sinyali tespit edilmedi. İzlemeye devam edeceğiz."
    elif severity == "soft":
        title = "Dikkat: sınırda dalgalanma"
        body = "Birkaç kez anomali benzeri davranış görüldü. Bugün ekstra gözlem önerilir."
    else:
        title = "Kritik: tekrarlayan anomali"
        body = "Anomali davranışı tekrar ediyor veya art arda sürüyor. Fiziksel kontrol / servis değerlendirin."

    return AlertSummary(
        last_anomaly_ts=last_anom_ts,
        repeats_24h=repeats_24h,
        consecutive_windows=consecutive,
        anomaly_points_24h=points_24h,
        severity=severity,
        message_title=title,
        message_body=body,
    )
