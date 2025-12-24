from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure


def plot_score_timeseries(df_scores: pd.DataFrame, threshold: float) -> str:
    fig = plt.figure()
    plt.plot(df_scores["timestamp"], df_scores["anomaly_score"], label="anomaly_score")
    plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.4f}")
    plt.title("Model-2: Anomaly Score Over Time")
    plt.xlabel("time")
    plt.ylabel("score (recon MSE)")
    plt.legend()
    return save_figure(fig, "05_anomaly_score_timeseries.png")


def plot_score_hist(df_scores: pd.DataFrame, threshold: float) -> str:
    fig = plt.figure()
    plt.hist(df_scores["anomaly_score"].values, bins=100)
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold:.4f}")
    plt.title("Model-2: Score Histogram")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    return save_figure(fig, "05_anomaly_score_hist.png")


def plot_anomaly_flags(df_scores: pd.DataFrame, threshold: float) -> str:
    fig = plt.figure()
    flags = (df_scores["anomaly_score"].values > threshold).astype(int)
    plt.plot(df_scores["timestamp"], flags, label="anomaly_flag (0/1)")
    plt.title("Model-2: Anomaly Flags")
    plt.xlabel("time")
    plt.ylabel("flag")
    plt.legend()
    return save_figure(fig, "05_anomaly_flags.png")
