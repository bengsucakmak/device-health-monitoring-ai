from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure


def plot_pred_vs_gt_samples(pred: np.ndarray, gt: np.ndarray, n_samples: int = 3) -> str:
    n = min(n_samples, pred.shape[0])
    fig = plt.figure()
    for i in range(n):
        plt.plot(gt[i], label=f"gt_{i}")
        plt.plot(pred[i], label=f"pred_{i}", linestyle="--")
    plt.title("Model-1: Pred vs GT (sample windows)")
    plt.xlabel("t (within window)")
    plt.ylabel("Watt")
    plt.legend()
    return save_figure(fig, "03_pred_vs_gt_samples.png")


def plot_error_hist(pred: np.ndarray, gt: np.ndarray) -> str:
    err = (pred - gt).ravel()
    fig = plt.figure()
    plt.hist(err, bins=100)
    plt.title("Model-1: Error Histogram (pred - gt)")
    plt.xlabel("error (W)")
    plt.ylabel("count")
    return save_figure(fig, "03_error_hist.png")


def plot_scatter(pred: np.ndarray, gt: np.ndarray, max_points: int = 200000) -> str:
    p = pred.ravel()
    g = gt.ravel()

    if p.shape[0] > max_points:
        idx = np.random.choice(p.shape[0], size=max_points, replace=False)
        p = p[idx]
        g = g[idx]

    fig = plt.figure()
    plt.scatter(g, p, s=1)
    plt.title("Model-1: Scatter (GT vs Pred)")
    plt.xlabel("GT (W)")
    plt.ylabel("Pred (W)")
    return save_figure(fig, "03_scatter_pred_vs_gt.png")
