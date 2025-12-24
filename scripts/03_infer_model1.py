from __future__ import annotations

import numpy as np
import torch

from src.utils.config import load_yaml
from src.data.dataset import make_loader
from src.model1.tcn import TCNDisaggregator
from src.model1.infer import compute_metrics_np, export_fridge_pred_csv, infer_windows
from src.model1.plots import plot_error_hist, plot_pred_vs_gt_samples, plot_scatter


def main() -> None:
    cfg = load_yaml("src/config/default.yaml")

    sampling = cfg["data"]["sampling"]
    window_size = int(cfg["windowing"]["window_size"])

    batch_size = 64
    if "model1" in cfg and "batch_size" in cfg["model1"]:
        batch_size = int(cfg["model1"]["batch_size"])

    # Test loader (shuffle=False zorunlu, t0 sırası bozulmasın)
    test_loader = make_loader("data/processed/test_windows.npz", batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model
    model = TCNDisaggregator(
        hidden=64,
        num_blocks=6,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    # Load checkpoint
    ckpt_path = "outputs/checkpoints/model1_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {ckpt_path}")

    # Inference
    pred, gt, t0 = infer_windows(model, test_loader, device)

    # Metrics
    m = compute_metrics_np(pred, gt)
    print("Test metrics:", m)

    # Plots
    p1 = plot_pred_vs_gt_samples(pred, gt, n_samples=3)
    p2 = plot_error_hist(pred, gt)
    p3 = plot_scatter(pred, gt)
    print(" Saved figures:", p1, p2, p3)

    # Export fridge_pred.csv
    out_csv = "data/processed/fridge_pred.csv"
    saved_csv = export_fridge_pred_csv(
        pred=pred,
        t0=t0,
        window_size=window_size,
        sampling=sampling,
        out_csv=out_csv,
    )
    print(f"Exported: {saved_csv}")

    print("\nNEXT: Faz 2 -> Model-2 LSTM Autoencoder (anomaly scoring)")


if __name__ == "__main__":
    main()
