from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure


@dataclass
class TrainAEConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 6


def run_epoch(model, loader, device, optimizer=None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.MSELoss()

    total = 0.0
    n = 0

    for x in tqdm(loader, desc="train" if train_mode else "val", leave=False):
        x = x.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        recon = model(x)
        loss = loss_fn(recon, x)

        if train_mode:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


def save_recon_samples(model, X_val: np.ndarray, device, n_samples: int = 3) -> str:
    model.eval()
    idx = np.random.choice(X_val.shape[0], size=min(n_samples, X_val.shape[0]), replace=False)
    X = torch.from_numpy(X_val[idx]).to(device)

    with torch.no_grad():
        R = model(X).cpu().numpy()

    fig = plt.figure()
    for i in range(len(idx)):
        plt.plot(X_val[idx[i], :, 0], label=f"x_{i}")
        plt.plot(R[i, :, 0], label=f"recon_{i}", linestyle="--")
    plt.title("Model-2 AE: Recon Samples (val)")
    plt.xlabel("t")
    plt.ylabel("z-scored power")
    plt.legend()

    return save_figure(fig, "04_recon_samples.png")


def train_autoencoder(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainAEConfig,
    X_val_for_plot: np.ndarray,
    ckpt_dir: str = "outputs/checkpoints",
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0
    best_file = ckpt_dir / "model2_ae_best.pt"

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer=opt)
        va = run_epoch(model, val_loader, device, optimizer=None)

        train_losses.append(tr)
        val_losses.append(va)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={tr:.6f} | val_loss={va:.6f}")

        if va < best_val:
            best_val = va
            patience = 0
            torch.save({"model_state": model.state_dict(), "best_val_loss": best_val}, best_file.as_posix())
            print(f"Saved best: {best_file.as_posix()} (val_loss={best_val:.6f})")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(" Early stopping triggered.")
                break

    # Loss curve
    fig = plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title("Model-2 AE Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    p = save_figure(fig, "04_ae_loss_curve.png")
    print(f"Figure saved: {p}")

    # Recon samples
    p2 = save_recon_samples(model, X_val_for_plot, device, n_samples=3)
    print(f"Figure saved: {p2}")

    return best_file
