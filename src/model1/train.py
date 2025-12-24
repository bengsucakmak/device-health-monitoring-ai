from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.metrics import mae, rmse
from src.utils.plotting import save_figure
import matplotlib.pyplot as plt


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 5


def run_epoch(model, loader, device, optimizer=None) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train" if train_mode else "val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())
        total_mae += mae(pred, y)
        total_rmse += rmse(pred, y)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
        "rmse": total_rmse / max(n_batches, 1),
    }


def train_model1(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
    ckpt_dir: str = "outputs/checkpoints",
) -> Tuple[Path, Dict[str, list]]:
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    best_val = float("inf")
    patience = 0
    best_file = ckpt_path / "model1_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_mae"].append(tr["mae"])
        history["val_mae"].append(va["mae"])

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train: loss={tr['loss']:.4f} mae={tr['mae']:.2f} rmse={tr['rmse']:.2f} | "
            f"val: loss={va['loss']:.4f} mae={va['mae']:.2f} rmse={va['rmse']:.2f}"
        )

        # checkpoint
        if va["loss"] < best_val:
            best_val = va["loss"]
            patience = 0
            torch.save(
                {"model_state": model.state_dict(), "best_val_loss": best_val},
                best_file.as_posix(),
            )
            print(f"  Saved best checkpoint: {best_file.as_posix()} (val_loss={best_val:.4f})")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

    # Save loss curves
    fig = plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Model-1 Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    p = save_figure(fig, "02_model1_loss_curve.png")
    print(f"  Figure saved: {p}")

    fig2 = plt.figure()
    plt.plot(history["train_mae"], label="train_mae")
    plt.plot(history["val_mae"], label="val_mae")
    plt.title("Model-1 MAE Curve")
    plt.xlabel("epoch")
    plt.ylabel("MAE (W)")
    plt.legend()
    p2 = save_figure(fig2, "02_model1_mae_curve.png")
    print(f"  Figure saved: {p2}")

    return best_file, history
