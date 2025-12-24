from __future__ import annotations

import torch

from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.data.dataset import make_loader
from src.model1.tcn import TCNDisaggregator
from src.model1.train import TrainConfig, train_model1


def main() -> None:
    cfg = load_yaml("src/config/default.yaml")
    set_seed(int(cfg["project"]["seed"]))

    # Dataloader
    batch_size = 32
    if "model1" in cfg and "batch_size" in cfg["model1"]:
        batch_size = int(cfg["model1"]["batch_size"])

    train_loader = make_loader("data/processed/train_windows.npz", batch_size=batch_size, shuffle=True)
    val_loader = make_loader("data/processed/val_windows.npz", batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model
    model = TCNDisaggregator(
        hidden=64,
        num_blocks=6,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    # Train config
    tcfg = TrainConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        early_stop_patience=5,
    )

    best_ckpt, _ = train_model1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=tcfg,
        ckpt_dir="outputs/checkpoints",
    )

    print("\nDONE.")
    print("Best checkpoint:", best_ckpt.as_posix())
    print("Next: scripts/03_infer_model1.py -> fridge_pred.csv")


if __name__ == "__main__":
    main()
