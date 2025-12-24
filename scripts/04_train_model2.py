from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.model2.sequence import build_train_test_sequences
from src.model2.dataset import SequenceDataset
from src.model2.lstm_autoencoder import LSTMAutoencoder
from src.model2.train import TrainAEConfig, train_autoencoder


def main() -> None:
    cfg = load_yaml("src/config/default.yaml")
    set_seed(int(cfg["project"]["seed"]))

    m2 = cfg["model2"]
    csv_path = "data/processed/fridge_pred_dedup.csv"

    train_data, test_data = build_train_test_sequences(
        csv_path=csv_path,
        seq_len=int(m2["seq_len"]),
        stride=int(m2["stride"]),
        train_ratio=float(m2["train_ratio"]),
        normalize=str(m2["normalize"]),
    )

    # Train/Val split within train sequences
    X = train_data.X
    n = X.shape[0]
    n_tr = int(n * 0.9)
    X_tr = X[:n_tr]
    X_va = X[n_tr:]

    train_loader = DataLoader(SequenceDataset(X_tr), batch_size=int(m2["batch_size"]), shuffle=True, pin_memory=True)
    val_loader = DataLoader(SequenceDataset(X_va), batch_size=int(m2["batch_size"]), shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Train seq:", X_tr.shape, "Val seq:", X_va.shape, "Test seq:", test_data.X.shape)

    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=int(m2["hidden_dim"]),
        latent_dim=int(m2["latent_dim"]),
        num_layers=int(m2["num_layers"]),
        dropout=float(m2["dropout"]),
    ).to(device)

    tcfg = TrainAEConfig(
        epochs=int(m2["epochs"]),
        lr=float(m2["lr"]),
        weight_decay=float(m2["weight_decay"]),
        early_stop_patience=int(m2["early_stop_patience"]),
    )

    best = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=tcfg,
        X_val_for_plot=X_va,
        ckpt_dir="outputs/checkpoints",
    )

    print("\nDONE.")
    print("Best checkpoint:", best.as_posix())
    print("Next: Faz 2 / Adım 2 -> anomaly score + threshold + alarms")


if __name__ == "__main__":
    main()
