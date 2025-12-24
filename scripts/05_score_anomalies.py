from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_yaml
from src.model2.sequence import build_train_test_sequences
from src.model2.lstm_autoencoder import LSTMAutoencoder
from src.model2.score import (
    build_score_dataframe,
    percentile_threshold,
    reconstruction_scores,
)
from src.model2.plots import (
    plot_anomaly_flags,
    plot_score_hist,
    plot_score_timeseries,
)


def main() -> None:
    cfg = load_yaml("src/config/default.yaml")
    m2 = cfg["model2"]

    csv_path = "data/processed/fridge_pred_dedup.csv"
    ckpt_path = "outputs/checkpoints/model2_ae_best.pt"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError("model2_ae_best.pt yok. Önce: python -m scripts.04_train_model2")

    # Train/Test sequences (scaler train’den fit)
    train_data, test_data = build_train_test_sequences(
        csv_path=csv_path,
        seq_len=int(m2["seq_len"]),
        stride=int(m2["stride"]),
        train_ratio=float(m2["train_ratio"]),
        normalize=str(m2["normalize"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model init
    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=int(m2["hidden_dim"]),
        latent_dim=int(m2["latent_dim"]),
        num_layers=int(m2["num_layers"]),
        dropout=float(m2["dropout"]),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {ckpt_path}")

    # Score on TEST (daha anlamlı; train normal öğrenmiş olacak)
    X = test_data.X
    t0 = test_data.t0

    if X.shape[0] == 0:
        raise ValueError("Test sequence yok. seq_len/stride/train_ratio ayarlarını kontrol et.")

    scores = reconstruction_scores(
        model=model,
        X=X,
        device=device,
        batch_size=128,
    )

    # Threshold: ister train-score’dan ister test-score’dan belirlenebilir.
    # Pratik başlangıç: TRAIN skor dağılımından threshold (normal baseline) daha mantıklı.
    train_scores = reconstruction_scores(model, train_data.X, device=device, batch_size=128)

    q = 99.0
    threshold = percentile_threshold(train_scores, q=q)
    print(f"Threshold = P{q} of TRAIN scores = {threshold:.6f}")

    import json

    threshold_info = {
        "method": f"P{q}_train_percentile",
        "q": q,
        "threshold": threshold,
    }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, ensure_ascii=False, indent=2)

    print("Saved: data/processed/threshold.json")


    df_scores = build_score_dataframe(
        t0=t0,
        scores=scores,
        seq_len=int(m2["seq_len"]),
        sampling=cfg["data"]["sampling"],
    )

    df_scores["is_anomaly"] = df_scores["anomaly_score"] > threshold

    out_path = Path("data/processed/anomaly_scores.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(out_path.as_posix(), index=False)
    print(f" Saved: {out_path.as_posix()} | rows={len(df_scores)}")

    # Plots
    p1 = plot_score_timeseries(df_scores, threshold)
    p2 = plot_score_hist(df_scores, threshold)
    p3 = plot_anomaly_flags(df_scores, threshold)
    print(" Saved figures:", p1, p2, p3)

    # Quick summary
    n_anom = int(df_scores["is_anomaly"].sum())
    print(f"Anomalies flagged: {n_anom}/{len(df_scores)} ({100*n_anom/len(df_scores):.2f}%)")

    print("\nNEXT: Faz 3 -> Pipeline + Streamlit dashboard (health score + alerts)")


if __name__ == "__main__":
    main()
