from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import mae, rmse


@dataclass
class InferResult:
    pred: np.ndarray      # (N, L)
    gt: np.ndarray        # (N, L)
    t0: np.ndarray        # (N,)
    metrics: Dict[str, float]


@torch.no_grad()
def infer_windows(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    gts = []
    t0s = []

    # dataset'ten t0 almak için loader.dataset.t0 kullanacağız
    # bu yüzden batch'te idx yok; sırayla birikecek şekilde alıyoruz.
    # DataLoader shuffle=False olmalı.
    ds = loader.dataset
    if not hasattr(ds, "t0"):
        raise ValueError("Dataset'te t0 alanı bulunamadı. NilmWindowDataset'i kontrol et.")

    offset = 0
    for x, y in loader:
        bsz = x.shape[0]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        p = model(x)

        preds.append(p.detach().cpu().numpy()[:, 0, :])  # (B,L)
        gts.append(y.detach().cpu().numpy()[:, 0, :])    # (B,L)

        t0_batch = ds.t0[offset: offset + bsz]
        t0s.append(t0_batch)
        offset += bsz

    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    t0 = np.concatenate(t0s, axis=0)

    return pred, gt, t0


def compute_metrics_np(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_t = torch.tensor(pred)
    gt_t = torch.tensor(gt)
    return {
        "mae": mae(pred_t, gt_t),
        "rmse": rmse(pred_t, gt_t),
    }


def export_fridge_pred_csv(
    pred: np.ndarray,      # (N,L)
    t0: np.ndarray,        # (N,)
    window_size: int,
    sampling: str,
    out_csv: str,
) -> str:
    """
    Window başlangıç timestamp'lerinden yola çıkarak her window için L tane timestamp üretir.
    Basit yaklaşım: windows overlap olsa bile tüm noktaları yazıyoruz.
    (Sonraki adımda overlap birleştirmeyi "weighted merge" ile iyileştirebiliriz.)
    """
    # Her window için zaman indeksini üret
    start_times = pd.to_datetime(t0, unit="ns")
    delta = pd.to_timedelta(sampling)

    rows = []
    for i in range(pred.shape[0]):
        base = start_times[i]
        times = base + (np.arange(window_size) * delta)
        rows.append(pd.DataFrame({"timestamp": times, "predicted_power": pred[i]}))

    df_out = pd.concat(rows, ignore_index=True)
    df_out = df_out.sort_values("timestamp")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path.as_posix(), index=False)
    return out_path.as_posix()
