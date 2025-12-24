from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class WindowData:
    X: np.ndarray  # (N, L)
    y: np.ndarray  # (N, L)
    t0: np.ndarray  # (N,)


class NilmWindowDataset(Dataset):
    """
    NILM window dataset:
      X: aggregate window
      y: appliance window
    Shapes are expected: (N, L)
    Returned tensors are shaped for 1D models: (C=1, L)
    """

    def __init__(self, npz_path: str, normalize: bool = True):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.t0 = data["t0"].astype(np.int64)

        if self.X.ndim != 2 or self.y.ndim != 2:
            raise ValueError("X ve y shape'i (N, L) olmalı. NPZ üretimini kontrol et.")

        self.normalize = normalize
        if normalize:
            # Basit pencere bazlı normalize: aggregate'i ölçekle (modelin öğrenmesini kolaylaştırır)
            # y'yi normalize ETMİYORUZ (çıkış watt ölçeğinde kalsın)
            eps = 1e-6
            self.x_mean = self.X.mean(axis=1, keepdims=True)
            self.x_std = self.X.std(axis=1, keepdims=True) + eps
        else:
            self.x_mean = None
            self.x_std = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (L,)
        y = self.y[idx]  # (L,)

        if self.normalize:
            x = (x - self.x_mean[idx]) / self.x_std[idx]

        # 1D conv için (C, L)
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, L)
        y_t = torch.from_numpy(y).unsqueeze(0)  # (1, L)
        return x_t, y_t


def make_loader(npz_path: str, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    ds = NilmWindowDataset(npz_path=npz_path, normalize=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
