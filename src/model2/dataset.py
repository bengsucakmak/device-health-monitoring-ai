from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    X shape: (N, L, 1)
    returns: (L, 1) tensor
    """
    def __init__(self, X: np.ndarray):
        if X.ndim != 3:
            raise ValueError("X shape (N, L, 1) olmalı.")
        self.X = X.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (L,1)
        return x
