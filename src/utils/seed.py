from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across random, numpy, and torch (if installed).
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism flags (slower but stable)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
