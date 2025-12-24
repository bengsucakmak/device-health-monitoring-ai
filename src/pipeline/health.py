from __future__ import annotations

import numpy as np


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def health_score(score: np.ndarray, threshold: float) -> np.ndarray:
    ratio = 1.0 - (score / (threshold + 1e-9))
    return 100.0 * clamp01(ratio)


def health_band(hs: float) -> tuple[str, str]:
    """
    Returns: (label, emoji)
    """
    if hs >= 80:
        return ("Healthy", "🟢")
    if hs >= 50:
        return ("Watch", "🟠")
    return ("Alert", "🔴")
