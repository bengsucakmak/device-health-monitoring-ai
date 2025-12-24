from __future__ import annotations

import numpy as np
import pandas as pd


def int64ns_to_datetime(t0: np.ndarray) -> pd.DatetimeIndex:
    """
    t0: int64 nanoseconds timestamps (numpy array)
    """
    return pd.to_datetime(t0, unit="ns", utc=False)
