from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import pandas as pd


@dataclass
class UKDALEKeys:
    aggregate_key: str
    appliance_key: str


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    for cand in ["timestamp", "time", "datetime", "Date", "date"]:
        if cand in df.columns:
            df = df.copy()
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            df = df.dropna(subset=[cand]).set_index(cand)
            return df.sort_index()

    raise ValueError("Datetime index oluşturulamadı. H5 içindeki kolonlara bakmalıyız.")


def _select_power_series(df: pd.DataFrame, power_column: List[str]) -> pd.Series:
    """
    power_column config'ten ["power","apparent"] gibi gelir.
    H5 içindeki kolon MultiIndex ise tuple olarak arar.
    """
    df = df.copy()
    df = _ensure_datetime_index(df)

    if isinstance(df.columns, pd.MultiIndex):
        key = tuple(power_column)
        if key in df.columns:
            s = df[key]
        else:
            # fallback: ilk kolon
            s = df[df.columns[0]]
    else:
        # tek seviye kolon
        if power_column and power_column[0] in df.columns:
            s = df[power_column[0]]
        else:
            s = df[df.columns[0]]

    s = pd.to_numeric(s, errors="coerce")
    return s


def load_ukdale_agg_and_appliance(
    h5_path: str | Path,
    keys: UKDALEKeys,
    power_column: List[str],
) -> pd.DataFrame:
    """
    Aggregate (meter1) ve appliance (fridge meterX) ayrı key'lerden okunur,
    zaman ekseninde hizalanır ve DataFrame döndürür: [aggregate, appliance]
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path.resolve()}")

    with pd.HDFStore(h5_path.as_posix(), mode="r") as store:
        if keys.aggregate_key not in store.keys():
            raise ValueError(f"aggregate_key bulunamadı: {keys.aggregate_key}")
        if keys.appliance_key not in store.keys():
            raise ValueError(f"appliance_key bulunamadı: {keys.appliance_key}")

        try:
            df_agg = store.select(keys.aggregate_key)
        except Exception:
            df_agg = store[keys.aggregate_key]

        try:
            df_app = store.select(keys.appliance_key)
        except Exception:
            df_app = store[keys.appliance_key]

    s_agg = _select_power_series(df_agg, power_column).rename("aggregate")
    s_app = _select_power_series(df_app, power_column).rename("appliance")

    # inner join ile hizala
    df = pd.concat([s_agg, s_app], axis=1).dropna()
    df = df.sort_index()
    return df
