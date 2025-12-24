from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

POWER_COL = ("power", "apparent")  # default; config ile de okuyacağız


def get_series(store: pd.HDFStore, key: str, power_col=POWER_COL) -> pd.Series | None:
    try:
        df = store.select(key)
    except Exception:
        df = store[key]

    # index datetime değilse dönüştürmeyi dene
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ["timestamp", "time", "datetime", "Date", "date"]:
            if cand in df.columns:
                df = df.copy()
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
                df = df.dropna(subset=[cand]).set_index(cand)
                break

    df = df.sort_index()

    # MultiIndex kolon
    if isinstance(df.columns, pd.MultiIndex):
        if power_col in df.columns:
            s = df[power_col]
        else:
            # ilk kolon fallback
            s = df[df.columns[0]]
    else:
        # tek seviye kolon
        if "power" in df.columns:
            s = df["power"]
        else:
            s = df[df.columns[0]]

    # sayısal ve temiz
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 100:
        return None
    return s


def main() -> None:
    h5_path = Path("data/raw/ukdale.h5")
    if not h5_path.exists():
        raise FileNotFoundError(h5_path.resolve())

    with pd.HDFStore(h5_path.as_posix(), mode="r") as store:
        keys = [k for k in store.keys() if k.startswith("/building1/elec/meter")]
        keys = sorted(keys, key=lambda x: int(x.split("meter")[-1]))
        print("Found meter keys:", len(keys))

        rows = []
        for k in keys:
            s = get_series(store, k)
            if s is None:
                continue

            # hızlı özet (fridge genelde: düşük-orta mean, döngüsel, belli on/off)
            mean = float(s.mean())
            std = float(s.std())
            p95 = float(np.percentile(s, 95))
            on_ratio = float((s > 10).mean())  # 10W üstünü "on" say
            rows.append((k, len(s), mean, std, p95, on_ratio))

        df = pd.DataFrame(rows, columns=["key", "n", "mean", "std", "p95", "on_ratio"])
        df = df.sort_values(["mean", "p95"], ascending=True)

        pd.set_option("display.max_rows", 200)
        print("\n=== Meter Summary (sorted by mean, p95) ===")
        print(df.to_string(index=False))

        print("\nİPUCU:")
        print("- Fridge genelde mean ~ 30-200W bandında olur (eve göre değişir).")
        print("- on_ratio çok düşükse (0.01 gibi) muhtemelen nadir çalışan bir cihaz.")
        print("- p95 çok yüksekse kettle/washer gibi yüksek güçlü olabilir.")


if __name__ == "__main__":
    main()
