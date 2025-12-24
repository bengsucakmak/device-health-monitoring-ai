from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    h5_path = Path("data/raw/ukdale.h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing: {h5_path.resolve()}")

    with pd.HDFStore(h5_path.as_posix(), mode="r") as store:
        keys = store.keys()
        print("\n=== HDF5 KEYS ===")
        for k in keys:
            print(" -", k)

        if not keys:
            print("No keys found in H5.")
            return

        # İlk 1-2 key’i örnek olarak okumayı dene
        print("\n=== SAMPLE PREVIEW ===")
        for k in keys[:2]:
            try:
                df = store.select(k, start=0, stop=5)
            except Exception:
                df = store[k].head(5)
            print(f"\nKey: {k}")
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print(df.head())


if __name__ == "__main__":
    main()
