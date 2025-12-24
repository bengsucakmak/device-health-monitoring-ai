from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    in_csv = Path("data/processed/fridge_pred.csv")
    out_csv = Path("data/processed/fridge_pred_dedup.csv")

    if not in_csv.exists():
        raise FileNotFoundError("fridge_pred.csv bulunamadı. Önce 03_infer_model1 çalıştır.")

    df = pd.read_csv(in_csv, parse_dates=["timestamp"])

    print("Before dedup:", len(df), "rows")

    #  ASIL TEKİLLEŞTİRME
    df_dedup = (
        df
        .groupby("timestamp", as_index=False)
        .agg(predicted_power=("predicted_power", "mean"))
        .sort_values("timestamp")
    )

    print("After dedup:", len(df_dedup), "rows")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(out_csv, index=False)

    print(f" Saved deduplicated CSV: {out_csv}")


if __name__ == "__main__":
    main()
