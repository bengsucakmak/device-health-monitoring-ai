from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure


def main() -> None:
    path = Path("data/interim/building1_1min_agg_fridge.parquet")
    if not path.exists():
        raise FileNotFoundError("Önce 01_prepare_data çalıştır: data/interim parquet yok.")

    df = pd.read_parquet(path.as_posix()).sort_index()

    # İlk dolu gün (istersen daha sonra parametre yaparız)
    day = df.index[0].normalize()
    df_day = df.loc[day : day + pd.Timedelta(days=1)]

    fig = plt.figure()
    plt.plot(df_day.index, df_day["aggregate"], label="aggregate")
    plt.plot(df_day.index, df_day["appliance"], label="appliance (candidate fridge)")
    plt.title(f"Aggregate vs Appliance - One Day ({day.date()})")
    plt.xlabel("time")
    plt.ylabel("Watt")
    plt.legend()

    saved = save_figure(fig, f"01_one_day_agg_vs_app_{day.date()}.png")
    print(f"  Figure saved: {saved}")

    # İstersen ekranda da göster:
    # plt.show()


if __name__ == "__main__":
    main()
