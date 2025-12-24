from __future__ import annotations

from pathlib import Path

from src.utils.config import ensure_dir, load_yaml
from src.utils.seed import set_seed
from src.data.ukdale_reader import UKDALEKeys, load_ukdale_agg_and_appliance
from src.data.preprocess import PreprocessConfig, make_windows, preprocess_df, time_split


def save_npz(out_path: Path, X, y, t0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(out_path.as_posix(), X=X, y=y, t0=t0)


def main() -> None:
    cfg = load_yaml("src/config/default.yaml")
    set_seed(int(cfg["project"]["seed"]))

    raw_h5 = cfg["paths"]["raw_h5_path"]
    building = int(cfg["data"]["building"])

    #  H5 key’leri
    keys = UKDALEKeys(
        aggregate_key=cfg["data"]["aggregate_key"],
        appliance_key=cfg["data"]["appliance_key"],
    )

    #  MultiIndex power column (["power","apparent"])
    power_column = list(cfg["data"]["power_column"])

    sampling = cfg["data"]["sampling"]
    pp_cfg = PreprocessConfig(
        sampling=sampling,
        drop_negative=bool(cfg["preprocess"]["drop_negative"]),
        fillna_method=str(cfg["preprocess"]["fillna_method"]),
        clip_max_watt=float(cfg["preprocess"]["clip_max_watt"]),
    )

    window_size = int(cfg["windowing"]["window_size"])
    stride = int(cfg["windowing"]["stride"])
    split = cfg["windowing"]["split"]

    interim_dir = ensure_dir("data/interim")
    processed_dir = ensure_dir("data/processed")

    print("1) Loading UK-DALE from H5 (aggregate_key + appliance_key)...")
    df = load_ukdale_agg_and_appliance(
        h5_path=raw_h5,
        keys=keys,
        power_column=power_column,
    )
    print("   Raw aligned shape:", df.shape, "| range:", df.index.min(), "->", df.index.max())

    print("2) Preprocess (resample/clean/fill/clip)...")
    df_pp = preprocess_df(df, pp_cfg)
    print("   Preprocessed shape:", df_pp.shape)

    interim_path = Path(interim_dir) / f"building{building}_{sampling}_agg_fridge.parquet"
    df_pp.to_parquet(interim_path.as_posix())
    print(f"Saved interim: {interim_path.as_posix()}")

    print("3) Time split (train/val/test)...")
    df_train, df_val, df_test = time_split(
        df_pp, float(split["train"]), float(split["val"]), float(split["test"])
    )
    print("   train:", df_train.shape, "val:", df_val.shape, "test:", df_test.shape)

    print("4) Windowing...")
    Xtr, ytr, t0tr = make_windows(df_train, window_size, stride)
    Xva, yva, t0va = make_windows(df_val, window_size, stride)
    Xte, yte, t0te = make_windows(df_test, window_size, stride)

    print("   windows train/val/test:", Xtr.shape, Xva.shape, Xte.shape)

    print("5) Saving npz...")
    save_npz(Path(processed_dir) / "train_windows.npz", Xtr, ytr, t0tr)
    save_npz(Path(processed_dir) / "val_windows.npz", Xva, yva, t0va)
    save_npz(Path(processed_dir) / "test_windows.npz", Xte, yte, t0te)

    print("Saved: data/processed/train_windows.npz, val_windows.npz, test_windows.npz")
    print("\nNEXT: scripts/02_train_model1.py (TCN/1D-CNN baseline)")


if __name__ == "__main__":
    main()
