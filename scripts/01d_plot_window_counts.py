from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting import save_figure


def load_npz(p: str) -> int:
    data = np.load(p)
    return int(data["X"].shape[0])


def main() -> None:
    tr = load_npz("data/processed/train_windows.npz")
    va = load_npz("data/processed/val_windows.npz")
    te = load_npz("data/processed/test_windows.npz")

    fig = plt.figure()
    plt.bar(["train", "val", "test"], [tr, va, te])
    plt.title("Window Counts")
    plt.ylabel("num_windows")

    saved = save_figure(fig, "01_window_counts.png")
    print(f"Figure saved: {saved}")


if __name__ == "__main__":
    main()
