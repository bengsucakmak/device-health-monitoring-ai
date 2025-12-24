from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.config import ensure_dir


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """
    Save matplotlib Figure into outputs/figures/ and return saved path.
    """
    out_dir = ensure_dir("outputs/figures")
    out_path = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches="tight")
    return out_path.as_posix()
