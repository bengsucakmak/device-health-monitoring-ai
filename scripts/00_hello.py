from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.utils.config import ensure_dir, load_yaml
from src.utils.seed import set_seed


def main() -> None:
    # 1) Config oku
    cfg = load_yaml("src/config/default.yaml")

    project_name = cfg["project"]["name"]
    seed = int(cfg["project"]["seed"])

    raw_h5_path = Path(cfg["paths"]["raw_h5_path"])
    log_dir = ensure_dir(cfg["paths"]["log_dir"])

    # 2) Seed ata
    set_seed(seed)

    # 3) Basit kontrol + log yaz
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exists_str = "YES" if raw_h5_path.exists() else "NO"

    log_text = (
        f"[{now}] HELLO PIPELINE\n"
        f"Project: {project_name}\n"
        f"Seed: {seed}\n"
        f"Raw dataset path: {raw_h5_path.as_posix()}\n"
        f"Dataset exists: {exists_str}\n"
        f"Next step: scripts/01_prepare_data.py (UK-DALE read + resample + window)\n"
    )

    out_file = log_dir / "hello.txt"
    out_file.write_text(log_text, encoding="utf-8")

    print(log_text)
    print(f" Wrote log: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
