from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    """
    Run a command in project root. Fail fast on error.
    """
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cihaz Sağlığı Asistanı: end-to-end pipeline runner"
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from step N (1..7). Default: 1",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=7,
        help="Stop at step N (1..7). Default: 7",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not launch Streamlit dashboard at the end",
    )
    args = parser.parse_args()

    steps = [
        (1, ["python", "-m", "scripts.01_prepare_data"]),
        (2, ["python", "-m", "scripts.02_train_model1"]),
        (3, ["python", "-m", "scripts.03_infer_model1"]),
        (4, ["python", "-m", "scripts.03b_deduplicate_fridge_csv"]),
        (5, ["python", "-m", "scripts.04_train_model2"]),
        (6, ["python", "-m", "scripts.05_score_anomalies"]),
        (7, ["streamlit", "run", "scripts/06_dashboard.py"]),
    ]

    from_step = max(1, min(7, args.from_step))
    to_step = max(1, min(7, args.to_step))

    for step_id, cmd in steps:
        if step_id < from_step or step_id > to_step:
            continue
        if step_id == 7 and args.no_dashboard:
            print("\nSkipping dashboard (--no-dashboard).")
            continue
        run(cmd)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
