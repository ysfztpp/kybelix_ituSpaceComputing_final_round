from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def last_metrics(history_path: Path) -> dict:
    return json.loads(history_path.read_text())[-1]


def run_one(name: str, extra_args: list[str], epochs: int, python: str) -> dict:
    cmd = [python, "scripts/train.py", "--config", "configs/train_fast.json", "--epochs", str(epochs), *extra_args]
    print({"running": name, "cmd": " ".join(cmd)})
    subprocess.run(cmd, cwd=ROOT, check=True)
    suffix = ""
    if "--no-query-date" in extra_args:
        suffix += "_no_date"
    if "--no-time-date" in extra_args:
        suffix += "_no_time_date"
    if "--shuffle-labels" in extra_args:
        suffix += "_shuffled_labels"
    history = ROOT / f"artifacts/models/fast_cnn_transformer{suffix}/history.json"
    row = last_metrics(history)
    return {"name": name, "history": str(history), **row}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast sanity experiments for date shortcut and leakage checks.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()
    experiments = [
        ("normal_dates", []),
        ("no_query_date", ["--no-query-date"]),
        ("no_time_date", ["--no-query-date", "--no-time-date"]),
        ("shuffled_train_labels", ["--shuffle-labels"]),
    ]
    rows = [run_one(name, extra, args.epochs, args.python) for name, extra in experiments]
    out = ROOT / "artifacts/audit/fast_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
