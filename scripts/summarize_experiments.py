from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def best_row(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not history:
        return {}
    maximize = "loss" not in metric.lower()
    return max(history, key=lambda row: float(row.get(metric, float("-inf")))) if maximize else min(history, key=lambda row: float(row.get(metric, float("inf"))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank completed experiment histories.")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "models" / "experiments")
    parser.add_argument("--metric", default="val_competition_score")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    rows: list[dict[str, Any]] = []
    for history_path in sorted(root.glob("**/history.json")):
        try:
            history = json.loads(history_path.read_text())
        except json.JSONDecodeError:
            continue
        row = best_row(history, args.metric)
        if not row:
            continue
        rows.append(
            {
                "experiment": history_path.parent.name,
                "history": str(history_path),
                "best_epoch": row.get("epoch"),
                args.metric: row.get(args.metric),
                "val_loss": row.get("val_loss"),
                "val_crop_macro_f1": row.get("val_crop_macro_f1"),
                "val_rice_stage_macro_f1": row.get("val_rice_stage_macro_f1"),
                "val_crop_accuracy": row.get("val_crop_accuracy"),
                "val_rice_stage_accuracy": row.get("val_rice_stage_accuracy"),
            }
        )
    rows.sort(key=lambda row: float(row.get(args.metric) or float("-inf")), reverse=True)
    output = {"metric": args.metric, "root": str(root), "experiments": rows}
    print(json.dumps(output, indent=2))
    if args.output is not None:
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
