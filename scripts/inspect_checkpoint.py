from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_derived_scores(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for prefix in ("train", "val"):
        key = f"{prefix}_competition_score"
        if key not in row:
            row[key] = 0.4 * _safe_float(row.get(f"{prefix}_crop_macro_f1")) + 0.6 * _safe_float(row.get(f"{prefix}_rice_stage_macro_f1"))
    return row


def best_row(history: list[dict[str, Any]], metric: str, tie_breaker: str) -> dict[str, Any]:
    rows = [add_derived_scores(row) for row in history]
    if not rows:
        return {}
    maximize = "loss" not in metric.lower()
    best = rows[0]
    for row in rows[1:]:
        current = _safe_float(row.get(metric), float("-inf") if maximize else float("inf"))
        previous = _safe_float(best.get(metric), float("-inf") if maximize else float("inf"))
        current_tie = _safe_float(row.get(tie_breaker), float("inf"))
        previous_tie = _safe_float(best.get(tie_breaker), float("inf"))
        if (current > previous if maximize else current < previous) or (abs(current - previous) <= 1e-12 and current_tie < previous_tie):
            best = row
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one saved model checkpoint and report its best validation metrics.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=ROOT / "checkpoints" / "model.pt")
    parser.add_argument("--metric", default="val_competition_score")
    parser.add_argument("--tie-breaker", default="val_loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required to inspect checkpoints: {exc}") from exc

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    history = payload.get("history") or []
    best = best_row(history, args.metric, args.tie_breaker)
    final = add_derived_scores(history[-1]) if history else {}
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
        "saved_epoch": payload.get("epoch"),
        "stored_checkpoint_metric": payload.get("checkpoint_metric", "legacy_or_missing"),
        "stored_best_metric_value": payload.get("best_metric_value"),
        "stored_best_epoch": payload.get("best_epoch"),
        "stored_best_val_loss": payload.get("best_val_loss"),
        "model_config": payload.get("model_config"),
        "train_config": payload.get("train_config"),
        "history_rows": len(history),
        "selection_metric": args.metric,
        "tie_breaker": args.tie_breaker,
        "computed_best_epoch": best.get("epoch"),
        "computed_best_val_competition_score": best.get("val_competition_score"),
        "computed_best_val_loss": best.get("val_loss"),
        "computed_best_val_crop_macro_f1": best.get("val_crop_macro_f1"),
        "computed_best_val_rice_stage_macro_f1": best.get("val_rice_stage_macro_f1"),
        "computed_best_val_crop_accuracy": best.get("val_crop_accuracy"),
        "computed_best_val_rice_stage_accuracy": best.get("val_rice_stage_accuracy"),
        "final_epoch": final.get("epoch"),
        "final_val_competition_score": final.get("val_competition_score"),
        "final_val_loss": final.get("val_loss"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
