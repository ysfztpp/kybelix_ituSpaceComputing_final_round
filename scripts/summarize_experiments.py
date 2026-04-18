from __future__ import annotations

import argparse
import csv
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
    """Keep old and new histories comparable.

    Older checkpoints did not store `*_competition_score`. The competition-style
    validation score we use for model selection is:
    0.4 * crop macro F1 + 0.6 * rice-stage macro F1.
    """

    row = dict(row)
    for prefix in ("train", "val"):
        score_key = f"{prefix}_competition_score"
        if score_key not in row:
            row[score_key] = 0.4 * _safe_float(row.get(f"{prefix}_crop_macro_f1")) + 0.6 * _safe_float(row.get(f"{prefix}_rice_stage_macro_f1"))
    return row


def metric_is_maximized(metric: str) -> bool:
    return "loss" not in metric.lower()


def best_row(history: list[dict[str, Any]], metric: str, tie_breaker: str = "val_loss") -> dict[str, Any]:
    rows = [add_derived_scores(row) for row in history]
    if not rows:
        return {}
    maximize = metric_is_maximized(metric)
    best = rows[0]
    for row in rows[1:]:
        current = _safe_float(row.get(metric), float("-inf") if maximize else float("inf"))
        previous = _safe_float(best.get(metric), float("-inf") if maximize else float("inf"))
        current_tie = _safe_float(row.get(tie_breaker), float("inf"))
        previous_tie = _safe_float(best.get(tie_breaker), float("inf"))
        improved = current > previous if maximize else current < previous
        tied_but_better = abs(current - previous) <= 1e-12 and current_tie < previous_tie
        if improved or tied_but_better:
            best = row
    return best


def read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text())


def config_summary(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config_resolved.json"
    if not config_path.exists():
        return {}
    try:
        payload = read_json(config_path)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    model_config = payload.get("model_config", {})
    train_config = payload.get("train_config", {})
    git = payload.get("git", {})
    return {
        "use_query_doy": model_config.get("use_query_doy"),
        "use_time_doy": model_config.get("use_time_doy"),
        "in_channels": model_config.get("in_channels"),
        "aux_feature_dim": model_config.get("aux_feature_dim", 0),
        "dropout": model_config.get("dropout"),
        "label_smoothing": train_config.get("label_smoothing", 0.0),
        "weight_decay": train_config.get("weight_decay"),
        "output_dir": train_config.get("output_dir", str(model_dir)),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
    }


def checkpoint_summary(model_dir: Path) -> dict[str, Any]:
    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        return {"checkpoint": None}
    summary: dict[str, Any] = {"checkpoint": str(checkpoint_path), "checkpoint_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2)}
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - summary should not fail if torch is unavailable.
        summary["checkpoint_error"] = str(exc)
        return summary
    summary.update(
        {
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_metric": payload.get("checkpoint_metric", "legacy"),
            "checkpoint_best_metric": payload.get("best_metric_value"),
            "checkpoint_best_epoch": payload.get("best_epoch"),
            "checkpoint_best_val_loss": payload.get("best_val_loss"),
        }
    )
    return summary


def summarize_history(history_path: Path, metric: str, tie_breaker: str) -> dict[str, Any]:
    history = read_json(history_path)
    if not isinstance(history, list):
        raise ValueError(f"{history_path} must contain a list of epoch rows")
    best = best_row(history, metric, tie_breaker)
    final = add_derived_scores(history[-1]) if history else {}
    model_dir = history_path.parent
    row = {
        "experiment": model_dir.name,
        "history": str(history_path),
        "epochs_ran": len(history),
        "best_epoch": best.get("epoch"),
        "best_metric": metric,
        "best_score": best.get(metric),
        "best_val_competition_score": best.get("val_competition_score"),
        "best_val_loss": best.get("val_loss"),
        "best_val_crop_macro_f1": best.get("val_crop_macro_f1"),
        "best_val_rice_stage_macro_f1": best.get("val_rice_stage_macro_f1"),
        "best_val_crop_accuracy": best.get("val_crop_accuracy"),
        "best_val_rice_stage_accuracy": best.get("val_rice_stage_accuracy"),
        "best_train_competition_score": best.get("train_competition_score"),
        "train_val_score_gap": _safe_float(best.get("train_competition_score")) - _safe_float(best.get("val_competition_score")),
        "final_epoch": final.get("epoch"),
        "final_val_competition_score": final.get("val_competition_score"),
        "final_val_loss": final.get("val_loss"),
    }
    row.update(config_summary(model_dir))
    row.update(checkpoint_summary(model_dir))
    return row


def discover_histories(root: Path) -> list[Path]:
    if root.is_file() and root.name == "history.json":
        return [root]
    return sorted(root.glob("**/history.json"))


def print_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "experiment",
        "epochs_ran",
        "best_epoch",
        "best_val_competition_score",
        "best_val_crop_macro_f1",
        "best_val_rice_stage_macro_f1",
        "best_val_loss",
        "train_val_score_gap",
        "use_query_doy",
        "use_time_doy",
        "in_channels",
        "aux_feature_dim",
        "git_commit",
        "git_dirty",
        "checkpoint_epoch",
    ]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append("" if value is None else str(value))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return
    columns = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean comparison table from training histories.")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "models", help="Folder containing model output directories.")
    parser.add_argument("--metric", default="val_competition_score", help="Metric used to select the best epoch.")
    parser.add_argument("--tie-breaker", default="val_loss", help="Lower is better when the main metric is tied.")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the table/report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    rows: list[dict[str, Any]] = []
    for history_path in discover_histories(root):
        try:
            rows.append(summarize_history(history_path, args.metric, args.tie_breaker))
        except (json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"skipped": str(history_path), "reason": str(exc)}))
    rows.sort(key=lambda row: _safe_float(row.get(args.metric) or row.get("best_val_competition_score"), float("-inf")), reverse=True)

    if args.format == "json":
        text = json.dumps({"metric": args.metric, "tie_breaker": args.tie_breaker, "root": str(root), "experiments": rows}, indent=2)
    elif args.format == "csv":
        if args.output is None:
            raise SystemExit("--output is required when --format csv")
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out_path)
        text = json.dumps({"csv": str(out_path), "rows": len(rows)}, indent=2)
    else:
        text = print_markdown(rows)

    print(text)
    if args.output is not None and args.format != "csv":
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
