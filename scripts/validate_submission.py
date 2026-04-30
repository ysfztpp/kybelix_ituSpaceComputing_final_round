from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_factory import build_model, build_model_config, normalize_model_type


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fail(message: str) -> None:
    raise SystemExit(f"[validate_submission] ERROR: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing config: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def inspect_checkpoint(path: Path, *, max_checkpoint_mb: float) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing checkpoint: {path}")
    checkpoint_mb = path.stat().st_size / (1024 * 1024)
    if checkpoint_mb > max_checkpoint_mb:
        fail(f"checkpoint is {checkpoint_mb:.2f} MB; GitHub normal file limit is about 100 MB")

    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is required to validate the checkpoint: {exc}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["model_config", "model_state_dict"]:
        if key not in payload:
            fail(f"checkpoint missing key: {key}")

    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    model_config = build_model_config(model_type, payload["model_config"])
    if model_config.in_channels not in {12, 24}:
        fail(f"model in_channels must be 12 or 24, got {model_config.in_channels}")
    if model_config.patch_size != 15:
        fail(f"model patch_size must be 15, got {model_config.patch_size}")
    if model_config.num_crop_classes != 3:
        fail(f"model num_crop_classes must be 3, got {model_config.num_crop_classes}")
    if model_config.num_phenophase_classes != 7:
        fail(f"model num_phenophase_classes must be 7, got {model_config.num_phenophase_classes}")

    model = build_model(model_type, model_config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)

    return {
        "checkpoint": str(path),
        "checkpoint_mb": round(checkpoint_mb, 2),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metric": payload.get("checkpoint_metric", "val_loss_legacy"),
        "best_metric_value": payload.get("best_metric_value"),
        "best_val_loss": payload.get("best_val_loss"),
        "model_type": model_type,
        "model_config": payload["model_config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Track 1 submission files before pushing.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "submission.json")
    parser.add_argument("--max-checkpoint-mb", type=float, default=95.0)
    args = parser.parse_args()

    config = load_json(resolve_path(args.config))
    required_config_keys = ["input_root", "output_json", "checkpoint", "normalization_json", "preprocessing"]
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        fail(f"submission config missing keys: {missing_keys}")

    output_json = Path(config["output_json"])
    if not output_json.is_absolute() or output_json.name != "result.json":
        fail(f"Track 1 output must be an absolute path ending with result.json, got {output_json}")

    normalization_json = resolve_path(config["normalization_json"])
    if not normalization_json.exists():
        fail(f"missing normalization file: {normalization_json}")
    normalization = load_json(normalization_json)
    if "bands" not in normalization or "per_band" not in normalization:
        fail("normalization file must contain bands and per_band stats")
    if len(normalization["bands"]) != 12:
        fail(f"normalization bands must contain 12 values, got {len(normalization['bands'])}")
    for band in normalization["bands"]:
        band_stats = normalization["per_band"].get(band, {})
        for key in ["mean", "std"]:
            if key not in band_stats:
                fail(f"normalization stats for {band} missing key: {key}")

    patch_cfg = config["preprocessing"]
    bands = patch_cfg.get("bands", [])
    if len(bands) != 12:
        fail(f"submission preprocessing must use 12 bands, got {len(bands)}")
    if int(patch_cfg.get("patch_size", 0)) != 15:
        fail(f"submission patch_size must be 15, got {patch_cfg.get('patch_size')}")

    checkpoint_paths: dict[str, Path] = {
        "checkpoint": resolve_path(config["checkpoint"]),
    }
    if "crop_checkpoint" in config:
        checkpoint_paths["crop_checkpoint"] = resolve_path(config["crop_checkpoint"])
    if "stage_checkpoint" in config:
        checkpoint_paths["stage_checkpoint"] = resolve_path(config["stage_checkpoint"])
    for index, ensemble_value in enumerate(config.get("ensemble_checkpoints", [])):
        checkpoint_paths[f"ensemble_checkpoints[{index}]"] = resolve_path(ensemble_value)

    inspected: dict[str, dict[str, Any]] = {}
    seen: dict[Path, dict[str, Any]] = {}
    for label, path in checkpoint_paths.items():
        resolved = path.resolve()
        if resolved not in seen:
            seen[resolved] = inspect_checkpoint(path, max_checkpoint_mb=args.max_checkpoint_mb)
        inspected[label] = seen[resolved]

    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": inspected["checkpoint"]["checkpoint"],
                "checkpoint_mb": inspected["checkpoint"]["checkpoint_mb"],
                "checkpoint_epoch": inspected["checkpoint"]["checkpoint_epoch"],
                "checkpoint_metric": inspected["checkpoint"]["checkpoint_metric"],
                "best_metric_value": inspected["checkpoint"]["best_metric_value"],
                "best_val_loss": inspected["checkpoint"]["best_val_loss"],
                "model_type": inspected["checkpoint"]["model_type"],
                "model_config": inspected["checkpoint"]["model_config"],
                "checked_checkpoints": inspected,
                "output_json": str(output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
