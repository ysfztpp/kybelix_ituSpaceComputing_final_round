from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig


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

    checkpoint = resolve_path(config["checkpoint"])
    if not checkpoint.exists():
        fail(f"missing checkpoint: {checkpoint}. Put your trained file at checkpoints/model.pt")
    checkpoint_mb = checkpoint.stat().st_size / (1024 * 1024)
    if checkpoint_mb > args.max_checkpoint_mb:
        fail(f"checkpoint is {checkpoint_mb:.2f} MB; GitHub normal file limit is about 100 MB")

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

    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is required to validate the checkpoint: {exc}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ["model_config", "model_state_dict"]:
        if key not in payload:
            fail(f"checkpoint missing key: {key}")

    model_config = QueryCNNTransformerConfig(**payload["model_config"])
    if model_config.in_channels not in {12, 24}:
        fail(f"model in_channels must be 12 or 24, got {model_config.in_channels}")
    if model_config.patch_size != 15:
        fail(f"model patch_size must be 15, got {model_config.patch_size}")
    if model_config.num_crop_classes != 3:
        fail(f"model num_crop_classes must be 3, got {model_config.num_crop_classes}")
    if model_config.num_phenophase_classes != 7:
        fail(f"model num_phenophase_classes must be 7, got {model_config.num_phenophase_classes}")

    model = QueryCNNTransformerClassifier(model_config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)

    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "checkpoint_mb": round(checkpoint_mb, 2),
                "checkpoint_epoch": payload.get("epoch"),
                "checkpoint_metric": payload.get("checkpoint_metric", "val_loss_legacy"),
                "best_metric_value": payload.get("best_metric_value"),
                "best_val_loss": payload.get("best_val_loss"),
                "model_config": payload["model_config"],
                "output_json": str(output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
