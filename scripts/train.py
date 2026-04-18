from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.query_dataset_npz import QueryDatePatchDataset
from models.query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig
from training.query_engine import fit_query

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. In Colab run with a PyTorch runtime.") from exc


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer, epochs: int):
    scheduler_name = str(config.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    warmup_epochs = max(0, int(config.get("warmup_epochs", 0)))
    base_lr = float(config["learning_rate"])
    min_lr = max(0.0, float(config.get("min_lr", 0.0)))
    min_lr_factor = min(min_lr / base_lr, 1.0) if base_lr > 0 else 0.0

    def lr_lambda(epoch_index: int) -> float:
        # LambdaLR uses zero-based epoch index after each scheduler.step().
        epoch_number = epoch_index + 1
        if warmup_epochs and epoch_number <= warmup_epochs:
            return max(epoch_number / warmup_epochs, min_lr_factor, 1e-8)
        decay_epochs = max(1, epochs - warmup_epochs)
        progress = min(max((epoch_number - warmup_epochs) / decay_epochs, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

    for group in optimizer.param_groups:
        group["lr"] = base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CNN+Transformer query-date model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-query-date", action="store_true", help="Ablation: remove query date from the model to test calendar leakage.")
    parser.add_argument("--no-time-date", action="store_true", help="Ablation: remove acquisition-date encoding from the temporal Transformer.")
    parser.add_argument("--shuffle-labels", action="store_true", help="Sanity check: shuffle train labels only; validation should collapse toward random.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    preserve_output_dir = bool(config.get("preserve_output_dir", False))

    model_config_data = dict(config.get("model", {}))
    if args.no_query_date:
        model_config_data["use_query_doy"] = False
        if not preserve_output_dir:
            config["output_dir"] = str(config.get("output_dir", "artifacts/models/cnn_transformer")) + "_no_date"
    if args.no_time_date:
        model_config_data["use_time_doy"] = False
        if not preserve_output_dir:
            config["output_dir"] = str(config.get("output_dir", "artifacts/models/cnn_transformer")) + "_no_time_date"
    if args.shuffle_labels and not preserve_output_dir:
        config["output_dir"] = str(config.get("output_dir", "artifacts/models/cnn_transformer")) + "_shuffled_labels"
    use_aux_features = bool(config.get("use_aux_features", False))

    train_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="train",
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        shuffle_labels_seed=int(config.get("seed", 42)) if args.shuffle_labels else None,
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=use_aux_features,
    )
    val_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="val",
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=use_aux_features,
    )
    if use_aux_features:
        model_config_data["aux_feature_dim"] = int(train_ds.aux_feature_dim)
    else:
        model_config_data["aux_feature_dim"] = 0
    model_config = QueryCNNTransformerConfig(**{key: value for key, value in model_config_data.items() if key in QueryCNNTransformerConfig.__annotations__})

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)), pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), pin_memory=pin_memory)

    model = QueryCNNTransformerClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(args.epochs or config.get("epochs", 10))
    scheduler = build_scheduler(config, optimizer, epochs)
    output_dir = resolve_path(config["output_dir"])

    payload = {
        "model_config": asdict(model_config),
        "train_config": config,
        "device": str(device),
        "task": "point_date_crop_stage_classification",
        "aux_feature_names": train_ds.aux_feature_names,
    }
    history = fit_query(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        stage_loss_weight=float(config.get("stage_loss_weight", 0.6)),
        output_dir=output_dir,
        scheduler=scheduler,
        amp=bool(config.get("amp", False)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        clip_grad_norm=float(config.get("clip_grad_norm", 1.0)),
        early_stopping_patience=config.get("early_stopping_patience"),
        save_best_only=bool(config.get("save_best_only", True)),
        checkpoint_payload=payload,
        checkpoint_metric=str(config.get("checkpoint_metric", "val_loss")),
        tie_breaker_metric=str(config.get("tie_breaker_metric", "val_loss")),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config_resolved.json").write_text(json.dumps({**payload, "history_file": str(output_dir / "history.json")}, indent=2))
    print(json.dumps({"model": str(output_dir / "model.pt"), "history": str(output_dir / "history.json"), "device": str(device), "train_queries": len(train_ds), "val_queries": len(val_ds)}, indent=2))


if __name__ == "__main__":
    main()
