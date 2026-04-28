from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. In Colab run with a PyTorch runtime.") from exc

from data.query_dataset_npz import QueryDatePatchDataset
from models.model_factory import build_model, build_model_config, config_asdict, normalize_model_type
from scripts.train import build_dataloader_kwargs, build_loss_weight_tensors, build_scheduler, collect_git_metadata, resolve_path, seed_everything, select_device
from training.query_engine import run_query_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a selected query model on all labeled data for a fixed epoch count.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_full_data_c03_epoch75.json")
    parser.add_argument("--epochs", type=int, default=None, help="Override fixed epoch count. Use only for deliberate experiments.")
    return parser.parse_args()


def save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    epoch: int,
    history: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(payload)
    checkpoint.update(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "checkpoint_metric": "fixed_epoch_full_data",
            "best_metric_value": None,
            "best_epoch": epoch,
            "tie_breaker_metric": None,
            "best_tie_breaker_value": None,
            "best_val_loss": None,
            "history": history,
            "full_data_training": True,
            "selection_rule": str(payload.get("selection_rule", "Fixed epoch selected before full-data training.")),
        }
    )
    torch.save(checkpoint, output_dir / "model.pt")


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    model_type = normalize_model_type(config.get("model_type", "query_cnn_transformer"))

    use_aux_features = bool(config.get("use_aux_features", False))
    aux_feature_set = str(config.get("aux_feature_set", "summary"))
    use_spectral_indices = bool(config.get("use_spectral_indices", False))
    _idx_stats_raw = config.get("spectral_index_stats_json", None)
    spectral_index_stats_json = resolve_path(_idx_stats_raw) if _idx_stats_raw else None

    train_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=None,
        split=None,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", True)),
        use_aux_features=use_aux_features,
        aux_feature_set=aux_feature_set,
        random_time_shift_days=int(config.get("random_time_shift_days", 0)),
        random_query_shift_days=int(config.get("random_query_shift_days", 0)),
        random_query_shift_prob=float(config.get("random_query_shift_prob", 1.0)),
        query_doy_dropout_prob=float(config.get("query_doy_dropout_prob", 0.0)),
        time_doy_dropout_prob=float(config.get("time_doy_dropout_prob", 0.0)),
        use_spectral_indices=use_spectral_indices,
        spectral_index_stats_json=spectral_index_stats_json,
        use_relative_doy=bool(config.get("use_relative_doy", False)),
    )

    model_config_data = dict(config.get("model", {}))
    model_config_data["aux_feature_dim"] = int(train_ds.aux_feature_dim) if use_aux_features else 0
    model_config = build_model_config(model_type, model_config_data)
    crop_class_weights, stage_class_weights = build_loss_weight_tensors(train_ds, config, device)

    pin_memory = device.type == "cuda"
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        **build_dataloader_kwargs(
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            config=config,
        ),
    )

    model = build_model(model_type, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(args.epochs or config.get("epochs", 75))
    scheduler = build_scheduler(config, optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", False)) and device.type == "cuda")
    output_dir = resolve_path(config["output_dir"])
    git_metadata = collect_git_metadata()

    payload = {
        "model_type": model_type,
        "model_config": config_asdict(model_config),
        "train_config": config,
        "device": str(device),
        "task": "point_date_crop_stage_classification_full_data_fixed_epoch",
        "aux_feature_names": train_ds.aux_feature_names,
        "aux_feature_set": aux_feature_set if use_aux_features else None,
        "git": git_metadata,
        "selection_rule": str(config.get("selection_rule", "Fixed epoch selected before full-data training.")),
    }

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_query_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            stage_loss_weight=float(config.get("stage_loss_weight", 0.6)),
            crop_loss_weight=float(config.get("crop_loss_weight", 1.0)),
            amp=bool(config.get("amp", False)),
            scaler=scaler,
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            clip_grad_norm=float(config.get("clip_grad_norm", 1.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            stage_label_smoothing=config.get("stage_label_smoothing"),
            stage_ordinal_loss_weight=float(config.get("stage_ordinal_loss_weight", 0.0)),
            stage_sequence_loss_weight=float(config.get("stage_sequence_loss_weight", 0.0)),
            stage_max_forward_step=float(config.get("stage_max_forward_step", 1.75)),
            stage_postprocess=str(config.get("stage_postprocess", "none")),
            crop_class_weights=crop_class_weights,
            stage_class_weights=stage_class_weights,
            point_crop_consistency_loss_weight=float(config.get("point_crop_consistency_loss_weight", 0.0)),
        )
        row = {"epoch": epoch, "lr": lr_used, **{f"train_{key}": value for key, value in train_metrics.items()}}
        row["train_competition_score"] = 0.4 * row["train_crop_macro_f1"] + 0.6 * row["train_rice_stage_macro_f1"]
        row["train_competition_score_consistent"] = 0.4 * row["train_crop_macro_f1_consistent"] + 0.6 * row["train_rice_stage_macro_f1"]
        history.append(row)
        print(row)
        if scheduler is not None:
            scheduler.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config_resolved.json").write_text(json.dumps({**payload, "history_file": str(output_dir / "history.json")}, indent=2))
    save_checkpoint(output_dir=output_dir, model=model, epoch=epochs, history=history, payload=payload)
    summary = {
        "experiment": output_dir.name,
        "training_mode": "full_data_fixed_epoch",
        "epoch": epochs,
        "selection_rule": str(config.get("selection_rule", "Fixed epoch selected before full-data training.")),
        "train_queries": len(train_ds),
        "final_train_competition_score": history[-1].get("train_competition_score") if history else None,
        "final_train_loss": history[-1].get("train_loss") if history else None,
        "model_config": config_asdict(model_config),
        "git": git_metadata,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"model": str(output_dir / "model.pt"), "history": str(output_dir / "history.json"), "device": str(device), "train_queries": len(train_ds)}, indent=2))


if __name__ == "__main__":
    main()
