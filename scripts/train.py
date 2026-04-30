from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.query_dataset_npz import QueryDatePatchDataset
from models.model_factory import build_model, build_model_config, config_asdict, normalize_model_type
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


def build_dataloader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
        prefetch_factor = config.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_git_metadata() -> dict[str, Any]:
    """Record the exact code state used for a training run."""

    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "--short", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = run_git("status", "--short")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status or "",
    }


def build_class_weights(
    labels: list[int],
    num_classes: int,
    mode: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    normalized_mode = str(mode or "none").strip().lower()
    if normalized_mode in {"none", "", "false"}:
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_classes)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    if normalized_mode == "inverse_freq":
        weights = 1.0 / counts
    elif normalized_mode == "sqrt_inverse_freq":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"unsupported class-weight mode: {mode}")
    weights = weights * (float(num_classes) / float(weights.sum()))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss_weight_tensors(
    train_ds: QueryDatePatchDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    crop_labels = [int(row[3]) for row in train_ds.rows]
    stage_labels = [int(row[1]) for row in train_ds.rows if float(row[4]) > 0.0]
    crop_weights = build_class_weights(
        crop_labels,
        num_classes=3,
        mode=config.get("crop_class_weight_mode", "none"),
        device=device,
    )
    stage_weights = build_class_weights(
        stage_labels,
        num_classes=7,
        mode=config.get("stage_class_weight_mode", "none"),
        device=device,
    )
    return crop_weights, stage_weights


def _with_derived_scores(row: dict[str, Any]) -> dict[str, Any]:
    """Make every history row contain the same score fields."""

    row = dict(row)
    for prefix in ("train", "val"):
        key = f"{prefix}_competition_score"
        if key not in row:
            row[key] = 0.4 * float(row.get(f"{prefix}_crop_macro_f1", 0.0)) + 0.6 * float(row.get(f"{prefix}_rice_stage_macro_f1", 0.0))
    return row


def _best_history_row(history: list[dict[str, Any]], metric: str, tie_breaker: str) -> dict[str, Any]:
    rows = [_with_derived_scores(row) for row in history]
    if not rows:
        return {}
    maximize = "loss" not in metric.lower()
    best = rows[0]
    for row in rows[1:]:
        current = float(row.get(metric, float("-inf") if maximize else float("inf")))
        previous = float(best.get(metric, float("-inf") if maximize else float("inf")))
        current_tie = float(row.get(tie_breaker, float("inf")))
        previous_tie = float(best.get(tie_breaker, float("inf")))
        improved = current > previous if maximize else current < previous
        tied_better = abs(current - previous) <= 1e-12 and current_tie < previous_tie
        if improved or tied_better:
            best = row
    return best


def write_metrics_summary(
    output_dir: Path,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    model_config: Any,
    git_metadata: dict[str, Any],
) -> None:
    """Write a compact table row next to every trained model."""

    metric = str(config.get("checkpoint_metric", "val_loss"))
    tie_breaker = str(config.get("tie_breaker_metric", "val_loss"))
    best = _best_history_row(history, metric, tie_breaker)
    final = _with_derived_scores(history[-1]) if history else {}
    summary = {
        "experiment": output_dir.name,
        "output_dir": str(output_dir),
        "selection_metric": metric,
        "tie_breaker": tie_breaker,
        "epochs_ran": len(history),
        "best_epoch": best.get("epoch"),
        "best_val_competition_score": best.get("val_competition_score"),
        "best_val_loss": best.get("val_loss"),
        "best_val_crop_macro_f1": best.get("val_crop_macro_f1"),
        "best_val_crop_macro_f1_consistent": best.get("val_crop_macro_f1_consistent"),
        "best_val_rice_stage_macro_f1": best.get("val_rice_stage_macro_f1"),
        "best_val_crop_accuracy": best.get("val_crop_accuracy"),
        "best_val_crop_accuracy_consistent": best.get("val_crop_accuracy_consistent"),
        "best_val_rice_stage_accuracy": best.get("val_rice_stage_accuracy"),
        "best_val_joint_accuracy": best.get("val_joint_accuracy"),
        "best_val_joint_accuracy_consistent": best.get("val_joint_accuracy_consistent"),
        "best_train_competition_score": best.get("train_competition_score"),
        "train_val_score_gap": float(best.get("train_competition_score", 0.0)) - float(best.get("val_competition_score", 0.0)),
        "final_epoch": final.get("epoch"),
        "final_val_competition_score": final.get("val_competition_score"),
        "final_val_competition_score_consistent": final.get("val_competition_score_consistent"),
        "final_val_loss": final.get("val_loss"),
        "model_config": config_asdict(model_config),
        "git": git_metadata,
    }
    for key in (
        "best_val_query_shift_avg_rice_stage_macro_f1",
        "best_val_query_shift_worst_rice_stage_macro_f1",
        "best_val_query_shift_plus_rice_stage_macro_f1",
        "best_val_query_shift_minus_rice_stage_macro_f1",
        "final_val_query_shift_avg_rice_stage_macro_f1",
        "final_val_query_shift_worst_rice_stage_macro_f1",
        "final_val_query_shift_plus_rice_stage_macro_f1",
        "final_val_query_shift_minus_rice_stage_macro_f1",
    ):
        source = best if key.startswith("best_") else final
        metric_key = key.replace("best_", "").replace("final_", "")
        summary[key] = source.get(metric_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    markdown = "\n".join(
        [
            "|experiment|best_epoch|best_val_score|crop_f1|rice_stage_f1|val_loss|gap|",
            "|---|---|---|---|---|---|---|",
            "|{experiment}|{best_epoch}|{best_val_competition_score:.6f}|{best_val_crop_macro_f1:.6f}|{best_val_rice_stage_macro_f1:.6f}|{best_val_loss:.6f}|{train_val_score_gap:.6f}|".format(
                **{key: (0.0 if value is None and key.startswith(("best_", "train_")) else value) for key, value in summary.items()}
            ),
        ]
    )
    (output_dir / "metrics_summary.md").write_text(markdown + "\n")


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
    model_type = normalize_model_type(config.get("model_type", "query_cnn_transformer"))

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
    aux_feature_set = str(config.get("aux_feature_set", "summary"))
    use_spectral_indices = bool(config.get("use_spectral_indices", False))
    disabled_bands = list(config.get("disabled_bands", []))
    _idx_stats_raw = config.get("spectral_index_stats_json", None)
    spectral_index_stats_json = resolve_path(_idx_stats_raw) if _idx_stats_raw else None
    _split_csv_raw = config.get("split_csv")
    split_csv_path = resolve_path(_split_csv_raw) if _split_csv_raw else None
    split_train = "train" if split_csv_path else None
    split_val = "val" if split_csv_path else None

    train_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=split_csv_path,
        split=split_train,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        shuffle_labels_seed=int(config.get("seed", 42)) if args.shuffle_labels else None,
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
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
        disabled_bands=disabled_bands,
    )
    val_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=split_csv_path,
        split=split_val,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=use_aux_features,
        aux_feature_set=aux_feature_set,
        use_spectral_indices=use_spectral_indices,
        spectral_index_stats_json=spectral_index_stats_json,
        use_relative_doy=bool(config.get("use_relative_doy", False)),
        disabled_bands=disabled_bands,
    )
    if use_aux_features:
        model_config_data["aux_feature_dim"] = int(train_ds.aux_feature_dim)
    else:
        model_config_data["aux_feature_dim"] = 0
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
    val_loader = DataLoader(
        val_ds,
        **build_dataloader_kwargs(
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            config=config,
        ),
    )
    extra_val_loaders: dict[str, DataLoader] = {}
    robust_val_query_shift_days = int(config.get("robust_val_query_shift_days", 0))
    if robust_val_query_shift_days > 0:
        for name, query_shift in (
            ("query_shift_plus", float(robust_val_query_shift_days)),
            ("query_shift_minus", float(-robust_val_query_shift_days)),
        ):
            extra_val_ds = QueryDatePatchDataset(
                npz_path=resolve_path(config["dataset_npz"]),
                split_csv=split_csv_path,
                split=split_val,
                normalization_json=resolve_path(config["normalization_json"]),
                rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
                include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
                use_aux_features=use_aux_features,
                aux_feature_set=aux_feature_set,
                use_spectral_indices=use_spectral_indices,
                spectral_index_stats_json=spectral_index_stats_json,
                use_relative_doy=bool(config.get("use_relative_doy", False)),
                disabled_bands=disabled_bands,
                fixed_time_shift_days=0.0,
                fixed_query_doy_shift_days=query_shift,
            )
            extra_val_loaders[name] = DataLoader(
                extra_val_ds,
                **build_dataloader_kwargs(
                    batch_size=int(config["batch_size"]),
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    config=config,
                ),
            )

    model = build_model(model_type, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(args.epochs or config.get("epochs", 10))
    scheduler = build_scheduler(config, optimizer, epochs)
    output_dir = resolve_path(config["output_dir"])
    git_metadata = collect_git_metadata()

    payload = {
        "model_type": model_type,
        "model_config": config_asdict(model_config),
        "train_config": config,
        "device": str(device),
        "task": "point_date_crop_stage_classification",
        "aux_feature_names": train_ds.aux_feature_names,
        "aux_feature_set": aux_feature_set if use_aux_features else None,
        "disabled_bands": disabled_bands,
        "git": git_metadata,
    }
    history = fit_query(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        stage_loss_weight=float(config.get("stage_loss_weight", 0.6)),
        crop_loss_weight=float(config.get("crop_loss_weight", 1.0)),
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
        stage_label_smoothing=config.get("stage_label_smoothing"),
        stage_ordinal_loss_weight=float(config.get("stage_ordinal_loss_weight", 0.0)),
        stage_sequence_loss_weight=float(config.get("stage_sequence_loss_weight", 0.0)),
        stage_max_forward_step=float(config.get("stage_max_forward_step", 1.75)),
        stage_postprocess=str(config.get("stage_postprocess", "none")),
        crop_class_weights=crop_class_weights,
        stage_class_weights=stage_class_weights,
        point_crop_consistency_loss_weight=float(config.get("point_crop_consistency_loss_weight", 0.0)),
        extra_val_loaders=extra_val_loaders or None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config_resolved.json").write_text(json.dumps({**payload, "history_file": str(output_dir / "history.json")}, indent=2))
    write_metrics_summary(output_dir, history, config, model_config, git_metadata)
    print(json.dumps({"model": str(output_dir / "model.pt"), "history": str(output_dir / "history.json"), "device": str(device), "train_queries": len(train_ds), "val_queries": len(val_ds)}, indent=2))


if __name__ == "__main__":
    main()
