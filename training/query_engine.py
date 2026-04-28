from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for training.") from exc

from preprocessing.constants import PHENOPHASE_STAGE_ID_TO_DOY_RANK
from .stage_decoding import maybe_decode_stages


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type in {"cuda", "cpu"})


def query_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage_loss_weight: float,
    crop_loss_weight: float = 1.0,
    label_smoothing: float = 0.0,
    stage_label_smoothing: float | None = None,
    stage_ordinal_loss_weight: float = 0.0,
    stage_sequence_loss_weight: float = 0.0,
    stage_max_forward_step: float = 1.75,
    crop_class_weights: torch.Tensor | None = None,
    stage_class_weights: torch.Tensor | None = None,
    point_crop_consistency_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    crop_loss = nn.functional.cross_entropy(
        outputs["crop_logits"],
        batch["crop_type_id"],
        weight=crop_class_weights,
        label_smoothing=label_smoothing,
    )
    stage_smoothing = float(label_smoothing if stage_label_smoothing is None else stage_label_smoothing)
    per_row_stage = nn.functional.cross_entropy(
        outputs["stage_logits"],
        batch["phenophase_stage_id"],
        reduction="none",
        weight=stage_class_weights,
        label_smoothing=stage_smoothing,
    )
    weights = batch["stage_loss_weight"].float()
    stage_rank_positions = torch.tensor(
        PHENOPHASE_STAGE_ID_TO_DOY_RANK,
        device=outputs["stage_logits"].device,
        dtype=outputs["stage_logits"].dtype,
    )
    if weights.sum() > 0:
        stage_loss = (per_row_stage * weights).sum() / weights.sum().clamp_min(1.0)
        stage_probabilities = torch.softmax(outputs["stage_logits"], dim=1)
        stage_positions = stage_rank_positions
        expected_stage = (stage_probabilities * stage_positions.unsqueeze(0)).sum(dim=1)
        target_stage = stage_rank_positions[batch["phenophase_stage_id"]]
        per_row_ordinal = nn.functional.smooth_l1_loss(expected_stage, target_stage, reduction="none")
        ordinal_stage_loss = (per_row_ordinal * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        stage_loss = per_row_stage.mean() * 0.0
        ordinal_stage_loss = per_row_stage.mean() * 0.0
    sequence_stage_loss = stage_loss * 0.0
    sequence_group_count = 0.0
    if float(stage_sequence_loss_weight) > 0.0:
        sequence_losses: list[torch.Tensor] = []
        stage_probabilities = torch.softmax(outputs["stage_logits"], dim=1)
        stage_positions = stage_rank_positions
        expected_stage = (stage_probabilities * stage_positions.unsqueeze(0)).sum(dim=1)
        point_ids = batch["point_id"].detach().long()
        query_doys = batch["query_doy"].float()
        weights = batch["stage_loss_weight"].float()
        unique_points = torch.unique(point_ids)
        for point_id in unique_points:
            indices = torch.where((point_ids == point_id) & (weights > 0.0))[0]
            if indices.numel() < 2:
                continue
            ordered = indices[torch.argsort(query_doys[indices])]
            diffs = expected_stage[ordered][1:] - expected_stage[ordered][:-1]
            backward_penalty = torch.relu(-diffs)
            overshoot_penalty = torch.relu(diffs - float(stage_max_forward_step))
            sequence_losses.append(backward_penalty.mean() + 0.5 * overshoot_penalty.mean())
        if sequence_losses:
            sequence_group_count = float(len(sequence_losses))
            sequence_stage_loss = torch.stack(sequence_losses).mean()
    point_crop_consistency_loss = crop_loss * 0.0
    point_crop_group_count = 0.0
    if float(point_crop_consistency_loss_weight) > 0.0:
        crop_probabilities = torch.softmax(outputs["crop_logits"], dim=1)
        point_ids = batch["point_id"].detach().long()
        group_losses: list[torch.Tensor] = []
        for point_id in torch.unique(point_ids):
            indices = torch.where(point_ids == point_id)[0]
            if indices.numel() < 2:
                continue
            group_probabilities = crop_probabilities[indices]
            group_mean = group_probabilities.mean(dim=0, keepdim=True)
            group_losses.append((group_probabilities - group_mean).pow(2).mean())
        if group_losses:
            point_crop_group_count = float(len(group_losses))
            point_crop_consistency_loss = torch.stack(group_losses).mean()
    total = (
        float(crop_loss_weight) * crop_loss
        + stage_loss_weight * stage_loss
        + float(stage_ordinal_loss_weight) * ordinal_stage_loss
        + float(stage_sequence_loss_weight) * sequence_stage_loss
        + float(point_crop_consistency_loss_weight) * point_crop_consistency_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "crop_loss": float(crop_loss.detach().cpu()),
        "stage_loss": float(stage_loss.detach().cpu()),
        "ordinal_stage_loss": float(ordinal_stage_loss.detach().cpu()),
        "sequence_stage_loss": float(sequence_stage_loss.detach().cpu()),
        "point_crop_consistency_loss": float(point_crop_consistency_loss.detach().cpu()),
        "stage_supervised_count": float(weights.sum().detach().cpu()),
        "sequence_group_count": sequence_group_count,
        "point_crop_group_count": point_crop_group_count,
    }


def _macro_f1(y_true: list[int], y_pred: list[int], labels: range) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return sum(scores) / max(len(scores), 1)


def _apply_crop_consistency(crop_logits: torch.Tensor, point_ids: torch.Tensor) -> list[int]:
    if crop_logits.numel() == 0:
        return []
    consistent = torch.empty(crop_logits.shape[0], dtype=torch.long)
    for point_id in torch.unique(point_ids):
        indices = torch.where(point_ids == point_id)[0]
        summed_logits = crop_logits[indices].sum(dim=0)
        majority_crop = int(summed_logits.argmax().item())
        consistent[indices] = majority_crop
    return consistent.tolist()


def run_query_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    stage_loss_weight: float,
    crop_loss_weight: float = 1.0,
    amp: bool = False,
    scaler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    label_smoothing: float = 0.0,
    stage_label_smoothing: float | None = None,
    stage_ordinal_loss_weight: float = 0.0,
    stage_sequence_loss_weight: float = 0.0,
    stage_max_forward_step: float = 1.75,
    stage_postprocess: str = "none",
    crop_class_weights: torch.Tensor | None = None,
    stage_class_weights: torch.Tensor | None = None,
    point_crop_consistency_loss_weight: float = 0.0,
) -> dict[str, float]:
    model.train(train)
    totals = {
        "loss": 0.0,
        "crop_loss": 0.0,
        "stage_loss": 0.0,
        "ordinal_stage_loss": 0.0,
        "sequence_stage_loss": 0.0,
        "point_crop_consistency_loss": 0.0,
        "count": 0.0,
        "stage_supervised_count": 0.0,
        "sequence_group_count": 0.0,
        "point_crop_group_count": 0.0,
    }
    crop_true: list[int] = []
    crop_pred_all: list[int] = []
    stage_true_all: list[int] = []
    stage_weight_all: list[float] = []
    point_id_all: list[int] = []
    query_doy_all: list[float] = []
    stage_logits_all: list[torch.Tensor] = []
    crop_logits_all: list[torch.Tensor] = []
    accumulation = max(1, int(gradient_accumulation_steps))
    context = torch.enable_grad() if train else torch.no_grad()
    if train:
        optimizer.zero_grad(set_to_none=True)
    with context:
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
            with _autocast_context(device, amp):
                outputs = model(
                    batch["patches"],
                    batch["time_mask"],
                    batch["time_doy"],
                    batch["query_doy"],
                    batch.get("aux_features"),
                    batch.get("query_doy_mask"),
                )
                loss, parts = query_loss(
                    outputs,
                    batch,
                    stage_loss_weight,
                    crop_loss_weight,
                    label_smoothing,
                    stage_label_smoothing,
                    stage_ordinal_loss_weight,
                    stage_sequence_loss_weight,
                    stage_max_forward_step,
                    crop_class_weights,
                    stage_class_weights,
                    point_crop_consistency_loss_weight,
                )
                loss_for_backward = loss / accumulation
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                    if step % accumulation == 0 or step == len(loader):
                        scaler.unscale_(optimizer)
                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if step % accumulation == 0 or step == len(loader):
                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch["patches"].shape[0])
            crop_pred = outputs["crop_logits"].argmax(dim=1)
            totals["count"] += batch_size
            crop_true.extend(batch["crop_type_id"].detach().cpu().tolist())
            crop_pred_all.extend(crop_pred.detach().cpu().tolist())
            stage_true_all.extend(batch["phenophase_stage_id"].detach().cpu().tolist())
            stage_weight_all.extend(batch["stage_loss_weight"].detach().cpu().tolist())
            point_id_all.extend(batch["point_id"].detach().cpu().tolist())
            query_doy_all.extend(batch["query_doy"].detach().cpu().tolist())
            stage_logits_all.append(outputs["stage_logits"].detach().cpu())
            crop_logits_all.append(outputs["crop_logits"].detach().cpu())
            totals["loss"] += parts["loss"] * batch_size
            totals["crop_loss"] += parts["crop_loss"] * batch_size
            totals["stage_loss"] += parts["stage_loss"] * parts["stage_supervised_count"]
            totals["ordinal_stage_loss"] += parts["ordinal_stage_loss"] * parts["stage_supervised_count"]
            totals["sequence_stage_loss"] += parts["sequence_stage_loss"] * parts["sequence_group_count"]
            totals["point_crop_consistency_loss"] += parts["point_crop_consistency_loss"] * parts["point_crop_group_count"]
            totals["stage_supervised_count"] += parts["stage_supervised_count"]
            totals["sequence_group_count"] += parts["sequence_group_count"]
            totals["point_crop_group_count"] += parts["point_crop_group_count"]
    count = max(totals["count"], 1.0)
    stage_supervised_count = max(totals["stage_supervised_count"], 1.0)
    sequence_group_count = max(totals["sequence_group_count"], 1.0)
    point_crop_group_count = max(totals["point_crop_group_count"], 1.0)
    stage_logits_tensor = torch.cat(stage_logits_all, dim=0) if stage_logits_all else torch.empty((0, 7), dtype=torch.float32)
    crop_logits_tensor = torch.cat(crop_logits_all, dim=0) if crop_logits_all else torch.empty((0, 3), dtype=torch.float32)
    point_ids_tensor = torch.tensor(point_id_all, dtype=torch.long)
    stage_pred_all = maybe_decode_stages(
        stage_logits_tensor,
        point_ids_tensor,
        torch.tensor(query_doy_all, dtype=torch.float32),
        mode=stage_postprocess,
    ).tolist()
    crop_pred_consistent_all = _apply_crop_consistency(crop_logits_tensor, point_ids_tensor)
    stage_correct = sum(1.0 for truth, pred in zip(stage_true_all, stage_pred_all) if int(truth) == int(pred))
    rice_indices = [index for index, weight in enumerate(stage_weight_all) if float(weight) > 0.0]
    rice_stage_true = [int(stage_true_all[index]) for index in rice_indices]
    rice_stage_pred = [int(stage_pred_all[index]) for index in rice_indices]
    rice_stage_correct = sum(1.0 for truth, pred in zip(rice_stage_true, rice_stage_pred) if truth == pred)
    joint_correct = sum(
        1.0
        for crop_truth, crop_pred, stage_truth, stage_pred in zip(crop_true, crop_pred_all, stage_true_all, stage_pred_all)
        if int(crop_truth) == int(crop_pred) and int(stage_truth) == int(stage_pred)
    )
    joint_correct_consistent = sum(
        1.0
        for crop_truth, crop_pred, stage_truth, stage_pred in zip(crop_true, crop_pred_consistent_all, stage_true_all, stage_pred_all)
        if int(crop_truth) == int(crop_pred) and int(stage_truth) == int(stage_pred)
    )
    rice_joint_correct = sum(
        1.0
        for index in rice_indices
        if int(crop_true[index]) == int(crop_pred_all[index]) and int(stage_true_all[index]) == int(stage_pred_all[index])
    )
    rice_joint_correct_consistent = sum(
        1.0
        for index in rice_indices
        if int(crop_true[index]) == int(crop_pred_consistent_all[index]) and int(stage_true_all[index]) == int(stage_pred_all[index])
    )
    rice_count = max(float(len(rice_indices)), 1.0)
    return {
        "loss": totals["loss"] / count,
        "crop_loss": totals["crop_loss"] / count,
        "stage_loss": totals["stage_loss"] / stage_supervised_count,
        "ordinal_stage_loss": totals["ordinal_stage_loss"] / stage_supervised_count,
        "sequence_stage_loss": totals["sequence_stage_loss"] / sequence_group_count,
        "point_crop_consistency_loss": totals["point_crop_consistency_loss"] / point_crop_group_count,
        "crop_accuracy": sum(1.0 for truth, pred in zip(crop_true, crop_pred_all) if int(truth) == int(pred)) / count,
        "crop_macro_f1": _macro_f1(crop_true, crop_pred_all, range(3)),
        "crop_accuracy_consistent": sum(1.0 for truth, pred in zip(crop_true, crop_pred_consistent_all) if int(truth) == int(pred)) / count,
        "crop_macro_f1_consistent": _macro_f1(crop_true, crop_pred_consistent_all, range(3)),
        "stage_accuracy_all_crops": stage_correct / count,
        "rice_stage_accuracy": rice_stage_correct / rice_count,
        "rice_stage_macro_f1": _macro_f1(rice_stage_true, rice_stage_pred, range(7)),
        "joint_accuracy": joint_correct / count,
        "joint_accuracy_consistent": joint_correct_consistent / count,
        "rice_joint_accuracy": rice_joint_correct / rice_count,
        "rice_joint_accuracy_consistent": rice_joint_correct_consistent / rice_count,
    }


def fit_query(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    stage_loss_weight: float,
    crop_loss_weight: float = 1.0,
    output_dir: Path | None = None,
    scheduler: Any | None = None,
    amp: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    early_stopping_patience: int | None = None,
    save_best_only: bool = True,
    checkpoint_payload: dict[str, Any] | None = None,
    checkpoint_metric: str = "val_loss",
    tie_breaker_metric: str = "val_loss",
    label_smoothing: float = 0.0,
    stage_label_smoothing: float | None = None,
    stage_ordinal_loss_weight: float = 0.0,
    stage_sequence_loss_weight: float = 0.0,
    stage_max_forward_step: float = 1.75,
    stage_postprocess: str = "none",
    crop_class_weights: torch.Tensor | None = None,
    stage_class_weights: torch.Tensor | None = None,
    point_crop_consistency_loss_weight: float = 0.0,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    maximize_checkpoint_metric = "loss" not in checkpoint_metric.lower()
    best_metric = float("-inf") if maximize_checkpoint_metric else float("inf")
    best_tie_breaker = float("inf")
    best_epoch = 0
    best_val_loss_seen = float("inf")
    stale_epochs = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for epoch in range(1, epochs + 1):
        # Log the learning rate actually used during this epoch. The scheduler is
        # stepped at the end so the printed value is not accidentally one epoch ahead.
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_query_epoch(
            model,
            train_loader,
            optimizer,
            device,
            True,
            stage_loss_weight,
            crop_loss_weight,
            amp,
            scaler,
            gradient_accumulation_steps,
            clip_grad_norm,
            label_smoothing,
            stage_label_smoothing,
            stage_ordinal_loss_weight,
            stage_sequence_loss_weight,
            stage_max_forward_step,
            stage_postprocess,
            crop_class_weights,
            stage_class_weights,
            point_crop_consistency_loss_weight,
        )
        val_metrics = run_query_epoch(
            model,
            val_loader,
            optimizer,
            device,
            False,
            stage_loss_weight,
            crop_loss_weight,
            False,
            None,
            1,
            clip_grad_norm,
            0.0,
            0.0 if stage_label_smoothing is None else stage_label_smoothing,
            stage_ordinal_loss_weight,
            stage_sequence_loss_weight,
            stage_max_forward_step,
            stage_postprocess,
            crop_class_weights,
            stage_class_weights,
            point_crop_consistency_loss_weight,
        )
        row = {"epoch": epoch, "lr": lr_used, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        for prefix in ("train", "val"):
            row[f"{prefix}_competition_score"] = 0.4 * row[f"{prefix}_crop_macro_f1"] + 0.6 * row[f"{prefix}_rice_stage_macro_f1"]
            row[f"{prefix}_competition_score_consistent"] = 0.4 * row[f"{prefix}_crop_macro_f1_consistent"] + 0.6 * row[f"{prefix}_rice_stage_macro_f1"]
        history.append(row)
        print(row)

        if checkpoint_metric not in row:
            raise KeyError(f"checkpoint_metric '{checkpoint_metric}' is not available. Available metrics: {sorted(row)}")
        if tie_breaker_metric not in row:
            raise KeyError(f"tie_breaker_metric '{tie_breaker_metric}' is not available. Available metrics: {sorted(row)}")

        best_val_loss_seen = min(best_val_loss_seen, row["val_loss"])
        metric_value = float(row[checkpoint_metric])
        tie_value = float(row[tie_breaker_metric])
        epsilon = 1e-12
        if maximize_checkpoint_metric:
            improved = metric_value > best_metric + epsilon or (abs(metric_value - best_metric) <= epsilon and tie_value < best_tie_breaker)
        else:
            improved = metric_value < best_metric - epsilon or (abs(metric_value - best_metric) <= epsilon and tie_value < best_tie_breaker)
        if improved:
            best_metric = metric_value
            best_tie_breaker = tie_value
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if output_dir is not None and (improved or not save_best_only):
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(checkpoint_payload or {})
            payload.update(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "checkpoint_metric": checkpoint_metric,
                    "best_metric_value": best_metric,
                    "best_epoch": best_epoch,
                    "tie_breaker_metric": tie_breaker_metric,
                    "best_tie_breaker_value": best_tie_breaker,
                    "best_val_loss": best_val_loss_seen,
                    "history": history,
                }
            )
            torch.save(payload, output_dir / "model.pt")
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            print({"early_stopped_epoch": epoch, "checkpoint_metric": checkpoint_metric, "best_metric_value": best_metric, "best_epoch": best_epoch})
            break
        if scheduler is not None:
            scheduler.step()
    return history
