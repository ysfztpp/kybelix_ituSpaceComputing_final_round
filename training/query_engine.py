from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for training.") from exc


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type in {"cuda", "cpu"})


def query_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], stage_loss_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    crop_loss = nn.functional.cross_entropy(outputs["crop_logits"], batch["crop_type_id"])
    per_row_stage = nn.functional.cross_entropy(outputs["stage_logits"], batch["phenophase_stage_id"], reduction="none")
    weights = batch["stage_loss_weight"].float()
    if weights.sum() > 0:
        stage_loss = (per_row_stage * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        stage_loss = per_row_stage.mean() * 0.0
    total = crop_loss + stage_loss_weight * stage_loss
    return total, {"loss": float(total.detach().cpu()), "crop_loss": float(crop_loss.detach().cpu()), "stage_loss": float(stage_loss.detach().cpu())}


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


def run_query_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    stage_loss_weight: float,
    amp: bool = False,
    scaler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "crop_loss": 0.0, "stage_loss": 0.0, "crop_correct": 0.0, "stage_correct": 0.0, "rice_stage_correct": 0.0, "count": 0.0, "rice_stage_count": 0.0}
    crop_true: list[int] = []
    crop_pred_all: list[int] = []
    rice_stage_true: list[int] = []
    rice_stage_pred: list[int] = []
    accumulation = max(1, int(gradient_accumulation_steps))
    context = torch.enable_grad() if train else torch.no_grad()
    if train:
        optimizer.zero_grad(set_to_none=True)
    with context:
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
            with _autocast_context(device, amp):
                outputs = model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"])
                loss, parts = query_loss(outputs, batch, stage_loss_weight)
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
            stage_pred = outputs["stage_logits"].argmax(dim=1)
            stage_weight = batch["stage_loss_weight"].float()
            totals["crop_correct"] += float((crop_pred == batch["crop_type_id"]).sum().detach().cpu())
            totals["stage_correct"] += float((stage_pred == batch["phenophase_stage_id"]).sum().detach().cpu())
            totals["rice_stage_correct"] += float(((stage_pred == batch["phenophase_stage_id"]).float() * stage_weight).sum().detach().cpu())
            totals["rice_stage_count"] += float(stage_weight.sum().detach().cpu())
            totals["count"] += batch_size
            crop_true.extend(batch["crop_type_id"].detach().cpu().tolist())
            crop_pred_all.extend(crop_pred.detach().cpu().tolist())
            rice_mask = stage_weight > 0
            rice_stage_true.extend(batch["phenophase_stage_id"][rice_mask].detach().cpu().tolist())
            rice_stage_pred.extend(stage_pred[rice_mask].detach().cpu().tolist())
            for key in ["loss", "crop_loss", "stage_loss"]:
                totals[key] += parts[key] * batch_size
    count = max(totals["count"], 1.0)
    rice_count = max(totals["rice_stage_count"], 1.0)
    return {
        "loss": totals["loss"] / count,
        "crop_loss": totals["crop_loss"] / count,
        "stage_loss": totals["stage_loss"] / count,
        "crop_accuracy": totals["crop_correct"] / count,
        "crop_macro_f1": _macro_f1(crop_true, crop_pred_all, range(3)),
        "stage_accuracy_all_crops": totals["stage_correct"] / count,
        "rice_stage_accuracy": totals["rice_stage_correct"] / rice_count,
        "rice_stage_macro_f1": _macro_f1(rice_stage_true, rice_stage_pred, range(7)),
    }


def fit_query(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    stage_loss_weight: float,
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
        train_metrics = run_query_epoch(model, train_loader, optimizer, device, True, stage_loss_weight, amp, scaler, gradient_accumulation_steps, clip_grad_norm)
        val_metrics = run_query_epoch(model, val_loader, optimizer, device, False, stage_loss_weight, False, None, 1, clip_grad_norm)
        row = {"epoch": epoch, "lr": lr_used, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        for prefix in ("train", "val"):
            row[f"{prefix}_competition_score"] = 0.4 * row[f"{prefix}_crop_macro_f1"] + 0.6 * row[f"{prefix}_rice_stage_macro_f1"]
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
