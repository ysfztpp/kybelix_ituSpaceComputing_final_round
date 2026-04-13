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
            for key in ["loss", "crop_loss", "stage_loss"]:
                totals[key] += parts[key] * batch_size
    count = max(totals["count"], 1.0)
    rice_count = max(totals["rice_stage_count"], 1.0)
    return {
        "loss": totals["loss"] / count,
        "crop_loss": totals["crop_loss"] / count,
        "stage_loss": totals["stage_loss"] / count,
        "crop_accuracy": totals["crop_correct"] / count,
        "stage_accuracy_all_crops": totals["stage_correct"] / count,
        "rice_stage_accuracy": totals["rice_stage_correct"] / rice_count,
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
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    stale_epochs = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for epoch in range(1, epochs + 1):
        train_metrics = run_query_epoch(model, train_loader, optimizer, device, True, stage_loss_weight, amp, scaler, gradient_accumulation_steps, clip_grad_norm)
        val_metrics = run_query_epoch(model, val_loader, optimizer, device, False, stage_loss_weight, False, None, 1, clip_grad_norm)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        improved = row["val_loss"] < best_val_loss
        if improved:
            best_val_loss = row["val_loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if output_dir is not None and (improved or not save_best_only):
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(checkpoint_payload or {})
            payload.update({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_loss": best_val_loss, "history": history})
            torch.save(payload, output_dir / "model.pt")
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            print({"early_stopped_epoch": epoch, "best_val_loss": best_val_loss})
            break
    return history
