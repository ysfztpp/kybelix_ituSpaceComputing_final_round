from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for training. Install PyTorch before running training scripts.") from exc


def multitask_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], phenophase_weight: float = 0.2) -> tuple[torch.Tensor, dict[str, float]]:
    crop_loss = nn.functional.cross_entropy(outputs["crop_logits"], batch["crop_type_id"])
    target = batch["phenophase_target"]
    valid = target > 0
    if valid.any():
        pheno_loss = nn.functional.smooth_l1_loss(outputs["phenophase_norm"][valid], target[valid])
    else:
        pheno_loss = outputs["phenophase_norm"].sum() * 0.0
    total = crop_loss + phenophase_weight * pheno_loss
    return total, {"loss": float(total.detach().cpu()), "crop_loss": float(crop_loss.detach().cpu()), "phenophase_loss": float(pheno_loss.detach().cpu())}


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type in {"cuda", "cpu"})


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    phenophase_weight: float,
    amp: bool = False,
    scaler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {"loss": 0.0, "crop_loss": 0.0, "phenophase_loss": 0.0, "crop_correct": 0.0, "count": 0.0}
    accumulation = max(1, int(gradient_accumulation_steps))
    context = torch.enable_grad() if train else torch.no_grad()
    if train:
        optimizer.zero_grad(set_to_none=True)

    with context:
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
            with _autocast_context(device, amp):
                outputs = model(batch["patches"], batch["time_mask"], batch["time_doy"])
                loss, parts = multitask_loss(outputs, batch, phenophase_weight)
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
            preds = outputs["crop_logits"].argmax(dim=1)
            totals["crop_correct"] += float((preds == batch["crop_type_id"]).sum().detach().cpu())
            totals["count"] += batch_size
            for key in ["loss", "crop_loss", "phenophase_loss"]:
                totals[key] += parts[key] * batch_size
    count = max(totals.pop("count"), 1.0)
    return {**{key: value / count for key, value in totals.items() if key != "crop_correct"}, "crop_accuracy": totals["crop_correct"] / count}


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    phenophase_weight: float,
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
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            phenophase_weight=phenophase_weight,
            amp=amp,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_grad_norm=clip_grad_norm,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            phenophase_weight=phenophase_weight,
            amp=False,
            scaler=None,
            gradient_accumulation_steps=1,
            clip_grad_norm=clip_grad_norm,
        )
        if scheduler is not None:
            scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
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
