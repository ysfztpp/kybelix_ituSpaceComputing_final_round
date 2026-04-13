from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.dataset_npz import PatchTimeSeriesDataset
from models.cnn_transformer import CNNTransformerBaseline, CNNTransformerConfig
from training.engine import fit

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. In Colab run: pip install torch torchvision") from exc


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN + Transformer baseline.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_baseline.json")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_scheduler(config: dict, optimizer, epochs: int):
    name = str(config.get("scheduler", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "cosine":
        warmup_epochs = max(0, int(config.get("warmup_epochs", 0)))
        if warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    raise ValueError(f"unknown scheduler: {name}")


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    model_config_data = json.loads(resolve_path(config["model_config"]).read_text())
    model_config = CNNTransformerConfig(**{key: value for key, value in model_config_data.items() if key in CNNTransformerConfig.__annotations__})

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))

    train_ds = PatchTimeSeriesDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="train",
        normalization_json=resolve_path(config["normalization_json"]),
    )
    val_ds = PatchTimeSeriesDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="val",
        normalization_json=resolve_path(config["normalization_json"]),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=pin_memory,
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=pin_memory,
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )

    model = CNNTransformerBaseline(model_config).to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(args.epochs or config.get("epochs", 5))
    scheduler = build_scheduler(config, optimizer, epochs)

    output_dir = resolve_path(config["output_dir"])
    checkpoint_payload = {"model_config": asdict(model_config), "train_config": config, "device": str(device)}
    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs,
        float(config.get("phenophase_loss_weight", 0.2)),
        output_dir=output_dir,
        scheduler=scheduler,
        amp=bool(config.get("amp", False)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        clip_grad_norm=float(config.get("clip_grad_norm", 1.0)),
        early_stopping_patience=config.get("early_stopping_patience"),
        save_best_only=bool(config.get("save_best_only", True)),
        checkpoint_payload=checkpoint_payload,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "model.pt").exists():
        torch.save({**checkpoint_payload, "model_state_dict": model.state_dict(), "history": history}, output_dir / "model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"model": str(output_dir / "model.pt"), "history": str(output_dir / "history.json"), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
