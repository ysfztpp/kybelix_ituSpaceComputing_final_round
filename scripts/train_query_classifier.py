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

from data.query_dataset_npz import QueryDatePatchDataset
from models.query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig
from scripts.train_baseline import build_scheduler, resolve_path, select_device
from training.query_engine import fit_query

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. In Colab run: pip install torch torchvision") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PDF-aligned query-date crop + phenophase-stage classifier.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_query_colab_gpu.json")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    model_config_data = json.loads(resolve_path(config["model_config"]).read_text())
    model_config = QueryCNNTransformerConfig(**{key: value for key, value in model_config_data.items() if key in QueryCNNTransformerConfig.__annotations__})
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))

    train_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="train",
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
    )
    val_ds = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split="val",
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)), pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), pin_memory=pin_memory)

    model = QueryCNNTransformerClassifier(model_config).to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(args.epochs or config.get("epochs", 5))
    scheduler = build_scheduler(config, optimizer, epochs)
    output_dir = resolve_path(config["output_dir"])
    payload = {"model_config": asdict(model_config), "train_config": config, "device": str(device), "task": "query_date_crop_stage_classification"}
    history = fit_query(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs,
        float(config.get("stage_loss_weight", 0.6)),
        output_dir=output_dir,
        scheduler=scheduler,
        amp=bool(config.get("amp", False)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        clip_grad_norm=float(config.get("clip_grad_norm", 1.0)),
        early_stopping_patience=config.get("early_stopping_patience"),
        save_best_only=bool(config.get("save_best_only", True)),
        checkpoint_payload=payload,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "model.pt").exists():
        torch.save({**payload, "model_state_dict": model.state_dict(), "history": history}, output_dir / "model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"model": str(output_dir / "model.pt"), "history": str(output_dir / "history.json"), "device": str(device), "train_queries": len(train_ds), "val_queries": len(val_ds)}, indent=2))


if __name__ == "__main__":
    main()
