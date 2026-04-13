from __future__ import annotations

from pathlib import Path

import numpy as np

from data.transforms import NpzPatchNormalizer

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for inference. Install PyTorch before prediction.") from exc

from models.cnn_transformer import CNNTransformerBaseline, CNNTransformerConfig


def load_model(checkpoint_path: Path, device: torch.device) -> CNNTransformerBaseline:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = CNNTransformerConfig(**checkpoint["model_config"])
    model = CNNTransformerBaseline(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_npz(npz_path: Path, checkpoint_path: Path, normalization_json: Path, batch_size: int = 16, device_name: str = "auto") -> dict[str, np.ndarray]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model = load_model(checkpoint_path, device)
    data = np.load(npz_path, allow_pickle=False)
    normalizer = NpzPatchNormalizer(normalization_json)
    crop_logits: list[np.ndarray] = []
    phenophase_norm: list[np.ndarray] = []

    for start in range(0, data["patches"].shape[0], batch_size):
        end = min(start + batch_size, data["patches"].shape[0])
        patches = normalizer(data["patches"][start:end], data["valid_pixel_mask"][start:end].astype(bool))
        batch = {
            "patches": torch.from_numpy(patches).to(device),
            "time_mask": torch.from_numpy(data["time_mask"][start:end].astype(bool)).to(device),
            "time_doy": torch.from_numpy(data["time_doy"][start:end].astype(np.float32)).to(device),
        }
        with torch.no_grad():
            outputs = model(batch["patches"], batch["time_mask"], batch["time_doy"])
        crop_logits.append(outputs["crop_logits"].detach().cpu().numpy())
        phenophase_norm.append(outputs["phenophase_norm"].detach().cpu().numpy())

    return {
        "point_id": data["point_id"],
        "crop_logits": np.concatenate(crop_logits, axis=0),
        "crop_type_id": np.concatenate(crop_logits, axis=0).argmax(axis=1),
        "phenophase_doy": np.rint(np.concatenate(phenophase_norm, axis=0) * 366.0).clip(1, 366).astype(np.int16),
    }
