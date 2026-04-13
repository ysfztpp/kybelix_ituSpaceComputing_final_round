from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import BAND_ORDER


def compute_band_stats(npz_path: Path, output_json: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    patches = data["patches"]
    valid_pixel_mask = data["valid_pixel_mask"].astype(bool)
    bands = data["bands"].astype(str).tolist() if "bands" in data.files else BAND_ORDER

    stats: dict[str, Any] = {
        "source_npz": str(npz_path),
        "policy": "computed from training patches only using valid_pixel_mask; raw patch artifact is not modified",
        "bands": bands,
        "per_band": {},
    }
    for band_index, band_id in enumerate(bands):
        values = patches[:, :, band_index][valid_pixel_mask[:, :, band_index]].astype(np.float64)
        if values.size == 0:
            band_stats = {
                "valid_count": 0,
                "mean": 0.0,
                "std": 1.0,
                "median": 0.0,
                "iqr": 1.0,
                "p02": 0.0,
                "p98": 1.0,
            }
        else:
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            std = float(values.std())
            iqr = float(q75 - q25)
            band_stats = {
                "valid_count": int(values.size),
                "mean": float(values.mean()),
                "std": std if std > 1e-6 else 1.0,
                "median": float(median),
                "iqr": iqr if iqr > 1e-6 else 1.0,
                "p02": float(np.percentile(values, 2)),
                "p98": float(np.percentile(values, 98)),
            }
        stats["per_band"][band_id] = band_stats

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2))
    return stats


class PatchNormalizer:
    def __init__(self, stats_path: Path, method: str = "zscore") -> None:
        self.stats = json.loads(Path(stats_path).read_text())
        self.bands = list(self.stats["bands"])
        if method not in {"zscore", "robust"}:
            raise ValueError("method must be 'zscore' or 'robust'")
        self.method = method

    def __call__(self, patches: np.ndarray, valid_pixel_mask: np.ndarray | None = None) -> np.ndarray:
        out = patches.astype(np.float32, copy=True)
        for band_index, band_id in enumerate(self.bands):
            band_stats = self.stats["per_band"][band_id]
            if self.method == "zscore":
                center = float(band_stats["mean"])
                scale = float(band_stats["std"])
            else:
                center = float(band_stats["median"])
                scale = float(band_stats["iqr"])
            out[..., band_index, :, :] = (out[..., band_index, :, :] - center) / max(scale, 1e-6)
            if valid_pixel_mask is not None:
                out[..., band_index, :, :] = np.where(valid_pixel_mask[..., band_index, :, :], out[..., band_index, :, :], 0.0)
        return out
