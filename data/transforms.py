from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NpzPatchNormalizer:
    """Apply train-only band stats to [T, B, H, W] or [N, T, B, H, W] patches."""

    def __init__(self, stats_path: Path, method: str = "zscore") -> None:
        stats = json.loads(Path(stats_path).read_text())
        if method not in {"zscore", "robust"}:
            raise ValueError("method must be 'zscore' or 'robust'")
        self.stats = stats
        self.bands = list(stats["bands"])
        self.method = method
        if method == "zscore":
            self.center = np.asarray([stats["per_band"][band]["mean"] for band in self.bands], dtype=np.float32)
            self.scale = np.asarray([stats["per_band"][band]["std"] for band in self.bands], dtype=np.float32)
        else:
            self.center = np.asarray([stats["per_band"][band]["median"] for band in self.bands], dtype=np.float32)
            self.scale = np.asarray([stats["per_band"][band]["iqr"] for band in self.bands], dtype=np.float32)
        self.scale = np.maximum(self.scale, 1e-6)

    def __call__(self, patches: np.ndarray, valid_pixel_mask: np.ndarray | None = None) -> np.ndarray:
        out = patches.astype(np.float32, copy=True)
        shape = (1,) * (out.ndim - 3) + (len(self.bands), 1, 1)
        out = (out - self.center.reshape(shape)) / self.scale.reshape(shape)
        if valid_pixel_mask is not None:
            out = np.where(valid_pixel_mask, out, 0.0).astype(np.float32, copy=False)
        return out
