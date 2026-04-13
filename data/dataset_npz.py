from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .transforms import NpzPatchNormalizer

try:  # Keep preprocessing usable on machines where torch is not installed yet.
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    Dataset = object


class PatchTimeSeriesDataset(Dataset):
    """PyTorch dataset for NPZ tensors shaped [N, T, B, H, W]."""

    def __init__(
        self,
        npz_path: Path,
        split_csv: Path | None = None,
        split: str | None = None,
        normalization_json: Path | None = None,
        normalization_method: str = "zscore",
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for PatchTimeSeriesDataset. Install PyTorch before training.")
        self.data = np.load(npz_path, allow_pickle=False)
        self.indices = np.arange(self.data["patches"].shape[0], dtype=np.int64)
        if split_csv is not None and split is not None:
            split_df = pd.read_csv(split_csv)
            split_indices = split_df.loc[split_df["split"] == split, "sample_index"].to_numpy(dtype=np.int64)
            self.indices = split_indices
        self.normalizer = NpzPatchNormalizer(normalization_json, normalization_method) if normalization_json else None
        self.has_labels = "crop_type_id" in self.data.files and "phenophase_doy" in self.data.files

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        patches = self.data["patches"][idx]
        valid_pixel_mask = self.data["valid_pixel_mask"][idx].astype(bool)
        if self.normalizer is not None:
            patches = self.normalizer(patches, valid_pixel_mask)
        example = {
            "patches": torch.from_numpy(patches.astype(np.float32, copy=False)),
            "valid_pixel_mask": torch.from_numpy(valid_pixel_mask),
            "band_mask": torch.from_numpy(self.data["band_mask"][idx].astype(bool)),
            "time_mask": torch.from_numpy(self.data["time_mask"][idx].astype(bool)),
            "time_doy": torch.from_numpy(self.data["time_doy"][idx].astype(np.float32)),
            "point_id": int(self.data["point_id"][idx]),
        }
        if self.has_labels:
            pheno = self.data["phenophase_doy"][idx].astype(np.float32)
            example["crop_type_id"] = torch.tensor(int(self.data["crop_type_id"][idx]), dtype=torch.long)
            example["phenophase_doy"] = torch.from_numpy(pheno)
            example["phenophase_target"] = torch.from_numpy(np.where(pheno > 0, pheno / 366.0, -1.0).astype(np.float32))
        return example
