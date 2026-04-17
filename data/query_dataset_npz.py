from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .transforms import NpzPatchNormalizer

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


class QueryDatePatchDataset(Dataset):
    """Expands each point into point-date queries for crop + phenophase-stage classification.

    Each training row corresponds to one known phenophase label:
    full patch time series + queried day-of-year -> crop class and stage class.
    """

    def __init__(
        self,
        npz_path: Path,
        split_csv: Path | None = None,
        split: str | None = None,
        normalization_json: Path | None = None,
        normalization_method: str = "zscore",
        rice_stage_loss_only: bool = True,
        shuffle_labels_seed: int | None = None,
        include_valid_mask_as_channels: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for QueryDatePatchDataset. Install PyTorch before training.")
        with np.load(npz_path, allow_pickle=False) as data:
            self.arrays = {name: data[name] for name in data.files}
        if "phenophase_doy" not in self.arrays or "crop_type_id" not in self.arrays:
            raise ValueError("query training requires phenophase_doy and crop_type_id in the NPZ")

        sample_indices = np.arange(self.arrays["patches"].shape[0], dtype=np.int64)
        if split_csv is not None and split is not None:
            split_df = pd.read_csv(split_csv)
            sample_indices = split_df.loc[split_df["split"] == split, "sample_index"].to_numpy(dtype=np.int64)

        self.normalizer = NpzPatchNormalizer(normalization_json, normalization_method) if normalization_json else None
        self.rice_stage_loss_only = bool(rice_stage_loss_only)
        self.include_valid_mask_as_channels = bool(include_valid_mask_as_channels)
        rice_id = self._rice_class_id()

        rows: list[tuple[int, int, int, int, float]] = []
        for sample_index in sample_indices:
            crop_id = int(self.arrays["crop_type_id"][sample_index])
            for stage_index, doy in enumerate(self.arrays["phenophase_doy"][sample_index].astype(np.int16)):
                if doy <= 0:
                    continue
                stage_weight = 1.0 if (not self.rice_stage_loss_only or crop_id == rice_id) else 0.0
                rows.append((int(sample_index), int(stage_index), int(doy), crop_id, float(stage_weight)))
        if shuffle_labels_seed is not None:
            rng = np.random.default_rng(int(shuffle_labels_seed))
            crop_labels = np.asarray([row[3] for row in rows], dtype=np.int16)
            stage_labels = np.asarray([row[1] for row in rows], dtype=np.int16)
            rng.shuffle(crop_labels)
            rng.shuffle(stage_labels)
            rows = [
                (
                    sample_index,
                    int(stage_labels[row_index]),
                    query_doy,
                    int(crop_labels[row_index]),
                    1.0 if (not self.rice_stage_loss_only or int(crop_labels[row_index]) == rice_id) else 0.0,
                )
                for row_index, (sample_index, _stage_index, query_doy, _crop_id, _stage_weight) in enumerate(rows)
            ]
        self.rows = rows

    def _rice_class_id(self) -> int:
        names = self.arrays.get("crop_type_names")
        if names is None:
            return 1
        names = names.astype(str).tolist()
        return names.index("rice") if "rice" in names else 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample_index, stage_index, query_doy, crop_id, stage_weight = self.rows[item]
        patches = self.arrays["patches"][sample_index]
        valid_pixel_mask = self.arrays["valid_pixel_mask"][sample_index].astype(bool)
        if self.normalizer is not None:
            patches = self.normalizer(patches, valid_pixel_mask)
        if self.include_valid_mask_as_channels:
            patches = np.concatenate([patches, valid_pixel_mask.astype(np.float32)], axis=1)
        return {
            "patches": torch.from_numpy(patches.astype(np.float32, copy=False)),
            "valid_pixel_mask": torch.from_numpy(valid_pixel_mask),
            "band_mask": torch.from_numpy(self.arrays["band_mask"][sample_index].astype(bool)),
            "time_mask": torch.from_numpy(self.arrays["time_mask"][sample_index].astype(bool)),
            "time_doy": torch.from_numpy(self.arrays["time_doy"][sample_index].astype(np.float32)),
            "query_doy": torch.tensor(float(query_doy), dtype=torch.float32),
            "crop_type_id": torch.tensor(crop_id, dtype=torch.long),
            "phenophase_stage_id": torch.tensor(stage_index, dtype=torch.long),
            "stage_loss_weight": torch.tensor(stage_weight, dtype=torch.float32),
            "sample_index": int(sample_index),
            "point_id": int(self.arrays["point_id"][sample_index]),
        }
