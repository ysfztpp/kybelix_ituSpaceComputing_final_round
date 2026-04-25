from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aux_features import aux_feature_names, compute_aux_features
from .transforms import NpzPatchNormalizer

# Band order in the NPZ: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
_B02, _B04, _B08, _B11 = 1, 3, 7, 10
_IDX_EPS = 1e-6


def _spectral_indices(patches: np.ndarray, valid_mask: np.ndarray):
    """Compute NDVI, EVI, LSWI from raw reflectance patches.

    patches:    [T, 12, H, W] float32 raw reflectance (0-fill for invalid pixels)
    valid_mask: [T, 12, H, W] bool

    Returns:
        index_patches [T, 3, H, W]: NDVI, EVI, LSWI (clipped, 0 where invalid)
        index_valid   [T, 3, H, W]: bool validity mask for each index
    """
    b02 = patches[:, _B02]   # [T, H, W]
    b04 = patches[:, _B04]
    b08 = patches[:, _B08]
    b11 = patches[:, _B11]

    ndvi = (b08 - b04) / (b08 + b04 + _IDX_EPS)
    evi  = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0 + _IDX_EPS)
    lswi = (b08 - b11) / (b08 + b11 + _IDX_EPS)

    ndvi = np.clip(ndvi, -1.0,  1.0)
    evi  = np.clip(evi,  -1.0,  1.5)
    lswi = np.clip(lswi, -1.0,  1.0)

    m02 = valid_mask[:, _B02]
    m04 = valid_mask[:, _B04]
    m08 = valid_mask[:, _B08]
    m11 = valid_mask[:, _B11]

    ndvi_v = m04 & m08
    evi_v  = m02 & m04 & m08
    lswi_v = m08 & m11

    index_patches = np.stack([ndvi, evi, lswi], axis=1).astype(np.float32)   # [T, 3, H, W]
    index_valid   = np.stack([ndvi_v, evi_v, lswi_v], axis=1)                # [T, 3, H, W] bool

    index_patches = np.where(index_valid, index_patches, 0.0).astype(np.float32)
    return index_patches, index_valid

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
        use_aux_features: bool = False,
        aux_feature_set: str = "summary",
        random_time_shift_days: int = 0,
        query_doy_dropout_prob: float = 0.0,
        time_doy_dropout_prob: float = 0.0,
        use_spectral_indices: bool = False,
        spectral_index_stats_json: Path | None = None,
        use_relative_doy: bool = False,
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
        self.split = str(split or "")
        self.rice_stage_loss_only = bool(rice_stage_loss_only)
        self.include_valid_mask_as_channels = bool(include_valid_mask_as_channels)
        self.use_aux_features = bool(use_aux_features)
        self.aux_feature_set = str(aux_feature_set)
        self.random_time_shift_days = int(random_time_shift_days)
        self.query_doy_dropout_prob = float(query_doy_dropout_prob)
        self.time_doy_dropout_prob = float(time_doy_dropout_prob)
        self.enable_temporal_augmentation = self.split == "train" and (
            self.random_time_shift_days > 0 or self.query_doy_dropout_prob > 0.0 or self.time_doy_dropout_prob > 0.0
        )
        self.use_spectral_indices = bool(use_spectral_indices)
        self.use_relative_doy = bool(use_relative_doy)
        if self.use_spectral_indices:
            if spectral_index_stats_json is None:
                raise ValueError("spectral_index_stats_json must be provided when use_spectral_indices=True")
            idx_stats = json.loads(Path(spectral_index_stats_json).read_text())
            # shape [3, 1, 1] for broadcasting with [T, 3, H, W]
            self._idx_mean = np.array([idx_stats["NDVI"]["mean"], idx_stats["EVI"]["mean"], idx_stats["LSWI"]["mean"]], dtype=np.float32).reshape(3, 1, 1)
            self._idx_std  = np.maximum(np.array([idx_stats["NDVI"]["std"],  idx_stats["EVI"]["std"],  idx_stats["LSWI"]["std"]],  dtype=np.float32), 1e-6).reshape(3, 1, 1)
        self.bands = self.arrays.get("bands", np.asarray([f"B{i:02d}" for i in range(1, self.arrays["patches"].shape[2] + 1)])).astype(str).tolist()
        self.aux_feature_names = aux_feature_names(self.bands, feature_set=self.aux_feature_set) if self.use_aux_features else []
        self.aux_feature_dim = len(self.aux_feature_names)
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
        self._aux_features: np.ndarray | None = None
        if self.use_aux_features:
            self._aux_features = np.stack(
                [
                    compute_aux_features(
                        self.arrays["patches"][sample_index],
                        self.arrays["valid_pixel_mask"][sample_index],
                        self.arrays["time_mask"][sample_index],
                        self.arrays["time_doy"][sample_index],
                        float(query_doy),
                        self.bands,
                        feature_set=self.aux_feature_set,
                    )
                    for sample_index, _stage_index, query_doy, _crop_id, _stage_weight in self.rows
                ]
            ).astype(np.float32, copy=False)

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
        raw_patches = self.arrays["patches"][sample_index]   # [T, 12, H, W]
        valid_pixel_mask = self.arrays["valid_pixel_mask"][sample_index].astype(bool)  # [T, 12, H, W]

        # Compute spectral indices from raw (unnormalized) patches before band normalization
        if self.use_spectral_indices:
            index_patches, index_valid = _spectral_indices(raw_patches, valid_pixel_mask)  # [T, 3, H, W] each

        patches = raw_patches
        if self.normalizer is not None:
            patches = self.normalizer(patches, valid_pixel_mask)

        if self.use_spectral_indices:
            norm_idx = np.where(index_valid, (index_patches - self._idx_mean) / self._idx_std, 0.0).astype(np.float32)
            patches = np.concatenate([patches, norm_idx], axis=1)  # [T, 15, H, W]

        if self.include_valid_mask_as_channels:
            if self.use_spectral_indices:
                full_valid = np.concatenate([valid_pixel_mask, index_valid], axis=1)  # [T, 15, H, W]
            else:
                full_valid = valid_pixel_mask  # [T, 12, H, W]
            patches = np.concatenate([patches, full_valid.astype(np.float32)], axis=1)
        time_doy = self.arrays["time_doy"][sample_index].astype(np.float32).copy()
        query_doy_value = float(query_doy)
        query_doy_mask = 1.0
        if self.enable_temporal_augmentation:
            if self.random_time_shift_days > 0:
                shift = float(np.random.randint(-self.random_time_shift_days, self.random_time_shift_days + 1))
                positive_mask = time_doy > 0
                time_doy[positive_mask] = np.clip(time_doy[positive_mask] + shift, 1.0, 366.0)
                query_doy_value = float(np.clip(query_doy_value + shift, 1.0, 366.0))
            if self.time_doy_dropout_prob > 0.0:
                existing = self.arrays["time_mask"][sample_index].astype(bool)
                drop_mask = (np.random.random(size=time_doy.shape[0]) < self.time_doy_dropout_prob) & existing
                time_doy[drop_mask] = 0.0
            if self.query_doy_dropout_prob > 0.0 and float(np.random.random()) < self.query_doy_dropout_prob:
                query_doy_value = 0.0
                query_doy_mask = 0.0
        if self.use_relative_doy:
            # Subtract series temporal center so the model sees relative dates.
            # Computed after augmentation so shifts cancel out automatically.
            time_mask_bool = self.arrays["time_mask"][sample_index].astype(bool)
            valid_doys = time_doy[time_mask_bool]
            series_center = float(valid_doys.mean()) if len(valid_doys) > 0 else 183.0
            time_doy = time_doy.copy()
            time_doy[time_mask_bool] -= series_center
            query_doy_value = query_doy_value - series_center
        sample = {
            "patches": torch.from_numpy(patches.astype(np.float32, copy=False)),
            "valid_pixel_mask": torch.from_numpy(valid_pixel_mask),
            "band_mask": torch.from_numpy(self.arrays["band_mask"][sample_index].astype(bool)),
            "time_mask": torch.from_numpy(self.arrays["time_mask"][sample_index].astype(bool)),
            "time_doy": torch.from_numpy(time_doy),
            "query_doy": torch.tensor(query_doy_value, dtype=torch.float32),
            "query_doy_mask": torch.tensor(query_doy_mask, dtype=torch.float32),
            "crop_type_id": torch.tensor(crop_id, dtype=torch.long),
            "phenophase_stage_id": torch.tensor(stage_index, dtype=torch.long),
            "stage_loss_weight": torch.tensor(stage_weight, dtype=torch.float32),
            "sample_index": int(sample_index),
            "point_id": int(self.arrays["point_id"][sample_index]),
        }
        if self.use_aux_features:
            if self._aux_features is None:
                raise RuntimeError("aux feature cache was not initialized")
            sample["aux_features"] = torch.from_numpy(self._aux_features[item])
        return sample
