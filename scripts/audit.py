from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def query_rows(data: dict[str, np.ndarray], split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    crop_names = data["crop_type_names"].astype(str).tolist()
    stage_names = data["phenophase_names"].astype(str).tolist()
    split_by_sample = split.set_index("sample_index")["split"].to_dict()
    for sample_index in range(data["patches"].shape[0]):
        crop_id = int(data["crop_type_id"][sample_index])
        for stage_id, doy in enumerate(data["phenophase_doy"][sample_index].astype(int)):
            if doy <= 0:
                continue
            rows.append(
                {
                    "sample_index": sample_index,
                    "split": split_by_sample.get(sample_index, "unknown"),
                    "crop_id": crop_id,
                    "crop_type": crop_names[crop_id],
                    "stage_id": stage_id,
                    "stage_name": stage_names[stage_id],
                    "query_doy": int(doy),
                }
            )
    return pd.DataFrame(rows)


def date_only_stage_baseline(rows: pd.DataFrame, stage_count: int) -> dict[str, Any]:
    train = rows[rows["split"] == "train"].copy()
    val = rows[rows["split"] == "val"].copy()
    stage_mean = train.groupby("stage_id")["query_doy"].mean().sort_index()
    stage_ids = stage_mean.index.to_numpy(dtype=int)
    stage_means = stage_mean.to_numpy(dtype=float)
    pred = stage_ids[np.abs(val["query_doy"].to_numpy(dtype=float)[:, None] - stage_means[None, :]).argmin(axis=1)]
    rice_val = val[val["crop_type"] == "rice"].copy()
    rice_pred = stage_ids[np.abs(rice_val["query_doy"].to_numpy(dtype=float)[:, None] - stage_means[None, :]).argmin(axis=1)] if len(rice_val) else np.array([], dtype=int)
    return {
        "stage_mean_doy_from_train": {str(int(k)): float(v) for k, v in stage_mean.items()},
        "date_only_stage_accuracy_all_crops": float((pred == val["stage_id"].to_numpy()).mean()) if len(val) else 0.0,
        "date_only_stage_macro_f1_all_crops": macro_f1(val["stage_id"].to_numpy(), pred, list(range(stage_count))) if len(val) else 0.0,
        "date_only_rice_stage_accuracy": float((rice_pred == rice_val["stage_id"].to_numpy()).mean()) if len(rice_val) else 0.0,
        "date_only_rice_stage_macro_f1": macro_f1(rice_val["stage_id"].to_numpy(), rice_pred, list(range(stage_count))) if len(rice_val) else 0.0,
    }


def constant_doy_mae(data: dict[str, np.ndarray], split: pd.DataFrame) -> dict[str, Any]:
    train_idx = split.loc[split["split"] == "train", "sample_index"].to_numpy(dtype=int)
    val_idx = split.loc[split["split"] == "val", "sample_index"].to_numpy(dtype=int)
    y_train = data["phenophase_doy"][train_idx].astype(float)
    y_val = data["phenophase_doy"][val_idx].astype(float)
    means = np.where(y_train > 0, y_train, np.nan)
    means = np.nanmean(means, axis=0)
    abs_err = np.abs(y_val - means[None, :])
    abs_err[y_val <= 0] = np.nan
    per_stage = np.nanmean(abs_err, axis=0)
    names = data["phenophase_names"].astype(str).tolist()
    return {
        "constant_train_mean_phenophase_mae_days_per_stage": {names[i]: float(per_stage[i]) for i in range(len(names))},
        "constant_train_mean_phenophase_mae_days_mean": float(np.nanmean(per_stage)),
    }


def observed_valid_ratio_per_sample(data: dict[str, np.ndarray]) -> np.ndarray:
    patch_size = int(data["patch_size"])
    observed_pixels = data["band_mask"].sum(axis=(1, 2)) * patch_size * patch_size
    valid_pixels = data["valid_pixel_mask"].sum(axis=(1, 2, 3, 4))
    return np.divide(valid_pixels, np.maximum(observed_pixels, 1))


def valid_ratio_crop_baseline(data: dict[str, np.ndarray], split: pd.DataFrame) -> dict[str, Any]:
    valid_ratio = observed_valid_ratio_per_sample(data)
    crop_id = data["crop_type_id"].astype(int)
    train_idx = split.loc[split["split"] == "train", "sample_index"].to_numpy(dtype=int)
    val_idx = split.loc[split["split"] == "val", "sample_index"].to_numpy(dtype=int)
    crop_names = data["crop_type_names"].astype(str).tolist()
    train_means = {crop: float(valid_ratio[train_idx][crop_id[train_idx] == crop].mean()) for crop in sorted(set(crop_id.tolist()))}
    labels = np.array(sorted(train_means.keys()), dtype=int)
    means = np.array([train_means[int(label)] for label in labels], dtype=float)
    pred = labels[np.abs(valid_ratio[val_idx, None] - means[None, :]).argmin(axis=1)]
    true = crop_id[val_idx]
    return {
        "train_mean_valid_ratio_by_crop": {crop_names[int(k)]: float(v) for k, v in train_means.items()},
        "valid_ratio_only_crop_accuracy": float((pred == true).mean()) if len(true) else 0.0,
        "valid_ratio_only_crop_macro_f1": macro_f1(true, pred, labels.tolist()) if len(true) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transparent data and leakage audit for the current dataset.")
    parser.add_argument("--dataset-npz", type=Path, default=ROOT / "artifacts" / "patches_clean" / "train_cnn_transformer_15x15.npz")
    parser.add_argument("--split-csv", type=Path, default=ROOT / "artifacts" / "splits" / "train_val_split.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "audit" / "data_audit.json")
    args = parser.parse_args()

    with np.load(resolve_path(args.dataset_npz), allow_pickle=False) as npz:
        data = {name: npz[name] for name in npz.files}
    split = pd.read_csv(resolve_path(args.split_csv))
    rows = query_rows(data, split)
    patch_size = int(data["patch_size"])
    observed_band_cells = int(data["band_mask"].sum())
    observed_pixels = int(observed_band_cells * patch_size * patch_size)
    valid_pixels = int(data["valid_pixel_mask"].sum())
    total_array_pixels = int(data["valid_pixel_mask"].size)
    padded_pixels = int(total_array_pixels - observed_pixels)
    observed_valid_ratio = valid_pixels / max(observed_pixels, 1)
    all_array_valid_ratio = valid_pixels / max(total_array_pixels, 1)
    per_sample_valid_ratio = observed_valid_ratio_per_sample(data)
    observed_band_valid_ratios = data["band_valid_ratio"][data["band_mask"]]

    train = split[split["split"] == "train"]
    val = split[split["split"] == "val"]
    warnings: list[str] = []
    date_baseline = date_only_stage_baseline(rows, len(data["phenophase_names"]))
    if date_baseline["date_only_rice_stage_accuracy"] >= 0.95:
        warnings.append("Rice stage is almost fully predictable from query date on this split; do not interpret near-zero stage loss as image-based phenology learning.")
    if observed_valid_ratio < 0.90:
        warnings.append("More than 10% of observed patch pixels are invalid; missingness may become a shortcut unless monitored.")
    if len(set(train["resolved_region_id"]) & set(val["resolved_region_id"])) == 0:
        warnings.append("No train/val region overlap was found; the suspicious stage result is not from region overlap.")

    report = {
        "dataset_npz": str(resolve_path(args.dataset_npz)),
        "patches_shape": list(data["patches"].shape),
        "patch_value_dtype": str(data["patches"].dtype),
        "observed_valid_pixel_ratio_excluding_padding": float(observed_valid_ratio),
        "observed_invalid_pixel_ratio_excluding_padding": float(1.0 - observed_valid_ratio),
        "all_array_valid_ratio_including_padding": float(all_array_valid_ratio),
        "padding_pixel_ratio_of_array": float(padded_pixels / max(total_array_pixels, 1)),
        "observed_band_cells": observed_band_cells,
        "observed_pixels": observed_pixels,
        "valid_pixels": valid_pixels,
        "invalid_observed_pixels": int(observed_pixels - valid_pixels),
        "time_steps_min": int(data["time_mask"].sum(axis=1).min()),
        "time_steps_max": int(data["time_mask"].sum(axis=1).max()),
        "sample_count": int(data["patches"].shape[0]),
        "query_row_count_after_expanding_7_stages": int(len(rows)),
        "train_samples": int(len(train)),
        "val_samples": int(len(val)),
        "train_queries": int((rows["split"] == "train").sum()),
        "val_queries": int((rows["split"] == "val").sum()),
        "train_val_sample_overlap": int(len(set(train["sample_index"]) & set(val["sample_index"]))),
        "train_val_point_overlap": int(len(set(train["point_id"]) & set(val["point_id"]))),
        "train_val_region_overlap": int(len(set(train["resolved_region_id"]) & set(val["resolved_region_id"]))),
        "class_counts_by_split": pd.crosstab(split["split"], split["crop_type"]).to_dict(),
        "observed_valid_pixel_ratio_by_crop": split.assign(valid_ratio=per_sample_valid_ratio).groupby(["split", "crop_type"])["valid_ratio"].mean().unstack(0).to_dict(),
        "observed_band_patch_valid_ratio_distribution": {
            "full_invalid_band_patches": int((observed_band_valid_ratios == 0).sum()),
            "perfect_band_patches": int((observed_band_valid_ratios == 1).sum()),
            "partial_invalid_band_patches": int(((observed_band_valid_ratios > 0) & (observed_band_valid_ratios < 1)).sum()),
        },
        "sample_valid_ratio_distribution": {
            "min": float(per_sample_valid_ratio.min()),
            "median": float(np.median(per_sample_valid_ratio)),
            "mean": float(per_sample_valid_ratio.mean()),
            "samples_below_0_99": int((per_sample_valid_ratio < 0.99).sum()),
            "samples_below_0_95": int((per_sample_valid_ratio < 0.95).sum()),
            "samples_below_0_90": int((per_sample_valid_ratio < 0.90).sum()),
        },
        "date_only_stage_baseline": date_baseline,
        "constant_doy_regression_baseline": constant_doy_mae(data, split),
        "valid_ratio_only_crop_baseline": valid_ratio_crop_baseline(data, split),
        "warnings": warnings,
        "interpretation": [
            "Crop labels are repeated across seven query rows per point, so crop metrics should be interpreted at point level, not as seven independent examples.",
            "Stage labels are built from the same phenophase dates used as query dates; this is why date-only stage prediction is extremely strong.",
            "Do not use all_array_valid_ratio_including_padding as a data-quality metric; padded timesteps are intentionally false in the mask.",
            "Most observed invalid pixels come from complete zero/nodata band patches, not from random single-pixel corruption.",
            "The current dataset is small enough for a 3M-parameter CNN+Transformer to memorize. Use no-date ablation and simpler baselines before claiming generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
