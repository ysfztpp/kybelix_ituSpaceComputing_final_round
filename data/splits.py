from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _flat_split(metadata: pd.DataFrame, val_fraction: float, seed: int, stratify_by: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    split_rows: list[pd.DataFrame] = []
    grouped = metadata.groupby(stratify_by, dropna=False, sort=True) if stratify_by in metadata.columns else [("all", metadata)]
    for _, group in grouped:
        group = group.copy().reset_index(drop=True)
        order = rng.permutation(len(group))
        val_count = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        group["split"] = "train"
        if val_count:
            group.loc[order[:val_count], "split"] = "val"
        split_rows.append(group)
    return pd.concat(split_rows, ignore_index=True)


def _group_split(metadata: pd.DataFrame, val_fraction: float, seed: int, group_by: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_sizes = metadata.groupby(group_by).size().reset_index(name="count")
    group_sizes = group_sizes.iloc[rng.permutation(len(group_sizes))].reset_index(drop=True)
    target_val = max(1, int(round(len(metadata) * val_fraction)))
    chosen_groups: list[str] = []
    running = 0
    for row in group_sizes.itertuples(index=False):
        chosen_groups.append(str(getattr(row, group_by)))
        running += int(row.count)
        if running >= target_val:
            break
    out = metadata.copy()
    out["split"] = np.where(out[group_by].astype(str).isin(chosen_groups), "val", "train")
    return out


def make_train_val_split(
    metadata_csv: Path,
    output_csv: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratify_by: str = "crop_type",
    group_by: str | None = None,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    metadata = pd.read_csv(metadata_csv)
    if "sample_index" not in metadata.columns:
        raise ValueError(f"{metadata_csv} must contain sample_index")

    if group_by and group_by in metadata.columns:
        split_df = _group_split(metadata, val_fraction, seed, group_by)
        split_mode = "grouped"
    else:
        split_df = _flat_split(metadata, val_fraction, seed, stratify_by)
        split_mode = "stratified" if stratify_by in metadata.columns else "random"
        group_by = None

    keep_columns = ["sample_index", "point_id", "resolved_region_id", "split"]
    for column in [stratify_by, group_by]:
        if column and column in split_df.columns and column not in keep_columns:
            keep_columns.append(column)
    split_df = split_df[keep_columns].sort_values("sample_index").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    summary = {
        "split_csv": str(output_csv),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "split_mode": split_mode,
        "stratify_by": stratify_by if stratify_by in metadata.columns else "none",
        "group_by": group_by or "none",
        "train_count": int((split_df["split"] == "train").sum()),
        "val_count": int((split_df["split"] == "val").sum()),
        "total_count": int(len(split_df)),
        "train_regions": int(split_df.loc[split_df["split"] == "train", "resolved_region_id"].nunique()),
        "val_regions": int(split_df.loc[split_df["split"] == "val", "resolved_region_id"].nunique()),
    }
    if stratify_by in split_df.columns:
        summary["class_counts"] = split_df.groupby(["split", stratify_by]).size().unstack(fill_value=0).to_dict()
    (output_csv.parent / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
