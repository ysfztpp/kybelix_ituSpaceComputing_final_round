from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit split leakage and simple label baselines.")
    parser.add_argument("--dataset-npz", type=Path, default=Path("artifacts/patches_clean/train_cnn_transformer_15x15.npz"))
    parser.add_argument("--split-csv", type=Path, default=Path("artifacts/splits/train_val_split.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.read_csv(args.split_csv)
    data = np.load(args.dataset_npz, allow_pickle=False)
    train = split[split["split"] == "train"]
    val = split[split["split"] == "val"]
    train_idx = train["sample_index"].to_numpy(dtype=np.int64)
    val_idx = val["sample_index"].to_numpy(dtype=np.int64)

    phenophase = data["phenophase_doy"].astype(float)
    train_mean = phenophase[train_idx].mean(axis=0)
    val_mae = np.abs(phenophase[val_idx] - train_mean).mean(axis=0)

    report = {
        "train_count": int(len(train)),
        "val_count": int(len(val)),
        "train_val_sample_overlap": int(len(set(train["sample_index"]) & set(val["sample_index"]))),
        "train_val_point_overlap": int(len(set(train["point_id"]) & set(val["point_id"]))),
        "train_val_region_overlap": int(len(set(train["resolved_region_id"]) & set(val["resolved_region_id"]))),
        "train_regions": int(train["resolved_region_id"].nunique()),
        "val_regions": int(val["resolved_region_id"].nunique()),
        "class_counts_by_split": pd.crosstab(split["split"], split["crop_type"]).to_dict(),
        "constant_train_mean_phenophase_mae_days_per_phase": {
            str(name): float(value) for name, value in zip(data["phenophase_names"].astype(str).tolist(), val_mae)
        },
        "constant_train_mean_phenophase_mae_days_mean": float(val_mae.mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
