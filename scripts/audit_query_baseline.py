from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Date-only baseline audit for the PDF-aligned query classifier.")
    parser.add_argument("--dataset-npz", type=Path, default=Path("artifacts/patches_clean/train_cnn_transformer_15x15.npz"))
    parser.add_argument("--split-csv", type=Path, default=Path("artifacts/splits/train_val_split.csv"))
    return parser.parse_args()


def expand_queries(data: np.lib.npyio.NpzFile, split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    subset = split[split["split"] == split_name]
    for row in subset.itertuples(index=False):
        sample_index = int(row.sample_index)
        crop_id = int(data["crop_type_id"][sample_index])
        crop_name = str(data["crop_type_names"][crop_id])
        for stage_id, doy in enumerate(data["phenophase_doy"][sample_index].astype(int)):
            if doy <= 0:
                continue
            rows.append(
                {
                    "sample_index": sample_index,
                    "point_id": int(row.point_id),
                    "region_id": str(row.resolved_region_id),
                    "crop_id": crop_id,
                    "crop_type": crop_name,
                    "stage_id": int(stage_id),
                    "stage_name": str(data["phenophase_names"][stage_id]),
                    "query_doy": int(doy),
                }
            )
    return pd.DataFrame(rows)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    scores = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    split = pd.read_csv(args.split_csv)
    with np.load(args.dataset_npz, allow_pickle=False) as data:
        train = expand_queries(data, split, "train")
        val = expand_queries(data, split, "val")
        stage_names = data["phenophase_names"].astype(str).tolist()

    # Predict stage by nearest train mean day-of-year for each phenophase stage.
    stage_mean_doy = train.groupby("stage_id")["query_doy"].mean().sort_index()
    stage_ids = stage_mean_doy.index.to_numpy(dtype=int)
    stage_means = stage_mean_doy.to_numpy(dtype=float)
    distances = np.abs(val["query_doy"].to_numpy(dtype=float)[:, None] - stage_means[None, :])
    pred_stage = stage_ids[distances.argmin(axis=1)]

    rice_val = val[val["crop_type"] == "rice"].copy()
    rice_distances = np.abs(rice_val["query_doy"].to_numpy(dtype=float)[:, None] - stage_means[None, :])
    rice_pred_stage = stage_ids[rice_distances.argmin(axis=1)]

    # Crop baseline: majority class from train, included only to show crop cannot be solved by date-only here.
    majority_crop = int(train["crop_id"].mode().iloc[0])
    crop_pred = np.full(len(val), majority_crop, dtype=int)

    report = {
        "train_queries": int(len(train)),
        "val_queries": int(len(val)),
        "val_rice_queries": int(len(rice_val)),
        "stage_mean_doy_from_train": {stage_names[int(k)]: float(v) for k, v in stage_mean_doy.items()},
        "date_only_stage_accuracy_all_crops": float((pred_stage == val["stage_id"].to_numpy()).mean()),
        "date_only_stage_macro_f1_all_crops": macro_f1(val["stage_id"].to_numpy(), pred_stage, list(range(len(stage_names)))),
        "date_only_rice_stage_accuracy": float((rice_pred_stage == rice_val["stage_id"].to_numpy()).mean()) if len(rice_val) else 0.0,
        "date_only_rice_stage_macro_f1": macro_f1(rice_val["stage_id"].to_numpy(), rice_pred_stage, list(range(len(stage_names)))) if len(rice_val) else 0.0,
        "majority_crop_accuracy": float((crop_pred == val["crop_id"].to_numpy()).mean()),
        "majority_crop_macro_f1": macro_f1(val["crop_id"].to_numpy(), crop_pred, [0, 1, 2]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
