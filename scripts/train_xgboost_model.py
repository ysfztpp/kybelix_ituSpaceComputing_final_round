from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.aux_features import compute_aux_features


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: range) -> float:
    scores: list[float] = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(np.mean(scores))


def band_time_medians(patches: np.ndarray, valid_mask: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    timesteps, bands = patches.shape[:2]
    out = np.zeros((timesteps, bands), dtype=np.float32)
    for t in range(timesteps):
        if not bool(time_mask[t]):
            continue
        for b in range(bands):
            valid = valid_mask[t, b].astype(bool)
            if valid.any():
                out[t, b] = float(np.median(patches[t, b][valid]))
    return out


def build_rows(config: dict[str, Any], split: str | None, limit_rows: int | None = None):
    with np.load(resolve_path(config["dataset_npz"]), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    bands = arrays.get("bands", np.asarray([f"B{i:02d}" for i in range(1, arrays["patches"].shape[2] + 1)])).astype(str).tolist()
    disabled = {str(band) for band in config.get("disabled_bands", [])}
    enabled_band_indices = [i for i, band in enumerate(bands) if band not in disabled]

    sample_indices = np.arange(arrays["patches"].shape[0], dtype=np.int64)
    if split and config.get("split_csv"):
        split_df = pd.read_csv(resolve_path(config["split_csv"]))
        sample_indices = split_df.loc[split_df["split"] == split, "sample_index"].to_numpy(dtype=np.int64)

    feature_rows: list[np.ndarray] = []
    crop_labels: list[int] = []
    stage_labels: list[int] = []
    stage_weights: list[float] = []
    feature_set = str(config.get("aux_feature_set", "phenology"))
    max_timesteps = int(config.get("max_timesteps", arrays["patches"].shape[1]))
    rice_id = 1
    names = arrays.get("crop_type_names")
    if names is not None:
        names_list = names.astype(str).tolist()
        rice_id = names_list.index("rice") if "rice" in names_list else 1

    for sample_index in sample_indices:
        patches = arrays["patches"][sample_index].astype(np.float32, copy=False)
        valid = arrays["valid_pixel_mask"][sample_index].astype(bool, copy=False)
        time_mask = arrays["time_mask"][sample_index].astype(bool, copy=False)
        time_doy = arrays["time_doy"][sample_index].astype(np.float32, copy=False)
        medians = band_time_medians(patches, valid, time_mask)[:max_timesteps, enabled_band_indices]
        valid_ratio = valid[:max_timesteps, enabled_band_indices].mean(axis=(2, 3)).astype(np.float32)
        time_features = np.concatenate(
            [
                medians.reshape(-1),
                valid_ratio.reshape(-1),
                (time_doy[:max_timesteps] / 366.0).astype(np.float32),
                time_mask[:max_timesteps].astype(np.float32),
            ]
        )
        crop_id = int(arrays["crop_type_id"][sample_index])
        for stage_index, query_doy in enumerate(arrays["phenophase_doy"][sample_index].astype(np.int16)):
            if query_doy <= 0:
                continue
            aux = compute_aux_features(patches, valid, time_mask, time_doy, float(query_doy), bands, feature_set=feature_set)
            feature_rows.append(np.concatenate([time_features, aux.astype(np.float32, copy=False), np.asarray([float(query_doy) / 366.0], dtype=np.float32)]))
            crop_labels.append(crop_id)
            stage_labels.append(int(stage_index))
            stage_weights.append(1.0 if crop_id == rice_id or not bool(config.get("rice_stage_loss_only", True)) else 0.0)
            if limit_rows is not None and len(feature_rows) >= limit_rows:
                return arrays, np.stack(feature_rows), np.asarray(crop_labels), np.asarray(stage_labels), np.asarray(stage_weights)
    return arrays, np.stack(feature_rows), np.asarray(crop_labels), np.asarray(stage_labels), np.asarray(stage_weights)


def xgb_classifier(config: dict[str, Any], num_classes: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost is not installed in this venv. Install it with `.venv/bin/python -m pip install xgboost`.") from exc

    params = dict(config.get("xgboost", {}))
    params.setdefault("objective", "multi:softprob")
    params.setdefault("num_class", int(num_classes))
    params.setdefault("tree_method", "hist")
    params.setdefault("device", "cuda")
    params.setdefault("n_estimators", 3000)
    params.setdefault("learning_rate", 0.025)
    params.setdefault("max_depth", 6)
    params.setdefault("min_child_weight", 2.0)
    params.setdefault("subsample", 0.9)
    params.setdefault("colsample_bytree", 0.85)
    params.setdefault("reg_lambda", 4.0)
    params.setdefault("reg_alpha", 0.05)
    params.setdefault("eval_metric", "mlogloss")
    params.setdefault("random_state", int(config.get("seed", 42)))
    params.setdefault("n_jobs", int(config.get("n_jobs", 16)))
    params.setdefault("early_stopping_rounds", int(config.get("early_stopping_rounds", 150)))
    return XGBClassifier(**params)


def evaluate(crop_model, stage_model, x_val: np.ndarray, crop_val: np.ndarray, stage_val: np.ndarray, stage_weight_val: np.ndarray) -> dict[str, float]:
    crop_pred = crop_model.predict(x_val)
    stage_pred = stage_model.predict(x_val)
    rice_mask = stage_weight_val > 0.0
    return {
        "crop_macro_f1": macro_f1(crop_val, crop_pred, range(3)),
        "crop_accuracy": float(np.mean(crop_val == crop_pred)),
        "rice_stage_macro_f1": macro_f1(stage_val[rice_mask], stage_pred[rice_mask], range(7)) if rice_mask.any() else 0.0,
        "rice_stage_accuracy": float(np.mean(stage_val[rice_mask] == stage_pred[rice_mask])) if rice_mask.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic heavy XGBoost crop + stage models.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_xgboost_ultimate_a100.json")
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke-test feature building on a subset.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(resolve_path(args.config).read_text())
    arrays, x_train, crop_train, stage_train, stage_weight_train = build_rows(config, "train", args.limit_rows)
    _, x_val, crop_val, stage_val, stage_weight_val = build_rows(config, "val", args.limit_rows)
    if args.dry_run:
        print(json.dumps({"x_train": list(x_train.shape), "x_val": list(x_val.shape), "feature_dtype": str(x_train.dtype)}, indent=2))
        return

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_model = xgb_classifier(config, 3)
    stage_model = xgb_classifier(config, 7)
    crop_model.fit(x_train, crop_train, eval_set=[(x_val, crop_val)], verbose=bool(config.get("verbose", True)))
    train_stage_mask = stage_weight_train > 0.0
    val_stage_mask = stage_weight_val > 0.0
    stage_model.fit(x_train[train_stage_mask], stage_train[train_stage_mask], eval_set=[(x_val[val_stage_mask], stage_val[val_stage_mask])], verbose=bool(config.get("verbose", True)))

    metrics = evaluate(crop_model, stage_model, x_val, crop_val, stage_val, stage_weight_val)
    metrics["competition_score"] = 0.4 * metrics["crop_macro_f1"] + 0.6 * metrics["rice_stage_macro_f1"]
    report = {
        "config": config,
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "bands": arrays.get("bands", np.asarray([])).astype(str).tolist(),
    }
    with (output_dir / "xgboost_models.pkl").open("wb") as f:
        pickle.dump({"crop_model": crop_model, "stage_model": stage_model, "report": report}, f)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
