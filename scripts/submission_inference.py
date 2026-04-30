from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aux_features import compute_aux_features
from data.transforms import NpzPatchNormalizer
from models.model_factory import build_model, build_model_config, normalize_model_type
from preprocessing.constants import BAND_ORDER, INVALID_FILL_VALUE, PATCH_SIZE
from preprocessing.dataset import build_patch_dataset
from preprocessing.raster_io import rasterio
from training.stage_decoding import maybe_decode_stages

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for submission inference.") from exc

CROP_TYPE_NAMES = ["corn", "rice", "soybean"]
PHENOPHASE_NAMES = ["Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence", "Dormancy"]

# Every point in training data (all 778, all crops) has exactly this DOY ordering.
# Sorted by ascending query_doy: Greenup(0) < MidGreenup(1) < Maturity(3) < Peak(2) <
# Senescence(5) < MidSenescence(4) < Dormancy(6).
# Verified: unique_orderings=1 across 778 training points (0 exceptions).
_DOY_STAGE_ORDER = [0, 1, 3, 2, 5, 4, 6]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def find_points_csv(input_root: Path) -> Path:
    for name in ["test_point.csv", "points_test.csv"]:
        candidate = input_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected {input_root}/test_point.csv or {input_root}/points_test.csv")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing required column. Tried {candidates}. Available columns: {list(df.columns)}")


def read_query_rows(points_csv: Path, npz_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    raw = pd.read_csv(points_csv, dtype=str).reset_index(drop=False).rename(columns={"index": "query_row_index"})
    lon_col = find_column(raw, ["Longitude", "Longtitude", "longitude", "longtitude", "lon", "lng"])
    lat_col = find_column(raw, ["Latitude", "latitude", "lat"])
    date_col = find_column(raw, ["phenophase_date", "Date", "date", "query_date", "datetime", "time"])
    try:
        point_id_col = find_column(raw, ["point_id", "pointid", "id"])
    except ValueError:
        raw["point_id"] = np.arange(1, len(raw) + 1, dtype=np.int64).astype(str)
        point_id_col = "point_id"

    sample_by_point = {int(point_id): sample_index for sample_index, point_id in enumerate(npz_arrays["point_id"])}
    rows: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        row_dict = row._asdict()
        point_id = int(float(row_dict[point_id_col]))
        if point_id not in sample_by_point:
            raise ValueError(f"point_id {point_id} was not extracted into the test patch NPZ")
        date_text = str(row_dict[date_col]).strip()
        parsed = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Could not parse query date {date_text!r} for point_id {point_id}")
        rows.append(
            {
                "sample_index": int(sample_by_point[point_id]),
                "point_id": point_id,
                "longitude_key": str(row_dict[lon_col]).strip(),
                "latitude_key": str(row_dict[lat_col]).strip(),
                "date_key": date_text,
                "query_doy": int(parsed.dayofyear),
            }
        )
    return pd.DataFrame(rows)


def apply_point_stage_bijection(stage_pred: np.ndarray, query_rows: pd.DataFrame) -> np.ndarray:
    """Assign stages to all query rows of each point by DOY rank.

    For a point with N queries (1 ≤ N ≤ 7), sort the N query dates by ascending
    DOY and assign _DOY_STAGE_ORDER[rank] to rank. This gives the first N stages
    in biological DOY order (Greenup → MidGreenup → Maturity → Peak → Senescence
    → MidSenescence → Dormancy), which is the only ordering observed in all 778
    training points (verified: 0 exceptions).

    Test data has 5–6 queries per point (not 7), so Dormancy and sometimes
    MidSenescence are absent — the bijection handles this correctly by assigning
    only the first N stages.
    """
    result = stage_pred.copy()
    rows_reset = query_rows.reset_index(drop=True)
    for _point_id, group in rows_reset.groupby("point_id"):
        n = len(group)
        if n < 1 or n > 7:
            continue
        doy_argsort = group["query_doy"].argsort().values
        for rank, pos_in_group in enumerate(doy_argsort):
            full_idx = group.index[pos_in_group]
            result[full_idx] = _DOY_STAGE_ORDER[rank]
    return result


def apply_crop_consistency(crop_logits: np.ndarray, query_rows: pd.DataFrame) -> np.ndarray:
    """Force the same crop prediction for all query rows of the same point.

    Sums logits across all query rows for each point and takes argmax.
    This prevents the model from predicting different crops for different
    query dates of the same spatial location (a semantic impossibility).
    """
    result = np.empty(len(query_rows), dtype=np.int64)
    rows_reset = query_rows.reset_index(drop=True)
    for _point_id, group in rows_reset.groupby("point_id"):
        group_indices = group.index.tolist()
        summed_logits = crop_logits[group_indices].sum(axis=0)
        majority_crop = int(summed_logits.argmax())
        for idx in group_indices:
            result[idx] = majority_crop
    return result


def apply_output_key_consistency(
    crop_pred: np.ndarray,
    stage_pred: np.ndarray,
    query_rows: pd.DataFrame,
    crop_logits: np.ndarray | None = None,
    stage_logits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Force identical predictions for rows that collapse to the same output key.

    The evaluation output is keyed by longitude/latitude/date, so duplicate rows
    must resolve to one label pair. This removes arbitrary "keep first row"
    behavior when overlapping source regions produce the same logical query key.
    """

    crop_out = crop_pred.copy()
    stage_out = stage_pred.copy()
    rows_reset = query_rows.reset_index(drop=True)
    key_to_indices: dict[str, list[int]] = {}
    for idx, row in rows_reset.iterrows():
        key = f"{row['longitude_key']}_{row['latitude_key']}_{row['date_key']}"
        key_to_indices.setdefault(key, []).append(int(idx))

    for indices in key_to_indices.values():
        if len(indices) <= 1:
            continue

        crop_vote = np.bincount(crop_out[indices], minlength=len(CROP_TYPE_NAMES))
        crop_best = np.flatnonzero(crop_vote == crop_vote.max())
        if len(crop_best) == 1 or crop_logits is None:
            crop_label = int(crop_best[0])
        else:
            crop_label = int(crop_best[np.argmax(crop_logits[indices][:, crop_best].sum(axis=0))])

        stage_vote = np.bincount(stage_out[indices], minlength=len(PHENOPHASE_NAMES))
        stage_best = np.flatnonzero(stage_vote == stage_vote.max())
        if len(stage_best) == 1 or stage_logits is None:
            stage_label = int(stage_best[0])
        else:
            stage_label = int(stage_best[np.argmax(stage_logits[indices][:, stage_best].sum(axis=0))])

        crop_out[indices] = crop_label
        stage_out[indices] = stage_label

    return crop_out, stage_out


def load_model(checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}. Train in Colab and place it at checkpoints/model.pt before submitting.")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    config = build_model_config(model_type, payload["model_config"])
    model = build_model(model_type, config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    train_config = payload.get("train_config", {})
    model.model_type = model_type
    model.aux_feature_set = str(payload.get("aux_feature_set") or train_config.get("aux_feature_set", "summary"))
    model.use_relative_doy = bool(train_config.get("use_relative_doy", False))
    model.to(device)
    model.eval()
    return model


def prepare_patches(arrays: dict[str, np.ndarray], indices: np.ndarray, normalizer: NpzPatchNormalizer, include_mask_channels: bool) -> np.ndarray:
    valid_mask = arrays["valid_pixel_mask"][indices].astype(bool)
    patches = normalizer(arrays["patches"][indices], valid_mask)
    if include_mask_channels:
        patches = np.concatenate([patches, valid_mask.astype(np.float32)], axis=2)
    return patches.astype(np.float32, copy=False)


def _apply_relative_doy(
    time_doy_batch: np.ndarray,
    time_mask_batch: np.ndarray,
    query_doys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-sample series-center DOY from time_doy and query_doy."""
    time_doy_out = time_doy_batch.copy()
    query_doys_out = query_doys.copy()
    for i in range(len(time_doy_batch)):
        valid = time_mask_batch[i].astype(bool)
        valid_doys = time_doy_batch[i][valid]
        series_center = float(valid_doys.mean()) if len(valid_doys) > 0 else 183.0
        time_doy_out[i][valid] -= series_center
        query_doys_out[i] -= series_center
    return time_doy_out, query_doys_out


def prepare_model_batch(
    *,
    model,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    query_doys: np.ndarray,
    normalizer: NpzPatchNormalizer,
    device: torch.device,
    use_relative_doy: bool = False,
) -> dict[str, torch.Tensor]:
    include_mask_channels = int(model.config.in_channels) == 24
    patches = prepare_patches(arrays, indices, normalizer, include_mask_channels)
    time_doy_np = arrays["time_doy"][indices].astype(np.float32)
    query_doys_np = query_doys.astype(np.float32, copy=False)
    if use_relative_doy:
        time_doy_np, query_doys_np = _apply_relative_doy(
            time_doy_np, arrays["time_mask"][indices], query_doys_np
        )
    batch = {
        "patches": torch.from_numpy(patches).to(device),
        "time_mask": torch.from_numpy(arrays["time_mask"][indices].astype(bool)).to(device),
        "time_doy": torch.from_numpy(time_doy_np).to(device),
        "query_doy": torch.from_numpy(query_doys_np).to(device),
    }
    if int(model.config.aux_feature_dim) > 0:
        bands = arrays.get("bands", np.asarray(BAND_ORDER)).astype(str).tolist()
        aux = np.stack(
            [
                compute_aux_features(
                    arrays["patches"][sample_index],
                    arrays["valid_pixel_mask"][sample_index],
                    arrays["time_mask"][sample_index],
                    arrays["time_doy"][sample_index],
                    float(query_doy),
                    bands,
                    feature_set=str(getattr(model, "aux_feature_set", "summary")),
                )
                for sample_index, query_doy in zip(indices, query_doys)
            ]
        )
        if aux.shape[1] != int(model.config.aux_feature_dim):
            raise ValueError(f"aux feature dimension mismatch: model expects {model.config.aux_feature_dim}, computed {aux.shape[1]}")
        batch["aux_features"] = torch.from_numpy(aux.astype(np.float32, copy=False)).to(device)
    return batch


def forward_model(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"], batch.get("aux_features"))


def _count_names(values: np.ndarray, names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in names}
    for value in values:
        counts[names[int(value)]] += 1
    return counts


def write_result(query_rows: pd.DataFrame, crop_pred: np.ndarray, stage_pred: np.ndarray, output_json: Path) -> dict[str, Any]:
    result: dict[str, list[str]] = {}
    duplicate_rows = 0
    duplicate_conflicts = 0
    duplicate_examples: list[dict[str, Any]] = []
    for i, row in query_rows.reset_index(drop=True).iterrows():
        crop = CROP_TYPE_NAMES[int(crop_pred[i])]
        stage = PHENOPHASE_NAMES[int(stage_pred[i])]
        key = f"{row['longitude_key']}_{row['latitude_key']}_{row['date_key']}"
        value = [crop, stage]
        if key in result:
            duplicate_rows += 1
            if result[key] != value:
                duplicate_conflicts += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append({"key": key, "kept": result[key], "ignored": value})
            # JSON cannot contain duplicate keys. Keep the first deterministic value.
            continue
        result[key] = value
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return {
        "query_rows": int(len(query_rows)),
        "unique_output_keys": int(len(result)),
        "duplicate_output_key_rows": int(duplicate_rows),
        "duplicate_output_key_conflicts": int(duplicate_conflicts),
        "duplicate_output_key_examples": duplicate_examples,
        "crop_prediction_counts": _count_names(crop_pred, CROP_TYPE_NAMES),
        "stage_prediction_counts": _count_names(stage_pred, PHENOPHASE_NAMES),
    }


def _gdal_env_options(config: dict[str, Any]) -> dict[str, Any]:
    options = dict(config.get("gdal_env", {}))
    normalized: dict[str, Any] = {}
    for key, value in options.items():
        if isinstance(value, bool):
            normalized[str(key)] = "TRUE" if value else "FALSE"
        else:
            normalized[str(key)] = value
    return normalized


def _patch_dataset_context(config: dict[str, Any]):
    if not bool(config.get("use_gdal_env", False)):
        return nullcontext()
    return rasterio.Env(**_gdal_env_options(config))


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "/input"))
    points_csv = find_points_csv(input_root)
    tiff_dir = input_root / "region_test"
    output_json = Path(config.get("output_json", "/output/result.json"))
    work_dir = Path(config.get("work_dir", "/workspace/submission_work"))
    test_npz = work_dir / "test_patches.npz"
    patch_cfg = config.get("preprocessing", {})

    work_dir.mkdir(parents=True, exist_ok=True)
    with _patch_dataset_context(config):
        report = build_patch_dataset(
            points_csv=points_csv,
            tiff_dirs=[tiff_dir],
            output_npz=test_npz,
            output_dir=work_dir,
            root=input_root,
            mode="test",
            patch_size=int(patch_cfg.get("patch_size", PATCH_SIZE)),
            allow_nearest_fallback=True,
            band_order=list(patch_cfg.get("bands", BAND_ORDER)),
            valid_min_exclusive=float(patch_cfg.get("valid_min_exclusive", 0.0)),
            valid_max_inclusive=float(patch_cfg.get("valid_max_inclusive", 2.0)),
            invalid_fill_value=float(patch_cfg.get("invalid_fill_value", INVALID_FILL_VALUE)),
            write_reports=False,
            skip_bands=list(config.get("skip_bands", patch_cfg.get("skip_bands", []))),
            batch_raster_reads=bool(config.get("batch_raster_reads", False)),
            max_batch_union_pixels=int(config.get("max_batch_union_pixels", 262144)),
            max_batch_union_overread_ratio=float(config.get("max_batch_union_overread_ratio", 6.0)),
            block_raster_reads=bool(config.get("block_raster_reads", False)),
            max_block_pixels=int(config.get("max_block_pixels", 1048576)),
            max_block_overread_ratio=float(config.get("max_block_overread_ratio", 12.0)),
        )

    device = select_device(str(config.get("device", "auto")))
    default_checkpoint = resolve_path(config.get("checkpoint", "checkpoints/model.pt"))
    crop_checkpoint = resolve_path(config.get("crop_checkpoint", default_checkpoint))
    stage_checkpoint = resolve_path(config.get("stage_checkpoint", default_checkpoint))
    crop_model = load_model(crop_checkpoint, device)
    if stage_checkpoint.resolve() == crop_checkpoint.resolve():
        stage_model = crop_model
    else:
        stage_model = load_model(stage_checkpoint, device)

    # Optional ensemble: extra checkpoints whose logits are averaged in.
    ensemble_paths = [resolve_path(p) for p in config.get("ensemble_checkpoints", [])]
    ensemble_models: list = []
    for ep in ensemble_paths:
        if ep.resolve() != crop_checkpoint.resolve():
            ensemble_models.append(load_model(ep, device))

    normalizer = NpzPatchNormalizer(resolve_path(config.get("normalization_json", "artifacts/normalization/train_patch_band_stats.json")))
    with np.load(test_npz, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    query_rows = read_query_rows(points_csv, arrays)

    batch_size = int(config.get("batch_size", 64))
    crop_logit_chunks: list[np.ndarray] = []
    stage_logit_chunks: list[np.ndarray] = []
    sample_indices = query_rows["sample_index"].to_numpy(dtype=np.int64)
    query_doys = query_rows["query_doy"].to_numpy(dtype=np.float32)
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        indices = sample_indices[start:end]
        batch_doys = query_doys[start:end]
        crop_batch = prepare_model_batch(
            model=crop_model,
            arrays=arrays,
            indices=indices,
            query_doys=batch_doys,
            normalizer=normalizer,
            device=device,
            use_relative_doy=bool(getattr(crop_model, "use_relative_doy", False)),
        )
        with torch.no_grad():
            crop_outputs = forward_model(crop_model, crop_batch)
            if stage_model is crop_model:
                stage_outputs = crop_outputs
            else:
                stage_batch = prepare_model_batch(
                    model=stage_model,
                    arrays=arrays,
                    indices=indices,
                    query_doys=batch_doys,
                    normalizer=normalizer,
                    device=device,
                    use_relative_doy=bool(getattr(stage_model, "use_relative_doy", False)),
                )
                stage_outputs = forward_model(stage_model, stage_batch)
            crop_logits_np = crop_outputs["crop_logits"].detach().cpu().numpy()
            stage_logits_np = stage_outputs["stage_logits"].detach().cpu().numpy()
            # Ensemble: average softmax over additional models
            for em in ensemble_models:
                em_batch = prepare_model_batch(
                    model=em,
                    arrays=arrays,
                    indices=indices,
                    query_doys=batch_doys,
                    normalizer=normalizer,
                    device=device,
                    use_relative_doy=bool(getattr(em, "use_relative_doy", False)),
                )
                em_out = forward_model(em, em_batch)
                crop_logits_np = crop_logits_np + em_out["crop_logits"].detach().cpu().numpy()
                stage_logits_np = stage_logits_np + em_out["stage_logits"].detach().cpu().numpy()
        crop_logit_chunks.append(crop_logits_np)
        stage_logit_chunks.append(stage_logits_np)

    crop_logits_all = np.concatenate(crop_logit_chunks, axis=0)
    # Crop type should be point-consistent regardless of stage decoding policy.
    use_crop_consistency = bool(config.get("use_crop_consistency", True))
    if use_crop_consistency:
        crop_pred = apply_crop_consistency(crop_logits_all, query_rows)
    else:
        crop_pred = crop_logits_all.argmax(axis=1)
    stage_logits = np.concatenate(stage_logit_chunks, axis=0)
    stage_postprocess = str(config.get("stage_postprocess", "none"))
    use_bijection = bool(config.get("use_point_stage_bijection", False))
    stage_pred = maybe_decode_stages(
        torch.from_numpy(stage_logits),
        torch.from_numpy(query_rows["point_id"].to_numpy(dtype=np.int64)),
        torch.from_numpy(query_rows["query_doy"].to_numpy(dtype=np.float32)),
        mode=stage_postprocess,
    ).numpy()
    # Override with point-level DOY bijection: sort all N queries for each point
    # by DOY and assign the first N stages in biological order. Works for any N.
    if use_bijection:
        stage_pred = apply_point_stage_bijection(stage_pred, query_rows)
    crop_pred, stage_pred = apply_output_key_consistency(
        crop_pred,
        stage_pred,
        query_rows,
        crop_logits=crop_logits_all,
        stage_logits=stage_logits,
    )
    result_stats = write_result(query_rows, crop_pred, stage_pred, output_json)
    return {
        "output_json": str(output_json),
        "queries": int(len(query_rows)),
        "unique_output_keys": result_stats["unique_output_keys"],
        "duplicate_output_key_rows": result_stats["duplicate_output_key_rows"],
        "duplicate_output_key_conflicts": result_stats["duplicate_output_key_conflicts"],
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "crop_checkpoint": str(crop_checkpoint),
        "stage_checkpoint": str(stage_checkpoint),
        "ensemble_checkpoints": [str(p) for p in ensemble_paths],
        "use_gdal_env": bool(config.get("use_gdal_env", False)),
        "gdal_env": _gdal_env_options(config) if bool(config.get("use_gdal_env", False)) else {},
        "skip_bands": list(config.get("skip_bands", patch_cfg.get("skip_bands", []))),
        "batch_raster_reads": bool(config.get("batch_raster_reads", False)),
        "block_raster_reads": bool(config.get("block_raster_reads", False)),
        "use_crop_consistency": bool(use_crop_consistency),
        "use_point_stage_bijection": bool(use_bijection),
        "crop_model_uses_mask_channels": bool(int(crop_model.config.in_channels) == 24),
        "stage_model_uses_mask_channels": bool(int(stage_model.config.in_channels) == 24),
        "crop_model_uses_aux_features": bool(getattr(crop_model, "crop_aux_proj", None) is not None),
        "stage_model_uses_aux_features": bool(getattr(stage_model, "stage_aux_proj", None) is not None),
        "crop_model_type": str(getattr(crop_model, "model_type", "query_cnn_transformer")),
        "stage_model_type": str(getattr(stage_model, "model_type", "query_cnn_transformer")),
        "crop_model_uses_relative_doy": bool(getattr(crop_model, "use_relative_doy", False)),
        "stage_model_uses_relative_doy": bool(getattr(stage_model, "use_relative_doy", False)),
        "model_uses_mask_channels": bool(int(crop_model.config.in_channels) == 24 or int(stage_model.config.in_channels) == 24),
        "model_uses_aux_features": bool(
            getattr(crop_model, "crop_aux_proj", None) is not None
            or getattr(crop_model, "stage_aux_proj", None) is not None
            or getattr(stage_model, "crop_aux_proj", None) is not None
            or getattr(stage_model, "stage_aux_proj", None) is not None
        ),
        "stage_postprocess": stage_postprocess,
        "result_stats": result_stats,
        "patch_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 submission inference: write /output/result.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "submission.json")
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    print(json.dumps(run_inference(config), indent=2))


if __name__ == "__main__":
    main()
