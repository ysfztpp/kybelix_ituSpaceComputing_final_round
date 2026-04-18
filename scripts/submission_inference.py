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

from data.transforms import NpzPatchNormalizer
from models.query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig
from preprocessing.constants import BAND_ORDER, INVALID_FILL_VALUE, PATCH_SIZE
from preprocessing.dataset import build_patch_dataset

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for submission inference.") from exc

CROP_TYPE_NAMES = ["corn", "rice", "soybean"]
PHENOPHASE_NAMES = ["Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence", "Dormancy"]


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


def load_model(checkpoint: Path, device: torch.device) -> QueryCNNTransformerClassifier:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}. Train in Colab and place it at checkpoints/model.pt before submitting.")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = QueryCNNTransformerConfig(**payload["model_config"])
    model = QueryCNNTransformerClassifier(config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_patches(arrays: dict[str, np.ndarray], indices: np.ndarray, normalizer: NpzPatchNormalizer, include_mask_channels: bool) -> np.ndarray:
    valid_mask = arrays["valid_pixel_mask"][indices].astype(bool)
    patches = normalizer(arrays["patches"][indices], valid_mask)
    if include_mask_channels:
        patches = np.concatenate([patches, valid_mask.astype(np.float32)], axis=2)
    return patches.astype(np.float32, copy=False)


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


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "/input"))
    points_csv = find_points_csv(input_root)
    tiff_dir = input_root / "region_test"
    output_json = Path(config.get("output_json", "/output/result.json"))
    work_dir = Path(config.get("work_dir", "/workspace/submission_work"))
    test_npz = work_dir / "test_patches.npz"
    patch_cfg = config.get("preprocessing", {})

    work_dir.mkdir(parents=True, exist_ok=True)
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
    )

    device = select_device(str(config.get("device", "auto")))
    model = load_model(resolve_path(config.get("checkpoint", "checkpoints/model.pt")), device)
    include_mask_channels = int(model.config.in_channels) == 24
    normalizer = NpzPatchNormalizer(resolve_path(config.get("normalization_json", "artifacts/normalization/train_patch_band_stats.json")))
    with np.load(test_npz, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    query_rows = read_query_rows(points_csv, arrays)

    batch_size = int(config.get("batch_size", 64))
    crop_chunks: list[np.ndarray] = []
    stage_chunks: list[np.ndarray] = []
    sample_indices = query_rows["sample_index"].to_numpy(dtype=np.int64)
    query_doys = query_rows["query_doy"].to_numpy(dtype=np.float32)
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        indices = sample_indices[start:end]
        patches = prepare_patches(arrays, indices, normalizer, include_mask_channels)
        batch = {
            "patches": torch.from_numpy(patches).to(device),
            "time_mask": torch.from_numpy(arrays["time_mask"][indices].astype(bool)).to(device),
            "time_doy": torch.from_numpy(arrays["time_doy"][indices].astype(np.float32)).to(device),
            "query_doy": torch.from_numpy(query_doys[start:end]).to(device),
        }
        with torch.no_grad():
            outputs = model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"])
        crop_chunks.append(outputs["crop_logits"].argmax(dim=1).cpu().numpy())
        stage_chunks.append(outputs["stage_logits"].argmax(dim=1).cpu().numpy())

    crop_pred = np.concatenate(crop_chunks)
    stage_pred = np.concatenate(stage_chunks)
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
        "model_uses_mask_channels": bool(include_mask_channels),
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
