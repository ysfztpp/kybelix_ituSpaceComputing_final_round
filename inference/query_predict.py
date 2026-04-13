from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.transforms import NpzPatchNormalizer
from models.query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for inference. Install PyTorch before prediction.") from exc


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def load_query_model(checkpoint_path: Path, device: torch.device) -> QueryCNNTransformerClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = QueryCNNTransformerConfig(**checkpoint["model_config"])
    model = QueryCNNTransformerClassifier(config)
    model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()
    return model


def _find_column(columns: list[str], candidates: list[str]) -> str:
    lookup = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"missing required column; tried: {candidates}")


def _format_date_key(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty query date")
    return text


def _date_to_doy(value: Any) -> int:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"could not parse query date: {value!r}")
    return int(parsed.dayofyear)


def build_query_table(points_csv: Path, npz_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    raw = pd.read_csv(points_csv, dtype=str).reset_index(drop=False).rename(columns={"index": "query_row_index"})
    lon_col = _find_column(raw.columns.tolist(), ["longitude", "longtitude", "lon", "lng"])
    lat_col = _find_column(raw.columns.tolist(), ["latitude", "lat"])
    date_col = _find_column(raw.columns.tolist(), ["phenophase_date", "date", "query_date", "datetime", "time"])

    point_id_col: str | None = None
    try:
        point_id_col = _find_column(raw.columns.tolist(), ["point_id", "pointid", "id"])
    except ValueError:
        raw["point_id"] = np.arange(1, len(raw) + 1, dtype=np.int64).astype(str)
        point_id_col = "point_id"

    sample_by_point_id = {int(point_id): sample_index for sample_index, point_id in enumerate(npz_arrays["point_id"])}
    rows: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        row_dict = row._asdict()
        point_id = int(float(row_dict[point_id_col]))
        if point_id not in sample_by_point_id:
            raise ValueError(f"point_id {point_id} from {points_csv} was not found in extracted test NPZ")
        date_key = _format_date_key(row_dict[date_col])
        rows.append(
            {
                "sample_index": int(sample_by_point_id[point_id]),
                "point_id": point_id,
                "longitude_key": str(row_dict[lon_col]).strip(),
                "latitude_key": str(row_dict[lat_col]).strip(),
                "date_key": date_key,
                "query_doy": _date_to_doy(date_key),
            }
        )
    return pd.DataFrame(rows)


def predict_point_date_queries(
    npz_path: Path,
    points_csv: Path,
    checkpoint_path: Path,
    normalization_json: Path,
    batch_size: int = 32,
    device_name: str = "auto",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model = load_query_model(checkpoint_path, device)
    normalizer = NpzPatchNormalizer(normalization_json)

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    query_table = build_query_table(points_csv, arrays)

    crop_logits: list[np.ndarray] = []
    stage_logits: list[np.ndarray] = []
    sample_indices = query_table["sample_index"].to_numpy(dtype=np.int64)
    query_doys = query_table["query_doy"].to_numpy(dtype=np.float32)

    for start in range(0, len(query_table), batch_size):
        end = min(start + batch_size, len(query_table))
        idx = sample_indices[start:end]
        patches = normalizer(arrays["patches"][idx], arrays["valid_pixel_mask"][idx].astype(bool))
        batch = {
            "patches": torch.from_numpy(patches.astype(np.float32, copy=False)).to(device),
            "time_mask": torch.from_numpy(arrays["time_mask"][idx].astype(bool)).to(device),
            "time_doy": torch.from_numpy(arrays["time_doy"][idx].astype(np.float32)).to(device),
            "query_doy": torch.from_numpy(query_doys[start:end]).to(device),
        }
        with torch.no_grad():
            outputs = model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"])
        crop_logits.append(outputs["crop_logits"].detach().cpu().numpy())
        stage_logits.append(outputs["stage_logits"].detach().cpu().numpy())

    crop_logits_np = np.concatenate(crop_logits, axis=0)
    stage_logits_np = np.concatenate(stage_logits, axis=0)
    return query_table, {
        "crop_logits": crop_logits_np,
        "stage_logits": stage_logits_np,
        "crop_type_id": crop_logits_np.argmax(axis=1),
        "phenophase_stage_id": stage_logits_np.argmax(axis=1),
    }
