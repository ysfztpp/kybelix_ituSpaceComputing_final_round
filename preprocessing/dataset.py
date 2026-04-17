from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import BAND_ORDER, CROP_TYPE_ORDER, INVALID_FILL_VALUE, PATCH_SIZE, PHENOPHASE_ORDER
from .filename import doy_from_timestamp
from .inventory import audit_tiff_files, build_region_catalog, select_file_index
from .mapping import map_points_to_regions, unique_points
from .raster_io import clean_patch_values, extract_patch_edge_from_src, raster_meta_from_src, rasterio


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _build_time_index(selected_df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for region_id, group in selected_df.groupby("region_id", sort=True):
        out[region_id] = sorted(group["start_norm"].unique().tolist())
    return out


def _candidate_paths_from_selected(row: pd.Series, root: Path) -> list[Path]:
    return [root / path for path in str(row["candidate_paths"]).split("|") if path]


def _open_first_readable(candidate_paths: list[Path]):
    attempts: list[dict] = []
    for path in candidate_paths:
        try:
            src = rasterio.open(path)
            attempts.append({"attempt_path": str(path), "attempt_status": "used", "attempt_error": ""})
            return src, path, attempts
        except Exception as exc:  # pragma: no cover - defensive logging
            attempts.append({"attempt_path": str(path), "attempt_status": "failed", "attempt_error": str(exc)})
    return None, None, attempts


def _attach_training_labels(metadata: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    labels["phenophase_doy"] = pd.to_datetime(labels["phenophase_date"], errors="coerce").dt.dayofyear
    pheno = (
        labels.pivot_table(index="point_id", columns="phenophase_name", values="phenophase_doy", aggfunc="first")
        .reindex(columns=PHENOPHASE_ORDER)
        .reset_index()
    )
    crop_map = {name: index for index, name in enumerate(CROP_TYPE_ORDER)}
    metadata = metadata.merge(pheno, on="point_id", how="left")
    metadata["crop_type_id"] = metadata["crop_type"].map(crop_map).fillna(-1).astype(np.int16)
    return metadata


def _normalise_sample_groups(sample_groups: dict[str, int] | None, sample_patch_count: int) -> dict[str, int]:
    if sample_groups:
        return {str(key): max(0, int(value)) for key, value in sample_groups.items()}
    random_n = max(0, sample_patch_count // 2)
    invalid_n = max(0, sample_patch_count - random_n)
    return {"random": random_n, "invalid": invalid_n}


def _normalise_point_columns(points_df: pd.DataFrame) -> pd.DataFrame:
    """Accept minor point CSV column-name variants without changing the canonical schema."""
    rename: dict[str, str] = {}
    lower_to_original = {str(col).strip().lower(): col for col in points_df.columns}
    for canonical, candidates in {
        "Longitude": ["longitude", "longtitude", "lon", "lng"],
        "Latitude": ["latitude", "lat"],
        "point_id": ["point_id", "pointid", "id"],
        "phenophase_date": ["phenophase_date", "date", "query_date", "datetime"],
    }.items():
        if canonical in points_df.columns:
            continue
        for candidate in candidates:
            original = lower_to_original.get(candidate)
            if original is not None:
                rename[original] = canonical
                break
    return points_df.rename(columns=rename)


def _reservoir_consider(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen_counts: dict[str, int],
    group: str,
    limit: int,
    rng: np.random.Generator,
    record: dict[str, Any],
) -> None:
    if limit <= 0:
        return
    seen_counts[group] += 1
    bucket = reservoirs[group]
    if len(bucket) < limit:
        bucket.append(record)
        return
    replace_index = int(rng.integers(0, seen_counts[group]))
    if replace_index < limit:
        bucket[replace_index] = record


def _record_sample_candidates(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen_counts: dict[str, int],
    sample_groups: dict[str, int],
    rng: np.random.Generator,
    sample_bands: set[str],
    invalid_ratio_threshold: float,
    *,
    point_id: Any,
    sample_index: int,
    region_id: str,
    date: str,
    time_index: int,
    band_id: str,
    raw_patch: np.ndarray,
    cleaned_patch: np.ndarray,
    valid_patch: np.ndarray,
    border_margin_pixels: int,
    center_clamped: bool,
) -> None:
    if band_id not in sample_bands:
        return
    valid_ratio = float(valid_patch.mean())
    base_record = {
        "point_id": point_id,
        "sample_index": int(sample_index),
        "region_id": str(region_id),
        "date": str(date),
        "time_index": int(time_index),
        "band_id": str(band_id),
        "raw_patch": raw_patch.astype(np.float32, copy=True),
        "cleaned_patch": cleaned_patch.astype(np.float32, copy=True),
        "valid_patch": valid_patch.astype(bool, copy=True),
        "valid_ratio": valid_ratio,
        "border_margin_pixels": int(border_margin_pixels),
        "center_clamped": bool(center_clamped),
    }
    if "random" in sample_groups:
        _reservoir_consider(reservoirs, seen_counts, "random", sample_groups["random"], rng, {**base_record, "sample_group": "random"})
    if "invalid" in sample_groups and valid_ratio < invalid_ratio_threshold:
        _reservoir_consider(reservoirs, seen_counts, "invalid", sample_groups["invalid"], rng, {**base_record, "sample_group": "invalid"})
    if "edge" in sample_groups and (border_margin_pixels < raw_patch.shape[0] // 2 or center_clamped):
        _reservoir_consider(reservoirs, seen_counts, "edge", sample_groups["edge"], rng, {**base_record, "sample_group": "edge"})


def _sample_records_to_rows(reservoirs: dict[str, list[dict[str, Any]]], patch_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in sorted(reservoirs):
        for idx, record in enumerate(reservoirs[group_name], start=1):
            sample_patch_id = f"{group_name}{idx:02d}_p{record['point_id']}_t{record['time_index']:02d}_{record['band_id']}"
            raw_patch = record["raw_patch"]
            cleaned_patch = record["cleaned_patch"]
            valid_patch = record["valid_patch"]
            for patch_row in range(patch_size):
                for patch_col in range(patch_size):
                    rows.append(
                        {
                            "sample_patch_id": sample_patch_id,
                            "sample_group": record["sample_group"],
                            "point_id": record["point_id"],
                            "sample_index": record["sample_index"],
                            "region_id": record["region_id"],
                            "date": record["date"],
                            "time_index": record["time_index"],
                            "band_id": record["band_id"],
                            "patch_row": patch_row,
                            "patch_col": patch_col,
                            "raw_value": float(raw_patch[patch_row, patch_col]),
                            "cleaned_value": float(cleaned_patch[patch_row, patch_col]),
                            "valid": bool(valid_patch[patch_row, patch_col]),
                            "patch_valid_ratio": float(record["valid_ratio"]),
                            "border_margin_pixels": int(record["border_margin_pixels"]),
                            "center_clamped": bool(record["center_clamped"]),
                        }
                    )
    return pd.DataFrame(rows)


def build_patch_dataset(
    points_csv: Path,
    tiff_dirs: list[Path],
    output_npz: Path,
    output_dir: Path,
    root: Path,
    mode: str,
    patch_size: int = PATCH_SIZE,
    allow_nearest_fallback: bool | None = None,
    sample_patch_count: int = 8,
    band_order: list[str] | None = None,
    valid_min_exclusive: float = 0.0,
    valid_max_inclusive: float = 2.0,
    invalid_fill_value: float = INVALID_FILL_VALUE,
    report_sample_groups: dict[str, int] | None = None,
    report_sample_bands: list[str] | None = None,
    report_random_seed: int = 42,
    invalid_sample_valid_ratio_below: float = 0.98,
    write_reports: bool = True,
) -> dict[str, Any]:
    if mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd so a point can sit at the center pixel")
    if allow_nearest_fallback is None:
        allow_nearest_fallback = mode == "test"

    band_order = list(band_order or BAND_ORDER)
    sample_groups = _normalise_sample_groups(report_sample_groups, sample_patch_count)
    sample_bands = set(report_sample_bands or [band for band in ["B04", "B08", "B11"] if band in band_order])
    rng = np.random.default_rng(report_random_seed)
    reservoirs = {name: [] for name in sample_groups}
    seen_counts = {name: 0 for name in sample_groups}

    output_dir.mkdir(parents=True, exist_ok=True)
    points_df = _normalise_point_columns(pd.read_csv(points_csv))
    point_meta, query_rows = unique_points(points_df)

    file_df = audit_tiff_files(tiff_dirs, root=root, band_order=band_order)
    selected_df = select_file_index(file_df)
    region_df = build_region_catalog(selected_df, root=root)
    candidates_df, summary_df, resolved_df = map_points_to_regions(
        point_meta=point_meta,
        region_df=region_df,
        patch_size=patch_size,
        allow_nearest_fallback=allow_nearest_fallback,
    )

    kept = resolved_df[resolved_df["keep_for_dataset"]].copy().reset_index(drop=True)
    kept["sample_index"] = np.arange(len(kept), dtype=np.int32)
    if mode == "train":
        kept = _attach_training_labels(kept, points_df)

    selected_df = selected_df.copy().reset_index(drop=True)
    selected_df["selected_file_index"] = np.arange(len(selected_df), dtype=np.int32)
    candidate_lookup: dict[tuple[str, str, str], tuple[int, list[Path]]] = {}
    for row in selected_df.itertuples(index=False):
        selected_row = pd.Series(row._asdict())
        candidate_lookup[(row.region_id, row.start_norm, row.band_id)] = (
            int(row.selected_file_index),
            _candidate_paths_from_selected(selected_row, root=root),
        )

    region_dates = _build_time_index(selected_df)
    max_timesteps = max((len(region_dates.get(region, [])) for region in kept["resolved_region_id"].astype(str)), default=0)
    n_samples = len(kept)
    n_bands = len(band_order)

    patches = np.full((n_samples, max_timesteps, n_bands, patch_size, patch_size), invalid_fill_value, dtype=np.float32)
    valid_pixel_mask = np.zeros((n_samples, max_timesteps, n_bands, patch_size, patch_size), dtype=bool)
    band_mask = np.zeros((n_samples, max_timesteps, n_bands), dtype=bool)
    time_mask = np.zeros((n_samples, max_timesteps), dtype=bool)
    time_doy = np.full((n_samples, max_timesteps), -1, dtype=np.int16)
    time_dates = np.full((n_samples, max_timesteps), "", dtype="<U16")
    border_margin = np.full((n_samples, max_timesteps, n_bands), -9999, dtype=np.int16)
    center_clamped = np.zeros((n_samples, max_timesteps, n_bands), dtype=bool)
    source_file_index = np.full((n_samples, max_timesteps, n_bands), -1, dtype=np.int32)
    band_valid_ratio = np.zeros((n_samples, max_timesteps, n_bands), dtype=np.float32)

    kept["resolved_time_steps"] = 0
    file_attempt_rows: list[dict] = []
    extraction_error_rows: list[dict] = []

    points_by_region: dict[str, list[Any]] = defaultdict(list)
    for row in kept.itertuples(index=False):
        points_by_region[str(row.resolved_region_id)].append(row)

    for region_id in sorted(points_by_region):
        region_points = points_by_region[region_id]
        dates = region_dates.get(region_id, [])
        kept.loc[kept["resolved_region_id"] == region_id, "resolved_time_steps"] = len(dates)

        for row in region_points:
            sample_index = int(row.sample_index)
            for time_index, start_norm in enumerate(dates):
                time_mask[sample_index, time_index] = True
                time_doy[sample_index, time_index] = doy_from_timestamp(start_norm)
                time_dates[sample_index, time_index] = start_norm[:10]

        for time_index, start_norm in enumerate(dates):
            for band_index, band_id in enumerate(band_order):
                selected = candidate_lookup.get((region_id, start_norm, band_id))
                if selected is None:
                    continue
                selected_file_index, candidate_paths = selected
                src, used_path, attempts = _open_first_readable(candidate_paths)
                for attempt in attempts:
                    file_attempt_rows.append(
                        {
                            "region_id": region_id,
                            "start_norm": start_norm,
                            "band_id": band_id,
                            "selected_file_index": selected_file_index,
                            "selected_candidate_count": len(candidate_paths),
                            "attempt_path": attempt["attempt_path"],
                            "attempt_status": attempt["attempt_status"],
                            "attempt_error": attempt["attempt_error"],
                        }
                    )
                if src is None or used_path is None:
                    continue

                try:
                    meta = raster_meta_from_src(used_path, src)
                    for row in region_points:
                        sample_index = int(row.sample_index)
                        try:
                            extracted = extract_patch_edge_from_src(
                                src=src,
                                meta=meta,
                                lon=float(row.Longitude),
                                lat=float(row.Latitude),
                                patch_size=patch_size,
                            )
                            cleaned, valid = clean_patch_values(
                                extracted.patch,
                                min_exclusive=valid_min_exclusive,
                                max_inclusive=valid_max_inclusive,
                                fill_value=invalid_fill_value,
                            )
                            patches[sample_index, time_index, band_index] = cleaned
                            valid_pixel_mask[sample_index, time_index, band_index] = valid
                            band_mask[sample_index, time_index, band_index] = True
                            border_margin[sample_index, time_index, band_index] = extracted.border_margin_pixels
                            center_clamped[sample_index, time_index, band_index] = extracted.center_clamped
                            source_file_index[sample_index, time_index, band_index] = selected_file_index
                            band_valid_ratio[sample_index, time_index, band_index] = float(valid.mean())
                            if write_reports:
                                _record_sample_candidates(
                                    reservoirs=reservoirs,
                                    seen_counts=seen_counts,
                                    sample_groups=sample_groups,
                                    rng=rng,
                                    sample_bands=sample_bands,
                                    invalid_ratio_threshold=invalid_sample_valid_ratio_below,
                                    point_id=row.point_id,
                                    sample_index=sample_index,
                                    region_id=region_id,
                                    date=start_norm[:10],
                                    time_index=time_index,
                                    band_id=band_id,
                                    raw_patch=extracted.patch,
                                    cleaned_patch=cleaned,
                                    valid_patch=valid,
                                    border_margin_pixels=extracted.border_margin_pixels,
                                    center_clamped=extracted.center_clamped,
                                )
                        except Exception as exc:  # pragma: no cover - defensive logging
                            extraction_error_rows.append(
                                {
                                    "point_id": row.point_id,
                                    "sample_index": sample_index,
                                    "Longitude": float(row.Longitude),
                                    "Latitude": float(row.Latitude),
                                    "region_id": region_id,
                                    "start_norm": start_norm,
                                    "band_id": band_id,
                                    "selected_file_index": selected_file_index,
                                    "path": str(used_path),
                                    "error": str(exc),
                                }
                            )
                finally:
                    src.close()

    metadata = kept.copy()
    metadata["time_steps_kept"] = time_mask.sum(axis=1).astype(np.int16)
    metadata["observed_band_cells"] = band_mask.sum(axis=(1, 2)).astype(np.int32)
    metadata["expected_band_cells"] = (time_mask.sum(axis=1) * n_bands).astype(np.int32)
    metadata["missing_band_cells"] = metadata["expected_band_cells"] - metadata["observed_band_cells"]
    metadata["has_missing_band"] = metadata["missing_band_cells"] > 0
    metadata["valid_pixel_ratio"] = np.divide(
        valid_pixel_mask.sum(axis=(1, 2, 3, 4)),
        np.maximum(band_mask.sum(axis=(1, 2)) * patch_size * patch_size, 1),
    )
    edge_replication_cells = (border_margin >= 0) & (border_margin < (patch_size // 2)) & band_mask
    metadata["requires_edge_replication"] = edge_replication_cells.any(axis=(1, 2)) if n_samples else False
    metadata["has_center_clamping"] = center_clamped.any(axis=(1, 2)) if n_samples else False

    file_attempt_df = pd.DataFrame(file_attempt_rows)
    extraction_error_df = pd.DataFrame(extraction_error_rows)

    if write_reports:
        file_df.to_csv(output_dir / "filename_audit.csv", index=False)
        selected_df.to_csv(output_dir / "selected_file_index.csv", index=False)
        region_df.to_csv(output_dir / "region_catalog.csv", index=False)
        candidates_df.to_csv(output_dir / "point_region_candidates.csv", index=False)
        summary_df.to_csv(output_dir / "point_region_summary.csv", index=False)
        resolved_df.to_csv(output_dir / "point_region_resolved.csv", index=False)
        metadata.to_csv(output_dir / f"{mode}_metadata.csv", index=False)
        query_rows.to_csv(output_dir / f"{mode}_query_rows.csv", index=False)
        file_attempt_df.to_csv(output_dir / "file_read_attempts.csv", index=False)
        extraction_error_df.to_csv(output_dir / "patch_extraction_errors.csv", index=False)

    save_kwargs: dict[str, Any] = {
        "patches": patches,
        "valid_pixel_mask": valid_pixel_mask,
        "band_mask": band_mask,
        "time_mask": time_mask,
        "time_doy": time_doy,
        "time_dates": time_dates,
        "border_margin_pixels": border_margin,
        "center_clamped": center_clamped,
        "source_file_index": source_file_index,
        "band_valid_ratio": band_valid_ratio,
        "point_id": metadata["point_id"].to_numpy(dtype=np.int32),
        "longitude": metadata["Longitude"].to_numpy(dtype=np.float64),
        "latitude": metadata["Latitude"].to_numpy(dtype=np.float64),
        "resolved_region_id": metadata["resolved_region_id"].astype(str).to_numpy(dtype="<U32"),
        "bands": np.asarray(band_order, dtype="<U4"),
        "schema_version": np.asarray("cnn_transformer_patch_timeseries_v2", dtype="<U64"),
        "invalid_fill_value": np.asarray(invalid_fill_value, dtype=np.float32),
        "valid_min_exclusive": np.asarray(valid_min_exclusive, dtype=np.float32),
        "valid_max_inclusive": np.asarray(valid_max_inclusive, dtype=np.float32),
        "patch_size": np.asarray(patch_size, dtype=np.int16),
    }
    if mode == "train":
        save_kwargs.update(
            {
                "crop_type_id": metadata["crop_type_id"].to_numpy(dtype=np.int16),
                "crop_type_names": np.asarray(CROP_TYPE_ORDER, dtype="<U16"),
                "phenophase_names": np.asarray(PHENOPHASE_ORDER, dtype="<U16"),
                "phenophase_doy": metadata[PHENOPHASE_ORDER].fillna(-1).to_numpy(dtype=np.int16),
            }
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **save_kwargs)

    total_observed_pixels = int(band_mask.sum()) * patch_size * patch_size
    missing_by_band = {
        band_id: int(time_mask.sum() - band_mask[:, :, band_index].sum()) for band_index, band_id in enumerate(band_order)
    }
    valid_ratio_by_band = {
        band_id: float(valid_pixel_mask[:, :, band_index].sum() / max(int(band_mask[:, :, band_index].sum()) * patch_size * patch_size, 1))
        for band_index, band_id in enumerate(band_order)
    }
    sample_seen_counts = {key: int(value) for key, value in seen_counts.items()}
    sample_kept_counts = {key: int(len(value)) for key, value in reservoirs.items()}

    invalid_policy = (
        f"valid iff finite and {valid_min_exclusive} < value <= {valid_max_inclusive}; "
        f"invalid pixels are filled with {invalid_fill_value} in patches and preserved in valid_pixel_mask"
    )
    report = {
        "mode": mode,
        "points_csv": str(points_csv),
        "tiff_dirs": [str(path) for path in tiff_dirs],
        "output_npz": str(output_npz),
        "schema_version": "cnn_transformer_patch_timeseries_v2",
        "patch_shape": list(patches.shape),
        "patch_dtype": str(patches.dtype),
        "valid_pixel_mask_shape": list(valid_pixel_mask.shape),
        "band_mask_shape": list(band_mask.shape),
        "time_mask_shape": list(time_mask.shape),
        "bands": band_order,
        "patch_size": patch_size,
        "invalid_fill_value": float(invalid_fill_value),
        "valid_min_exclusive": float(valid_min_exclusive),
        "valid_max_inclusive": float(valid_max_inclusive),
        "invalid_value_policy": invalid_policy,
        "pixel_rounding_policy": "nearest pixel using round((lon-xmin)/pixel_size_x), round((ymax-lat)/pixel_size_y)",
        "border_policy": "edge_replication",
        "points_input_rows": int(len(points_df)),
        "unique_points": int(len(point_meta)),
        "samples_kept": int(n_samples),
        "samples_dropped": int((~resolved_df["keep_for_dataset"]).sum()) if not resolved_df.empty else 0,
        "allow_nearest_fallback": bool(allow_nearest_fallback),
        "filename_status_counts": file_df["status"].value_counts(dropna=False).to_dict() if not file_df.empty else {},
        "region_count": int(len(region_df)),
        "selected_file_rows": int(len(selected_df)),
        "selected_file_collision_groups": int((selected_df["candidate_count"] > 1).sum()),
        "selected_l2a_rows": int((selected_df["selected_level"] == "L2A").sum()),
        "selected_l1c_rows": int((selected_df["selected_level"] == "L1C").sum()),
        "total_time_cells": int(time_mask.sum()),
        "max_timesteps": int(max_timesteps),
        "padded_time_cells": int(time_mask.size - time_mask.sum()),
        "samples_with_time_padding": int((~time_mask).any(axis=1).sum()) if n_samples else 0,
        "observed_band_cells": int(band_mask.sum()),
        "missing_band_cells": int((time_mask.sum() * n_bands) - band_mask.sum()),
        "missing_band_cells_by_band": missing_by_band,
        "samples_with_missing_bands": int(metadata["has_missing_band"].sum()) if n_samples else 0,
        "observed_patch_pixels": int(total_observed_pixels),
        "valid_patch_pixels": int(valid_pixel_mask.sum()),
        "invalid_patch_pixels": int(total_observed_pixels - valid_pixel_mask.sum()),
        "global_valid_pixel_ratio": float(valid_pixel_mask.sum() / max(total_observed_pixels, 1)),
        "valid_pixel_ratio_by_band": valid_ratio_by_band,
        "samples_with_any_invalid_pixel": int((band_valid_ratio < 1.0).any(axis=(1, 2)).sum()) if n_samples else 0,
        "requires_edge_replication_count": int(metadata["requires_edge_replication"].sum()) if n_samples else 0,
        "edge_replication_band_cells": int(edge_replication_cells.sum()),
        "center_clamped_band_cells": int(center_clamped.sum()),
        "samples_with_center_clamping": int(metadata["has_center_clamping"].sum()) if n_samples else 0,
        "file_read_failures": int((file_attempt_df.get("attempt_status", pd.Series(dtype=str)) == "failed").sum()),
        "patch_extraction_errors": int(len(extraction_error_df)),
        "mapping_status_counts": resolved_df["mapping_status"].value_counts(dropna=False).to_dict() if not resolved_df.empty else {},
        "report_sample_groups_requested": sample_groups,
        "report_sample_groups_seen": sample_seen_counts,
        "report_sample_groups_kept": sample_kept_counts,
        "report_sample_bands": sorted(sample_bands),
        "report_random_seed": int(report_random_seed),
    }
    if write_reports:
        from .reporting import write_preprocessing_report

        (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2, default=_json_default))
        sample_rows_df = _sample_records_to_rows(reservoirs, patch_size)
        write_preprocessing_report(output_dir, report, metadata, sample_rows_df)
    return report
