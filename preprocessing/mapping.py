from __future__ import annotations

import math
import pandas as pd

from .raster_io import RasterMeta, bbox_distance_deg, border_margin_pixels, contains_point, lonlat_to_pixel, patch_fits_without_padding


def _region_row_to_meta(row: pd.Series) -> RasterMeta:
    return RasterMeta(
        path=str(row.get("reference_path", "")),
        xmin=float(row["xmin"]),
        ymin=float(row["ymin"]),
        xmax=float(row["xmax"]),
        ymax=float(row["ymax"]),
        width=int(row["width"]),
        height=int(row["height"]),
        pixel_size_x=float(row["pixel_size_x"]),
        pixel_size_y=float(row["pixel_size_y"]),
        crs=str(row.get("crs", "")),
        transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def unique_points(points_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"point_id", "Longitude", "Latitude"}
    missing = required - set(points_df.columns)
    if missing:
        raise ValueError(f"point CSV is missing required columns: {sorted(missing)}")

    query_rows = points_df.copy().reset_index(drop=False).rename(columns={"index": "query_row_index"})
    agg = {"Longitude": "first", "Latitude": "first"}
    if "crop_type" in points_df.columns:
        agg["crop_type"] = "first"
    point_meta = points_df.groupby("point_id", as_index=False).agg(agg)

    coord_counts = points_df.groupby("point_id")[["Longitude", "Latitude"]].nunique().reset_index()
    coord_counts["coordinate_conflict"] = (coord_counts["Longitude"] > 1) | (coord_counts["Latitude"] > 1)
    point_meta = point_meta.merge(coord_counts[["point_id", "coordinate_conflict"]], on="point_id", how="left")
    point_meta["coordinate_conflict"] = point_meta["coordinate_conflict"].fillna(False).astype(bool)
    return point_meta.sort_values("point_id").reset_index(drop=True), query_rows


def map_points_to_regions(
    point_meta: pd.DataFrame,
    region_df: pd.DataFrame,
    patch_size: int,
    allow_nearest_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_records = [(row["region_id"], _region_row_to_meta(row), row) for _, row in region_df.iterrows()]
    candidate_rows: list[dict] = []
    resolved_rows: list[dict] = []
    summary_rows: list[dict] = []

    for point in point_meta.itertuples(index=False):
        lon = float(point.Longitude)
        lat = float(point.Latitude)
        contains_matches: list[dict] = []
        nearest_match: dict | None = None
        nearest_distance = math.inf

        for region_id, meta, region_row in region_records:
            distance = bbox_distance_deg(meta, lon, lat)
            pixel_x_float, pixel_y_float, pixel_x, pixel_y = lonlat_to_pixel(meta, lon, lat)
            row = {
                "point_id": point.point_id,
                "Longitude": lon,
                "Latitude": lat,
                "match_kind": "contains" if contains_point(meta, lon, lat) else "nearest_candidate",
                "region_id": region_id,
                "pixel_x_float": pixel_x_float,
                "pixel_y_float": pixel_y_float,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "width": meta.width,
                "height": meta.height,
                "border_margin_pixels": border_margin_pixels(meta, max(0, min(pixel_x, meta.width - 1)), max(0, min(pixel_y, meta.height - 1))),
                "patch_fits_15x15": patch_fits_without_padding(meta, pixel_x, pixel_y, patch_size),
                "bbox_distance_deg": distance,
                "time_steps": int(region_row["time_steps"]),
                "band_count": int(region_row["band_count"]),
            }
            if row["match_kind"] == "contains":
                contains_matches.append(row)
            if distance < nearest_distance or (distance == nearest_distance and str(region_id) < str(nearest_match["region_id"] if nearest_match else "zzzz")):
                nearest_distance = distance
                nearest_match = row

        candidate_rows.extend(contains_matches)
        mapping_status = "unique_region" if len(contains_matches) == 1 else "overlap_multi_region" if contains_matches else "no_covering_region"
        if contains_matches:
            best_candidates = sorted(
                contains_matches,
                key=lambda row: (-row["time_steps"], -row["band_count"], -row["border_margin_pixels"], str(row["region_id"])),
            )
            best = best_candidates[0]
            keep = True
            reason = "best_region_by_time_steps_then_band_count_then_border_margin_then_region_id"
            match_kind = best["match_kind"]
        elif allow_nearest_fallback and nearest_match is not None:
            best = nearest_match.copy()
            best["match_kind"] = "nearest_fallback"
            candidate_rows.append(best)
            keep = True
            reason = "nearest_region_for_inference_only"
            match_kind = "nearest_fallback"
        else:
            best = nearest_match or {}
            keep = False
            reason = "drop_no_covering_region"
            match_kind = "dropped"

        summary_rows.append(
            {
                "point_id": point.point_id,
                "Longitude": lon,
                "Latitude": lat,
                "candidate_region_count": len(contains_matches),
                "mapping_status": mapping_status,
                "nearest_region_id": best.get("region_id", ""),
                "nearest_distance_deg": float(nearest_distance) if nearest_distance < math.inf else math.nan,
                "resolved_region_id": best.get("region_id", "") if keep else "",
                "resolution_reason": reason,
                "keep_for_dataset": bool(keep),
            }
        )

        resolved_row = {
            "point_id": point.point_id,
            "Longitude": lon,
            "Latitude": lat,
            "mapping_status": mapping_status,
            "resolved_region_id": best.get("region_id", "") if keep else "",
            "resolution_reason": reason,
            "keep_for_dataset": bool(keep),
            "match_kind": match_kind,
            "patch_padding_mode": "edge_replication",
            "coordinate_conflict": bool(getattr(point, "coordinate_conflict", False)),
        }
        if "crop_type" in point_meta.columns:
            resolved_row["crop_type"] = getattr(point, "crop_type")
        for key in ["pixel_x", "pixel_y", "pixel_x_float", "pixel_y_float", "time_steps", "band_count", "border_margin_pixels", "patch_fits_15x15"]:
            resolved_row[key] = best.get(key, math.nan)
        resolved_rows.append(resolved_row)

    candidates_df = pd.DataFrame(candidate_rows).sort_values(["point_id", "match_kind", "region_id"]).reset_index(drop=True) if candidate_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values("point_id").reset_index(drop=True)
    resolved_df = pd.DataFrame(resolved_rows).sort_values("point_id").reset_index(drop=True)
    return candidates_df, summary_df, resolved_df
