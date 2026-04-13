from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .constants import BAND_ORDER
from .filename import canonical_name, level_rank, parse_tiff_name
from .raster_io import read_raster_meta


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_tiff_paths(tiff_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in tiff_dirs:
        if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}:
            paths.append(item)
        elif item.is_dir():
            paths.extend(sorted(p for p in item.glob("*.tif*") if p.is_file()))
        else:
            paths.extend(sorted(Path().glob(str(item))))
    return sorted(set(paths))


def audit_tiff_files(tiff_dirs: list[Path], root: Path, band_order: list[str] | None = None) -> pd.DataFrame:
    supported_bands = set(band_order or BAND_ORDER)
    paths = iter_tiff_paths(tiff_dirs)
    candidates_by_canonical: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        candidates_by_canonical[(path.parent.name, canonical_name(path.name))].append(path)

    duplicate_of: dict[Path, Path] = {}
    for (_, canonical), group_paths in candidates_by_canonical.items():
        exact = [path for path in group_paths if path.name == canonical]
        keeper = exact[0] if exact else sorted(group_paths)[0]
        for path in group_paths:
            if path != keeper:
                duplicate_of[path] = keeper

    rows: list[dict] = []
    for path in paths:
        parsed = parse_tiff_name(path.name)
        canonical = canonical_name(path.name)
        duplicate_path = duplicate_of.get(path)
        if parsed is None:
            status = "unparsed"
            reason = "filename_does_not_match_expected_pattern"
        elif parsed.band_id not in supported_bands:
            status = "unsupported_band"
            reason = "band_not_used_for_12_band_patch_timeseries"
        elif duplicate_path is not None:
            status = "duplicate_canonical"
            reason = "canonical_file_already_exists_in_same_folder"
        else:
            status = "valid"
            reason = ""

        row = {
            "folder": path.parent.name,
            "path": _rel(path, root),
            "name": path.name,
            "canonical_name": canonical,
            "status": status,
            "reason": reason,
            "duplicate_of": _rel(duplicate_path, root) if duplicate_path else "",
            "use_for_timeseries": status == "valid",
        }
        if parsed is not None:
            row.update(
                {
                    "region_id": parsed.region_id,
                    "start_raw": parsed.start_raw,
                    "end_raw": parsed.end_raw,
                    "start_norm": parsed.start_norm,
                    "end_norm": parsed.end_norm,
                    "level": parsed.level,
                    "band_token": parsed.band_token,
                    "band_id": parsed.band_id,
                }
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["status", "folder", "name"]).reset_index(drop=True)


def select_file_index(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        raise ValueError("no TIFF files found")
    valid = file_df[file_df["status"] == "valid"].copy()
    if valid.empty:
        raise ValueError("no valid Sentinel-2 band TIFFs found after filename audit")
    valid["level_rank"] = valid["level"].map(level_rank)

    rows: list[dict] = []
    for (region_id, start_norm, band_id), group in valid.groupby(["region_id", "start_norm", "band_id"], sort=True):
        group = group.sort_values(["level_rank", "path"]).reset_index(drop=True)
        best = group.iloc[0]
        rows.append(
            {
                "region_id": region_id,
                "start_norm": start_norm,
                "band_id": band_id,
                "selected_path": best["path"],
                "selected_level": best["level"],
                "candidate_count": int(len(group)),
                "candidate_levels": "|".join(sorted(group["level"].astype(str).unique())),
                "candidate_paths": "|".join(group["path"].astype(str).tolist()),
                "selection_reason": "prefer_L2A_then_path",
            }
        )
    return pd.DataFrame(rows).sort_values(["region_id", "start_norm", "band_id"]).reset_index(drop=True)


def build_region_catalog(selected_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for region_id, group in selected_df.groupby("region_id", sort=True):
        first_path = root / str(group.iloc[0]["selected_path"])
        meta = read_raster_meta(first_path)
        rows.append(
            {
                "region_id": region_id,
                "file_count_used": int(len(group)),
                "time_steps": int(group["start_norm"].nunique()),
                "band_count": int(group["band_id"].nunique()),
                "band_ids": "|".join(sorted(group["band_id"].unique().tolist())),
                "levels": "|".join(sorted(group["selected_level"].astype(str).unique().tolist())),
                "xmin": meta.xmin,
                "ymin": meta.ymin,
                "xmax": meta.xmax,
                "ymax": meta.ymax,
                "width": meta.width,
                "height": meta.height,
                "pixel_size_x": meta.pixel_size_x,
                "pixel_size_y": meta.pixel_size_y,
                "crs": meta.crs,
                "bbox_area_deg2": (meta.xmax - meta.xmin) * (meta.ymax - meta.ymin),
                "reference_path": str(group.iloc[0]["selected_path"]),
            }
        )
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
