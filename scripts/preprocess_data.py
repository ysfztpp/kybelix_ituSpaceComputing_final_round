from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.splits import make_train_val_split
from preprocessing.constants import BAND_ORDER, INVALID_FILL_VALUE, PATCH_SIZE
from preprocessing.dataset import build_patch_dataset
from preprocessing.normalization import compute_band_stats


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = ["points_csv", "tiff_dirs", "output_npz", "normalization_json", "report_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean 15x15 Sentinel-2 patch dataset plus logs/charts.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.json")
    parser.add_argument("--keep-old-report", action="store_true", help="Do not clear the previous report folder before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))

    patch_config = config.get("patch", {})
    report_config = config.get("report", {})
    split_config = config.get("split", {})

    points_csv = resolve_path(config["points_csv"])
    tiff_dirs = [resolve_path(path) for path in config["tiff_dirs"]]
    output_npz = resolve_path(config["output_npz"])
    normalization_json = resolve_path(config["normalization_json"])
    report_dir = resolve_path(config["report_dir"])

    if report_dir.exists() and not args.keep_old_report:
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(patch_config.get("size", config.get("patch_size", PATCH_SIZE)))
    band_order = list(patch_config.get("bands", config.get("bands", BAND_ORDER)))
    invalid_fill_value = float(patch_config.get("invalid_fill_value", INVALID_FILL_VALUE))
    valid_min_exclusive = float(patch_config.get("valid_min_exclusive", 0.0))
    valid_max_inclusive = float(patch_config.get("valid_max_inclusive", 2.0))

    report = build_patch_dataset(
        points_csv=points_csv,
        tiff_dirs=tiff_dirs,
        output_npz=output_npz,
        output_dir=report_dir,
        root=ROOT,
        mode="train",
        patch_size=patch_size,
        allow_nearest_fallback=False,
        sample_patch_count=int(config.get("sample_patch_count", 8)),
        band_order=band_order,
        valid_min_exclusive=valid_min_exclusive,
        valid_max_inclusive=valid_max_inclusive,
        invalid_fill_value=invalid_fill_value,
        report_sample_groups=report_config.get("sample_groups"),
        report_sample_bands=report_config.get("sample_bands"),
        report_random_seed=int(report_config.get("random_seed", 42)),
        invalid_sample_valid_ratio_below=float(report_config.get("invalid_sample_valid_ratio_below", 0.98)),
        write_reports=bool(report_config.get("enabled", True)),
    )
    stats = compute_band_stats(output_npz, normalization_json)

    split_summary = None
    if split_config.get("enabled", True):
        split_summary = make_train_val_split(
            metadata_csv=report_dir / "train_metadata.csv",
            output_csv=resolve_path(split_config.get("output_csv", "artifacts/splits/train_val_split.csv")),
            val_fraction=float(split_config.get("val_fraction", 0.2)),
            seed=int(split_config.get("random_seed", 42)),
            stratify_by=str(split_config.get("stratify_by", "crop_type")),
            group_by=split_config.get("group_by"),
        )

    summary = {
        "dataset": str(output_npz.relative_to(ROOT)),
        "normalization": str(normalization_json.relative_to(ROOT)),
        "report_dir": str(report_dir.relative_to(ROOT)),
        "patch_shape": report["patch_shape"],
        "samples_kept": report["samples_kept"],
        "missing_band_cells": report["missing_band_cells"],
        "valid_pixel_ratio": report["global_valid_pixel_ratio"],
        "edge_replication_samples": report["requires_edge_replication_count"],
        "sample_groups_kept": report.get("report_sample_groups_kept", {}),
        "normalization_bands": stats["bands"],
        "split": split_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
