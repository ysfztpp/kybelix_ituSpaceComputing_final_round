from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.constants import BAND_ORDER, INVALID_FILL_VALUE, PATCH_SIZE
from preprocessing.dataset import build_patch_dataset


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def find_points_csv(input_root: Path) -> Path:
    for name in ["test_point.csv", "points_test.csv"]:
        candidate = input_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"expected {input_root}/test_point.csv or {input_root}/points_test.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract test-time patches with the same core preprocessing used for training.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.json")
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    parser.add_argument("--points-csv", type=Path, default=None)
    parser.add_argument("--tiff-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/patches_clean_test"))
    parser.add_argument("--output-npz", type=Path, default=Path("/workspace/patches_clean_test/test_cnn_transformer_15x15.npz"))
    parser.add_argument("--write-report", action="store_true", help="Write CSV/PNG diagnostics. Keep off for final inference unless debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    patch_config = config.get("patch", {})
    report_config = config.get("report", {})

    points_csv = args.points_csv or find_points_csv(args.input_root)
    tiff_dir = args.tiff_dir or (args.input_root / "region_test")
    root = args.input_root if args.input_root.exists() else ROOT

    report = build_patch_dataset(
        points_csv=points_csv,
        tiff_dirs=[tiff_dir],
        output_npz=args.output_npz,
        output_dir=args.output_dir,
        root=root,
        mode="test",
        patch_size=int(patch_config.get("size", PATCH_SIZE)),
        allow_nearest_fallback=True,
        band_order=list(patch_config.get("bands", BAND_ORDER)),
        valid_min_exclusive=float(patch_config.get("valid_min_exclusive", 0.0)),
        valid_max_inclusive=float(patch_config.get("valid_max_inclusive", 2.0)),
        invalid_fill_value=float(patch_config.get("invalid_fill_value", INVALID_FILL_VALUE)),
        report_sample_groups=report_config.get("sample_groups"),
        report_sample_bands=report_config.get("sample_bands"),
        report_random_seed=int(report_config.get("random_seed", 42)),
        invalid_sample_valid_ratio_below=float(report_config.get("invalid_sample_valid_ratio_below", 0.98)),
        write_reports=args.write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
