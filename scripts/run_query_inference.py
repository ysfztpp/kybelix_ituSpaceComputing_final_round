from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.query_predict import predict_point_date_queries
from inference.write_submission import write_point_date_result_json
from preprocessing.constants import BAND_ORDER, INVALID_FILL_VALUE, PATCH_SIZE
from preprocessing.dataset import build_patch_dataset
from scripts.extract_test_patches import find_points_csv, resolve_path


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-round inference: extract test patches and write /output/result.json.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.json")
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    parser.add_argument("--points-csv", type=Path, default=None)
    parser.add_argument("--tiff-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/patches_clean_test"))
    parser.add_argument("--test-npz", type=Path, default=Path("/workspace/patches_clean_test/test_cnn_transformer_15x15.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/query_cnn_transformer_colab/model.pt"))
    parser.add_argument("--normalization-json", type=Path, default=Path("artifacts/normalization/train_patch_band_stats.json"))
    parser.add_argument("--output-json", type=Path, default=Path("/output/result.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--write-report", action="store_true")
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
        output_npz=args.test_npz,
        output_dir=args.work_dir,
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
    query_rows, predictions = predict_point_date_queries(
        npz_path=args.test_npz,
        points_csv=points_csv,
        checkpoint_path=resolve_path(args.checkpoint),
        normalization_json=resolve_path(args.normalization_json),
        batch_size=args.batch_size,
        device_name=args.device,
    )
    write_point_date_result_json(query_rows, predictions, args.output_json)
    print(json.dumps({"output_json": str(args.output_json), "queries": int(len(query_rows)), "patch_report": report}, indent=2))


if __name__ == "__main__":
    main()
