from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.predict import predict_npz
from inference.write_submission import write_generic_result_json


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference from an extracted test NPZ and write result.json.")
    parser.add_argument("--test-npz", type=Path, default=Path("/workspace/patches_clean_test/test_cnn_transformer_15x15.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, default=Path("artifacts/normalization/train_patch_band_stats.json"))
    parser.add_argument("--output-json", type=Path, default=Path("/output/result.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_npz(resolve_path(args.test_npz), resolve_path(args.checkpoint), resolve_path(args.normalization_json), args.batch_size, args.device)
    write_generic_result_json(predictions, args.output_json)
    print({"output_json": str(args.output_json), "rows": int(len(predictions["point_id"]))})


if __name__ == "__main__":
    main()
