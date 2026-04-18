from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aux_features import aux_feature_names, compute_aux_features


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract optional auxiliary features from the clean patch NPZ.")
    parser.add_argument("--npz", type=Path, default=ROOT / "artifacts" / "patches_clean" / "train_cnn_transformer_15x15.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "features" / "train_aux_features_v1.npz")
    parser.add_argument("--query-stage-features", action="store_true", help="Also build one feature row per valid phenophase query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_path(args.npz)
    output_path = resolve_path(args.output)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    bands = arrays["bands"].astype(str).tolist()
    names = aux_feature_names(bands)
    sample_features = np.stack(
        [
            compute_aux_features(
                arrays["patches"][sample_index],
                arrays["valid_pixel_mask"][sample_index],
                arrays["time_mask"][sample_index],
                arrays["time_doy"][sample_index],
                float(np.nanmedian(arrays["time_doy"][sample_index][arrays["time_mask"][sample_index].astype(bool)])),
                bands,
            )
            for sample_index in range(arrays["patches"].shape[0])
        ]
    ).astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "sample_features": sample_features,
        "sample_index": np.arange(arrays["patches"].shape[0], dtype=np.int32),
        "point_id": arrays["point_id"].astype(np.int32),
        "feature_names": np.asarray(names, dtype="U64"),
    }
    report = {
        "input_npz": str(npz_path),
        "output_npz": str(output_path),
        "sample_count": int(sample_features.shape[0]),
        "sample_feature_dim": int(sample_features.shape[1]),
        "query_stage_features": bool(args.query_stage_features),
    }

    if args.query_stage_features:
        rows: list[np.ndarray] = []
        row_sample_index: list[int] = []
        row_stage_index: list[int] = []
        row_query_doy: list[int] = []
        for sample_index in range(arrays["patches"].shape[0]):
            for stage_index, query_doy in enumerate(arrays["phenophase_doy"][sample_index].astype(np.int16)):
                if query_doy <= 0:
                    continue
                rows.append(
                    compute_aux_features(
                        arrays["patches"][sample_index],
                        arrays["valid_pixel_mask"][sample_index],
                        arrays["time_mask"][sample_index],
                        arrays["time_doy"][sample_index],
                        float(query_doy),
                        bands,
                    )
                )
                row_sample_index.append(sample_index)
                row_stage_index.append(stage_index)
                row_query_doy.append(int(query_doy))
        payload.update(
            {
                "query_features": np.stack(rows).astype(np.float32),
                "query_sample_index": np.asarray(row_sample_index, dtype=np.int32),
                "query_stage_index": np.asarray(row_stage_index, dtype=np.int16),
                "query_doy": np.asarray(row_query_doy, dtype=np.int16),
            }
        )
        report["query_count"] = int(len(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
