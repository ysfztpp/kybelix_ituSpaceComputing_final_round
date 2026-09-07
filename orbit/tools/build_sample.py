"""Build the self-contained input bundle that ships inside the spaceborne image.

The bundle lets the on-orbit run execute without depending on the satellite's
`/rs` mount, which is what makes verification reproducible. It carries:

  * a stratified subset of labelled points (patches + masks + acquisition DOYs),
  * the query rows derived exactly as in training (one row per phenophase DOY),
  * ground-computed reference logits, so the orbit run can prove it reproduces
    the ground result bit-for-bit rather than merely "looking plausible".

Run from the project root:

    python3 orbit/tools/build_sample.py --points-per-crop 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.transforms import NpzPatchNormalizer  # noqa: E402

CROP_TYPE_NAMES = ["corn", "rice", "soybean"]
PHENOPHASE_NAMES = ["Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence", "Dormancy"]


def select_points(crop_type_id: np.ndarray, per_crop: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for crop_id in range(len(CROP_TYPE_NAMES)):
        candidates = np.flatnonzero(crop_type_id == crop_id)
        take = min(per_crop, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def build_query_rows(points: np.ndarray, phenophase_doy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One query row per positive phenophase DOY, matching QueryDatasetNPZ."""
    row_point: list[int] = []
    row_stage: list[int] = []
    row_doy: list[int] = []
    for local_index, point_index in enumerate(points):
        for stage_index, doy in enumerate(phenophase_doy[point_index].astype(np.int16)):
            if doy <= 0:
                continue
            row_point.append(local_index)
            row_stage.append(int(stage_index))
            row_doy.append(int(doy))
    return (
        np.asarray(row_point, dtype=np.int32),
        np.asarray(row_stage, dtype=np.int16),
        np.asarray(row_doy, dtype=np.int16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="artifacts/patches_clean/train_cnn_transformer_15x15.npz")
    parser.add_argument("--onnx", default="orbit/model/c03.onnx")
    parser.add_argument("--stats", default="artifacts/normalization/train_patch_band_stats.json")
    parser.add_argument("--output", default="orbit/sample/demo_input.npz")
    parser.add_argument("--points-per-crop", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arrays = np.load(args.npz, allow_pickle=True)
    points = select_points(arrays["crop_type_id"], args.points_per_crop, args.seed)
    row_point, row_stage, row_doy = build_query_rows(points, arrays["phenophase_doy"])

    patches = arrays["patches"][points].astype(np.float32)
    valid_pixel_mask = arrays["valid_pixel_mask"][points]
    time_mask = arrays["time_mask"][points]
    time_doy = arrays["time_doy"][points].astype(np.int16)

    # Output keys are "lon_lat_YYYY/M/D", so the runtime needs the acquisition
    # year to turn a query DOY back into a calendar date.
    years = np.asarray(
        [
            int(next(str(d)[:4] for d in arrays["time_dates"][point_index] if str(d)[:4].isdigit()))
            for point_index in points
        ],
        dtype=np.int16,
    )

    # Ground reference: run the exported graph here so the satellite has
    # something exact to compare against.
    import onnxruntime  # noqa: PLC0415

    normalizer = NpzPatchNormalizer(args.stats)
    stacked = np.concatenate(
        [normalizer(patches, valid_pixel_mask.astype(bool)), valid_pixel_mask.astype(np.float32)], axis=2
    ).astype(np.float32)

    session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    crop_logits, stage_logits = session.run(
        None,
        {
            "patches": stacked[row_point],
            "time_mask": time_mask[row_point].astype(bool),
            "time_doy": time_doy[row_point].astype(np.float32),
            "query_doy": row_doy.astype(np.float32),
        },
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        patches=patches,
        valid_pixel_mask=valid_pixel_mask,
        time_mask=time_mask,
        time_doy=time_doy,
        longitude=arrays["longitude"][points],
        latitude=arrays["latitude"][points],
        year=years,
        crop_type_id=arrays["crop_type_id"][points],
        bands=arrays["bands"],
        row_point=row_point,
        row_stage=row_stage,
        row_query_doy=row_doy,
        reference_crop_logits=crop_logits.astype(np.float32),
        reference_stage_logits=stage_logits.astype(np.float32),
    )

    crop_pred = crop_logits.argmax(1)
    truth = arrays["crop_type_id"][points][row_point]
    manifest = {
        "points": int(len(points)),
        "query_rows": int(len(row_point)),
        "timesteps": int(patches.shape[1]),
        "bytes": int(output.stat().st_size),
        "crop_accuracy_ground": float((crop_pred == truth).mean()),
        "stage_accuracy_ground": float((stage_logits.argmax(1) == row_stage).mean()),
        "note": "Labels are in-sample for the full-data C03 checkpoint. The orbit metric of record is ground-vs-orbit logit parity, not accuracy.",
    }
    Path(output.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(f"[sample] wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    print(f"[sample] {manifest['points']} points, {manifest['query_rows']} query rows, T={manifest['timesteps']}")
    print(f"[sample] ground crop acc {manifest['crop_accuracy_ground']:.4f}, stage acc {manifest['stage_accuracy_ground']:.4f}")


if __name__ == "__main__":
    main()
