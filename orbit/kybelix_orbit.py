"""Kybelix C03 spaceborne inference entry point.

Runs the exported C03 crop-type / phenophase model on an NVIDIA Orin node of the
Three-Body Computing Constellation and writes results to the downlink directory.

Design constraints this file exists to satisfy:

  * ARM64 Orin, so no x86 wheels and no assumption that PyTorch is installed.
    Only numpy and onnxruntime are imported; the model is a portable ONNX graph.
  * 300 MB image delta cap, so no pandas, no rasterio/GDAL.
  * Limited downlink, so /output carries a small JSON result plus a compact
    report rather than intermediate tensors.
  * One shot per scheduled pass: every failure path still writes a report to
    /output, because a run that dies silently tells us nothing on the ground.

Usage:

    python3 /workspace/kybelix_orbit.py --output-dir /output
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

CROP_TYPE_NAMES = ["corn", "rice", "soybean"]
PHENOPHASE_NAMES = ["Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence", "Dormancy"]

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model" / "c03.onnx"
DEFAULT_STATS = HERE / "model" / "band_stats.json"
DEFAULT_INPUT = HERE / "sample" / "demo_input.npz"

# Preference order. TensorRT and CUDA are used when the Orin runtime exposes
# them; CPU always works and is fast enough for a 3.3M-parameter model.
PROVIDER_PREFERENCE = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def log(message: str) -> None:
    print(f"[orbit] {message}", flush=True)


def environment_report() -> dict:
    """Everything we would want downlinked if the run misbehaves."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        import onnxruntime  # noqa: PLC0415

        info["onnxruntime"] = onnxruntime.__version__
        info["available_providers"] = list(onnxruntime.get_available_providers())
    except Exception as exc:  # pragma: no cover - reported, not raised
        info["onnxruntime_error"] = repr(exc)
    return info


def normalize(patches: np.ndarray, valid_pixel_mask: np.ndarray, stats_path: Path, bands: list[str]) -> np.ndarray:
    """Train-derived z-score per band, invalid pixels zeroed, mask appended.

    Mirrors data.transforms.NpzPatchNormalizer plus the 12 mask channels the
    24-channel checkpoint expects. Reimplemented here so the flight image does
    not need the training package on its path.
    """
    stats = json.loads(stats_path.read_text())
    stat_bands = list(stats["bands"])
    if stat_bands != bands:
        raise ValueError(f"band order mismatch: bundle {bands} vs stats {stat_bands}")
    center = np.asarray([stats["per_band"][band]["mean"] for band in stat_bands], dtype=np.float32)
    scale = np.maximum(np.asarray([stats["per_band"][band]["std"] for band in stat_bands], dtype=np.float32), 1e-6)

    shape = (1,) * (patches.ndim - 3) + (len(stat_bands), 1, 1)
    out = (patches.astype(np.float32) - center.reshape(shape)) / scale.reshape(shape)
    out = np.where(valid_pixel_mask, out, 0.0).astype(np.float32)
    return np.concatenate([out, valid_pixel_mask.astype(np.float32)], axis=2).astype(np.float32)


def fit_timesteps(array: np.ndarray, timesteps: int, axis: int = 1) -> np.ndarray:
    """Pad or truncate the acquisition axis to the graph's fixed T.

    The exported graph has a static T (nn.MultiheadAttention bakes the sequence
    length into its reshapes). Padded slots are marked invalid by time_mask, so
    this matches how the model was trained on 29 slots with ~18 valid.
    """
    current = array.shape[axis]
    if current == timesteps:
        return array
    if current > timesteps:
        return np.take(array, range(timesteps), axis=axis)
    pad = [(0, 0)] * array.ndim
    pad[axis] = (0, timesteps - current)
    return np.pad(array, pad, mode="constant")


def doy_to_date(year: int, doy: int) -> str:
    date = dt.date(int(year), 1, 1) + dt.timedelta(days=int(doy) - 1)
    return f"{date.year}/{date.month}/{date.day}"


def format_coordinate(value: float) -> str:
    text = f"{float(value):.7f}".rstrip("0").rstrip(".")
    return text or "0"


def make_session(model_path: Path, requested: str | None, threads: int = 2):
    import onnxruntime  # noqa: PLC0415

    available = list(onnxruntime.get_available_providers())
    if requested:
        providers = [requested] if requested in available else ["CPUExecutionProvider"]
        if requested not in available:
            log(f"WARNING requested provider {requested} unavailable; available={available}")
    else:
        providers = [name for name in PROVIDER_PREFERENCE if name in available] or ["CPUExecutionProvider"]

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Thread counts MUST be set explicitly on this node. Its sysfs advertises 12
    # CPUs while the cgroup allows 8, so onnxruntime's default thread-pool sizing
    # calls pthread_setaffinity_np with an out-of-range mask. Older builds only
    # warn ("Specify the number of threads explicitly so the affinity is not
    # set"); 1.19.2+ turns the same condition into a std::vector out-of-range
    # assertion and aborts the process. Setting these keeps ORT off that path.
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    session = onnxruntime.InferenceSession(str(model_path), options, providers=providers)
    log(f"providers available: {available}")
    log(f"providers in use:    {session.get_providers()}")
    log(f"threads: intra={options.intra_op_num_threads} inter={options.inter_op_num_threads}")
    return session


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    model_path = Path(args.model)
    input_path = Path(args.input)
    stats_path = Path(args.stats)
    for path in (model_path, input_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"missing required file: {path}")

    bundle = np.load(input_path, allow_pickle=True)
    bands = [str(band) for band in bundle["bands"]]
    row_point = bundle["row_point"].astype(np.int64)
    row_query_doy = bundle["row_query_doy"].astype(np.float32)
    log(f"bundle: {bundle['patches'].shape[0]} points, {len(row_point)} query rows, bands={len(bands)}")

    session = make_session(model_path, args.provider, args.threads)
    graph_timesteps = int(session.get_inputs()[0].shape[1])
    log(f"graph fixed timesteps: T={graph_timesteps}")

    prep_started = time.time()
    stacked = normalize(bundle["patches"], bundle["valid_pixel_mask"].astype(bool), stats_path, bands)
    stacked = fit_timesteps(stacked, graph_timesteps)
    time_mask = fit_timesteps(bundle["time_mask"].astype(bool), graph_timesteps)
    time_doy = fit_timesteps(bundle["time_doy"].astype(np.float32), graph_timesteps)
    prep_seconds = time.time() - prep_started

    crop_chunks: list[np.ndarray] = []
    stage_chunks: list[np.ndarray] = []
    infer_started = time.time()
    for start in range(0, len(row_point), args.batch_size):
        rows = row_point[start : start + args.batch_size]
        crop_logits, stage_logits = session.run(
            None,
            {
                "patches": stacked[rows],
                "time_mask": time_mask[rows],
                "time_doy": time_doy[rows],
                "query_doy": row_query_doy[start : start + args.batch_size],
            },
        )
        crop_chunks.append(crop_logits)
        stage_chunks.append(stage_logits)
    infer_seconds = time.time() - infer_started

    crop_logits = np.concatenate(crop_chunks).astype(np.float32)
    stage_logits = np.concatenate(stage_chunks).astype(np.float32)
    crop_pred = crop_logits.argmax(1)
    stage_pred = stage_logits.argmax(1)

    longitude = bundle["longitude"]
    latitude = bundle["latitude"]
    year = bundle["year"]
    result: dict[str, list[str]] = {}
    for index, point_index in enumerate(row_point):
        key = (
            f"{format_coordinate(longitude[point_index])}_"
            f"{format_coordinate(latitude[point_index])}_"
            f"{doy_to_date(year[point_index], row_query_doy[index])}"
        )
        result[key] = [CROP_TYPE_NAMES[int(crop_pred[index])], PHENOPHASE_NAMES[int(stage_pred[index])]]

    report: dict = {
        "status": "ok",
        "model": model_path.name,
        "environment": environment_report(),
        "providers_in_use": session.get_providers(),
        "counts": {
            "points": int(bundle["patches"].shape[0]),
            "query_rows": int(len(row_point)),
            "unique_output_keys": int(len(result)),
        },
        "timing_seconds": {
            "preprocess": round(prep_seconds, 3),
            "inference": round(infer_seconds, 3),
            "total": round(time.time() - started, 3),
            "per_row_ms": round(1000.0 * infer_seconds / max(len(row_point), 1), 3),
        },
        "crop_prediction_counts": {
            name: int((crop_pred == index).sum()) for index, name in enumerate(CROP_TYPE_NAMES)
        },
        "stage_prediction_counts": {
            name: int((stage_pred == index).sum()) for index, name in enumerate(PHENOPHASE_NAMES)
        },
    }

    # The metric of record: does flight hardware reproduce the ground result?
    if "reference_crop_logits" in bundle.files:
        ref_crop = bundle["reference_crop_logits"]
        ref_stage = bundle["reference_stage_logits"]
        report["ground_parity"] = {
            "max_abs_delta_crop_logits": float(np.abs(crop_logits - ref_crop).max()),
            "max_abs_delta_stage_logits": float(np.abs(stage_logits - ref_stage).max()),
            "crop_prediction_agreement": float((crop_pred == ref_crop.argmax(1)).mean()),
            "stage_prediction_agreement": float((stage_pred == ref_stage.argmax(1)).mean()),
        }

    # Secondary, and in-sample for the full-data C03 checkpoint: reported for
    # completeness, not as evidence of generalization.
    if "crop_type_id" in bundle.files:
        crop_truth = bundle["crop_type_id"][row_point]
        stage_truth = bundle["row_stage"]
        rice = CROP_TYPE_NAMES.index("rice")
        rice_rows = crop_truth == rice
        report["accuracy_in_sample"] = {
            "crop_accuracy": float((crop_pred == crop_truth).mean()),
            "stage_accuracy_all_crops": float((stage_pred == stage_truth).mean()),
            "rice_stage_accuracy": float((stage_pred[rice_rows] == stage_truth[rice_rows]).mean())
            if rice_rows.any()
            else None,
            "note": "C03 was trained on all labelled points; these rows are in-sample.",
        }

    report["sample_results"] = [
        {"key": key, "prediction": value} for key, value in list(result.items())[:5]
    ]
    return {"result": result, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kybelix C03 spaceborne inference")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--provider", default=None, help="Force an ONNX Runtime execution provider.")
    parser.add_argument("--threads", type=int, default=2, help="intra_op threads; must be explicit on the Orin node.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Fail with a non-zero exit code unless orbit output matches the ground reference.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"output directory: {output_dir}")

    try:
        payload = run(args)
    except Exception as exc:
        # Never die without leaving something to downlink.
        report = {
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "environment": environment_report(),
        }
        (output_dir / "orbit_report.json").write_text(json.dumps(report, indent=2))
        log(f"FAILED: {exc!r}")
        log(f"wrote {output_dir / 'orbit_report.json'}")
        return 1

    (output_dir / "result.json").write_text(json.dumps(payload["result"], indent=1))
    (output_dir / "orbit_report.json").write_text(json.dumps(payload["report"], indent=2))

    report = payload["report"]
    log(f"wrote {output_dir / 'result.json'} ({report['counts']['unique_output_keys']} keys)")
    log(f"wrote {output_dir / 'orbit_report.json'}")
    log(f"timing: {report['timing_seconds']}")
    parity = report.get("ground_parity")
    if parity:
        log(
            "ground parity: max|delta| crop={:.3e} stage={:.3e}, agreement crop={:.4f} stage={:.4f}".format(
                parity["max_abs_delta_crop_logits"],
                parity["max_abs_delta_stage_logits"],
                parity["crop_prediction_agreement"],
                parity["stage_prediction_agreement"],
            )
        )
        if args.selftest:
            if parity["crop_prediction_agreement"] < 1.0 or parity["stage_prediction_agreement"] < 1.0:
                log("SELFTEST FAILED: predictions diverge from the ground reference")
                return 2
            if max(parity["max_abs_delta_crop_logits"], parity["max_abs_delta_stage_logits"]) > 1e-3:
                log("SELFTEST FAILED: logit drift exceeds 1e-3")
                return 2
            log("SELFTEST PASSED")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
