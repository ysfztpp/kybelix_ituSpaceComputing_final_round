from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.query_dataset_npz import QueryDatePatchDataset
from models.model_factory import build_model, build_model_config, normalize_model_type
from training.query_engine import run_query_epoch

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for band importance analysis.") from exc


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    model_config = build_model_config(model_type, payload["model_config"])
    model = build_model(model_type, model_config).to(device)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, payload, model_type, model_config


def build_dataset(config: dict[str, Any], disabled_bands: list[str]) -> QueryDatePatchDataset:
    return QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]) if config.get("split_csv") else None,
        split=str(config.get("split", "val")) if config.get("split_csv") else None,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=bool(config.get("use_aux_features", False)),
        aux_feature_set=str(config.get("aux_feature_set", "summary")),
        use_spectral_indices=bool(config.get("use_spectral_indices", False)),
        spectral_index_stats_json=resolve_path(config["spectral_index_stats_json"]) if config.get("spectral_index_stats_json") else None,
        use_relative_doy=bool(config.get("use_relative_doy", False)),
        disabled_bands=disabled_bands,
    )


def build_loader(dataset: QueryDatePatchDataset, config: dict[str, Any], device: torch.device) -> DataLoader:
    num_workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(config.get("batch_size", 1024)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def add_scores(metrics: dict[str, float]) -> dict[str, float]:
    row = dict(metrics)
    row["competition_score"] = 0.4 * float(row["crop_macro_f1"]) + 0.6 * float(row["rice_stage_macro_f1"])
    row["competition_score_consistent"] = 0.4 * float(row["crop_macro_f1_consistent"]) + 0.6 * float(row["rice_stage_macro_f1"])
    return row


def evaluate_case(name: str, disabled_bands: list[str], model, config: dict[str, Any], device: torch.device, baseline: dict[str, float] | None) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = build_dataset(config, disabled_bands)
    metrics = add_scores(
        run_query_epoch(
            model=model,
            loader=build_loader(dataset, config, device),
            optimizer=None,
            device=device,
            train=False,
            stage_loss_weight=float(config.get("stage_loss_weight", 0.6)),
            crop_loss_weight=float(config.get("crop_loss_weight", 1.0)),
            amp=bool(config.get("amp", False)) and device.type == "cuda",
            stage_postprocess=str(config.get("stage_postprocess", "none")),
        )
    )
    row: dict[str, Any] = {
        "name": name,
        "disabled_bands": disabled_bands,
        "rows": len(dataset),
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    if baseline is not None:
        keys = ("competition_score", "competition_score_consistent", "crop_macro_f1", "rice_stage_macro_f1", "rice_stage_accuracy", "rice_joint_accuracy", "loss")
        row["delta_vs_baseline"] = {key: float(metrics[key]) - float(baseline[key]) for key in keys}
    return row


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# C03 Band Importance",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- split: `{report['split']}`",
        f"- device: `{report['device']}`",
        f"- order: {', '.join(report['candidate_bands_order'])}",
        "",
        "## Single-Band Removal",
        "",
        "|rank|band|score_delta|crop_f1_delta|rice_stage_f1_delta|loss_delta|seconds|",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    singles = [row for row in report["cases"] if row["name"].startswith("single:")]
    singles.sort(key=lambda row: row["delta_vs_baseline"]["competition_score"], reverse=True)
    for rank, row in enumerate(singles, start=1):
        delta = row["delta_vs_baseline"]
        lines.append(
            f"|{rank}|{row['disabled_bands'][0]}|{delta['competition_score']:.6f}|{delta['crop_macro_f1']:.6f}|"
            f"{delta['rice_stage_macro_f1']:.6f}|{delta['loss']:.6f}|{row['seconds']:.2f}|"
        )
    lines.extend(["", "## Cumulative Removal", "", "|step|disabled_bands|score_delta|rice_stage_f1_delta|loss_delta|", "|---:|---|---:|---:|---:|"])
    for row in [case for case in report["cases"] if case["name"].startswith("cumulative:")]:
        delta = row["delta_vs_baseline"]
        lines.append(f"|{len(row['disabled_bands'])}|{', '.join(row['disabled_bands'])}|{delta['competition_score']:.6f}|{delta['rice_stage_macro_f1']:.6f}|{delta['loss']:.6f}|")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic C03 band-ablation importance analysis.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "band_importance_c03_p5_a100.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only the first N candidate bands.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(resolve_path(args.config).read_text())
    if args.device:
        config["device"] = args.device
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    device = select_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(config.get("allow_tf32", True))

    model, payload, model_type, model_config = load_model(resolve_path(config["checkpoint"]), device)
    baseline_case = evaluate_case("baseline", [], model, config, device, None)
    baseline = baseline_case["metrics"]
    order = [str(band) for band in config["candidate_bands_order"]]
    if args.limit is not None:
        order = order[: int(args.limit)]

    cases = [baseline_case]
    for band in order:
        cases.append(evaluate_case(f"single:{band}", [band], model, config, device, baseline))
    cumulative: list[str] = []
    for band in order:
        cumulative.append(band)
        cases.append(evaluate_case(f"cumulative:{'+'.join(cumulative)}", list(cumulative), model, config, device, baseline))
    for name, bands in dict(config.get("band_groups", {})).items():
        cases.append(evaluate_case(f"group:{name}", [str(band) for band in bands], model, config, device, baseline))

    report = {
        "config": str(resolve_path(args.config)),
        "checkpoint": str(resolve_path(config["checkpoint"])),
        "checkpoint_epoch": payload.get("epoch"),
        "model_type": model_type,
        "model_config": getattr(model_config, "__dict__", {}),
        "split": str(config.get("split", "val")),
        "device": str(device),
        "candidate_bands_order": order,
        "cases": cases,
    }
    output_json = resolve_path(args.output_json or config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    if config.get("output_markdown") or args.output_markdown:
        write_markdown(resolve_path(args.output_markdown or config["output_markdown"]), report)
    print(json.dumps({"output_json": str(output_json), "cases": len(cases), "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
