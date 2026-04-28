from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. In Colab run with a PyTorch runtime.") from exc

from data.query_dataset_npz import QueryDatePatchDataset
from models.model_factory import build_model, build_model_config, normalize_model_type
from scripts.train import resolve_path, select_device
from training.stage_decoding import maybe_decode_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage robustness under deterministic query/time DOY shifts.")
    parser.add_argument("--config", type=Path, required=True, help="Training config used to build the validation dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to evaluate.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--shift-days", type=float, default=30.0)
    parser.add_argument("--stage-postprocess", type=str, default=None)
    return parser.parse_args()


def _macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    scores: list[float] = []
    for label in range(num_classes):
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return sum(scores) / max(len(scores), 1)


def _scenario_dataset(config: dict[str, Any], split: str, time_shift_days: float, query_shift_days: float) -> QueryDatePatchDataset:
    use_aux_features = bool(config.get("use_aux_features", False))
    aux_feature_set = str(config.get("aux_feature_set", "summary"))
    use_spectral_indices = bool(config.get("use_spectral_indices", False))
    spectral_index_stats_json = config.get("spectral_index_stats_json")
    return QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]),
        split=split,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=use_aux_features,
        aux_feature_set=aux_feature_set,
        use_spectral_indices=use_spectral_indices,
        spectral_index_stats_json=resolve_path(spectral_index_stats_json) if spectral_index_stats_json else None,
        use_relative_doy=bool(config.get("use_relative_doy", False)),
        fixed_time_shift_days=float(time_shift_days),
        fixed_query_doy_shift_days=float(query_shift_days),
    )


def _evaluate_dataset(
    model: torch.nn.Module,
    dataset: QueryDatePatchDataset,
    device: torch.device,
    batch_size: int,
    stage_postprocess: str,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    crop_true: list[int] = []
    crop_pred: list[int] = []
    stage_true: list[int] = []
    stage_weight: list[float] = []
    point_ids: list[int] = []
    query_doys: list[float] = []
    stage_logits_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(
                batch["patches"],
                batch["time_mask"],
                batch["time_doy"],
                batch["query_doy"],
                batch.get("aux_features"),
                batch.get("query_doy_mask"),
            )
            crop_true.extend(batch["crop_type_id"].detach().cpu().tolist())
            crop_pred.extend(outputs["crop_logits"].argmax(dim=1).detach().cpu().tolist())
            stage_true.extend(batch["phenophase_stage_id"].detach().cpu().tolist())
            stage_weight.extend(batch["stage_loss_weight"].detach().cpu().tolist())
            point_ids.extend(batch["point_id"].detach().cpu().tolist())
            query_doys.extend(batch["query_doy"].detach().cpu().tolist())
            stage_logits_all.append(outputs["stage_logits"].detach().cpu())

    stage_logits = torch.cat(stage_logits_all, dim=0) if stage_logits_all else torch.empty((0, 7), dtype=torch.float32)
    stage_pred = maybe_decode_stages(
        stage_logits,
        torch.tensor(point_ids, dtype=torch.long),
        torch.tensor(query_doys, dtype=torch.float32),
        mode=stage_postprocess,
    ).tolist()
    count = max(float(len(crop_true)), 1.0)
    rice_indices = [index for index, weight in enumerate(stage_weight) if float(weight) > 0.0]
    rice_truth = [int(stage_true[index]) for index in rice_indices]
    rice_pred = [int(stage_pred[index]) for index in rice_indices]
    rice_count = max(float(len(rice_indices)), 1.0)
    joint_correct = sum(
        1.0
        for crop_truth, crop_guess, stage_truth, stage_guess in zip(crop_true, crop_pred, stage_true, stage_pred)
        if int(crop_truth) == int(crop_guess) and int(stage_truth) == int(stage_guess)
    )
    return {
        "queries": float(len(crop_true)),
        "crop_macro_f1": _macro_f1([int(v) for v in crop_true], [int(v) for v in crop_pred], 3),
        "stage_accuracy_all_crops": sum(1.0 for truth, pred in zip(stage_true, stage_pred) if int(truth) == int(pred)) / count,
        "rice_stage_accuracy": sum(1.0 for truth, pred in zip(rice_truth, rice_pred) if truth == pred) / rice_count,
        "rice_stage_macro_f1": _macro_f1(rice_truth, rice_pred, 7),
        "joint_accuracy": joint_correct / count,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    device = select_device(args.device)

    model_type = normalize_model_type(checkpoint.get("model_type", config.get("model_type", "query_cnn_transformer")))
    model_config = build_model_config(model_type, checkpoint["model_config"])
    model = build_model(model_type, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_size = int(args.batch_size or config.get("batch_size", 256))
    stage_postprocess = str(
        args.stage_postprocess
        if args.stage_postprocess is not None
        else checkpoint.get("train_config", {}).get("stage_postprocess", config.get("stage_postprocess", "none"))
    )
    shift = float(args.shift_days)
    scenarios = {
        "base": (0.0, 0.0),
        "shift_all_plus": (shift, shift),
        "shift_all_minus": (-shift, -shift),
        "shift_query_plus": (0.0, shift),
        "shift_query_minus": (0.0, -shift),
    }

    results: dict[str, Any] = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "device": str(device),
        "stage_postprocess": stage_postprocess,
        "shift_days": shift,
        "scenarios": {},
    }
    for name, (time_shift, query_shift) in scenarios.items():
        dataset = _scenario_dataset(config, args.split, time_shift, query_shift)
        results["scenarios"][name] = _evaluate_dataset(model, dataset, device, batch_size, stage_postprocess)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
