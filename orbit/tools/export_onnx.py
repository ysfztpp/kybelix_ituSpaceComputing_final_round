"""Export a trained Kybelix checkpoint to ONNX for spaceborne (ARM64/Orin) inference.

Run from the project root:

    python3 orbit/tools/export_onnx.py \
        --checkpoint ../project_organized/checkpoints/c03_full_data_model.pt \
        --output orbit/model/c03.onnx

The exported graph takes the same four tensors the PyTorch model takes and
returns crop/stage logits, so the on-orbit runtime needs no model code at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model_factory import build_model, build_model_config, normalize_model_type  # noqa: E402


class ExportWrapper(nn.Module):
    """Tuple-output wrapper. ONNX handles tuples more predictably than dicts."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        patches: torch.Tensor,
        time_mask: torch.Tensor,
        time_doy: torch.Tensor,
        query_doy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(patches, time_mask, time_doy, query_doy)
        return out["crop_logits"], out["stage_logits"]


def load_checkpoint(checkpoint: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    config = build_model_config(model_type, payload["model_config"])
    model = build_model(model_type, config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # The nested-tensor fast path in nn.TransformerEncoder is not traceable.
    # Disabling it keeps numerics identical while making the graph exportable.
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoder):
            module.enable_nested_tensor = False
            module.use_nested_tensor = False

    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--timesteps", type=int, default=29, help="Example T used for tracing; T stays dynamic.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint(checkpoint)
    config = model.config
    in_channels = int(config.in_channels)
    patch_size = int(config.patch_size)
    if int(config.aux_feature_dim) > 0:
        raise SystemExit("Checkpoints with aux features are not supported by this exporter.")

    batch, timesteps = 2, int(args.timesteps)
    example = (
        torch.randn(batch, timesteps, in_channels, patch_size, patch_size),
        torch.ones(batch, timesteps, dtype=torch.bool),
        torch.full((batch, timesteps), 180.0),
        torch.full((batch,), 200.0),
    )

    wrapper = ExportWrapper(model).eval()
    with torch.no_grad():
        reference = wrapper(*example)

    torch.onnx.export(
        wrapper,
        example,
        str(output),
        input_names=["patches", "time_mask", "time_doy", "query_doy"],
        output_names=["crop_logits", "stage_logits"],
        # Only batch is dynamic. nn.MultiheadAttention bakes the sequence length
        # into its reshapes, and T is fixed by the trained contract anyway:
        # exactly `timesteps` acquisition slots, with time_mask marking which are
        # real. The runtime pads/truncates to this T. Static T also keeps the
        # TensorRT engine simple on Orin.
        dynamic_axes={
            "patches": {0: "batch"},
            "time_mask": {0: "batch"},
            "time_doy": {0: "batch"},
            "query_doy": {0: "batch"},
            "crop_logits": {0: "batch"},
            "stage_logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    onnx.checker.check_model(onnx.load(str(output)))

    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    got = session.run(
        None,
        {
            "patches": example[0].numpy(),
            "time_mask": example[1].numpy(),
            "time_doy": example[2].numpy(),
            "query_doy": example[3].numpy(),
        },
    )
    deltas = [float(np.abs(a.numpy() - b).max()) for a, b in zip(reference, got)]
    print(f"[export] wrote {output} ({output.stat().st_size / 1e6:.2f} MB, opset {args.opset})")
    print(f"[export] max |torch - onnx|: crop={deltas[0]:.3e} stage={deltas[1]:.3e}")
    if max(deltas) > 1e-4:
        raise SystemExit(f"ONNX parity check failed: {deltas}")

    # Batch must be dynamic; T must be exactly `timesteps`. Assert both, because a
    # silent shape regression here would only surface on the satellite.
    for size in (1, 7, 64):
        session.run(
            None,
            {
                "patches": np.zeros((size, timesteps, in_channels, patch_size, patch_size), np.float32),
                "time_mask": np.ones((size, timesteps), bool),
                "time_doy": np.full((size, timesteps), 180.0, np.float32),
                "query_doy": np.full((size,), 200.0, np.float32),
            },
        )
    print(f"[export] dynamic batch verified (1/7/64) at fixed T={timesteps}")

    metadata = {
        "source_checkpoint": checkpoint.name,
        "epoch": payload.get("epoch"),
        "task": payload.get("task"),
        "model_config": payload["model_config"],
        "in_channels": in_channels,
        "patch_size": patch_size,
        "opset": args.opset,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "git": payload.get("git"),
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[export] wrote {meta_path}")


if __name__ == "__main__":
    main()
