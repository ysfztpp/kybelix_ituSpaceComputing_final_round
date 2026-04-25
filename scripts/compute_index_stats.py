"""Compute normalization stats (mean/std) for NDVI, EVI, LSWI from the training NPZ.

Run once before training with use_spectral_indices=true:
    python scripts/compute_index_stats.py

Output: artifacts/normalization/spectral_index_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Band order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
_B02, _B04, _B08, _B11 = 1, 3, 7, 10
_EPS = 1e-6


def compute_indices(patches: np.ndarray, valid_mask: np.ndarray):
    """Return clipped NDVI/EVI/LSWI and their validity masks.

    patches:    [..., 12, H, W] raw reflectance float32
    valid_mask: [..., 12, H, W] bool
    Returns: (ndvi, evi, lswi) and (ndvi_valid, evi_valid, lswi_valid) — same spatial shape
    """
    b02 = patches[..., _B02, :, :]
    b04 = patches[..., _B04, :, :]
    b08 = patches[..., _B08, :, :]
    b11 = patches[..., _B11, :, :]

    ndvi = (b08 - b04) / (b08 + b04 + _EPS)
    evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0 + _EPS)
    lswi = (b08 - b11) / (b08 + b11 + _EPS)

    ndvi = np.clip(ndvi, -1.0, 1.0)
    evi = np.clip(evi, -1.0, 1.5)
    lswi = np.clip(lswi, -1.0, 1.0)

    m02 = valid_mask[..., _B02, :, :].astype(bool)
    m04 = valid_mask[..., _B04, :, :].astype(bool)
    m08 = valid_mask[..., _B08, :, :].astype(bool)
    m11 = valid_mask[..., _B11, :, :].astype(bool)

    return (ndvi, evi, lswi), (m04 & m08, m02 & m04 & m08, m08 & m11)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="artifacts/patches_clean/train_cnn_transformer_15x15.npz")
    parser.add_argument("--split_csv", default="artifacts/splits/train_val_split.csv")
    parser.add_argument("--output", default="artifacts/normalization/spectral_index_stats.json")
    args = parser.parse_args()

    npz_path = Path(args.npz) if Path(args.npz).is_absolute() else ROOT / args.npz
    split_csv_path = Path(args.split_csv) if Path(args.split_csv).is_absolute() else ROOT / args.split_csv
    output_path = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output

    print(f"Loading {npz_path} ...")
    with np.load(npz_path, allow_pickle=False) as d:
        patches = d["patches"]        # [N, T, 12, 15, 15]
        valid_mask = d["valid_pixel_mask"]  # [N, T, 12, 15, 15]

    if split_csv_path.exists():
        import pandas as pd
        df = pd.read_csv(split_csv_path)
        train_idx = df.loc[df["split"] == "train", "sample_index"].to_numpy(dtype=np.int64)
        print(f"Using {len(train_idx)} training samples (of {len(patches)} total)")
        patches = patches[train_idx]
        valid_mask = valid_mask[train_idx]
    else:
        print("No split CSV — using all samples")

    print("Computing indices ...")
    (ndvi, evi, lswi), (ndvi_v, evi_v, lswi_v) = compute_indices(patches, valid_mask)

    def stats(vals: np.ndarray, mask: np.ndarray) -> dict:
        v = vals[mask].astype(np.float64)
        return {"mean": float(v.mean()), "std": float(v.std()), "valid_count": int(mask.sum())}

    result = {"NDVI": stats(ndvi, ndvi_v), "EVI": stats(evi, evi_v), "LSWI": stats(lswi, lswi_v)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    for name, s in result.items():
        print(f"  {name}: mean={s['mean']:.4f}  std={s['std']:.4f}  n={s['valid_count']:,}")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
