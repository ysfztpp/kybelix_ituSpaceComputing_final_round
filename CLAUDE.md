# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Track 1 remote-sensing competition submission:

```
raw Sentinel-2 TIFFs + point/date rows → 15×15 patch time series → CNN + Transformer → result.json
```

Output: crop type (corn / rice / soybean) and phenophase stage (7 classes) per `Longitude_Latitude_Date` key.

## Commands

**Preprocessing** — build the training NPZ from raw TIFFs:
```bash
python3 scripts/preprocess.py --config configs/preprocess.json
```

**Training**:
```bash
python3 scripts/train.py --config configs/train.json
python3 scripts/train.py --config configs/train_submission_date.json   # stronger submission config
```

**Run an experiment suite** (controlled multi-run with config overrides):
```bash
python3 scripts/run_experiments.py --suite configs/experiment_suite.json
python3 scripts/run_experiments.py --suite configs/experiment_suite.json --dry-run   # preview only
```

**Summarize results**:
```bash
python3 scripts/summarize_experiments.py --root artifacts/models --format markdown
python3 scripts/summarize_experiments.py --root artifacts/models/experiments/<suite>
```

**Submission validation**:
```bash
python3 scripts/validate_submission.py
python3 scripts/inspect_checkpoint.py checkpoints/model.pt
```

**Local inference** (simulates platform environment):
```bash
INPUT_ROOT=/path/to/input OUTPUT_DIR=/path/to/output ./run.sh
```

**Data audit**:
```bash
python3 scripts/audit.py
```

**Tests**:
```bash
python3 -m pytest tests/
```

Dependencies: `pip install -r requirements-train.txt` for training; `pip install -r requirements.txt` for inference only (CPU PyTorch).

## Architecture

### Model (`models/`)

`QueryCNNTransformerClassifier` in `models/query_cnn_transformer.py`:

1. **CNN encoder** (`models/cnn_encoder.py`): `PatchCNNEncoder` maps each `[bands, 15, 15]` patch to a 256-d embedding. All timesteps are processed in a single batched call.
2. **Temporal Transformer**: 4-layer, 8-head encoder over the date sequence. Both acquisition DOY and query DOY use sinusoidal positional encodings (`models/temporal_transformer.py`).
3. **Pooling + query fusion**: `MaskedTemporalPool` averages over valid timesteps; the pooled vector is concatenated with the query-date encoding.
4. **Two heads**: `crop_head` (3 classes) and `stage_head` (7 phenophases).
5. **Optional aux branch**: compact spectral/phenology summaries fused after pooling; controlled by `aux_feature_dim` in config. Disabled for the safe baseline.

Forward signature: `patches [B, T, 12, 15, 15]`, `time_mask [B, T]`, `time_doy [B, T]`, `query_doy [B]`.

### Data Pipeline (`preprocessing/`, `data/`, `artifacts/`)

- `scripts/preprocess.py` reads raw TIFFs via `preprocessing/` and writes `artifacts/patches_clean/train_cnn_transformer_15x15.npz`.
- NPZ schema: `patches [N, T, 12, 15, 15]`, `valid_pixel_mask`, `band_mask`, `time_mask [N, T]`, `time_doy [N, T]`, `crop_type_id [N]`, `phenophase_doy [N, 7]`.
- Valid reflectance: `0.0 < value ≤ 2.0`; out-of-range values filled with `0.0` (constants in `preprocessing/constants.py`).
- Normalization stats (`artifacts/normalization/train_patch_band_stats.json`) are computed on training data only and applied at load time by `NpzPatchNormalizer`.
- Train/val split is by `resolved_region_id` — no region appears in both sets.

`data/query_dataset_npz.py:QueryDatePatchDataset` expands each point into one row per known phenophase label (full time series + query DOY → crop + stage).

### Training Loop (`training/`)

`training/query_engine.py:fit_query` drives the loop. Key config knobs:

| Field | Default | Note |
|---|---|---|
| `stage_loss_weight` | 0.6 | Weight on phenophase CE loss |
| `rice_stage_loss_only` | `true` | Stage loss only on rice rows |
| `checkpoint_metric` | `val_score` | Metric used to save best model |
| `scheduler` | `cosine` | With warmup via `warmup_epochs` |
| `use_query_doy` | `true` | **Never disable** — see below |

Each run saves: `model.pt`, `history.json`, `config_resolved.json`, `metrics_summary.json`.

### Submission Path

Platform entrypoint: `run.sh` → `inference.py` → `scripts/submission_inference.py`.

Input: `/input/test_point.csv` (or `points_test.csv`) + `/input/region_test/*.tiff`.
Output: `${OUTPUT_DIR}/result.json` (format: `{"Lon_Lat_Date": ["CropType", "PhenophaseStage"]}`).

The Dockerfile uses CPU PyTorch intentionally to avoid the build timeout on the platform. Duplicate `Longitude_Latitude_Date` keys in test input collapse to one output key (first prediction kept).

## Key Findings That Affect Code Decisions

**Query DOY is mandatory.** C07 ablation: removing it collapsed rice-stage F1 from 1.0 to 0.067. `use_query_doy` must stay `true` in any submission config.

**Stage accuracy alone is not evidence of image learning.** Date-only rice-stage accuracy is 1.0 on this dataset. Always check that the model degrades meaningfully without query date when evaluating new architectures.

**Valid-mask channels must stay.** C06 (no mask channels) dropped val score from 1.0 to 0.974.

**Current `checkpoints/model.pt`** is the safe C00/E1 checkpoint (val score 1.0, val loss 0.063). Do not replace it until a candidate passes `scripts/validate_submission.py`. C03 is the best reproduced current-code candidate.

## Experiment Suite Config Format

Suite JSONs (`configs/experiment_suite_*.json`) follow:
```json
{
  "name": "suite_name",
  "base_config": "configs/train_submission_date.json",
  "output_root": "artifacts/models/experiments/suite_name",
  "experiments": [
    { "name": "C10_...", "reason": "...", "overrides": { "epochs": 80, "model": { "dropout": 0.2 } } }
  ]
}
```
Overrides deep-merge into the base config. The runner skips experiments whose `model.pt` already exists unless `--force` is passed.

## Model Selection Rule

When choosing between checkpoints: platform result > higher val score > higher rice-stage macro F1 > higher crop macro F1 > lower val loss > smaller train/val gap > simpler model.
