# Remote Sensing CNN + Transformer Project

This project builds a clean, reproducible Sentinel-2 `15x15` patch time-series dataset and a baseline CNN + Transformer model scaffold.

The primary dataset is patch-based, not hand-engineered tabular features:

```text
patches: [N, T, 12, 15, 15]
```

## Main Notebook

Use this in Colab or locally:

```text
notebooks/project_pipeline.ipynb
```

The notebook runs preprocessing, shows charts/samples, inspects tensors, and can start a 1-epoch model smoke test if PyTorch is installed.

## Core Commands

Build the clean training dataset:

```bash
python3 scripts/preprocess_data.py --config configs/preprocessing.json
```

Train the baseline after preprocessing:

```bash
python3 scripts/train_baseline.py --config configs/train_baseline.json
```

Extract test patches in the final inference environment:

```bash
python3 scripts/extract_test_patches.py --input-root /input --output-npz /workspace/patches_clean_test/test_cnn_transformer_15x15.npz
```

Run prediction to `/output/result.json` after a model is trained:

```bash
python3 scripts/run_inference.py --checkpoint artifacts/models/cnn_transformer_baseline/model.pt --output-json /output/result.json
```

Important: the original guide/submission manual was removed from this workspace earlier. `inference/write_submission.py` writes a transparent generic JSON, but the exact competition schema must be confirmed before final upload.

## Config Files

```text
configs/preprocessing.json
configs/model_cnn_transformer.json
configs/train_baseline.json
```

`configs/preprocessing.json` now actually controls:

```text
patch.size
patch.bands
patch.valid_min_exclusive
patch.valid_max_inclusive
patch.invalid_fill_value
report.random_seed
report.sample_bands
report.sample_groups
split.val_fraction
split.stratify_by
```

## Raw Data Location

Raw competition data is kept local in:

```text
downloadedRawData/
```

This folder is ignored by Git because it is about `70GB`. I did not edit raw TIFF contents; I only moved the raw folders and label CSV into `downloadedRawData/` and updated `configs/preprocessing.json` to point there.

The compact training artifact is kept trackable for GitHub:

```text
artifacts/patches_clean/train_cnn_transformer_15x15.npz
artifacts/normalization/train_patch_band_stats.json
artifacts/splits/train_val_split.csv
```

## Produced Data

Main dataset:

```text
artifacts/patches_clean/train_cnn_transformer_15x15.npz
```

Normalization stats:

```text
artifacts/normalization/train_patch_band_stats.json
```

Train/validation split:

```text
artifacts/splits/train_val_split.csv
```

Preprocessing report:

```text
artifacts/preprocessing_report/
```

## NPZ Schema

The training NPZ contains:

```text
patches: [N, T, 12, 15, 15]
valid_pixel_mask: [N, T, 12, 15, 15]
band_mask: [N, T, 12]
time_mask: [N, T]
time_doy: [N, T]
time_dates: [N, T]
border_margin_pixels: [N, T, 12]
center_clamped: [N, T, 12]
source_file_index: [N, T, 12]
band_valid_ratio: [N, T, 12]
point_id
longitude
latitude
resolved_region_id
crop_type_id
crop_type_names
phenophase_names
phenophase_doy
bands
schema_version
```

Invalid pixels are not silently trusted. Valid raw values are preserved, invalid positions are filled only in the tensor, and `valid_pixel_mask` records exactly where that happened.

## Model Direction

The baseline model is intentionally simple and modular:

```text
models/cnn_encoder.py          CNN per timestep over [12, 15, 15]
models/temporal_transformer.py DOY encoding and masked temporal pooling
models/cnn_transformer.py      CNN encoder + Transformer + crop/phenophase heads
training/engine.py             multitask training loop
```

The intended architecture is:

```text
[T, 12, 15, 15] -> CNN timestep encoder -> temporal embeddings -> Transformer -> heads
```

Masks are used so padded timesteps and invalid pixels are not treated as normal observations.

## Colab GPU Training

Use the larger GPU-oriented config after cloning the repo. The compact NPZ artifact is configured to be trackable in GitHub:

```bash
python3 scripts/train_baseline.py --config configs/train_colab_gpu.json
```

This config uses a larger CNN + Transformer, mixed precision on CUDA, cosine LR schedule, warmup, gradient clipping, early stopping, and best-checkpoint saving. It keeps `num_workers` at `0` because the compact compressed NPZ is loaded into memory and multiprocessing workers can trigger zip/zlib read issues. If Colab runs out of memory, lower `batch_size` in `configs/train_colab_gpu.json` from `32` to `16` or `8`.

## Latest Local Smoke Test

A local `.venv` was created with system site packages and PyTorch was installed inside it. The 1-epoch baseline smoke test completed on CPU and wrote:

```text
artifacts/models/cnn_transformer_baseline/model.pt
artifacts/models/cnn_transformer_baseline/history.json
```

This confirms the model code runs. Treat the 1-epoch metric as a pipeline check, not as final model quality. The default split is grouped by `resolved_region_id` to reduce spatial leakage.
