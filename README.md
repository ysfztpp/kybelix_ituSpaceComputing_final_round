# Kybelix ITU Space Computing Final Round

Clean, reproducible Track 1 remote-sensing project for:

```text
raw Sentinel-2 TIFFs + point/date rows -> 15x15 patch time series -> CNN + Transformer -> result.json
```

The repository now contains both the research/training pipeline and a tested inference package.

## Current Status

Successful platform test:

```text
job id: jb-aitrain-155347848884412160
submitted commit: 28d94fb
output file: /mnt/si000886fq1w/default/output/result.json
runtime result rows: 930
runtime query rows: 942
```

The model ran on CPU in the successful submission because the large CUDA PyTorch image exceeded the project build timeout. CPU inference completed successfully.

First platform result snapshot:

```text
shown values: 0.969318, 0.862764, 0.0197
```

## Important Finding

Phenophase stage is strongly date-driven in this dataset.

Current audit result:

```text
date-only rice stage accuracy: 1.0
date-only all-crop stage accuracy: 0.934
```

So very high stage performance is not proof that the model learned phenology only from pixels. The query date is an important competition input and is used by the submission model.

## Main Dataset

Main training artifact:

```text
artifacts/patches_clean/train_cnn_transformer_15x15.npz
```

Patch tensor schema:

```text
patches: [N, T, 12, 15, 15]
```

Meaning:

```text
N = sample point
T = Sentinel-2 acquisition dates
12 = Sentinel-2 bands
15x15 = image patch
```

Known data facts:

```text
samples: 778
patch shape: [778, 29, 12, 15, 15]
observed valid pixel ratio excluding padded timesteps: about 0.923
observed invalid pixel ratio excluding padded timesteps: about 0.0768
train/val point overlap: 0
train/val region overlap: 0
```

## Submission Files

The inference package uses:

```text
run.sh
inference.py
scripts/submission_inference.py
scripts/validate_submission.py
configs/submission.json
checkpoints/model.pt
artifacts/normalization/train_patch_band_stats.json
```

Runtime input expected by the platform:

```text
/input/test_point.csv or /input/points_test.csv
/input/region_test/*.tiff
```

Runtime output:

```text
${OUTPUT_DIR}/result.json
```

If `OUTPUT_DIR` is not set, local fallback is:

```text
/output/result.json
```

The submission logs include:

```text
Python and torch version
CUDA availability and device count
input/output paths
checkpoint metadata
patch extraction report
query rows vs unique output keys
duplicate output-key counts
crop/stage prediction counts
sample result rows
```


## Dependencies

For submission/inference image:

```bash
pip install -r requirements.txt
```

For local training and reports:

```bash
pip install -r requirements-train.txt
```

The Dockerfile installs CPU PyTorch separately to avoid the large CUDA image build timeout.

## Commands

Validate the submission package:

```bash
python3 scripts/validate_submission.py
```

Run local inference with platform-like folders:

```bash
INPUT_ROOT=/path/to/input OUTPUT_DIR=/path/to/output ./run.sh
```

Build/rebuild the training dataset:

```bash
python3 scripts/preprocess.py --config configs/preprocess.json
```

Audit data and leakage risks:

```bash
python3 scripts/audit.py
```

Train the main CNN+Transformer:

```bash
python3 scripts/train.py --config configs/train.json
```

Train the stronger submission configuration:

```bash
python3 scripts/train.py --config configs/train_submission_date.json
```

Train the date-aware model with auxiliary phenology/index features:

```bash
python3 scripts/train.py --config configs/train_submission_date_aux_features.json
```

Run the controlled experiment suite:

```bash
python3 scripts/run_experiments.py --suite configs/experiment_suite.json
```

Preview the suite without training:

```bash
python3 scripts/run_experiments.py --suite configs/experiment_suite.json --dry-run
```

Summarize experiment histories:

```bash
python3 scripts/summarize_experiments.py --root artifacts/models/experiments/date_query_feature_search_v1
```

Export optional auxiliary features for analysis:

```bash
python3 scripts/extract_aux_features.py --query-stage-features
```

Run the no-query-date ablation:

```bash
python3 scripts/train.py --config configs/train.json --no-query-date
```

## Repository Structure

```text
configs/                  preprocessing, training, and submission configs
data/                     PyTorch datasets and transforms
docs/                     experiment plan and research notes
models/                   CNN encoder + temporal Transformer
preprocessing/            TIFF inventory, point mapping, patch extraction
scripts/                  preprocess, audit, train, inference, validation
training/                 training loop and metrics
artifacts/normalization/  train-only normalization stats
artifacts/patches_clean/  compact training NPZ artifact
checkpoints/model.pt      trained submission checkpoint
run.sh                    platform entrypoint
inference.py              root inference wrapper
```

## Notes

- The submission Dockerfile intentionally uses a lightweight Python base image and CPU PyTorch to avoid the project build timeout.
- Duplicate `Longitude_Latitude_Date` rows in test input collapse to one JSON key; the code logs duplicate counts and keeps the first deterministic prediction.
- `.gitlab-ci.yml` in this GitHub repo is only a lightweight reference. The official GitLab project uses its own CI template.
- See `docs/EXPERIMENT_PLAN.md` before starting the next Colab sweep.
