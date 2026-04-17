# Preprocessing And Training Track

## Current Simplified Workflow

1. Read the two PDF files and treat them as the source of truth.
2. Keep raw data under `downloadedRawData/` and do not modify raw TIFF or label contents.
3. Build real 15x15 Sentinel-2 patches, not tabular feature vectors.
4. Save the patch dataset to `artifacts/patches_clean/train_cnn_transformer_15x15.npz`.
5. Save masks and metadata with the dataset.
6. Compute normalization statistics from training data and save them separately.
7. Split train/val by `resolved_region_id` so regions do not overlap.
8. Audit the data before trusting model results.
9. Train one CNN+Transformer model.
10. Run the no-date ablation to check whether stage predictions are actually image-based.

## Dataset Schema

The main NPZ contains:

```text
patches: [N, T, 12, 15, 15]
valid_pixel_mask: [N, T, 12, 15, 15]
band_mask: [N, T, 12]
time_mask: [N, T]
time_doy: [N, T]
time_dates: [N, T]
point_id: [N]
longitude: [N]
latitude: [N]
resolved_region_id: [N]
crop_type_id: [N]
phenophase_doy: [N, 7]
```

The valid-value rule is:

```text
finite and 0.0 < value <= 2.0
```

Invalid pixels are filled with `0.0` in `patches` and preserved in `valid_pixel_mask`.

## Current Audit Findings

Current dataset:

```text
samples: 778
patch shape: [778, 29, 12, 15, 15]
observed valid pixel ratio, excluding padded timesteps: about 0.923
valid-ratio-only crop macro-F1: about 0.370
train samples: 609
val samples: 169
train/val point overlap: 0
train/val region overlap: 0
```

The most important issue:

```text
date-only rice stage accuracy: 1.0
date-only all-crop stage accuracy: 0.934
```

Interpretation:

```text
Near-zero phenophase/stage loss is mostly caused by the query date being highly predictive of the stage label.
It is not enough evidence that the model learned phenology from image patches.
```

Other risks:

```text
Each point becomes 7 query rows, so crop labels are repeated 7 times.
The dataset is small for a 3M-parameter model.
The observed invalid-pixel ratio is about 7.68%. Most invalid observed pixels come from complete zero/nodata band-patches, while padded timesteps are separate and should not be counted as invalid observed pixels.
```

## Files Kept

```text
configs/preprocess.json
configs/train.json
scripts/preprocess.py
scripts/audit.py
scripts/train.py
preprocessing/
data/query_dataset_npz.py
models/
training/query_engine.py
```

## Files Removed From Active Workflow

Submission-only and old-baseline files were removed from the active repo structure:

```text
Dockerfile
run.sh
inference/
old train_baseline path
old date-regression CNN+Transformer path
multiple old model/train configs
old notebook that referenced stale commands
```

Reason:

```text
The project was too hard to reason about. We need one clean training path before returning to final submission packaging.
```


Important mask clarification:

```text
observed valid pixel ratio excluding padding: 0.923
observed invalid pixel ratio excluding padding: 0.0768
all-array valid ratio including padded timesteps: 0.566
padding pixel ratio of NPZ array: 0.3865
full-invalid observed band-patches: 12,564
perfect observed band-patches: 152,902
partial-invalid observed band-patches: 631
```

The `0.566` number is not the real invalid-pixel rate. It counts padded timesteps as invalid. Use the observed-only ratio from `scripts/audit.py`.

## Commands

Rebuild data:

```bash
python3 scripts/preprocess.py --config configs/preprocess.json
```

Audit data:

```bash
python3 scripts/audit.py
```

Train normal model:

```bash
python3 scripts/train.py --config configs/train.json
```

Train no-date ablation:

```bash
python3 scripts/train.py --config configs/train.json --no-query-date
```

## Decision Rule

If normal stage performance is high but no-date stage performance drops strongly, the model is using date as a shortcut.

If crop macro-F1 remains strong in both runs, crop learning is probably image-based.

If both crop and stage are unrealistically high, next step is to train a much smaller sanity model and/or an LSTM/RNN baseline to check whether the issue is model capacity or data leakage.


## 1-epoch smoke test

Normal model with query date, CPU smoke run:

```text
val_crop_macro_f1: 0.999
val_rice_stage_macro_f1: 0.120
```

No-date ablation, CPU smoke run:

```text
val_crop_macro_f1: 0.995
val_rice_stage_macro_f1: 0.090
```

Interpretation: after only 1 epoch, crop is already very easy for the CNN+Transformer, while stage is still poor. The previous very high stage score after many epochs should be compared against the date-only baseline before trusting it.

## Fast Date/Leakage Tests

Added fast experiment controls:

```text
configs/train_fast.json
scripts/run_fast_checks.py
scripts/train.py --no-query-date
scripts/train.py --no-time-date
scripts/train.py --shuffle-labels
```

One-epoch CPU fast-model results:

```text
normal_dates val_crop_macro_f1: 0.995
normal_dates val_rice_stage_macro_f1: 0.410
no_query_date val_crop_macro_f1: 0.995
no_query_date val_rice_stage_macro_f1: 0.064
no_query_or_time_date val_crop_macro_f1: 0.989
no_query_or_time_date val_rice_stage_macro_f1: 0.042
shuffled_train_labels val_crop_macro_f1: 0.120
```

Interpretation:

```text
The query date strongly drives phenophase-stage prediction.
Crop prediction remains high even without query/acquisition date encoding, so crop seems mostly image-based.
When train labels are shuffled, validation crop macro-F1 collapses to majority-class behavior, so there is no obvious label leakage through the code path.
```
