# Kybelix ITU Space Computing Final Round

This repository is now intentionally simple. It focuses on one training workflow:

```text
raw Sentinel-2 TIFFs + point labels -> 15x15 patch dataset -> audit -> CNN + Transformer
```

Submission/Docker files were removed for now because we first need to understand the data and the suspicious training results.

## What The PDFs Say

`Space Intelligence Empowering Zero Hunger Track_Task Description of Final Round_en.pdf` defines the data and scoring:

- Training labels are rows with `point_id`, `Longitude`, `Latitude`, `phenophase_date`, `crop_type`, `phenophase_name`.
- The target is crop type plus phenophase stage for point-date rows.
- Crop score uses macro-F1 over `corn`, `rice`, `soybean`.
- Rice phenology score uses strict crop+stage exact matching for rice samples.

`Round2 Project Submission Manual_en_0407.pdf` says the final platform is inference-only, but we are not focusing on submission files right now.

## Important Finding

The model looked too good because phenophase stage is almost predictable from date alone.

Current audit result:

```text
date-only rice stage accuracy: 1.0
date-only all-crop stage accuracy: 0.934
```

This means near-zero stage loss is not reliable evidence that the model learned phenology from image pixels. It is mostly learning calendar timing. This is not a code crash or train/val region overlap issue; it is a dataset/label-design issue.

## Current Data

Main artifact:

```text
artifacts/patches_clean/train_cnn_transformer_15x15.npz
```

Main tensor:

```text
patches: [N, T, 12, 15, 15]
```

Meaning:

```text
N = sample point
T = acquisition dates
12 = Sentinel-2 bands
15x15 = image patch
```

The dataset also contains masks, dates, coordinates, crop labels, and phenophase DOY labels.

Known current facts:

```text
samples: 778
patch shape: [778, 29, 12, 15, 15]
observed valid pixel ratio, excluding padded timesteps: about 0.923
valid-ratio-only crop macro-F1: about 0.370
train/val point overlap: 0
train/val region overlap: 0
```

## Simple Commands

Build/rebuild the dataset:

```bash
python3 scripts/preprocess.py --config configs/preprocess.json
```

Audit the dataset and leakage risks:

```bash
python3 scripts/audit.py
```

Train the CNN+Transformer with query date enabled:

```bash
python3 scripts/train.py --config configs/train.json
```

Run the important no-date ablation:

```bash
python3 scripts/train.py --config configs/train.json --no-query-date
```

If the no-date model performs much worse on rice stage, then the original stage result was calendar-driven.

## Current File Structure

```text
configs/preprocess.json       one preprocessing config
configs/train.json            one training config
scripts/preprocess.py         builds the patch dataset
scripts/audit.py              checks data, split, date-only baseline, leakage risks
scripts/train.py              trains the single CNN+Transformer model
preprocessing/                reusable TIFF/patch/mapping code
data/query_dataset_npz.py     PyTorch dataset for point-date rows
models/                       CNN encoder + temporal Transformer
training/query_engine.py      training loop and metrics
artifacts/                    compact data, splits, stats, generated model outputs
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

## How To Interpret Training

Do not only look at loss.

Use these checks:

1. Compare normal training with `--no-query-date`.
2. Compare model stage metrics with `scripts/audit.py` date-only baseline.
3. Watch crop macro-F1, not only crop accuracy.
4. Remember that each point is expanded into 7 query rows, so crop examples are repeated 7 times.
5. Do not confuse padded timesteps with invalid observed pixels. The observed invalid-pixel ratio is about 7.68%; the lower 0.566 array ratio only happens if padded timesteps are counted as invalid.

## Next Model Direction

Keep one main model for now:

```text
CNN encoder per date -> Transformer over time -> crop head + stage head
```

Only add LSTM/RNN or other models after the audit proves whether the current issue is data/date leakage or model behavior.


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
