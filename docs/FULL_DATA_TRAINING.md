# Full-Data Training Setup

Branch: `fullDataTraining`

This branch adds a minimal final-training path for the selected C03-style model. It uses all labeled training samples and does not create a validation loader.

## Why There Is No Best Epoch During This Run

Full-data training uses all labels for training. That means there is no validation set left for model selection.

So we cannot honestly say "best epoch" during this run. The epoch must be selected before training using earlier validation experiments.

The fixed stopping rule is:

`train for 75 epochs and save the final epoch`

Reason:

- C00 selected epoch 75.
- C03 selected epoch 75.
- C03 is the best reproduced current-code baseline.
- C03 used the same C00-style hyperparameters we want for final training.

This is standard final-model practice: tune on validation first, then retrain once on all data using the selected hyperparameters and epoch count.

## Config

Use:

```bash
configs/train_full_data_c03_epoch75.json
```

Important settings:

- batch size: `512`
- learning rate: `0.0004`
- epochs: `75`
- weight decay: `0.03`
- dropout: `0.18`
- valid-mask channels: enabled, `in_channels = 24`
- query day-of-year: enabled
- acquisition day-of-year: enabled
- auxiliary features: disabled
- min LR: `0.0`

## Train

In Colab or the training GPU environment:

```bash
python scripts/train_full_data.py --config configs/train_full_data_c03_epoch75.json
```

For the C11 stage-phenology-light model, use the fixed epoch selected from the
C11 validation run:

```bash
python scripts/train_full_data.py --config configs/train_full_data_c11_stage_phenology_light_epoch112.json
```

For the planned C19 strong-date-shift hedge model, use the fixed epoch selected
from the corresponding validation run:

```bash
python scripts/train_full_data.py --config configs/train_full_data_c19_strong_date_shift_dropout_epoch58.json
```

Expected output:

```text
artifacts/models/full_data_c03_epoch75/model.pt
artifacts/models/full_data_c03_epoch75/history.json
artifacts/models/full_data_c03_epoch75/config_resolved.json
artifacts/models/full_data_c03_epoch75/metrics_summary.json
```

## Submit This Model

After training, copy the checkpoint into the submission path:

```bash
cp artifacts/models/full_data_c03_epoch75/model.pt checkpoints/model.pt
python scripts/validate_submission.py
```

Then submit the normal repository/package.

For C11, prepare the GitLab submission repo from this project folder:

```bash
python scripts/prepare_c11_submission.py
```

For C19, prepare the GitLab submission repo from this project folder:

```bash
python scripts/prepare_c19_submission.py
```

This copies:

- `artifacts/models/full_data_c11_stage_phenology_light_epoch112/model.pt` to the submission repo as `checkpoints/model.pt`
- `configs/submission_c11_stage_phenology_light.json` to the submission repo as `configs/submission.json`
- the matching aux-feature, model, inference, and validation code

Then commit and push the GitLab submission repo only after validating the copied files.

## What To Watch While Training

Because there is no validation set, only use train metrics as a sanity check:

- train loss should decrease,
- crop F1 should approach 1.0,
- rice-stage F1 should approach the C03 range,
- no NaNs or sudden explosions.

Do not use train metrics to choose a different epoch after the fact. If you want to hedge, train separate fixed-epoch variants such as 70, 75, and 80, but selecting between them requires platform submissions or an external holdout.
