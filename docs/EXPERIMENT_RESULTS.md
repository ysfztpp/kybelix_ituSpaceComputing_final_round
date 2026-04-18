# Experiment Results

Last updated: 2026-04-18

This file summarizes the completed model artifacts found in `../models`. The detailed explanation is in `docs/DETAILED_MODEL_REPORT.md`.

## Source Of Truth

For completed runs, keep the sidecar files with each model:

- `model.pt`: checkpoint.
- `history.json`: epoch-by-epoch metrics.
- `config_resolved.json`: exact training/model config.
- `metrics_summary.json`: best-epoch summary when available.
- `metrics_summary.md`: one-row markdown summary when available.

The upper-folder model archive currently includes:

- `../models/E1-baselinemodel/model.pt`
- `../models/E2-auxfeatures+labelsmoothing/`
- `../models/E3-baselinemodelRepeat/`
- `../models/C03/checkpoint/`
- `../models/C04/`
- `../models/C04_aux_features_onlyRepeat/`
- `../models/C05_aux_features_small_branch/`
- `../models/C06_aux_features_no_mask_channels/`
- `../models/C07_no_query_date_ablation/`
- `../models/C08_no_time_date_ablation/`
- `../models/C09_shuffle_labels_sanity/`

Local Python currently lacks PyTorch, so checkpoint internals were not loaded here. The table uses JSON sidecar files. C00/E1 metrics are preserved from the prior project record because the upper-folder E1 artifact only has `model.pt`.

## Current Result Table

|ID|Experiment|Batch/LR|Aux|Mask Channels|Query DOY|Time DOY|Best Epoch|Best Val Score|Crop F1|Rice Stage F1|Val Loss|Status|Meaning|
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|
|C00/E1|Safe platform baseline|512 / 0.0004|No|Yes|Yes|Yes|75|1.000000|1.000000|1.000000|0.063148|Submitted|Safe fallback; passed platform.|
|E2|Aux + label smoothing|1024 / 0.00025|Yes|Yes|Yes|Yes|21|0.935991|1.000000|0.893318|0.366777|Done|Worse; not a clean feature test because smoothing/regularization also changed.|
|E3|Baseline repeat|1024 / 0.00025|No|Yes|Yes|Yes|67|1.000000|1.000000|1.000000|0.267392|Done|Confirms model family is strong, but confidence/loss is worse than C00.|
|C03|C00 hyperparameter reproduction|512 / 0.0004|No|Yes|Yes|Yes|75|1.000000|1.000000|1.000000|0.061087|Done|Reproduces C00-style low loss with current code.|
|C04|Aux features only, original-speed run|128 / 0.00025|Yes|Yes|Yes|Yes|79|0.995363|1.000000|0.992272|0.010718|Done|Very low loss, but stage F1 is not perfect.|
|C04 repeat|Aux features only, fast run|512 / 0.0004|Yes|Yes|Yes|Yes|22|0.952619|0.994655|0.924595|0.198616|Done|Fast hyperparameters made this aux run worse.|
|C05|Aux features, small branch|512 / 0.0004|Yes|Yes|Yes|Yes|73|0.976170|1.000000|0.960283|0.064275|Done|Better than C04 repeat, still below C03.|
|C06|Aux features, no mask channels|512 / 0.0004|Yes|No|Yes|Yes|36|0.974086|1.000000|0.956810|0.113064|Done|Removing mask channels hurts.|
|C07|No query date ablation|512 / 0.0004|No|Yes|No|Yes|2|0.436115|0.989268|0.067347|1.111821|Done|Query date is essential for phenophase.|
|C08|No acquisition date ablation|512 / 0.0004|No|Yes|Yes|No|40|0.997683|1.000000|0.996138|0.197197|Done|Time DOY helps confidence but is less critical than query DOY on this split.|
|C09|Shuffle-label sanity|512 / 0.0004|No|Yes|Yes|Yes|7|0.178458|0.119741|0.217603|2.241805|Done|Correctly fails; no obvious leakage signal.|

## Main Conclusions

### C03 is the strongest reproduced baseline

C03 reached validation score `1.000000` and validation loss `0.061087`. This is effectively the same behavior as the safe C00 checkpoint, whose recorded validation loss was `0.063148`.

Interpretation: the earlier low-loss C00 result is reproducible with the current code when using C00-style batch size, learning rate, and validation-loss checkpointing.

### Query date must stay enabled

C07 removed query day-of-year and stage prediction collapsed. Crop F1 stayed high, but rice-stage macro F1 fell to `0.067347`.

Interpretation: the model can identify crops from the image time series, but cannot reliably answer phenophase stage without knowing the requested date.

### Acquisition date encoding should stay enabled

C08 removed acquisition-date encoding but kept query-date encoding. It still scored highly at `0.997683`, but validation loss worsened to `0.197197`.

Interpretation: acquisition date is not as critical as query date on this split, but it improves confidence and should remain enabled.

### Valid-mask channels should stay enabled

C06 removed mask channels and reached only `0.974086`, worse than C03 and worse than the best original-speed C04 run.

Interpretation: explicit per-pixel validity information is useful. Do not remove mask channels for the final baseline.

### Auxiliary features are not yet a safe replacement

Auxiliary features are mixed:

- C04 original-speed run: very low loss `0.010718`, but score `0.995363`.
- C04 fast repeat: score dropped to `0.952619`.
- C05 small branch: recovered to `0.976170`, still below C03.
- C06 without mask channels: `0.974086`.

Interpretation: engineered spectral features may help confidence in some settings, but the current evidence does not justify replacing the C03/C00-style baseline.

### Shuffle-label sanity check passed

C09 failed badly, as intended. This reduces concern that the training loop is leaking labels directly.

## Current Recommendation

Use this decision order:

1. Keep C00/E1 as the platform-passed fallback.
2. Treat C03 as the best reproduced current-code baseline.
3. Do not switch to auxiliary-feature checkpoints unless platform validation or a stronger holdout confirms a gain.
4. Keep query date, acquisition date, and valid-mask channels enabled.
5. Keep C09 as a required sanity check after any data/split changes.

## Useful Commands

Summarize local project model histories:

```bash
python scripts/summarize_experiments.py --root artifacts/models --format markdown
```

Inspect one checkpoint in an environment with PyTorch installed:

```bash
python scripts/inspect_checkpoint.py ../models/C03/checkpoint/model\ \(1\).pt
```

Before replacing the submission checkpoint, validate the package:

```bash
python scripts/validate_submission.py
```
