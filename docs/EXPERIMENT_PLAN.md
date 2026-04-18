# Experiment Plan

This branch keeps the successful submission path intact and uses controlled experiments for model search.

## Baseline We Trust

C00 is the safe checkpoint because it passed the platform. Do not overwrite `checkpoints/model.pt` until a new model is clearly better and validation passes locally.

The model family that works is:

- Sentinel-2 patch time series as the main input.
- CNN patch encoder for each date.
- Transformer over the date sequence.
- Query day-of-year enabled.
- Acquisition day-of-year enabled.
- Valid-pixel mask channels enabled.

The first submission result was strong, so the default path should remain date-aware.

## Why We Are Testing Features

The literature direction is consistent with what our results show: crop and phenophase prediction depend heavily on seasonal trajectories. Raw time-series Transformers can learn those trajectories, but explicit vegetation and moisture indices can sometimes help because they expose phenology signals directly.

The feature branch is therefore a side branch, not a replacement for the CNN/Transformer path. The raw patch sequence remains the main signal.

## Current Feature Set

Implemented in `data/aux_features.py`.

For each query, we compute:

- Per-band median, standard deviation, min, max, amplitude, and valid ratio.
- Vegetation-index summaries for `NDVI`, `EVI`, `NDMI`, `NBR`, `NDRE`, `SAVI`, and `GNDVI`.
- Query day-of-year scaled to `[0, 1]`.
- Distance from the query date to the nearest acquisition date.
- Nearest-date band medians and nearest-date vegetation-index values.

Invalid pixels are ignored when computing features. Missing values are converted to `0.0` only inside the auxiliary vector.

## What We Learned So Far

| ID | Result | Lesson |
|---|---|---|
| C00 | Platform passed, validation loss `0.063148` | Safe checkpoint. Do not overwrite. |
| C01 | Aux + smoothing score `0.935991` | Not good enough. Also not a clean feature test because several regularization settings changed. |
| C02 | Baseline repeat score `1.000000`, loss `0.267392` | Baseline is reproducible, but lower-confidence than C00. |

## Next Controlled Suite

The active suite is `configs/experiment_suite.json`.

Run only C03 first:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json --only C03_reproduce_C00_lr0004_bs512_valloss
```

Reason: this tells us whether the C00 low loss was mostly caused by `batch_size=512` and `learning_rate=0.0004`.

Then run C04:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json --only C04_aux_features_only
```

Reason: this tests engineered features without label smoothing, extra weight decay, or higher dropout.

After C03 and C04, use the results to decide:

| If Result | Decision |
|---|---|
| C03 gets loss near C00 | Use C00-style hyperparameters for the next serious candidates. |
| C03 stays near C02 loss | Loss difference is not only hyperparameters; keep comparing by score and platform validation. |
| C04 beats C02 | Keep auxiliary features and tune branch size. |
| C04 loses to C02 | Features are not yet useful; try smaller branch C05 or feature selection. |

## Full Suite Meaning

| ID | Experiment | Why It Exists |
|---|---|---|
| C03 | Reproduce C00 hyperparameters | Separates code changes from hyperparameter effects. |
| C04 | Aux features only | Clean test of explicit phenology/spectral features. |
| C05 | Aux features small branch | Checks if the feature branch is too large/noisy. |
| C06 | Aux features without mask channels | Tests whether features can replace valid-mask channels. |
| C07 | No query date ablation | Measures query-date dependence. |
| C08 | No acquisition date ablation | Measures acquisition-time dependence. |
| C09 | Shuffle-label sanity | Must fail; protects us from leakage mistakes. |

## Decision Rule

Pick candidates in this order:

1. Higher `val_competition_score`.
2. Higher `val_rice_stage_macro_f1`.
3. Higher `val_crop_macro_f1`.
4. Lower `val_loss`.
5. Smaller train-validation score gap.
6. Simpler model if metrics are close.

Before a second real submission:

- Check the new checkpoint's `metrics_summary.json`.
- Check the git commit in `config_resolved.json`.
- Copy only the chosen checkpoint to `checkpoints/model.pt`.
- Run `python scripts/validate_submission.py`.
- Keep a backup folder for the old C00 checkpoint.
