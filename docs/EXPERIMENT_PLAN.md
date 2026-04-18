# Experiment Plan

This branch keeps the successful submission path intact and adds a controlled training program for model search.

## Current Baseline

- Model: date-aware CNN + Transformer.
- Input: clean Sentinel-2 patch time series `[N, T, 12, 15, 15]`.
- Extra input: valid-pixel mask channels, so the model receives 24 channels when configured with `include_valid_mask_as_channels=true`.
- Query input: phenophase query day-of-year.
- Checkpoint selection: `val_competition_score`, with `val_loss` as tie breaker.
- First submission result was strong, so date/query inputs remain enabled for main experiments.

## Research Direction

- Keep full patch tensors as the main data path. Recent SITS work supports modeling the whole temporal sequence instead of collapsing early.
- Keep date encodings. Crop and phenophase signatures are seasonal, and multiple papers use temporal/day embeddings or explicit phenological trajectories.
- Add auxiliary features, but only as a side branch. Phenology indices help the model see NDVI/EVI/moisture/fire-stress style signals directly without replacing the CNN patch encoder.
- Add ablations to catch leakage or overfitting: no query date, no acquisition date, and shuffled labels.
- Prefer reproducible experiment configs over one-off notebooks.

## Papers That Informed This Branch

- SITS-Former: patch-based Sentinel-2 time-series Transformer pretraining with missing-data reconstruction.
- ViTs for SITS / TSViT: temporo-spatial Transformer factorization and acquisition-time positional encodings for satellite image time series.
- IncepTAE: hybrid local/global temporal attention for fine-grained SITS crop classification.
- 2025 crop-quality-control work: Sentinel-2 crop classification improves with time-series phenology features and noisy-label quality control.
- 2025/2026 agriculture foundation-model work: full-season and multi-scale spatiotemporal modeling are becoming the strongest direction, but this repository keeps the architecture lightweight enough for the competition.

## Added Feature Set

Implemented in `data/aux_features.py`.

Per sample/query, the auxiliary vector includes:

- Per-band median, standard deviation, min, max, amplitude, and valid ratio.
- Vegetation-index summaries for `NDVI`, `EVI`, `NDMI`, `NBR`, `NDRE`, `SAVI`, and `GNDVI`.
- Query day-of-year scaled to `[0, 1]`.
- Nearest acquisition distance from the query date.
- Nearest-date band medians and vegetation-index values.

Invalid pixels are ignored using `valid_pixel_mask`. Missing values are converted to `0.0` only inside the auxiliary feature vector.

## How To Run

Run one auxiliary-feature model:

```bash
python scripts/train.py --config configs/train_submission_date_aux_features.json
```

Run the full experiment suite:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json
```

Dry-run without training:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json --dry-run
```

Summarize completed experiments:

```bash
python scripts/summarize_experiments.py --root artifacts/models/experiments/date_query_feature_search_v1
```

Optionally export features for external analysis:

```bash
python scripts/extract_aux_features.py --query-stage-features
```

## Experiments In `configs/experiment_suite.json`

- `baseline_date_mask`: current strong model family.
- `aux_features_date_mask`: baseline plus auxiliary feature branch.
- `aux_features_no_mask_channels`: tests whether explicit features can replace mask channels.
- `strong_regularization_aux`: stronger dropout, label smoothing, and weight decay.
- `no_query_date_ablation`: checks how much query date matters.
- `no_time_date_ablation`: checks how much acquisition dates matter.
- `shuffle_labels_sanity`: should fail; if it performs well, there is leakage.

## Decision Rule

Pick the checkpoint with best `val_competition_score`, not the lowest loss alone. If scores tie, prefer lower validation loss and simpler model.

Before a second real submission:

- Run at least one full suite on Colab.
- Check shuffled-label sanity fails.
- Compare date ablations.
- Package only the best checkpoint into `checkpoints/model.pt`.
- Run `python scripts/validate_submission.py`.
