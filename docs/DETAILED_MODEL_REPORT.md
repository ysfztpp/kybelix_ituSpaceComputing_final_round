# Detailed Model, Data, Training, And Experiment Report

Last updated: 2026-04-18

This report is based on the current project code plus the completed model artifacts in `../models`. It should be treated as the main plain-English reference for the article and presentation.

## 1. High-Level Goal

The project predicts crop type and rice phenophase stage from Sentinel-2 satellite imagery.

For each field point, the system builds a small multi-date image time series around that point. At prediction time, the model receives:

- the full satellite patch time series for the point,
- the acquisition day-of-year for each satellite image,
- a query day-of-year for the requested phenophase date,
- optional engineered spectral/phenology features in some experiments.

The model outputs:

- crop type: `corn`, `rice`, or `soybean`,
- phenophase stage: one of 7 stage classes.

The seven phenophase stages are:

- `Greenup`
- `MidGreenup`
- `Peak`
- `Maturity`
- `MidSenescence`
- `Senescence`
- `Dormancy`

The model is a query-date model: the same point can be asked about different dates, and the stage output should change depending on the query date.

## 2. Current Source Files That Define The System

Important implementation files:

- `preprocessing/dataset.py`: builds the patch time-series NPZ from point CSVs and GeoTIFFs.
- `preprocessing/raster_io.py`: converts lon/lat to pixels, extracts 15 x 15 patches, pads edge cases, and cleans invalid values.
- `preprocessing/mapping.py`: maps points to raster regions.
- `preprocessing/inventory.py`: audits TIFF filenames and selects one file per region/date/band.
- `preprocessing/normalization.py`: computes train-only per-band normalization stats.
- `data/query_dataset_npz.py`: expands each point into point-date query rows for training.
- `data/aux_features.py`: computes optional engineered vegetation-index and validity features.
- `data/transforms.py`: applies per-band normalization at train/inference time.
- `models/query_cnn_transformer.py`: full CNN + Transformer + query-date classifier.
- `models/cnn_encoder.py`: per-date patch CNN encoder.
- `models/temporal_transformer.py`: day-of-year encoding and masked temporal pooling.
- `training/query_engine.py`: loss, metrics, epoch loop, checkpointing, and early stopping.
- `scripts/train.py`: training entrypoint for one config.
- `scripts/run_experiments.py`: controlled experiment-suite runner.
- `scripts/submission_inference.py`: submission inference path.

## 3. Raw Data And File Selection

The raw imagery consists of Sentinel-2 GeoTIFF files. The current 12-band order is:

`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`

The preprocessing code scans input TIFF paths and audits filenames. It rejects files that:

- do not match the expected filename pattern,
- refer to unsupported bands,
- are duplicate canonical files in the same folder.

For every `(region_id, acquisition_date, band)` group, it selects exactly one TIFF. If multiple processing levels are available, it prefers L2A before L1C, then uses path order as a deterministic tie breaker.

## 4. Point And Region Mapping

The point CSV is normalized to accept common column variants such as `Longitude`, `Longtitude`, `longitude`, `lon`, `Latitude`, `lat`, `date`, `query_date`, and `phenophase_date`.

The code groups rows by `point_id`. A point can have multiple phenophase-date labels, but it has one longitude/latitude location.

For each unique point:

1. The code checks all raster region bounding boxes.
2. If one region contains the point, that region is used.
3. If multiple regions contain the point, the selected region is chosen by:
   - highest number of time steps,
   - then highest band count,
   - then largest border margin,
   - then region ID for deterministic tie breaking.
4. In training mode, points outside all raster regions are dropped.
5. In test/inference mode, nearest-region fallback is allowed.

Current local cleaned artifact stats:

- Input point rows: 5,446.
- Unique points: 778.
- Samples kept: 778.
- Samples dropped: 0.
- Unique-region matches: 751.
- Multi-region overlaps: 27.

## 5. Patch Extraction

For every kept point, every selected date, and every selected band, the system extracts a `15 x 15` patch.

The pixel center is computed as:

- `pixel_x = round((lon - xmin) / pixel_size_x)`
- `pixel_y = round((ymax - lat) / pixel_size_y)`

Because the patch size is odd, a `15 x 15` patch has one center pixel and 7 pixels of context on each side.

### Edge Cases Near TIFF Borders

If the full patch would extend outside the raster, the point is not discarded. The code reads the valid part of the raster window and pads the missing side by edge replication.

Edge replication means the nearest real edge row or column is repeated. This is not zero padding.

Example: if a point is close to the left border, the missing left columns are filled by repeating the first real column that was read from the TIFF.

The code records border behavior with:

- `border_margin_pixels`: distance from the center pixel to the nearest border.
- `center_clamped`: whether the rounded center pixel had to be clamped into the raster.
- `requires_edge_replication`: whether any patch cell required edge replication.

Current local cleaned artifact stats:

- Samples requiring edge replication somewhere: 136.
- Region/date/band cells requiring edge replication: 8,354.
- Samples with center clamping somewhere: 97.
- Region/date/band cells with center clamping: 1,406.

## 6. Invalid Pixel Handling

A pixel is valid only if:

`finite and 0.0 < value <= 2.0`

Invalid pixels are handled in two ways:

1. The cleaned patch value is filled with `0.0`.
2. A boolean `valid_pixel_mask` records which pixels were valid.

This is important because `0.0` can mean either a filled invalid pixel or, after normalization, a centered value. The mask lets the model distinguish validity when mask channels are enabled.

Current local cleaned artifact stats:

- Global valid pixel ratio: 0.923175.
- Every sample has at least one invalid pixel somewhere in its full time/band patch tensor.

## 7. Time, Band, And Tensor Shapes

The cleaned NPZ stores the full dataset as a rectangular tensor. The local artifact has:

`patches: [778, 29, 12, 15, 15]`

This means:

- 778 unique spatial points,
- maximum 29 acquisition dates,
- 12 Sentinel-2 bands,
- 15 x 15 pixels per band/date.

Different regions can have different date counts. Shorter sequences are padded to 29 dates. The `time_mask` marks real dates as `True` and padded dates as `False`.

The `band_mask` records whether an individual band exists for a real date. The model does not directly consume `band_mask`, but when valid-mask channels are enabled, missing bands are visible as invalid mask channels.

Current local cleaned artifact stats:

- Max time steps: 29.
- Real time cells: 13,872.
- Padded time cells: 8,690.
- Samples with time padding: 767 of 778.
- Missing band cells: 367.
- Samples with missing bands: 38.

## 8. Normalization

Band normalization is computed from training patches only and only from valid pixels.

The active training path uses z-score normalization:

`normalized = (raw_value - training_band_mean) / training_band_std`

After normalization, invalid pixels are reset to `0.0`.

This prevents invalid raw fill values from becoming misleading normalized values.

## 9. How Point-Date Training Queries Are Built

The NPZ stores one tensor per unique point. The PyTorch dataset expands each point into query rows.

For each point:

1. The code reads its seven phenophase day-of-year labels.
2. For every known day-of-year greater than zero, it creates one training query.
3. That query receives the full satellite time series for the point.
4. That query also receives the requested `query_doy`.
5. The crop target is the point crop type.
6. The stage target is the phenophase stage corresponding to that query date.

Current local query counts:

|Split|Point samples|Query rows|Rice-stage loss rows|
|---|---:|---:|---:|
|Train|609|4,263|2,310|
|Validation|169|1,183|259|

The train/validation split is grouped by `resolved_region_id`, so validation regions are separated from training regions.

Split stats:

- Train regions: 41.
- Validation regions: 8.
- Train samples: 609.
- Validation samples: 169.

## 10. Exact Model Input

For the main date-aware model with mask channels enabled, each query item contains:

- `patches`: `[T, 24, 15, 15]`
- `time_mask`: `[T]`
- `time_doy`: `[T]`
- `query_doy`: scalar
- optional `aux_features`: `[135]` only for auxiliary-feature experiments

The 24 channels are:

- 12 normalized Sentinel-2 spectral bands,
- 12 binary valid-pixel mask channels.

Without mask channels, `patches` has 12 channels instead of 24.

## 11. Model Architecture

The current model class is `QueryCNNTransformerClassifier`.

### 11.1 Per-Date CNN Encoder

Each acquisition date is encoded separately by the same CNN.

CNN structure:

1. Conv2D input channels to 32 channels.
2. BatchNorm.
3. GELU.
4. Conv2D 32 to 64 channels.
5. BatchNorm.
6. GELU.
7. MaxPool2D.
8. Conv2D 64 to 96 channels.
9. BatchNorm.
10. GELU.
11. Adaptive average pooling to one spatial cell.
12. Flatten.
13. Dropout.
14. Linear projection to 256 dimensions.
15. LayerNorm.

Output: one 256-dimensional embedding per date.

### 11.2 Acquisition Day Encoding

If `use_time_doy = true`, each acquisition day-of-year is encoded as sine/cosine seasonal values and projected into 256 dimensions.

This encoding is added to the CNN date embedding. It tells the Transformer when each satellite observation happened in the growing season.

### 11.3 Temporal Transformer

The sequence of date embeddings is processed by a Transformer encoder.

Current main architecture:

- Transformer dimension: 256.
- Transformer layers: 4.
- Attention heads: 8.
- Feed-forward dimension: 1,024.
- Activation: GELU.
- Dropout: 0.18 in the latest main suite.

The Transformer receives `src_key_padding_mask = ~time_mask`, so padded dates are ignored.

### 11.4 Masked Temporal Pooling

After the Transformer, the model averages only real dates according to `time_mask`. This produces one 256-dimensional seasonal representation for the point.

### 11.5 Query Day Encoding

If `use_query_doy = true`, the requested query day-of-year is also encoded as sine/cosine seasonal values and projected into 256 dimensions.

This is critical for phenophase prediction because stage is date-dependent.

### 11.6 Optional Auxiliary Feature Branch

Auxiliary-feature experiments compute a 135-dimensional feature vector from the raw patch time series.

The auxiliary features include:

- per-band median, standard deviation, min, max, amplitude, and valid ratio for all 12 bands,
- vegetation index summaries for NDVI, EVI, NDMI, NBR, NDRE, SAVI, and GNDVI,
- scaled query day-of-year,
- distance from query date to nearest acquisition date,
- nearest-date band medians,
- nearest-date vegetation-index values.

The auxiliary branch is intentionally a side branch, not a replacement for the raw image time-series learner.

Auxiliary MLP:

1. LayerNorm over the 135 features.
2. Linear layer to `aux_hidden_dim`.
3. GELU.
4. Dropout.
5. LayerNorm.

Experiments used `aux_hidden_dim = 128` and `aux_hidden_dim = 32`.

### 11.7 Output Heads

The model concatenates:

- pooled temporal vector: 256 dimensions,
- query-date vector: 256 dimensions if enabled,
- optional auxiliary vector: 0, 32, or 128 dimensions.

Then it uses two heads:

- crop head: LayerNorm -> Dropout -> Linear to 3 classes,
- stage head: LayerNorm -> Dropout -> Linear to 7 classes.

## 12. Training Objective

The training loss is:

`total_loss = crop_loss + 0.6 * stage_loss`

Details:

- `crop_loss`: cross entropy over 3 crop classes, used for all query rows.
- `stage_loss`: cross entropy over 7 phenophase stages.
- `rice_stage_loss_only = true`: stage loss is counted only for rice query rows.

The model still produces stage logits for all crops, but only rice rows contribute to stage loss in the active configs.

## 13. Metrics And Checkpointing

The training loop records:

- loss,
- crop loss,
- stage loss,
- crop accuracy,
- crop macro F1,
- stage accuracy over all crops,
- rice-stage accuracy,
- rice-stage macro F1.

The project score used for model selection is:

`competition_score = 0.4 * crop_macro_f1 + 0.6 * rice_stage_macro_f1`

Newer runs usually checkpoint on `val_competition_score`, using `val_loss` as a tie breaker. C03 intentionally checkpoints on `val_loss` to reproduce the older C00 behavior.

## 14. Current Training Configuration

The base submission-date config is `configs/train_submission_date.json`.

Important base settings:

- Optimizer: AdamW.
- Scheduler: cosine decay.
- Warmup: 5 epochs.
- Mixed precision AMP: enabled.
- Gradient clipping: 1.0.
- Early stopping patience: 12.
- Weight decay: 0.03.
- Stage loss weight: 0.6.
- Dropout: 0.18.
- Valid-mask channels: enabled.
- Query date encoding: enabled.
- Acquisition date encoding: enabled.

The latest upper-folder suite uses `batch_size = 512` and `learning_rate = 0.0004` for C03 and for the repeated C04-C09 runs, except the separate `../models/C04` run, which used batch size 128 and learning rate 0.00025.

## 15. GPU And Runtime Notes For L4

Using only 7-8 GB of a 22.5 GB L4 is expected.

Reasons:

- The patches are only 15 x 15.
- The sequence length is at most 29.
- Transformer dimension is 256.
- There are only 4 Transformer layers.
- AMP mixed precision is enabled.
- The dataset is small enough that GPU memory is not the limiting factor.

The run can still be slow if the bottleneck is CPU-side work:

- reading batches,
- NumPy normalization,
- optional auxiliary feature computation,
- Python DataLoader overhead.

Practical interpretation:

- Batch size 512 reduces optimizer steps per epoch, which can speed training.
- Higher learning rate changes the experiment, not only the speed.
- `num_workers = 2` may underuse an L4 if CPU loading is the bottleneck.
- Trying `num_workers = 4` or `8` is reasonable, but monitor CPU RAM and stability.
- Auxiliary-feature runs can be slower or more CPU-bound because features are computed in Python/NumPy per query.

## 16. Upper-Folder Model Artifact Audit

The upper folder `../models` contains completed runs and model checkpoints.

Important files found:

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

Local Python does not have PyTorch installed, so `model.pt` internals were not loaded locally. The report below uses sidecar `history.json`, `config_resolved.json`, and `metrics_summary.json` files. E1/C00 has only a checkpoint in `../models`, so its metrics are kept from the previous project record.

## 17. Experiment Results From `../models`

|ID|Experiment|Batch/LR|Aux|Mask Channels|Query DOY|Time DOY|Best Epoch|Best Val Score|Crop F1|Rice Stage F1|Val Loss|Meaning|
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
|C00/E1|Safe platform baseline|512 / 0.0004|No|Yes|Yes|Yes|75|1.000000|1.000000|1.000000|0.063148|Safe submitted checkpoint; platform-passed fallback.|
|E2|Aux + label smoothing|1024 / 0.00025|Yes|Yes|Yes|Yes|21|0.935991|1.000000|0.893318|0.366777|Worse; changed too many regularization variables.|
|E3|Baseline repeat|1024 / 0.00025|No|Yes|Yes|Yes|67|1.000000|1.000000|1.000000|0.267392|Strong score but much worse confidence/loss than C00.|
|C03|C00 hyperparameter reproduction|512 / 0.0004|No|Yes|Yes|Yes|75|1.000000|1.000000|1.000000|0.061087|Successfully reproduces C00-style low loss.|
|C04|Aux features only, original-speed run|128 / 0.00025|Yes|Yes|Yes|Yes|79|0.995363|1.000000|0.992272|0.010718|Very low loss, but not perfect stage F1.|
|C04 repeat|Aux features only, fast run|512 / 0.0004|Yes|Yes|Yes|Yes|22|0.952619|0.994655|0.924595|0.198616|Fast hyperparameters made this aux run worse.|
|C05|Aux features, small branch|512 / 0.0004|Yes|Yes|Yes|Yes|73|0.976170|1.000000|0.960283|0.064275|Smaller branch improves over C04 repeat but not over C03.|
|C06|Aux features, no mask channels|512 / 0.0004|Yes|No|Yes|Yes|36|0.974086|1.000000|0.956810|0.113064|Removing mask channels hurts versus C03 and C04 original.|
|C07|No query date|512 / 0.0004|No|Yes|No|Yes|2|0.436115|0.989268|0.067347|1.111821|Query date is essential for stage prediction.|
|C08|No acquisition date|512 / 0.0004|No|Yes|Yes|No|40|0.997683|1.000000|0.996138|0.197197|Acquisition DOY helps confidence/loss but is less critical than query DOY on this split.|
|C09|Shuffle-label sanity|512 / 0.0004|No|Yes|Yes|Yes|7|0.178458|0.119741|0.217603|2.241805|Correctly fails; no obvious leakage signal.|

## 18. Main Findings

### C03 Confirms The C00 Hyperparameter Hypothesis

C03 used C00-style `batch_size = 512`, `learning_rate = 0.0004`, and validation-loss checkpointing. It reached:

- validation score: 1.000000,
- validation loss: 0.061087.

This is extremely close to the recorded C00 loss of 0.063148 and better than E3's repeat loss of 0.267392. The low-loss C00 behavior is therefore mostly reproducible with the current code and C00-style hyperparameters.

### Query Date Is Essential

C07 removed query day-of-year and collapsed to:

- validation score: 0.436115,
- rice-stage macro F1: 0.067347.

Crop F1 stayed high at 0.989268, but stage prediction failed. This is exactly what we expect: crop type can be learned from the seasonal image series, but phenophase stage needs the requested date.

### Acquisition Date Encoding Helps But Is Less Critical Than Query Date

C08 removed acquisition-date encoding but kept query-date encoding. It still reached:

- validation score: 0.997683,
- rice-stage macro F1: 0.996138,
- validation loss: 0.197197.

This means the model can still perform well on this split without explicit acquisition DOY, probably because image content and sequence order carry strong seasonal information. However, the loss is much worse than C03, so acquisition DOY still improves confidence and should remain enabled.

### Valid-Mask Channels Should Stay

C06 removed the 12 valid-mask channels and used auxiliary validity summaries instead. It reached:

- validation score: 0.974086,
- validation loss: 0.113064.

This is worse than C03 and worse than the best original-speed C04 run. The current evidence says valid-mask channels should stay.

### Auxiliary Features Are Not Yet A Clear Win

The auxiliary-feature picture is mixed:

- C04 original-speed run had very low validation loss: 0.010718, but score was 0.995363 rather than 1.000000.
- C04 fast repeat dropped to 0.952619.
- C05 small branch improved over the fast repeat but reached only 0.976170.
- C06 without mask channels reached only 0.974086.

This suggests auxiliary features can make predictions more confident in some settings, but they are not consistently improving the main validation score. They should not replace the C03/C00-style baseline yet.

### Shuffle-Label Sanity Check Passed

C09 intentionally shuffled training labels. It failed badly:

- validation score: 0.178458,
- crop macro F1: 0.119741,
- validation loss: 2.241805.

This is the desired result. It reduces concern that the high validation scores are caused by direct label leakage in the training loop.

## 19. Recommended Current Model Choice

For a safe final candidate right now:

1. Keep C00/E1 as the platform-passed fallback.
2. Treat C03 as the best reproduced baseline candidate from the new suite.
3. Do not switch to auxiliary-feature checkpoints unless platform validation or further tests show a real gain.
4. Keep valid-mask channels, query date encoding, and acquisition date encoding enabled.

Best current technical configuration:

- 12 Sentinel-2 bands plus 12 valid-mask channels.
- CNN patch encoder.
- 4-layer temporal Transformer.
- Query day-of-year enabled.
- Acquisition day-of-year enabled.
- No auxiliary feature branch for the safest baseline.
- Batch size 512.
- Learning rate 0.0004.
- Checkpoint selection by validation loss if trying to reproduce C00 confidence, or by validation score with loss tie breaker for competition ranking.

## 20. Future Experiment Plan

Recommended next steps:

1. Validate C03 locally through the submission pipeline before replacing `checkpoints/model.pt`.
2. If C03 passes local validation, compare it against the current C00 checkpoint on the real platform or final holdout if available.
3. Rerun C04 auxiliary features under a controlled setting if you want to study why the original-speed C04 had very low loss but imperfect score.
4. Do not remove query-date encoding.
5. Do not remove valid-mask channels.
6. Keep acquisition-date encoding unless deployment tests prove it unnecessary.
7. If speed remains a problem on L4, test `num_workers = 4` or `8` separately from model-quality experiments.
8. Keep C09 as a required sanity check whenever major data/split logic changes.

## 21. Short Presentation Script

Our pipeline first finds the Sentinel-2 GeoTIFF files for each region, date, and spectral band. For every field point, it extracts a 15 x 15 pixel patch across 12 bands for every available satellite date. If a point is near the image edge, the missing patch area is filled by repeating the nearest real edge pixels. Invalid reflectance values are filled with zero, but a validity mask is stored and passed to the model as extra channels. The final input is a satellite patch time series plus a requested query date.

The model encodes each date's patch with a CNN, adds date information, processes the sequence with a Transformer, and combines the seasonal representation with the query date. It then predicts both crop type and phenophase stage. Training uses crop classification loss for all crops and rice-stage loss for rice samples, because the stage score focuses on rice phenology. Our experiments show that query date is essential, valid-mask channels should stay, and the C00-style large-batch setup is reproducible. Auxiliary features are promising but not yet better than the safest CNN-Transformer baseline.
