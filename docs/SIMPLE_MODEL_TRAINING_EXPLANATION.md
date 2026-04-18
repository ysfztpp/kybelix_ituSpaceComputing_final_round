# Simple Explanation Of The Current Model, Training, And Experiments

Last updated: 2026-04-18

This is the simpler explanation for teammates, article writing, and presentation preparation. The deeper report is `docs/DETAILED_MODEL_REPORT.md`. The latest experiment table is `docs/EXPERIMENT_RESULTS.md`.

## Short Summary

The model looks at a small Sentinel-2 satellite image time series around each field point and predicts two things:

1. the crop type: `corn`, `rice`, or `soybean`,
2. the rice phenophase stage for a requested date.

The key idea is that crop growth changes over the season. The model uses the image sequence to understand the crop's seasonal behavior, and it uses the query date to answer which stage the crop is in on that date.

## What Goes Into The Model

For each point, preprocessing builds a tensor of satellite patches:

`[dates, channels, height, width]`

The current main model uses:

- up to 29 satellite acquisition dates,
- 12 Sentinel-2 spectral bands,
- 15 x 15 pixels around the point,
- 12 extra valid-mask channels.

So the main model input per point-date query is usually:

`[T, 24, 15, 15]`

The 24 channels are:

- 12 normalized Sentinel-2 bands,
- 12 binary masks showing which pixels are valid.

The model also receives:

- `time_doy`: the day-of-year for each satellite acquisition,
- `query_doy`: the requested day-of-year for the prediction,
- `time_mask`: which dates are real and which are padding.

Some experiments add a 135-value auxiliary feature vector with vegetation-index summaries such as NDVI, EVI, NDMI, NBR, NDRE, SAVI, and GNDVI.

## How TIFF Files Are Found And Used

The code scans the TIFF folders and parses the filenames. It keeps supported Sentinel-2 band files and removes duplicate canonical files.

For every region, date, and band, it selects one TIFF. If both L2A and L1C exist, it prefers L2A.

Then it maps each field point to a raster region. If multiple regions contain the point, it chooses the region with more dates, more bands, better border margin, and then deterministic region ID.

For training, points outside all regions are dropped. For test/inference, nearest-region fallback is allowed.

In the current local cleaned training artifact:

- 778 unique points were kept.
- 0 points were dropped.
- 751 points matched one region.
- 27 points had overlapping regions and were resolved by the tie-break rule.

## How The 15 x 15 Crop Works

For every point, every selected date, and every selected band, the code extracts a `15 x 15` pixel patch.

The point is placed at the center pixel when possible. Since 15 is odd, the crop has 7 pixels of context on each side.

If the point is near a TIFF edge and the full crop does not fit, the code pads by edge replication. That means it repeats the nearest real edge pixels. It does not use zero padding for image borders.

The code also records whether this happened with fields such as `border_margin_pixels` and `center_clamped`.

Current local artifact edge stats:

- 136 samples required edge replication somewhere.
- 97 samples had center clamping somewhere.

## Invalid Pixels And Masks

A pixel is valid only when:

`finite and 0.0 < value <= 2.0`

Invalid pixels are filled with `0.0`, but the validity mask records that they were invalid.

This is why the main model uses mask channels. It can see not only the spectral values but also which pixels were trustworthy.

The current local cleaned artifact has a valid pixel ratio of about `0.9232`.

## Time Padding

Not every region has the same number of satellite dates. The code pads all point sequences to the maximum number of dates, which is 29 in the current local artifact.

The model gets a `time_mask`, so it knows which dates are real and which are padding. The Transformer ignores padded dates.

Current local artifact shape:

`patches: [778, 29, 12, 15, 15]`

## How Training Rows Are Created

The NPZ stores one time series per unique point. Training expands that into query rows.

For every known phenophase date of a point:

1. the full patch time series is reused,
2. the query date is set to that phenophase date's day-of-year,
3. the crop label is the point's crop type,
4. the stage label is the phenophase stage for that date.

Current query counts:

|Split|Point samples|Query rows|Rice-stage loss rows|
|---|---:|---:|---:|
|Train|609|4,263|2,310|
|Validation|169|1,183|259|

The split is grouped by region, so validation regions are separate from training regions.

## The Model In Simple Steps

1. A CNN reads each date's `15 x 15` patch.
2. Each date becomes a 256-dimensional vector.
3. The model adds acquisition day-of-year information to each date vector.
4. A Transformer reads the sequence of date vectors.
5. The model averages only the real dates, ignoring padded dates.
6. The query day-of-year is encoded separately.
7. The seasonal vector and query-date vector are combined.
8. Two heads predict crop type and phenophase stage.

The current main Transformer has:

- 4 layers,
- 8 attention heads,
- 256 hidden dimension,
- dropout around 0.18 in the latest suite.

## Training Loss

Training uses two losses:

- crop cross-entropy for all crops,
- stage cross-entropy for rice rows only.

The combined loss is:

`total_loss = crop_loss + 0.6 * stage_loss`

This matches the project focus: crop type matters for all points, and rice phenophase matters especially for rice.

The validation score used in the project is:

`0.4 * crop_macro_f1 + 0.6 * rice_stage_macro_f1`

## Why L4 VRAM Usage Is Low

Using only 7-8 GB out of 22.5 GB on an L4 is expected.

The model is not huge:

- patches are only 15 x 15,
- max sequence length is 29 dates,
- hidden dimension is 256,
- Transformer has only 4 layers,
- AMP mixed precision is enabled.

If training is slow, the bottleneck may be CPU-side loading, normalization, auxiliary feature computation, or DataLoader overhead, not GPU memory.

Batch size 512 reduces the number of optimizer steps per epoch. But increasing learning rate changes the training behavior, so it should be reported as a real experimental change, not only a speed change.

If the GPU is waiting for data, try testing `num_workers = 4` or `8` separately.

## What The Completed Experiments Show

The upper-folder `../models` directory now contains completed C03-C09 runs.

|ID|Main Change|Best Val Score|Val Loss|Meaning|
|---|---|---:|---:|---|
|C03|Reproduce C00 with batch 512, LR 0.0004|1.000000|0.061087|Strongest reproduced baseline.|
|C04|Aux features, original-speed run|0.995363|0.010718|Very low loss, but not perfect score.|
|C04 repeat|Aux features, batch 512/LR 0.0004|0.952619|0.198616|Fast aux run got worse.|
|C05|Aux features with smaller branch|0.976170|0.064275|Better than C04 repeat, still below C03.|
|C06|Aux features without mask channels|0.974086|0.113064|Mask channels should stay.|
|C07|No query date|0.436115|1.111821|Query date is essential.|
|C08|No acquisition date|0.997683|0.197197|Acquisition date helps confidence but is less critical than query date.|
|C09|Shuffled labels|0.178458|2.241805|Correctly fails, which supports no obvious label leakage.|

## Current Decision

The safest choice is still the C00/E1 platform-passed checkpoint as fallback. The best reproduced current-code baseline is C03.

For the final baseline, keep:

- valid-mask channels,
- query day-of-year,
- acquisition day-of-year,
- CNN + Transformer architecture,
- C00/C03-style batch size and learning rate.

Do not switch to auxiliary features yet unless platform validation or a stronger holdout proves they improve generalization.

## Simple Reporter Version

We extract a 15 x 15 Sentinel-2 patch around each field point for every available satellite date and each of 12 spectral bands. If the point is near the image edge, missing crop pixels are filled by repeating the nearest real edge pixels. Invalid reflectance values are filled with zero, but a validity mask is passed to the model so it knows which pixels are reliable. A CNN encodes each date's patch, a Transformer learns the seasonal sequence, and the model combines this seasonal representation with the requested query date. It then predicts crop type and phenophase stage. Experiments show that the query date is essential, valid-mask channels should stay, and the C00-style larger-batch setup is reproducible in C03.
