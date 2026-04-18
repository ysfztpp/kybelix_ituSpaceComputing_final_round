# Experiment Plan

Last updated: 2026-04-18

The previous C03-C09 suite has completed in the upper-folder archive `../models`. Current detailed results are in `docs/EXPERIMENT_RESULTS.md`, and the full model explanation is in `docs/DETAILED_MODEL_REPORT.md`.

## Current Baseline Decision

The safest model family remains:

- Sentinel-2 patch time series as the main input.
- 12 spectral bands plus 12 valid-mask channels.
- CNN patch encoder for each acquisition date.
- Transformer over the date sequence.
- Query day-of-year enabled.
- Acquisition day-of-year enabled.
- No auxiliary feature branch for the safest baseline.

The safe fallback is still C00/E1 because it passed the platform. The best reproduced current-code baseline is C03.

## What The Completed Suite Proved

|Experiment|Result|Decision|
|---|---|---|
|C03 C00 hyperparameter reproduction|Val score `1.000000`, val loss `0.061087`|C00-style batch/LR and loss checkpointing reproduce the strong low-loss baseline.|
|C04 original-speed aux features|Val score `0.995363`, val loss `0.010718`|Interesting low-loss result, but not better than C03 on score.|
|C04 fast repeat|Val score `0.952619`|Aux features are unstable under 512/0.0004 in this run.|
|C05 small aux branch|Val score `0.976170`|Smaller aux branch helps versus C04 repeat but still loses to C03.|
|C06 no mask channels|Val score `0.974086`|Keep valid-mask channels.|
|C07 no query date|Val score `0.436115`|Query date is essential. Do not remove it.|
|C08 no acquisition date|Val score `0.997683`, val loss `0.197197`|Acquisition date is less critical than query date, but should stay for confidence.|
|C09 shuffled labels|Val score `0.178458`|Sanity check correctly fails; no obvious leakage signal.|

## Immediate Next Steps

1. Keep `checkpoints/model.pt` unchanged until a replacement passes validation.
2. If testing C03 as a replacement, copy it into `checkpoints/model.pt` only in a controlled branch or backup the current file first.
3. Run local submission validation after any checkpoint replacement.
4. If platform submissions are limited, prioritize C03 over auxiliary-feature checkpoints.
5. Keep the C00/E1 checkpoint available as rollback.

## Model Selection Rule

Use this order for candidate selection:

1. Platform result, if available.
2. Higher validation competition score.
3. Higher rice-stage macro F1.
4. Higher crop macro F1.
5. Lower validation loss.
6. Smaller train-validation score gap.
7. Simpler model if metrics are close.

For current artifacts, this selects C00/E1 as the submitted fallback and C03 as the best reproduced current-code candidate.

## Recommended Future Experiments

### 1. Validate C03 End To End

Goal: confirm that C03 can replace C00 without packaging or inference issues.

Actions:

- Inspect C03 checkpoint metadata in a PyTorch environment.
- Copy C03 checkpoint to `checkpoints/model.pt` only after backing up C00.
- Run `python scripts/validate_submission.py`.
- Compare generated output format and prediction distribution.

### 2. Revisit C04 Carefully

C04 original-speed run had very low validation loss but not perfect score. That makes it interesting but not immediately safe.

Questions:

- Was the very low loss caused by auxiliary features or by training dynamics?
- Why did the 512/0.0004 repeat perform much worse?
- Does C04 generalize better on platform despite lower validation score?

Possible controlled reruns:

- C04 with batch 128 and LR 0.00025, same seed.
- C04 with batch 512 and lower LR.
- C04 with frozen or smaller auxiliary branch.
- C04 with auxiliary feature normalization checks.

### 3. Keep Query-Date Ablation As Evidence

C07 is useful for the article/presentation because it demonstrates why the model is query-date-aware.

Main message:

- Crop prediction survives without query date.
- Phenophase prediction fails without query date.

### 4. Use C09 As A Required Data-Safety Check

Run the shuffle-label sanity check whenever changing:

- preprocessing,
- train/validation split logic,
- query-row construction,
- label handling,
- date handling.

If C09 ever performs well, stop and debug leakage.

### 5. Runtime Optimization Separate From Science Experiments

On L4, VRAM usage around 7-8 GB is expected. Speed may be limited by CPU/DataLoader work rather than GPU memory.

If training is too slow, test these separately from model-quality experiments:

- `num_workers = 4`
- `num_workers = 8`
- persistent workers, if added to the code later
- precomputing auxiliary features, if aux experiments remain important

Do not combine runtime changes with scientific ablations unless the table clearly says so.

## Article/Presentation Claims That Are Currently Supported

Supported:

- The model uses a CNN to encode local image patches and a Transformer to model the time sequence.
- Query date is essential for phenophase prediction.
- Valid-pixel masks improve robustness and should stay.
- C00-style larger-batch training is reproducible in C03.
- The shuffle-label check fails, which supports the absence of direct label leakage in the training loop.

Avoid overstating:

- Do not claim auxiliary features improve the final model yet.
- Do not claim acquisition-date encoding is unnecessary; C08 kept high F1 but had worse loss.
- Do not claim C04 is strictly better than C03; it has lower loss but lower score.
