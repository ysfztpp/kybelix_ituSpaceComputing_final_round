# Experiment Results

This file is the human-readable experiment tracker. For future runs, the source of truth is each model output folder.

Keep these files for every serious run:

- `model.pt`: saved checkpoint.
- `history.json`: every epoch.
- `config_resolved.json`: exact resolved config, now including git commit information.
- `metrics_summary.json`: compact best-epoch summary for new runs.
- `metrics_summary.md`: one-row markdown summary for new runs.

Generate a fresh comparison table with:

```bash
python scripts/summarize_experiments.py --root artifacts/models --format markdown
```

Inspect one checkpoint with:

```bash
python scripts/inspect_checkpoint.py artifacts/models/submission_cnn_transformer_date/model.pt
```

## Current Results

| Chrono ID | Former ID | Experiment | Config / Model | Code Commit | Best Epoch | Best Val Score | Crop F1 | Rice Stage F1 | Val Loss | Status | Meaning |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| C00 | E00 | First platform submission | `checkpoints/model.pt` | platform: `28d94fb6`; repo checkpoint: `424868d`; cleaned package: `254d04e` | 75 | 1.000000 post-computed | 1.000000 | 1.000000 | 0.063148 | Submitted | Safe fallback. This checkpoint passed the real platform. |
| C01 | E02 | Aux features + label smoothing | `configs/train_submission_date_aux_features.json` | likely `7696d84` | 21 | 0.935991 | 1.000000 | 0.893318 | 0.366777 | Done | Worse than baseline. The result is not a clean feature test because smoothing, dropout, and weight decay also changed. |
| C02 | E01 | Latest baseline repeat | `configs/train_submission_date.json` | likely `334fe92` or `7696d84` | 67 | 1.000000 | 1.000000 | 1.000000 | 0.267392 | Done | Confirms the date-aware baseline is strong, but its loss is higher than C00. |

## Why C00 And C02 Have Different Loss

Both C00 and C02 reached perfect validation score. The loss is different because cross-entropy also measures confidence, not just correctness.

The important differences are:

- C00 used `batch_size=512` and `learning_rate=0.0004`.
- C02 used `batch_size=128` and `learning_rate=0.00025`.
- C00 was selected mainly by validation loss.
- C02 was selected by `val_competition_score`, using `val_loss` only as a tie breaker.
- The baseline model path did not change in a way that explains this by itself.

So the next clean check is C03: reproduce the C00-style hyperparameters with the current code.

## Current Decision

Do not replace `checkpoints/model.pt` yet.

The safe checkpoint is still C00 because it already passed the platform and has the lowest validation loss. C02 is useful because it confirms the model family is reproducible. C01 should not be used as a submission candidate.

## Next Feature Search

The next suite is `configs/experiment_suite.json`. It uses chronological IDs starting from C03.

| Order | ID | Experiment | Reason | What We Learn |
|---:|---|---|---|---|
| 1 | C03 | Reproduce C00 hyperparameters | Before testing more features, we need to know whether the low C00 loss comes from batch size and learning rate. | If loss returns near 0.06, the difference was mostly hyperparameters/checkpointing. |
| 2 | C04 | Aux features only | C01 changed too many things at once. This tests features without label smoothing or extra regularization. | If this improves over C02, engineered features help. |
| 3 | C05 | Aux features with smaller branch | A large feature branch can inject noisy shortcuts. | If C05 beats C04, features help but need weaker capacity. |
| 4 | C06 | Aux features without mask channels | Tests whether explicit validity/statistics features can replace raw valid-mask channels. | If it drops, keep mask channels. |
| 5 | C07 | No query date ablation | Query date is probably important for stage. | If score drops, query date is real signal, not optional. |
| 6 | C08 | No acquisition date ablation | Acquisition timing tells the transformer where observations are in the season. | If score drops, keep time-date encoding. |
| 7 | C09 | Shuffle-label sanity | This must fail. | If it does not fail, there is leakage or split contamination. |

Run one experiment first:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json --only C03_reproduce_C00_lr0004_bs512_valloss
```

Then run the first clean feature test:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json --only C04_aux_features_only
```

Run the full suite only after these two finish cleanly:

```bash
python scripts/run_experiments.py --suite configs/experiment_suite.json
```

## Decision Rule

Use this order when choosing a new candidate:

1. Higher `val_competition_score`.
2. Higher `val_rice_stage_macro_f1`.
3. Higher `val_crop_macro_f1`.
4. Lower `val_loss`.
5. Smaller train-validation score gap.
6. Simpler model if all metrics are close.

Only replace `checkpoints/model.pt` after the candidate beats C00/C02 and passes:

```bash
python scripts/validate_submission.py
```
