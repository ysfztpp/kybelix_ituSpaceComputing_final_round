# Experiment Results

This file is the human-readable tracker. The source of truth for future runs is each model folder:

- `history.json`: every epoch.
- `model.pt`: saved checkpoint.
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

## Current Table

| ID | Experiment | Config | Feature Branch | Query Date | Acquisition Date | Best Epoch | Best Val Score | Crop F1 | Rice Stage F1 | Val Loss | Status | Interpretation |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| E00 | First platform submission | `checkpoints/model.pt` | No | Yes | Yes | 75 | legacy checkpoint | not stored as score | not stored as score | 0.063148 | Submitted | Safe checkpoint that passed the platform. Keep until a better submission candidate is proven. |
| E01 | Baseline repeat on `dev` | `configs/train_submission_date.json` | No | Yes | Yes | 67 | 1.000000 | 1.000000 | 1.000000 | 0.267392 | Done | Strongest current validation result. This is the model family to beat. |
| E02 | Aux features with smoothing | `configs/train_submission_date_aux_features.json` | Yes | Yes | Yes | 21 | 0.935991 | 1.000000 | 0.893318 | 0.366777 | Done | Useful but worse than the baseline repeat. Label smoothing and/or feature noise likely hurt. |

## What The Latest Baseline Run Means

The latest baseline repeat used:

- CNN patch encoder.
- Transformer temporal encoder.
- Query day-of-year enabled.
- Acquisition day-of-year enabled.
- Valid-pixel mask channels enabled.
- No auxiliary feature branch.

It reached perfect validation macro-F1 at multiple epochs. The selected checkpoint should be epoch 67 because the checkpoint metric is `val_competition_score` and `val_loss` is the tie breaker.

Important detail:

- The last epoch is not necessarily the saved checkpoint.
- The selected checkpoint is the best validation score, then lower validation loss if scores tie.

## Next Experiments

| Order | ID | Experiment | Reason |
|---:|---|---|---|
| 1 | E03 | Aux features without label smoothing | Tests if smoothing caused the weaker aux result. |
| 2 | E04 | Aux features with lower dropout | Tests whether the aux model was over-regularized. |
| 3 | E05 | Shuffle-label sanity check | Must fail; if it does not fail, there is leakage. |
| 4 | E06 | No query date ablation | Measures dependence on the query date. |
| 5 | E07 | No acquisition date ablation | Measures dependence on image acquisition dates. |
| 6 | E08 | CNN + BiGRU attention | Next model family if baseline remains strongest. |

## Decision Rule

Use this order:

1. Highest `val_competition_score`.
2. Higher `val_rice_stage_macro_f1`.
3. Higher `val_crop_macro_f1`.
4. Lower `val_loss`.
5. Smaller train-validation score gap.

Do not replace `checkpoints/model.pt` until a new candidate is clearly better and validated with `scripts/validate_submission.py`.
