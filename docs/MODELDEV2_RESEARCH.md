# ModelDev2 Research Notes

Last updated: 2026-04-27

## Scope

Goal: identify model families from nearby remote-sensing and time-series literature that are most likely to outperform the current CNN+Transformer submission family on our point-date crop/stage task.

This note intentionally looks at adjacent problems, not only this exact competition setup:

- satellite image time series classification,
- crop mapping with seasonal shift,
- phenology-aware temporal modeling,
- remote-sensing self-supervised pretraining,
- global/local temporal encoders for sparse or irregular season signals.

## Strong Patterns From The Literature

### 1. Temporal-then-spatial factorization is a strong default for SITS

Paper:

- TSViT, CVPR 2023
  https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html

Why it matters:

- The paper reports state-of-the-art performance on multiple satellite image time-series benchmarks.
- The key idea is not “just use a ViT”; it is specifically to process time first, then aggregate spatially.
- Multiple learnable class tokens are used to improve discriminative capacity.

What I took from it:

- Replace per-timestep CNN pooling with patch tokens.
- Encode temporal evolution before collapsing local spatial structure.
- Use task-specific/class-token decoding rather than one generic pooled vector.

### 2. Calendar-time shortcuts are fragile across regions; season-shift robustness matters

Paper:

- Thermal Positional Encoding (TPE), CVPRW 2022
  https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Nyborg_Generalized_Classification_of_Satellite_Image_Time_Series_With_Thermal_Positional_CVPRW_2022_paper.html

Why it matters:

- The paper argues that cross-region errors often come from seasonal shifts rather than poor spatial recognition.
- Raw calendar DOY is not a biologically stable coordinate across regions.
- Their reported gain comes from replacing brittle time encoding with growth-aligned temporal encoding.

What I took from it:

- The next model should not rely on plain absolute query day alone.
- Signed query-to-acquisition lag is a better inductive bias than only absolute DOY.
- Mild date-robustness augmentation should stay in the experimental plan.

### 3. Global-only temporal attention is often not enough; local temporal dynamics help

Paper:

- GL-TAE, Remote Sensing 2023
  https://www.mdpi.com/2072-4292/15/3/618

Why it matters:

- The paper reports gains from combining global attention with local temporal extraction.
- Remote-sensing crop signals often contain both long seasonal context and short transition events.

What I took from it:

- Keep the model fully seasonal, but avoid making every decision from one pooled global token.
- The new model should preserve local temporal structure longer and only aggregate after temporal reasoning.

### 4. Self-supervised pretraining is increasingly the right scaling path

Papers:

- Presto, ICLR 2024 submission
  https://openreview.net/forum?id=Iip7rt9UL3
- Galileo, TerraBytes 2025
  https://openreview.net/forum?id=a2Qn7lTYWL
- Masked Vision Transformers for Hyperspectral Image Classification, CVPRW 2023
  https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Scheibenreif_Masked_Vision_Transformers_for_Hyperspectral_Image_Classification_CVPRW_2023_paper.html

Why it matters:

- Remote-sensing labels are scarce compared with available unlabeled EO time series.
- The strongest newer results increasingly rely on pretraining, masked modeling, or large multimodal encoders.
- Even when the downstream task is different, the representation-learning lesson transfers.

What I took from it:

- The heavy supervised model is step one.
- The likely highest-ceiling follow-up is self-supervised pretraining on unlabeled Sentinel time series, then fine-tuning on the crop/stage task.

### 5. Phenology-aware conditioning is promising when stage structure matters

Related recent paper:

- Phenology-Aware Transformer for Semantic Segmentation of Non-Food Crops from Multi-Source Remote Sensing Time Series, Remote Sensing 2025
  https://www.mdpi.com/2072-4292/17/14/2346

Why it matters:

- Even though the task is segmentation, the point is useful: explicit phenology cues can guide attention toward the temporally most discriminative parts of the season.

What I took from it:

- Our stage head should remain more structured than the crop head.
- Sequence-aware losses and postprocessing remain justified.
- A later variant should fuse auxiliary phenology/index summaries into the stage pathway only.

## What Looks Weak In The Current Family

The current CNN+Transformer family is strong but probably capped by three issues:

1. It compresses each timestep too early into one vector, losing fine local spatial structure before temporal modeling.
2. It still leans heavily on handcrafted pooling choices instead of task-specific token routing.
3. The existing “cross-series query” idea is directionally good, but the legacy time encoding stack was built for positive DOY values, not signed lags.

## New Model Added On `modelDev2`

Implemented model:

- `models/query_tsvit.py`
- model type: `query_tsvit`
- first config: `configs/train_c28_tsvit_signed_relative_query_val.json`

Architecture summary:

- Patch-token embedding over each 15x15 timestep patch.
- Factorized temporal encoder first, spatial encoder second.
- Signed relative query-time encoding for `query_doy - time_doy`.
- Task-specific temporal pooling queries for crop and stage.
- Class-token spatial decoding for both heads.
- Optional aux-feature fusion kept compatible with the existing training API.

Why this is the right first “heavy” model:

- It directly incorporates the strongest structural lesson from TSViT.
- It explicitly addresses seasonal shift using signed relative lags.
- It stays compatible with the current dataset, loss, and training scripts.

## Immediate Experiment Plan

### C28: heavy supervised TSViT baseline

Run:

```bash
python scripts/train.py --config configs/train_c28_tsvit_signed_relative_query_val.json
```

This is the first serious challenger to the current CNN+Transformer line.

### If C28 is promising, next variants should be:

1. `C29`: same model plus stage-only auxiliary phenology/index branch.
2. `C30`: same model but with less aggressive date dropout if crop F1 slips.
3. `C31`: same backbone plus self-supervised pretraining.
4. `C32`: multimodal extension if temperature/weather or SAR can be added.

## C28b Correction

The first heavy TSViT run exposed a stage-collapse pattern:

- crop learned extremely well,
- stage validation accuracy stayed near `1/7`,
- transition-Viterbi decoding and strong date corruption were the main suspects.

Recommended retry:

- `configs/train_c28b_tsvit_l4_stage_recovery_val.json`

Changes from C28:

- `stage_postprocess: none`
- `stage_sequence_loss_weight: 0.0`
- `stage_loss_weight: 1.0`
- `query_doy_dropout_prob: 0.0`
- `time_doy_dropout_prob: 0.0`
- `random_time_shift_days: 20`
- `batch_size: 160`
- `gradient_accumulation_steps: 1`
- `num_workers: 8`

Why:

- First recover raw stage learning before adding structured decoding back.
- Use larger per-step batches to better utilize an L4 GPU.

## Recommendation

For now, the most defensible path is:

1. keep the inference bug fix separate from model search,
2. train the new TSViT-style supervised model first,
3. only after that decide whether the real next leap is architectural or pretraining-based.

My current belief:

- The best short-term upside is the new `query_tsvit` model.
- The best medium-term upside is `query_tsvit` plus self-supervised pretraining.
