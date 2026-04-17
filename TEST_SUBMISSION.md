# Test Submission Branch

Branch purpose:

```text
test_submission
```

This branch restores only the files needed to package a trained Track 1 model and test the official output format. It is separate from `main` so the research/debug workflow stays simple.

## PDF Requirements Used

From the submission manual and task PDF:

```text
input points: /input/test_point.csv or /input/points_test.csv
input TIFFs: /input/region_test/*.tiff
output file: /output/result.json
Track 1 output name: result.json
working dir in image: /workspace
recommended entry: ./run.sh
```

The output JSON format is:

```json
{
  "Longitude_Latitude_Date": ["CropType", "PhenophaseStage"]
}
```

## Files Added For Submission Test

```text
Dockerfile
run.sh
.gitlab-ci.yml
configs/submission.json
configs/train_submission_date.json
configs/train_submission_no_query_date.json
scripts/submission_inference.py
checkpoints/.gitkeep
```

The actual trained model is not included yet. After Colab training, put the selected checkpoint here:

```text
checkpoints/model.pt
```

## Model For The Test

The submission training config uses a stronger but still optimized CNN + Transformer:

```text
CNN encoder per timestep
Transformer over Sentinel-2 dates
query-date embedding for point-date task
valid-pixel mask concatenated as extra channels
crop head: 3 classes
phenophase head: 7 classes
```

Input channel count is `24`:

```text
12 normalized Sentinel-2 bands + 12 valid-mask channels
```

Two configs are provided:

```text
configs/train_submission_date.json
configs/train_submission_no_query_date.json
```

Use the first one for the likely best competition score. Use the second one to test the date shortcut.

## Colab Training Commands

Clone and switch branch:

```bash
git clone https://github.com/ysfztpp/kybelix_ituSpaceComputing_final_round.git
cd kybelix_ituSpaceComputing_final_round
git checkout test_submission
```

Install only if needed:

```bash
pip install rasterio pandas numpy matplotlib nbformat
```

Train date model:

```bash
python3 scripts/train.py --config configs/train_submission_date.json
```

Train no-query-date model:

```bash
python3 scripts/train.py --config configs/train_submission_no_query_date.json
```

Copy the checkpoint you want to submit:

```bash
mkdir -p checkpoints
cp artifacts/models/submission_cnn_transformer_date/model.pt checkpoints/model.pt
```

or for no-query-date:

```bash
mkdir -p checkpoints
cp artifacts/models/submission_cnn_transformer_no_query_date/model.pt checkpoints/model.pt
```

Then commit and push the checkpoint on this branch.

## Local Format Smoke Test

The inference script was smoke-tested locally using real training-region TIFFs arranged as `/input/region_test` and produced `/output/result.json` with keys like:

```text
127.6879707_49.58568126_2018/6/5
```

This verifies the code path and JSON format, not model quality.

## GitLab Warning

The included `.gitlab-ci.yml` only sets the path variables from the manual:

```text
DATASETS=/input
OUTPUT_DIR=/output
START_CMD=./run.sh
```

If the official GitLab project already contains a platform-specific `.gitlab-ci.yml`, prefer their template and only copy these path values and `START_CMD`. Do not change `/input`, `/output`, or the Track 1 output filename `result.json`.
