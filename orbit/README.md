# Kybelix C03 — On-Orbit Deployment Package

Flies the **C03 full-data** crop-type / phenophase model on the Three-Body Computing
Constellation Open Science Platform (`spaceapp.zero2x.org`).

C03 is the best model in this project: hidden-test score **0.994132**, versus
**0.907497** for C20 Fourier. Note that `project/checkpoints/model.pt` contains
**C20**, not C03 — this package deliberately exports from
`project_organized/checkpoints/c03_full_data_model.pt` (epoch 75, 3,275,533 parameters).

## What is in here

| Path | Purpose |
|---|---|
| `kybelix_orbit.py` | Flight entry point. numpy + onnxruntime only. |
| `model/c03.onnx` | C03 exported to ONNX (13.3 MB, opset 17). |
| `model/band_stats.json` | Train-derived per-band normalization statistics. |
| `sample/demo_input.npz` | Self-contained input: 30 points, 210 query rows, plus ground reference logits (2.7 MB). |
| `Dockerfile` | Application image on the platform base image. |
| `app.yaml` | Platform deployment spec. **Two values must be edited before upload.** |
| `build_delta.sh` | Builds the delta package inside the dev environment, with guards. |
| `tools/export_onnx.py` | Regenerates `model/c03.onnx` from a checkpoint. |
| `tools/build_sample.py` | Regenerates `sample/demo_input.npz`. |

Total image payload: **16 MB**, against a 300 MB delta-package cap.

## Three design decisions worth knowing

**No PyTorch.** The satellite node is ARM64 (NVIDIA Orin) and the platform base
image is not documented as carrying PyTorch. An aarch64 CUDA torch wheel would
blow the 300 MB delta cap on its own. The model ships as an ONNX graph instead;
`pandas` and `rasterio`/GDAL are gone for the same reason. Verified parity
against PyTorch on real data: **max |torch − onnx| = 7.6e-06**, identical
predictions.

**Static T=29.** `nn.MultiheadAttention` bakes the sequence length into its
reshapes, so the exported graph has a fixed 29-slot acquisition axis (batch stays
dynamic — verified at 1/7/64). This matches the trained contract: 29 padded slots
with `time_mask` marking the ~18 real ones. `fit_timesteps()` pads or truncates
to 29 explicitly rather than letting a shape mismatch surface in orbit.

**Self-contained input, `rs: false`.** v1.0.0 does not read the satellite's `/rs`
mount. We do not yet know what the assigned satellite group's `/rs` actually
contains, and the model needs a 29-date Sentinel-2 L2A stack that a satellite
does not carry onboard. Bundling the input removes that dependency entirely, so
verification and the first orbit pass can proceed while the `/rs` question is
resolved separately. Consuming real `/rs` data is a later version.

Because of this, the metric of record is **ground-vs-orbit logit parity**, not
accuracy: the bundle carries reference logits computed on the ground, and the
flight run reports `max|delta|` and prediction agreement. Accuracy against labels
is also reported, but C03 was trained on all 778 labelled points, so those rows
are **in-sample** and are not evidence of generalization. Parity is the honest
claim — and it is exactly the evidence the ITU paper needs for "validation on
flight-representative hardware".

## Ground test result

```
$ python3 orbit/kybelix_orbit.py --output-dir /tmp/out --selftest
[orbit] bundle: 30 points, 210 query rows, bands=12
[orbit] graph fixed timesteps: T=29
[orbit] timing: preprocess 0.02s, inference 0.873s, total 1.001s, 4.16 ms/row
[orbit] ground parity: max|delta| crop=0.000e+00 stage=0.000e+00, agreement crop=1.0000 stage=1.0000
[orbit] SELFTEST PASSED
```

Outputs `result.json` in the competition format
(`125.5616746_49.2809009_2018/6/8 -> ["soybean", "Greenup"]`) and
`orbit_report.json` with environment, providers, timings and parity.

Runtime is ~1 s against a 30-minute platform job limit, so timeout is not a risk.

---

# Runbook

## Current status (2026-09-07)

Steps 1–3 are **done**. Confirmed values from the platform, already applied to
`Dockerfile`, `app.yaml` and `build_delta.sh`:

| Field | Value |
|---|---|
| Model | `C03_fulldatatrained` (`app-100`), status **Awaiting Submission** |
| Version | `v1.0.0` |
| Image address | `app/c03_fulldatatrained:v1.0.0` |
| Base image | `base/base:base` — Satellite SDK, **Python 3.9**, scheduling runtime |
| Device | SCS-01 DevPod (Orin), 4 cores / 16 GB / 100 GB |
| Workdir | `/app` (the platform's own template; not `/workspace`) |
| Dev environment | `c03-orbit-dev`, **Developing** |

**Next action is Step 4**, in the dev environment's editor. Steps 1–3 below are
kept for the record and for future versions.

## Step 1 — Create the verification task (do this first) — DONE

Submitted 2026-09-05, approved by platform review.

`spaceapp.zero2x.org` → **Verification** → **Create Verification Task**.

Platform review takes **1–3 business days**, and it gates everything downstream,
so start it before the engineering work is finished. Fields:

- **English name**: `kybelix-crop-phenology` — at least 5 characters, letters,
  digits, `-` and `_` only, must start and end alphanumeric. Lowercased into the
  image repository address.
- **Chinese name**: optional.
- **Description**: use `VERIFICATION_TASK.md` in this directory.
- **Attachment**: optional. Attach the ITU paper PDF; the manual says richer
  documentation helps staff assign a suitable satellite group, and the satellite
  group determines what `/rs` holds.

## Step 2 — Initialize the image repository

Once approved: model details page → **Initialize Image Repository** → then
**Create Verification Task and Prepare Image**.

Record the **image address** shown on the model details page. It goes into
`app.yaml` at `pods.kybelix.image` verbatim — a mismatch here is the most common
image-upload failure in the manual.

## Step 3 — Create the spaceborne development environment

Model details page → **Create Development Environment**. The SCS-01 DevPod image
(Orin, ARM64, 4C / 16G / 100G / 1 GPU) is matched from the satellite group.

Wait for state **Developing**, then open the access URL for VS Code Server.

Watch the lifecycle: max 2 concurrent environments per project; auto-released
after 7 idle days; **deleted 7 days after that, which permanently destroys
everything under `/home/spaceapp/project/data`**. Download outputs promptly.

## Step 4 — Validate on the real Orin before building anything

In the dev environment IDE terminal, get this `orbit/` directory onto the box
(git clone, or drag-and-drop upload into `/home/spaceapp/project`), then:

```bash
cat /home/spaceapp/project/base.image          # the FROM value for this device
python3 -c "import numpy, onnxruntime; print(numpy.__version__, onnxruntime.__version__)"
python3 -c "import onnxruntime as o; print(o.get_available_providers())"
pip install -r orbit/requirements.txt          # only if the imports above failed
python3 orbit/kybelix_orbit.py --output-dir /tmp/orbit-test --selftest
```

This is the highest-value step in the whole process. It answers, on the actual
target hardware and before you spend a verification round: does onnxruntime exist
for this ARM64 image, is there a CUDA or TensorRT provider, and does C03 reproduce
the ground logits? The manual is explicit that reproducing problems here rather
than in verification is what keeps the round count down.

If a `CUDAExecutionProvider` or `TensorrtExecutionProvider` shows up and the
selftest passes with it (`--provider TensorrtExecutionProvider`), you may raise
`gpu: 0` to `gpu: 1` in `app.yaml`. If not, leave it at 0 — CPU runs in ~1 s and
requesting a GPU you cannot use only adds a failure mode.

## Step 5 — Build the delta package

```bash
cd /home/spaceapp/project/<your-path>/orbit
./build_delta.sh
```

The script reads `base.image`, rewrites the Dockerfile `FROM` to match, checks the
payload is present, calls `/usr/local/bin/image_tool.sh`, and refuses to hand you
a package over 300 MB. Output lands in `/home/spaceapp/project/data`.

Download it via **Output File List** on the environment details page. (Individual
files download; whole directories do not.)

## Step 6 — Start model verification

Model details page → **Model Verification**. Upload in order:

1. **Image file** — the `.tar` delta package from Step 5.
2. **Model file package** — *skip*. The model is inside the image, which the
   manual permits.
3. **YAML** — `app.yaml`, after editing:
   - `pods.kybelix.image` → the exact address from Step 2.
   - the pod key `kybelix` → whatever pod directory name the platform expects.

## Step 7 — Unit-level verification

Runs once on a ground digital-twin node for one satellite:

```
Parse YAML → Upload image → Upload algorithm files → Device self-check
  → Start container → Run application → Return and inspect results
```

Read the logs even when it passes. Expected in `orbit_report.json`:
`status: ok`, 210 query rows, 210 unique keys, and `ground_parity` agreement of
1.0000 on both heads. Then click through to the next step.

If it fails, the manual's mapping covers nearly every case: image-upload failures
are almost always a YAML `image`/`tag` mismatch or an oversized package;
container-start failures are usually a resource over-request or an architecture
mismatch (rebuild from the correct base); application failures are missing
dependencies or bad paths.

## Step 8 — Simulation workflow verification

The twin constellation schedules the app as a real on-orbit task:

```
Create twin task → Start task → Run application with task
  → Collect results → Clean up environment
```

Allow 10–15 minutes for simulated satellite ingress/egress. If Step 7 passed but
this fails, the cause is scheduling-related — runtime, input timing, or results
written after task completion. Our ~1 s runtime makes that unlikely.

## Step 9 — Verification result review

Platform staff review. On approval the verification report is available from the
verification record.

**The version freezes here.** Its image and model files can no longer be changed
and it cannot be re-verified. Any change needs **Update Version**.

## Step 10 — Submit the on-orbit task

**On-orbit Deployment** → **Submit On-orbit Task**, associating the verified model
and version. You are Project Administrator, so you can do this; Developers cannot.

The task enters **Pending Scheduling** and waits for platform scheduling. Once
scheduled, the task details page shows the scheduled time and the executing
satellite. After execution, download the results from that page —
`result.json` and `orbit_report.json`.

The number to look for is `ground_parity` in `orbit_report.json`. If prediction
agreement is 1.0000, C03 produced identical results in orbit and on the ground.

---

## Open items

1. **`/rs` contents are unknown.** Ask 3body-compute@zhejianglab.org what the
   assigned satellite group's `/rs` mount holds: sensor, bands, GSD, format, and
   coverage. This decides whether a future version can do real on-orbit inference
   rather than replaying a bundled scene, and it is the difference between "our
   model ran in space" and "our model processed satellite data in space".
2. **`clean_paths` semantics** are not defined in the manual beyond path depth.
   Currently set to a scratch path; confirm it never touches `/output`.
3. **Pod key name** in `app.yaml` must match the platform's expected pod directory.
4. **The Dockerfile has not been built against the real base image** — it is on an
   internal address unreachable from here. Step 4 and `build_delta.sh` are what
   catch problems there.
