# Verification task submission text

Paste into **Verification → Create Verification Task** on `spaceapp.zero2x.org`.

---

**English name**

```
kybelix-crop-phenology
```

**Chinese name** (optional)

```
作物类型与物候期识别模型
```

---

**Description**

```
Kybelix C03 is a query-aware CNN-Transformer model for crop type classification and
rice phenological stage estimation from Sentinel-2 optical time series.

FUNCTION
Given a 15x15 pixel patch time series over one ground point (12 Sentinel-2 bands,
29 acquisition slots) and a query date, the model returns two labels for that point
on that date: crop type (corn / rice / soybean) and phenophase stage (Greenup,
MidGreenup, Peak, Maturity, MidSenescence, Senescence, Dormancy).

TECHNICAL APPROACH
A small CNN encodes each acquisition's patch into an embedding. A 4-layer
Transformer encoder (d_model 256, 8 heads) attends across the acquisition axis
with a padding mask, using a sinusoidal day-of-year encoding. A separate query-date
encoding is concatenated to the masked temporal pooling output, and two linear
heads produce crop and stage logits. Input is 24 channels: 12 normalized reflectance
bands plus 12 per-band valid-pixel mask channels, so cloud and border invalidity is
an explicit model input rather than silent noise.

SCALE
Parameters: 3,275,533
Model file: 13.3 MB (ONNX, opset 17)
Total image payload: 16 MB (graph + normalization statistics + bundled test input)
Runtime: approximately 1 second for 210 inference rows on CPU
Requested resources: 2 CPU, 4096 MB memory, 0 GPU

The model runs on CPU within the job time limit, so no GPU is requested. Inference
uses ONNX Runtime; no PyTorch is installed, which keeps the image delta far below
the 300 MB limit on ARM64.

WHY ON-ORBIT OPERATION IS NECESSARY
Agricultural monitoring is bandwidth-bound, not compute-bound. A multi-band optical
scene is large; the decision derived from it is a few bytes per ground point. Running
this model on the spaceborne node lets the satellite downlink compact crop and
phenology labels instead of raw imagery, which is the difference between a link
budget that supports frequent revisit-rate monitoring and one that does not. Timeliness
matters for the same reason: phenological stage transitions are the trigger for
irrigation, fertilization and harvest advisories, and their value decays with latency.

This work was developed for the Space Intelligence Empowering Zero Hunger track and
is documented in an ITU Journal submission. Ground evaluation on a fully hidden test
set (942 query rows, 171 points, 2,235 TIFFs) gave a competition score of 0.9941. The
paper currently states that on-orbit claims require validation on flight-representative
hardware; this verification is intended to supply exactly that evidence.

VERSION 1.0.0 SCOPE
This first version is deliberately self-contained. It carries a bundled input of 30
labelled points and 210 query rows inside the image and does not read the satellite
`/rs` mount, so the run has no external data dependency. It also carries reference
logits computed on the ground, and reports the maximum absolute logit difference and
prediction agreement between the orbit run and the ground run. The purpose of this
version is to establish that the model deploys, executes and reproduces ground results
bit-comparably on the spaceborne compute node.

A subsequent version will consume real remote sensing data from `/rs`. We would
appreciate guidance on the sensor, band set, ground sample distance, file format and
coverage of the data available to the assigned satellite group, so that version can
be designed against the actual on-orbit input.
```

---

**Attachment** (optional but recommended)

Attach the ITU Journal paper PDF. The manual notes that detailed design
documentation helps platform staff assign a suitable satellite group — and the
satellite group determines what `/rs` provides, which is open item 1 in the runbook.

Candidates in this project:
- `query_aware_sentinel2_article_readable.pdf`
- `revision_ar20/ar20_revised.docx` (export to PDF first)
