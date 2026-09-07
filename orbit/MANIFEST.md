# Shipping manifest — orbit/ v1.0.0

Pins exactly what goes into the image. Regenerate after any change:

```bash
shasum -a 256 kybelix_orbit.py requirements.txt model/* sample/*
```

Verified: 2026-09-07T08:22:09Z

## Platform binding

| Field | Value |
|---|---|
| Model | C03_fulldatatrained (app-100) |
| Version | v1.0.0 |
| Image address | app/c03_fulldatatrained:v1.0.0 |
| Base image | base/base:base (Satellite SDK, Python 3.9) |
| Device | SCS-01 DevPod / Orin, 4C / 16G / 100G |
| Workdir | /app |

## Files copied into the image

```
5201e9c2cb71f7e71446c47a64532b664e3599e209e370eb58dd2715107f64dc  kybelix_orbit.py
4c8b262f4cd68fc26e698abf5dfc0b4f1da2e40ca68f88eb8b7c3f338c9c83a4  requirements.txt
446f38ad04f27992eba55c17b7e597cd4e50870668439fff11f02707f8b619ba  model/c03.onnx
f80d1d424ad43b47e7896989c0e356ce24513c54b0a33e8210f68300cb662dd8  model/c03.meta.json
7185e525c92578817a93d27285db010f077697efd27990331210beb098a3f9dd  model/band_stats.json
03c564a4f51ff9d59581153c417f7da4f725c149b6b9bed2ca54266e42531a89  sample/demo_input.npz
5ca0aac69b9f652d9ea815a1ffa1d60ee132d0a3ef5786c7d06df73b65c25b71  sample/demo_input.manifest.json
```

Payload:  15M (cap: 300 MB)

## Not copied into the image

`Dockerfile`, `app.yaml`, `build_delta.sh`, `tools/`, `README.md`, `VERIFICATION_TASK.md`, `MANIFEST.md` — build-time and documentation only.

## Last verification

```
layout      : simulated /app, run from cwd=/
bundle      : 30 points, 210 query rows, T=29
parity      : max|delta| crop=0.000e+00 stage=0.000e+00
agreement   : crop=1.0000 stage=1.0000
timing      : 0.95 s total, 4.1 ms/row (CPU, Python 3.9.6)
selftest    : PASSED (exit 0)
```
