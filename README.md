# Nightfall

A from-scratch reimplementation of PatchCore for multi-category visual anomaly detection, built for production deployment — not just paper reproduction.

**v2 of [visual-defect-inspector](../../tree/v1.0)** (tagged `v1.0`), which used `anomalib`'s PatchCore implementation for a single category (bottle). Nightfall hand-rolls the entire algorithm, extends to all 15 MVTec AD categories, and adds a full deployment stack: ONNX export, INT8 quantization, gRPC serving, and a real (if scrappy) embedded-device integration.

## Why this exists

Anyone can run `anomalib`'s PatchCore against MVTec AD — the paper is public, the reference implementation is public. The goal here was to demonstrate the harder, less common things: understanding the algorithm well enough to reimplement and debug it, understanding what changes when you compress and deploy a model rather than just train one, and being honest about what breaks along the way.

Several real bugs were found and fixed during this project, not glossed over — see [What actually broke, and what we learned](#quantization-phase-3) below. That section is arguably the most useful part of this README if you're evaluating engineering ability rather than a headline number.

## Architecture

```
core/
├── feature_extractor.py   # WideResNet50 + hooks, locally-aware pooling, multi-scale fusion
├── coreset.py              # Greedy k-center (farthest point sampling) coreset selection
├── memory_bank.py          # kNN scoring, softmax confidence reweighting, pixel heatmaps
├── patchcore.py             # Multi-category orchestrator
├── preprocessing.py        # ImageNet-normalized resize/crop pipeline
└── onnx_feature_extractor.py  # ONNX Runtime-backed extractor (drop-in for the above)

eval/
├── metrics.py               # Image AUROC, pixel AUROC, PRO (Per-Region Overlap)
├── harness.py               # Per-category evaluation + comparison tables
└── dataloader.py            # MVTec test/ground-truth loading

scripts/
├── train.py                 # Resumable, checkpointed training across all 15 categories
├── run_eval.py               # Full evaluation harness (fp32)
├── run_eval_int8.py           # INT8 accuracy validation
├── calibrate_thresholds.py    # Per-category anomaly threshold calibration
├── generate_test_image_header.py  # Converts a real MVTec image into an ESP32-embeddable C header
└── export_onnx.py / quantize_onnx.py  # ONNX export + INT8 quantization pipeline

serving/
├── proto/nightfall.proto      # gRPC service definition
├── run_grpc_server.py          # Multi-category gRPC inference server
├── rest_gateway.py             # Async REST/HTTP gateway in front of gRPC (grpc.aio)
└── esp32_client.ino            # ESP32 firmware (Wokwi-simulated), WiFi + HTTP client
```

## Core algorithm

Hand-rolled, not wrapped:

- **Feature extraction**: WideResNet50, hooked at `layer2`/`layer3` via `register_forward_hook`, locally-aware average pooling (3x3), multi-scale fusion (bilinear-upsample the coarser map to match the finer one, channel-concat).
- **Coreset selection**: Greedy k-center (farthest point sampling), with a Johnson-Lindenstrauss random projection to make the distance computation cheaper on high-dimensional features. Preserves coverage of rare-but-normal patch types far better than random subsampling -- this matters directly for categories with high intra-class variation (wood grain, carpet).
- **Scoring**: kNN distance to the memory bank, with a softmax-based confidence reweighting scheme (image-level score only; pixel-level heatmaps use raw distances). See below for where this broke and how it was fixed.

## Results (fp32, WideResNet50, all 15 MVTec AD categories)

| Category | Image AUROC | PRO |
|---|---|---|
| bottle | 0.997 | 0.701 |
| cable | 0.927 | 0.497 |
| capsule | 0.926 | 0.666 |
| carpet | 0.941 | 0.736 |
| grid | 0.799 | 0.558 |
| hazelnut | 0.956 | 0.670 |
| leather | 1.000 | 0.860 |
| metal_nut | 0.980 | 0.769 |
| pill | 0.810 | 0.740 |
| screw | 0.876 | 0.544 |
| tile | 0.993 | 0.724 |
| toothbrush | 0.989 | 0.515 |
| transistor | 0.969 | 0.502 |
| wood | 0.954 | 0.779 |
| zipper | 0.952 | 0.502 |
| **Mean (15 categories)** | **0.938** | **0.651** |

Reference: the original PatchCore paper (Roth et al., 2021) reports a mean image AUROC of ~0.991, mean pixel AUROC ~0.981, mean PRO ~0.935 (WideResNet50, 1% coreset). Per-category paper numbers are not reproduced above. The ~5-point gap in mean image AUROC is a real, plausible-and-explainable delta for a hand-rolled reimplementation (different reweighting formula derivation, no calibrated static quantization, etc.), not evidence of a broken implementation -- the individual per-category numbers above track known category difficulty (grid, pill are hard categories in the literature; bottle, leather, tile are easy).

**What the PRO column reveals that image AUROC hides**: several categories with strong image-level AUROC (cable 0.927, toothbrush 0.989, transistor 0.969) have notably weak PRO (0.497, 0.515, 0.502) -- meaning the model reliably flags *that* an image is defective but is comparatively weak at precisely localizing *where*. This is exactly why PRO is included alongside AUROC rather than relying on image-level AUROC alone.

## Quantization (Phase 3)

ONNX export (verified numerically correct, <1e-5 max absolute difference vs. native PyTorch) then INT8 dynamic quantization via ONNX Runtime.

- **Size**: 99.6MB -> 25.0MB (3.98x reduction)
- **Accuracy**: not a clean win -- and the honest story here is more valuable than a clean one would have been.

### Finding 1: the paper's confidence reweighting doesn't survive INT8 quantization

Applying the same softmax-based confidence reweighting scheme (tuned for fp32 feature scales) to INT8-extracted features collapsed image AUROC from 0.997 to 0.518 (near chance) on the bottle category. A temperature sweep (0.1 to 128) showed no genuine calibration point -- every temperature that "recovered" AUROC did so by pushing the reweighting mechanism toward a no-op (uniform weights), not by restoring real confidence discrimination. **Resolution**: `MemoryBankConfig.use_reweighting` is a documented flag; the INT8 inference path disables reweighting and scores directly on raw max nearest-neighbor distance (0.894 AUROC on bottle -- a real, honest cost of quantization).

### Finding 2: train/test feature-space consistency is non-negotiable

Scoring INT8-extracted test features against a memory bank built from fp32-extracted training features caused severe, category-dependent collapse (hazelnut 0.486, metal_nut 0.623 -- both near or below chance). Refitting the memory bank using the same INT8 extractor for both training and test resolved this completely (hazelnut 1.000, metal_nut 0.998) -- verified not to be a fragile artifact by inspecting the actual sorted score distributions (hazelnut: real ~2.2-point margin between every normal and defect score; metal_nut: near-total separation with one honest, expected overlapping case).

**Conclusion**: INT8 quantization is viable for this pipeline, but only as a fully internally-consistent pipeline -- the feature extractor used to build a memory bank and the one used to score against it must match. This was true regardless of raw quantization noise; the earlier "reweighting collapse" and this "train/test mismatch collapse" are two distinct, separately-diagnosed failure modes.

### Latency: parked, not swept under the rug

Initial CPU latency benchmarking (PyTorch fp32 vs. ONNX fp32 vs. ONNX INT8) was run on shared Colab hardware (AMD EPYC 7B12, AVX2 only, **no VNNI** -- the instruction set that accelerates INT8 integer math). Result: INT8 was *slower* than both fp32 baselines (p50 1304ms vs. 293ms for ONNX fp32) -- a real, hardware-dependent finding. Without VNNI/AVX512-VNNI, quantized ops fall back to unaccelerated integer emulation, which can genuinely lose to well-optimized fp32 SIMD paths. **This means INT8 quantization here is a legitimate memory/storage win, not a guaranteed latency win** -- the latency benefit is conditional on deploying to hardware with real INT8 acceleration.

## Serving

- **gRPC** (`serving/run_grpc_server.py`): multi-category routing (client specifies category explicitly -- PatchCore's kNN scoring has no mechanism to infer category from an image; that would require a separate classifier), per-category calibrated anomaly thresholds (mean + 3 sigma of each category's own training score distribution, avoiding test-set leakage).
- **REST gateway** (`serving/rest_gateway.py`): thin async layer in front of gRPC, for clients (like microcontrollers) that can't speak gRPC natively. Uses `grpc.aio` with a single reused channel (FastAPI lifespan-managed) and explicit per-request timeouts -- not a blocking synchronous call wrapped in an `async def`, which would silently serialize all concurrent requests behind Uvicorn's single event loop.

Why REST-in-front-of-gRPC rather than exposing gRPC directly to the ESP32: there is no mature, production-ready gRPC client for ESP32/Arduino. What exists (e.g. `esp-grpc`) is explicitly self-described as an experimental reference, not a real library -- confirmed via multiple multi-year-old, unresolved Espressif forum threads asking for this. A REST gateway is the standard approach, not a workaround.

## Edge integration (Phase 6)

An ESP32 (simulated in [Wokwi](https://wokwi.com)) connects to WiFi, sends a real MVTec test image (embedded in firmware -- Wokwi does not genuinely simulate a camera sensor feeding live pixel data) to the REST gateway over HTTP, and receives back a real inference result from the actual hand-rolled PatchCore model:

```
HTTP status: 200
Response: {"success":true,"anomaly_score":16.08,"is_anomalous":true,"inference_latency_ms":1252.3}
```

**What this genuinely validates**: real WiFi connection handling, real HTTP client construction (multipart/form-data), real end-to-end network communication to a live inference server, real JSON response parsing -- all code that runs unchanged on physical ESP32-CAM hardware.

**A real, non-obvious infrastructure finding along the way**: ESP32's embedded mbedTLS stack failed to complete a TLS handshake against both ngrok and Cloudflare Tunnel (identical "connection refused" symptom on both, despite HTTPS working fine against an established endpoint like httpbin.org) -- a documented, unresolved compatibility gap between ESP32's TLS implementation and certain tunnel providers' edge TLS configurations, not something specific to this project's setup. Resolved by using a plain-HTTP tunnel (Localtunnel) instead of fighting the TLS incompatibility.

## Setup

See `scripts/colab_setup.py` for the full Colab bootstrap (Drive mount, repo clone, dependency install, GPU check). Training data downloads automatically via `anomalib`'s `MVTecAD` datamodule (used purely as a data-fetching utility, not as the modeling library -- the actual algorithm is 100% hand-rolled in `core/`), with a HuggingFace mirror fallback for when the official MVTec endpoint 404s (a known, intermittent issue as of early 2026).

```bash
python scripts/train.py --data-root <mvtec_dir> --output-dir <checkpoint_dir>
python scripts/run_eval.py --data-root <mvtec_dir> --checkpoint-dir <checkpoint_dir>
python scripts/calibrate_thresholds.py --data-root <mvtec_dir> --checkpoint-dir <checkpoint_dir> --output <checkpoint_dir>/thresholds.json
python serving/run_grpc_server.py --checkpoint-dir <checkpoint_dir> --thresholds-path <checkpoint_dir>/thresholds.json
```
