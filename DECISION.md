# Technical decisions

The reasoning behind each choice, made explicit and defensible. This
reflects Nightfall v2 (the hand-rolled reimplementation); v1 notes from
visual-defect-inspector that no longer apply were removed.

## Why PatchCore

PatchCore achieves higher accuracy than most anomaly-detection methods
while cutting inference cost by a good margin: a frozen ImageNet backbone
plus a memory bank of patch embeddings needs no task training at all.

## Why wide_resnet50_2 (not resnet18)

v1 used resnet18 for size; v2 uses wide_resnet50_2 (the paper's choice)
because the wider backbone's layer2/layer3 features are measurably better
at localizing subtle defects, and the backbone is frozen -- its size only
matters at export time, where ONNX/INT8 handles it (99.6MB -> 25.0MB).

## Why layers 2 and 3

Layer 2 captures low-to-mid level features (textures, edges, local
patterns); layer 3 captures mid-level semantic features (shapes, object
parts). Together they give fine-grained texture information plus broader
structural context, which is what detecting subtle surface defects needs.
Layer 2 alone under-detects small anomalies; layer 3 alone over-smooths
their boundaries.

## Why 1% coreset subsampling (not 10%)

The default sampling_ratio is 0.01 (the paper's setting), not v1's 0.1.
Greedy k-center (farthest point sampling) selection preserves coverage of
rare-but-normal patch types far better than random subsampling at this
size, and the Johnson-Lindenstrauss projection keeps the greedy loop
affordable on high-dimensional features. Empirically this mattered most
for categories with high intra-class variation (wood grain, carpet).

## Why one memory bank per category

kNN scoring has no notion of category boundaries; a shared bank would let
a bottle-shaped normal patch mask a genuinely anomalous cable patch.
Category identity is an explicit routing key in training and serving.

## Why reweighting exists -- and why it has a kill switch

The image-level score is reweighted by a softmax confidence factor over
each patch's top-k bank distances: a clearly-dominant nearest neighbor is
stronger anomaly evidence than one of several equidistant candidates. This
is tuned for fp32 feature scales; under INT8 quantization it collapsed
image AUROC to near-chance (bottle 0.997 -> 0.518, no honest temperature
recalibration found). Hence `MemoryBankConfig.use_reweighting`: enabled by
default (fp32), disabled on the quantized path, which scores raw max
nearest-neighbor distance.

## Why train/test feature spaces must match exactly

Scoring INT8-extracted test features against a bank built from fp32 train
features collapsed categories to near-chance AUROC; refitting the bank
with the same INT8 extractor fully resolved it. The pipeline therefore
treats "same extractor for fit and score" as an invariant, not an option.

## Why gRPC with a REST gateway

gRPC gives typed contracts and efficient transport between server-side
clients; the REST gateway exists because there is no mature gRPC client
for ESP32/Arduino (existing options are self-described experiments). The
gateway uses grpc.aio with a single reused channel and explicit per-call
timeouts -- a blocking sync call inside an async endpoint would silently
serialize all concurrent requests behind the event loop.

## Why per-category calibrated thresholds (train mean + 3 sigma)

Score scales differ meaningfully per category, so no global default is
safe. Thresholds are calibrated from each category's own *training*
scores (mean + 3 sigma), avoiding test-set leakage, and the serving layer
refuses a category that has a bank but no calibrated threshold.

## Why FastAPI (and why the old Flask argument was incomplete)

FastAPI alone doesn't make concurrency work: the real fix was grpc.aio +
a reused channel + deadlines. FastAPI is still the right shell (async
native, pydantic validation, maintained), but the gateway's concurrency
properties come from the gRPC call design, not the framework choice.

## Why plain HTTP to the ESP32 (not HTTPS tunnels)

ESP32's embedded mbedTLS stack fails TLS handshakes against several
tunnel providers (ngrok, Cloudflare Tunnel) with an identical
connection-refused symptom -- a documented compatibility gap. The edge
path uses a plain-HTTP localtunnel endpoint instead; anything more
sensitive would need a real TLS-capable endpoint, not a workaround.
