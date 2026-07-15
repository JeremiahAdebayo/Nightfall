"""
gRPC server for Nightfall anomaly detection inference.

Before running this server, generate the protobuf/gRPC Python stubs from
nightfall.proto (this is a one-time step, or repeated whenever the .proto
changes):

    python -m grpc_tools.protoc \
        -I serving/proto \
        --python_out=serving \
        --grpc_python_out=serving \
        serving/proto/nightfall.proto

This generates serving/nightfall_pb2.py and serving/nightfall_pb2_grpc.py,
which this server imports. These generated files are NOT hand-written and
should be regenerated from the .proto, not edited directly.

Usage:
    python scripts/run_grpc_server.py \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --port 50051
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent import futures
from pathlib import Path

# Ensure the repo root (parent of this file's `serving/` directory) is on
# sys.path, regardless of the working directory this script is invoked
# from. Without this, `from scripts.train import ...` below fails when
# run as `python serving/run_grpc_server.py` from the repo root, since
# Python adds the SCRIPT's own directory (serving/) to sys.path by
# default, not the repo root -- a different situation from scripts/
# themselves, which are invoked directly and therefore have the repo
# root as their natural working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import grpc
import numpy as np
import torch
from PIL import Image

from core.patchcore import PatchCore
from core.memory_bank import MemoryBank
from scripts.train import ALL_MVTEC_CATEGORIES, checkpoint_path

import nightfall_pb2
import nightfall_pb2_grpc


class NightfallServicer(nightfall_pb2_grpc.NightfallInferenceServicer):
    """
    Implements the NightfallInference gRPC service defined in
    serving/proto/nightfall.proto.

    Loads all available category checkpoints once at startup, keeping
    them in memory for the server's lifetime -- avoids reloading a
    memory bank from disk on every request, which would add meaningful,
    unnecessary latency per call.
    """

    def __init__(self, checkpoint_dir: Path, thresholds_path: Path):
        self.model = PatchCore()
        self.thresholds = self._load_thresholds(thresholds_path)
        self._load_available_categories(checkpoint_dir)

    @staticmethod
    def _load_thresholds(thresholds_path: Path) -> dict:
        if not thresholds_path.exists():
            raise FileNotFoundError(
                f"No thresholds file found at {thresholds_path}. Run "
                f"scripts/calibrate_thresholds.py first -- there is no "
                f"safe global default threshold, since score scales differ "
                f"meaningfully per category (confirmed empirically: e.g. "
                f"bottle's defective-image scores were far below a naive "
                f"placeholder threshold that happened to work for a "
                f"different category)."
            )
        import json
        return json.loads(thresholds_path.read_text())

    def _load_available_categories(self, checkpoint_dir: Path) -> None:
        loaded = []
        for category in ALL_MVTEC_CATEGORIES:
            ckpt_path = checkpoint_path(checkpoint_dir, category)
            if not ckpt_path.exists():
                continue
            bank = MemoryBank(self.model.bank_config)
            bank.fit(torch.load(ckpt_path))
            self.model.banks[category] = bank
            loaded.append(category)

        if not loaded:
            raise RuntimeError(
                f"No category checkpoints found under {checkpoint_dir} -- "
                f"the server has nothing to serve. Run training first."
            )
        print(f"Loaded {len(loaded)} categories: {loaded}")

    def ListCategories(self, request, context):
        return nightfall_pb2.ListCategoriesResponse(
            categories=list(self.model.banks.keys())
        )

    def DetectAnomaly(self, request, context):
        start = time.perf_counter()

        # Validate category has BOTH a fitted bank AND a calibrated
        # threshold before doing any real work -- a bank without a
        # threshold entry is a real, checkable inconsistency (e.g. a
        # category trained after the last calibration run), not
        # something to silently paper over with a guessed default.
        if request.category not in self.model.banks:
            return nightfall_pb2.AnomalyResponse(
                success=False,
                error_message=(
                    f"Unknown category '{request.category}'. "
                    f"Available categories: {list(self.model.banks.keys())}"
                ),
            )
        if request.category not in self.thresholds:
            return nightfall_pb2.AnomalyResponse(
                success=False,
                error_message=(
                    f"Category '{request.category}' has a fitted memory bank "
                    f"but no calibrated threshold -- re-run "
                    f"scripts/calibrate_thresholds.py to include it."
                ),
            )

        try:
            image = Image.open(io.BytesIO(request.image_data)).convert("RGB")
        except Exception as e:
            return nightfall_pb2.AnomalyResponse(
                success=False,
                error_message=f"Failed to decode image_data: {e}",
            )

        try:
            image_tensor = self.model.preprocessor(image).unsqueeze(0)  # add batch dim
            scoring = self.model.predict(request.category, image_tensor)

            anomaly_score = float(scoring.image_score.item())
            threshold = self.thresholds[request.category]["threshold"]
            is_anomalous = anomaly_score > threshold

            heatmap_png = self._encode_heatmap(scoring.pixel_map[0])

            latency_ms = (time.perf_counter() - start) * 1000

            return nightfall_pb2.AnomalyResponse(
                success=True,
                anomaly_score=anomaly_score,
                is_anomalous=is_anomalous,
                heatmap_png=heatmap_png,
                inference_latency_ms=latency_ms,
            )
        except Exception as e:
            return nightfall_pb2.AnomalyResponse(
                success=False,
                error_message=f"Inference failed: {e}",
            )

    @staticmethod
    def _encode_heatmap(pixel_map: torch.Tensor) -> bytes:
        """
        Normalizes a raw pixel-level anomaly map to 0-255 grayscale and
        encodes as PNG bytes for direct client-side display. Normalization
        is per-image (min-max of THIS image's map), not a fixed global
        scale -- appropriate for visualization, but means heatmap
        intensity is NOT directly comparable across different images or
        categories. The raw anomaly_score field (not derived from this
        normalized map) is what should be used for actual thresholding
        and cross-image comparison.
        """
        arr = pixel_map.detach().cpu().numpy()
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)

        img = Image.fromarray(normalized, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def serve(checkpoint_dir: Path, thresholds_path: Path, port: int, max_workers: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = NightfallServicer(checkpoint_dir, thresholds_path)
    nightfall_pb2_grpc.add_NightfallInferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Nightfall gRPC server listening on port {port}")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds-path", type=Path, required=True,
        help=(
            "Path to a JSON file mapping category -> calibrated threshold, "
            "produced by scripts/calibrate_thresholds.py. Required, no "
            "default: score scales differ meaningfully per category, so a "
            "single global threshold silently mislabels some categories "
            "(confirmed empirically during development)."
        ),
    )
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    serve(args.checkpoint_dir, args.thresholds_path, args.port, args.max_workers)


if __name__ == "__main__":
    main()