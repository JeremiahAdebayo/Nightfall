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
import time
from concurrent import futures
from pathlib import Path

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

    def __init__(self, checkpoint_dir: Path, anomaly_threshold: float):
        self.model = PatchCore()
        self.anomaly_threshold = anomaly_threshold
        self._load_available_categories(checkpoint_dir)

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

        # Validate category BEFORE doing any image decoding or inference
        # work -- fail fast and cheaply on a bad request rather than
        # spend compute on a request we're going to reject anyway.
        if request.category not in self.model.banks:
            return nightfall_pb2.AnomalyResponse(
                success=False,
                error_message=(
                    f"Unknown category '{request.category}'. "
                    f"Available categories: {list(self.model.banks.keys())}"
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
            is_anomalous = anomaly_score > self.anomaly_threshold

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


def serve(checkpoint_dir: Path, port: int, anomaly_threshold: float, max_workers: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = NightfallServicer(checkpoint_dir, anomaly_threshold)
    nightfall_pb2_grpc.add_NightfallInferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Nightfall gRPC server listening on port {port}")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--anomaly-threshold", type=float, default=None,
        help=(
            "Score threshold above which is_anomalous=True. NOT calibrated "
            "here -- this needs to be set per-deployment based on the "
            "actual score distributions observed in Phase 2 eval (see "
            "scripts/run_eval.py's output for real score ranges per "
            "category). Passing no value raises an error rather than "
            "silently using an arbitrary guess."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if args.anomaly_threshold is None:
        raise ValueError(
            "--anomaly-threshold is required. This is a deliberate design "
            "choice: PatchCore's raw distance scores have no universal "
            "'anomalous' cutoff -- the right threshold depends on category "
            "and the score distributions seen during eval. Pick a value "
            "informed by Phase 2's actual per-category score ranges rather "
            "than an arbitrary default that would silently mislabel results."
        )

    serve(args.checkpoint_dir, args.port, args.anomaly_threshold, args.max_workers)


if __name__ == "__main__":
    main()