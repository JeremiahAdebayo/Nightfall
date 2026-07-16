"""
Minimal REST/HTTP gateway in front of Nightfall's gRPC service.

Why this exists: there is no mature, production-ready gRPC client for
ESP32/Arduino. What does exist (e.g. esp-grpc, chrisomatic/esp-grpc) is
explicitly described by its own authors as an experimental reference,
not a library meant for real projects -- and the official Espressif
forums have multi-year-old unresolved threads asking for gRPC support
that never materialized. A thin HTTP/REST layer is the standard,
correct approach for microcontroller clients: ESP32's HTTPClient.h and
WiFiClient.h are mature, first-party libraries that just work, and even
production IoT systems commonly gateway constrained devices through
REST rather than exposing gRPC directly to them.

Concurrency design: uses grpc.aio (gRPC's genuine async API), not the
synchronous grpc.insecure_channel + a blocking stub call inside an
`async def` endpoint. A synchronous gRPC call inside an async endpoint
blocks Uvicorn's single event loop for the call's full duration --
under concurrent load (multiple ESP32s, or any concurrent clients),
every request queues behind whichever one is currently blocking the
loop, defeating the entire point of an async framework. grpc.aio's
channel and stub calls are real awaitables that yield control back to
the event loop while waiting on the network, letting FastAPI actually
serve other requests concurrently. The channel is created ONCE at
startup and reused across all requests (gRPC channels are designed to
be reused/multiplexed, not recreated per-call -- recreating one per
request adds real, unnecessary connection-setup overhead per request).

Timeout design: every gRPC call passes an explicit `timeout` (a client-
side deadline). Without this, a stuck or unusually slow inference call
(model load contention, a huge image, a wedged server) would hang the
requesting client indefinitely -- there's no default gRPC timeout, a
call waits forever unless told otherwise. A fixed deadline bounds worst-
case request latency and ensures a client gets a clear, timely error
instead of hanging, which matters for anything claiming to be
production-grade, not just a demo that only has to work once.

Usage:
    !python serving/rest_gateway.py \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --thresholds-path {DRIVE_ROOT}/checkpoints/thresholds.json \
        --grpc-target localhost:50051 \
        --port 8000

Requires the gRPC server (serving/run_grpc_server.py) to already be
running and reachable at --grpc-target.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SERVING_DIR = Path(__file__).resolve().parent
if str(_SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVING_DIR))

import grpc
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

import nightfall_pb2
import nightfall_pb2_grpc

# Set from CLI args in main(), read inside the lifespan handler and
# request handlers below -- module-level config is a pragmatic choice
# for this single-file MVP.
_grpc_target = "localhost:50051"
_grpc_timeout_seconds = 10.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Creates ONE grpc.aio channel + stub at startup, stores them on
    app.state, and cleanly closes the channel at shutdown. This is the
    standard FastAPI pattern for a long-lived async resource: avoids
    recreating a channel per-request (real overhead) and avoids the
    module-level-global pattern for the channel itself, since
    app.state is FastAPI's intended place for this.
    """
    channel = grpc.aio.insecure_channel(_grpc_target)
    app.state.grpc_channel = channel
    app.state.grpc_stub = nightfall_pb2_grpc.NightfallInferenceStub(channel)
    print(f"REST gateway: gRPC async channel opened to {_grpc_target}")
    yield
    await channel.close()
    print("REST gateway: gRPC channel closed cleanly")


app = FastAPI(title="Nightfall REST Gateway (MVP)", lifespan=lifespan)


def _grpc_error_response(e: grpc.aio.AioRpcError) -> JSONResponse:
    """
    Shared error-to-HTTP-response mapping for both endpoints, so a gRPC
    failure (server down, timeout, etc.) always produces the same kind
    of clean, structured JSON error rather than an unhandled exception
    turning into a bare 500 with no useful body -- exactly what
    happened before this helper existed, when list_categories had no
    error handling at all and a connection failure to the gRPC server
    propagated as a raw, uninformative 500.
    """
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error_message": (
                    f"Request did not complete within "
                    f"{_grpc_timeout_seconds}s -- timed out rather than "
                    f"hanging indefinitely."
                ),
            },
        )
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error_message": (
                    f"Cannot reach the gRPC server at {_grpc_target} -- "
                    f"is it running? ({e.details()})"
                ),
            },
        )
    return JSONResponse(
        status_code=502,
        content={"success": False, "error_message": f"gRPC call failed: {e.details()}"},
    )


@app.get("/categories")
async def list_categories(request: Request):
    """Lets a client (or a curious human) check available categories
    before sending an image -- mirrors the gRPC ListCategories RPC."""
    stub = request.app.state.grpc_stub
    try:
        response = await stub.ListCategories(
            nightfall_pb2.ListCategoriesRequest(), timeout=_grpc_timeout_seconds
        )
    except grpc.aio.AioRpcError as e:
        return _grpc_error_response(e)
    return {"categories": list(response.categories)}


@app.post("/detect")
async def detect_anomaly(
    request: Request, category: str = Form(...), image: UploadFile = File(...)
):
    """
    MVP endpoint: accepts a category string and an image file (as
    multipart/form-data, the format ESP32's HTTPClient can send with a
    manually-constructed multipart body, or trivially from any HTTP
    client library). Returns a plain JSON response -- no protobuf, no
    binary heatmap in this MVP response (the heatmap PNG bytes from the
    underlying gRPC response are dropped here; an ESP32 with a serial
    monitor has no way to display an image anyway, and returning it
    would bloat the response for no benefit to this specific client).

    The await on the gRPC call below is what actually makes concurrent
    requests work correctly: while THIS request is waiting on the
    network/inference, the event loop is free to make progress on
    other requests, rather than blocking everyone behind this one call.
    """
    image_bytes = await image.read()

    grpc_request = nightfall_pb2.AnomalyRequest(
        category=category,
        image_data=image_bytes,
        image_format=(image.filename.split(".")[-1] if image.filename else "png"),
    )

    stub = request.app.state.grpc_stub
    try:
        grpc_response = await stub.DetectAnomaly(
            grpc_request, timeout=_grpc_timeout_seconds
        )
    except grpc.aio.AioRpcError as e:
        return _grpc_error_response(e)

    if not grpc_response.success:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error_message": grpc_response.error_message},
        )

    return {
        "success": True,
        "anomaly_score": grpc_response.anomaly_score,
        "is_anomalous": grpc_response.is_anomalous,
        "inference_latency_ms": grpc_response.inference_latency_ms,
        # heatmap_png intentionally omitted from this MVP response --
        # see docstring above.
    }


def main():
    global _grpc_target, _grpc_timeout_seconds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpc-target", type=str, default="localhost:50051")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--grpc-timeout-seconds", type=float, default=10.0,
        help=(
            "Client-side deadline for each gRPC call. Bounds worst-case "
            "request latency -- without this, a stuck or slow inference "
            "call would hang the calling client indefinitely, since gRPC "
            "has no default timeout of its own."
        ),
    )
    args = parser.parse_args()

    _grpc_target = args.grpc_target
    _grpc_timeout_seconds = args.grpc_timeout_seconds

    print(f"REST gateway starting on {args.host}:{args.port}, forwarding to gRPC at {args.grpc_target}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()