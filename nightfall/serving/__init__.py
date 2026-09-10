"""
Serving layer: gRPC inference server, async REST gateway, and the ESP32
client + proto sources.

The two `nightfall_pb2*.py` modules in here are generated from
proto/nightfall.proto (see run_grpc_server.py's docstring for the exact
grpc_tools command, and the one-line post-generation import fix that keeps
them importable as part of this package).
"""
