#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p controller/generated
python -m grpc_tools.protoc \
  -I proto \
  --python_out=controller/generated \
  --grpc_python_out=controller/generated \
  proto/veilsight/runner/v1/runner.proto
