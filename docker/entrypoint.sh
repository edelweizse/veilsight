#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-serve}" != "serve" ]]; then
  exec "$@"
fi

export PYTHONPATH="${PYTHONPATH:-/opt/veilsight/controller/generated:/opt/veilsight}"
if [[ -f /opt/veilsight/configs/docker_full_reference.yaml ]]; then
  export VEILSIGHT_CONFIG="${VEILSIGHT_CONFIG:-/opt/veilsight/configs/docker_full_reference.yaml}"
else
  export VEILSIGHT_CONFIG="${VEILSIGHT_CONFIG:-/opt/veilsight/configs/full_reference.yaml}"
fi
export VEILSIGHT_CONFIG_DIR="${VEILSIGHT_CONFIG_DIR:-/opt/veilsight/configs}"
export VEILSIGHT_WEB_DIST="${VEILSIGHT_WEB_DIST:-/opt/veilsight/web/dist}"
export VEILSIGHT_RUNNER_GRPC="${VEILSIGHT_RUNNER_GRPC:-unix:///tmp/veilsight-runner.sock}"
export VEILSIGHT_RENDER_BINARY="${VEILSIGHT_RENDER_BINARY:-/opt/veilsight/build/apps/render_video/veilsight_render_video}"
export VEILSIGHT_ANALYTICS_DB_PATH="${VEILSIGHT_ANALYTICS_DB_PATH:-/opt/veilsight/data/veilsight_analytics.sqlite3}"
export VEILSIGHT_GALLERY_DB_PATH="${VEILSIGHT_GALLERY_DB_PATH:-/opt/veilsight/data/mobilefacenet_gallery.sqlite3}"
export VEILSIGHT_NCNN_VULKAN="${VEILSIGHT_NCNN_VULKAN:-0}"

mkdir -p /opt/veilsight/assets /opt/veilsight/results /opt/veilsight/data /tmp
rm -f /tmp/veilsight-runner.sock

if [[ ! -f "$VEILSIGHT_GALLERY_DB_PATH" ]]; then
  python3 scripts/identity/init_mobilefacenet_gallery_db.py "$VEILSIGHT_GALLERY_DB_PATH"
fi

runner_bin="/opt/veilsight/build/apps/core_service/veilsight_core_service"
"$runner_bin" "$VEILSIGHT_CONFIG" &
runner_pid=$!
python3 -m uvicorn controller.veilsight_controller.main:app --host 0.0.0.0 --port 8000 &
controller_pid=$!

terminate() {
  kill "$controller_pid" "$runner_pid" 2>/dev/null || true
  wait "$controller_pid" "$runner_pid" 2>/dev/null || true
}
trap terminate INT TERM EXIT

wait -n "$controller_pid" "$runner_pid"
status=$?
exit "$status"
