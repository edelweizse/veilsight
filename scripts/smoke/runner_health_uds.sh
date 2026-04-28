#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${ROOT_DIR}/build/apps/core_service/veilsight_core_service"
CONFIG="${VEILSIGHT_CONFIG:-${ROOT_DIR}/configs/dual_example.yaml}"

if [[ ! -x "$RUNNER" ]]; then
  echo "runner binary not found at $RUNNER" >&2
  exit 2
fi

"$RUNNER" "$CONFIG" --no-autostart &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT
sleep 2

python - <<'PY'
import asyncio
from controller.veilsight_controller.runner_client import RunnerClient

async def main():
    client = RunnerClient("unix:///tmp/veilsight-runner.sock")
    health = await client.health()
    await client.close()
    assert health["ok"] is True, health
    print("ok=true")

asyncio.run(main())
PY
