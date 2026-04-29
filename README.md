# Veilsight

Veilsight is a two-service privacy-preserving video pipeline.

## Architecture

```text
Browser  -> Runner      WebRTC/WHEP video, MJPEG fallback, snapshots, metadata
Browser  -> Controller  HTTP API, WebSockets, React dashboard
Controller -> Runner    gRPC control/config/telemetry over UDS by default
Runner   -> Controller  Telemetry stream cached and fanned out over WebSockets
```

The Runner owns ingest, inference, anonymization, metrics, MJPEG fallback, and the direct browser video endpoint. The Controller owns browser APIs, config persistence, lifecycle commands, telemetry cache, WebSocket fanout, and production serving of the React app in `web/dist`. `ui/app.py` remains as a legacy development fallback.

## Build

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Python controller setup:

```bash
python -m pip install -r controller/requirements.txt
scripts/generate_proto_python.sh
PYTHONPATH=controller/generated:. pytest controller/tests
```

React dashboard:

```bash
npm --prefix web install
npm --prefix web test
npm --prefix web run build
```

## Runtime

Runner:

```bash
./build/apps/core_service/veilsight_core_service configs/dual_example.yaml
```

Controller:

```bash
PYTHONPATH=controller/generated:. fastapi dev
```

React dev server:

```bash
npm --prefix web run dev
```

In production, build the React app and run the Controller; FastAPI serves `web/dist` at `/`.

## Configuration

Person tracking uses a configurable person detector followed by ByteTrack. Detector boxes are expected in inference-frame coordinates, and anonymization remains fail-closed for all emitted person tracks:

```yaml
modules:
  detector:
    type: "yolox" # yolox | future_person_backend
    workers: 2
    yolox:
      variant: "nano"
      param_path: "models/detector/bytetrack_nano.ncnn.param"
      bin_path: "models/detector/bytetrack_nano.ncnn.bin"
      input_w: 1088
      input_h: 608
      score_threshold: 0.35
      nms_threshold: 0.45
      top_k: 300
      class_id: 0
      ncnn_threads: 1
      letterbox: true
      decoded_output: false
  tracker:
    type: "bytetrack"
    bytetrack:
      high_thresh: 0.45
      low_thresh: 0.2
      new_track_thresh: 0.45
      match_iou_thresh: 0.35
      low_match_iou_thresh: 0.25
      unconfirmed_match_iou_thresh: 0.35
      duplicate_iou_thresh: 0.85
      track_buffer: 100
      min_box_area: 0.0
      fuse_score: true
      scene_grid:
        enabled: true
        rows: 6
        cols: 8
        association_weight: 0.15
        max_extra_cost: 0.30
```

Default same-device gRPC transport:

```yaml
runner:
  grpc:
    listen: "unix:///tmp/veilsight-runner.sock"
    fallback_tcp: "127.0.0.1:9090"
  public_base_url: "http://localhost:8080"

streaming:
  primary: "webrtc"
  fallback: "mjpeg"
  codec: "h264"
  encoder: "auto"
  bitrate_kbps: 2500
  keyframe_interval_frames: 30
  webrtc:
    enabled: true
    max_peers_per_stream: 2
    ice_gathering_timeout_ms: 2000
    session_idle_timeout_s: 30
    stun_servers: []
    cors_allowed_origins:
      - "http://localhost:8000"
      - "http://127.0.0.1:8000"
      - "http://localhost:5173"
      - "http://127.0.0.1:5173"
```

Reload is whole-pipeline: the Runner validates first, stops the old pipeline, starts the new one, and attempts rollback to the previous config if the new start fails.

## Controller Analytics

The Controller owns MVP analytics in SQLite. The Runner still sends per-frame `FrameAnalytics` track telemetry; the Controller enqueues it with a bounded non-blocking queue, computes overlays/events/snapshots in a background worker, and persists only numeric geometry, rule definitions, events, and aggregate snapshots.

Default controller analytics settings can be overridden with environment variables:

```yaml
analytics:
  enabled: true
  db_path: "data/veilsight_analytics.sqlite3"
  ingest_queue_capacity: 2000
  snapshot_interval_ms: 500
  heatmap:
    rows: 36
    cols: 64
    decay_half_life_s: 900
  routes:
    max_points_per_track: 256
    simplify_epsilon_px: 3
    min_route_support: 3
  retention:
    raw_track_samples_days: null
    events_days: null
    aggregate_snapshots_days: null
```

`null` retention means the Controller does not prune that table automatically. Current retention and queue drop counters are returned by `GET /api/analytics/summary`, so operators can verify pruning is explicitly configured.

## APIs

Runner gRPC services:

- `PipelineControlService`: `Health`, `GetStatus`, `GetStreams`, `ValidateConfig`, `Start`, `Stop`, `Reload`
- `RunnerTelemetryService`: `WatchTelemetry`, `GetMetricsSnapshot`

Controller HTTP/WS:

- `GET /api/health`
- `GET /api/config`, `PUT /api/config`, `POST /api/config/validate`
- `POST /api/pipeline/start`, `POST /api/pipeline/stop`, `POST /api/pipeline/reload`
- `GET /api/pipeline/status`, `GET /api/streams`, `GET /api/metrics`, `GET /api/analytics/latest`
- `GET /api/analytics/rules`, `POST /api/analytics/rules`, `PUT /api/analytics/rules/{rule_id}`, `DELETE /api/analytics/rules/{rule_id}`
- `GET /api/analytics/summary`, `GET /api/analytics/events`, `GET /api/analytics/snapshot`
- `WS /ws/analytics`, `WS /ws/analytics/overlays`, `WS /ws/metrics`

Runner HTTP:

- `GET /health`
- `GET /streams`
- `GET /metrics`
- `GET /meta/<stream_id>/<profile>`
- `GET /snapshot/<stream_id>/<profile>`
- `GET /video/<stream_id>/<profile>`
- `POST /webrtc/whep/<stream_id>/<profile>`
- `DELETE /webrtc/whep/sessions/<session_id>`
- `PATCH /webrtc/whep/sessions/<session_id>` returns `501` until trickle ICE is implemented

## Troubleshooting

- Missing `webrtcbin`: install GStreamer WebRTC plugins. Runner still starts with MJPEG fallback and reports WebRTC unavailable.
- Missing H.264 encoder: install one of `v4l2h264enc`, `vaapih264enc`, `nvh264enc`, `x264enc`, or `openh264enc`, or set `streaming.encoder` to an installed encoder.
- Protobuf mismatch: CMake checks `protoc` against pkg-config protobuf and fails early if major/minor versions differ.
- Unix socket permissions: remove stale `/tmp/veilsight-runner.sock` or run Runner/Controller under users that can access it.
- CORS failures: add the Controller or Vite origin to `streaming.webrtc.cors_allowed_origins`.
