#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-all}"
export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/controller/generated:${ROOT_DIR}}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/ncnn}"

BINARIES=(
  "build/apps/core_service/veilsight_core_service"
  "build/apps/eval_mot20/veilsight_eval_mot20"
  "build/apps/eval_chokepoint/veilsight_eval_chokepoint"
  "build/apps/enroll_faces/enroll_faces"
  "build/apps/render_video/veilsight_render_video"
)

run_unit() {
  cmake --build build -j2
  ctest --test-dir build --output-on-failure
  PYTHONPATH=controller/generated:. pytest controller/tests
  npm --prefix web test
}

check_native_deps() {
  for binary in "${BINARIES[@]}"; do
    if [[ ! -x "$binary" ]]; then
      echo "missing executable: $binary" >&2
      exit 1
    fi
    if ldd "$binary" | grep "not found"; then
      echo "missing native dependency for $binary" >&2
      exit 1
    fi
  done
}

run_health() {
  docker/entrypoint.sh &
  local stack_pid=$!
  cleanup_health() {
    kill "$stack_pid" 2>/dev/null || true
    wait "$stack_pid" 2>/dev/null || true
  }
  trap cleanup_health RETURN

  local ok=0
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/api/health > /tmp/veilsight-health.json; then
      if python3 - <<'PY'
import json
payload = json.load(open("/tmp/veilsight-health.json"))
raise SystemExit(0 if payload.get("ok") and payload.get("runner", {}).get("ok") else 1)
PY
      then
        ok=1
        break
      fi
    fi
    sleep 1
  done

  if [[ "$ok" != "1" ]]; then
    cat /tmp/veilsight-health.json 2>/dev/null || true
    echo "controller health did not report a reachable runner" >&2
    exit 1
  fi

  curl -fsS http://127.0.0.1:8000/ > /tmp/veilsight-dashboard.html
  if ! grep -q 'id="root"' /tmp/veilsight-dashboard.html; then
    echo "controller did not serve the dashboard shell" >&2
    exit 1
  fi

  cleanup_health
  trap - RETURN
}

write_detector_config() {
  local detector="$1"
  local out="$2"
  local port="$3"
  local socket_path="$4"
  local detector_block

  if [[ "$detector" == "yolox" ]]; then
    detector_block='    type: "yolox"
    model_instances: 1
    yolox:
      variant: "nano"
      param_path: "/opt/veilsight/models/people_detectors/yolox_nano/bytetrack_nano.ncnn.param"
      bin_path: "/opt/veilsight/models/people_detectors/yolox_nano/bytetrack_nano.ncnn.bin"
      input_w: 640
      input_h: 640
      score_threshold: 0.01
      nms_threshold: 0.45
      top_k: 100
      class_id: 0
      ncnn_threads: 1
      letterbox: true
      decoded_output: false'
  elif [[ "$detector" == "uhd" ]]; then
    detector_block='    type: "uhd"
    model_instances: 1
    uhd:
      variant: "s_anc8_w80_64x64_opencv_inter_nearest_static_nopost"
      model_path: "/opt/veilsight/models/people_detectors/UHD/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_static_nopost"
      input_w: 64
      input_h: 64
      score_threshold: 0.05
      nms_threshold: 0.45
      top_k: 100
      ncnn_threads: 1'
  else
    echo "unknown detector: $detector" >&2
    exit 1
  fi

  cat > "$out" <<YAML
server:
  host: "0.0.0.0"
  port: ${port}
controller:
  host: "0.0.0.0"
  port: 8000
runner:
  id: "docker-${detector}-smoke"
  grpc:
    listen: "unix://${socket_path}"
    fallback_tcp: ""
  public_base_url: "http://localhost:${port}"
streaming:
  primary: "mjpeg"
  fallback: "mjpeg"
  codec: "h264"
  encoder: "auto"
  bitrate_kbps: 1000
  keyframe_interval_frames: 30
  webrtc:
    enabled: false
    max_peers_per_stream: 1
    ice_gathering_timeout_ms: 500
    session_idle_timeout_s: 10
    stun_servers: []
    cors_allowed_origins: []
runtime:
  reorder_window: 2
  pending_state_limit: 32
  jpeg_quality: 70
  queues:
    global:
      person_detector_in_capacity: 4
      face_detector_in_capacity: 4
      recognizer_in_capacity: 4
      identity_in_capacity: 4
      anonymizer_in_capacity: 4
    per_stream:
      frames_in_capacity: 2
      person_detections_in_capacity: 4
      faces_in_capacity: 4
      recognitions_in_capacity: 4
      identities_in_capacity: 4
      encoder_in_capacity: 2
  anonymizer:
    model_instances: 1
    method: "pixelate"
    pixelation_divisor: 10
    blur_kernel: 31
    face_only_when_available: false
modules:
  person_detector:
${detector_block}
  tracker:
    type: "bytetrack"
  face_detector:
    type: "none"
    model_instances: 1
  recognizer:
    type: "noop"
    model_instances: 1
  identity:
    type: "noop"
    model_instances: 1
metrics:
  enabled: false
  enable_http: false
  enable_ui_payload: false
  log_interval_ms: 5000
streams:
  - id: "smoke0"
    type: "webcam"
    webcam:
      device: "/dev/null"
      width: 64
      height: 64
      fps: 5
      mjpg: false
    outputs:
      fps: 5
      profiles:
        inference:
          width: 64
          height: 64
          keep_aspect: true
          interp: "linear"
          format: "BGR"
        ui:
          width: 64
          height: 64
          keep_aspect: true
          interp: "linear"
          jpeg_quality: 70
YAML
}

build_model_smoke_binary() {
  local work="/tmp/veilsight-model-smoke"
  rm -rf "$work"
  mkdir -p "$work"
  cat > "$work/model_smoke.cpp" <<'CPP'
#include <common/config.hpp>
#include <person_detector/person_detector.hpp>

#include <opencv2/core.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: model_smoke <config.yaml>\n";
        return 2;
    }
    try {
        auto cfg = veilsight::load_config_yaml(argv[1]);
        auto detector = veilsight::create_person_detector(cfg.modules.person_detector);
        cv::Mat frame(128, 128, CV_8UC3, cv::Scalar(12, 34, 56));
        const auto boxes = detector->detect(frame);
        std::cout << "detections=" << boxes.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
CPP
  cat > "$work/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(veilsight_model_smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)
pkg_check_modules(GST_ALLOCATORS REQUIRED IMPORTED_TARGET gstreamer-allocators-1.0)
pkg_check_modules(GST_WEBRTC REQUIRED IMPORTED_TARGET gstreamer-webrtc-1.0)
pkg_check_modules(GST_SDP REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
pkg_check_modules(SQLITE3 REQUIRED IMPORTED_TARGET sqlite3)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(ncnn REQUIRED)

add_executable(model_smoke model_smoke.cpp)
target_include_directories(model_smoke PRIVATE /opt/veilsight/core/include /opt/veilsight/thirdparty)
target_link_libraries(model_smoke PRIVATE
  /opt/veilsight/build/core/libveilsight_core.a
  ${OpenCV_LIBS}
  PkgConfig::GST
  PkgConfig::GST_APP
  PkgConfig::GST_VIDEO
  PkgConfig::GST_ALLOCATORS
  PkgConfig::GST_WEBRTC
  PkgConfig::GST_SDP
  PkgConfig::PCRE2
  PkgConfig::SQLITE3
  yaml-cpp::yaml-cpp
  Threads::Threads
  ncnn
)
CMAKE
  cmake -S "$work" -B "$work/build" >/tmp/veilsight-model-smoke-cmake.log
  cmake --build "$work/build" -j2 >/tmp/veilsight-model-smoke-build.log
  echo "$work/build/model_smoke"
}

run_models() {
  local smoke_bin
  smoke_bin="$(build_model_smoke_binary)"
  local yolox_cfg="/tmp/veilsight-yolox-smoke.yaml"
  local uhd_cfg="/tmp/veilsight-uhd-smoke.yaml"
  write_detector_config yolox "$yolox_cfg" 19080 /tmp/veilsight-yolox-smoke.sock
  write_detector_config uhd "$uhd_cfg" 19081 /tmp/veilsight-uhd-smoke.sock
  "$smoke_bin" "$yolox_cfg"
  "$smoke_bin" "$uhd_cfg"
}

run_eval_smoke() {
  python3 scripts/run_mot20_eval.py --help >/tmp/veilsight-run-mot20-help.txt
  python3 scripts/run_chokepoint_eval.py --help >/tmp/veilsight-run-chokepoint-help.txt
  ./build/apps/eval_mot20/veilsight_eval_mot20 configs/eval_configs/eval_mot20.yaml --help || true
  ./build/apps/eval_chokepoint/veilsight_eval_chokepoint --help || true

  if [[ "${VEILSIGHT_DOCKER_REAL_EVAL_SMOKE:-0}" == "1" && -d assets/MOT20/train/MOT20-01 ]]; then
    ./build/apps/eval_mot20/veilsight_eval_mot20 \
      configs/eval_configs/eval_mot20.yaml \
      --split train \
      --sequences MOT20-01 \
      --output /opt/veilsight/results \
      --detections-only
  fi
}

run_app() {
  check_native_deps
  run_health
}

case "$MODE" in
  all)
    run_unit
    run_app
    run_models
    run_eval_smoke
    ;;
  unit)
    run_unit
    ;;
  app)
    run_app
    ;;
  models)
    run_models
    ;;
  eval-smoke)
    run_eval_smoke
    ;;
  *)
    echo "usage: docker/test.sh [all|unit|app|models|eval-smoke]" >&2
    exit 2
    ;;
esac
