import streamlit as st
import requests
import math
import time
from datetime import datetime

st.set_page_config(layout="wide")
st.title("SecureSurveillance")

# ----------------------------
# Helpers
# ----------------------------
def get_streams(base_url: str, timeout=1.0):
    r = requests.get(f"{base_url}/streams", timeout=timeout)
    r.raise_for_status()
    return r.json()  # ["file0_0/main", "file0_1/main"]

def split_stream_id(s: str):
    if "/" not in s:
        return s, "main"
    src, channel = s.split("/", 1)
    return src, channel

def compose_urls(base: str, stream_id: str):
    src, channel = split_stream_id(stream_id)
    base = base.rstrip("/")
    video = f"{base}/video/{src}/{channel}"
    meta  = f"{base}/meta/{src}/{channel}"
    return video, meta

def safe_get_json(url: str, timeout: float):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_metrics(base_url: str, timeout=1.0):
    return safe_get_json(f"{base_url.rstrip('/')}/metrics", timeout)

def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Backend")

base_url = st.sidebar.text_input("Base URL", value="http://localhost:8080")

refresh_ms = st.sidebar.slider("Refresh interval (ms)", 200, 2000, 500, step=100)
timeout_s  = st.sidebar.slider("HTTP timeout (s)", 0.2, 3.0, 0.7, step=0.1)
cols       = st.sidebar.slider("Grid columns", 1, 4, 2)

show_meta  = st.sidebar.checkbox("Show metadata", True)
show_metrics = st.sidebar.checkbox("Show metrics dashboard", True)
show_links = st.sidebar.checkbox("Show stream links", False)
run        = st.sidebar.toggle("Run", value=True)

# ----------------------------
# Load streams from backend
# ----------------------------
try:
    streams = get_streams(base_url, timeout=timeout_s)
except Exception as e:
    st.error(f"Failed to fetch /streams: {e}")
    st.stop()

if not streams:
    st.warning("No streams available.")
    st.stop()

# ----------------------------
# Stream selector
# ----------------------------
selected = st.multiselect(
    "Select streams",
    options=streams,
    default=streams[: min(2, len(streams))],
)

if not selected:
    st.info("Select at least one stream.")
    st.stop()

# ----------------------------
# Session state: last-known meta per stream
# ----------------------------
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}      # stream_id -> dict
if "last_meta_ts" not in st.session_state:
    st.session_state.last_meta_ts = {}   # stream_id -> float (time.time())
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

if show_metrics:
    st.subheader("Metrics Dashboard")
    try:
        metrics = get_metrics(base_url, timeout=timeout_s)
        global_stats = metrics.get("global", {})
        streams_stats = metrics.get("streams", {})
        queues = metrics.get("queues", {})

        def stage_val(stage: str, key: str, default=0.0):
            payload = global_stats.get(stage, {})
            v = payload.get(key, default)
            return v if isinstance(v, (int, float)) else default

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detector FPS", f"{stage_val('detector', 'fps'):.2f}", f"p95 {stage_val('detector', 'p95_ms'):.1f} ms")
        c2.metric("Tracker FPS", f"{stage_val('tracker', 'fps'):.2f}", f"p95 {stage_val('tracker', 'p95_ms'):.1f} ms")
        c3.metric("Anonymizer FPS", f"{stage_val('anonymizer', 'fps'):.2f}", f"p95 {stage_val('anonymizer', 'p95_ms'):.1f} ms")
        c4.metric("Encoder FPS", f"{stage_val('encoder', 'fps'):.2f}", f"p95 {stage_val('end_to_end', 'p95_ms'):.1f} ms e2e")

        global_rows = []
        for stage, payload in global_stats.items():
            global_rows.append(
                {
                    "stage": stage,
                    "fps": round(payload.get("fps", 0.0), 3),
                    "avg_ms": round(payload.get("avg_ms", 0.0), 3),
                    "p50_ms": round(payload.get("p50_ms", 0.0), 3),
                    "p95_ms": round(payload.get("p95_ms", 0.0), 3),
                    "p99_ms": round(payload.get("p99_ms", 0.0), 3),
                    "count": int(payload.get("count", 0)),
                    "errors": int(payload.get("errors", 0)),
                }
            )
        st.markdown("**Global Stages**")
        st.dataframe(global_rows, use_container_width=True, hide_index=True)

        stream_rows = []
        for stream_id, stage_map in streams_stats.items():
            for stage_name in ["tracker", "recognizer", "anonymizer", "encoder", "end_to_end"]:
                payload = stage_map.get(stage_name)
                if not isinstance(payload, dict):
                    continue
                stream_rows.append(
                    {
                        "stream": stream_id,
                        "stage": stage_name,
                        "fps": round(payload.get("fps", 0.0), 3),
                        "avg_ms": round(payload.get("avg_ms", 0.0), 3),
                        "p95_ms": round(payload.get("p95_ms", 0.0), 3),
                        "count": int(payload.get("count", 0)),
                        "errors": int(payload.get("errors", 0)),
                    }
                )

        st.markdown("**Per Stream**")
        if stream_rows:
            st.dataframe(stream_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No per-stream metrics yet.")

        queue_rows = []
        for name, payload in queues.items():
            queue_rows.append(
                {
                    "queue": name,
                    "size": int(payload.get("size", 0)),
                    "capacity": int(payload.get("capacity", 0)),
                    "dropped": int(payload.get("dropped", 0)),
                }
            )
        with st.expander("Queue Health", expanded=False):
            st.dataframe(queue_rows, use_container_width=True, hide_index=True)

        history = st.session_state.metrics_history
        history.append(
            {
                "detector_fps": stage_val("detector", "fps"),
                "tracker_fps": stage_val("tracker", "fps"),
                "anonymizer_fps": stage_val("anonymizer", "fps"),
                "encoder_fps": stage_val("encoder", "fps"),
            }
        )
        if len(history) > 120:
            st.session_state.metrics_history = history[-120:]
            history = st.session_state.metrics_history

        chart_data = {
            "detector_fps": [x["detector_fps"] for x in history],
            "tracker_fps": [x["tracker_fps"] for x in history],
            "anonymizer_fps": [x["anonymizer_fps"] for x in history],
            "encoder_fps": [x["encoder_fps"] for x in history],
        }
        st.markdown("**FPS Trends**")
        st.line_chart(chart_data)

        st.caption(
            f"metrics updated {now_str()} | uptime {metrics.get('uptime_s', 0):.1f}s"
        )
    except Exception as e:
        st.warning(f"Failed to fetch /metrics: {e}")

# ----------------------------
# Render grid
# ----------------------------
n = len(selected)
rows = math.ceil(n / cols)

for r in range(rows):
    row_streams = selected[r * cols : (r + 1) * cols]
    grid = st.columns(cols)

    for c, stream_id in enumerate(row_streams):
        with grid[c]:
            video_url, meta_url = compose_urls(base_url, stream_id)

            st.subheader(stream_id)

            if show_links:
                st.code(video_url)
                st.code(meta_url)

            # Video (MJPEG)
            st.markdown(
                f'<img src="{video_url}" style="width:100%; height:auto; border-radius:10px;" />',
                unsafe_allow_html=True,
            )

            # --- REAL-TIME META (updates in-place) ---
            if show_meta:
                meta_box = st.empty()   # placeholder that updates smoothly
                status   = st.empty()   # small status line

                try:
                    meta = safe_get_json(meta_url, timeout_s)
                    st.session_state.last_meta[stream_id] = meta
                    st.session_state.last_meta_ts[stream_id] = time.time()

                    status.caption(f"meta: updated {now_str()}")
                    meta_box.json(meta)

                except Exception as e:
                    # Keep last known meta instead of flickering / disappearing
                    last = st.session_state.last_meta.get(stream_id)
                    ts   = st.session_state.last_meta_ts.get(stream_id)

                    age = ""
                    if ts is not None:
                        age = f" (last OK {time.time() - ts:.1f}s ago)"

                    status.caption(f"meta: error{age} — {type(e).__name__}: {e}")

                    if last is not None:
                        meta_box.json(last)
                    else:
                        meta_box.warning("No metadata received yet.")

# ----------------------------
# Auto refresh (after rendering)
# ----------------------------
if run:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
