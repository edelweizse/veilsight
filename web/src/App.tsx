import {
  Activity,
  AreaChart,
  FileCheck,
  GitCommitHorizontal,
  Layers,
  MousePointer2,
  Play,
  RefreshCw,
  Route,
  Square,
  TrendingUp
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { AnalyticsSnapshot, DrawMode, OverlayLayers, Point } from "./analytics/types";
import { api, type PipelineStatus, type StreamInfo } from "./api/client";
import { connectJsonWebSocket } from "./api/ws";
import { AnalyticsOverlay } from "./components/AnalyticsOverlay";
import { WebRTCPlayer } from "./components/WebRTCPlayer";

function streamKey(stream: Pick<StreamInfo, "stream_id" | "profile">) {
  return `${stream.stream_id}/${stream.profile}`;
}

const defaultLayers: OverlayLayers = {
  tracks: true,
  heatmap: false,
  density: false,
  directions: true,
  routes: false,
  rules: true,
  events: true
};

function LayerToggles({ layers, onChange }: { layers: OverlayLayers; onChange: (layers: OverlayLayers) => void }) {
  const items: Array<[keyof OverlayLayers, string]> = [
    ["tracks", "Tracks"],
    ["heatmap", "Heatmap"],
    ["density", "Density"],
    ["directions", "Directions"],
    ["routes", "Routes"],
    ["rules", "Rules"],
    ["events", "Events"]
  ];
  return (
    <div className="toggle-grid" aria-label="Overlay layers">
      {items.map(([key, label]) => (
        <label key={key}>
          <input type="checkbox" checked={layers[key]} onChange={(event) => onChange({ ...layers, [key]: event.target.checked })} />
          {label}
        </label>
      ))}
    </div>
  );
}

function DrawTools({ mode, onChange }: { mode: DrawMode; onChange: (mode: DrawMode) => void }) {
  return (
    <div className="tool-row" aria-label="Rule drawing tools">
      <button className={mode === "select" ? "active" : ""} onClick={() => onChange("select")} title="Select">
        <MousePointer2 size={16} />Select
      </button>
      <button className={mode === "line" ? "active" : ""} onClick={() => onChange("line")} title="Line rule">
        <GitCommitHorizontal size={16} />Line
      </button>
      <button className={mode === "area" ? "active" : ""} onClick={() => onChange("area")} title="Area rule">
        <AreaChart size={16} />Area
      </button>
    </div>
  );
}

function StreamVideo({
  stream,
  snapshot,
  layers,
  drawMode,
  onCreateRule,
  onUpdateRule
}: {
  stream: StreamInfo;
  snapshot?: AnalyticsSnapshot;
  layers: OverlayLayers;
  drawMode: DrawMode;
  onCreateRule: (kind: "line" | "area", points: Point[]) => Promise<void>;
  onUpdateRule: (ruleId: string, points: Point[]) => Promise<void>;
}) {
  return (
    <div className="video-wrap">
      <WebRTCPlayer stream={stream} />
      <AnalyticsOverlay
        snapshot={snapshot}
        layers={layers}
        drawMode={drawMode}
        onCreateRule={onCreateRule}
        onUpdateRule={(rule, points) => onUpdateRule(rule.id, points)}
      />
    </div>
  );
}

function StreamWorkspace({
  streams,
  selectedKey,
  snapshots,
  layers,
  drawMode,
  onSelect,
  onCreateRule,
  onUpdateRule
}: {
  streams: StreamInfo[];
  selectedKey: string;
  snapshots: Record<string, AnalyticsSnapshot>;
  layers: OverlayLayers;
  drawMode: DrawMode;
  onSelect: (key: string) => void;
  onCreateRule: (kind: "line" | "area", points: Point[]) => Promise<void>;
  onUpdateRule: (ruleId: string, points: Point[]) => Promise<void>;
}) {
  const selectedStream = streams.find((stream) => streamKey(stream) === selectedKey) ?? streams[0];
  if (!selectedStream) return <div className="empty">No streams reported by runner</div>;
  const selected = streamKey(selectedStream);
  const otherStreams = streams.filter((stream) => streamKey(stream) !== selected);

  return (
    <section className="focus-workspace">
      <article className="focused-stream">
        <header>
          <strong>{selected}</strong>
          <span>{selectedStream.webrtc_available ? "WebRTC" : "MJPEG fallback"}</span>
        </header>
        <StreamVideo
          stream={selectedStream}
          snapshot={snapshots[selected]}
          layers={layers}
          drawMode={drawMode}
          onCreateRule={onCreateRule}
          onUpdateRule={onUpdateRule}
        />
      </article>
      <div className="stream-strip">
        {otherStreams.map((stream) => {
          const key = streamKey(stream);
          return (
            <button className="stream-thumb" key={key} onClick={() => onSelect(key)}>
              <div className="thumb-media">
                <WebRTCPlayer stream={stream} />
              </div>
              <span>{key}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export function App() {
  const [health, setHealth] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<PipelineStatus>({});
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [snapshots, setSnapshots] = useState<Record<string, AnalyticsSnapshot>>({});
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [layers, setLayers] = useState<OverlayLayers>(defaultLayers);
  const [drawMode, setDrawMode] = useState<DrawMode>("select");
  const [configPath, setConfigPath] = useState("");
  const [yamlText, setYamlText] = useState("");
  const [configMessage, setConfigMessage] = useState("");

  async function refresh() {
    const [healthValue, statusValue, streamsValue, metricsValue, configValue] = await Promise.all([
      api.health().catch((error) => ({ ok: false, error: String(error) })),
      api.status().catch((error) => ({ state: "disconnected", message: String(error) })),
      api.streams().catch(() => ({ streams: [], public_base_url: "" })),
      api.metrics().catch(() => ({})),
      api.config().catch(() => ({ path: "", yaml_text: "" }))
    ]);
    setHealth(healthValue);
    setStatus(statusValue);
    setStreams(streamsValue.streams);
    setMetrics(metricsValue);
    setConfigPath(configValue.path);
    setYamlText(configValue.yaml_text);
    setSelectedKey((current) => current || (streamsValue.streams[0] ? streamKey(streamsValue.streams[0]) : ""));
  }

  useEffect(() => {
    void refresh();
    const metricsSocket = connectJsonWebSocket("/ws/metrics", (value: any) => {
      if (value.metrics) setMetrics(value.metrics);
      if (value.status) setStatus(value.status);
    });
    const overlaySocket = connectJsonWebSocket("/ws/analytics/overlays", (value: any) => {
      const stream = value.stream ?? {};
      if (stream.stream_id) {
        setSnapshots((prev) => ({ ...prev, [`${stream.stream_id}/${stream.profile ?? "ui"}`]: value as AnalyticsSnapshot }));
      }
    });
    return () => {
      metricsSocket.close();
      overlaySocket.close();
    };
  }, []);

  const selectedStream = useMemo(
    () => streams.find((stream) => streamKey(stream) === selectedKey) ?? streams[0],
    [streams, selectedKey]
  );
  const selectedSnapshot = selectedStream ? snapshots[streamKey(selectedStream)] : undefined;

  async function runCommand(command: () => Promise<Record<string, unknown>>) {
    const result = await command();
    if ((result as any).status) setStatus((result as any).status);
    await refresh();
  }

  async function validateConfig() {
    const result = await api.validateConfig(yamlText);
    setConfigMessage(JSON.stringify(result, null, 2));
  }

  async function saveConfig() {
    const result = await api.saveConfig(yamlText);
    setConfigMessage(JSON.stringify(result, null, 2));
  }

  async function createRule(kind: "line" | "area", points: Point[]) {
    if (!selectedStream) return;
    const name = `${kind === "line" ? "Line" : "Area"} ${new Date().toLocaleTimeString()}`;
    const rule = await api.createAnalyticsRule({
      stream_id: selectedStream.stream_id,
      profile: selectedStream.profile,
      kind,
      name,
      enabled: true,
      geometry: { points },
      settings: kind === "line" ? { min_gap_ms: 1000 } : {}
    });
    setSnapshots((prev) => {
      const key = streamKey(selectedStream);
      const snapshot = prev[key];
      if (!snapshot) return prev;
      return { ...prev, [key]: { ...snapshot, rules: [...snapshot.rules.filter((item) => item.id !== (rule as any).id), rule as any] } };
    });
    setDrawMode("select");
  }

  async function updateRule(ruleId: string, points: Point[]) {
    const rule = await api.updateAnalyticsRule(ruleId, { geometry: { points } });
    if (!selectedStream) return;
    setSnapshots((prev) => {
      const key = streamKey(selectedStream);
      const snapshot = prev[key];
      if (!snapshot) return prev;
      return { ...prev, [key]: { ...snapshot, rules: snapshot.rules.map((item) => (item.id === ruleId ? (rule as any) : item)) } };
    });
  }

  const globalStages = Array.isArray(metrics.global) ? metrics.global : [];
  const queues = Array.isArray(metrics.queues) ? metrics.queues : [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Veilsight</h1>
          <span>{configPath || "No active config"}</span>
        </div>
        <div className="status-cluster">
          <span className="pill">Runner: {String((health as any).connection_state ?? "unknown")}</span>
          <span className="pill">Pipeline: {status.state ?? "unknown"}</span>
          <button onClick={() => runCommand(api.start)} title="Start"><Play size={16} />Start</button>
          <button onClick={() => runCommand(api.stop)} title="Stop"><Square size={16} />Stop</button>
          <button onClick={() => runCommand(api.reload)} title="Reload"><RefreshCw size={16} />Reload</button>
        </div>
      </header>

      <section className="workspace">
        <StreamWorkspace
          streams={streams}
          selectedKey={selectedKey}
          snapshots={snapshots}
          layers={layers}
          drawMode={drawMode}
          onSelect={setSelectedKey}
          onCreateRule={createRule}
          onUpdateRule={updateRule}
        />

        <aside className="side-panel">
          <section>
            <h2><Activity size={16} />Metrics</h2>
            <div className="metric-grid">
              {["detector", "tracker", "anonymizer", "encoder"].map((stage) => {
                const row = globalStages.find((item: any) => item.stage === stage) ?? {};
                return <div className="metric" key={stage}><span>{stage}</span><strong>{Number(row.fps ?? 0).toFixed(1)} fps</strong><small>p95 {Number(row.p95_ms ?? 0).toFixed(1)} ms</small></div>;
              })}
            </div>
            <table>
              <thead><tr><th>Queue</th><th>Size</th><th>Drop</th></tr></thead>
              <tbody>{queues.map((q: any) => <tr key={q.name}><td>{q.name}</td><td>{q.size}/{q.capacity}</td><td>{q.dropped}</td></tr>)}</tbody>
            </table>
          </section>

          <section>
            <h2><Layers size={16} />Layers</h2>
            <LayerToggles layers={layers} onChange={setLayers} />
            <DrawTools mode={drawMode} onChange={setDrawMode} />
          </section>

          <section>
            <h2><TrendingUp size={16} />Analytics</h2>
            <div className="stat-row"><span>Frame</span><strong>{selectedSnapshot?.frame_id ?? 0}</strong></div>
            <div className="stat-row"><span>Active</span><strong>{selectedSnapshot?.counts.active_tracks ?? 0}</strong></div>
            <div className="stat-row"><span>Unique</span><strong>{selectedSnapshot?.counts.unique_tracks ?? 0}</strong></div>
            <div className="stat-row"><span>Rules</span><strong>{selectedSnapshot?.rules.length ?? 0}</strong></div>
            <div className="event-list">
              {(selectedSnapshot?.recent_events ?? []).slice(0, 6).map((event) => (
                <div key={`${event.id ?? event.ts_ms}-${event.kind}-${event.track_id}`}>
                  <span>{event.kind}</span>
                  <small>track {event.track_id ?? "-"}</small>
                </div>
              ))}
            </div>
            <div className="routes-note"><Route size={14} />{selectedSnapshot?.routes.length ?? 0} learned routes</div>
          </section>
        </aside>
      </section>

      <section className="config-editor">
        <h2><FileCheck size={16} />Config</h2>
        <textarea value={yamlText} onChange={(event) => setYamlText(event.target.value)} spellCheck={false} />
        <div className="editor-actions">
          <button onClick={validateConfig}>Validate</button>
          <button onClick={saveConfig}>Save</button>
          <button onClick={() => runCommand(api.reload)}>Reload pipeline</button>
        </div>
        {configMessage && <pre>{configMessage}</pre>}
      </section>
    </main>
  );
}
