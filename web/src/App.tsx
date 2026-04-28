import { Activity, FileCheck, Play, RefreshCw, Square } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type PipelineStatus, type StreamInfo } from "./api/client";
import { connectJsonWebSocket } from "./api/ws";
import { TrackOverlay } from "./components/TrackOverlay";
import { WebRTCPlayer } from "./components/WebRTCPlayer";

type Analytics = { stream?: { stream_id?: string; profile?: string }; frame_id?: number; tracks?: unknown[]; width?: number; height?: number };

function streamKey(stream: StreamInfo) {
  return `${stream.stream_id}/${stream.profile}`;
}

export function App() {
  const [health, setHealth] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<PipelineStatus>({});
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [analytics, setAnalytics] = useState<Record<string, Analytics>>({});
  const [selectedKey, setSelectedKey] = useState<string>("");
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
    const analyticsSocket = connectJsonWebSocket("/ws/analytics", (value: any) => {
      const stream = value.stream ?? {};
      setAnalytics((prev) => ({ ...prev, [`${stream.stream_id}/${stream.profile}`]: value }));
    });
    return () => {
      metricsSocket.close();
      analyticsSocket.close();
    };
  }, []);

  const selectedStream = useMemo(
    () => streams.find((stream) => streamKey(stream) === selectedKey) ?? streams[0],
    [streams, selectedKey]
  );

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
        <div className="streams-grid">
          {streams.map((stream) => {
            const key = streamKey(stream);
            return (
              <article className={`stream-tile ${key === selectedKey ? "selected" : ""}`} key={key} onClick={() => setSelectedKey(key)}>
                <div className="video-wrap">
                  <WebRTCPlayer stream={stream} />
                  <TrackOverlay analytics={analytics[key] as any} />
                </div>
                <footer>
                  <strong>{key}</strong>
                  <span>{stream.webrtc_available ? "WebRTC" : "MJPEG fallback"}</span>
                </footer>
              </article>
            );
          })}
          {streams.length === 0 && <div className="empty">No streams reported by runner</div>}
        </div>

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
            <h2>Analytics</h2>
            <p>Frame {selectedStream ? analytics[streamKey(selectedStream)]?.frame_id ?? 0 : 0}</p>
            <p>Tracks {selectedStream ? analytics[streamKey(selectedStream)]?.tracks?.length ?? 0 : 0}</p>
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
