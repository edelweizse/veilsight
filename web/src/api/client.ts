export type PipelineStatus = {
  runner_id?: string;
  running?: boolean;
  state?: string;
  message?: string;
  timestamp_ms?: number;
};

export type StreamInfo = {
  stream_id: string;
  profile: string;
  webrtc_available: boolean;
  webrtc_offer_url: string;
  mjpeg_url: string;
  snapshot_url: string;
  metadata_url: string;
};

export type StreamsResponse = {
  streams: StreamInfo[];
  public_base_url: string;
};

export type ConfigResponse = {
  path: string;
  yaml_text: string;
  config?: Record<string, unknown>;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Record<string, unknown>>("/api/health"),
  status: () => request<PipelineStatus>("/api/pipeline/status"),
  streams: () => request<StreamsResponse>("/api/streams"),
  metrics: () => request<Record<string, unknown>>("/api/metrics"),
  analyticsLatest: () => request<Record<string, unknown>>("/api/analytics/latest"),
  config: () => request<ConfigResponse>("/api/config"),
  validateConfig: (yaml_text: string) =>
    request<Record<string, unknown>>("/api/config/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text })
    }),
  saveConfig: (yaml_text: string) =>
    request<Record<string, unknown>>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text })
    }),
  start: () => request<Record<string, unknown>>("/api/pipeline/start", { method: "POST" }),
  stop: () => request<Record<string, unknown>>("/api/pipeline/stop", { method: "POST" }),
  reload: () => request<Record<string, unknown>>("/api/pipeline/reload", { method: "POST" })
};
