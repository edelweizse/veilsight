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

export type ConfigFileInfo = {
  path: string;
  name: string;
  active: boolean;
};

export type ConfigListResponse = {
  active_path: string;
  files: ConfigFileInfo[];
};

export type AnalyticsRulePayload = {
  stream_id: string;
  profile: string;
  kind: "line" | "area";
  name: string;
  enabled?: boolean;
  geometry: Record<string, unknown>;
  settings?: Record<string, unknown>;
};

export type RenderLayerOptions = {
  tracks: boolean;
  faces: boolean;
  directions: boolean;
  rules: boolean;
  events: boolean;
};

export type RenderRule = {
  id: string;
  kind: "line" | "area";
  name: string;
  enabled: boolean;
  geometry: { points?: Array<{ x: number; y: number }>; [key: string]: unknown };
  settings: Record<string, unknown>;
};

export type RenderJobCreate = {
  config_path: string;
  input_path: string;
  output_path: string;
  gallery_db?: string | null;
  layers: Array<keyof RenderLayerOptions>;
  rules: RenderRule[];
  overwrite: boolean;
  timing_mode: "source" | "custom";
  render_mode?: string;
  fps?: number | null;
  source_fps?: number | null;
  no_gallery: boolean;
  preview_every_frames?: number;
};

export type RenderJobStatus = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  created_at_ms: number;
  updated_at_ms: number;
  config_path: string;
  input_path: string;
  output_path: string;
  gallery_db?: string | null;
  layers: Array<keyof RenderLayerOptions>;
  rules_yaml_path?: string | null;
  events_jsonl_path?: string | null;
  manifest_path?: string | null;
  progress_frame?: number | null;
  total_frames?: number | null;
  progress_percent?: number | null;
  preview_jpeg_path?: string | null;
  preview_updated_at_ms?: number | null;
  timing_mode: "source" | "custom";
  render_mode?: string;
  no_gallery: boolean;
  started_at_ms?: number | null;
  finished_at_ms?: number | null;
  returncode?: number | null;
  stderr_tail: string;
  message: string;
};

export type RenderSettingsResponse = {
  binary_path: string;
  default_gallery_db: string;
};

export type GalleryIdentity = {
  identity_key: string;
  display_name?: string | null;
  active: boolean;
  embedding_count: number;
  created_at_ms?: number | null;
  updated_at_ms?: number | null;
};

export type GalleryEmbedding = {
  id: number;
  identity_key: string;
  model: string;
  dim: number;
  active: boolean;
  source_type?: string | null;
  source_ref?: string | null;
  quality?: Record<string, unknown> | null;
  face_bbox?: Record<string, unknown> | null;
  created_at_ms?: number | null;
};

export type EnrollmentCandidate = {
  candidate_id: string;
  image_id: string;
  bbox: { x: number; y: number; w: number; h: number };
  landmarks: Array<{ x: number; y: number }>;
  score: number;
  image_width: number;
  image_height: number;
  usable: boolean;
  reject_reasons: string[];
};

export type EnrollmentCandidateResponse = {
  enrollment_id: string;
  expires_at_ms: number;
  image_id: string;
  candidates: EnrollmentCandidate[];
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

async function requestBlob(url: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.blob();
}

export function renderJobPreviewUrl(job_id: string, updated?: number | null): string {
  return `/api/renders/jobs/${encodeURIComponent(job_id)}/preview.jpg?ts=${updated ?? Date.now()}`;
}

export const api = {
  health: () => request<Record<string, unknown>>("/api/health"),
  status: () => request<PipelineStatus>("/api/pipeline/status"),
  streams: () => request<StreamsResponse>("/api/streams"),
  metrics: () => request<Record<string, unknown>>("/api/metrics"),
  analyticsLatest: () => request<Record<string, unknown>>("/api/analytics/latest"),
  analyticsSnapshot: (stream_id: string, profile = "ui") =>
    request<Record<string, unknown>>(`/api/analytics/snapshot?stream_id=${encodeURIComponent(stream_id)}&profile=${encodeURIComponent(profile)}`),
  createAnalyticsRule: (payload: AnalyticsRulePayload) =>
    request<Record<string, unknown>>("/api/analytics/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  updateAnalyticsRule: (rule_id: string, payload: Partial<AnalyticsRulePayload>) =>
    request<Record<string, unknown>>(`/api/analytics/rules/${encodeURIComponent(rule_id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  deleteAnalyticsRule: (rule_id: string) =>
    request<Record<string, unknown>>(`/api/analytics/rules/${encodeURIComponent(rule_id)}`, {
      method: "DELETE"
    }),
  config: () => request<ConfigResponse>("/api/config"),
  configs: () => request<ConfigListResponse>("/api/configs"),
  selectConfig: (path: string) =>
    request<ConfigResponse>("/api/config/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    }),
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
  ,
  galleryIdentities: () => request<GalleryIdentity[]>("/api/gallery/identities"),
  createGalleryIdentity: (payload: { identity_key?: string; display_name: string; active?: boolean }) =>
    request<GalleryIdentity>("/api/gallery/identities", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  updateGalleryIdentity: (identity_key: string, payload: { display_name?: string; active?: boolean }) =>
    request<GalleryIdentity>(`/api/gallery/identities/${encodeURIComponent(identity_key)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  deleteGalleryIdentity: (identity_key: string) =>
    request<Record<string, unknown>>(`/api/gallery/identities/${encodeURIComponent(identity_key)}`, { method: "DELETE" }),
  galleryEmbeddings: (identity_key: string) =>
    request<GalleryEmbedding[]>(`/api/gallery/identities/${encodeURIComponent(identity_key)}/embeddings`),
  deleteGalleryEmbedding: (embedding_id: number) =>
    request<GalleryEmbedding>(`/api/gallery/embeddings/${embedding_id}`, { method: "DELETE" }),
  enrollmentCandidates: (file: Blob, filename = "capture.jpg") => {
    const form = new FormData();
    form.append("file", file, filename);
    return request<EnrollmentCandidateResponse>("/api/gallery/enrollment-candidates", { method: "POST", body: form });
  },
  enrollmentCandidatesFromStream: (stream_id: string, profile = "ui") =>
    request<EnrollmentCandidateResponse>("/api/gallery/enrollment-candidates/from-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_id, profile })
    }),
  commitGalleryEmbeddings: (identity_key: string, enrollment_id: string, candidate_ids: string[]) =>
    request<GalleryEmbedding[]>(`/api/gallery/identities/${encodeURIComponent(identity_key)}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enrollment_id, candidate_ids })
    }),
  refreshGallery: () => request<Record<string, unknown>>("/api/gallery/refresh", { method: "POST" })
  ,
  renderSettings: () => request<RenderSettingsResponse>("/api/renders/settings"),
  renderPreview: (input_path: string) =>
    requestBlob("/api/renders/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path })
    }),
  createRenderJob: (payload: RenderJobCreate) =>
    request<RenderJobStatus>("/api/renders/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  renderJob: (job_id: string) => request<RenderJobStatus>(`/api/renders/jobs/${encodeURIComponent(job_id)}`),
  cancelRenderJob: (job_id: string) =>
    request<RenderJobStatus>(`/api/renders/jobs/${encodeURIComponent(job_id)}/cancel`, { method: "POST" })
};
