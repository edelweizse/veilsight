export type Point = { x: number; y: number };
export type Rect = { x: number; y: number; w: number; h: number };

export type AnalyticsRule = {
  id: string;
  stream_id: string;
  profile: string;
  kind: "line" | "area";
  name: string;
  enabled: boolean;
  geometry: { points?: Point[]; [key: string]: unknown };
  settings: Record<string, unknown>;
  created_at_ms: number;
  updated_at_ms: number;
};

export type AnalyticsEvent = {
  id?: number;
  stream_id: string;
  profile: string;
  rule_id?: string | null;
  kind: string;
  track_id?: number | null;
  direction?: string | null;
  frame_id: number;
  pts_ns: number;
  ts_ms: number;
  position: Record<string, unknown>;
  payload: Record<string, unknown>;
};

export type AnalyticsSnapshot = {
  stream: { stream_id: string; profile: string };
  frame_id: number;
  timestamp_ms: number;
  width: number;
  height: number;
  counts: { active_tracks: number; unique_tracks: number };
  tracks: Array<{ id: number; bbox: Rect; foot: Point; dwell_s: number; velocity_px_s: number }>;
  heatmap: { rows: number; cols: number; values: number[]; max_value: number };
  density: { rows: number; cols: number; values: number[]; max_value: number };
  directions: Array<{ track_id: number; from: Point; to: Point; speed_px_s: number }>;
  routes: Array<{ id: string; points: Point[]; support: number; last_seen_ms: number }>;
  rules: AnalyticsRule[];
  recent_events: AnalyticsEvent[];
};

export type OverlayLayers = {
  tracks: boolean;
  heatmap: boolean;
  density: boolean;
  directions: boolean;
  routes: boolean;
  rules: boolean;
  events: boolean;
};

export type DrawMode = "select" | "line" | "area";
