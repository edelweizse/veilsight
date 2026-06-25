import {
  Activity,
  AreaChart,
  Camera,
  Check,
  CircleX,
  Download,
  FileCheck,
  Film,
  FolderOpen,
  GitCommitHorizontal,
  Image,
  Layers,
  MousePointer2,
  Pencil,
  Play,
  RefreshCw,
  Route,
  Square,
  Trash2,
  TrendingUp,
  Upload,
  UserPlus,
  Users
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalyticsRule, AnalyticsSnapshot, DrawMode, OverlayLayers, Point } from "./analytics/types";
import {
  api,
  renderJobPreviewUrl,
  type ConfigFileInfo,
  type EnrollmentCandidateResponse,
  type GalleryEmbedding,
  type GalleryIdentity,
  type PipelineStatus,
  type RenderJobStatus,
  type RenderLayerOptions,
  type RenderRule,
  type StreamInfo
} from "./api/client";
import { connectJsonWebSocket } from "./api/ws";
import { AnalyticsOverlay } from "./components/AnalyticsOverlay";
import { WebRTCPlayer, type StreamProtocol } from "./components/WebRTCPlayer";

function streamKey(stream: Pick<StreamInfo, "stream_id" | "profile">) {
  return `${stream.stream_id}/${stream.profile}`;
}

function queueRows(raw: unknown): Array<{ name: string; size?: number; capacity?: number; dropped?: number; producer?: string; consumer?: string; description?: string }> {
  if (Array.isArray(raw)) return raw as Array<{ name: string; size?: number; capacity?: number; dropped?: number; producer?: string; consumer?: string; description?: string }>;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, any>).map(([name, value]) => ({ name, ...(value ?? {}) }));
}

function stageRows(raw: unknown): Array<Record<string, any>> {
  if (Array.isArray(raw)) return raw as Array<Record<string, any>>;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, any>).map(([stage, value]) => ({ stage, ...(value ?? {}) }));
}

const defaultLayers: OverlayLayers = {
  tracks: true,
  faces: true,
  heatmap: false,
  density: false,
  directions: true,
  routes: false,
  rules: true,
  events: true
};

const defaultRenderLayers: RenderLayerOptions = {
  tracks: false,
  faces: false,
  directions: false,
  rules: false,
  events: false
};

function selectedRenderLayers(layers: RenderLayerOptions): Array<keyof RenderLayerOptions> {
  return (Object.keys(layers) as Array<keyof RenderLayerOptions>).filter((key) => layers[key]);
}

function LayerToggles({ layers, onChange }: { layers: OverlayLayers; onChange: (layers: OverlayLayers) => void }) {
  const items: Array<[keyof OverlayLayers, string]> = [
    ["tracks", "Tracks"],
    ["faces", "Faces"],
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

function DrawTools({ mode, onChange, disabled = false }: { mode: DrawMode; onChange: (mode: DrawMode) => void; disabled?: boolean }) {
  return (
    <div className="tool-row" aria-label="Rule drawing tools">
      <button className={mode === "select" ? "active" : ""} onClick={() => onChange("select")} title="Select" disabled={disabled}>
        <MousePointer2 size={16} />Select
      </button>
      <button className={mode === "line" ? "active" : ""} onClick={() => onChange("line")} title="Line rule" disabled={disabled}>
        <GitCommitHorizontal size={16} />Line
      </button>
      <button className={mode === "area" ? "active" : ""} onClick={() => onChange("area")} title="Area rule" disabled={disabled}>
        <AreaChart size={16} />Area
      </button>
    </div>
  );
}

function RuleManager({
  rules,
  onRename,
  onDelete
}: {
  rules: AnalyticsRule[];
  onRename: (ruleId: string, name: string) => Promise<void>;
  onDelete: (ruleId: string) => Promise<void>;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [pendingId, setPendingId] = useState<string | null>(null);

  function beginRename(rule: AnalyticsRule) {
    setEditingId(rule.id);
    setDraftName(rule.name);
  }

  async function saveRename(rule: AnalyticsRule) {
    const nextName = draftName.trim();
    if (!nextName || nextName === rule.name) {
      setEditingId(null);
      return;
    }
    setPendingId(rule.id);
    try {
      await onRename(rule.id, nextName);
      setEditingId(null);
    } finally {
      setPendingId(null);
    }
  }

  async function removeRule(rule: AnalyticsRule) {
    setPendingId(rule.id);
    try {
      await onDelete(rule.id);
      if (editingId === rule.id) setEditingId(null);
    } finally {
      setPendingId(null);
    }
  }

  return (
    <div className="rule-list" aria-label="Drawn analytical events">
      {rules.length === 0 ? (
        <div className="empty-inline">No drawn analytical events</div>
      ) : (
        rules.map((rule) => {
          const isEditing = editingId === rule.id;
          const pending = pendingId === rule.id;
          return (
            <div className="rule-row" key={rule.id}>
              <div>
                {isEditing ? (
                  <input
                    value={draftName}
                    onChange={(event) => setDraftName(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") void saveRename(rule);
                      if (event.key === "Escape") setEditingId(null);
                    }}
                    aria-label={`Rename ${rule.name}`}
                    autoFocus
                  />
                ) : (
                  <strong>{rule.name}</strong>
                )}
                <small>{rule.kind}</small>
              </div>
              <div className="rule-actions">
                {isEditing ? (
                  <>
                    <button onClick={() => void saveRename(rule)} disabled={pending} title="Save name" aria-label={`Save ${rule.name}`}>
                      <Check size={15} />
                    </button>
                    <button onClick={() => setEditingId(null)} disabled={pending} title="Cancel rename" aria-label={`Cancel ${rule.name}`}>
                      <CircleX size={15} />
                    </button>
                  </>
                ) : (
                  <button onClick={() => beginRename(rule)} disabled={pending} title="Rename event" aria-label={`Rename ${rule.name}`}>
                    <Pencil size={15} />
                  </button>
                )}
                <button onClick={() => void removeRule(rule)} disabled={pending} title="Delete event" aria-label={`Delete ${rule.name}`}>
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

function StreamVideo({
  stream,
  snapshot,
  layers,
  drawMode,
  onCreateRule,
  onUpdateRule,
  onProtocolChange
}: {
  stream: StreamInfo;
  snapshot?: AnalyticsSnapshot;
  layers: OverlayLayers;
  drawMode: DrawMode;
  onCreateRule: (kind: "line" | "area", points: Point[]) => Promise<void>;
  onUpdateRule: (ruleId: string, points: Point[]) => Promise<void>;
  onProtocolChange?: (protocol: StreamProtocol) => void;
}) {
  return (
    <div className="video-wrap">
      <WebRTCPlayer stream={stream} onProtocolChange={onProtocolChange} />
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
  onUpdateRule,
  protocols,
  onProtocolChange
}: {
  streams: StreamInfo[];
  selectedKey: string;
  snapshots: Record<string, AnalyticsSnapshot>;
  layers: OverlayLayers;
  drawMode: DrawMode;
  onSelect: (key: string) => void;
  onCreateRule: (kind: "line" | "area", points: Point[]) => Promise<void>;
  onUpdateRule: (ruleId: string, points: Point[]) => Promise<void>;
  protocols: Record<string, StreamProtocol>;
  onProtocolChange: (key: string, protocol: StreamProtocol) => void;
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
          <span>Protocol: {protocols[selected] ?? (selectedStream.webrtc_available ? "WebRTC" : "MJPEG")}</span>
        </header>
        <StreamVideo
          stream={selectedStream}
          snapshot={snapshots[selected]}
          layers={layers}
          drawMode={drawMode}
          onCreateRule={onCreateRule}
          onUpdateRule={onUpdateRule}
          onProtocolChange={(protocol) => onProtocolChange(selected, protocol)}
        />
      </article>
      <div className="stream-strip">
        {otherStreams.map((stream) => {
          const key = streamKey(stream);
          return (
            <button className="stream-thumb" key={key} onClick={() => onSelect(key)}>
              <div className="thumb-media">
                <WebRTCPlayer stream={stream} preferWebRTC={false} />
              </div>
              <span>{key}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

async function captureWebcamBlob(): Promise<Blob> {
  const media = await navigator.mediaDevices.getUserMedia({ video: true });
  try {
    const video = document.createElement("video");
    video.srcObject = media;
    video.muted = true;
    await video.play().catch(() => undefined);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve) => {
      canvas.toBlob((blob) => resolve(blob ?? new Blob([], { type: "image/jpeg" })), "image/jpeg", 0.9);
    });
  } finally {
    media.getTracks().forEach((track) => track.stop());
  }
}

function GalleryView({ streams }: { streams: StreamInfo[] }) {
  const [identities, setIdentities] = useState<GalleryIdentity[]>([]);
  const [selectedKey, setSelectedKey] = useState("");
  const [embeddings, setEmbeddings] = useState<GalleryEmbedding[]>([]);
  const [search, setSearch] = useState("");
  const [draftName, setDraftName] = useState("");
  const [newName, setNewName] = useState("");
  const [source, setSource] = useState<"upload" | "webcam" | "stream">("upload");
  const [enrollment, setEnrollment] = useState<EnrollmentCandidateResponse | null>(null);
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [galleryMessage, setGalleryMessage] = useState("");
  const [pending, setPending] = useState(false);
  const [streamChoice, setStreamChoice] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  async function loadIdentities(nextSelected?: string) {
    const rows = await api.galleryIdentities();
    setIdentities(rows);
    const key = nextSelected || selectedKey || rows[0]?.identity_key || "";
    setSelectedKey(key);
    const selected = rows.find((identity) => identity.identity_key === key);
    setDraftName(selected?.display_name || selected?.identity_key || "");
  }

  useEffect(() => {
    void loadIdentities();
  }, []);

  useEffect(() => {
    if (!selectedKey) {
      setEmbeddings([]);
      return;
    }
    setDeleteConfirm(false);
    void api.galleryEmbeddings(selectedKey).then(setEmbeddings).catch(() => setEmbeddings([]));
    const identity = identities.find((item) => item.identity_key === selectedKey);
    setDraftName(identity?.display_name || identity?.identity_key || "");
  }, [selectedKey, identities]);

  const filteredIdentities = identities.filter((identity) => {
    const haystack = `${identity.identity_key} ${identity.display_name ?? ""}`.toLowerCase();
    return haystack.includes(search.toLowerCase());
  });
  const selectedIdentity = identities.find((identity) => identity.identity_key === selectedKey);
  const previewUrl = enrollment ? `/api/gallery/enrollment-candidates/${enrollment.enrollment_id}/image/${enrollment.image_id}` : "";
  const selectedStream = streams.find((stream) => streamKey(stream) === streamChoice) ?? streams[0];

  async function createIdentity() {
    if (!newName.trim()) return;
    const created = await api.createGalleryIdentity({ display_name: newName.trim() });
    setNewName("");
    await loadIdentities(created.identity_key);
  }

  async function saveIdentity() {
    if (!selectedIdentity) return;
    const updated = await api.updateGalleryIdentity(selectedIdentity.identity_key, { display_name: draftName, active: selectedIdentity.active });
    setGalleryMessage(`Saved ${updated.identity_key}`);
    await loadIdentities(updated.identity_key);
  }

  async function deactivateIdentity() {
    if (!selectedIdentity) return;
    const updated = await api.updateGalleryIdentity(selectedIdentity.identity_key, { active: false });
    setGalleryMessage(`Deactivated ${updated.identity_key}`);
    await loadIdentities(updated.identity_key);
  }

  async function activateIdentity() {
    if (!selectedIdentity) return;
    const updated = await api.updateGalleryIdentity(selectedIdentity.identity_key, { active: true });
    setGalleryMessage(`Activated ${updated.identity_key}`);
    await loadIdentities(updated.identity_key);
  }

  async function permanentlyDeleteIdentity() {
    if (!selectedIdentity) return;
    await api.deleteGalleryIdentity(selectedIdentity.identity_key);
    const refresh = await api.refreshGallery().catch((error) => ({ accepted: false, message: String(error) }));
    const success = (refresh as any).accepted ? "success" : "failure";
    setGalleryMessage(`Deleted ${selectedIdentity.identity_key} permanently; refresh ${success}`);
    
    await loadIdentities();
    setSelectedKey("");
    setEmbeddings([]);
    setEnrollment(null);
    setSelectedCandidates([]);
    setDeleteConfirm(false);
  }

  async function analyzeBlob(blob: Blob, filename = "capture.jpg") {
    setPending(true);
    try {
      const result = await api.enrollmentCandidates(blob, filename);
      setEnrollment(result);
      setSelectedCandidates(result.candidates.filter((candidate) => candidate.usable).map((candidate) => candidate.candidate_id));
      setGalleryMessage(`${result.candidates.length} face candidates`);
    } finally {
      setPending(false);
    }
  }

  async function analyzeUpload(file?: File) {
    if (!file) return;
    await analyzeBlob(file, file.name);
  }

  async function analyzeWebcam() {
    const blob = await captureWebcamBlob();
    await analyzeBlob(blob, "webcam.jpg");
  }

  async function analyzeStream() {
    if (!selectedStream) return;
    setPending(true);
    try {
      const result = await api.enrollmentCandidatesFromStream(selectedStream.stream_id, selectedStream.profile);
      setEnrollment(result);
      setSelectedCandidates(result.candidates.filter((candidate) => candidate.usable).map((candidate) => candidate.candidate_id));
      setGalleryMessage(`${result.candidates.length} face candidates`);
    } finally {
      setPending(false);
    }
  }

  async function commitCandidates() {
    if (!selectedIdentity || !enrollment || selectedCandidates.length === 0) return;
    setPending(true);
    try {
      await api.commitGalleryEmbeddings(selectedIdentity.identity_key, enrollment.enrollment_id, selectedCandidates);
      const refresh = await api.refreshGallery().catch((error) => ({ accepted: false, message: String(error) }));
      
      const wasInactive = selectedIdentity.active === false;
      const count = selectedCandidates.length;
      
      if (wasInactive) {
        if ((refresh as any).accepted === false) {
           setGalleryMessage(`Committed ${count} sample(s); identity activated but refresh failed: ${(refresh as any).message}. Start or reload pipeline to load gallery.`);
        } else {
           setGalleryMessage(`Committed ${count} sample(s); identity activated; refresh ${String((refresh as any).message || (refresh as any).accepted)}`);
        }
      } else {
        if ((refresh as any).accepted === false) {
          setGalleryMessage(`Committed ${count} sample(s); samples saved but refresh failed: ${(refresh as any).message}. Start or reload pipeline to load gallery.`);
        } else {
          setGalleryMessage(`Committed ${count}; refresh ${String((refresh as any).message || (refresh as any).accepted)}`);
        }
      }
      
      setEnrollment(null);
      setSelectedCandidates([]);
      await loadIdentities(selectedIdentity.identity_key);
      setEmbeddings(await api.galleryEmbeddings(selectedIdentity.identity_key));
    } finally {
      setPending(false);
    }
  }

  async function removeEmbedding(id: number) {
    await api.deleteGalleryEmbedding(id);
    if (selectedKey) setEmbeddings(await api.galleryEmbeddings(selectedKey));
    setGalleryMessage("Sample deactivated");
  }

  return (
    <section className="gallery-view">
      <aside className="gallery-list">
        <h2><Users size={16} />Gallery</h2>
        <input aria-label="Search identities" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search identities" />
        <div className="identity-list">
          {filteredIdentities.map((identity) => (
            <button
              key={identity.identity_key}
              className={identity.identity_key === selectedKey ? "identity-row active" : "identity-row"}
              onClick={() => setSelectedKey(identity.identity_key)}
            >
              <span>
                <strong>{identity.display_name || identity.identity_key}</strong>
                <small>{identity.identity_key}</small>
              </span>
              <small>{identity.active ? "active" : "inactive"} · {identity.embedding_count}</small>
            </button>
          ))}
        </div>
        <div className="new-identity">
          <input aria-label="New identity name" value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="New identity" />
          <button onClick={createIdentity} title="Create identity"><UserPlus size={16} />Create</button>
        </div>
      </aside>

      <section className="gallery-detail">
        <div className="gallery-editor">
          <h2><Pencil size={16} />Identity</h2>
          {selectedIdentity ? (
            <>
              {!selectedIdentity.active && (
                <div style={{ color: 'var(--muted)', marginBottom: '1rem', fontWeight: 'bold' }}>This identity is inactive</div>
              )}
              <label>
                <span>Display name</span>
                <input aria-label="Identity display name" value={draftName} onChange={(event) => setDraftName(event.target.value)} />
              </label>
              <div className="editor-actions">
                <button onClick={saveIdentity}>Save</button>
                {selectedIdentity.active ? (
                  <button onClick={deactivateIdentity}><CircleX size={16} />Deactivate</button>
                ) : (
                  <button onClick={activateIdentity}><Check size={16} />Activate</button>
                )}
                <button onClick={() => setDeleteConfirm(true)}><Trash2 size={16} />Delete</button>
              </div>
              {deleteConfirm && (
                <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: '4px' }}>
                  <p>Permanently delete {selectedIdentity.display_name || selectedIdentity.identity_key} and all {selectedIdentity.embedding_count} samples? This cannot be undone.</p>
                  <div className="editor-actions" style={{ marginTop: '1rem' }}>
                    <button onClick={() => setDeleteConfirm(false)}>Cancel</button>
                    <button onClick={permanentlyDeleteIdentity} style={{ color: '#ef4444' }}>Delete permanently</button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="empty-inline">No identity selected</div>
          )}
        </div>

        <div className="sample-list">
          <h2><Layers size={16} />Samples</h2>
          {embeddings.length === 0 ? (
            <div className="empty-inline">No samples</div>
          ) : (
            embeddings.map((embedding) => (
              <div className="sample-row" key={embedding.id}>
                <span>#{embedding.id} · {embedding.source_type || "manual"} · {embedding.active ? "active" : "inactive"}</span>
                <button onClick={() => removeEmbedding(embedding.id)} title={`Deactivate sample ${embedding.id}`}><Trash2 size={15} /></button>
              </div>
            ))
          )}
        </div>

        <div className="enrollment-panel">
          <h2><Camera size={16} />Enrollment</h2>
          <div className="tab-row">
            <button className={source === "upload" ? "active" : ""} onClick={() => setSource("upload")}><Upload size={16} />Upload photo</button>
            <button className={source === "webcam" ? "active" : ""} onClick={() => setSource("webcam")}><Camera size={16} />Browser webcam</button>
            <button className={source === "stream" ? "active" : ""} onClick={() => setSource("stream")}><RefreshCw size={16} />Runner snapshot</button>
          </div>
          {source === "upload" && (
            <div className="source-row">
              <input ref={fileRef} aria-label="Upload enrollment photo" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => void analyzeUpload(event.target.files?.[0])} />
            </div>
          )}
          {source === "webcam" && <button onClick={analyzeWebcam} disabled={pending}>Capture webcam</button>}
          {source === "stream" && (
            <div className="source-row">
              <select aria-label="Runner stream" value={streamChoice || (streams[0] ? streamKey(streams[0]) : "")} onChange={(event) => setStreamChoice(event.target.value)}>
                {streams.map((stream) => <option key={streamKey(stream)} value={streamKey(stream)}>{streamKey(stream)}</option>)}
              </select>
              <button onClick={analyzeStream} disabled={!selectedStream || pending}>Capture stream</button>
            </div>
          )}
          {enrollment && (
            <div className="candidate-preview">
              <div className="preview-frame">
                <img src={previewUrl} alt="Enrollment preview" />
                {enrollment.candidates.map((candidate) => (
                  <button
                    key={candidate.candidate_id}
                    className={selectedCandidates.includes(candidate.candidate_id) ? "face-box active" : "face-box"}
                    style={{
                      left: `${(candidate.bbox.x / Math.max(1, candidate.image_width)) * 100}%`,
                      top: `${(candidate.bbox.y / Math.max(1, candidate.image_height)) * 100}%`,
                      width: `${(candidate.bbox.w / Math.max(1, candidate.image_width)) * 100}%`,
                      height: `${(candidate.bbox.h / Math.max(1, candidate.image_height)) * 100}%`
                    }}
                    onClick={() => setSelectedCandidates((current) =>
                      current.includes(candidate.candidate_id)
                        ? current.filter((id) => id !== candidate.candidate_id)
                        : [...current, candidate.candidate_id]
                    )}
                    aria-label={`Select ${candidate.candidate_id}`}
                  />
                ))}
              </div>
              <div className="candidate-list">
                {enrollment.candidates.map((candidate) => (
                  <label key={candidate.candidate_id}>
                    <input
                      type="checkbox"
                      checked={selectedCandidates.includes(candidate.candidate_id)}
                      disabled={!candidate.usable}
                      onChange={() => setSelectedCandidates((current) =>
                        current.includes(candidate.candidate_id)
                          ? current.filter((id) => id !== candidate.candidate_id)
                          : [...current, candidate.candidate_id]
                      )}
                    />
                    <span>{candidate.candidate_id} · {candidate.usable ? "usable" : candidate.reject_reasons.join(", ")}</span>
                  </label>
                ))}
              </div>
              <button onClick={commitCandidates} disabled={!selectedIdentity || selectedCandidates.length === 0 || pending}>Commit selected</button>
            </div>
          )}
          {galleryMessage && <pre>{galleryMessage}</pre>}
        </div>
      </section>
    </section>
  );
}

function RenderLayerToggles({ layers, onChange }: { layers: RenderLayerOptions; onChange: (layers: RenderLayerOptions) => void }) {
  const items: Array<[keyof RenderLayerOptions, string]> = [
    ["tracks", "Tracks"],
    ["faces", "Faces"],
    ["directions", "Directions"],
    ["rules", "Rules"],
    ["events", "Events"]
  ];
  return (
    <div className="toggle-grid" aria-label="Render layers">
      {items.map(([key, label]) => (
        <label key={key}>
          <input type="checkbox" checked={layers[key]} onChange={(event) => onChange({ ...layers, [key]: event.target.checked })} />
          {label}
        </label>
      ))}
    </div>
  );
}

function RenderRuleList({ rules, onDelete }: { rules: RenderRule[]; onDelete: (ruleId: string) => void }) {
  return (
    <div className="rule-list" aria-label="Render rules">
      {rules.length === 0 ? (
        <div className="empty-inline">No render rules</div>
      ) : (
        rules.map((rule) => (
          <div className="rule-row" key={rule.id}>
            <div>
              <strong>{rule.name}</strong>
              <small>{rule.kind}</small>
            </div>
            <div className="rule-actions">
              <button onClick={() => onDelete(rule.id)} title="Delete render rule" aria-label={`Delete ${rule.name}`}>
                <Trash2 size={15} />
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function renderFrameLabel(job: RenderJobStatus | null) {
  if (!job?.progress_frame) return "-";
  return job.total_frames ? `${job.progress_frame} / ${job.total_frames}` : String(job.progress_frame);
}

function renderPercentLabel(job: RenderJobStatus | null) {
  return job?.progress_percent !== undefined && job.progress_percent !== null ? `${job.progress_percent.toFixed(1)}%` : "";
}

function RenderPlayer({
  previewUrl,
  livePreviewUrl,
  job,
  imageSize,
  renderRules,
  renderLayers,
  drawMode,
  onImageLoad,
  onCreateRule,
  onUpdateRule
}: {
  previewUrl: string;
  livePreviewUrl: string;
  job: RenderJobStatus | null;
  imageSize: { width: number; height: number };
  renderRules: RenderRule[];
  renderLayers: RenderLayerOptions;
  drawMode: DrawMode;
  onImageLoad: (width: number, height: number) => void;
  onCreateRule: (kind: "line" | "area", points: Point[]) => void;
  onUpdateRule: (ruleId: string, points: Point[]) => void;
}) {
  const activeRendering = job?.status === "queued" || job?.status === "running";
  const showLive = Boolean(livePreviewUrl && (activeRendering || job?.status === "succeeded" || job?.status === "failed" || job?.status === "cancelled"));
  const imageUrl = showLive ? livePreviewUrl : previewUrl;
  const status = job?.status ?? "idle";
  const snapshot: AnalyticsSnapshot | undefined = imageUrl && !showLive && !activeRendering && imageSize.width > 0 && imageSize.height > 0 ? {
    stream: { stream_id: "render", profile: "preview" },
    frame_id: 0,
    timestamp_ms: 0,
    width: imageSize.width,
    height: imageSize.height,
    counts: { active_tracks: 0, unique_tracks: 0 },
    tracks: [],
    heatmap: { rows: 1, cols: 1, values: [0], max_value: 0 },
    density: { rows: 1, cols: 1, values: [0], max_value: 0 },
    directions: [],
    routes: [],
    rules: renderRules.map((rule) => ({
      ...rule,
      stream_id: "render",
      profile: "preview",
      created_at_ms: 0,
      updated_at_ms: 0
    })),
    recent_events: []
  } : undefined;
  const overlayLayers: OverlayLayers = {
    tracks: renderLayers.tracks,
    faces: renderLayers.faces,
    heatmap: false,
    density: false,
    directions: renderLayers.directions,
    routes: false,
    rules: renderLayers.rules,
    events: renderLayers.events
  };

  return (
    <div className="render-preview">
      <header className="render-player-status">
        <strong>{status}</strong>
        <span>Frame {renderFrameLabel(job)}</span>
        {renderPercentLabel(job) && <span>{renderPercentLabel(job)}</span>}
      </header>
      {imageUrl ? (
        <div className="video-wrap">
          <img
            className="stream-media"
            src={imageUrl}
            alt="Render preview"
            onLoad={(event) => onImageLoad(event.currentTarget.naturalWidth, event.currentTarget.naturalHeight)}
          />
          {snapshot && (
            <AnalyticsOverlay
              snapshot={snapshot}
              layers={overlayLayers}
              drawMode={drawMode}
              onCreateRule={onCreateRule}
              onUpdateRule={(rule, points) => onUpdateRule(rule.id, points)}
            />
          )}
        </div>
      ) : (
        <div className="render-preview-empty"><Image size={22} />Preview frame</div>
      )}
    </div>
  );
}

function RenderView({ configPath, configFiles }: { configPath: string; configFiles: ConfigFileInfo[] }) {
  const [inputPath, setInputPath] = useState("assets/output.mp4");
  const [renderConfigPath, setRenderConfigPath] = useState(configPath);
  const [galleryDb, setGalleryDb] = useState("");
  const [outputPath, setOutputPath] = useState("/tmp/veilsight_render.mp4");
  const [overwrite, setOverwrite] = useState(false);
  const [timingMode, setTimingMode] = useState<"source" | "custom">("source");
  const [renderMode, setRenderMode] = useState("face+body");
  const [fps, setFps] = useState("25");
  const [sourceFps, setSourceFps] = useState("");
  const [noGallery, setNoGallery] = useState(true);
  const [previewEveryFrames, setPreviewEveryFrames] = useState(5);
  const [layers, setLayers] = useState<RenderLayerOptions>(defaultRenderLayers);
  const [drawMode, setDrawMode] = useState<DrawMode>("select");
  const [renderRules, setRenderRules] = useState<RenderRule[]>([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [job, setJob] = useState<RenderJobStatus | null>(null);
  const [message, setMessage] = useState("");
  const activeRendering = job?.status === "queued" || job?.status === "running";
  const livePreviewUrl = job?.preview_updated_at_ms ? renderJobPreviewUrl(job.job_id, job.preview_updated_at_ms) : "";

  useEffect(() => {
    setRenderConfigPath((current) => current || configPath);
  }, [configPath]);

  useEffect(() => {
    void api.renderSettings().then((settings) => {
      setGalleryDb((current) => current || settings.default_gallery_db);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      void api.renderJob(job.job_id).then(setJob).catch((error) => setMessage(String(error)));
    }, 500);
    return () => window.clearInterval(timer);
  }, [job]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function loadPreview() {
    setMessage("");
    const blob = await api.renderPreview(inputPath);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(blob));
  }

  function createRenderRule(kind: "line" | "area", points: Point[]) {
    const count = renderRules.filter((rule) => rule.kind === kind).length + 1;
    const rule: RenderRule = {
      id: `rule_${Date.now()}_${count}`,
      kind,
      name: `${kind === "line" ? "Line" : "Area"} ${count}`,
      enabled: true,
      geometry: { points },
      settings: kind === "line" ? { min_gap_ms: 1000 } : { dwell_threshold_s: 3.0 }
    };
    setRenderRules((current) => [...current, rule]);
    setDrawMode("select");
  }

  function updateRenderRule(ruleId: string, points: Point[]) {
    setRenderRules((current) => current.map((rule) => rule.id === ruleId ? { ...rule, geometry: { ...rule.geometry, points } } : rule));
  }

  async function startRender() {
    setMessage("");
    const created = await api.createRenderJob({
      config_path: renderConfigPath,
      input_path: inputPath,
      output_path: outputPath,
      gallery_db: noGallery ? null : galleryDb || null,
      layers: selectedRenderLayers(layers),
      rules: renderRules,
      overwrite,
      timing_mode: timingMode,
      render_mode: renderMode,
      fps: timingMode === "custom" ? Number(fps) : null,
      source_fps: timingMode === "custom" && sourceFps ? Number(sourceFps) : null,
      no_gallery: noGallery,
      preview_every_frames: previewEveryFrames
    });
    setJob(created);
  }

  async function cancelRender() {
    if (!job) return;
    setJob(await api.cancelRenderJob(job.job_id));
  }

  return (
    <section className="render-view">
      <aside className="render-panel">
        <h2><Film size={16} />Render</h2>
        <label>
          <span>Input video path</span>
          <input aria-label="Render input video path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} />
        </label>
        <label>
          <span>Config file</span>
          <select aria-label="Render config path" value={renderConfigPath} onChange={(event) => setRenderConfigPath(event.target.value)}>
            {!renderConfigPath && <option value="">Choose a config</option>}
            {configFiles.map((file) => <option value={file.path} key={file.path}>{file.name}</option>)}
          </select>
        </label>
        <label>
          <span>Gallery DB path</span>
          <input aria-label="Render gallery DB path" value={galleryDb} disabled={noGallery} onChange={(event) => setGalleryDb(event.target.value)} />
        </label>
        <section>
          <h2><FileCheck size={16} />Mode</h2>
          <div className="segmented" aria-label="Render timing mode">
            <button className={timingMode === "source" ? "active" : ""} onClick={() => setTimingMode("source")}>Original FPS</button>
            <button className={timingMode === "custom" ? "active" : ""} onClick={() => setTimingMode("custom")}>Custom FPS</button>
          </div>
          {timingMode === "custom" && (
            <div className="field-row">
              <label>
                <span>Output FPS</span>
                <input aria-label="Render output FPS" type="number" min="0.01" step="0.01" value={fps} onChange={(event) => setFps(event.target.value)} />
              </label>
            </div>
          )}
          <div className="segmented" style={{ marginTop: 12 }} aria-label="Render anonymization mode">
            <button className={renderMode === "face+body" ? "active" : ""} onClick={() => setRenderMode("face+body")}>Face + Body</button>
            <button className={renderMode === "face" ? "active" : ""} onClick={() => setRenderMode("face")}>Face Only</button>
          </div>
          <label className="check-row">
            <input aria-label="No gallery: anonymize everyone" type="checkbox" checked={noGallery} onChange={(event) => setNoGallery(event.target.checked)} />
            <span>No gallery: anonymize everyone</span>
          </label>
          <label>
            <span>Preview update interval</span>
            <select aria-label="Render preview update interval" value={previewEveryFrames} onChange={(event) => setPreviewEveryFrames(Number(event.target.value))}>
              <option value={1}>Every 1 frame</option>
              <option value={5}>Every 5 frames</option>
              <option value={10}>Every 10 frames</option>
            </select>
          </label>
        </section>
        <label>
          <span>Output MP4 path</span>
          <input aria-label="Render output MP4 path" value={outputPath} onChange={(event) => setOutputPath(event.target.value)} />
        </label>
        <label className="check-row">
          <input type="checkbox" checked={overwrite} onChange={(event) => setOverwrite(event.target.checked)} />
          <span>Overwrite existing output</span>
        </label>
        <div className="editor-actions">
          <button onClick={loadPreview}><Image size={16} />Preview</button>
          <button aria-label="Start render job" onClick={startRender} disabled={!renderConfigPath || !inputPath || !outputPath}><Play size={16} />Start</button>
          <button onClick={cancelRender} disabled={!job || !["queued", "running"].includes(job.status)}><CircleX size={16} />Cancel</button>
        </div>
      </aside>

      <section className="render-canvas">
        <RenderPlayer
          previewUrl={previewUrl}
          livePreviewUrl={livePreviewUrl}
          job={job}
          imageSize={imageSize}
          renderRules={renderRules}
          renderLayers={layers}
          drawMode={activeRendering ? "select" : drawMode}
          onImageLoad={(width, height) => setImageSize({ width, height })}
          onCreateRule={createRenderRule}
          onUpdateRule={updateRenderRule}
        />
      </section>

      <aside className="render-panel">
        <section>
          <h2><Layers size={16} />Layers</h2>
          <RenderLayerToggles layers={layers} onChange={setLayers} />
          <DrawTools mode={drawMode} onChange={setDrawMode} disabled={activeRendering} />
        </section>
        <section>
          <h2><TrendingUp size={16} />Rules</h2>
          <RenderRuleList rules={renderRules} onDelete={(ruleId) => setRenderRules((current) => current.filter((rule) => rule.id !== ruleId))} />
        </section>
        <section>
          <h2><Download size={16} />Job</h2>
          {job ? (
            <div className="render-job">
              <div className="stat-row"><span>Status</span><strong>{job.status}</strong></div>
              <div className="stat-row"><span>Frame</span><strong>{renderFrameLabel(job)}</strong></div>
              <div className="stat-row"><span>Percent</span><strong>{renderPercentLabel(job) || "-"}</strong></div>
              <div className="path-block"><span>MP4</span><code>{job.output_path}</code></div>
              <div className="path-block"><span>Events</span><code>{job.events_jsonl_path || "-"}</code></div>
              {job.stderr_tail && <pre>{job.stderr_tail}</pre>}
              {job.status === "succeeded" && <div className="empty-inline">Completed output is ready</div>}
            </div>
          ) : (
            <div className="empty-inline">No render job</div>
          )}
          {message && <pre>{message}</pre>}
        </section>
      </aside>
    </section>
  );
}

export function App() {
  const [activeView, setActiveView] = useState<"monitor" | "gallery" | "render" | "config">("monitor");
  const [health, setHealth] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<PipelineStatus>({});
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [snapshots, setSnapshots] = useState<Record<string, AnalyticsSnapshot>>({});
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [layers, setLayers] = useState<OverlayLayers>(defaultLayers);
  const [drawMode, setDrawMode] = useState<DrawMode>("select");
  const [streamProtocols, setStreamProtocols] = useState<Record<string, StreamProtocol>>({});
  const [configPath, setConfigPath] = useState("");
  const [configFiles, setConfigFiles] = useState<ConfigFileInfo[]>([]);
  const [yamlText, setYamlText] = useState("");
  const [configMessage, setConfigMessage] = useState("");

  async function refresh() {
    const [healthValue, statusValue, streamsValue, metricsValue, configValue, configsValue] = await Promise.all([
      api.health().catch((error) => ({ ok: false, error: String(error) })),
      api.status().catch((error) => ({ state: "disconnected", message: String(error) })),
      api.streams().catch(() => ({ streams: [], public_base_url: "" })),
      api.metrics().catch(() => ({})),
      api.config().catch(() => ({ path: "", yaml_text: "" })),
      api.configs().catch(() => ({ active_path: "", files: [] }))
    ]);
    setHealth(healthValue);
    setStatus(statusValue);
    setStreams(streamsValue.streams);
    setMetrics(metricsValue);
    setConfigPath(configValue.path);
    setConfigFiles(configsValue.files);
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

  function updateStreamProtocol(key: string, protocol: StreamProtocol) {
    setStreamProtocols((prev) => (prev[key] === protocol ? prev : { ...prev, [key]: protocol }));
  }

  async function validateConfig() {
    const result = await api.validateConfig(yamlText);
    setConfigMessage(JSON.stringify(result, null, 2));
  }

  async function saveConfig() {
    const result = await api.saveConfig(yamlText);
    setConfigMessage(JSON.stringify(result, null, 2));
  }

  async function selectConfig(path: string) {
    const result = await api.selectConfig(path);
    setConfigPath(result.path);
    setYamlText(result.yaml_text);
    setConfigMessage(`Loaded ${result.path}`);
    setConfigFiles((files) => files.map((file) => ({ ...file, active: file.path === result.path })));
    await refresh();
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

  async function renameRule(ruleId: string, name: string) {
    const rule = await api.updateAnalyticsRule(ruleId, { name });
    setSnapshots((prev) => {
      const next = { ...prev };
      for (const [key, snapshot] of Object.entries(prev)) {
        next[key] = { ...snapshot, rules: snapshot.rules.map((item) => (item.id === ruleId ? (rule as any) : item)) };
      }
      return next;
    });
  }

  async function deleteRule(ruleId: string) {
    await api.deleteAnalyticsRule(ruleId);
    setSnapshots((prev) => {
      const next = { ...prev };
      for (const [key, snapshot] of Object.entries(prev)) {
        next[key] = { ...snapshot, rules: snapshot.rules.filter((item) => item.id !== ruleId) };
      }
      return next;
    });
  }

  const globalStages = stageRows(metrics.global);
  const queues = queueRows(metrics.queues);
  const selectedStreamId = selectedStream?.stream_id ?? "";
  const visibleQueues = queues.filter((q: any) => {
    const name = String(q.name ?? "");
    if (name.startsWith("global/")) return true;
    if (selectedStreamId && name.startsWith(`stream/${selectedStreamId}/`)) return true;
    const slash = name.indexOf("/");
    if (slash < 0) return true;
    return name.slice(0, slash) === selectedStreamId;
  });

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Veilsight</h1>
          <span>{configPath || "No active config"}</span>
        </div>
        <div className="status-cluster">
          <nav className="top-nav" aria-label="Primary">
            <button className={activeView === "monitor" ? "active" : ""} onClick={() => setActiveView("monitor")}>Monitor</button>
            <button className={activeView === "gallery" ? "active" : ""} onClick={() => setActiveView("gallery")}>Gallery</button>
            <button className={activeView === "render" ? "active" : ""} onClick={() => setActiveView("render")}>Render</button>
            <button className={activeView === "config" ? "active" : ""} onClick={() => setActiveView("config")}>Config</button>
          </nav>
          <span className="pill">Runner: {String((health as any).connection_state ?? "unknown")}</span>
          <span className="pill">Pipeline: {status.state ?? "unknown"}</span>
          <button onClick={() => runCommand(api.start)} title="Start"><Play size={16} />Start</button>
          <button onClick={() => runCommand(api.stop)} title="Stop"><Square size={16} />Stop</button>
          <button onClick={() => runCommand(api.reload)} title="Reload"><RefreshCw size={16} />Reload</button>
        </div>
      </header>

      {activeView === "monitor" && <section className="workspace">
        <StreamWorkspace
          streams={streams}
          selectedKey={selectedKey}
          snapshots={snapshots}
          layers={layers}
          drawMode={drawMode}
          onSelect={setSelectedKey}
          onCreateRule={createRule}
          onUpdateRule={updateRule}
          protocols={streamProtocols}
          onProtocolChange={updateStreamProtocol}
        />

        <aside className="side-panel">
          <section>
            <h2><Activity size={16} />Metrics</h2>
            <div className="metric-grid">
              {[
                ["person_detector", "person_detector"],
                ["tracker", "tracker"],
                ["face_detector", "face_detector"],
                ["recognizer", "recognizer"],
                ["identity", "identity"],
                ["anonymizer", "anonymizer"],
                ["encoder", "encoder"]
              ].map(([stage, label]) => {
                const row = globalStages.find((item: any) => item.stage === stage) ?? {};
                return <div className="metric" key={stage}><span>{label}</span><strong>{Number(row.fps ?? 0).toFixed(1)} fps</strong><small>p95 {Number(row.p95_ms ?? 0).toFixed(1)} ms</small></div>;
              })}
            </div>
            <table>
              <thead><tr><th>Queue</th><th>Flow</th><th>Size</th><th>Drop</th></tr></thead>
              <tbody>{visibleQueues.map((q: any) => <tr key={q.name}><td>{q.name}</td><td>{q.producer && q.consumer ? `${q.producer} -> ${q.consumer}` : "-"}</td><td>{q.size}/{q.capacity}</td><td>{q.dropped}</td></tr>)}</tbody>
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
            <RuleManager rules={selectedSnapshot?.rules ?? []} onRename={renameRule} onDelete={deleteRule} />
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
      </section>}

      {activeView === "gallery" && <GalleryView streams={streams} />}

      {activeView === "render" && <RenderView configPath={configPath} configFiles={configFiles} />}

      {activeView === "config" && <section className="config-editor">
        <h2><FileCheck size={16} />Config</h2>
        <div className="config-toolbar">
          <label>
            <FolderOpen size={16} />
            <span>Config file</span>
            <select
              value={configPath}
              onChange={(event) => void selectConfig(event.target.value)}
              disabled={configFiles.length === 0}
              aria-label="Config file"
            >
              {configFiles.length === 0 ? (
                <option value="">No config files found</option>
              ) : (
                <>
                  {!configPath && <option value="">Choose a config</option>}
                  {configFiles.map((file) => (
                    <option value={file.path} key={file.path}>
                      {file.name}
                    </option>
                  ))}
                </>
              )}
            </select>
          </label>
          <span>{configPath || "No active config"}</span>
        </div>
        <textarea aria-label="Config YAML" value={yamlText} onChange={(event) => setYamlText(event.target.value)} spellCheck={false} />
        <div className="editor-actions">
          <button onClick={validateConfig}>Validate</button>
          <button onClick={saveConfig}>Save</button>
          <button onClick={() => runCommand(api.reload)}>Reload pipeline</button>
        </div>
        {configMessage && <pre>{configMessage}</pre>}
      </section>}
    </main>
  );
}
