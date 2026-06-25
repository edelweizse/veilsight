import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { WebRTCPlayer } from "../src/components/WebRTCPlayer";

const sockets: MockSocket[] = [];

class MockSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  close = vi.fn();
  constructor(public url: string) {
    sockets.push(this);
  }
  emit(value: unknown) {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

global.WebSocket = MockSocket as any;

class MockMediaStream {
  constructor(public tracks: unknown[] = []) {}
  getTracks() {
    return this.tracks as MediaStreamTrack[];
  }
}

class MockPeerConnection {
  static instances: MockPeerConnection[] = [];
  iceGatheringState = "complete";
  connectionState = "new";
  localDescription = { type: "offer", sdp: "mock-offer" };
  ontrack: ((event: RTCTrackEvent) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  addTransceiver = vi.fn();
  addEventListener = vi.fn();
  close = vi.fn(() => {
    this.connectionState = "closed";
  });
  createOffer = vi.fn(async () => ({ type: "offer", sdp: "mock-offer" }));
  setLocalDescription = vi.fn(async () => undefined);
  setRemoteDescription = vi.fn(async () => undefined);

  constructor() {
    MockPeerConnection.instances.push(this);
  }
}

global.MediaStream = MockMediaStream as any;

function stream(stream_id: string) {
  return {
    stream_id,
    profile: "ui",
    webrtc_available: false,
    webrtc_offer_url: "",
    mjpeg_url: `http://localhost:8080/video/${stream_id}/ui`,
    snapshot_url: "",
    metadata_url: ""
  };
}

function rule(name = "Entry line") {
  return {
    id: "rule-1",
    stream_id: "cam0",
    profile: "ui",
    kind: "line",
    name,
    enabled: true,
    geometry: { points: [{ x: 10, y: 10 }, { x: 50, y: 50 }] },
    settings: {},
    created_at_ms: 1,
    updated_at_ms: 1
  };
}

function snapshot(stream_id: string, frame_id = 5, rules: any[] = []) {
  return {
    stream: { stream_id, profile: "ui" },
    frame_id,
    timestamp_ms: 1000,
    width: 100,
    height: 100,
    counts: { active_tracks: 1, unique_tracks: 1 },
    tracks: [{ id: 1, bbox: { x: 10, y: 10, w: 20, h: 30 }, foot: { x: 20, y: 38 }, dwell_s: 1.2, velocity_px_s: 3 }],
    heatmap: { rows: 2, cols: 2, values: [1, 0, 0, 0], max_value: 1 },
    density: { rows: 2, cols: 2, values: [1, 0, 0, 0], max_value: 1 },
    directions: [],
    routes: [],
    rules,
    recent_events: []
  };
}

function mockFetch(streams = [stream("cam0")], metrics: Record<string, unknown> = {}) {
  let config = { path: "/tmp/config.yaml", yaml_text: "streams: []" };
  let identities = [{ identity_key: "alice", display_name: "Alice", active: true, embedding_count: 1 }];
  let embeddings = [{ id: 7, identity_key: "alice", model: "mobilefacenet", dim: 128, active: true, source_type: "upload" }];
  const candidateResponse = {
    enrollment_id: "enroll-1",
    expires_at_ms: 999999,
    image_id: "source",
    candidates: [
      {
        candidate_id: "face-0",
        image_id: "source",
        bbox: { x: 10, y: 10, w: 20, h: 20 },
        landmarks: [],
        score: 0.95,
        image_width: 100,
        image_height: 100,
        usable: true,
        reject_reasons: []
      }
    ]
  };
  const files = [
    { path: "/tmp/config.yaml", name: "config.yaml" },
    { path: "/tmp/alternate.yaml", name: "alternate.yaml" }
  ];
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/streams") return Response.json({ public_base_url: "http://localhost:8080", streams });
    if (url === "/api/config") return Response.json(config);
    if (url === "/api/configs") {
      return Response.json({
        active_path: config.path,
        files: files.map((file) => ({ ...file, active: file.path === config.path }))
      });
    }
    if (url === "/api/config/select" && init?.method === "POST") {
      const body = JSON.parse(String(init.body));
      config = { path: body.path, yaml_text: "streams:\n  - id: alternate" };
      return Response.json(config);
    }
    if (url === "/api/pipeline/status") return Response.json({ state: "stopped" });
    if (url === "/api/health") return Response.json({ ok: true, connection_state: "connected", runner: { ok: true } });
    if (url === "/api/metrics") return Response.json(metrics);
    if (url === "/api/gallery/identities") {
      if (init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        const created = { identity_key: "bob", display_name: body.display_name, active: true, embedding_count: 0 };
        identities = [...identities, created];
        return Response.json(created);
      }
      return Response.json(identities);
    }
    if (String(url).startsWith("/api/gallery/identities/alice") && init?.method === "PATCH") {
      const body = JSON.parse(String(init.body));
      identities = identities.map((identity) => identity.identity_key === "alice" ? { ...identity, ...body } : identity);
      return Response.json(identities[0]);
    }
    if (url === "/api/gallery/identities/alice/embeddings" && init?.method === "POST") {
      embeddings = [...embeddings, { id: 8, identity_key: "alice", model: "mobilefacenet", dim: 128, active: true, source_type: "upload" }];
      identities = identities.map((identity) => identity.identity_key === "alice" ? { ...identity, embedding_count: identity.embedding_count + 1 } : identity);
      return Response.json([embeddings[embeddings.length - 1]]);
    }
    if (url === "/api/gallery/identities/alice/embeddings") return Response.json(embeddings);
    if (url === "/api/gallery/embeddings/7" && init?.method === "DELETE") {
      embeddings = embeddings.map((embedding) => embedding.id === 7 ? { ...embedding, active: false } : embedding);
      return Response.json(embeddings[0]);
    }
    if (url === "/api/gallery/enrollment-candidates" && init?.method === "POST") return Response.json(candidateResponse);
    if (url === "/api/gallery/enrollment-candidates/from-stream" && init?.method === "POST") return Response.json(candidateResponse);
    if (url === "/api/gallery/refresh" && init?.method === "POST") return Response.json({ accepted: true, message: "gallery reloaded" });
    if (url === "/api/renders/settings") return Response.json({ binary_path: "/tmp/render", default_gallery_db: "/tmp/gallery.sqlite3" });
    if (url === "/api/renders/preview" && init?.method === "POST") return new Response(new Blob(["jpeg"], { type: "image/jpeg" }));
    if (url === "/api/renders/jobs" && init?.method === "POST") {
      const body = JSON.parse(String(init.body));
      return Response.json({
        job_id: "job-1",
        status: "succeeded",
        created_at_ms: 1,
        updated_at_ms: 2,
        config_path: body.config_path,
        input_path: body.input_path,
        output_path: body.output_path,
        gallery_db: body.gallery_db,
        layers: body.layers,
        rules_yaml_path: "/tmp/rules.yaml",
        events_jsonl_path: "/tmp/veilsight_render.events.jsonl",
        manifest_path: "/tmp/veilsight_render.manifest.json",
        progress_frame: 30,
        total_frames: 120,
        progress_percent: 25,
        preview_jpeg_path: "/tmp/latest.jpg",
        preview_updated_at_ms: 10,
        timing_mode: body.timing_mode,
        render_mode: body.render_mode ?? "face+body",
        no_gallery: body.no_gallery,
        returncode: 0,
        stderr_tail: "",
        message: "render complete"
      });
    }
    if (url === "/api/renders/jobs/job-1") {
      return Response.json({
        job_id: "job-1",
        status: "succeeded",
        created_at_ms: 1,
        updated_at_ms: 2,
        config_path: "/tmp/config.yaml",
        input_path: "assets/output.mp4",
        output_path: "/tmp/veilsight_render.mp4",
        layers: ["tracks"],
        events_jsonl_path: "/tmp/veilsight_render.events.jsonl",
        progress_frame: 30,
        total_frames: 120,
        progress_percent: 25,
        preview_jpeg_path: "/tmp/latest.jpg",
        preview_updated_at_ms: 10,
        timing_mode: "source",
        no_gallery: true,
        stderr_tail: "",
        message: "render complete"
      });
    }
    if (url === "/api/pipeline/start") return Response.json({ accepted: true, status: { state: "running" } });
    if (url === "/api/analytics/rules" && init?.method === "POST") {
      const body = JSON.parse(String(init.body));
      return Response.json({ id: "rule-1", created_at_ms: 1, updated_at_ms: 1, enabled: true, settings: {}, ...body });
    }
    if (String(url).startsWith("/api/analytics/rules/") && init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      return Response.json({ ...rule("Entry line"), updated_at_ms: 2, ...body });
    }
    if (String(url).startsWith("/api/analytics/rules/") && init?.method === "DELETE") {
      return Response.json({ deleted: true });
    }
    return Response.json({ valid: true });
  });
}

describe("dashboard", () => {
  beforeEach(() => {
    sockets.length = 0;
    MockPeerConnection.instances.length = 0;
    global.RTCPeerConnection = MockPeerConnection as any;
    HTMLMediaElement.prototype.play = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => new MockMediaStream([{ stop: vi.fn() }])) }
    });
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      arc: vi.fn(),
      beginPath: vi.fn(),
      clearRect: vi.fn(),
      closePath: vi.fn(),
      drawImage: vi.fn(),
      fill: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      restore: vi.fn(),
      save: vi.fn(),
      setLineDash: vi.fn(),
      setTransform: vi.fn(),
      stroke: vi.fn(),
      strokeRect: vi.fn()
    })) as any;
    HTMLCanvasElement.prototype.toBlob = vi.fn((callback: BlobCallback) => callback(new Blob(["jpeg"], { type: "image/jpeg" })));
    HTMLCanvasElement.prototype.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders streams and calls pipeline commands", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;

    render(<App />);
    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    expect(screen.getAllByText("Protocol: MJPEG").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByTitle("Start"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/start", { method: "POST" }));
  });

  it("loads a selected config file into the editor", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Config" }));

    const selector = (await screen.findByLabelText("Config file")) as HTMLSelectElement;
    await userEvent.selectOptions(selector, "/tmp/alternate.yaml");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/config/select", expect.objectContaining({ method: "POST" })));
    expect((screen.getByLabelText("Config file") as HTMLSelectElement).value).toBe("/tmp/alternate.yaml");
    await waitFor(() => expect((screen.getByLabelText("Config YAML") as HTMLTextAreaElement).value).toBe("streams:\n  - id: alternate"));
  });

  it("focuses a clicked stream thumbnail", async () => {
    global.fetch = mockFetch([stream("cam0"), stream("cam1")]) as any;
    const { container } = render(<App />);

    expect(await screen.findByText("cam1/ui")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /cam1\/ui/ }));
    expect(container.querySelector(".focused-stream header strong")?.textContent).toBe("cam1/ui");
  });

  it("filters queue rows to the selected stream and exposes face overlays", async () => {
    global.fetch = mockFetch(
      [stream("file0_0"), stream("file0_1")],
      {
        queues: [
          { name: "global/person_detector.in", size: 1, capacity: 50, dropped: 0, producer: "Ingestor", consumer: "PersonDetectorPool" },
          { name: "stream/file0_0/frames.in", size: 2, capacity: 50, dropped: 0, producer: "Ingestor", consumer: "StreamCoordinator" },
          { name: "stream/file0_1/frames.in", size: 3, capacity: 50, dropped: 0, producer: "Ingestor", consumer: "StreamCoordinator" }
        ]
      }
    ) as any;

    render(<App />);

    expect(await screen.findByText("file0_0/ui")).toBeInTheDocument();
    expect(screen.getByText("global/person_detector.in")).toBeInTheDocument();
    expect(screen.getByText("stream/file0_0/frames.in")).toBeInTheDocument();
    expect(screen.queryByText("stream/file0_1/frames.in")).not.toBeInTheDocument();
    expect(screen.getAllByText("Ingestor -> StreamCoordinator").length).toBeGreaterThan(0);
    expect((screen.getByLabelText("Faces") as HTMLInputElement).checked).toBe(true);

    await userEvent.click(screen.getByRole("button", { name: /file0_1\/ui/ }));

    expect(screen.getByText("stream/file0_1/frames.in")).toBeInTheDocument();
    expect(screen.queryByText("stream/file0_0/frames.in")).not.toBeInTheDocument();
  });

  it("attaches streamless WebRTC tracks to the video element", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("mock-answer", {
        status: 201,
        headers: { "Content-Type": "application/sdp", Location: "/webrtc/whep/sessions/sess-1" }
      })
    );
    global.fetch = fetchMock as any;
    const rtcStream = {
      ...stream("cam0"),
      webrtc_available: true,
      webrtc_offer_url: "http://localhost:8080/webrtc/whep/cam0/ui"
    };

    const { container } = render(<WebRTCPlayer stream={rtcStream} />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(rtcStream.webrtc_offer_url, expect.any(Object)));
    const peer = MockPeerConnection.instances[0];
    const track = { kind: "video" };
    peer.ontrack?.({ streams: [], track } as unknown as RTCTrackEvent);

    const video = container.querySelector("video") as HTMLVideoElement;
    expect((video.srcObject as unknown as MockMediaStream).tracks).toEqual([track]);
  });

  it("updates the analytics panel from overlay websocket snapshots and toggles layers", async () => {
    global.fetch = mockFetch() as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0", 12));
    await waitFor(() => expect(screen.getByText("12")).toBeInTheDocument());

    const heatmap = screen.getByLabelText("Heatmap") as HTMLInputElement;
    expect(heatmap.checked).toBe(false);
    await userEvent.click(heatmap);
    expect(heatmap.checked).toBe(true);
  });

  it("submits line rule geometry in frame coordinates", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0"));
    await userEvent.click(screen.getByTitle("Line rule"));
    const canvas = document.querySelector("canvas.analytics-overlay");
    expect(canvas).not.toBeNull();
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 10, clientY: 10 });
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 50, clientY: 50 });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules", expect.any(Object)));
    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/analytics/rules");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.geometry.points).toEqual([{ x: 10, y: 10 }, { x: 50, y: 50 }]);
  });

  it("renames and deletes drawn analytical events", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0", 5, [rule("Entry line")]));
    expect(await screen.findByText("Entry line")).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText("Rename Entry line"));
    await userEvent.clear(screen.getByLabelText("Rename Entry line"));
    await userEvent.type(screen.getByLabelText("Rename Entry line"), "Lobby line");
    await userEvent.click(screen.getByLabelText("Save Entry line"));

    await waitFor(() => expect(screen.getByText("Lobby line")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules/rule-1", expect.objectContaining({ method: "PUT" }));

    await userEvent.click(screen.getByLabelText("Delete Lobby line"));
    await waitFor(() => expect(screen.queryByText("Lobby line")).not.toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules/rule-1", { method: "DELETE" });
  });

  it("render tab submits selected layer flags", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    await userEvent.click(screen.getByLabelText("Tracks"));
    await userEvent.click(screen.getByLabelText("Faces"));
    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/renders/jobs", expect.objectContaining({ method: "POST" })));
    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/renders/jobs");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.layers).toEqual(["tracks", "faces"]);
    expect(body.output_path).toBe("/tmp/veilsight_render.mp4");
    expect(body.timing_mode).toBe("source");
    expect(body.render_mode).toBe("face+body");
    expect(body.fps).toBeNull();
    expect(body.no_gallery).toBe(true);
    expect(body.gallery_db).toBeNull();
  });

  it("render no-gallery toggle disables gallery DB and submits false when cleared", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    const galleryInput = screen.getByLabelText("Render gallery DB path");
    expect(galleryInput).toBeDisabled();
    await userEvent.click(screen.getByLabelText("No gallery: anonymize everyone"));
    expect(galleryInput).not.toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));

    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/renders/jobs");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.no_gallery).toBe(false);
    expect(body.gallery_db).toBe("/tmp/gallery.sqlite3");
  });

  it("render custom FPS mode submits numeric FPS", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    await userEvent.click(screen.getByRole("button", { name: "Custom FPS" }));
    await userEvent.clear(screen.getByLabelText("Render output FPS"));
    await userEvent.type(screen.getByLabelText("Render output FPS"), "12.5");
    await userEvent.selectOptions(screen.getByLabelText("Render preview update interval"), "10");
    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));

    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/renders/jobs");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.timing_mode).toBe("custom");
    expect(body.render_mode).toBe("face+body");
    expect(body.fps).toBe(12.5);
    expect(body.source_fps).toBe(null);
    expect(body.preview_every_frames).toBe(10);
  });

  it("render drawing line and area creates rule payloads", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    const image = await screen.findByAltText("Render preview");
    Object.defineProperty(image, "naturalWidth", { configurable: true, value: 100 });
    Object.defineProperty(image, "naturalHeight", { configurable: true, value: 100 });
    fireEvent.load(image);

    await userEvent.click(screen.getByTitle("Line rule"));
    const canvas = document.querySelector("canvas.analytics-overlay");
    expect(canvas).not.toBeNull();
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 10, clientY: 10 });
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 50, clientY: 50 });

    await userEvent.click(screen.getByTitle("Area rule"));
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 20, clientY: 20 });
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 60, clientY: 20 });
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 60, clientY: 60 });
    fireEvent.doubleClick(canvas as HTMLCanvasElement, { clientX: 20, clientY: 60 });

    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));
    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/renders/jobs");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.rules[0].kind).toBe("line");
    expect(body.rules[0].geometry.points).toEqual([{ x: 10, y: 10 }, { x: 50, y: 50 }]);
    expect(body.rules[1].kind).toBe("area");
    expect(body.rules[1].geometry.points).toHaveLength(3);
  });

  it("completed render job displays MP4 and events paths", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));

    await waitFor(() => expect(screen.getAllByText("succeeded").length).toBeGreaterThan(0));
    expect(screen.getByText("/tmp/veilsight_render.mp4")).toBeInTheDocument();
    expect(screen.getByText("/tmp/veilsight_render.events.jsonl")).toBeInTheDocument();
    expect(screen.getAllByText("30 / 120").length).toBeGreaterThan(0);
    expect(screen.getAllByText("25.0%").length).toBeGreaterThan(0);
    expect(screen.getByAltText("Render preview")).toHaveAttribute("src", "/api/renders/jobs/job-1/preview.jpg?ts=10");
  });

  it("disables render rule drawing controls while a job is running", async () => {
    const baseFetch = mockFetch();
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/renders/jobs" && init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        return Response.json({
          job_id: "job-running",
          status: "running",
          created_at_ms: 1,
          updated_at_ms: 2,
          config_path: body.config_path,
          input_path: body.input_path,
          output_path: body.output_path,
          layers: body.layers,
          progress_frame: 5,
          total_frames: 100,
          progress_percent: 5,
          timing_mode: body.timing_mode,
          render_mode: body.render_mode ?? "face+body",
          no_gallery: body.no_gallery,
          stderr_tail: "",
          message: "render process started"
        });
      }
      if (url === "/api/renders/jobs/job-running") {
        return Response.json({
          job_id: "job-running",
          status: "running",
          created_at_ms: 1,
          updated_at_ms: 3,
          config_path: "/tmp/config.yaml",
          input_path: "assets/output.mp4",
          output_path: "/tmp/veilsight_render.mp4",
          layers: [],
          progress_frame: 6,
          total_frames: 100,
          progress_percent: 6,
          timing_mode: "source",
          no_gallery: true,
          stderr_tail: "",
          message: "render process started"
        });
      }
      return baseFetch(url, init);
    });
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Render" }));
    await userEvent.click(screen.getByRole("button", { name: "Start render job" }));

    await waitFor(() => expect(screen.getAllByText("running").length).toBeGreaterThan(0));
    expect(screen.getByTitle("Line rule")).toBeDisabled();
    expect(screen.getByTitle("Area rule")).toBeDisabled();
  });

  it("renders gallery identities and commits upload candidates", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Gallery" }));
    expect(await screen.findByText("Alice")).toBeInTheDocument();

    const file = new File(["image"], "alice.jpg", { type: "image/jpeg" });
    await userEvent.upload(screen.getByLabelText("Upload enrollment photo"), file);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/enrollment-candidates", expect.objectContaining({ method: "POST" })));
    expect(await screen.findByText("face-0 · usable")).toBeInTheDocument();

    await userEvent.click(screen.getByText("Commit selected"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/identities/alice/embeddings", expect.objectContaining({ method: "POST" })));
    expect(fetchMock).toHaveBeenCalledWith("/api/gallery/refresh", { method: "POST" });
    expect(await screen.findByText(/gallery reloaded/)).toBeInTheDocument();
  });

  it("permanently deletes a gallery identity after confirmation", async () => {
    const baseFetch = mockFetch();
    let identities = [{ identity_key: "alice", display_name: "Alice", active: true, embedding_count: 1 }];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/gallery/identities" && !init?.method) return Response.json(identities);
      if (url === "/api/gallery/identities/alice" && init?.method === "DELETE") {
        identities = [];
        return Response.json({ deleted: true });
      }
      return baseFetch(url, init);
    });
    global.fetch = fetchMock as any;

    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: "Gallery" }));
    expect(await screen.findByText("Alice")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(await screen.findByText(/Permanently delete/)).toBeInTheDocument();

    await userEvent.click(screen.getByText("Delete permanently"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/identities/alice", { method: "DELETE" }));
    expect(fetchMock).toHaveBeenCalledWith("/api/gallery/refresh", { method: "POST" });
    await waitFor(() => expect(screen.queryByText("Alice")).not.toBeInTheDocument());
  });

  it("commits candidates for inactive identities and shows activation message", async () => {
    const baseFetch = mockFetch();
    let identities = [{ identity_key: "alice", display_name: "Alice", active: false, embedding_count: 0 }];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/gallery/identities" && !init?.method) return Response.json(identities);
      if (url === "/api/gallery/identities/alice/embeddings" && init?.method === "POST") {
        const body = JSON.parse(String(init.body ?? "{}")) as { candidate_ids?: string[] };
        identities = identities.map((identity) =>
          identity.identity_key === "alice"
            ? { ...identity, active: true, embedding_count: identity.embedding_count + (body.candidate_ids?.length ?? 0) }
            : identity
        );
        return Response.json([{ id: 8, identity_key: "alice", model: "mobilefacenet", dim: 128, active: true, source_type: "upload" }]);
      }
      return baseFetch(url, init);
    });
    global.fetch = fetchMock as any;

    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: "Gallery" }));
    expect(await screen.findByText(/This identity is inactive/)).toBeInTheDocument();

    const file = new File(["image"], "alice.jpg", { type: "image/jpeg" });
    await userEvent.upload(screen.getByLabelText("Upload enrollment photo"), file);
    expect(await screen.findByText("face-0 · usable")).toBeInTheDocument();

    await userEvent.click(screen.getByText("Commit selected"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/identities/alice/embeddings", expect.objectContaining({ method: "POST" })));
    expect(fetchMock).toHaveBeenCalledWith("/api/gallery/refresh", { method: "POST" });
    expect(await screen.findByText(/identity activated/)).toBeInTheDocument();
  });

  it("captures webcam and runner stream candidates", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Gallery" }));
    await userEvent.click(screen.getByRole("button", { name: /Browser webcam/ }));
    await userEvent.click(screen.getByText("Capture webcam"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/enrollment-candidates", expect.objectContaining({ method: "POST" })));

    await userEvent.click(screen.getByRole("button", { name: /Runner snapshot/ }));
    await userEvent.click(screen.getByText("Capture stream"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/enrollment-candidates/from-stream", expect.objectContaining({ method: "POST" })));
  });

  it("deactivates gallery identity and samples", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Gallery" }));
    await userEvent.click(await screen.findByText("Deactivate"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/identities/alice", expect.objectContaining({ method: "PATCH" })));

    await userEvent.click(screen.getByTitle("Deactivate sample 7"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/gallery/embeddings/7", { method: "DELETE" }));
  });
});
