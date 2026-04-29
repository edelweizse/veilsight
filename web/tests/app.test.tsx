import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

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

function snapshot(stream_id: string, frame_id = 5) {
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
    rules: [],
    recent_events: []
  };
}

function mockFetch(streams = [stream("cam0")]) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/streams") return Response.json({ public_base_url: "http://localhost:8080", streams });
    if (url === "/api/config") return Response.json({ path: "/tmp/config.yaml", yaml_text: "streams: []" });
    if (url === "/api/pipeline/status") return Response.json({ state: "stopped" });
    if (url === "/api/health") return Response.json({ ok: true, connection_state: "connected", runner: { ok: true } });
    if (url === "/api/metrics") return Response.json({});
    if (url === "/api/pipeline/start") return Response.json({ accepted: true, status: { state: "running" } });
    if (url === "/api/analytics/rules" && init?.method === "POST") {
      const body = JSON.parse(String(init.body));
      return Response.json({ id: "rule-1", created_at_ms: 1, updated_at_ms: 1, enabled: true, settings: {}, ...body });
    }
    return Response.json({ valid: true });
  });
}

describe("dashboard", () => {
  beforeEach(() => {
    sockets.length = 0;
    HTMLCanvasElement.prototype.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  });

  afterEach(() => {
    cleanup();
  });

  it("renders streams and calls pipeline commands", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;

    render(<App />);
    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    await userEvent.click(screen.getByTitle("Start"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/start", { method: "POST" }));
  });

  it("focuses a clicked stream thumbnail", async () => {
    global.fetch = mockFetch([stream("cam0"), stream("cam1")]) as any;
    const { container } = render(<App />);

    expect(await screen.findByText("cam1/ui")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /cam1\/ui/ }));
    expect(container.querySelector(".focused-stream header strong")?.textContent).toBe("cam1/ui");
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
});
