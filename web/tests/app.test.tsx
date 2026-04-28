import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

class MockSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  close = vi.fn();
  constructor(public url: string) {}
}

global.WebSocket = MockSocket as any;

describe("dashboard", () => {
  it("renders streams and calls pipeline commands", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/streams") return Response.json({ public_base_url: "http://localhost:8080", streams: [{ stream_id: "cam0", profile: "ui", webrtc_available: false, webrtc_offer_url: "", mjpeg_url: "http://localhost:8080/video/cam0/ui", snapshot_url: "", metadata_url: "" }] });
      if (url === "/api/config") return Response.json({ path: "/tmp/config.yaml", yaml_text: "streams: []" });
      if (url === "/api/pipeline/status") return Response.json({ state: "stopped" });
      if (url === "/api/health") return Response.json({ ok: true, connection_state: "connected", runner: { ok: true } });
      if (url === "/api/metrics") return Response.json({});
      if (url === "/api/pipeline/start") return Response.json({ accepted: true, status: { state: "running" } });
      return Response.json({ valid: true });
    });
    global.fetch = fetchMock as any;

    render(<App />);
    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    await userEvent.click(screen.getByTitle("Start"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/start", { method: "POST" }));
  });
});
