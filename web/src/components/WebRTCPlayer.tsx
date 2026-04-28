import { useEffect, useRef, useState } from "react";
import type { StreamInfo } from "../api/client";

async function waitForIce(peer: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (peer.iceGatheringState === "complete") return;
  await new Promise<void>((resolve) => {
    const timer = window.setTimeout(resolve, timeoutMs);
    peer.addEventListener("icegatheringstatechange", () => {
      if (peer.iceGatheringState === "complete") {
        window.clearTimeout(timer);
        resolve();
      }
    });
  });
}

export function WebRTCPlayer({ stream }: { stream: StreamInfo }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [failed, setFailed] = useState(!stream.webrtc_available);

  useEffect(() => {
    setFailed(!stream.webrtc_available);
    if (!stream.webrtc_available) return;

    let closed = false;
    const peer = new RTCPeerConnection();
    peer.addTransceiver("video", { direction: "recvonly" });
    peer.ontrack = (event) => {
      if (videoRef.current) videoRef.current.srcObject = event.streams[0];
    };

    async function connect() {
      try {
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        await waitForIce(peer, 1500);
        const response = await fetch(stream.webrtc_offer_url, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: peer.localDescription?.sdp ?? offer.sdp ?? ""
        });
        if (!response.ok) throw new Error(`WHEP failed: ${response.status}`);
        const answer = await response.text();
        if (closed) return;
        await peer.setRemoteDescription({ type: "answer", sdp: answer });
      } catch {
        setFailed(true);
        peer.close();
      }
    }

    void connect();
    return () => {
      closed = true;
      peer.close();
    };
  }, [stream]);

  if (failed) {
    return <img className="stream-media" src={stream.mjpeg_url} alt={`${stream.stream_id} MJPEG`} />;
  }

  return <video ref={videoRef} className="stream-media" autoPlay playsInline muted />;
}
