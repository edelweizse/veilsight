import { useEffect, useRef } from "react";

type Track = { id: number; x: number; y: number; w: number; h: number; score: number; occluded: boolean };
type Analytics = { width: number; height: number; tracks?: Track[] };

export function TrackOverlay({ analytics }: { analytics?: Analytics }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * scale));
    canvas.height = Math.max(1, Math.floor(rect.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!analytics || !analytics.width || !analytics.height) return;
    ctx.scale(scale, scale);
    const sx = rect.width / analytics.width;
    const sy = rect.height / analytics.height;
    ctx.font = "12px system-ui";
    for (const track of analytics.tracks ?? []) {
      const x = track.x * sx;
      const y = track.y * sy;
      const w = track.w * sx;
      const h = track.h * sy;
      ctx.strokeStyle = track.occluded ? "#d98c00" : "#1f9d55";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(`id:${track.id} score:${track.score.toFixed(2)}`, x + 4, Math.max(14, y - 4));
    }
  }, [analytics]);

  return <canvas ref={ref} className="track-overlay" />;
}
