import { useEffect, useRef, useState } from "react";
import { clientToFrame, fitFrameToRect, frameRectToCanvas, frameToCanvas } from "../analytics/coordinates";
import type { AnalyticsRule, AnalyticsSnapshot, DrawMode, OverlayLayers, Point } from "../analytics/types";

type Props = {
  snapshot?: AnalyticsSnapshot;
  layers: OverlayLayers;
  drawMode: DrawMode;
  onCreateRule: (kind: "line" | "area", points: Point[]) => Promise<void> | void;
  onUpdateRule: (rule: AnalyticsRule, points: Point[]) => Promise<void> | void;
};

type DragState = {
  rule: AnalyticsRule;
  pointIndex: number;
  points: Point[];
};

function colorWithAlpha(hex: string, alpha: number) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function rulePoints(rule: AnalyticsRule): Point[] {
  return Array.isArray(rule.geometry.points) ? (rule.geometry.points as Point[]) : [];
}

function drawPolyline(ctx: CanvasRenderingContext2D, points: Point[], close: boolean) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  if (close) ctx.closePath();
}

export function AnalyticsOverlay({ snapshot, layers, drawMode, onCreateRule, onUpdateRule }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);

  const width = snapshot?.width ?? 0;
  const height = snapshot?.height ?? 0;

  useEffect(() => {
    if (drawMode === "select") {
      setDraftPoints([]);
      setHoverPoint(null);
    }
  }, [drawMode]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setDraftPoints([]);
        setHoverPoint(null);
        setDrag(null);
      }
      if (event.key === "Enter" && drawMode === "area" && draftPoints.length >= 3) {
        event.preventDefault();
        void onCreateRule("area", draftPoints);
        setDraftPoints([]);
        setHoverPoint(null);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawMode, draftPoints, onCreateRule]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * pixelRatio));
    canvas.height = Math.max(1, Math.floor(rect.height * pixelRatio));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!snapshot || !width || !height) return;

    const fit = fitFrameToRect({ width, height }, rect);
    ctx.save();
    ctx.strokeStyle = "#d9dde2";
    ctx.strokeRect(fit.dx, fit.dy, fit.displayWidth, fit.displayHeight);
    ctx.restore();

    if (layers.heatmap && snapshot.heatmap.max_value > 0) {
      const cellW = width / snapshot.heatmap.cols;
      const cellH = height / snapshot.heatmap.rows;
      snapshot.heatmap.values.forEach((value, index) => {
        if (value <= 0) return;
        const col = index % snapshot.heatmap.cols;
        const row = Math.floor(index / snapshot.heatmap.cols);
        const topLeft = frameToCanvas({ x: col * cellW, y: row * cellH }, fit);
        ctx.fillStyle = colorWithAlpha("#d84a3a", Math.min(0.55, 0.08 + value / snapshot.heatmap.max_value * 0.47));
        ctx.fillRect(topLeft.x, topLeft.y, cellW * fit.scale, cellH * fit.scale);
      });
    }

    if (layers.density && snapshot.density.max_value > 0) {
      const cellW = width / snapshot.density.cols;
      const cellH = height / snapshot.density.rows;
      snapshot.density.values.forEach((value, index) => {
        if (value <= 0) return;
        const col = index % snapshot.density.cols;
        const row = Math.floor(index / snapshot.density.cols);
        const topLeft = frameToCanvas({ x: col * cellW, y: row * cellH }, fit);
        ctx.fillStyle = colorWithAlpha("#2d7dd2", Math.min(0.45, 0.1 + value / snapshot.density.max_value * 0.35));
        ctx.fillRect(topLeft.x, topLeft.y, cellW * fit.scale, cellH * fit.scale);
      });
    }

    if (layers.routes) {
      ctx.lineWidth = 3;
      for (const route of snapshot.routes) {
        const points = route.points.map((point) => frameToCanvas(point, fit));
        ctx.strokeStyle = "#7a4cc2";
        drawPolyline(ctx, points, false);
        ctx.stroke();
        const last = points[points.length - 1];
        if (last) {
          ctx.fillStyle = "#7a4cc2";
          ctx.font = "12px system-ui";
          ctx.fillText(`${route.support}`, last.x + 5, last.y - 5);
        }
      }
    }

    if (layers.tracks) {
      ctx.font = "12px system-ui";
      for (const track of snapshot.tracks) {
        const box = frameRectToCanvas(track.bbox, fit);
        const foot = frameToCanvas(track.foot, fit);
        ctx.strokeStyle = "#159957";
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = "#159957";
        ctx.beginPath();
        ctx.arc(foot.x, foot.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(`id:${track.id} ${track.dwell_s.toFixed(1)}s`, box.x + 4, Math.max(14, box.y - 4));
      }
    }

    if (layers.directions) {
      for (const direction of snapshot.directions) {
        const from = frameToCanvas(direction.from, fit);
        const to = frameToCanvas(direction.to, fit);
        ctx.strokeStyle = "#f28c28";
        ctx.fillStyle = "#f28c28";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        const angle = Math.atan2(to.y - from.y, to.x - from.x);
        ctx.beginPath();
        ctx.moveTo(to.x, to.y);
        ctx.lineTo(to.x - Math.cos(angle - 0.5) * 9, to.y - Math.sin(angle - 0.5) * 9);
        ctx.lineTo(to.x - Math.cos(angle + 0.5) * 9, to.y - Math.sin(angle + 0.5) * 9);
        ctx.closePath();
        ctx.fill();
      }
    }

    if (layers.rules) {
      for (const rule of snapshot.rules) {
        const sourcePoints = drag?.rule.id === rule.id ? drag.points : rulePoints(rule);
        const points = sourcePoints.map((point) => frameToCanvas(point, fit));
        ctx.lineWidth = 2;
        ctx.strokeStyle = rule.enabled ? "#ffcc33" : "#8c939d";
        ctx.fillStyle = "rgba(255, 204, 51, 0.16)";
        drawPolyline(ctx, points, rule.kind === "area");
        if (rule.kind === "area") ctx.fill();
        ctx.stroke();
        for (const point of points) {
          ctx.fillStyle = "#101418";
          ctx.strokeStyle = "#ffcc33";
          ctx.beginPath();
          ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        const label = points[0];
        if (label) {
          ctx.fillStyle = "#ffcc33";
          ctx.font = "12px system-ui";
          ctx.fillText(rule.name, label.x + 7, label.y - 7);
        }
      }
    }

    if (layers.events) {
      for (const event of snapshot.recent_events.slice(0, 10)) {
        const x = Number(event.position.x);
        const y = Number(event.position.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const point = frameToCanvas({ x, y }, fit);
        ctx.fillStyle = "#c33d57";
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(event.kind.replace("area_", ""), point.x + 8, point.y + 4);
      }
    }

    const draft = [...draftPoints, ...(hoverPoint ? [hoverPoint] : [])].map((point) => frameToCanvas(point, fit));
    if (draft.length) {
      ctx.strokeStyle = "#49a3ff";
      ctx.fillStyle = "rgba(73, 163, 255, 0.16)";
      ctx.lineWidth = 2;
      drawPolyline(ctx, draft, false);
      ctx.stroke();
      for (const point of draft) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [snapshot, layers, width, height, draftPoints, hoverPoint, drag]);

  function pointerToFrame(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = ref.current;
    if (!canvas || !width || !height) return null;
    const rect = canvas.getBoundingClientRect();
    const fit = fitFrameToRect({ width, height }, rect);
    return clientToFrame(event.clientX, event.clientY, rect, fit);
  }

  function findHandle(event: React.PointerEvent<HTMLCanvasElement>): DragState | null {
    const canvas = ref.current;
    if (!canvas || !snapshot || !width || !height) return null;
    const rect = canvas.getBoundingClientRect();
    const fit = fitFrameToRect({ width, height }, rect);
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    for (const rule of snapshot.rules) {
      const points = rulePoints(rule);
      for (let index = 0; index < points.length; index += 1) {
        const canvasPoint = frameToCanvas(points[index], fit);
        if (Math.hypot(canvasPoint.x - x, canvasPoint.y - y) <= 10) {
          return { rule, pointIndex: index, points: points.map((point) => ({ ...point })) };
        }
      }
    }
    return null;
  }

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (drawMode !== "select") return;
    const handle = findHandle(event);
    if (handle) {
      event.currentTarget.setPointerCapture(event.pointerId);
      setDrag(handle);
    }
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = pointerToFrame(event);
    if (drag && point) {
      setDrag({ ...drag, points: drag.points.map((item, index) => (index === drag.pointIndex ? point : item)) });
      return;
    }
    setHoverPoint(drawMode === "select" ? null : point);
  }

  function onPointerUp() {
    if (!drag) return;
    void onUpdateRule(drag.rule, drag.points);
    setDrag(null);
  }

  function onClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (drag || drawMode === "select") return;
    const point = pointerToFrame(event as unknown as React.PointerEvent<HTMLCanvasElement>);
    if (!point) return;
    if (drawMode === "line") {
      if (draftPoints.length === 0) {
        setDraftPoints([point]);
      } else {
        void onCreateRule("line", [draftPoints[0], point]);
        setDraftPoints([]);
        setHoverPoint(null);
      }
      return;
    }
    setDraftPoints((current) => [...current, point]);
  }

  function onDoubleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (drawMode !== "area" || draftPoints.length < 3) return;
    event.preventDefault();
    void onCreateRule("area", draftPoints);
    setDraftPoints([]);
    setHoverPoint(null);
  }

  return (
    <canvas
      ref={ref}
      className={`analytics-overlay ${drawMode !== "select" ? "drawing" : ""}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => setDrag(null)}
      onMouseLeave={() => setHoverPoint(null)}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    />
  );
}
