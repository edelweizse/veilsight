import type { Point } from "./types";

export type FrameSize = { width: number; height: number };
export type CanvasFit = {
  scale: number;
  dx: number;
  dy: number;
  displayWidth: number;
  displayHeight: number;
};

export function fitFrameToRect(frame: FrameSize, rect: DOMRect | { width: number; height: number }): CanvasFit {
  if (!frame.width || !frame.height || !rect.width || !rect.height) {
    return { scale: 1, dx: 0, dy: 0, displayWidth: rect.width, displayHeight: rect.height };
  }
  const scale = Math.min(rect.width / frame.width, rect.height / frame.height);
  const displayWidth = frame.width * scale;
  const displayHeight = frame.height * scale;
  return {
    scale,
    dx: (rect.width - displayWidth) / 2,
    dy: (rect.height - displayHeight) / 2,
    displayWidth,
    displayHeight
  };
}

export function frameToCanvas(point: Point, fit: CanvasFit): Point {
  return { x: fit.dx + point.x * fit.scale, y: fit.dy + point.y * fit.scale };
}

export function frameRectToCanvas(rect: { x: number; y: number; w: number; h: number }, fit: CanvasFit) {
  return {
    x: fit.dx + rect.x * fit.scale,
    y: fit.dy + rect.y * fit.scale,
    w: rect.w * fit.scale,
    h: rect.h * fit.scale
  };
}

export function clientToFrame(clientX: number, clientY: number, rect: DOMRect, fit: CanvasFit): Point | null {
  const x = clientX - rect.left - fit.dx;
  const y = clientY - rect.top - fit.dy;
  if (x < 0 || y < 0 || x > fit.displayWidth || y > fit.displayHeight) return null;
  return { x: x / fit.scale, y: y / fit.scale };
}
