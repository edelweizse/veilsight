import "@testing-library/jest-dom/vitest";

HTMLCanvasElement.prototype.getContext = (() => ({
  setTransform: () => undefined,
  clearRect: () => undefined,
  scale: () => undefined,
  save: () => undefined,
  restore: () => undefined,
  strokeRect: () => undefined,
  fillRect: () => undefined,
  beginPath: () => undefined,
  moveTo: () => undefined,
  lineTo: () => undefined,
  closePath: () => undefined,
  stroke: () => undefined,
  fill: () => undefined,
  arc: () => undefined,
  fillText: () => undefined,
  set font(_value: string) {},
  set strokeStyle(_value: string) {},
  set fillStyle(_value: string) {},
  set lineWidth(_value: number) {}
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;
