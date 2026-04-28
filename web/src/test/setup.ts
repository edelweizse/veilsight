import "@testing-library/jest-dom/vitest";

HTMLCanvasElement.prototype.getContext = (() => ({
  clearRect: () => undefined,
  scale: () => undefined,
  strokeRect: () => undefined,
  fillText: () => undefined,
  set font(_value: string) {},
  set strokeStyle(_value: string) {},
  set fillStyle(_value: string) {},
  set lineWidth(_value: number) {}
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;
