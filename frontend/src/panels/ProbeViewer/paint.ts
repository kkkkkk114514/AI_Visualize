import type { SnapshotPayload } from "../../api/types";

export type ColormapName = "viridis" | "magma" | "gray" | "coolwarm";

export const COLORMAP_NAMES: ColormapName[] = ["viridis", "magma", "gray", "coolwarm"];

/** 控制点线性插值成 256 级 LUT，避免每个像素都做插值。 */
const STOPS: Record<ColormapName, [number, number, number][]> = {
  viridis: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ],
  magma: [
    [0, 0, 4],
    [81, 18, 124],
    [183, 55, 121],
    [252, 137, 97],
    [252, 253, 191],
  ],
  gray: [
    [12, 14, 18],
    [246, 248, 250],
  ],
  coolwarm: [
    [59, 76, 192],
    [221, 221, 221],
    [180, 4, 38],
  ],
};

const luts = new Map<ColormapName, Uint8ClampedArray>();

export function colormapLut(name: ColormapName): Uint8ClampedArray {
  const cached = luts.get(name);
  if (cached) return cached;

  const stops = STOPS[name];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let index = 0; index < 256; index += 1) {
    const t = (index / 255) * (stops.length - 1);
    const low = Math.min(Math.floor(t), stops.length - 2);
    const ratio = t - low;
    const [r0, g0, b0] = stops[low];
    const [r1, g1, b1] = stops[low + 1];
    lut[index * 4] = r0 + (r1 - r0) * ratio;
    lut[index * 4 + 1] = g0 + (g1 - g0) * ratio;
    lut[index * 4 + 2] = b0 + (b1 - b0) * ratio;
    lut[index * 4 + 3] = 255;
  }
  luts.set(name, lut);
  return lut;
}

export interface DecodedSnapshot {
  shape: number[];
  layout: string;
  dtype: string;
  /** 量化前真实值域（histogram 为 bin 范围） */
  min: number;
  max: number;
  meta: SnapshotPayload["meta"];
  values: Uint8Array | Uint32Array;
}

function base64ToBytes(encoded: string): Uint8Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export function decodePayload(payload: SnapshotPayload): DecodedSnapshot {
  const bytes = base64ToBytes(payload.data_b64);
  const values =
    payload.dtype === "uint32"
      ? new Uint32Array(bytes.buffer, 0, Math.floor(bytes.byteLength / 4))
      : bytes;
  return {
    shape: payload.shape,
    layout: payload.layout,
    dtype: payload.dtype,
    min: payload.min,
    max: payload.max,
    meta: payload.meta,
    values,
  };
}

/** 量化值 → 量化前真实值（histogram 的计数不做反量化）。 */
export function dequantize(decoded: DecodedSnapshot, quantized: number): number {
  if (decoded.dtype !== "uint8") return quantized;
  return decoded.min + (quantized / 255) * (decoded.max - decoded.min);
}

export interface MatrixView {
  /** 该矩阵在 values 中的起始下标 */
  offset: number;
  rows: number;
  cols: number;
}

/** 按 layout 切出二维矩阵视图：CHW/HTT 是多块，TH 单块，histogram 交给直方图渲染器。 */
export function matrixViews(decoded: DecodedSnapshot): MatrixView[] {
  const [first, second, third] = decoded.shape;
  if (decoded.layout === "CHW" || decoded.layout === "HTT") {
    if (!first || !second || !third) return [];
    const stride = second * third;
    return Array.from({ length: first }, (_, index) => ({
      offset: index * stride,
      rows: second,
      cols: third,
    }));
  }
  if (decoded.layout === "TH" && first && second) {
    return [{ offset: 0, rows: first, cols: second }];
  }
  return [];
}

export interface GridLayout {
  cols: number;
  rows: number;
  tileW: number;
  tileH: number;
  gap: number;
  width: number;
  height: number;
}

/** 平铺布局：优先「整屏放得下」里单块最大的方案，放不下则取最矮的一档（滚动查看）。 */
export function gridLayout(
  count: number,
  containerWidth: number,
  aspect: number,
  maxHeight: number,
  gap = 4,
  minTile = 20,
): GridLayout {
  const width = Math.max(0, Math.floor(containerWidth));
  let fit: GridLayout | null = null;
  let smallest: GridLayout | null = null;
  for (let cols = 1; cols <= count; cols += 1) {
    const rows = Math.ceil(count / cols);
    const tileW = Math.floor((width - gap * (cols - 1)) / cols);
    const tileH = Math.max(1, Math.round(tileW * aspect));
    if (tileW < minTile || tileH < minTile) break;
    const candidate: GridLayout = {
      cols,
      rows,
      tileW,
      tileH,
      gap,
      width,
      height: rows * tileH + (rows - 1) * gap,
    };
    if (candidate.height <= maxHeight) {
      if (!fit || tileW * tileH > fit.tileW * fit.tileH) fit = candidate;
    }
    if (!smallest || candidate.height < smallest.height) smallest = candidate;
  }
  const layout = fit ?? smallest;
  if (layout) return layout;
  const tileW = Math.max(width, minTile);
  const tileH = Math.max(minTile, Math.round(tileW * aspect));
  return { cols: 1, rows: count, tileW, tileH, gap, width, height: count * tileH + (count - 1) * gap };
}

/** 把单通道矩阵画成位图：先 write 成 cols×rows 的 ImageData，再用最近邻放大到瓦片。 */
export function paintMatrix(
  target: CanvasRenderingContext2D,
  view: MatrixView,
  decoded: DecodedSnapshot,
  lut: Uint8ClampedArray,
  tile: { x: number; y: number; width: number; height: number },
): void {
  const offscreen = document.createElement("canvas");
  offscreen.width = view.cols;
  offscreen.height = view.rows;
  const context = offscreen.getContext("2d");
  if (!context) return;
  const image = context.createImageData(view.cols, view.rows);
  const isUint8 = decoded.dtype === "uint8";
  for (let index = 0; index < view.rows * view.cols; index += 1) {
    const quantized = Number(decoded.values[view.offset + index] ?? 0);
    const level = isUint8 ? Math.min(255, Math.max(0, quantized)) : 0;
    const base = index * 4;
    image.data[base] = lut[level * 4];
    image.data[base + 1] = lut[level * 4 + 1];
    image.data[base + 2] = lut[level * 4 + 2];
    image.data[base + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  target.imageSmoothingEnabled = false;
  target.drawImage(offscreen, 0, 0, view.cols, view.rows, tile.x, tile.y, tile.width, tile.height);
}

/** 像素级读数：4 位有效数字，兼顾 1e-4 量级与 O(1) 量级。 */
export function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 0.001 || magnitude >= 100000) return value.toExponential(2);
  return value.toPrecision(4);
}
