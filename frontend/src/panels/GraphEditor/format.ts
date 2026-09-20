import type { TFunction } from "i18next";

const BATCH_SENTINEL = 2;

export function formatShape(shape?: number[] | null): string {
  if (!shape || shape.length === 0) return "—";
  return shape
    .map((dim, index) => (index === 0 && dim === BATCH_SENTINEL ? "B" : String(dim)))
    .join("×");
}

export function formatShapes(shapes?: number[][] | null): string {
  if (!shapes || shapes.length === 0) return "—";
  return shapes.map((shape) => formatShape(shape)).join(" + ");
}

export function formatParams(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString("en-US");
}

export function compactParams(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1_000_000).toFixed(2)}M`;
}

/** 后端 args 直接插值会打印成 JSON：形状与枚举先转成可读文本。 */
export function formatIssueArgs(
  args: Record<string, unknown> | undefined,
  t: TFunction,
): Record<string, unknown> {
  if (!args) return {};
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (key === "kind" && typeof value === "string") {
      out[key] = t(`errors.kind.${value}`, { defaultValue: value });
    } else if (key === "shapes" && Array.isArray(value)) {
      out[key] = formatShapes(value as number[][]);
    } else if (key === "shape" && Array.isArray(value)) {
      out[key] = formatShape(value as number[]);
    } else if (Array.isArray(value)) {
      out[key] = value.join(", ");
    } else {
      out[key] = value;
    }
  }
  return out;
}
