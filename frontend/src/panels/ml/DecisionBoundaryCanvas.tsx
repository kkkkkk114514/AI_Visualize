import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchDatasetPoints } from "../../api/datasets";
import type { SnapshotPayload } from "../../api/types";
import { useAlgoStore } from "../../stores/algoStore";
import { useCanvasPainter } from "../algo/canvas";
import { useAlgoSnapshot } from "../algo/snapshot";
import { colormapLut, decodePayload, dequantize } from "../ProbeViewer/paint";

/** 区域与散点配色：区域用冷端 / 暖端，散点提亮以便压在区域上仍可辨。 */
const CLASS_RGB: [number, number, number][] = [
  [59, 76, 192],
  [180, 4, 38],
];
const POINT_COLORS = ["#5b8ff9", "#f2637b"];

interface Points {
  points: [number, number][];
  labels: number[];
}

function useDatasetPoints(datasetId: string | null, split: "train" | "val"): Points | null {
  const [state, setState] = useState<Points | null>(null);
  useEffect(() => {
    if (!datasetId) {
      setState(null);
      return;
    }
    let cancelled = false;
    fetchDatasetPoints(datasetId, split)
      .then((payload) => {
        if (!cancelled) setState({ points: payload.points, labels: payload.labels });
      })
      .catch(() => {
        if (!cancelled) setState(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, split]);
  return state;
}

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  sx: (x: number) => number;
  sy: (y: number) => number;
}

function resolveRange(payload: SnapshotPayload | null, points: Points | null) {
  const xr = payload?.meta?.x_range;
  const yr = payload?.meta?.y_range;
  if (xr && yr) return { x0: xr[0], x1: xr[1], y0: yr[0], y1: yr[1] };
  if (points && points.points.length > 0) {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const [x, y] of points.points) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const mx = (x1 - x0) * 0.05 || 0.1;
    const my = (y1 - y0) * 0.05 || 0.1;
    return { x0: x0 - mx, x1: x1 + mx, y0: y0 - my, y1: y1 + my };
  }
  return { x0: -1, x1: 1, y0: -1, y1: 1 };
}

/** 决策场：按阈值把实值映射到发散色带，边界落在色带中点。
 *  回归与森林的场是「0/1 目标上的实值 / 软概率」，判界在 0.5；其余（逻辑、SVM）是 s(x)，判界在 0。 */
function paintField(
  ctx: CanvasRenderingContext2D,
  payload: SnapshotPayload,
  box: Box,
  band: boolean,
): void {
  const decoded = decodePayload(payload);
  const [rows, cols] = decoded.shape;
  if (!rows || !cols) return;
  const mode = decoded.meta?.mode ?? "score";
  const algo = decoded.meta?.algo;
  const threshold = algo === "linear_regression" || algo === "random_forest" ? 0.5 : 0;
  const span = Math.max(Math.abs(decoded.max - threshold), Math.abs(decoded.min - threshold)) || 1;
  const lut = colormapLut("coolwarm");

  const field = document.createElement("canvas");
  field.width = cols;
  field.height = rows;
  const fieldCtx = field.getContext("2d");
  if (!fieldCtx) return;
  const image = fieldCtx.createImageData(cols, rows);
  const bandImage = band ? fieldCtx.createImageData(cols, rows) : null;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      // 网格行序是 y 升序（(0,0) 在左下），画布行序自上而下，因此上下翻转
      const source = (rows - 1 - row) * cols + col;
      const value = dequantize(decoded, Number(decoded.values[source] ?? 0));
      const base = (row * cols + col) * 4;
      if (mode === "label") {
        const [r, g, b] = CLASS_RGB[value >= 0.5 ? 1 : 0];
        image.data[base] = r;
        image.data[base + 1] = g;
        image.data[base + 2] = b;
        image.data[base + 3] = 110;
      } else {
        const t = 0.5 + ((value - threshold) / span) * 0.5;
        const level = Math.round(Math.min(1, Math.max(0, t)) * 255);
        image.data[base] = lut[level * 4];
        image.data[base + 1] = lut[level * 4 + 1];
        image.data[base + 2] = lut[level * 4 + 2];
        image.data[base + 3] = 255;
      }
      if (bandImage) {
        // SVM 的间隔带：hinge 的 ±1 水平线，即 margin 所在的区域
        const inside = mode === "score" && Math.abs(value) <= 1;
        bandImage.data[base] = 255;
        bandImage.data[base + 1] = 255;
        bandImage.data[base + 2] = 255;
        bandImage.data[base + 3] = inside ? 60 : 0;
      }
    }
  }

  fieldCtx.putImageData(image, 0, 0);
  ctx.drawImage(field, box.x, box.y, box.width, box.height);
  if (bandImage) {
    fieldCtx.putImageData(bandImage, 0, 0);
    ctx.drawImage(field, box.x, box.y, box.width, box.height);
  }
}

function drawPoints(ctx: CanvasRenderingContext2D, points: Points | null, box: Box): void {
  if (!points) return;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
  for (let index = 0; index < points.points.length; index += 1) {
    const [x, y] = points.points[index];
    ctx.beginPath();
    ctx.arc(box.sx(x), box.sy(y), 3, 0, Math.PI * 2);
    ctx.fillStyle = POINT_COLORS[points.labels[index] === 1 ? 1 : 0];
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawSupportVectors(ctx: CanvasRenderingContext2D, payload: SnapshotPayload, box: Box): void {
  const vectors = payload.meta?.support_vectors ?? [];
  ctx.save();
  for (const [x, y] of vectors) {
    const cx = box.sx(x);
    const cy = box.sy(y);
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
    ctx.lineWidth = 2.6;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }
  ctx.restore();
}

export function DecisionBoundaryCanvas() {
  const { t } = useTranslation();
  const spec = useAlgoStore((state) => state.spec);
  const [split, setSplit] = useState<"train" | "val">("train");
  const datasetId = spec?.kind === "ml" ? spec.dataset_id ?? null : null;
  const points = useDatasetPoints(datasetId, split);
  const { runId, meta, payload, stale } = useAlgoSnapshot("model", "boundary");

  const canvasRef = useCanvasPainter(
    (ctx, width, height) => {
      if (!payload && !points) return;
      const pad = 10;
      const range = resolveRange(payload, points);
      const scale = Math.min(
        (width - pad * 2) / (range.x1 - range.x0),
        (height - pad * 2) / (range.y1 - range.y0),
      );
      const plotWidth = (range.x1 - range.x0) * scale;
      const plotHeight = (range.y1 - range.y0) * scale;
      const box: Box = {
        x: (width - plotWidth) / 2,
        y: (height - plotHeight) / 2,
        width: plotWidth,
        height: plotHeight,
        sx: (x) => (width - plotWidth) / 2 + (x - range.x0) * scale,
        sy: (y) => (height - plotHeight) / 2 + plotHeight - (y - range.y0) * scale,
      };
      if (payload) {
        const band = payload.meta?.mode === "score" && payload.meta?.algo === "svm";
        paintField(ctx, payload, box, band);
      }
      drawPoints(ctx, points, box);
      if (payload && payload.meta?.support_vectors) drawSupportVectors(ctx, payload, box);
      ctx.strokeStyle = "rgba(147, 161, 177, 0.35)";
      ctx.lineWidth = 1;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    },
    [payload, points],
  );

  if (!spec) return null;
  const info = payload?.meta;
  const badges: string[] = [];
  if (meta) badges.push(`step ${meta.step}${meta.epoch ? ` · e${meta.epoch}` : ""}`);
  if (info?.mode) badges.push(t(`algo.boundary.mode.${info.mode}`));
  if (info?.mode === "label") {
    if (info.depth !== undefined) badges.push(t("algo.boundary.depth", { value: info.depth }));
    if (info.leaves !== undefined) badges.push(t("algo.boundary.leaves", { value: info.leaves }));
    if (info.n_trees !== undefined) badges.push(t("algo.boundary.trees", { value: info.n_trees }));
  } else if (info?.algo === "svm") {
    if (info.margin !== undefined) badges.push(t("algo.boundary.margin", { value: info.margin }));
    if (info.n_sv !== undefined) badges.push(t("algo.boundary.nSv", { value: info.n_sv }));
  }
  if (stale) badges.push(t("common.loading"));

  return (
    <div className="algo-canvas">
      <div className="algo-canvas__toolbar">
        <select
          className="probe__select"
          value={split}
          onChange={(event) => setSplit(event.target.value as "train" | "val")}
          title={t("algo.boundary.splitHint")}
        >
          <option value="train">{t("algo.boundary.split.train")}</option>
          <option value="val">{t("algo.boundary.split.val")}</option>
        </select>
        <span className="algo-canvas__spacer" />
        {badges.length > 0 ? <span className="probe__meta">{badges.join(" · ")}</span> : null}
      </div>
      <div className="algo-canvas__stage">
        <canvas className="algo-canvas__bitmap" ref={canvasRef} />
        {!payload && !points ? (
          <div className="algo-canvas__empty">
            {runId ? t("algo.boundary.waiting") : t("algo.boundary.noRun")}
          </div>
        ) : null}
      </div>
      <div className="algo-canvas__legend">
        <span className="algo-legend">
          <i className="algo-legend__dot" style={{ background: POINT_COLORS[0] }} />
          {t("algo.boundary.class0")}
        </span>
        <span className="algo-legend">
          <i className="algo-legend__dot" style={{ background: POINT_COLORS[1] }} />
          {t("algo.boundary.class1")}
        </span>
        {info?.algo === "svm" && info.mode === "score" ? (
          <span className="algo-legend">
            <i className="algo-legend__ring" />
            {t("algo.boundary.svLegend")}
          </span>
        ) : null}
        <span className="algo-canvas__hint">{t("algo.boundary.hint")}</span>
      </div>
    </div>
  );
}
