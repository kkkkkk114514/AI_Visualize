import { useTranslation } from "react-i18next";

import type { GridEnv, SnapshotPayload } from "../../api/types";
import { useAlgoStore } from "../../stores/algoStore";
import { useCanvasPainter } from "../algo/canvas";
import { useAlgoSnapshot } from "../algo/snapshot";
import { colormapLut, decodePayload, dequantize } from "../ProbeViewer/paint";

/** 动作下标 → 单位向量（0=上 / 1=下 / 2=左 / 3=右，y 向上；docs/02 §13.5）。 */
const ACTION_DELTA: [number, number][] = [
  [0, 1],
  [0, -1],
  [-1, 0],
  [1, 0],
];

interface GridBox {
  cell: number;
  width: number;
  height: number;
  /** 格子 (x, y) 左上角的画布坐标（y 向上，(0,0) 在左下） */
  px: (x: number) => number;
  py: (y: number) => number;
  cx: (x: number) => number;
  cy: (y: number) => number;
}

function fitBox(env: GridEnv, width: number, height: number): GridBox {
  const cell = Math.max(6, Math.floor(Math.min(width / env.width, height / env.height)));
  const ox = Math.floor((width - cell * env.width) / 2);
  const oy = Math.floor((height - cell * env.height) / 2);
  return {
    cell,
    width: cell * env.width,
    height: cell * env.height,
    px: (x) => ox + x * cell,
    py: (y) => oy + (env.height - 1 - y) * cell,
    cx: (x) => ox + x * cell + cell / 2,
    cy: (y) => oy + (env.height - 1 - y) * cell + cell / 2,
  };
}

function paintValues(
  ctx: CanvasRenderingContext2D,
  payload: SnapshotPayload,
  env: GridEnv,
  box: GridBox,
): void {
  const decoded = decodePayload(payload);
  const [rows, cols] = decoded.shape;
  if (rows !== env.height || cols !== env.width) return;
  const lut = colormapLut("viridis");
  const span = decoded.max - decoded.min;
  for (let y = 0; y < env.height; y += 1) {
    for (let x = 0; x < env.width; x += 1) {
      const value = dequantize(decoded, Number(decoded.values[y * env.width + x] ?? 0));
      const t = span > 0 ? (value - decoded.min) / span : 0.5;
      const level = Math.round(Math.min(1, Math.max(0, t)) * 255);
      ctx.fillStyle = `rgb(${lut[level * 4]}, ${lut[level * 4 + 1]}, ${lut[level * 4 + 2]})`;
      ctx.fillRect(box.px(x), box.py(y), box.cell, box.cell);
    }
  }
}

function paintGridLines(ctx: CanvasRenderingContext2D, env: GridEnv, box: GridBox): void {
  ctx.save();
  ctx.strokeStyle = "rgba(147, 161, 177, 0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= env.width; x += 1) {
    const px = box.px(0) + x * box.cell;
    ctx.moveTo(px, box.py(0));
    ctx.lineTo(px, box.py(0) + box.height);
  }
  for (let y = 0; y <= env.height; y += 1) {
    const py = box.py(0) + y * box.cell;
    ctx.moveTo(box.px(0), py);
    ctx.lineTo(box.px(0) + box.width, py);
  }
  ctx.stroke();
  ctx.restore();
}

function paintPolicy(ctx: CanvasRenderingContext2D, payload: SnapshotPayload, env: GridEnv, box: GridBox): void {
  const policy = payload.meta?.policy ?? [];
  if (policy.length !== env.width * env.height) return;
  const reach = box.cell * 0.3;
  ctx.save();
  ctx.lineWidth = Math.max(1.2, box.cell * 0.08);
  ctx.lineCap = "round";
  for (let y = 0; y < env.height; y += 1) {
    for (let x = 0; x < env.width; x += 1) {
      const action = policy[y * env.width + x];
      if (action !== 0 && action !== 1 && action !== 2 && action !== 3) continue;
      const [dx, dy] = ACTION_DELTA[action];
      const cx = box.cx(x);
      const cy = box.cy(y);
      // 画布 y 向下，世界 y 向上，方向分量取反
      const tipX = cx + dx * reach;
      const tipY = cy - dy * reach;
      ctx.strokeStyle = "rgba(12, 16, 22, 0.5)";
      ctx.beginPath();
      ctx.moveTo(cx - dx * reach, cy + dy * reach);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.beginPath();
      ctx.moveTo(cx - dx * reach, cy + dy * reach);
      ctx.lineTo(tipX, tipY);
      const wing = reach * 0.5;
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - dx * wing - dy * wing, tipY + dy * wing - dx * wing);
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - dx * wing + dy * wing, tipY + dy * wing + dx * wing);
      ctx.stroke();
    }
  }
  ctx.restore();
}

/** 轨迹折线：最近一个已完成 episode 的访问序列（cell 中心连线）。 */
function paintTrajectory(ctx: CanvasRenderingContext2D, payload: SnapshotPayload, box: GridBox): void {
  const path = payload.meta?.trajectory ?? [];
  if (path.length < 2) return;
  ctx.save();
  ctx.lineWidth = Math.max(1.5, box.cell * 0.12);
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(12, 16, 22, 0.55)";
  ctx.beginPath();
  ctx.moveTo(box.cx(path[0][0]), box.cy(path[0][1]));
  for (const [x, y] of path.slice(1)) ctx.lineTo(box.cx(x), box.cy(y));
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 235, 130, 0.95)";
  ctx.lineWidth = Math.max(1, box.cell * 0.07);
  ctx.stroke();
  ctx.restore();
}

function paintMarkers(ctx: CanvasRenderingContext2D, env: GridEnv, box: GridBox): void {
  ctx.save();
  for (const [x, y] of env.obstacles) {
    ctx.fillStyle = "rgba(38, 44, 52, 0.92)";
    ctx.fillRect(box.px(x), box.py(y), box.cell, box.cell);
  }
  for (const [x, y] of env.rewards) {
    ctx.beginPath();
    ctx.arc(box.cx(x), box.cy(y), box.cell * 0.26, 0, Math.PI * 2);
    ctx.strokeStyle = "#f5a524";
    ctx.lineWidth = Math.max(2, box.cell * 0.12);
    ctx.stroke();
  }
  const [sx, sy] = env.start;
  ctx.beginPath();
  ctx.arc(box.cx(sx), box.cy(sy), box.cell * 0.26, 0, Math.PI * 2);
  ctx.fillStyle = "#39b54a";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 1.4;
  ctx.stroke();
  const [gx, gy] = env.goal;
  const cx = box.cx(gx);
  const cy = box.cy(gy);
  const r = box.cell * 0.32;
  ctx.beginPath();
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx + r, cy);
  ctx.lineTo(cx, cy + r);
  ctx.lineTo(cx - r, cy);
  ctx.closePath();
  ctx.fillStyle = "#ffd666";
  ctx.fill();
  ctx.strokeStyle = "rgba(12, 16, 22, 0.45)";
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
}

/** RL 网格世界画布（docs/02 §13.6）：V(s) 热力 + 策略箭头 + 环境标记 + 轨迹折线。 */
export function GridWorldCanvas() {
  const { t } = useTranslation();
  const spec = useAlgoStore((state) => state.spec);
  const { runId, meta, payload, stale } = useAlgoSnapshot("agent", "grid");
  const env = spec?.kind === "rl" ? spec.env ?? null : null;

  const canvasRef = useCanvasPainter(
    (ctx, width, height) => {
      if (!env) return;
      const box = fitBox(env, width, height);
      if (payload) paintValues(ctx, payload, env, box);
      else {
        ctx.fillStyle = "rgba(147, 161, 177, 0.12)";
        ctx.fillRect(box.px(0), box.py(0), box.width, box.height);
      }
      paintGridLines(ctx, env, box);
      if (payload) paintPolicy(ctx, payload, env, box);
      paintMarkers(ctx, env, box);
      if (payload) paintTrajectory(ctx, payload, box);
    },
    [env, payload],
  );

  if (!spec) return null;
  const info = payload?.meta;
  const badges: string[] = [];
  if (meta) badges.push(`step ${meta.step}`);
  if (info?.episode !== undefined) badges.push(t("algo.grid.episode", { value: info.episode }));
  if (info?.epsilon !== undefined) badges.push(t("algo.grid.epsilon", { value: info.epsilon.toFixed(3) }));
  if (info?.success !== undefined) {
    badges.push(t(`algo.grid.success.${info.success ? "yes" : "no"}`));
  }
  if (stale) badges.push(t("common.loading"));

  return (
    <div className="algo-canvas">
      <div className="algo-canvas__toolbar">
        <span className="algo-canvas__spacer" />
        {badges.length > 0 ? <span className="probe__meta">{badges.join(" · ")}</span> : null}
      </div>
      <div className="algo-canvas__stage">
        <canvas className="algo-canvas__bitmap" ref={canvasRef} />
        {!payload ? (
          <div className="algo-canvas__empty">
            {runId ? t("algo.grid.waiting") : t("algo.grid.noRun")}
          </div>
        ) : null}
      </div>
      <div className="algo-canvas__legend">
        <span className="algo-legend">
          <i className="algo-legend__dot" style={{ background: "#39b54a" }} />
          {t("algo.grid.start")}
        </span>
        <span className="algo-legend">
          <i className="algo-legend__diamond" />
          {t("algo.grid.goal")}
        </span>
        <span className="algo-legend">
          <i className="algo-legend__ring" style={{ borderColor: "#f5a524" }} />
          {t("algo.grid.reward")}
        </span>
        <span className="algo-canvas__hint">{t("algo.grid.hint")}</span>
      </div>
    </div>
  );
}
