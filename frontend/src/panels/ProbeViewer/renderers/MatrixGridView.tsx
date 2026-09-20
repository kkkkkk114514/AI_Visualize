import { useEffect, useRef } from "react";

import type { DecodedSnapshot, GridLayout, MatrixView } from "../paint";
import { gridLayout, paintMatrix } from "../paint";

interface MatrixGridViewProps {
  matrices: MatrixView[];
  decoded: DecodedSnapshot;
  lut: Uint8ClampedArray;
  /** 每块矩阵的说明（通道号 / 头号） */
  caption?: (index: number) => string;
  /** 行列刻度（注意力矩阵的 token 序号） */
  axes?: boolean;
  /** 每块矩阵的宽高比（= 行数 / 列数） */
  aspect: number;
  /** hover 读数：由调用方拼好文案（含真实值） */
  readout: (index: number, row: number, col: number, real: number) => string;
}

interface Frame {
  views: MatrixView[];
  layout: GridLayout;
  pad: number;
}

/**
 * 矩阵平铺渲染（FeatureGridCanvas / HeatmapCanvas 共用）：
 * 像素画在 canvas 上；hover 高亮框与读数直接写 DOM，不触发 React 重渲染。
 */
export function MatrixGridView({
  matrices,
  decoded,
  lut,
  caption,
  axes,
  aspect,
  readout,
}: MatrixGridViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const markerRef = useRef<HTMLDivElement | null>(null);
  const readoutRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<Frame | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas || matrices.length === 0) return;

    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const ratio = window.devicePixelRatio || 1;
      const pad = axes ? 20 : caption ? 13 : 2;
      const layout = gridLayout(
        matrices.length,
        Math.max(40, container.clientWidth - pad * 2),
        aspect,
        Math.max(120, container.clientHeight - pad * 2),
      );
      const width = layout.width + pad * 2;
      const height = layout.height + pad * 2;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = "#0b0e13";
      context.fillRect(0, 0, width, height);
      context.font = "9px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      context.textBaseline = "top";

      matrices.forEach((view, index) => {
        const column = index % layout.cols;
        const row = Math.floor(index / layout.cols);
        const x = pad + column * (layout.tileW + layout.gap);
        const y = pad + row * (layout.tileH + layout.gap);
        if (caption) {
          context.fillStyle = "#93a1b1";
          context.fillText(caption(index), x, y - 11);
        }
        paintMatrix(context, view, decoded, lut, {
          x,
          y,
          width: layout.tileW,
          height: layout.tileH,
        });
        context.strokeStyle = "#212a36";
        context.lineWidth = 1;
        context.strokeRect(x + 0.5, y + 0.5, layout.tileW - 1, layout.tileH - 1);
        if (axes) {
          // token 序号按块宽抽稀，避免标签互相压字
          const step = Math.max(1, Math.ceil(view.cols / 8));
          context.fillStyle = "#5d6b7c";
          for (let tick = 0; tick < view.cols; tick += step) {
            const at = x + ((tick + 0.5) / view.cols) * layout.tileW;
            context.fillText(String(tick), at - 3, y + layout.tileH + 2);
          }
        }
      });
      frameRef.current = { views: matrices, layout, pad };
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [matrices, decoded, lut, caption, axes, aspect]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const marker = markerRef.current;
    const readoutEl = readoutRef.current;
    if (!canvas || !marker || !readoutEl) return;

    const clear = () => {
      marker.hidden = true;
      readoutEl.hidden = true;
    };

    const onMove = (event: MouseEvent) => {
      const frame = frameRef.current;
      if (!frame) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const { layout, pad, views } = frame;
      const column = Math.floor((x - pad) / (layout.tileW + layout.gap));
      const row = Math.floor((y - pad) / (layout.tileH + layout.gap));
      if (column < 0 || column >= layout.cols || row < 0) return clear();
      const view = views[row * layout.cols + column];
      if (!view) return clear();

      const tileX = pad + column * (layout.tileW + layout.gap);
      const tileY = pad + row * (layout.tileH + layout.gap);
      const innerX = x - tileX;
      const innerY = y - tileY;
      if (innerX < 0 || innerX >= layout.tileW || innerY < 0 || innerY >= layout.tileH) {
        return clear();
      }

      const cellCol = Math.min(view.cols - 1, Math.floor((innerX / layout.tileW) * view.cols));
      const cellRow = Math.min(view.rows - 1, Math.floor((innerY / layout.tileH) * view.rows));
      const quantized = Number(decoded.values[view.offset + cellRow * view.cols + cellCol] ?? 0);
      const real =
        decoded.dtype === "uint8"
          ? decoded.min + (quantized / 255) * (decoded.max - decoded.min)
          : quantized;

      const cellW = layout.tileW / view.cols;
      const cellH = layout.tileH / view.rows;
      marker.hidden = false;
      marker.style.transform = `translate(${tileX + cellCol * cellW}px, ${tileY + cellRow * cellH}px)`;
      marker.style.width = `${Math.max(2, cellW)}px`;
      marker.style.height = `${Math.max(2, cellH)}px`;
      const index = row * layout.cols + column;
      readoutEl.hidden = false;
      readoutEl.textContent = readout(index, cellRow, cellCol, real);
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", clear);
    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", clear);
    };
  }, [decoded, readout]);

  return (
    <div className="probe-render" ref={containerRef}>
      <canvas className="probe-render__canvas" ref={canvasRef} />
      <div className="probe-render__marker" ref={markerRef} hidden />
      <div className="probe-render__readout" ref={readoutRef} hidden />
    </div>
  );
}
