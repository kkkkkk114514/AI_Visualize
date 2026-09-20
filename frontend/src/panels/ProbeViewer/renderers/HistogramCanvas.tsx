import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import type { SnapshotPayload } from "../../../api/types";
import { decodePayload, formatValue } from "../paint";

interface HistogramCanvasProps {
  payload: SnapshotPayload;
}

const PAD = { left: 34, right: 8, top: 8, bottom: 16 };

/** 直方图：uint32 计数不做反量化，读数给出 bin 区间与计数。 */
export function HistogramCanvas({ payload }: HistogramCanvasProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const markerRef = useRef<HTMLDivElement | null>(null);
  const readoutRef = useRef<HTMLDivElement | null>(null);
  const layoutRef = useRef<{ plotW: number; plotH: number; bins: number } | null>(null);
  const decoded = useMemo(() => decodePayload(payload), [payload]);
  const counts = decoded.values as Uint32Array;

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(80, container.clientWidth);
      const height = Math.max(60, container.clientHeight);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = "#0b0e13";
      context.fillRect(0, 0, width, height);

      const plotW = width - PAD.left - PAD.right;
      const plotH = height - PAD.top - PAD.bottom;
      const bins = counts.length;
      const max = Math.max(1, ...Array.from(counts));
      const barW = plotW / bins;
      context.strokeStyle = "#212a36";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(PAD.left + 0.5, PAD.top);
      context.lineTo(PAD.left + 0.5, PAD.top + plotH + 0.5);
      context.lineTo(PAD.left + plotW, PAD.top + plotH + 0.5);
      context.stroke();

      context.fillStyle = "#4c9aff";
      for (let index = 0; index < bins; index += 1) {
        const barH = (Number(counts[index]) / max) * (plotH - 2);
        if (barH <= 0) continue;
        context.fillRect(
          PAD.left + index * barW + 1,
          PAD.top + plotH - barH,
          Math.max(1, barW - 2),
          barH,
        );
      }

      context.font = "9px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      context.fillStyle = "#93a1b1";
      context.textBaseline = "top";
      context.fillText(formatValue(decoded.max), PAD.left, PAD.top + plotH + 3);
      context.fillText(formatValue(decoded.min), PAD.left + plotW - 34, PAD.top + plotH + 3);
      context.fillText(String(max), 4, PAD.top - 2);
      layoutRef.current = { plotW, plotH, bins };
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [counts, decoded]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const marker = markerRef.current;
    const readout = readoutRef.current;
    if (!canvas || !marker || !readout) return;

    const clear = () => {
      marker.hidden = true;
      readout.hidden = true;
    };

    const onMove = (event: MouseEvent) => {
      const frame = layoutRef.current;
      if (!frame) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left - PAD.left;
      const y = event.clientY - rect.top;
      const index = Math.floor((x / frame.plotW) * frame.bins);
      if (index < 0 || index >= frame.bins || y < PAD.top || y > PAD.top + frame.plotH) {
        return clear();
      }
      const barW = frame.plotW / frame.bins;
      const span = decoded.max - decoded.min;
      const low = decoded.min + (index / frame.bins) * span;
      const high = decoded.min + ((index + 1) / frame.bins) * span;
      marker.hidden = false;
      marker.style.transform = `translate(${PAD.left + index * barW}px, ${PAD.top}px)`;
      marker.style.width = `${Math.max(2, barW)}px`;
      marker.style.height = `${frame.plotH}px`;
      readout.hidden = false;
      readout.textContent = t("run.probe.readout.bin", {
        bin: index,
        count: Number(counts[index]),
        low: formatValue(low),
        high: formatValue(high),
      });
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", clear);
    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", clear);
    };
  }, [counts, decoded, t]);

  return (
    <div className="probe-render" ref={containerRef}>
      <canvas className="probe-render__canvas" ref={canvasRef} />
      <div className="probe-render__marker" ref={markerRef} hidden />
      <div className="probe-render__readout" ref={readoutRef} hidden />
    </div>
  );
}
