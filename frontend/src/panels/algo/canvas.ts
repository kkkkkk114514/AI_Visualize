import { useEffect, useRef } from "react";
import type { DependencyList, RefObject } from "react";

/**
 * 画布绘制钩子：按 devicePixelRatio 设置位图尺寸、清除上一帧后调用 `paint`
 * （坐标为 CSS 像素），并在容器尺寸变化时重画。
 * `deps` 必须包含 `paint` 读取的一切：paint 不参与依赖比较（每次渲染都是新函数）。
 */
export function useCanvasPainter(
  paint: (ctx: CanvasRenderingContext2D, width: number, height: number) => void,
  deps: DependencyList,
): RefObject<HTMLCanvasElement | null> {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const paintRef = useRef(paint);
  paintRef.current = paint;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width === 0 || height === 0) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      paintRef.current(ctx, width, height);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}
