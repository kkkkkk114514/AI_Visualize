import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import uPlot from "uplot";

import "uplot/dist/uPlot.min.css";

import type { MetricName } from "../../api/types";
import { METRIC_NAMES, useRunStore, metricsBuffer } from "../../stores/runStore";

const COLORS: Record<MetricName, string> = {
  loss: "#4c9aff",
  acc: "#3fb950",
  val_loss: "#8ab4f8",
  val_acc: "#7ee787",
  lr: "#d29922",
  grad_norm: "#bc8cff",
  throughput: "#39c5cf",
  vram_mb: "#f85149",
};

const MAIN_SERIES: MetricName[] = ["loss", "val_loss", "acc", "val_acc"];
const OPTIONAL_METRICS: MetricName[] = ["lr", "grad_norm", "throughput"];

function axisStyle(): uPlot.Axis {
  return {
    stroke: "#93a1b1",
    grid: { stroke: "#212a36", width: 1 },
    ticks: { stroke: "#212a36", width: 1 },
    font: "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
  };
}

function baseOptions(): Omit<uPlot.Options, "series"> {
  return {
    width: 320,
    height: 180,
    padding: [10, 14, 0, 0],
    cursor: { sync: { key: "metrics-sync" }, drag: { x: true, y: false } },
    legend: { show: true, live: true },
    scales: { x: { time: false }, y: { auto: true } },
    axes: [
      { ...axisStyle(), label: undefined, size: 30 },
      { ...axisStyle(), size: 48 },
    ],
  };
}

/** 图例数值：uPlot 默认按 Intl（最多 3 位小数）格式化，lr=1e-4 会显示成 0，小量级改给 2 位有效数字。 */
function legendValue(value: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude > 0 && magnitude < 0.001) return String(Number(value.toPrecision(2)));
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function seriesValue(_self: uPlot, value: number, _seriesIdx: number, idx: number | null): string {
  return idx === null ? "--" : legendValue(value);
}

/** 时间游标 ref：插件在 draw 时读取，绕开 uPlot 实例重建；null = 跟随最新。 */
type CursorRef = { current: number | null };

/** ≤ step 的最后一个下标（与探针「游标处最近快照」同口径）；越界取最近端点。 */
function indexAtOrBefore(xs: number[], step: number): number {
  if (xs.length === 0) return -1;
  if (xs[0] > step) return 0;
  let found = 0;
  for (let i = 0; i < xs.length; i += 1) {
    if (xs[i] <= step) found = i;
    else break;
  }
  return found;
}

/** 无光标时图例跟随读数：跟随最新时用末点，定位到游标后用游标处的点。 */
function liveLegendPlugin(cursorRef: CursorRef): uPlot.Plugin {
  return {
    hooks: {
      setLegend: (u: uPlot) => {
        const xs = u.data[0] as number[] | undefined;
        if (u.cursor.idx !== null || !xs || xs.length === 0) return;
        const step = cursorRef.current;
        u.setLegend({ idx: step === null ? xs.length - 1 : indexAtOrBefore(xs, step) }, false);
      },
    },
  };
}

/** 时间游标：在曲线上画一条持久竖线（draw hook，不受鼠标移出影响）。 */
function timeCursorPlugin(cursorRef: CursorRef): uPlot.Plugin {
  return {
    hooks: {
      draw: (u: uPlot) => {
        const step = cursorRef.current;
        const xs = u.data[0] as number[] | undefined;
        if (step === null || !xs || xs.length === 0) return;
        const x = Math.round(u.valToPos(xs[indexAtOrBefore(xs, step)], "x", true));
        if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) return;
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = "#d29922";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, u.bbox.top);
        ctx.lineTo(x, u.bbox.top + u.bbox.height);
        ctx.stroke();
        ctx.restore();
      },
    },
  };
}

/** 点击曲线定位游标：用 4px 位移守卫区分「点击」与 uPlot 的拖拽框选，命中最近的真实 step。 */
function enableClickToPin(plot: uPlot, onPin: (step: number) => void): () => void {
  let origin: { x: number; y: number } | null = null;
  const onDown = (event: MouseEvent) => {
    origin = { x: event.clientX, y: event.clientY };
  };
  const onClick = (event: MouseEvent) => {
    if (event.button !== 0) return;
    const start = origin;
    origin = null;
    if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
    const xs = plot.data[0] as number[] | undefined;
    if (!xs || xs.length === 0) return;
    const rect = plot.over.getBoundingClientRect();
    const value = plot.posToVal(event.clientX - rect.left, "x");
    let nearest = xs[0];
    for (const step of xs) {
      if (Math.abs(step - value) < Math.abs(nearest - value)) nearest = step;
    }
    onPin(nearest);
  };
  plot.over.addEventListener("mousedown", onDown);
  plot.over.addEventListener("click", onClick);
  return () => {
    plot.over.removeEventListener("mousedown", onDown);
    plot.over.removeEventListener("click", onClick);
  };
}

/** 图例项点击即隐藏 / 显示对应曲线（uPlot 原生行为）；补上指针与提示，否则入口完全不可见。 */
function markLegendToggles(plot: uPlot, hint: string): void {
  plot.root.querySelectorAll<HTMLElement>(".u-legend .u-series").forEach((row, index) => {
    if (index === 0) return; // 第 0 行是 x 轴（步）读数，没有曲线可切换
    row.classList.add("u-toggle");
    row.querySelector<HTMLElement>("th")?.setAttribute("title", hint);
  });
}

/** 每个 epoch 末的 val_* 点位置画竖线，标出 epoch 边界。 */
function epochMarkersPlugin(valColumn: number): uPlot.Plugin {
  return {
    hooks: {
      draw: (u: uPlot) => {
        const data = u.data as uPlot.AlignedData;
        const values = data[valColumn] as (number | null)[] | undefined;
        if (!values) return;
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = "rgba(147, 161, 177, 0.35)";
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
        ctx.fillStyle = "#5d6b7c";
        let epoch = 0;
        values.forEach((value, index) => {
          if (value === null || value === undefined) return;
          epoch += 1;
          const x = Math.round(u.valToPos(data[0][index], "x", true));
          if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) return;
          ctx.beginPath();
          ctx.moveTo(x, u.bbox.top);
          ctx.lineTo(x, u.bbox.top + u.bbox.height);
          ctx.stroke();
          ctx.fillText(`E${epoch}`, x + 3, u.bbox.top + 10);
        });
        ctx.restore();
      },
    },
  };
}

function buildMainOptions(container: HTMLElement, stepLabel: string, cursorRef: CursorRef): uPlot.Options {
  const names = MAIN_SERIES;
  const valColumn = names.indexOf("val_loss") + 1;
  return {
    ...baseOptions(),
    width: container.clientWidth || 320,
    series: [
      { label: stepLabel, value: seriesValue },
      {
        label: names[0],
        stroke: COLORS.loss,
        width: 1.8,
        points: { show: false },
        value: seriesValue,
      },
      {
        label: names[1],
        stroke: COLORS.val_loss,
        width: 1.4,
        dash: [5, 4],
        spanGaps: true,
        points: { show: true, size: 5 },
        value: seriesValue,
      },
      {
        label: names[2],
        stroke: COLORS.acc,
        width: 1.6,
        scale: "acc",
        points: { show: false },
        value: seriesValue,
      },
      {
        label: names[3],
        stroke: COLORS.val_acc,
        width: 1.4,
        dash: [5, 4],
        spanGaps: true,
        scale: "acc",
        points: { show: true, size: 5 },
        value: seriesValue,
      },
    ],
    scales: {
      x: { time: false },
      y: { auto: true },
      acc: { auto: true },
    },
    axes: [
      { ...axisStyle(), size: 30 },
      { ...axisStyle(), size: 48 },
      { ...axisStyle(), scale: "acc", side: 1, size: 40 },
    ],
    plugins: [liveLegendPlugin(cursorRef), timeCursorPlugin(cursorRef), epochMarkersPlugin(valColumn)],
  };
}

function buildExtraOptions(
  name: MetricName,
  container: HTMLElement,
  stepLabel: string,
  cursorRef: CursorRef,
): uPlot.Options {
  return {
    ...baseOptions(),
    width: container.clientWidth || 320,
    height: 96,
    series: [
      { label: stepLabel, value: seriesValue },
      { label: name, stroke: COLORS[name], width: 1.6, points: { show: false }, value: seriesValue },
    ],
    plugins: [liveLegendPlugin(cursorRef), timeCursorPlugin(cursorRef)],
  };
}

export function MetricsCharts() {
  const { t } = useTranslation();
  const stepLabel = t("run.metrics.step");
  const legendHint = t("run.chart.legendToggle");
  // 数据源标注看 metricsRunId（缓冲当前承载的 run）：已挂 run A + 回看 run B 时标 B
  const chartRunId = useRunStore((state) => state.metricsRunId);
  const chartName = useRunStore((state) => {
    const runId = state.metricsRunId;
    if (!runId) return null;
    if (state.current?.id === runId) return state.current.name;
    return state.history.find((run) => run.id === runId)?.name ?? null;
  });
  const cursorStep = useRunStore((state) => state.cursorStep);
  const setCursorStep = useRunStore((state) => state.setCursorStep);
  const [optional, setOptional] = useState<MetricName[]>(["lr"]);
  const [empty, setEmpty] = useState(true);
  const [range, setRange] = useState<{ min: number; max: number }>({ min: 0, max: 0 });

  const cursorRef = useRef<number | null>(cursorStep);
  const mainHost = useRef<HTMLDivElement | null>(null);
  const mainPlot = useRef<uPlot | null>(null);
  const extraHosts = useRef(new Map<MetricName, HTMLDivElement>());
  const extraPlots = useRef(new Map<MetricName, uPlot>());

  // 游标变化：重画各图（插件读 cursorRef）——不重建实例
  useEffect(() => {
    cursorRef.current = cursorStep;
    mainPlot.current?.redraw();
    extraPlots.current.forEach((plot) => plot.redraw());
  }, [cursorStep]);

  useEffect(() => {
    const host = mainHost.current;
    if (!host) return;
    const plot = new uPlot(
      buildMainOptions(host, stepLabel, cursorRef),
      metricsBuffer.alignedData(MAIN_SERIES),
      host,
    );
    markLegendToggles(plot, legendHint);
    mainPlot.current = plot;
    const detachPin = enableClickToPin(plot, setCursorStep);
    const off = metricsBuffer.subscribe(() => {
      plot.setData(metricsBuffer.alignedData(MAIN_SERIES));
    });
    const onResize = () => {
      if (host.clientWidth > 0) plot.setSize({ width: host.clientWidth, height: host.clientHeight - 4 });
    };
    onResize();
    const observer = new ResizeObserver(onResize);
    observer.observe(host);
    return () => {
      off();
      observer.disconnect();
      detachPin();
      plot.destroy();
      mainPlot.current = null;
    };
  }, [legendHint, stepLabel, setCursorStep]);

  useEffect(() => {
    const created: uPlot[] = [];
    const detachers: (() => void)[] = [];
    // 图例的 x 轴标签来自 i18n：语言切换时已有小图整体重建
    for (const [name, plot] of [...extraPlots.current.entries()]) {
      plot.destroy();
      extraPlots.current.delete(name);
    }
    for (const name of optional) {
      const host = extraHosts.current.get(name);
      if (!host || extraPlots.current.has(name)) continue;
      const plot = new uPlot(
        buildExtraOptions(name, host, stepLabel, cursorRef),
        metricsBuffer.alignedData([name]),
        host,
      );
      extraPlots.current.set(name, plot);
      created.push(plot);
      detachers.push(enableClickToPin(plot, setCursorStep));
    }
    for (const [name, plot] of [...extraPlots.current.entries()]) {
      if (optional.includes(name)) continue;
      plot.destroy();
      extraPlots.current.delete(name);
    }
    const off = metricsBuffer.subscribe(() => {
      for (const name of optional) {
        extraPlots.current.get(name)?.setData(metricsBuffer.alignedData([name]));
      }
    });

    const observers: ResizeObserver[] = [];
    for (const name of optional) {
      const host = extraHosts.current.get(name);
      if (!host) continue;
      const observer = new ResizeObserver(() => {
        const plot = extraPlots.current.get(name);
        if (plot && host.clientWidth > 0) plot.setSize({ width: host.clientWidth, height: 96 });
      });
      observer.observe(host);
      observers.push(observer);
    }
    return () => {
      off();
      observers.forEach((observer) => observer.disconnect());
      detachers.forEach((detach) => detach());
      created.forEach((plot) => plot.destroy());
      for (const [name, plot] of [...extraPlots.current.entries()]) {
        if (created.includes(plot)) extraPlots.current.delete(name);
      }
    };
  }, [optional, stepLabel, setCursorStep]);

  // 缓冲区变化：同步「是否为空」与时间轴范围（range 不变时不触发渲染）
  useEffect(() => {
    const sync = () => {
      setEmpty(metricsBuffer.length === 0);
      setRange((prev) => {
        const min = metricsBuffer.firstStep() ?? 0;
        const max = metricsBuffer.lastStep() ?? 0;
        return prev.min === min && prev.max === max ? prev : { min, max };
      });
    };
    const off = metricsBuffer.subscribe(sync);
    sync();
    return off;
  }, [chartRunId]);

  const toggles = useMemo(
    () =>
      OPTIONAL_METRICS.map((name) => ({
        name,
        active: optional.includes(name),
        disabled: !METRIC_NAMES.includes(name),
      })),
    [optional],
  );

  return (
    <div className="charts">
      <div className="charts__toolbar">
        <span className="charts__source">
          {chartRunId
            ? chartName
              ? t("run.chart.source", { name: chartName, run: chartRunId })
              : t("run.chart.sourceId", { run: chartRunId })
            : t("run.chart.noSource")}
        </span>
        <span className="charts__toggles">
          {toggles.map((item) => (
            <label key={item.name} className={`charts__toggle ${item.active ? "charts__toggle--on" : ""}`}>
              <input
                type="checkbox"
                checked={item.active}
                onChange={() =>
                  setOptional((current) =>
                    current.includes(item.name)
                      ? current.filter((name) => name !== item.name)
                      : [...current, item.name],
                  )
                }
              />
              <span>{t(`run.chart.metric.${item.name}`)}</span>
            </label>
          ))}
        </span>
      </div>
      <div className="charts__body">
        <div className="charts__main" ref={mainHost} />
        {optional.map((name) => (
          <div
            key={name}
            className="charts__extra"
            ref={(node) => {
              if (node) extraHosts.current.set(name, node);
              else extraHosts.current.delete(name);
            }}
          />
        ))}
        {empty ? <div className="charts__empty">{t("run.chart.empty")}</div> : null}
      </div>
      <div className="charts__timeline">
        <span className="charts__timeline-label">{t("run.timeline.label")}</span>
        <input
          className="charts__timeline-range"
          type="range"
          min={range.min}
          max={range.max}
          step={1}
          value={cursorStep ?? range.max}
          disabled={range.max <= range.min}
          onChange={(event) => setCursorStep(Number(event.target.value))}
          title={t("run.timeline.hint")}
        />
        <span
          className={`charts__timeline-readout ${cursorStep === null ? "charts__timeline-readout--live" : ""}`}
        >
          {cursorStep === null
            ? t("run.timeline.latest", { step: range.max })
            : t("run.timeline.at", { step: cursorStep })}
        </span>
        <button
          type="button"
          className="btn btn--ghost charts__timeline-follow"
          disabled={cursorStep === null}
          onClick={() => setCursorStep(null)}
          title={t("run.timeline.followHint")}
        >
          {t("run.timeline.follow")}
        </button>
      </div>
    </div>
  );
}
