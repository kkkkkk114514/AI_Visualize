import { create } from "zustand";

import { downloadDataset, listDatasets } from "../api/datasets";
import { ApiError } from "../api/rest";
import { controlRun, deleteRun, fetchRunMetrics, getRun, listRuns, startRun } from "../api/runs";
import type {
  DatasetInfo,
  MetricName,
  MetricSeries,
  RunStatus,
  RunSummary,
  ServerEvent,
} from "../api/types";
import { wsClient } from "../api/ws";

export const METRIC_NAMES: MetricName[] = [
  "loss",
  "acc",
  "val_loss",
  "val_acc",
  "lr",
  "grad_norm",
  "throughput",
  "vram_mb",
];

/** 环形缓冲窗口：最多保留最近 N 个 step 点（docs/02 §9.2 实时曲线） */
export const METRICS_WINDOW = 4000;
const TRIM_CHUNK = 512;

export interface MetricPoint {
  step: number;
  epoch?: number | null;
  values: Partial<Record<MetricName, number>>;
}

type Column = (number | null)[];

/**
 * 指标环形缓冲：按 step 对齐成列，供 uPlot 直接读取。
 * 数据只在 WS 线程写入、由图表订阅者消费，不进入 React 渲染。
 */
class MetricsBuffer {
  private steps: number[] = [];
  private readonly columns = new Map<MetricName, Column>();
  private readonly listeners = new Set<() => void>();

  constructor() {
    this.reset();
  }

  reset(): void {
    this.steps = [];
    this.columns.clear();
    for (const name of METRIC_NAMES) this.columns.set(name, []);
  }

  get length(): number {
    return this.steps.length;
  }

  lastStep(): number | null {
    return this.steps.length === 0 ? null : this.steps[this.steps.length - 1];
  }

  push(points: MetricPoint[]): void {
    let changed = false;
    for (const point of points) {
      const step = Number(point.step);
      if (!Number.isFinite(step)) continue;
      const last = this.lastStep();
      if (last !== null && step < last) continue;
      if (last !== null && step === last) {
        this.merge(this.steps.length - 1, point.values);
        changed = true;
        continue;
      }
      this.append(step, point.values);
      changed = true;
    }
    if (!changed) return;
    if (this.steps.length > METRICS_WINDOW + TRIM_CHUNK) {
      const drop = this.steps.length - METRICS_WINDOW;
      this.steps.splice(0, drop);
      for (const column of this.columns.values()) column.splice(0, drop);
    }
    this.listeners.forEach((listener) => listener());
  }

  /** 用 REST 拉回的历史序列重建缓冲（回放 & 刷新后补齐）。 */
  loadSeries(series: Partial<Record<MetricName, MetricSeries>>): void {
    this.reset();
    const union = new Set<number>();
    for (const entry of Object.values(series)) {
      for (const step of entry.steps) union.add(step);
    }
    const sorted = [...union].sort((a, b) => a - b).slice(-METRICS_WINDOW);
    const index = new Map<number, number>();
    sorted.forEach((step, position) => {
      this.steps.push(step);
      index.set(step, position);
      for (const column of this.columns.values()) column.push(null);
    });
    for (const name of METRIC_NAMES) {
      const entry = series[name];
      const column = this.columns.get(name);
      if (!entry || !column) continue;
      entry.steps.forEach((step, position) => {
        const row = index.get(step);
        if (row === undefined) return;
        column[row] = entry.values[position];
      });
    }
    this.listeners.forEach((listener) => listener());
  }

  /** uPlot 对齐数据：第一行是 x（step），之后按 names 顺序各一行。 */
  alignedData(names: MetricName[]): [number[], ...Column[]] {
    return [this.steps, ...names.map((name) => this.columns.get(name) ?? [])];
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private append(step: number, values: Partial<Record<MetricName, number>>): void {
    this.steps.push(step);
    for (const name of METRIC_NAMES) {
      const value = values[name];
      this.columns.get(name)!.push(typeof value === "number" ? value : null);
    }
  }

  private merge(row: number, values: Partial<Record<MetricName, number>>): void {
    for (const name of METRIC_NAMES) {
      const value = values[name];
      if (typeof value !== "number") continue;
      this.columns.get(name)![row] = value;
    }
  }
}

export const metricsBuffer = new MetricsBuffer();

export interface RunConfig {
  optimizer: "adam" | "adamw" | "sgd";
  lr: number;
  batch_size: number;
  epochs: number;
  loss: "cross_entropy" | "mse";
  grad_clip: number;
  train_size: number;
  val_size: number;
}

export const DEFAULT_RUN_CONFIG: RunConfig = {
  optimizer: "adam",
  lr: 0.001,
  batch_size: 64,
  epochs: 3,
  loss: "cross_entropy",
  grad_clip: 5.0,
  train_size: 20000,
  val_size: 5000,
};

export interface LogEntry {
  id: number;
  level: string;
  key: string;
  args: Record<string, unknown>;
  at: number;
}

export interface RunErrorState {
  code: string;
  messageKey: string;
  args: Record<string, unknown>;
  detail?: string;
}

export interface DatasetProgress {
  phase: string;
  progress: number;
  file?: string;
  detail?: string;
}

interface RunState {
  datasets: DatasetInfo[] | null;
  datasetsError: string | null;
  datasetId: string | null;
  datasetProgress: Record<string, DatasetProgress>;
  config: RunConfig;
  current: RunSummary | null;
  replay: RunSummary | null;
  status: RunStatus;
  step: number;
  epoch: number;
  plannedSteps: number | null;
  device: string | null;
  elapsedS: number;
  bestMetric: number | null;
  runError: RunErrorState | null;
  startError: RunErrorState | null;
  starting: boolean;
  logs: LogEntry[];
  history: RunSummary[];
  historyTotal: number;
  historyError: string | null;

  loadDatasets: () => Promise<void>;
  downloadDataset: (datasetId: string) => Promise<void>;
  setDataset: (datasetId: string) => void;
  setConfig: (patch: Partial<RunConfig>) => void;
  applyConfigDefaults: (defaults: Record<string, unknown> | null | undefined) => void;
  start: (graph: unknown, modelId?: string) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  applyLr: (lr: number) => Promise<void>;
  applyBatchSize: (size: number) => Promise<void>;
  loadHistory: () => Promise<void>;
  selectRun: (runId: string | null) => Promise<void>;
  removeRun: (runId: string) => Promise<void>;
  attachRun: (runId: string) => Promise<void>;
  refresh: () => Promise<void>;
  tickElapsed: () => void;
  handleEvent: (event: ServerEvent) => void;
}

let logSeq = 1;
let elapsedAnchor = 0;
let elapsedBase = 0;

const TERMINAL: RunStatus[] = ["finished", "stopped", "failed", "interrupted"];
const ACTIVE: RunStatus[] = ["created", "running", "paused"];

export function isActiveStatus(status: RunStatus): boolean {
  return ACTIVE.includes(status);
}

export function isTerminalStatus(status: RunStatus): boolean {
  return TERMINAL.includes(status);
}

function errorState(error: unknown, fallbackKey = "errors.run.unknown"): RunErrorState {
  if (error instanceof ApiError) {
    return {
      code: error.payload?.code ?? `http_${error.status}`,
      messageKey: error.payload?.message_key ?? fallbackKey,
      args: error.payload?.args ?? {},
      detail: error.payload?.detail ?? error.message,
    };
  }
  return {
    code: "client_error",
    messageKey: fallbackKey,
    args: {},
    detail: error instanceof Error ? error.message : String(error),
  };
}

export const useRunStore = create<RunState>((set, get) => {
  const pushLog = (entry: LogEntry) => {
    set((state) => ({ logs: [...state.logs, entry].slice(-200) }));
  };

  const applySummary = (run: RunSummary) => {
    elapsedBase = run.elapsed_s ?? 0;
    elapsedAnchor = performance.now();
    set({
      current: run,
      status: run.status,
      step: run.step ?? run.total_steps ?? 0,
      epoch: run.epoch ?? 0,
      plannedSteps: run.planned_steps ?? null,
      device: run.device,
      elapsedS: run.elapsed_s ?? 0,
      bestMetric: run.best_metric,
    });
  };

  return {
    datasets: null,
    datasetsError: null,
    datasetId: null,
    datasetProgress: {},
    config: { ...DEFAULT_RUN_CONFIG },
    current: null,
    replay: null,
    status: "idle",
    step: 0,
    epoch: 0,
    plannedSteps: null,
    device: null,
    elapsedS: 0,
    bestMetric: null,
    runError: null,
    startError: null,
    starting: false,
    logs: [],
    history: [],
    historyTotal: 0,
    historyError: null,

    loadDatasets: async () => {
      try {
        const datasets = await listDatasets();
        const cached = datasets.find((item) => item.cached);
        set((state) => ({
          datasets,
          datasetsError: null,
          datasetId: state.datasetId ?? cached?.id ?? datasets[0]?.id ?? null,
        }));
      } catch (error) {
        set({ datasetsError: error instanceof Error ? error.message : String(error) });
      }
    },

    downloadDataset: async (datasetId) => {
      try {
        await downloadDataset(datasetId);
        set((state) => ({
          datasetProgress: {
            ...state.datasetProgress,
            [datasetId]: { phase: "download", progress: 0 },
          },
        }));
      } catch (error) {
        set({ datasetsError: error instanceof Error ? error.message : String(error) });
      }
    },

    setDataset: (datasetId) => set({ datasetId }),

    setConfig: (patch) => set((state) => ({ config: { ...state.config, ...patch } })),

    applyConfigDefaults: (defaults) => {
      if (!defaults) return;
      const patch: Partial<RunConfig> = {};
      for (const key of Object.keys(DEFAULT_RUN_CONFIG) as (keyof RunConfig)[]) {
        const value = defaults[key];
        if (value !== undefined && value !== null) {
          patch[key] = value as never;
        }
      }
      set((state) => ({ config: { ...state.config, ...patch } }));
    },

    start: async (graph, modelId) => {
      const state = get();
      if (state.starting) return;
      const datasetId = state.datasetId;
      if (!datasetId) {
        set({ startError: { code: "no_dataset", messageKey: "errors.run.needDataset", args: {} } });
        return;
      }
      set({ starting: true, startError: null, runError: null, logs: [], replay: null });
      try {
        const run = await startRun({
          graph: modelId ? undefined : graph,
          model_id: modelId,
          dataset_id: datasetId,
          hyperparams: state.config,
        });
        metricsBuffer.reset();
        wsClient.subscribe(run.id);
        applySummary(run);
        set({ starting: false });
        pushLog({ id: logSeq++, level: "info", key: "log.run.queued", args: {}, at: Date.now() });
        void get().loadHistory();
      } catch (error) {
        const payload = error instanceof ApiError ? error.payload : null;
        set({ starting: false, startError: errorState(error, "errors.run.startFailed") });
        // 已有活动 run（多标签页/未刷新）：直接切到它，而不是留一个死错误
        if (payload?.code === "errors.run.busy") {
          const busyId = payload.args?.run_id;
          if (typeof busyId === "string" && busyId) {
            await get().attachRun(busyId);
            set({ startError: null });
          }
        }
      }
    },

    pause: async () => {
      const runId = get().current?.id;
      if (!runId) return;
      try {
        await controlRun(runId, "pause");
      } catch (error) {
        set({ runError: errorState(error, "errors.run.controlFailed") });
      }
    },

    resume: async () => {
      const runId = get().current?.id;
      if (!runId) return;
      try {
        await controlRun(runId, "resume");
      } catch (error) {
        set({ runError: errorState(error, "errors.run.controlFailed") });
      }
    },

    stop: async () => {
      const runId = get().current?.id;
      if (!runId) return;
      try {
        await controlRun(runId, "stop");
      } catch (error) {
        set({ runError: errorState(error, "errors.run.controlFailed") });
      }
    },

    applyLr: async (lr) => {
      set((state) => ({ config: { ...state.config, lr } }));
      const run = get().current;
      if (!run || !isActiveStatus(run.status)) return;
      try {
        await controlRun(run.id, "set_lr", lr);
        pushLog({ id: logSeq++, level: "info", key: "log.run.lrApplied", args: { lr }, at: Date.now() });
      } catch (error) {
        set({ runError: errorState(error, "errors.run.controlFailed") });
      }
    },

    applyBatchSize: async (size) => {
      set((state) => ({ config: { ...state.config, batch_size: size } }));
      const run = get().current;
      if (!run || !isActiveStatus(run.status)) return;
      try {
        await controlRun(run.id, "set_batch_size", size);
        pushLog({
          id: logSeq++,
          level: "info",
          key: "log.run.batchApplied",
          args: { batch_size: size },
          at: Date.now(),
        });
      } catch (error) {
        set({ runError: errorState(error, "errors.run.controlFailed") });
      }
    },

    loadHistory: async () => {
      try {
        const payload = await listRuns(50, 0);
        set({ history: payload.runs, historyTotal: payload.total, historyError: null });
      } catch (error) {
        set({ historyError: error instanceof Error ? error.message : String(error) });
      }
    },

    selectRun: async (runId) => {
      if (runId === null) {
        set({ replay: null });
        metricsBuffer.reset();
        return;
      }
      const known = get().history.find((item) => item.id === runId) ?? null;
      if (known && isActiveStatus(known.status)) {
        await get().attachRun(runId);
        return;
      }
      set({ replay: known });
      try {
        const [detail, metrics] = await Promise.all([getRun(runId), fetchRunMetrics(runId)]);
        set({ replay: detail });
        metricsBuffer.loadSeries(metrics.series);
      } catch (error) {
        set({ runError: errorState(error, "errors.run.loadFailed") });
      }
    },

    removeRun: async (runId) => {
      try {
        await deleteRun(runId);
        set((state) => ({
          history: state.history.filter((item) => item.id !== runId),
          historyTotal: Math.max(0, state.historyTotal - 1),
          replay: state.replay?.id === runId ? null : state.replay,
        }));
      } catch (error) {
        set({ runError: errorState(error, "errors.run.deleteFailed") });
      }
    },

    attachRun: async (runId) => {
      try {
        const detail = await getRun(runId);
        wsClient.subscribe(runId);
        applySummary(detail);
        set({ replay: null, runError: null, logs: [] });
        const metrics = await fetchRunMetrics(runId);
        metricsBuffer.loadSeries(metrics.series);
      } catch (error) {
        set({ runError: errorState(error, "errors.run.loadFailed") });
      }
    },

    refresh: async () => {
      await get().loadDatasets();
      await get().loadHistory();
      const history = get().history;
      const active = history.find((item) => isActiveStatus(item.status));
      if (active) {
        await get().attachRun(active.id);
        return;
      }
      if (get().current) return;
      const latest = history[0];
      if (latest) await get().selectRun(latest.id);
    },

    tickElapsed: () => {
      const status = get().status;
      if (!ACTIVE.includes(status)) return;
      set({ elapsedS: elapsedBase + (performance.now() - elapsedAnchor) / 1000 });
    },

    handleEvent: (event) => {
      const state = get();
      const runId = typeof event.run_id === "string" ? event.run_id : null;

      if (event.type === "run.status" && runId && state.current?.id === runId) {
        const status = String(event.status ?? state.current.status) as RunStatus;
        const step = Number(event.step ?? state.step);
        const elapsedS = typeof event.elapsed_s === "number" ? event.elapsed_s : state.elapsedS;
        const bestMetric =
          event.best_metric === undefined || event.best_metric === null
            ? state.bestMetric
            : Number(event.best_metric);
        if (typeof event.elapsed_s === "number") {
          elapsedBase = event.elapsed_s;
          elapsedAnchor = performance.now();
        }
        set({
          status,
          step,
          epoch: Number(event.epoch ?? state.epoch),
          plannedSteps:
            event.planned_steps === undefined || event.planned_steps === null
              ? state.plannedSteps
              : Number(event.planned_steps),
          device: typeof event.device === "string" && event.device ? event.device : state.device,
          bestMetric,
          elapsedS,
          current: { ...state.current, status, step, elapsed_s: elapsedS, best_metric: bestMetric },
          // 历史列表里的同一 run 同步到最新进度，避免行内还停在启动时的快照
          history: state.history.map((item) =>
            item.id === runId
              ? {
                  ...item,
                  status,
                  step,
                  total_steps: Math.max(item.total_steps, step),
                  elapsed_s: elapsedS,
                  best_metric: bestMetric,
                }
              : item,
          ),
        });
        if (isTerminalStatus(status)) {
          void get().loadHistory();
        }
        return;
      }

      if (event.type === "metrics" && runId) {
        const owner = state.current?.id === runId || state.replay?.id === runId;
        if (!owner) return;
        const points = Array.isArray(event.points) ? (event.points as MetricPoint[]) : [];
        if (points.length > 0) {
          metricsBuffer.push(points);
          const last = points[points.length - 1];
          set({
            step: Math.max(state.step, Number(last.step ?? 0)),
            epoch: Number(last.epoch ?? state.epoch),
          });
        }
        return;
      }

      if (event.type === "log") {
        pushLog({
          id: logSeq++,
          level: String(event.level ?? "info"),
          key: String(event.key ?? ""),
          args: (event.args as Record<string, unknown>) ?? {},
          at: Date.now(),
        });
        return;
      }

      if (event.type === "error") {
        set({
          runError: {
            code: String(event.code ?? "unknown"),
            messageKey: String(event.message_key ?? "errors.run.failed"),
            args: (event.args as Record<string, unknown>) ?? {},
            detail: typeof event.detail === "string" ? event.detail : undefined,
          },
        });
        return;
      }

      if (event.type === "dataset.progress") {
        const datasetId = typeof event.dataset_id === "string" ? event.dataset_id : null;
        if (!datasetId) return;
        const phase = String(event.phase ?? "download");
        set((current) => ({
          datasetProgress: {
            ...current.datasetProgress,
            [datasetId]: {
              phase,
              progress: Number(event.progress ?? 0),
              file: typeof event.file === "string" ? event.file : undefined,
              detail: typeof event.detail === "string" ? event.detail : undefined,
            },
          },
        }));
        if (phase === "done" || phase === "error") {
          void get().loadDatasets();
        }
        return;
      }

      if (event.type === "run.deleted" && runId) {
        set((current) => ({
          history: current.history.filter((item) => item.id !== runId),
          replay: current.replay?.id === runId ? null : current.replay,
        }));
      }
    },
  };
});

/** 把 WS 事件与重连补齐接到 store；返回取消订阅函数。 */
export function bindRunEvents(): () => void {
  const offEvent = wsClient.onEvent((event) => useRunStore.getState().handleEvent(event));
  const offReconnect = wsClient.onReconnect(() => {
    void useRunStore.getState().refresh();
  });
  return () => {
    offEvent();
    offReconnect();
  };
}
