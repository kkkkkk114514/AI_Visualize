import { create } from "zustand";

import { downloadDataset, listDatasets } from "../api/datasets";
import { ApiError } from "../api/rest";
import {
  clearRunHistory,
  controlRun,
  deleteRun,
  fetchRunMetrics,
  fetchRunStorage,
  getRun,
  listRuns,
  startRun,
} from "../api/runs";
import { clearSnapshotCache, listSnapshots } from "../api/snapshots";
import type {
  DatasetInfo,
  MetricName,
  MetricSeries,
  ProbeKind,
  ProbeSpec,
  RunStatus,
  RunStorage,
  RunDetail,
  RunHyperparams,
  RunSummary,
  ServerEvent,
  SnapshotMeta,
} from "../api/types";
import { wsClient } from "../api/ws";
import { derivedKind } from "../graph/probes";
import type { GraphIR } from "../graph/ir";

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
    // 清空也要通知：图表订阅者据此清掉旧 run 的曲线与「空」标记
    this.listeners.forEach((listener) => listener());
  }

  get length(): number {
    return this.steps.length;
  }

  firstStep(): number | null {
    return this.steps.length === 0 ? null : this.steps[0];
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

/** 探针勾选项：按 `(node_id, kind)` 标识一条流（docs/02 §6.1）。 */
export interface ProbeChoice {
  nodeId: string;
  kind: ProbeKind;
}

/** 默认采样间隔与每流快照上限（对齐后端 `config.PROBE_*`）。 */
export const DEFAULT_PROBE_EVERY_N = 50;
export const PROBE_SNAPSHOT_WINDOW = 200;

export function probeStreamKey(choice: ProbeChoice): string {
  return `${choice.nodeId}:${choice.kind}`;
}

export function parseStreamKey(key: string): ProbeChoice | null {
  const at = key.lastIndexOf(":");
  if (at <= 0) return null;
  return { nodeId: key.slice(0, at), kind: key.slice(at + 1) as ProbeKind };
}

interface RunState {
  datasets: DatasetInfo[] | null;
  datasetsError: string | null;
  datasetId: string | null;
  datasetProgress: Record<string, DatasetProgress>;
  config: RunConfig;
  current: RunSummary | null;
  /** 回放中的 run 详情（含 `graph`，供只读恢复图结构；docs/02 §9.2「回放模式」） */
  replay: RunDetail | null;
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

  probeChoices: ProbeChoice[];
  probeEveryN: number;
  /** streamKey → 该流的快照元数据（按 step 递增，实时追加 / 回放整批填充） */
  snapshots: Record<string, SnapshotMeta[]>;
  snapshotRunId: string | null;
  probeStream: string | null;
  /** 曲线与探针共用的时间游标（step）；null = 跟随最新（docs/02 §9.2「时间游标与回放」） */
  cursorStep: number | null;
  /** 指标缓冲当前承载的 run：曲线数据源标注与 WS 指标归属都看它，避免跨 run 混数据 */
  metricsRunId: string | null;
  /** 历史 run 的磁盘占用（docs/02 §8.1） */
  storage: RunStorage | null;
  /** 回放详情加载中：URL 指定的回放不被「自动选中最新 run」抢占 */
  replayLoading: boolean;

  loadDatasets: () => Promise<void>;
  downloadDataset: (datasetId: string) => Promise<void>;
  setDataset: (datasetId: string) => void;
  setConfig: (patch: Partial<RunConfig>) => void;
  applyConfigDefaults: (defaults: Record<string, unknown> | null | undefined) => void;
  applyProbeDefaults: (graph: Pick<GraphIR, "nodes" | "probe_defaults"> | null | undefined) => void;
  toggleProbe: (nodeId: string, defaultKind: ProbeKind) => void;
  setProbeKind: (nodeId: string, kind: ProbeKind) => void;
  setProbeEveryN: (everyN: number) => void;
  setProbeStream: (stream: string | null) => void;
  setCursorStep: (step: number | null) => void;
  start: (graph: unknown, modelId?: string) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  applyLr: (lr: number) => Promise<void>;
  applyBatchSize: (size: number) => Promise<void>;
  loadHistory: () => Promise<void>;
  loadStorage: () => Promise<void>;
  clearHistory: () => Promise<void>;
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

function firstStream(grouped: Record<string, SnapshotMeta[]>): string | null {
  const keys = Object.keys(grouped);
  return keys.find((key) => grouped[key].length > 0) ?? keys[0] ?? null;
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

/** 回放时把该 run 的超参回显到训练配置（只读展示，docs/02 §9.2「回放模式」）。 */
function configFromHyperparams(hyperparams: RunHyperparams | null | undefined): RunConfig {
  const patch: Partial<RunConfig> = {};
  for (const key of Object.keys(DEFAULT_RUN_CONFIG) as (keyof RunConfig)[]) {
    const value = hyperparams?.[key];
    if (value !== undefined && value !== null) patch[key] = value as never;
  }
  return { ...DEFAULT_RUN_CONFIG, ...patch };
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

  /** 把快照索引按流分组，供探针面板直接取用。 */
  const groupSnapshots = (metas: SnapshotMeta[]) => {
    const grouped: Record<string, SnapshotMeta[]> = {};
    for (const meta of metas) {
      if (!meta.node_id) continue;
      const key = probeStreamKey({ nodeId: meta.node_id, kind: meta.kind });
      (grouped[key] ??= []).push(meta);
    }
    for (const list of Object.values(grouped)) list.sort((a, b) => a.step - b.step || a.id.localeCompare(b.id));
    return grouped;
  };

  /** 挂到某个 run 上：勾选与采样间隔同步为该 run 实际提交的探针配置（训练中不可改，docs/02 §6.1）。 */
  const adoptRunProbes = (run: RunSummary) => {
    clearSnapshotCache();
    const specs = run.probes ?? [];
    const choices = specs.map((spec) => ({ nodeId: spec.node_id, kind: spec.kind }));
    set({
      probeChoices: choices,
      probeEveryN: specs[0]?.every_n_steps ?? get().probeEveryN,
      snapshots: {},
      snapshotRunId: run.id,
      probeStream: choices.length > 0 ? probeStreamKey(choices[0]) : null,
      cursorStep: null,
    });
  };

  /** 拉快照索引（回放整批 / 实时补齐未订阅期间的采样）。 */
  const loadSnapshots = async (runId: string) => {
    try {
      const payload = await listSnapshots(runId);
      if (get().snapshotRunId !== runId) return;
      const grouped = groupSnapshots(payload.snapshots);
      set((state) => ({
        snapshots: grouped,
        // 勾选的首条流即使还没采样到也保持选中；否则退到第一个有数据的流
        probeStream: state.probeStream ?? firstStream(grouped),
      }));
    } catch {
      // 索引拉取失败不影响其余面板：实时事件仍会补上快照
    }
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
    probeChoices: [],
    probeEveryN: DEFAULT_PROBE_EVERY_N,
    snapshots: {},
    snapshotRunId: null,
    probeStream: null,
    cursorStep: null,
    metricsRunId: null,
    storage: null,
    replayLoading: false,

    loadDatasets: async () => {
      try {
        const datasets = await listDatasets();
        const cached = datasets.find((item) => item.cached);
        set((state) => ({
          datasets,
          datasetsError: null,
          datasetId: datasets.some((item) => item.id === state.datasetId)
            ? state.datasetId
            : cached?.id ?? datasets[0]?.id ?? null,
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

    applyProbeDefaults: (graph) => {
      const ids = graph?.probe_defaults ?? [];
      if (!graph || ids.length === 0) return;
      const kinds = new Map(graph.nodes.map((node) => [node.id, derivedKind(node.type)]));
      const choices: ProbeChoice[] = [];
      for (const nodeId of ids) {
        const kind = kinds.get(nodeId);
        if (kind) choices.push({ nodeId, kind });
      }
      if (choices.length === 0) return;
      set((state) => ({
        probeChoices: choices,
        probeStream: state.snapshotRunId === null ? probeStreamKey(choices[0]) : state.probeStream,
      }));
    },

    toggleProbe: (nodeId, defaultKind) =>
      set((state) => {
        const existing = state.probeChoices.find((choice) => choice.nodeId === nodeId);
        if (existing) {
          const choices = state.probeChoices.filter((choice) => choice.nodeId !== nodeId);
          const dropped = probeStreamKey(existing);
          return {
            probeChoices: choices,
            probeStream: state.probeStream === dropped ? null : state.probeStream,
          };
        }
        return { probeChoices: [...state.probeChoices, { nodeId, kind: defaultKind }] };
      }),

    setProbeKind: (nodeId, kind) =>
      set((state) => {
        const previous = state.probeChoices.find((choice) => choice.nodeId === nodeId);
        const choices = state.probeChoices.map((choice) =>
          choice.nodeId === nodeId ? { nodeId, kind } : choice,
        );
        // 改 kind 等于换一条流：旧的视图选中项跟着迁移，避免指向已不存在的流
        const probeStream =
          previous && state.probeStream === probeStreamKey(previous)
            ? probeStreamKey({ nodeId, kind })
            : state.probeStream;
        return { probeChoices: choices, probeStream };
      }),

    setProbeEveryN: (everyN) => set({ probeEveryN: Math.max(1, Math.round(everyN)) }),

    // 游标按 step 记（与流无关），换流不重置：切换 (节点, kind) 后仍停在同一步
    setProbeStream: (stream) => set({ probeStream: stream }),

    setCursorStep: (step) => set({ cursorStep: step }),

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
        const probes: ProbeSpec[] = state.probeChoices.map((choice) => ({
          node_id: choice.nodeId,
          kind: choice.kind,
          every_n_steps: state.probeEveryN,
        }));
        const run = await startRun({
          graph: modelId ? undefined : graph,
          model_id: modelId,
          dataset_id: datasetId,
          hyperparams: state.config,
          probes,
        });
        metricsBuffer.reset();
        clearSnapshotCache();
        wsClient.subscribe(run.id);
        applySummary(run);
        set({
          starting: false,
          snapshots: {},
          snapshotRunId: run.id,
          metricsRunId: run.id,
          probeStream: probes.length > 0 ? probeStreamKey(state.probeChoices[0]) : null,
          cursorStep: null,
        });
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
        void get().loadStorage();
      } catch (error) {
        set({ historyError: error instanceof Error ? error.message : String(error) });
      }
    },

    loadStorage: async () => {
      try {
        set({ storage: await fetchRunStorage() });
      } catch {
        // 占用统计拿不到不影响历史列表本身
      }
    },

    clearHistory: async () => {
      try {
        await clearRunHistory();
      } catch (error) {
        set({ runError: errorState(error, "errors.run.deleteFailed") });
        return;
      }
      const state = get();
      // 被清理的 run 若正是曲线 / 探针当前承载的，一并清空；活动 run 保留
      await get().loadHistory();
      const remaining = get().history;
      const stillThere = (runId: string | null) =>
        runId !== null && remaining.some((item) => item.id === runId);
      const bufferDied = state.metricsRunId !== null && !stillThere(state.metricsRunId);
      const snapshotDied = state.snapshotRunId !== null && !stillThere(state.snapshotRunId);
      if (bufferDied) metricsBuffer.reset();
      if (snapshotDied) clearSnapshotCache();
      set({
        replay: stillThere(state.replay?.id ?? null) ? state.replay : null,
        metricsRunId: bufferDied ? null : state.metricsRunId,
        snapshotRunId: snapshotDied ? null : state.snapshotRunId,
        snapshots: snapshotDied ? {} : state.snapshots,
        probeStream: snapshotDied ? null : state.probeStream,
        cursorStep: snapshotDied ? null : state.cursorStep,
      });
    },

    selectRun: async (runId) => {
      if (runId === null) {
        set({
          replay: null,
          replayLoading: false,
          snapshots: {},
          snapshotRunId: null,
          probeStream: null,
          cursorStep: null,
          metricsRunId: null,
        });
        metricsBuffer.reset();
        clearSnapshotCache();
        return;
      }
      const known = get().history.find((item) => item.id === runId) ?? null;
      if (known && isActiveStatus(known.status)) {
        await get().attachRun(runId);
        set({ replay: null });
        return;
      }
      set({ replay: known, replayLoading: true });
      try {
        const [detail, metrics] = await Promise.all([getRun(runId), fetchRunMetrics(runId)]);
        metricsBuffer.loadSeries(metrics.series);
        adoptRunProbes(detail);
        set((state) => ({
          replay: detail,
          replayLoading: false,
          metricsRunId: runId,
          // 回放回显该 run 的 dataset 与超参（面板锁定编辑，docs/02 §9.2「回放模式」）
          config: configFromHyperparams(detail.hyperparams),
          datasetId: detail.dataset_id ?? state.datasetId,
        }));
        await loadSnapshots(runId);
      } catch (error) {
        const current = get().replay;
        set({
          replayLoading: false,
          // 失败时丢掉指向该 run 的回放态（含从列表乐观带出的 summary，它没有 graph）；
          // 仍在回放别的 run 时保持不动，无效的 `?run=` 由 URL 同步 effect 清掉。
          replay: current && current.id !== runId ? current : null,
          runError: errorState(error, "errors.run.loadFailed"),
        });
      }
    },

    removeRun: async (runId) => {
      try {
        await deleteRun(runId);
      } catch (error) {
        set({ runError: errorState(error, "errors.run.deleteFailed") });
        return;
      }
      const state = get();
      const wasReplay = state.replay?.id === runId;
      const bufferDied = state.metricsRunId === runId;
      const snapshotDied = state.snapshotRunId === runId;
      if (bufferDied) metricsBuffer.reset();
      if (snapshotDied) clearSnapshotCache();
      set({
        history: state.history.filter((item) => item.id !== runId),
        historyTotal: Math.max(0, state.historyTotal - 1),
        replay: wasReplay ? null : state.replay,
        snapshots: snapshotDied ? {} : state.snapshots,
        snapshotRunId: snapshotDied ? null : state.snapshotRunId,
        probeStream: snapshotDied ? null : state.probeStream,
        cursorStep: snapshotDied ? null : state.cursorStep,
        metricsRunId: bufferDied ? null : state.metricsRunId,
      });
      void get().loadStorage();
    },

    attachRun: async (runId) => {
      try {
        const detail = await getRun(runId);
        wsClient.subscribe(runId);
        applySummary(detail);
        set({ runError: null, logs: [] });
        adoptRunProbes(detail);
        const metrics = await fetchRunMetrics(runId);
        metricsBuffer.loadSeries(metrics.series);
        set({ metricsRunId: runId });
        await loadSnapshots(runId);
      } catch (error) {
        set({ runError: errorState(error, "errors.run.loadFailed") });
      }
    },

    refresh: async () => {
      await get().loadDatasets();
      await get().loadHistory();
      const history = get().history;
      const active = history.find((item) => isActiveStatus(item.status));
      // 回放优先：URL 指定的回放（已选中 / 加载中）不被「自动选中最新 run」抢占，
      // 此时活动 run 只同步 REST 状态与订阅，不动曲线缓冲与探针（docs/02 §9.2「回放模式」）。
      if (get().replay || get().replayLoading) {
        if (active) {
          try {
            const detail = await getRun(active.id);
            wsClient.subscribe(active.id);
            applySummary(detail);
          } catch {
            // 状态同步失败下个事件周期仍会补上
          }
        }
        return;
      }
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
        // 归属看缓冲当前承载的 run：已挂 run A + 回看 run B 时，A 的点不会混进 B 的曲线
        if (state.metricsRunId !== runId) return;
        const points = Array.isArray(event.points) ? (event.points as MetricPoint[]) : [];
        if (points.length > 0) {
          metricsBuffer.push(points);
          if (state.current?.id === runId) {
            const last = points[points.length - 1];
            set({
              step: Math.max(state.step, Number(last.step ?? 0)),
              epoch: Number(last.epoch ?? state.epoch),
            });
          }
        }
        return;
      }

      if (event.type === "probe.snapshot" && runId) {
        // 同理：只收探针面板当前承载 run（snapshotRunId）的快照
        if (state.snapshotRunId !== runId) return;
        const nodeId = typeof event.node_id === "string" ? event.node_id : null;
        const kind = event.kind as ProbeKind | undefined;
        const snapshotId = typeof event.snapshot_id === "string" ? event.snapshot_id : null;
        if (!nodeId || !kind || !snapshotId) return;
        const key = probeStreamKey({ nodeId, kind });
        const meta: SnapshotMeta = {
          id: snapshotId,
          run_id: runId,
          step: Number(event.step ?? 0),
          epoch: event.epoch === undefined || event.epoch === null ? null : Number(event.epoch),
          node_id: nodeId,
          kind,
          shape: Array.isArray(event.shape) ? (event.shape as number[]) : [],
        };
        set((current) => {
          const list = current.snapshots[key] ?? [];
          if (list.some((item) => item.id === meta.id)) return current;
          const next = [...list, meta];
          return {
            snapshots: {
              ...current.snapshots,
              [key]: next.length > PROBE_SNAPSHOT_WINDOW ? next.slice(-PROBE_SNAPSHOT_WINDOW) : next,
            },
            // 首条快照到达时自动选中该流，面板不必等用户手选
            probeStream: current.probeStream ?? key,
          };
        });
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
        const bufferDied = state.metricsRunId === runId;
        const snapshotDied = state.snapshotRunId === runId;
        if (bufferDied) metricsBuffer.reset();
        if (snapshotDied) clearSnapshotCache();
        set({
          history: state.history.filter((item) => item.id !== runId),
          historyTotal: Math.max(0, state.historyTotal - 1),
          replay: state.replay?.id === runId ? null : state.replay,
          snapshots: snapshotDied ? {} : state.snapshots,
          snapshotRunId: snapshotDied ? null : state.snapshotRunId,
          probeStream: snapshotDied ? null : state.probeStream,
          cursorStep: snapshotDied ? null : state.cursorStep,
          metricsRunId: bufferDied ? null : state.metricsRunId,
        });
        void get().loadStorage();
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
