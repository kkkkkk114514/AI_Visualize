export type ModelKind = "ml" | "dl" | "rl";

export type LocalizedText = string | { zh?: string; en?: string };

export interface ModelSummary {
  id: string;
  kind: ModelKind;
  task: string | null;
  name: LocalizedText | null;
  desc: LocalizedText | null;
  source: "preset" | "user";
}

export interface ModelsResponse {
  groups: Record<ModelKind, ModelSummary[]>;
}

export interface HealthInfo {
  status: string;
  service: string;
  version: string;
  python: string;
  device: "cpu" | "cuda";
  device_name: string;
  cpu_count: number | null;
  memory_gb: number | null;
  torch: string | null;
  cuda_available: boolean;
  cuda_version?: string;
}

export type RunStatus =
  | "idle"
  | "created"
  | "running"
  | "paused"
  | "finished"
  | "stopped"
  | "failed"
  | "interrupted";

export type MetricName =
  | "loss"
  | "acc"
  | "val_loss"
  | "val_acc"
  | "lr"
  | "grad_norm"
  | "throughput"
  | "vram_mb"
  // ML / RL 算法指标（docs/02 §5.4）
  | "margin"
  | "n_sv"
  | "depth"
  | "leaves"
  | "n_nodes"
  | "n_trees"
  | "reward"
  | "q_delta"
  | "episode_reward"
  | "epsilon"
  | "episode_steps"
  | "success";

export type RunControlAction =
  | "pause"
  | "resume"
  | "stop"
  | "kill"
  | "set_lr"
  | "set_batch_size";

export interface RunHyperparams {
  optimizer?: "adam" | "adamw" | "sgd";
  lr?: number;
  batch_size?: number;
  epochs?: number;
  loss?: "cross_entropy" | "mse";
  grad_clip?: number;
  train_size?: number;
  val_size?: number;
}

/** 探针类型：DL 五类 + ML / RL 的单流类型（docs/02 §6.1 / §13.4）。 */
export type ProbeKind = "feature_grid" | "attention" | "hidden" | "histogram" | "weights" | "boundary" | "grid";

/** 探针条目（docs/02 §6.1）：随 run 启动提交，训练中不变。 */
export interface ProbeSpec {
  node_id: string;
  kind: ProbeKind;
  every_n_steps?: number;
  sample_index?: number;
  max_items?: number;
}

/** WS `probe.snapshot` / 快照索引行：只有元数据，payload 另取。 */
export interface SnapshotMeta {
  id: string;
  run_id: string;
  step: number;
  epoch: number | null;
  node_id: string | null;
  kind: ProbeKind;
  shape: number[];
  min?: number | null;
  max?: number | null;
  created_at?: string | null;
}

/** 快照 payload 容器（docs/02 §6.2）：uint8 需按 min/max 反量化，uint32 为原始计数。 */
export interface SnapshotMetaInfo {
  heads?: number;
  tokens?: number;
  causal?: boolean;
  bins?: number;
  /** `boundary`（docs/02 §13.5）：score 为实值决策函数 / label 为类别下标 */
  mode?: "score" | "label";
  x_range?: [number, number];
  y_range?: [number, number];
  algo?: string;
  depth?: number;
  leaves?: number;
  nodes?: number;
  n_trees?: number;
  margin?: number;
  n_sv?: number;
  support_vectors?: [number, number][];
  /** `grid`（docs/02 §13.5）：策略动作下标（-1 = 障碍或终点）与最近一个已完成 episode */
  policy?: number[];
  trajectory?: [number, number][];
  episode?: number;
  epsilon?: number;
  success?: boolean;
}

export interface SnapshotPayload {
  shape: number[];
  dtype: string;
  layout: string;
  min: number;
  max: number;
  data_b64: string;
  meta?: SnapshotMetaInfo;
}

export interface SnapshotsResponse {
  snapshots: SnapshotMeta[];
}

export interface RunSummary {
  id: string;
  name: string;
  kind: string;
  model_id: string | null;
  dataset_id: string | null;
  hyperparams: RunHyperparams;
  probes?: ProbeSpec[];
  status: RunStatus;
  device: string | null;
  seed: number | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  best_metric: number | null;
  total_steps: number;
  elapsed_s: number | null;
  planned_steps: number | null;
  /** 仅 list/get 返回的活动 run 快照 */
  step?: number;
  epoch?: number;
}

export interface RunDetail extends RunSummary {
  graph?: unknown;
}

export interface RunsResponse {
  runs: RunSummary[];
  total: number;
}

/** 历史占用统计（docs/02 §8.1）：run 目录之和，不含数据集缓存。 */
export interface RunStorage {
  run_count: number;
  total_bytes: number;
  by_run: Record<string, number>;
  runs_dir: string;
}

export interface MetricSeries {
  steps: number[];
  values: number[];
  total: number;
}

export interface RunMetricsResponse {
  run_id: string;
  status: RunStatus;
  total_steps: number;
  max_points: number;
  series: Partial<Record<MetricName, MetricSeries>>;
}

export interface DatasetFileState {
  name: string;
  present: boolean;
  md5_ok: boolean;
  bytes: number;
  source?: "raw" | "manual" | "corpora" | string;
}

export interface DatasetInfo {
  id: string;
  name: LocalizedText;
  task: string;
  loader: string;
  input_shape: number[];
  num_classes: number;
  vocab_size: number | null;
  cached: boolean;
  size_bytes: number;
  total_bytes: number;
  path: string;
  manual_dir: string;
  files: DatasetFileState[];
  note: LocalizedText;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
}

/** `GET /api/datasets/{id}/points`（docs/02 §13.3）：ML 决策边界画布的散点底图。 */
export interface PointsResponse {
  split: string;
  points: [number, number][];
  labels: number[];
  count: number;
}

/** 算法参数 schema 字段（docs/02 §13.2）：`GET /api/algos` 与服务端校验共用同一份表。 */
export interface AlgoField {
  name: string;
  type: "int" | "float" | "choice";
  min?: number | null;
  max?: number | null;
  step?: number | null;
  choices?: string[];
  default: number | string;
  label_key: string;
}

export interface AlgoSchema {
  algo: string;
  kind: "ml" | "rl";
  fields: AlgoField[];
}

export interface AlgosResponse {
  algos: Record<string, AlgoSchema>;
}

/** GridWorld 环境（docs/02 §13.3）：`spec.env`。 */
export interface GridEnv {
  width: number;
  height: number;
  start: [number, number];
  goal: [number, number];
  obstacles: [number, number][];
  rewards: [number, number, number][];
}

/** ML / RL 的模型定义（docs/02 §13.1）：与 DL 图 IR 共用 `graph` 字段，按 `kind` 区分。 */
export interface AlgoSpec {
  ir_version?: number;
  id: string;
  kind: "ml" | "rl";
  task?: string;
  name?: LocalizedText;
  desc?: LocalizedText;
  algo: string;
  params: Record<string, number | string>;
  env?: GridEnv;
  probe_defaults?: { every_n_steps?: number };
  dataset_id?: string;
}

export interface ApiErrorPayload {
  code: string;
  message_key: string;
  args?: Record<string, unknown>;
  detail?: string;
}

export interface ServerEvent {
  type: string;
  run_id?: string;
  dataset_id?: string;
  [key: string]: unknown;
}
