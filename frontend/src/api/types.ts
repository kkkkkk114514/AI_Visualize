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
  | "vram_mb";

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

export interface RunSummary {
  id: string;
  name: string;
  kind: string;
  model_id: string | null;
  dataset_id: string | null;
  hyperparams: RunHyperparams;
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
  source?: "raw" | "manual";
}

export interface DatasetInfo {
  id: string;
  name: LocalizedText;
  task: string;
  input_shape: number[];
  num_classes: number;
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
