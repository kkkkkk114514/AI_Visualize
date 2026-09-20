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

export interface ServerEvent {
  type: string;
  run_id?: string;
  [key: string]: unknown;
}
