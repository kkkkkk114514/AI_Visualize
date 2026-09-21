import { apiDelete, apiGet, apiPost } from "./rest";
import type {
  MetricName,
  ProbeSpec,
  RunControlAction,
  RunDetail,
  RunHyperparams,
  RunMetricsResponse,
  RunStorage,
  RunSummary,
  RunsResponse,
} from "./types";

export interface StartRunBody {
  graph?: unknown;
  model_id?: string;
  dataset_id: string;
  hyperparams?: RunHyperparams;
  probes?: ProbeSpec[];
  seed?: number;
  name?: string;
}

export async function startRun(body: StartRunBody): Promise<RunSummary> {
  const payload = await apiPost<{ run: RunSummary }>("/api/runs", body);
  return payload.run;
}

export async function listRuns(limit = 20, offset = 0): Promise<RunsResponse> {
  return apiGet<RunsResponse>(`/api/runs?limit=${limit}&offset=${offset}`);
}

export async function getRun(runId: string): Promise<RunDetail> {
  const payload = await apiGet<{ run: RunDetail }>(`/api/runs/${encodeURIComponent(runId)}`);
  return payload.run;
}

export async function controlRun(
  runId: string,
  action: RunControlAction,
  value?: number,
): Promise<void> {
  await apiPost(`/api/runs/${encodeURIComponent(runId)}/control`, { action, value });
}

export async function fetchRunMetrics(
  runId: string,
  names?: MetricName[],
  maxPoints = 2000,
): Promise<RunMetricsResponse> {
  const params = new URLSearchParams({ max_points: String(maxPoints) });
  if (names && names.length > 0) params.set("names", names.join(","));
  return apiGet<RunMetricsResponse>(`/api/runs/${encodeURIComponent(runId)}/metrics?${params}`);
}

export async function deleteRun(runId: string): Promise<void> {
  await apiDelete(`/api/runs/${encodeURIComponent(runId)}`);
}

export async function fetchRunStorage(): Promise<RunStorage> {
  return apiGet<RunStorage>("/api/runs/storage");
}

/** 清理全部历史（跳过活动 run）；返回删除条数与保留的活动 run。 */
export async function clearRunHistory(): Promise<{ deleted: number; kept_active: string | null }> {
  return apiDelete<{ deleted: number; kept_active: string | null }>("/api/runs");
}
