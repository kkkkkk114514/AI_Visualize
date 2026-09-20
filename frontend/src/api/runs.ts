import { apiDelete, apiGet, apiPost } from "./rest";
import type {
  MetricName,
  ProbeSpec,
  RunControlAction,
  RunDetail,
  RunHyperparams,
  RunMetricsResponse,
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
