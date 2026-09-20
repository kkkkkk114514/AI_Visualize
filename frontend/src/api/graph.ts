import type { GraphIR } from "../graph/ir";
import { apiGet, apiPost } from "./rest";

export interface GraphIssueDto {
  code: string;
  message_key: string;
  severity: "error" | "warning";
  node_id?: string;
  edge_id?: string;
  args?: Record<string, unknown>;
  detail?: string;
}

export interface NodeInferInfo {
  in_shape?: number[];
  in_shapes?: number[][];
  out_shape?: number[] | null;
  params?: number | null;
}

export interface InferResponse {
  ok: boolean;
  nodes: Record<string, NodeInferInfo>;
  total_params: number;
  errors: GraphIssueDto[];
  warnings: GraphIssueDto[];
  elapsed_ms: number;
}

/** 带上当前选中的数据集：后端会按数据集绑定 Input.shape / 词表 / 分类头后再 dry-run。 */
export function inferGraph(ir: GraphIR, datasetId?: string | null): Promise<InferResponse> {
  return apiPost<InferResponse>("/api/graph/infer", { ...ir, dataset_id: datasetId ?? undefined });
}

export function fetchModelDetail(modelId: string): Promise<{ source: string; graph: GraphIR }> {
  return apiGet<{ source: string; graph: GraphIR }>(`/api/models/${encodeURIComponent(modelId)}`);
}
