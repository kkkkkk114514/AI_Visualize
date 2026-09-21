import { apiGet, apiPost } from "./rest";
import type { DatasetInfo, DatasetsResponse, PointsResponse } from "./types";

export async function listDatasets(): Promise<DatasetInfo[]> {
  const payload = await apiGet<DatasetsResponse>("/api/datasets");
  return payload.datasets;
}

/** 二维合成点集（docs/02 §13.3）：决策边界画布的散点底图，固定 seed 与后端训练逐位一致。 */
export function fetchDatasetPoints(
  datasetId: string,
  split: "train" | "val" = "train",
): Promise<PointsResponse> {
  return apiGet<PointsResponse>(
    `/api/datasets/${encodeURIComponent(datasetId)}/points?split=${split}`,
  );
}

export async function downloadDataset(datasetId: string): Promise<{ started: boolean }> {
  return apiPost<{ started: boolean }>(
    `/api/datasets/${encodeURIComponent(datasetId)}/download`,
    {},
  );
}
