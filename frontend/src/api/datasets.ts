import { apiGet, apiPost } from "./rest";
import type { DatasetInfo, DatasetsResponse } from "./types";

export async function listDatasets(): Promise<DatasetInfo[]> {
  const payload = await apiGet<DatasetsResponse>("/api/datasets");
  return payload.datasets;
}

export async function downloadDataset(datasetId: string): Promise<{ started: boolean }> {
  return apiPost<{ started: boolean }>(
    `/api/datasets/${encodeURIComponent(datasetId)}/download`,
    {},
  );
}
