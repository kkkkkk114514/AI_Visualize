import { apiDelete, apiGet, apiPost, apiUpload } from "./rest";
import type { DatasetInfo, DatasetsResponse, PointsResponse } from "./types";

export async function listDatasets(): Promise<DatasetInfo[]> {
  const payload = await apiGet<DatasetsResponse>("/api/datasets");
  return payload.datasets;
}

/** 二维点集（docs/02 §13.3）：合成集与上传的 csv2d 都可用，是决策边界画布的散点底图。 */
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

/** 上传自定义数据集（docs/02 §7.5）：`.zip` 图片集 / `.csv` 点集 / `.txt` 语料。 */
export async function uploadDataset(file: File): Promise<DatasetInfo> {
  const form = new FormData();
  form.append("file", file, file.name);
  const payload = await apiUpload<{ dataset: DatasetInfo }>("/api/datasets/upload", form);
  return payload.dataset;
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await apiDelete<{ deleted: string }>(`/api/datasets/${encodeURIComponent(datasetId)}`);
}
