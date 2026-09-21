import { apiGet } from "./rest";
import type { AlgosResponse } from "./types";

/** 算法参数 schema（docs/02 §13.2）：打开 ML / RL 工作台时取一次，参数表单与服务端共用同一份表。 */
export function fetchAlgoSchemas(): Promise<AlgosResponse> {
  return apiGet<AlgosResponse>("/api/algos");
}
