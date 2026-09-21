import { create } from "zustand";

import { fetchAlgoSchemas } from "../api/algos";
import type { AlgoSchema, AlgoSpec, GridEnv } from "../api/types";

/**
 * ML / RL 的 spec 存储（docs/02 §13.6）：这些模型不是 DL 图 IR，不进 `graphStore`。
 * 表单改动只作用本次 run 的请求体（预置文件不动）：`dirty` 时提交内联 `graph`，
 * 未改动时提交 `model_id`。
 */

/** 请求体里的 `graph` 是 DL 图 IR 还是 ML / RL 的 algo spec（kind 缺省按 dl）。 */
export function isAlgoSpec(graph: unknown): graph is AlgoSpec {
  const kind = (graph as { kind?: unknown } | null | undefined)?.kind;
  return kind === "ml" || kind === "rl";
}

export interface AlgoSource {
  /** 来自模型库的模型 id：未改动参数时提交它，避免把预置整份内联 */
  modelId: string | null;
  /** 来自历史 run 的 id：只读回显该 run 的 spec（回放，表单一律锁定） */
  fromRunId: string | null;
}

interface AlgoState {
  spec: AlgoSpec | null;
  /** 进入时的原样 spec（序列化）：与 `spec` 不同即 dirty */
  baseJson: string | null;
  dirty: boolean;
  modelId: string | null;
  fromRunId: string | null;
  schemas: Record<string, AlgoSchema> | null;
  schemasError: string | null;

  loadSpec: (spec: AlgoSpec, source: AlgoSource) => void;
  clear: () => void;
  loadSchemas: () => Promise<void>;
  setParam: (name: string, value: number | string) => void;
  setDataset: (datasetId: string) => void;
  setEnv: (env: GridEnv) => void;
  /** 提交给 `POST /api/runs` 的来源：改过 → 内联 spec，未改 → model_id */
  runSource: () => { graph?: AlgoSpec; modelId?: string };
}

let schemaTask: Promise<void> | null = null;

export const useAlgoStore = create<AlgoState>((set, get) => {
  const apply = (spec: AlgoSpec) => {
    set({ spec, dirty: get().baseJson !== JSON.stringify(spec) });
  };

  return {
    spec: null,
    baseJson: null,
    dirty: false,
    modelId: null,
    fromRunId: null,
    schemas: null,
    schemasError: null,

    loadSpec: (spec, source) => {
      set({
        spec,
        baseJson: JSON.stringify(spec),
        dirty: false,
        modelId: source.modelId,
        fromRunId: source.fromRunId,
      });
    },

    clear: () => {
      if (get().spec === null) return;
      set({ spec: null, baseJson: null, dirty: false, modelId: null, fromRunId: null });
    },

    loadSchemas: () => {
      if (get().schemas) return Promise.resolve();
      schemaTask ??= fetchAlgoSchemas()
        .then((payload) => set({ schemas: payload.algos, schemasError: null }))
        .catch((error: unknown) => {
          schemaTask = null;
          set({ schemasError: error instanceof Error ? error.message : String(error) });
        });
      return schemaTask;
    },

    setParam: (name, value) => {
      const spec = get().spec;
      if (!spec) return;
      apply({ ...spec, params: { ...spec.params, [name]: value } });
    },

    setDataset: (datasetId) => {
      const spec = get().spec;
      if (!spec) return;
      apply({ ...spec, dataset_id: datasetId });
    },

    setEnv: (env) => {
      const spec = get().spec;
      if (!spec) return;
      apply({ ...spec, env });
    },

    runSource: () => {
      const { spec, dirty, modelId } = get();
      if (!spec) return {};
      if (!dirty && modelId) return { modelId };
      return { graph: spec };
    },
  };
});
