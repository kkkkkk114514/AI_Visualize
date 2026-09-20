import { useEffect, useMemo } from "react";

import { inferGraph } from "../../api/graph";
import { useGraphStore } from "../../stores/graphStore";
import { useRunStore } from "../../stores/runStore";

const DEBOUNCE_MS = 300;

/** 结构（节点参数 + 连线 + 数据集）变化后 300ms 防抖调用 /api/graph/infer；拖动位置不触发。 */
export function useGraphInfer(enabled = true) {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const task = useGraphStore((state) => state.meta.task);
  const epoch = useGraphStore((state) => state.epoch);
  const datasetId = useRunStore((state) => state.datasetId);
  const setInferPending = useGraphStore((state) => state.setInferPending);
  const setInferResult = useGraphStore((state) => state.setInferResult);
  const setInferError = useGraphStore((state) => state.setInferError);

  const signature = useMemo(
    () =>
      JSON.stringify({
        epoch,
        task,
        datasetId,
        nodes: nodes.map((node) => [node.id, node.data.nodeType, node.data.params]),
        edges: edges.map((edge) => [edge.source, edge.sourceHandle, edge.target, edge.targetHandle]),
      }),
    [epoch, nodes, edges, task, datasetId],
  );

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      const ir = useGraphStore.getState().toGraphIR();
      setInferPending();
      inferGraph(ir, datasetId)
        .then((result) => {
          if (controller.signal.aborted) return;
          setInferResult({
            byNode: result.nodes,
            totalParams: result.ok || Object.keys(result.nodes).length > 0 ? result.total_params : null,
            errors: result.errors,
            warnings: result.warnings,
            elapsedMs: result.elapsed_ms,
          });
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          setInferError(error instanceof Error ? error.message : String(error));
        });
    }, DEBOUNCE_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [signature, datasetId, enabled, setInferPending, setInferResult, setInferError]);
}
