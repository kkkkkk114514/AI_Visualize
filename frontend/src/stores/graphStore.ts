import { create } from "zustand";

import type { GraphIssueDto, NodeInferInfo } from "../api/graph";
import type { GraphIR, GraphMeta, NodeType } from "../graph/ir";
import { createEdgeId, emptyGraph, metaFromIR, nextTargetPort, shortId } from "../graph/ir";
import { computeLayout } from "../graph/layout";
import type { FlowGraphEdge, FlowGraphNode } from "../graph/flow";
import { toFlowEdges, toFlowNodes, toIR } from "../graph/flow";
import { specOf } from "../graph/specs";
import type { LocalIssue } from "../graph/validate";
import { validateLocal } from "../graph/validate";

interface Snapshot {
  nodes: FlowGraphNode[];
  edges: FlowGraphEdge[];
}

export interface InferState {
  pending: boolean;
  byNode: Record<string, NodeInferInfo>;
  totalParams: number | null;
  errors: GraphIssueDto[];
  warnings: GraphIssueDto[];
  elapsedMs: number | null;
  requestError: string | null;
}

interface GraphState {
  meta: GraphMeta;
  nodes: FlowGraphNode[];
  edges: FlowGraphEdge[];
  readOnly: boolean;
  source: "preset" | "user" | "new";
  dirty: boolean;
  clipboard: { nodes: FlowGraphNode[]; edges: FlowGraphEdge[] } | null;
  past: Snapshot[];
  future: Snapshot[];
  localIssues: LocalIssue[];
  infer: InferState;
  /** 整图被替换（load / 克隆）时自增，用于强制重新推导：这类替换会清空 infer，但结构签名可能不变。 */
  epoch: number;
  load: (ir: GraphIR, options: { readOnly: boolean; source: "preset" | "user" | "new" }) => void;
  toGraphIR: () => GraphIR;
  applyPositions: (nodes: FlowGraphNode[]) => void;
  addNode: (type: NodeType, position: { x: number; y: number }) => void;
  connect: (source: string, target: string) => void;
  deleteElements: (nodeIds: string[], edgeIds: string[]) => void;
  updateParams: (nodeId: string, patch: Record<string, unknown>) => void;
  copySelection: (nodeIds: string[]) => void;
  pasteClipboard: (offset?: { x: number; y: number }) => void;
  cloneAsCopy: () => void;
  autoLayout: () => void;
  undo: () => void;
  redo: () => void;
  setInferPending: () => void;
  setInferResult: (result: Omit<InferState, "pending" | "requestError">) => void;
  setInferError: (message: string) => void;
}

const EMPTY_INFER: InferState = {
  pending: false,
  byNode: {},
  totalParams: null,
  errors: [],
  warnings: [],
  elapsedMs: null,
  requestError: null,
};

const HISTORY_LIMIT = 50;

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function snapshot(nodes: FlowGraphNode[], edges: FlowGraphEdge[]): Snapshot {
  return { nodes: deepCopy(nodes), edges: deepCopy(edges) };
}

export const useGraphStore = create<GraphState>((set, get) => {
  const commit = (
    updater: (state: GraphState) => { nodes: FlowGraphNode[]; edges: FlowGraphEdge[] },
    options: { history?: boolean } = {},
  ) => {
    const state = get();
    if (state.readOnly) return;
    const before = snapshot(state.nodes, state.edges);
    const next = updater(state);
    set({
      nodes: next.nodes,
      edges: next.edges,
      dirty: true,
      localIssues: validateLocal(toIR(state.meta, next.nodes, next.edges)),
      past: options.history === false ? state.past : [...state.past, before].slice(-HISTORY_LIMIT),
      future: options.history === false ? state.future : [],
    });
  };

  return {
    meta: metaFromIR(emptyGraph()),
    nodes: toFlowNodes(emptyGraph()),
    edges: [],
    readOnly: false,
    source: "new",
    dirty: false,
    clipboard: null,
    past: [],
    future: [],
    localIssues: [],
    infer: EMPTY_INFER,
    epoch: 0,

    load: (ir, { readOnly, source }) => {
      const nodes = toFlowNodes(ir);
      const edges = toFlowEdges(ir);
      const meta = metaFromIR(ir);
      set({
        meta,
        nodes,
        edges,
        readOnly,
        source,
        dirty: false,
        past: [],
        future: [],
        clipboard: null,
        localIssues: validateLocal(toIR(meta, nodes, edges)),
        infer: EMPTY_INFER,
        epoch: get().epoch + 1,
      });
    },

    toGraphIR: () => {
      const { meta, nodes, edges } = get();
      return toIR(meta, nodes, edges);
    },

    applyPositions: (nodes) => {
      set({ nodes });
    },

    addNode: (type, position) => {
      const spec = specOf(type);
      if (!spec) return;
      const id = shortId("n");
      commit((state) => ({
        nodes: [
          ...state.nodes,
          {
            id,
            type: "graphNode",
            position,
            data: { nodeType: type, params: { ...spec.defaultParams } },
          },
        ],
        edges: state.edges,
      }));
    },

    connect: (source, target) => {
      const state = get();
      if (state.readOnly || source === target) return;
      const targetNode = state.nodes.find((n) => n.id === target);
      if (!targetNode) return;
      const spec = specOf(targetNode.data.nodeType);
      if (!spec) return;
      const existing = state.edges.filter((e) => e.target === target);
      const limit = spec.multiInput ? 4 : 1;
      if (existing.length >= limit) return;
      const port = nextTargetPort(
        state.edges.map((e) => ({ id: e.id, source: e.source, source_port: e.sourceHandle ?? "out", target: e.target, target_port: e.targetHandle ?? "in" })),
        target,
        Boolean(spec.multiInput),
      );
      const edgeId = createEdgeId(source, target, port);
      const edge = {
        id: edgeId,
        source,
        target,
        sourceHandle: "out",
        targetHandle: port,
        type: "smoothstep" as const,
      };
      commit((current) => ({ nodes: current.nodes, edges: [...current.edges, edge] }));
    },

    deleteElements: (nodeIds, edgeIds) => {
      if (nodeIds.length === 0 && edgeIds.length === 0) return;
      commit((state) => {
        const nodeSet = new Set(nodeIds);
        const edgeSet = new Set(edgeIds);
        const nodes = state.nodes.filter((n) => !nodeSet.has(n.id));
        let edges = state.edges.filter(
          (e) => !edgeSet.has(e.id) && !nodeSet.has(e.source) && !nodeSet.has(e.target),
        );

        const portIndex = (edge: FlowGraphEdge): number => {
          const match = /^in(\d+)$/.exec(String(edge.targetHandle ?? ""));
          return match ? Number(match[1]) : 0;
        };

        // 删除后重排端口，保持 in1..inN 连续
        for (const node of nodes) {
          const spec = specOf(node.data.nodeType);
          if (!spec?.multiInput) {
            edges = edges.map((edge) =>
              edge.target === node.id && edge.targetHandle !== "in" ? { ...edge, targetHandle: "in" } : edge,
            );
            continue;
          }
          const incoming = edges
            .filter((edge) => edge.target === node.id)
            .sort((a, b) => portIndex(a) - portIndex(b));
          const renumbered = new Map<string, string>();
          incoming.forEach((edge, index) => renumbered.set(edge.id, `in${index + 1}`));
          edges = edges.map((edge) =>
            renumbered.has(edge.id) ? { ...edge, targetHandle: renumbered.get(edge.id)! } : edge,
          );
        }
        return { nodes, edges };
      });
    },

    updateParams: (nodeId, patch) => {
      commit((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, params: { ...node.data.params, ...patch } } } : node,
        ),
        edges: state.edges,
      }));
    },

    copySelection: (nodeIds) => {
      const { nodes, edges } = get();
      const selected = new Set(nodeIds);
      if (selected.size === 0) return;
      set({
        clipboard: {
          nodes: deepCopy(nodes.filter((n) => selected.has(n.id))),
          edges: deepCopy(edges.filter((e) => selected.has(e.source) && selected.has(e.target))),
        },
      });
    },

    pasteClipboard: (offset = { x: 40, y: 40 }) => {
      const clipboard = get().clipboard;
      if (!clipboard || clipboard.nodes.length === 0) return;
      const idMap = new Map<string, string>();
      for (const node of clipboard.nodes) {
        idMap.set(node.id, shortId("n"));
      }
      const newNodes: FlowGraphNode[] = clipboard.nodes.map((node) => ({
        ...deepCopy(node),
        id: idMap.get(node.id)!,
        position: { x: node.position.x + offset.x, y: node.position.y + offset.y },
        selected: false,
      }));
      const newEdges: FlowGraphEdge[] = clipboard.edges.map((edge) => ({
        ...deepCopy(edge),
        id: shortId("e"),
        source: idMap.get(edge.source)!,
        target: idMap.get(edge.target)!,
      }));
      commit((state) => ({ nodes: [...state.nodes, ...newNodes], edges: [...state.edges, ...newEdges] }));
    },

    cloneAsCopy: () => {
      const { meta, nodes, edges } = get();
      // 节点 id 原样保留，故 name/desc/探针配置可继续沿用；仅换模型 id 并解除只读。
      const nextMeta: GraphMeta = { ...meta, id: shortId("m") };
      set({
        meta: nextMeta,
        readOnly: false,
        source: "user",
        dirty: true,
        past: [],
        future: [],
        infer: EMPTY_INFER,
        epoch: get().epoch + 1,
        localIssues: validateLocal(toIR(nextMeta, nodes, edges)),
      });
    },

    autoLayout: () => {
      commit((state) => {
        const positions = computeLayout(toIR(state.meta, state.nodes, state.edges));
        return {
          nodes: state.nodes.map((node) => {
            const position = positions.get(node.id);
            return position ? { ...node, position } : node;
          }),
          edges: state.edges,
        };
      });
    },

    undo: () => {
      const state = get();
      if (state.readOnly || state.past.length === 0) return;
      const previous = state.past[state.past.length - 1];
      const current = snapshot(state.nodes, state.edges);
      set({
        nodes: previous.nodes,
        edges: previous.edges,
        past: state.past.slice(0, -1),
        future: [current, ...state.future].slice(0, HISTORY_LIMIT),
        dirty: true,
        localIssues: validateLocal(toIR(state.meta, previous.nodes, previous.edges)),
      });
    },

    redo: () => {
      const state = get();
      if (state.readOnly || state.future.length === 0) return;
      const next = state.future[0];
      const current = snapshot(state.nodes, state.edges);
      set({
        nodes: next.nodes,
        edges: next.edges,
        past: [...state.past, current].slice(-HISTORY_LIMIT),
        future: state.future.slice(1),
        dirty: true,
        localIssues: validateLocal(toIR(state.meta, next.nodes, next.edges)),
      });
    },

    setInferPending: () => {
      set((state) => ({ infer: { ...state.infer, pending: true, requestError: null } }));
    },

    setInferResult: (result) => {
      set({ infer: { ...result, pending: false, requestError: null } });
    },

    setInferError: (message) => {
      set((state) => ({ infer: { ...state.infer, pending: false, requestError: message } }));
    },
  };
});
