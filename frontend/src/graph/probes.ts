import type { ProbeKind } from "../api/types";
import type { FlowGraphNode } from "./flow";
import type { NodeType } from "./ir";

/** kind 默认按节点类型推导：与后端 `probes/encode.py` 的 DERIVED_KIND 一致（docs/02 §6.1）。 */
const DERIVED_KIND: Partial<Record<NodeType, ProbeKind>> = {
  Conv2d: "feature_grid",
  MultiHeadAttention: "attention",
  LSTM: "hidden",
};

/** 手动可选的额外 kind：`weights` 只列第一个 ≥2 维参数一定有值的节点。 */
const EXTRA_KINDS: Partial<Record<NodeType, ProbeKind[]>> = {
  Conv2d: ["weights", "histogram"],
  MultiHeadAttention: ["weights", "histogram"],
  LSTM: ["weights", "histogram"],
  Linear: ["weights"],
  Embedding: ["weights"],
};

export function derivedKind(nodeType: NodeType): ProbeKind {
  return DERIVED_KIND[nodeType] ?? "histogram";
}

export function kindsFor(nodeType: NodeType): ProbeKind[] {
  return [derivedKind(nodeType), ...(EXTRA_KINDS[nodeType] ?? [])];
}

export interface ProbeCandidate {
  nodeId: string;
  nodeType: NodeType;
  kinds: ProbeKind[];
}

export function probeCandidates(nodes: FlowGraphNode[]): ProbeCandidate[] {
  return nodes.map((node) => ({
    nodeId: node.id,
    nodeType: node.data.nodeType,
    kinds: kindsFor(node.data.nodeType),
  }));
}
