import type { Edge as FlowEdge, Node as FlowNode } from "@xyflow/react";

import type { GraphEdge, GraphIR, GraphMeta, NodeType } from "./ir";
import { defaultParamsFor } from "./specs";

export interface GraphNodeData extends Record<string, unknown> {
  nodeType: NodeType;
  params: Record<string, unknown>;
}

export type FlowGraphNode = FlowNode<GraphNodeData>;
export type FlowGraphEdge = FlowEdge;

export function toFlowNodes(ir: GraphIR): FlowGraphNode[] {
  return ir.nodes.map((node) => ({
    id: node.id,
    type: "graphNode",
    position: { x: node.ui?.x ?? 0, y: node.ui?.y ?? 0 },
    data: {
      nodeType: node.type,
      params: { ...defaultParamsFor(node.type), ...node.params },
    },
  }));
}

export function toFlowEdges(ir: GraphIR): FlowGraphEdge[] {
  return ir.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.source_port,
    targetHandle: edge.target_port,
    type: "smoothstep",
  }));
}

export function toIR(
  meta: GraphMeta,
  nodes: FlowGraphNode[],
  edges: FlowGraphEdge[],
): GraphIR {
  return {
    ir_version: 1,
    id: meta.id,
    kind: meta.kind,
    task: meta.task,
    name: meta.name,
    desc: meta.desc,
    hyper_defaults: meta.hyperDefaults,
    probe_defaults: meta.probeDefaults,
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.data.nodeType,
      params: node.data.params,
      ui: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      source_port: edge.sourceHandle ?? "out",
      target: edge.target,
      target_port: edge.targetHandle ?? "in",
    })),
  };
}

export function flowEdgeOf(
  source: string,
  target: string,
  targetPort: string,
  edgeId: string,
): FlowGraphEdge | null {
  if (source === target) return null;
  return { id: edgeId, source, target, sourceHandle: "out", targetHandle: targetPort, type: "smoothstep" };
}

export function asGraphEdge(edge: FlowGraphEdge): GraphEdge {
  return {
    id: edge.id,
    source: edge.source,
    source_port: edge.sourceHandle ?? "out",
    target: edge.target,
    target_port: edge.targetHandle ?? "in",
  };
}
