import type { LocalizedText, ModelKind } from "../api/types";

export const NODE_TYPES = [
  "Input",
  "Linear",
  "Conv2d",
  "MaxPool2d",
  "AvgPool2d",
  "AdaptiveAvgPool2d",
  "Flatten",
  "Dropout",
  "Activation",
  "BatchNorm2d",
  "BatchNorm1d",
  "LayerNorm",
  "Embedding",
  "PositionalEncoding",
  "LSTM",
  "MultiHeadAttention",
  "ResidualAdd",
  "Concat",
  "Output",
] as const;

export type NodeType = (typeof NODE_TYPES)[number];

export interface GraphNode {
  id: string;
  type: NodeType;
  params: Record<string, unknown>;
  ui?: { x?: number; y?: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  source_port: string;
  target: string;
  target_port: string;
}

export interface GraphIR {
  ir_version: number;
  id: string;
  kind: ModelKind;
  task: string;
  name?: LocalizedText | null;
  desc?: LocalizedText | null;
  hyper_defaults?: Record<string, unknown>;
  probe_defaults?: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphMeta {
  id: string;
  kind: ModelKind;
  task: string;
  name: LocalizedText | null;
  desc: LocalizedText | null;
  hyperDefaults: Record<string, unknown>;
  probeDefaults: string[];
}

const ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";

export function shortId(prefix: string): string {
  let suffix = "";
  for (let i = 0; i < 4; i += 1) {
    suffix += ID_ALPHABET[Math.floor(Math.random() * ID_ALPHABET.length)];
  }
  return `${prefix}${suffix}`;
}

export function createEdgeId(source: string, target: string, port: string): string {
  return `e_${source}_${port}_${target}`;
}

/** 源节点 → 目标端口的连线在 IR 中的规范写法（多输入节点按 in1..inN 自动选端口）。 */
export function nextTargetPort(edges: GraphEdge[], targetId: string, multiInput: boolean): string {
  if (!multiInput) return "in";
  const used = new Set(edges.filter((e) => e.target === targetId).map((e) => e.target_port));
  for (let i = 1; i <= 4; i += 1) {
    const port = `in${i}`;
    if (!used.has(port)) return port;
  }
  return "in1";
}

export function metaFromIR(ir: GraphIR): GraphMeta {
  return {
    id: ir.id,
    kind: ir.kind,
    task: ir.task,
    name: ir.name ?? null,
    desc: ir.desc ?? null,
    hyperDefaults: ir.hyper_defaults ?? {},
    probeDefaults: ir.probe_defaults ?? [],
  };
}

export function emptyGraph(): GraphIR {
  return {
    ir_version: 1,
    id: "new",
    kind: "dl",
    task: "image_classification",
    name: null,
    desc: null,
    hyper_defaults: { optimizer: "adam", lr: 0.001, batch_size: 64, epochs: 5, loss: "cross_entropy" },
    probe_defaults: [],
    nodes: [
      { id: "input", type: "Input", params: { shape: [1, 28, 28] }, ui: { x: 0, y: 120 } },
      { id: "output", type: "Output", params: { classes: 10 }, ui: { x: 320, y: 120 } },
    ],
    edges: [],
  };
}
