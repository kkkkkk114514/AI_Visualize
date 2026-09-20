import type { NodeType } from "./ir";

export interface EnumChoice {
  value: string;
  labelKey: string;
}

export interface ParamSpecUI {
  key: string;
  labelKey: string;
  kind: "int" | "float" | "bool" | "enum" | "shape";
  min?: number;
  max?: number;
  step?: number;
  choices?: EnumChoice[];
  /** 空值按 kernel_size 处理（池化 stride） */
  zeroMeansKernel?: boolean;
}

export type NodeCategory = "io" | "basic" | "conv" | "norm" | "seq" | "combine";

export interface NodeSpec {
  type: NodeType;
  category: NodeCategory;
  params: ParamSpecUI[];
  multiInput?: boolean;
  defaultParams: Record<string, unknown>;
}

const ACTIVATIONS: EnumChoice[] = [
  { value: "relu", labelKey: "activation.relu" },
  { value: "leaky_relu", labelKey: "activation.leaky_relu" },
  { value: "gelu", labelKey: "activation.gelu" },
  { value: "tanh", labelKey: "activation.tanh" },
  { value: "sigmoid", labelKey: "activation.sigmoid" },
];

const POS_MODES: EnumChoice[] = [
  { value: "learned", labelKey: "posMode.learned" },
  { value: "sinusoidal", labelKey: "posMode.sinusoidal" },
];

export const NODE_SPECS: Record<NodeType, NodeSpec> = {
  Input: {
    type: "Input",
    category: "io",
    params: [{ key: "shape", labelKey: "param.shape", kind: "shape" }],
    defaultParams: { shape: [1, 28, 28] },
  },
  Output: {
    type: "Output",
    category: "io",
    params: [
      { key: "classes", labelKey: "param.classes", kind: "int", min: 1, max: 1 << 20 },
      { key: "out_dim", labelKey: "param.outDim", kind: "int", min: 0, max: 1 << 20 },
    ],
    defaultParams: { classes: 10, out_dim: 0 },
  },
  Linear: {
    type: "Linear",
    category: "basic",
    params: [
      { key: "out_features", labelKey: "param.outFeatures", kind: "int", min: 1, max: 1 << 20 },
      { key: "bias", labelKey: "param.bias", kind: "bool" },
    ],
    defaultParams: { out_features: 128, bias: true },
  },
  Flatten: {
    type: "Flatten",
    category: "basic",
    params: [{ key: "start_dim", labelKey: "param.startDim", kind: "int", min: 0, max: 4 }],
    defaultParams: { start_dim: 1 },
  },
  Dropout: {
    type: "Dropout",
    category: "basic",
    params: [{ key: "p", labelKey: "param.dropoutP", kind: "float", min: 0, max: 0.99, step: 0.05 }],
    defaultParams: { p: 0.5 },
  },
  Activation: {
    type: "Activation",
    category: "basic",
    params: [{ key: "name", labelKey: "param.activation", kind: "enum", choices: ACTIVATIONS }],
    defaultParams: { name: "relu" },
  },
  Conv2d: {
    type: "Conv2d",
    category: "conv",
    params: [
      { key: "out_channels", labelKey: "param.outChannels", kind: "int", min: 1, max: 4096 },
      { key: "kernel_size", labelKey: "param.kernelSize", kind: "int", min: 1, max: 32 },
      { key: "stride", labelKey: "param.stride", kind: "int", min: 1, max: 32 },
      { key: "padding", labelKey: "param.padding", kind: "int", min: 0, max: 32 },
      { key: "dilation", labelKey: "param.dilation", kind: "int", min: 1, max: 8 },
      { key: "groups", labelKey: "param.groups", kind: "int", min: 1, max: 4096 },
    ],
    defaultParams: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  MaxPool2d: {
    type: "MaxPool2d",
    category: "conv",
    params: [
      { key: "kernel_size", labelKey: "param.kernelSize", kind: "int", min: 1, max: 32 },
      { key: "stride", labelKey: "param.stride", kind: "int", min: 0, max: 32, zeroMeansKernel: true },
      { key: "padding", labelKey: "param.padding", kind: "int", min: 0, max: 32 },
    ],
    defaultParams: { kernel_size: 2, stride: 0, padding: 0 },
  },
  AvgPool2d: {
    type: "AvgPool2d",
    category: "conv",
    params: [
      { key: "kernel_size", labelKey: "param.kernelSize", kind: "int", min: 1, max: 32 },
      { key: "stride", labelKey: "param.stride", kind: "int", min: 0, max: 32, zeroMeansKernel: true },
      { key: "padding", labelKey: "param.padding", kind: "int", min: 0, max: 32 },
    ],
    defaultParams: { kernel_size: 2, stride: 0, padding: 0 },
  },
  AdaptiveAvgPool2d: {
    type: "AdaptiveAvgPool2d",
    category: "conv",
    params: [{ key: "output_size", labelKey: "param.outputSize", kind: "int", min: 1, max: 256 }],
    defaultParams: { output_size: 1 },
  },
  BatchNorm2d: {
    type: "BatchNorm2d",
    category: "norm",
    params: [
      { key: "eps", labelKey: "param.eps", kind: "float", min: 0, max: 1, step: 0.00001 },
      { key: "momentum", labelKey: "param.momentum", kind: "float", min: 0, max: 1, step: 0.05 },
    ],
    defaultParams: { eps: 1e-5, momentum: 0.1 },
  },
  BatchNorm1d: {
    type: "BatchNorm1d",
    category: "norm",
    params: [
      { key: "eps", labelKey: "param.eps", kind: "float", min: 0, max: 1, step: 0.00001 },
      { key: "momentum", labelKey: "param.momentum", kind: "float", min: 0, max: 1, step: 0.05 },
    ],
    defaultParams: { eps: 1e-5, momentum: 0.1 },
  },
  LayerNorm: {
    type: "LayerNorm",
    category: "norm",
    params: [{ key: "normalized_shape", labelKey: "param.normalizedShape", kind: "int", min: 0, max: 1 << 20 }],
    defaultParams: { normalized_shape: 0 },
  },
  Embedding: {
    type: "Embedding",
    category: "seq",
    params: [
      { key: "num_embeddings", labelKey: "param.numEmbeddings", kind: "int", min: 2, max: 1 << 20 },
      { key: "embedding_dim", labelKey: "param.embeddingDim", kind: "int", min: 1, max: 4096 },
    ],
    defaultParams: { num_embeddings: 5000, embedding_dim: 64 },
  },
  PositionalEncoding: {
    type: "PositionalEncoding",
    category: "seq",
    params: [
      { key: "max_len", labelKey: "param.maxLen", kind: "int", min: 1, max: 8192 },
      { key: "mode", labelKey: "param.posMode", kind: "enum", choices: POS_MODES },
    ],
    defaultParams: { max_len: 128, mode: "learned" },
  },
  LSTM: {
    type: "LSTM",
    category: "seq",
    params: [
      { key: "hidden_size", labelKey: "param.hiddenSize", kind: "int", min: 1, max: 4096 },
      { key: "num_layers", labelKey: "param.numLayers", kind: "int", min: 1, max: 8 },
      { key: "bidirectional", labelKey: "param.bidirectional", kind: "bool" },
      { key: "return_sequences", labelKey: "param.returnSequences", kind: "bool" },
    ],
    defaultParams: { hidden_size: 128, num_layers: 1, bidirectional: false, return_sequences: true },
  },
  MultiHeadAttention: {
    type: "MultiHeadAttention",
    category: "seq",
    params: [
      { key: "num_heads", labelKey: "param.numHeads", kind: "int", min: 1, max: 64 },
      { key: "causal", labelKey: "param.causal", kind: "bool" },
      { key: "dropout", labelKey: "param.dropoutP", kind: "float", min: 0, max: 0.9, step: 0.05 },
    ],
    defaultParams: { num_heads: 4, causal: true, dropout: 0 },
  },
  ResidualAdd: {
    type: "ResidualAdd",
    category: "combine",
    params: [],
    multiInput: true,
    defaultParams: {},
  },
  Concat: {
    type: "Concat",
    category: "combine",
    params: [{ key: "dim", labelKey: "param.dim", kind: "int", min: -4, max: 3 }],
    multiInput: true,
    defaultParams: { dim: -1 },
  },
};

export const PALETTE_GROUPS: { category: NodeCategory; types: NodeType[] }[] = [
  { category: "io", types: ["Input", "Output"] },
  { category: "basic", types: ["Linear", "Activation", "Flatten", "Dropout"] },
  { category: "conv", types: ["Conv2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"] },
  { category: "norm", types: ["BatchNorm2d", "BatchNorm1d", "LayerNorm"] },
  { category: "seq", types: ["Embedding", "PositionalEncoding", "LSTM", "MultiHeadAttention"] },
  { category: "combine", types: ["ResidualAdd", "Concat"] },
];

export function specOf(type: string): NodeSpec | undefined {
  return NODE_SPECS[type as NodeType];
}

export function defaultParamsFor(type: NodeType): Record<string, unknown> {
  return { ...NODE_SPECS[type].defaultParams };
}
