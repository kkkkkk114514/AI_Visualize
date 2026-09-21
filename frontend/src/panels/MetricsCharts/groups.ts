import type { MetricName, ModelKind } from "../../api/types";

/**
 * 曲线分组（docs/02 §5.4 / §13.6）：按模型类别与算法挑主图曲线和附加指标，
 * 所有曲线名都取自后端 `METRIC_NAMES`（20 项）。
 */

export const COLORS: Record<MetricName, string> = {
  loss: "#4c9aff",
  acc: "#3fb950",
  val_loss: "#8ab4f8",
  val_acc: "#7ee787",
  lr: "#d29922",
  grad_norm: "#bc8cff",
  throughput: "#39c5cf",
  vram_mb: "#f85149",
  margin: "#f0883e",
  n_sv: "#db61a2",
  depth: "#7b8cff",
  leaves: "#2fbf9b",
  n_nodes: "#b1a0ff",
  n_trees: "#4fb3d9",
  reward: "#56d364",
  q_delta: "#e3b341",
  episode_reward: "#3fb950",
  epsilon: "#d29922",
  episode_steps: "#79c0ff",
  success: "#f778ba",
};

export interface SeriesSpec {
  name: MetricName;
  axis: "left" | "right";
}

export interface RightAxisSpec {
  scale: string;
  /** 固定范围（ε 恒 0..1），不填则自适应 */
  range?: [number, number];
  size: number;
}

export interface MetricGroup {
  main: SeriesSpec[];
  right?: RightAxisSpec;
  /** 附加小图的可选项（勾选显示） */
  optional: MetricName[];
  /** 默认勾选的项数（取 optional 前 n 个） */
  optionalDefault: number;
}

const ACC_RIGHT: RightAxisSpec = { scale: "acc", size: 40 };
const EPSILON_RIGHT: RightAxisSpec = { scale: "epsilon", range: [0, 1], size: 40 };

/** 算法专属附加指标：按后端实际上报的指标给（见 `app/algos/ml.py`）。 */
const ALGO_OPTIONAL: Record<string, MetricName[]> = {
  linear_regression: ["lr"],
  logistic_regression: ["lr"],
  svm: ["margin", "n_sv", "lr"],
  decision_tree: ["depth", "leaves", "n_nodes"],
  random_forest: ["n_trees", "depth", "leaves"],
};

const DL_OPTIONAL: MetricName[] = ["lr", "grad_norm", "throughput", "vram_mb"];

export function metricGroups(kind: ModelKind, algo: string | null): MetricGroup {
  if (kind === "rl") {
    return {
      main: [
        { name: "episode_reward", axis: "left" },
        { name: "epsilon", axis: "right" },
      ],
      right: EPSILON_RIGHT,
      optional: ["reward", "episode_steps", "success", "q_delta"],
      optionalDefault: 1,
    };
  }
  const optional = kind === "ml" ? ALGO_OPTIONAL[algo ?? ""] ?? [] : DL_OPTIONAL;
  return {
    main: [
      { name: "loss", axis: "left" },
      { name: "val_loss", axis: "left" },
      { name: "acc", axis: "right" },
      { name: "val_acc", axis: "right" },
    ],
    right: ACC_RIGHT,
    optional,
    optionalDefault: 1,
  };
}
