import type { ComponentType } from "react";
import { useTranslation } from "react-i18next";

import type { ModelKind } from "../api/types";
import { usePanelLayout } from "../components/panelLayout";
import { useAlgoStore } from "../stores/algoStore";
import { GraphEditor } from "./GraphEditor/GraphEditor";
import { IssuesPanel } from "./GraphEditor/IssuesPanel";
import { NodeDescriptions } from "./GraphEditor/NodeDescriptions";
import { NodePalette } from "./GraphEditor/NodePalette";
import { MetricsCharts } from "./MetricsCharts/MetricsCharts";
import { ProbeViewer } from "./ProbeViewer/ProbeViewer";
import { RunList } from "./RunList/RunList";
import { TrainingConfig } from "./TrainingConfig/TrainingConfig";
import { DecisionBoundaryCanvas } from "./ml/DecisionBoundaryCanvas";
import { ModelConfigForm } from "./ml/ModelConfigForm";
import { GridWorldCanvas } from "./rl/GridWorldCanvas";
import { RLConfigForm } from "./rl/RLConfigForm";

/**
 * 面板注册表（docs/02 §13.6）：`PanelId` 全集 + 三类模型的面板装配表。
 * 工作台按 `kind` 装配左 / 中 / 右三列，新增算法只注册面板、不改外壳。
 */

export type PanelId =
  | "nodes"
  | "config"
  | "graph"
  | "boundary"
  | "grid"
  | "metrics"
  | "probe"
  | "check"
  | "nodeHelp"
  | "runs";

export type ColumnKey = "left" | "center" | "right";

export const COLUMN_KEYS: ColumnKey[] = ["left", "center", "right"];

/** 恢复条顺序 = 布局顺序（左列 → 中列 → 右列，docs/02 §9.2「面板收起」） */
export const PANEL_ORDER: PanelId[] = [
  "nodes",
  "config",
  "graph",
  "boundary",
  "grid",
  "metrics",
  "probe",
  "check",
  "nodeHelp",
  "runs",
];

export const PANEL_TITLE_KEY: Record<PanelId, string> = {
  nodes: "lab.zones.nodes",
  config: "lab.zones.config",
  graph: "lab.zones.graph",
  boundary: "lab.zones.boundary",
  grid: "lab.zones.grid",
  metrics: "lab.zones.metrics",
  probe: "lab.zones.probe",
  check: "lab.zones.check",
  nodeHelp: "lab.zones.nodeHelp",
  runs: "lab.zones.runs",
};

export interface PanelSpec {
  id: PanelId;
  titleKey: string;
  component: ComponentType;
  column: ColumnKey;
}

export const panelRegistry: Record<ModelKind, PanelSpec[]> = {
  dl: [
    { id: "nodes", titleKey: PANEL_TITLE_KEY.nodes, component: NodePalette, column: "left" },
    { id: "config", titleKey: PANEL_TITLE_KEY.config, component: TrainingConfig, column: "left" },
    { id: "graph", titleKey: PANEL_TITLE_KEY.graph, component: GraphEditor, column: "center" },
    { id: "metrics", titleKey: PANEL_TITLE_KEY.metrics, component: MetricsCharts, column: "center" },
    { id: "probe", titleKey: PANEL_TITLE_KEY.probe, component: ProbeViewer, column: "center" },
    { id: "check", titleKey: PANEL_TITLE_KEY.check, component: IssuesPanel, column: "right" },
    { id: "nodeHelp", titleKey: PANEL_TITLE_KEY.nodeHelp, component: NodeDescriptions, column: "right" },
    { id: "runs", titleKey: PANEL_TITLE_KEY.runs, component: RunList, column: "right" },
  ],
  ml: [
    { id: "config", titleKey: PANEL_TITLE_KEY.config, component: ModelConfigForm, column: "left" },
    { id: "boundary", titleKey: PANEL_TITLE_KEY.boundary, component: DecisionBoundaryCanvas, column: "center" },
    { id: "metrics", titleKey: PANEL_TITLE_KEY.metrics, component: MetricsCharts, column: "center" },
    { id: "runs", titleKey: PANEL_TITLE_KEY.runs, component: RunList, column: "right" },
  ],
  rl: [
    { id: "config", titleKey: PANEL_TITLE_KEY.config, component: RLConfigForm, column: "left" },
    { id: "grid", titleKey: PANEL_TITLE_KEY.grid, component: GridWorldCanvas, column: "center" },
    { id: "metrics", titleKey: PANEL_TITLE_KEY.metrics, component: MetricsCharts, column: "center" },
    { id: "runs", titleKey: PANEL_TITLE_KEY.runs, component: RunList, column: "right" },
  ],
};

/** 面板尺寸：必须写在调用点的 `<Allotment.Pane>` 上（docs/02 §9.2 实现注意）。 */
export const PANEL_SIZE: Record<PanelId, { minSize: number; preferredSize?: number | string }> = {
  nodes: { minSize: 160 },
  config: { minSize: 240 },
  graph: { minSize: 200, preferredSize: "46%" },
  boundary: { minSize: 200, preferredSize: "56%" },
  grid: { minSize: 200, preferredSize: "56%" },
  metrics: { minSize: 150, preferredSize: "26%" },
  probe: { minSize: 150 },
  check: { minSize: 120, preferredSize: "30%" },
  nodeHelp: { minSize: 140, preferredSize: "34%" },
  runs: { minSize: 140 },
};

/** 列尺寸：ML / RL 的中列没有图编辑器，最小宽度放宽 */
export const COLUMN_SIZE: Record<
  ModelKind,
  Record<ColumnKey, { minSize: number; preferredSize?: number | string }>
> = {
  dl: {
    left: { minSize: 240, preferredSize: "20%" },
    center: { minSize: 460 },
    right: { minSize: 300, preferredSize: "26%" },
  },
  ml: {
    left: { minSize: 260, preferredSize: "24%" },
    center: { minSize: 420 },
    right: { minSize: 300, preferredSize: "26%" },
  },
  rl: {
    left: { minSize: 260, preferredSize: "24%" },
    center: { minSize: 420 },
    right: { minSize: 300, preferredSize: "26%" },
  },
};

/** 已收起面板的恢复条（只列当前模型类别装了的面板）；没有收起项时不渲染。 */
export function HiddenPanelsStrip() {
  const { t } = useTranslation();
  const { isHidden, restore } = usePanelLayout();
  const activeKind: ModelKind = useAlgoStore((state) => state.spec?.kind ?? "dl");
  const installed = new Set(panelRegistry[activeKind].map((spec) => spec.id));
  const hidden = PANEL_ORDER.filter((id) => installed.has(id) && isHidden(id));
  if (hidden.length === 0) return null;
  return (
    <div className="lab-hidden">
      <span className="lab-hidden__label">{t("lab.panels.hidden")}</span>
      {hidden.map((id) => (
        <button
          key={id}
          type="button"
          className="lab-hidden__chip"
          onClick={() => restore(id)}
          title={t("lab.panels.restore", { name: t(PANEL_TITLE_KEY[id]) })}
        >
          <svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M6 2.5v7M2.5 6h7" />
          </svg>
          {t(PANEL_TITLE_KEY[id])}
        </button>
      ))}
    </div>
  );
}
