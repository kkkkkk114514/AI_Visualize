import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

/**
 * 面板收起/恢复状态（docs/02 §9.2「面板收起」）。
 * 注意：Allotment 只把 `type.displayName === "Allotment.Pane"` 的直接子元素当 pane 配置，
 * minSize/preferredSize/visible 必须写在调用点的 <Allotment.Pane> 上——包一层自定义
 * 组件会被当成普通子元素，这些 prop 会被静默忽略。
 */

export type PanelId = "nodes" | "config" | "graph" | "metrics" | "probe" | "check" | "nodeHelp" | "runs";

/** 恢复条里的排列顺序 = 布局顺序（左列 → 中列 → 右列） */
export const PANEL_ORDER: PanelId[] = ["nodes", "config", "graph", "metrics", "probe", "check", "nodeHelp", "runs"];

const ZONE_KEY: Record<PanelId, string> = {
  nodes: "lab.zones.nodes",
  config: "lab.zones.config",
  graph: "lab.zones.graph",
  metrics: "lab.zones.metrics",
  probe: "lab.zones.probe",
  check: "lab.zones.check",
  nodeHelp: "lab.zones.nodeHelp",
  runs: "lab.zones.runs",
};

export interface PanelLayoutApi {
  isHidden: (id: PanelId) => boolean;
  collapse: (id: PanelId) => void;
  restore: (id: PanelId) => void;
}

const INERT: PanelLayoutApi = { isHidden: () => false, collapse: () => {}, restore: () => {} };

const PanelLayoutContext = createContext<PanelLayoutApi>(INERT);

/** 收起状态不持久化：刷新回到全展开（docs/02 §9.2「面板收起」）。 */
export function usePanelLayoutState(): PanelLayoutApi {
  const [hidden, setHidden] = useState<ReadonlySet<PanelId>>(() => new Set<PanelId>());
  const collapse = useCallback(
    (id: PanelId) => setHidden((prev) => (prev.has(id) ? prev : new Set(prev).add(id))),
    [],
  );
  const restore = useCallback(
    (id: PanelId) =>
      setHidden((prev) => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      }),
    [],
  );
  return useMemo<PanelLayoutApi>(
    () => ({ isHidden: (id) => hidden.has(id), collapse, restore }),
    [hidden, collapse, restore],
  );
}

export function PanelLayoutProvider({ value, children }: { value: PanelLayoutApi; children: ReactNode }) {
  return <PanelLayoutContext.Provider value={value}>{children}</PanelLayoutContext.Provider>;
}

export function usePanelLayout(): PanelLayoutApi {
  return useContext(PanelLayoutContext);
}

/** 已收起面板的恢复条；没有收起项时不渲染。 */
export function HiddenPanelsStrip() {
  const { t } = useTranslation();
  const { isHidden, restore } = usePanelLayout();
  const hidden = PANEL_ORDER.filter((id) => isHidden(id));
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
          title={t("lab.panels.restore", { name: t(ZONE_KEY[id]) })}
        >
          <svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M6 2.5v7M2.5 6h7" />
          </svg>
          {t(ZONE_KEY[id])}
        </button>
      ))}
    </div>
  );
}
