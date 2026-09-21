import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { PanelId } from "../panels/registry";

/**
 * 面板收起/恢复状态（docs/02 §9.2「面板收起」）。
 * 注意：Allotment 只把 `type.displayName === "Allotment.Pane"` 的直接子元素当 pane 配置，
 * minSize/preferredSize/visible 必须写在调用点的 <Allotment.Pane> 上——包一层自定义
 * 组件会被当成普通子元素，这些 prop 会被静默忽略。
 * `PanelId` 全集与装配表在 `panels/registry.tsx`（docs/02 §13.6）。
 */

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
