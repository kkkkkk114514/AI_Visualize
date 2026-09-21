import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { usePanelLayout } from "./panelLayout";
import type { PanelId } from "../panels/registry";

interface PanelProps {
  /** 传 id 才显示「收起」按钮（见 docs/02 §9.2「面板收起」） */
  id?: PanelId;
  title: string;
  actions?: ReactNode;
  empty?: string;
  children?: ReactNode;
}

export function Panel({ id, title, actions, empty, children }: PanelProps) {
  const { t } = useTranslation();
  const { collapse } = usePanelLayout();
  return (
    <section className="panel">
      <div className="panel__header">
        <h2 className="panel__title">{title}</h2>
        {actions}
        <span className="panel__gap" />
        {id ? (
          <button
            type="button"
            className="panel__fold"
            onClick={() => collapse(id)}
            title={t("lab.panels.collapse", { name: title })}
            aria-label={t("lab.panels.collapse", { name: title })}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M2.5 6h7" />
            </svg>
          </button>
        ) : null}
      </div>
      <div className="panel__body">
        {children ?? (empty ? <p className="empty-state">{empty}</p> : null)}
      </div>
    </section>
  );
}
