import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  actions?: ReactNode;
  empty?: string;
  children?: ReactNode;
}

export function Panel({ title, actions, empty, children }: PanelProps) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h2 className="panel__title">{title}</h2>
        {actions}
      </div>
      <div className="panel__body">
        {children ?? (empty ? <p className="empty-state">{empty}</p> : null)}
      </div>
    </section>
  );
}
