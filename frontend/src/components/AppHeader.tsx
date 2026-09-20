import type { ReactNode } from "react";

interface AppHeaderProps {
  left: ReactNode;
  actions?: ReactNode;
}

export function AppHeader({ left, actions }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__left">{left}</div>
      {actions ? <div className="app-header__actions">{actions}</div> : null}
    </header>
  );
}
