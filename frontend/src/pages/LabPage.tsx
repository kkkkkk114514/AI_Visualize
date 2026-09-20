import { Allotment } from "allotment";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { Panel } from "../components/Panel";

export default function LabPage() {
  const { t } = useTranslation();

  return (
    <div className="app-shell">
      <AppHeader
        left={
          <>
            <Link className="app-header__back" to="/">
              {t("nav.backToLibrary")}
            </Link>
            <span className="app-header__title">{t("lab.title")}</span>
          </>
        }
        actions={
          <>
            <DeviceBadge />
            <ConnectionStatus />
            <LanguageSwitch />
          </>
        }
      />
      <div className="run-bar">
        <button type="button" className="btn btn--primary" disabled>
          {t("run.controls.start")}
        </button>
        <button type="button" className="btn" disabled>
          {t("run.controls.pause")}
        </button>
        <button type="button" className="btn" disabled>
          {t("run.controls.stop")}
        </button>
        <span className="run-bar__status">{t("run.status.idle")}</span>
      </div>
      <div className="lab-body">
        <Allotment>
          <Allotment.Pane minSize={220} preferredSize="22%">
            <Panel title={t("lab.zones.config")} empty={t("lab.empty.config")} />
          </Allotment.Pane>
          <Allotment.Pane minSize={420}>
            <Panel title={t("lab.zones.main")} empty={t("lab.empty.main")} />
          </Allotment.Pane>
          <Allotment.Pane minSize={260} preferredSize="26%">
            <Panel title={t("lab.zones.metrics")} empty={t("lab.empty.metrics")} />
          </Allotment.Pane>
        </Allotment>
      </div>
    </div>
  );
}
