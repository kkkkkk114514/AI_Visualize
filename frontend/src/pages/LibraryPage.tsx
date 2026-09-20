import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { apiGet } from "../api/rest";
import type { ModelKind, ModelsResponse } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";

const SECTION_ORDER: ModelKind[] = ["ml", "dl", "rl"];

export default function LibraryPage() {
  const { t } = useTranslation();
  const [groups, setGroups] = useState<ModelsResponse["groups"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await apiGet<ModelsResponse>("/api/models");
      setGroups(data.groups);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="app-shell">
      <AppHeader
        left={
          <>
            <span className="app-header__title">{t("app.title")}</span>
            <span className="app-header__subtitle">{t("app.subtitle")}</span>
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
      <main className="page">
        <div className="page__intro">
          <div>
            <h1>{t("library.title")}</h1>
            <p>{t("library.intro")}</p>
          </div>
          <Link className="btn" to="/lab/new">
            {t("library.newModel")}
          </Link>
        </div>

        {error ? (
          <div className="error-state">
            <span>
              {t("library.loadError")}：{error}
            </span>
            <button type="button" className="btn" onClick={() => void load()}>
              {t("library.retry")}
            </button>
          </div>
        ) : null}

        <div className="sections">
          {SECTION_ORDER.map((kind) => {
            const count = groups?.[kind]?.length ?? 0;
            return (
              <section key={kind} className="section">
                <div className="section__header">
                  <h2>{t(`library.sections.${kind}`)}</h2>
                  {count > 0 ? <span className="section__count">{count}</span> : null}
                </div>
                {count === 0 ? <p className="section__empty">{t("library.empty")}</p> : null}
              </section>
            );
          })}
        </div>
      </main>
    </div>
  );
}
