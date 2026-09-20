import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { apiGet } from "../api/rest";
import type { ModelKind, ModelsResponse } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { localizedText } from "../i18n/localized";

const SECTION_ORDER: ModelKind[] = ["ml", "dl", "rl"];

export default function LibraryPage() {
  const { t, i18n } = useTranslation();
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
            const models = groups?.[kind] ?? [];
            return (
              <section key={kind} className="section">
                <div className="section__header">
                  <h2>{t(`library.sections.${kind}`)}</h2>
                  {models.length > 0 ? <span className="section__count">{models.length}</span> : null}
                </div>
                {models.length === 0 ? (
                  <p className="section__empty">{t("library.empty")}</p>
                ) : (
                  <div className="model-grid">
                    {models.map((model) => (
                      <Link key={model.id} className="model-card" to={`/lab/${encodeURIComponent(model.id)}`}>
                        <div className="model-card__name">{localizedText(model.name, i18n.language) || model.id}</div>
                        <div className="model-card__desc">{localizedText(model.desc, i18n.language)}</div>
                        <div className="model-card__meta">
                          <span className={`chip chip--${model.kind}`}>{model.kind.toUpperCase()}</span>
                          <span className="chip chip--muted">{t(`library.source.${model.source}`)}</span>
                        </div>
                      </Link>
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      </main>
    </div>
  );
}
