import { useCallback, useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { Allotment } from "allotment";
import { useTranslation } from "react-i18next";
import { Link, useParams } from "react-router-dom";

import { fetchModelDetail } from "../api/graph";
import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { Panel } from "../components/Panel";
import { GraphEditor } from "../panels/GraphEditor/GraphEditor";
import { IssuesPanel } from "../panels/GraphEditor/IssuesPanel";
import { NodeDescriptions } from "../panels/GraphEditor/NodeDescriptions";
import { NodePalette } from "../panels/GraphEditor/NodePalette";
import { useGraphInfer } from "../panels/GraphEditor/useGraphInfer";
import { emptyGraph } from "../graph/ir";
import { localizedText } from "../i18n/localized";
import { useGraphStore } from "../stores/graphStore";

export default function LabPage() {
  const { t, i18n } = useTranslation();
  const { graphId = "new" } = useParams();
  const load = useGraphStore((state) => state.load);
  const cloneAsCopy = useGraphStore((state) => state.cloneAsCopy);
  const readOnly = useGraphStore((state) => state.readOnly);
  const source = useGraphStore((state) => state.source);
  const dirty = useGraphStore((state) => state.dirty);
  const meta = useGraphStore((state) => state.meta);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setLoadError(null);
    if (graphId === "new") {
      load(emptyGraph(), { readOnly: false, source: "new" });
      setStatus("ready");
      return () => {
        cancelled = true;
      };
    }
    fetchModelDetail(graphId)
      .then((detail) => {
        if (cancelled) return;
        load(detail.graph, { readOnly: detail.source === "preset", source: detail.source === "preset" ? "preset" : "user" });
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setLoadError(error instanceof Error ? error.message : String(error));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [graphId, load]);

  useGraphInfer(status === "ready");

  const title = useMemo(() => {
    const localized = localizedText(meta.name, i18n.language);
    return localized || t("lab.untitled");
  }, [meta.name, i18n.language, t]);

  const clone = useCallback(() => {
    cloneAsCopy();
  }, [cloneAsCopy]);

  return (
    <div className="app-shell">
      <AppHeader
        left={
          <>
            <Link className="app-header__back" to="/">
              {t("nav.backToLibrary")}
            </Link>
            <span className="app-header__title">{title}</span>
            {readOnly ? <span className="badge badge--muted">{t("lab.readOnly")}</span> : null}
            {dirty && !readOnly ? <span className="badge badge--warn">{t("lab.unsaved")}</span> : null}
          </>
        }
        actions={
          <>
            {readOnly ? (
              <button type="button" className="btn btn--primary" onClick={clone}>
                {t("lab.cloneAsCopy")}
              </button>
            ) : null}
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
        <span className="run-bar__spacer" />
        {readOnly ? (
          <span className="run-bar__note">{t("lab.presetGraph")}</span>
        ) : source === "new" ? (
          <span className="run-bar__note">{t("lab.newGraph")}</span>
        ) : null}
      </div>
      <div className="lab-body">
        {status === "error" ? (
          <div className="error-state">
            <span>
              {t("lab.loadError")}：{loadError}
            </span>
            <Link className="btn" to="/">
              {t("nav.backToLibrary")}
            </Link>
          </div>
        ) : status === "loading" ? (
          <div className="lab-loading">{t("common.loading")}</div>
        ) : (
          <ReactFlowProvider>
            <Allotment>
              <Allotment.Pane minSize={220} preferredSize="18%">
                <Panel title={t("lab.zones.nodes")}>
                  <NodePalette />
                </Panel>
              </Allotment.Pane>
              <Allotment.Pane minSize={460}>
                <Panel title={t("lab.zones.graph")}>
                  <GraphEditor />
                </Panel>
              </Allotment.Pane>
              <Allotment.Pane minSize={280} preferredSize="24%">
                <Allotment vertical>
                  <Allotment.Pane minSize={120}>
                    <Panel title={t("lab.zones.check")}>
                      <IssuesPanel />
                    </Panel>
                  </Allotment.Pane>
                  <Allotment.Pane minSize={140}>
                    <Panel title={t("lab.zones.nodeHelp")}>
                      <NodeDescriptions />
                    </Panel>
                  </Allotment.Pane>
                </Allotment>
              </Allotment.Pane>
            </Allotment>
          </ReactFlowProvider>
        )}
      </div>
    </div>
  );
}
