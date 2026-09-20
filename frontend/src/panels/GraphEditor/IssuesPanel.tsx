import { useCallback, useMemo } from "react";
import { useReactFlow } from "@xyflow/react";
import { useTranslation } from "react-i18next";

import type { GraphIssueDto } from "../../api/graph";
import { useGraphStore } from "../../stores/graphStore";
import { formatIssueArgs, formatParams } from "./format";

interface Row {
  key: string;
  severity: "error" | "warning";
  messageKey: string;
  args: Record<string, unknown>;
  detail?: string;
  nodeId?: string;
  edgeId?: string;
}

export function IssuesPanel() {
  const { t } = useTranslation();
  const localIssues = useGraphStore((state) => state.localIssues);
  const infer = useGraphStore((state) => state.infer);
  const nodes = useGraphStore((state) => state.nodes);
  const { setCenter } = useReactFlow();

  const rows = useMemo<Row[]>(() => {
    const merged = new Map<string, Row>();
    const push = (row: Row) => {
      const key = `${row.severity}:${row.messageKey}:${row.nodeId ?? row.edgeId ?? ""}`;
      const existing = merged.get(key);
      if (!existing || (existing.args && Object.keys(existing.args).length === 0)) {
        merged.set(key, row);
      }
    };
    for (const issue of localIssues) {
      push({
        key: `${issue.code}:${issue.nodeId ?? issue.edgeId ?? "graph"}`,
        severity: issue.severity,
        messageKey: issue.messageKey,
        args: issue.args ?? {},
        nodeId: issue.nodeId,
        edgeId: issue.edgeId,
      });
    }
    for (const issue of infer.errors as GraphIssueDto[]) {
      push({
        key: `infer:${issue.code}:${issue.node_id ?? issue.edge_id ?? "graph"}`,
        severity: "error",
        messageKey: issue.message_key,
        args: issue.args ?? {},
        detail: issue.detail,
        nodeId: issue.node_id,
        edgeId: issue.edge_id,
      });
    }
    for (const issue of infer.warnings as GraphIssueDto[]) {
      push({
        key: `infer:${issue.code}:${issue.node_id ?? issue.edge_id ?? "graph"}`,
        severity: "warning",
        messageKey: issue.message_key,
        args: issue.args ?? {},
        detail: issue.detail,
        nodeId: issue.node_id,
        edgeId: issue.edge_id,
      });
    }
    return [...merged.values()];
  }, [localIssues, infer.errors, infer.warnings]);

  const focusNode = useCallback(
    (nodeId: string) => {
      const node = nodes.find((item) => item.id === nodeId);
      if (!node) return;
      setCenter(node.position.x + 110, node.position.y + 60, { zoom: 1, duration: 350 });
    },
    [nodes, setCenter],
  );

  const untouched = localIssues.length === 0 && infer.errors.length === 0 && infer.warnings.length === 0;

  return (
    <div className="issues">
      <div className="issues__summary">
        <span className={`issues__badge ${infer.errors.length ? "issues__badge--error" : "issues__badge--ok"}`}>
          {infer.errors.length ? t("editor.issueErrors", { count: infer.errors.length }) : t("editor.noErrors")}
        </span>
        {infer.warnings.length ? (
          <span className="issues__badge issues__badge--warn">
            {t("editor.issueWarnings", { count: infer.warnings.length })}
          </span>
        ) : null}
        <span className="issues__meta">
          {infer.pending
            ? t("editor.checking")
            : infer.elapsedMs !== null
              ? t("editor.checkMs", { ms: Math.round(infer.elapsedMs) })
              : t("editor.notChecked")}
        </span>
      </div>

      {infer.requestError ? <div className="issues__network">{t("editor.networkError", { detail: infer.requestError })}</div> : null}

      {untouched ? <div className="issues__empty">{t("editor.noIssues")}</div> : null}

      <ul className="issues__list">
        {rows.map((row) => (
          <li
            key={row.key}
            className={`issues__row issues__row--${row.severity} ${row.nodeId ? "issues__row--link" : ""}`}
            onClick={row.nodeId ? () => focusNode(row.nodeId!) : undefined}
          >
            <span className="issues__dot" />
            <div className="issues__body">
              <div className="issues__text">
                {t(row.messageKey, { ...formatIssueArgs(row.args, t), defaultValue: row.messageKey })}
              </div>
              <div className="issues__origin">
                {row.nodeId ? `${nodes.find((n) => n.id === row.nodeId)?.data.nodeType ?? ""} · ${row.nodeId}` : null}
                {row.edgeId ? `${t("editor.edge")} ${row.edgeId}` : null}
                {row.detail ? <span className="issues__detail" title={row.detail} /> : null}
              </div>
            </div>
          </li>
        ))}
      </ul>

      <div className="issues__footer">
        <span>{t("editor.totalParams")}</span>
        <strong title={infer.totalParams !== null ? String(infer.totalParams) : undefined}>
          {formatParams(infer.totalParams)}
        </strong>
      </div>
    </div>
  );
}
