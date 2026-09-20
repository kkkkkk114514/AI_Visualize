import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import type { RunSummary } from "../../api/types";
import { isActiveStatus, useRunStore } from "../../stores/runStore";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(run: RunSummary): string {
  if (typeof run.elapsed_s === "number" && run.elapsed_s > 0) {
    const total = Math.floor(run.elapsed_s);
    return total >= 60 ? `${Math.floor(total / 60)}m${total % 60}s` : `${total}s`;
  }
  if (run.started_at && run.finished_at) {
    const seconds = (new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()) / 1000;
    if (Number.isFinite(seconds) && seconds > 0) return `${Math.round(seconds)}s`;
  }
  return "—";
}

export function RunList() {
  const { t } = useTranslation();
  const history = useRunStore((state) => state.history);
  const historyTotal = useRunStore((state) => state.historyTotal);
  const historyError = useRunStore((state) => state.historyError);
  const currentId = useRunStore((state) => state.current?.id ?? null);
  const replayId = useRunStore((state) => state.replay?.id ?? null);
  const loadHistory = useRunStore((state) => state.loadHistory);
  const selectRun = useRunStore((state) => state.selectRun);
  const removeRun = useRunStore((state) => state.removeRun);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  if (historyError) {
    return (
      <div className="runs">
        <div className="runs__error">{historyError}</div>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="runs">
        <div className="runs__empty">{t("run.list.empty")}</div>
      </div>
    );
  }

  return (
    <div className="runs">
      <ul className="runs__list">
        {history.map((run) => {
          const selected = run.id === replayId || run.id === currentId;
          return (
            <li
              key={run.id}
              className={`runs__row ${selected ? "runs__row--selected" : ""}`}
              onClick={() => void selectRun(run.id)}
            >
              <div className="runs__head">
                <span className="runs__name" title={run.id}>
                  {run.name}
                </span>
                <span className={`run-state run-state--${run.status}`}>{t(`run.status.${run.status}`)}</span>
              </div>
              <div className="runs__meta">
                <span>{run.dataset_id}</span>
                <span>
                  {t("run.metrics.best")} {run.best_metric === null ? "—" : run.best_metric.toFixed(4)}
                </span>
                <span>
                  {t("run.metrics.step")} {run.total_steps}
                </span>
                <span>{formatDuration(run)}</span>
                <span>
                  {t("run.list.finishedAt")} {run.finished_at ? formatTime(run.finished_at) : "—"}
                </span>
              </div>
              <div className="runs__actions">
                {isActiveStatus(run.status) ? (
                  <span className="runs__live">{t("run.list.live")}</span>
                ) : (
                  <button
                    type="button"
                    className="btn btn--ghost runs__delete"
                    onClick={(event) => {
                      event.stopPropagation();
                      void removeRun(run.id);
                    }}
                  >
                    {t("run.list.delete")}
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="runs__footer">
        {t("run.list.count", { count: history.length, total: historyTotal })}
      </div>
    </div>
  );
}
