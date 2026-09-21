import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { useAlgoStore } from "../../stores/algoStore";
import { useGraphStore } from "../../stores/graphStore";
import { isActiveStatus, useRunStore } from "../../stores/runStore";

function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

export function RunControls() {
  const { t } = useTranslation();
  const datasets = useRunStore((state) => state.datasets);
  const datasetId = useRunStore((state) => state.datasetId);
  const current = useRunStore((state) => state.current);
  const status = useRunStore((state) => state.status);
  const step = useRunStore((state) => state.step);
  const epoch = useRunStore((state) => state.epoch);
  const plannedSteps = useRunStore((state) => state.plannedSteps);
  const device = useRunStore((state) => state.device);
  const elapsedS = useRunStore((state) => state.elapsedS);
  const bestMetric = useRunStore((state) => state.bestMetric);
  const starting = useRunStore((state) => state.starting);
  const startError = useRunStore((state) => state.startError);
  const runError = useRunStore((state) => state.runError);
  const logs = useRunStore((state) => state.logs);
  const tickElapsed = useRunStore((state) => state.tickElapsed);
  const start = useRunStore((state) => state.start);
  const pause = useRunStore((state) => state.pause);
  const resume = useRunStore((state) => state.resume);
  const stop = useRunStore((state) => state.stop);

  const readOnly = useGraphStore((state) => state.readOnly);
  const graphSource = useGraphStore((state) => state.source);
  const inferErrors = useGraphStore((state) => state.infer.errors);
  const localIssues = useGraphStore((state) => state.localIssues);
  const modelId = useGraphStore((state) => state.meta.id);

  // ML / RL 的 spec 不进 graphStore（docs/02 §13.6）：启动源与数据集都改看 algoStore
  const algoSpec = useAlgoStore((state) => state.spec);
  const activeKind = algoSpec?.kind ?? "dl";

  const replay = useRunStore((state) => state.replay);
  const replaying = replay !== null;

  const running = isActiveStatus(status);
  const blocked =
    activeKind === "dl" && (localIssues.some((issue) => issue.severity === "error") || inferErrors.length > 0);
  const effectiveDatasetId = algoSpec?.dataset_id ?? datasetId;
  const dataset = datasets?.find((item) => item.id === effectiveDatasetId) ?? null;

  const handleStart = () => {
    if (activeKind === "dl") {
      void start({
        kind: "dl",
        graph: useGraphStore.getState().toGraphIR(),
        modelId: graphSource === "preset" ? modelId : undefined,
      });
      return;
    }
    void start({ kind: "algo", ...useAlgoStore.getState().runSource() });
  };

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => tickElapsed(), 500);
    return () => window.clearInterval(timer);
  }, [running, tickElapsed]);

  const latestLog = logs.length > 0 ? logs[logs.length - 1] : null;
  const message = runError
    ? t(runError.messageKey, { ...runError.args, defaultValue: runError.messageKey })
    : startError
      ? t(startError.messageKey, { ...startError.args, defaultValue: startError.messageKey })
      : latestLog
        ? t(latestLog.key, { ...latestLog.args, defaultValue: latestLog.key })
        : null;

  return (
    <div className="run-bar">
      <div className="run-bar__buttons">
        <button
          type="button"
          className="btn btn--primary"
          disabled={running || starting || blocked || replaying}
          onClick={handleStart}
          title={
            replaying
              ? t("run.replayNote", { run: replay.id })
              : blocked
                ? t("run.blockedHint")
                : activeKind === "dl" && readOnly
                  ? t("run.presetHint")
                  : undefined
          }
        >
          {starting ? t("run.controls.starting") : t("run.controls.start")}
        </button>
        {status === "paused" ? (
          <button type="button" className="btn" onClick={() => void resume()}>
            {t("run.controls.resume")}
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            disabled={!running || status === "created"}
            onClick={() => void pause()}
          >
            {t("run.controls.pause")}
          </button>
        )}
        <button type="button" className="btn" disabled={!running} onClick={() => void stop()}>
          {t("run.controls.stop")}
        </button>
      </div>

      <span className={`run-state run-state--${status}`}>{t(`run.status.${status}`)}</span>

      {current ? (
        <span className="run-bar__metrics">
          <span className="run-bar__metric">
            <span className="run-bar__metric-label">{t("run.metrics.step")}</span>
            <strong>
              {step}
              {plannedSteps ? ` / ${plannedSteps}` : ""}
            </strong>
          </span>
          <span className="run-bar__metric">
            <span className="run-bar__metric-label">{t("run.metrics.epoch")}</span>
            <strong>{epoch}</strong>
          </span>
          <span className="run-bar__metric">
            <span className="run-bar__metric-label">{t("run.metrics.elapsed")}</span>
            <strong className="run-bar__mono">{formatElapsed(elapsedS)}</strong>
          </span>
          <span className="run-bar__metric">
            <span className="run-bar__metric-label">{t("run.metrics.best")}</span>
            <strong>{bestMetric === null ? "—" : bestMetric.toFixed(4)}</strong>
          </span>
          <span className="badge badge--muted">{device ?? effectiveDatasetId ?? "—"}</span>
        </span>
      ) : null}

      <span className="run-bar__spacer" />

      {message ? (
        <span className={`run-bar__note ${runError || startError ? "run-bar__note--error" : ""}`} title={runError?.detail}>
          {message}
        </span>
      ) : replaying ? (
        <span className="run-bar__note">{t("run.replayNote", { run: replay.id })}</span>
      ) : blocked ? (
        <span className="run-bar__note">{t("run.blockedHint")}</span>
      ) : activeKind === "dl" && readOnly ? (
        <span className="run-bar__note">{t("run.presetHint")}</span>
      ) : (
        <span className="run-bar__note">
          {dataset ? `${dataset.id} · ${t("run.dataset.ready")}` : t("run.dataset.missing")}
        </span>
      )}
    </div>
  );
}
