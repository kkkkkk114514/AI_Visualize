import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchSnapshot } from "../../api/snapshots";
import type { SnapshotPayload } from "../../api/types";
import { parseStreamKey, useRunStore } from "../../stores/runStore";
import type { ColormapName } from "./paint";
import { COLORMAP_NAMES } from "./paint";
import { FeatureGridCanvas } from "./renderers/FeatureGridCanvas";
import { HeatmapCanvas } from "./renderers/HeatmapCanvas";
import { HistogramCanvas } from "./renderers/HistogramCanvas";

interface LoadedPayload {
  runId: string;
  id: string;
  payload: SnapshotPayload;
}

/**
 * 按快照 id 取 payload（实时 / 回放同一条路径）：
 * 换步时保留上一帧直到新帧到位，避免实时刷新时画面闪空。
 */
function useSnapshotPayload(runId: string | null, snapshotId: string | null) {
  const [loaded, setLoaded] = useState<LoadedPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !snapshotId) return;
    let cancelled = false;
    fetchSnapshot(runId, snapshotId)
      .then((payload) => {
        if (cancelled) return;
        setLoaded({ runId, id: snapshotId, payload });
        setError(null);
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, snapshotId]);

  const payload = loaded && loaded.runId === runId ? loaded.payload : null;
  return { payload, stale: payload !== null && loaded?.id !== snapshotId, error };
}

export function ProbeViewer() {
  const { t } = useTranslation();
  // 流列表 / 步列表 / payload 全部来自快照索引，runId 必须与索引同源：
  // 若改从 `current ?? replay` 推断，出现「已挂 run A + 回看 run B」时会拿 A 的 id 去取 B 的快照（404）。
  const runId = useRunStore((state) => state.snapshotRunId);
  const snapshots = useRunStore((state) => state.snapshots);
  const streamKey = useRunStore((state) => state.probeStream);
  const cursorStep = useRunStore((state) => state.cursorStep);
  const probeCount = useRunStore((state) => state.probeChoices.length);
  const probeEveryN = useRunStore((state) => state.probeEveryN);
  const setProbeStream = useRunStore((state) => state.setProbeStream);
  const setCursorStep = useRunStore((state) => state.setCursorStep);
  const [colormap, setColormap] = useState<ColormapName>("viridis");

  const streams = useMemo(() => Object.keys(snapshots), [snapshots]);
  const list = streamKey ? (snapshots[streamKey] ?? []) : [];
  // 游标（step）→ 快照下标：取 ≤ 游标的最近一条；游标早于第一条时取第一条（docs/02 §9.2「时间游标与回放」）
  const index = useMemo(() => {
    if (list.length === 0) return -1;
    if (cursorStep === null) return list.length - 1;
    let found = 0;
    for (let i = 0; i < list.length; i += 1) {
      if (list[i].step <= cursorStep) found = i;
      else break;
    }
    return found;
  }, [list, cursorStep]);
  const meta = index >= 0 ? list[index] : null;
  const stream = streamKey ? parseStreamKey(streamKey) : null;
  const { payload, stale, error } = useSnapshotPayload(runId, meta?.id ?? null);

  return (
    <div className="probe">
      <div className="probe__toolbar">
        <select
          className="probe__select"
          value={streamKey ?? ""}
          disabled={streams.length === 0}
          onChange={(event) => setProbeStream(event.target.value || null)}
          title={t("run.probe.streamHint")}
        >
          {streams.length === 0 ? <option value="">{t("run.probe.noStream")}</option> : null}
          {streams.map((key) => {
            const item = parseStreamKey(key);
            return (
              <option key={key} value={key}>
                {item ? `${item.nodeId} · ${t(`run.probe.kind.${item.kind}`)}` : key}
                {` (${snapshots[key].length})`}
              </option>
            );
          })}
        </select>

        <select
          className="probe__select probe__select--step"
          value={cursorStep === null ? "" : meta ? String(meta.step) : ""}
          disabled={list.length === 0}
          onChange={(event) =>
            setCursorStep(event.target.value === "" ? null : Number(event.target.value))
          }
          title={t("run.probe.stepHint")}
        >
          <option value="">{t("run.probe.followLatest")}</option>
          {list.map((item) => (
            <option key={item.id} value={String(item.step)}>
              {`step ${item.step}${item.epoch ? ` · e${item.epoch}` : ""}`}
            </option>
          ))}
        </select>

        {stream && stream.kind !== "histogram" ? (
          <select
            className="probe__select probe__select--map"
            value={colormap}
            onChange={(event) => setColormap(event.target.value as ColormapName)}
            title={t("run.probe.colormapHint")}
          >
            {COLORMAP_NAMES.map((name) => (
              <option key={name} value={name}>
                {t(`run.probe.colormap.${name}`)}
              </option>
            ))}
          </select>
        ) : null}

        <span className="probe__spacer" />

        {meta ? (
          <span className="probe__meta">
            {`step ${meta.step}`}
            {meta.epoch ? ` · e${meta.epoch}` : ""}
            {meta.shape.length > 0 ? ` · ${meta.shape.join("×")}` : ""}
            {stale ? ` · ${t("common.loading")}` : ""}
          </span>
        ) : null}
      </div>

      <div className="probe__body">
        {!runId ? (
          <div className="probe__empty">{t("run.probe.empty.noRun")}</div>
        ) : streams.length === 0 ? (
          <div className="probe__empty">
            {probeCount > 0
              ? t("run.probe.empty.waiting", { n: probeEveryN })
              : t("run.probe.empty.noProbes")}
          </div>
        ) : !payload ? (
          <div className="probe__empty">{error ?? t("common.loading")}</div>
        ) : stream?.kind === "histogram" ? (
          <HistogramCanvas payload={payload} />
        ) : stream?.kind === "attention" || stream?.kind === "hidden" ? (
          <HeatmapCanvas payload={payload} colormap={colormap} kind={stream.kind} />
        ) : (
          <FeatureGridCanvas payload={payload} colormap={colormap} />
        )}
      </div>
    </div>
  );
}
