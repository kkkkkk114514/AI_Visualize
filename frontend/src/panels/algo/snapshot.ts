import { useEffect, useMemo, useState } from "react";

import { fetchSnapshot } from "../../api/snapshots";
import type { ProbeKind, SnapshotMeta, SnapshotPayload } from "../../api/types";
import { probeStreamKey, useRunStore } from "../../stores/runStore";

const NONE: SnapshotMeta[] = [];

interface LoadedPayload {
  runId: string;
  id: string;
  payload: SnapshotPayload;
}

/** 按快照 id 取 payload（实时 / 回放同一条路径）：换步时保留上一帧，避免画面闪空。 */
function usePayload(runId: string | null, snapshotId: string | null) {
  const [loaded, setLoaded] = useState<LoadedPayload | null>(null);

  useEffect(() => {
    if (!runId || !snapshotId) return;
    let cancelled = false;
    fetchSnapshot(runId, snapshotId)
      .then((payload) => {
        if (!cancelled) setLoaded({ runId, id: snapshotId, payload });
      })
      .catch(() => {
        // 单帧拉取失败不打断训练视图：下一帧到达时自然恢复
      });
    return () => {
      cancelled = true;
    };
  }, [runId, snapshotId]);

  const payload = loaded && loaded.runId === runId ? loaded.payload : null;
  return { payload, stale: payload !== null && loaded?.id !== snapshotId };
}

export interface AlgoSnapshot {
  runId: string | null;
  meta: SnapshotMeta | null;
  payload: SnapshotPayload | null;
  /** 正在切到新一帧（画的是上一帧） */
  stale: boolean;
}

/**
 * 单流探针快照（ML `model:boundary` / RL `agent:grid`）：取「≤ 游标的最近一条」，
 * 跟随最新时取最新（与探针面板同口径，docs/02 §13.6）。
 */
export function useAlgoSnapshot(nodeId: string, kind: ProbeKind): AlgoSnapshot {
  const runId = useRunStore((state) => state.snapshotRunId);
  const cursorStep = useRunStore((state) => state.cursorStep);
  const list = useRunStore(
    (state) => state.snapshots[probeStreamKey({ nodeId, kind })] ?? NONE,
  );
  const meta = useMemo(() => {
    if (list.length === 0) return null;
    if (cursorStep === null) return list[list.length - 1];
    let found: SnapshotMeta = list[0];
    for (const item of list) {
      if (item.step <= cursorStep) found = item;
      else break;
    }
    return found;
  }, [list, cursorStep]);
  const { payload, stale } = usePayload(runId, meta?.id ?? null);
  return { runId, meta, payload, stale };
}
