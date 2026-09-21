import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { Allotment, type AllotmentHandle } from "allotment";
import { useTranslation } from "react-i18next";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { fetchModelDetail } from "../api/graph";
import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { HiddenPanelsStrip, PanelLayoutProvider, usePanelLayoutState } from "../components/panelLayout";
import type { PanelId, PanelLayoutApi } from "../components/panelLayout";
import { Panel } from "../components/Panel";
import { GraphEditor } from "../panels/GraphEditor/GraphEditor";
import { IssuesPanel } from "../panels/GraphEditor/IssuesPanel";
import { NodeDescriptions } from "../panels/GraphEditor/NodeDescriptions";
import { NodePalette } from "../panels/GraphEditor/NodePalette";
import { useGraphInfer } from "../panels/GraphEditor/useGraphInfer";
import { MetricsCharts } from "../panels/MetricsCharts/MetricsCharts";
import { ProbeViewer } from "../panels/ProbeViewer/ProbeViewer";
import { RunControls } from "../panels/RunControls/RunControls";
import { RunList } from "../panels/RunList/RunList";
import { TrainingConfig } from "../panels/TrainingConfig/TrainingConfig";
import { emptyGraph } from "../graph/ir";
import type { GraphIR } from "../graph/ir";
import { localizedText } from "../i18n/localized";
import { useGraphStore } from "../stores/graphStore";
import { useRunStore } from "../stores/runStore";

type ColumnKey = "left" | "center" | "right";

/** 三列的面板归属（与下方 JSX 的列结构、COLUMN_OF 保持一致）；整列收起与尺寸还原按此分组（docs/02 §9.2） */
const COLUMN_PANES: Record<ColumnKey, PanelId[]> = {
  left: ["nodes", "config"],
  center: ["graph", "metrics", "probe"],
  right: ["check", "nodeHelp", "runs"],
};

const COLUMN_OF: Record<PanelId, ColumnKey> = {
  nodes: "left",
  config: "left",
  graph: "center",
  metrics: "center",
  probe: "center",
  check: "right",
  nodeHelp: "right",
  runs: "right",
};

export default function LabPage() {
  const { t, i18n } = useTranslation();
  const { graphId = "new" } = useParams();
  const load = useGraphStore((state) => state.load);
  const cloneAsCopy = useGraphStore((state) => state.cloneAsCopy);
  const readOnly = useGraphStore((state) => state.readOnly);
  const source = useGraphStore((state) => state.source);
  const dirty = useGraphStore((state) => state.dirty);
  const meta = useGraphStore((state) => state.meta);
  const refresh = useRunStore((state) => state.refresh);
  const applyConfigDefaults = useRunStore((state) => state.applyConfigDefaults);
  const applyProbeDefaults = useRunStore((state) => state.applyProbeDefaults);
  const setDataset = useRunStore((state) => state.setDataset);
  const replay = useRunStore((state) => state.replay);
  const replayLoading = useRunStore((state) => state.replayLoading);
  const selectRun = useRunStore((state) => state.selectRun);
  const [searchParams, setSearchParams] = useSearchParams();
  const urlRunId = searchParams.get("run");
  const panelLayout = usePanelLayoutState();
  const isHidden = panelLayout.isHidden;
  // 一列的面板全部收起时，整列一起收起，把宽度让给相邻列（docs/02 §9.2）
  const columnHidden: Record<ColumnKey, boolean> = {
    left: COLUMN_PANES.left.every(isHidden),
    center: COLUMN_PANES.center.every(isHidden),
    right: COLUMN_PANES.right.every(isHidden),
  };
  // 整列收起再恢复时 Allotment 的 cachedVisibleSize 已在级联中失真（逐个 setVisible(false)
  // 会把腾出的高度先分给仍可见的兄弟面板，恢复时按失真值撑开、溢出后被压回最小尺寸）。
  // 因此记住「整列都在」时的列内尺寸，恢复时用 resize() 还原（docs/02 §9.2）。
  const columnRefs = useRef<Record<ColumnKey, AllotmentHandle | null>>({ left: null, center: null, right: null });
  const pristineSizes = useRef<Partial<Record<ColumnKey, number[]>>>({});
  const pendingResize = useRef(new Set<ColumnKey>());

  const recordPristine = useCallback((col: ColumnKey, sizes: number[]) => {
    const prev = pristineSizes.current[col];
    pristineSizes.current[col] = sizes.map((size, i) => (size > 0 ? size : prev?.[i] ?? size));
  }, []);

  const layoutApi = useMemo<PanelLayoutApi>(
    () => ({
      ...panelLayout,
      restore: (id) => {
        const col = COLUMN_OF[id];
        if (COLUMN_PANES[col].some(isHidden)) pendingResize.current.add(col);
        panelLayout.restore(id);
      },
    }),
    [panelLayout, isHidden],
  );

  useLayoutEffect(() => {
    if (pendingResize.current.size === 0) return;
    for (const col of [...pendingResize.current]) {
      const pristine = pristineSizes.current[col];
      if (!pristine) {
        pendingResize.current.delete(col);
        continue;
      }
      columnRefs.current[col]?.resize(COLUMN_PANES[col].map((id, i) => (isHidden(id) ? 0 : pristine[i])));
      if (COLUMN_PANES[col].every((id) => !isHidden(id))) pendingResize.current.delete(col);
    }
  });

  // 快照与还原都以像素尺寸判断可见性（可见面板有 minSize，尺寸为 0 只可能是收起）：
  // Allotment 在 passive effect 里才换上新的 onChange 闭包，收起提交内的级联布局仍在用
  // 上一轮闭包，读 render 里的 isHidden 会把级联出来的 0 当成可见尺寸记下来。恢复中的
  // 列（pending）同样不记：恢复时级联出来的中间尺寸会覆盖掉真正要还原的快照（docs/02 §9.2）。
  const columnBind = (col: ColumnKey) => ({
    ref: (handle: AllotmentHandle | null) => {
      columnRefs.current[col] = handle;
    },
    onChange: (sizes: number[]) => {
      if (pendingResize.current.has(col) || !sizes.every((size) => size > 0)) return;
      pristineSizes.current[col] = sizes;
    },
    onDragEnd: (sizes: number[]) => recordPristine(col, sizes),
  });
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // `?run=` 是回放状态的唯一真源（docs/02 §9.2「回放模式」）：带参打开即拉详情，
  // 后退 / 手动清参退回非回放；同一参数只尝试一次，失败由下面的写 URL effect 移除无效参数。
  // 判定值一律现读 store：同一提交里前面的 effect 可能刚改过 store，用渲染期闭包会拿旧值做决定。
  const attemptedRun = useRef<string | null>(null);
  useEffect(() => {
    const state = useRunStore.getState();
    if (urlRunId === null) {
      attemptedRun.current = null;
      if (state.replay) void selectRun(null);
      return;
    }
    if (state.replay?.id === urlRunId || state.replayLoading) return;
    if (attemptedRun.current === urlRunId) return;
    attemptedRun.current = urlRunId;
    void selectRun(urlRunId);
  }, [urlRunId, replay, replayLoading, selectRun]);

  // 反向同步：列表点选 / 自动选中最新 / 删除回放中的 run，都把 `?run=` 写回 URL
  // （replace，不污染后退栈）；加载期间不写，否则会把刚打开的 URL 参数删掉。
  useEffect(() => {
    const state = useRunStore.getState();
    if (state.replayLoading) return;
    const activeRunId = state.replay?.id ?? null;
    if (activeRunId === urlRunId) return;
    const next = new URLSearchParams(searchParams);
    if (activeRunId) next.set("run", activeRunId);
    else next.delete("run");
    setSearchParams(next, { replace: true });
  }, [urlRunId, replay, replayLoading, searchParams, setSearchParams]);

  const replayGraph = (replay?.graph ?? null) as GraphIR | null;

  useEffect(() => {
    let cancelled = false;
    const state = useRunStore.getState();
    if (state.replayLoading) {
      // 回放加载中保持现状：首次进入停在加载态，切换回放时旧图继续显示，
      // 避免与即将恢复的回放图互相覆盖
      return () => {
        cancelled = true;
      };
    }
    const graph = (state.replay?.graph ?? null) as GraphIR | null;
    if (graph) {
      // 回放：图结构从 run 详情的 graph_json 恢复，只读（禁编辑 / 禁右键建节点 / 不显示克隆）
      load(graph, { readOnly: true, source: "replay" });
      setLoadError(null);
      setStatus("ready");
      return () => {
        cancelled = true;
      };
    }
    setStatus("loading");
    setLoadError(null);
    if (graphId === "new") {
      const graph = emptyGraph();
      load(graph, { readOnly: false, source: "new" });
      applyConfigDefaults(graph.hyper_defaults);
      applyProbeDefaults(graph);
      setStatus("ready");
      return () => {
        cancelled = true;
      };
    }
    fetchModelDetail(graphId)
      .then((detail) => {
        if (cancelled) return;
        load(detail.graph, { readOnly: detail.source === "preset", source: detail.source === "preset" ? "preset" : "user" });
        applyConfigDefaults(detail.graph.hyper_defaults);
        applyProbeDefaults(detail.graph);
        const hinted = (detail.graph as { dataset_id?: unknown }).dataset_id;
        if (typeof hinted === "string" && hinted) setDataset(hinted);
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
  }, [graphId, replayGraph, replayLoading, load, applyConfigDefaults, applyProbeDefaults, setDataset]);

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
            {readOnly ? (
              <span className="badge badge--muted">
                {source === "replay"
                  ? t("lab.replayMode", { run: replay?.id ?? "" })
                  : t("lab.readOnly")}
              </span>
            ) : null}
            {source === "replay" ? (
              <button
                type="button"
                className="btn btn--ghost lab-replay-exit"
                onClick={() => void selectRun(null)}
                title={t("lab.exitReplayHint")}
              >
                {t("lab.exitReplay")}
              </button>
            ) : null}
            {dirty && !readOnly ? <span className="badge badge--warn">{t("lab.unsaved")}</span> : null}
          </>
        }
        actions={
          <>
            {readOnly && source !== "replay" ? (
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
      <RunControls />
      <PanelLayoutProvider value={layoutApi}>
        <HiddenPanelsStrip />
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
                <Allotment.Pane minSize={240} preferredSize="20%" visible={!columnHidden.left}>
                  <Allotment vertical {...columnBind("left")}>
                    <Allotment.Pane minSize={160} visible={!isHidden("nodes")}>
                      <Panel id="nodes" title={t("lab.zones.nodes")}>
                        <NodePalette />
                      </Panel>
                    </Allotment.Pane>
                    <Allotment.Pane minSize={240} visible={!isHidden("config")}>
                      <Panel id="config" title={t("lab.zones.config")}>
                        <TrainingConfig />
                      </Panel>
                    </Allotment.Pane>
                  </Allotment>
                </Allotment.Pane>
                <Allotment.Pane minSize={460} visible={!columnHidden.center}>
                  <Allotment vertical {...columnBind("center")}>
                    <Allotment.Pane minSize={200} preferredSize="46%" visible={!isHidden("graph")}>
                      <Panel id="graph" title={t("lab.zones.graph")}>
                        <GraphEditor />
                      </Panel>
                    </Allotment.Pane>
                    <Allotment.Pane minSize={150} preferredSize="26%" visible={!isHidden("metrics")}>
                      <Panel id="metrics" title={t("lab.zones.metrics")}>
                        <MetricsCharts />
                      </Panel>
                    </Allotment.Pane>
                    <Allotment.Pane minSize={150} visible={!isHidden("probe")}>
                      <Panel id="probe" title={t("lab.zones.probe")}>
                        <ProbeViewer />
                      </Panel>
                    </Allotment.Pane>
                  </Allotment>
                </Allotment.Pane>
                <Allotment.Pane minSize={300} preferredSize="26%" visible={!columnHidden.right}>
                  <Allotment vertical {...columnBind("right")}>
                    <Allotment.Pane minSize={120} preferredSize="30%" visible={!isHidden("check")}>
                      <Panel id="check" title={t("lab.zones.check")}>
                        <IssuesPanel />
                      </Panel>
                    </Allotment.Pane>
                    <Allotment.Pane minSize={140} preferredSize="34%" visible={!isHidden("nodeHelp")}>
                      <Panel id="nodeHelp" title={t("lab.zones.nodeHelp")}>
                        <NodeDescriptions />
                      </Panel>
                    </Allotment.Pane>
                    <Allotment.Pane minSize={140} visible={!isHidden("runs")}>
                      <Panel id="runs" title={t("lab.zones.runs")}>
                        <RunList />
                      </Panel>
                    </Allotment.Pane>
                  </Allotment>
                </Allotment.Pane>
              </Allotment>
            </ReactFlowProvider>
          )}
        </div>
      </PanelLayoutProvider>
    </div>
  );
}
