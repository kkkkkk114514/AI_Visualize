import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import { useCallback, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import type { FlowGraphNode } from "../../graph/flow";
import { useGraphStore } from "../../stores/graphStore";
import { GraphNodeCard } from "./GraphNodeCard";

const nodeTypes = { graphNode: GraphNodeCard };

export function GraphEditor() {
  const { t } = useTranslation();
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const readOnly = useGraphStore((state) => state.readOnly);
  const applyPositions = useGraphStore((state) => state.applyPositions);
  const connect = useGraphStore((state) => state.connect);
  const deleteElements = useGraphStore((state) => state.deleteElements);
  const copySelection = useGraphStore((state) => state.copySelection);
  const pasteClipboard = useGraphStore((state) => state.pasteClipboard);
  const undo = useGraphStore((state) => state.undo);
  const redo = useGraphStore((state) => state.redo);
  const autoLayout = useGraphStore((state) => state.autoLayout);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { fitView } = useReactFlow();

  const onNodesChange = useCallback(
    (changes: NodeChange<FlowGraphNode>[]) => {
      const removals = changes.filter((change) => change.type === "remove").map((change) => change.id);
      if (removals.length > 0 && !readOnly) {
        deleteElements(removals, []);
        return;
      }
      applyPositions(applyNodeChanges(changes, nodes));
    },
    [applyPositions, deleteElements, nodes, readOnly],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const removals = changes.filter((change) => change.type === "remove").map((change) => change.id);
      if (removals.length > 0 && !readOnly) {
        deleteElements([], removals);
      }
    },
    [deleteElements, readOnly],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;
      connect(connection.source, connection.target);
    },
    [connect],
  );

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const modifier = event.ctrlKey || event.metaKey;
      const selectedNodes = useGraphStore.getState().nodes.filter((node) => node.selected).map((node) => node.id);

      if (modifier && event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
      } else if (modifier && (event.key.toLowerCase() === "y" || (event.key.toLowerCase() === "z" && event.shiftKey))) {
        event.preventDefault();
        redo();
      } else if (modifier && event.key.toLowerCase() === "c") {
        if (selectedNodes.length) {
          event.preventDefault();
          copySelection(selectedNodes);
        }
      } else if (modifier && event.key.toLowerCase() === "v") {
        event.preventDefault();
        pasteClipboard();
      } else if (modifier && event.key.toLowerCase() === "a") {
        event.preventDefault();
        applyPositions(nodes.map((node) => ({ ...node, selected: true })));
      } else if (event.key === "Escape") {
        applyPositions(nodes.map((node) => ({ ...node, selected: false })));
      }
    };
    element.addEventListener("keydown", onKeyDown);
    return () => element.removeEventListener("keydown", onKeyDown);
  }, [applyPositions, copySelection, nodes, pasteClipboard, redo, undo]);

  const handleLayout = useCallback(() => {
    autoLayout();
    window.setTimeout(() => void fitView({ padding: 0.2, duration: 400 }), 30);
  }, [autoLayout, fitView]);

  return (
    <div className="graph-editor" ref={wrapperRef} tabIndex={0}>
      <div className="graph-editor__toolbar">
        {!readOnly ? (
          <>
            <button type="button" className="btn btn--ghost" onClick={handleLayout}>
              {t("editor.autoLayout")}
            </button>
            <button type="button" className="btn btn--ghost" onClick={undo}>
              {t("editor.undo")}
            </button>
            <button type="button" className="btn btn--ghost" onClick={redo}>
              {t("editor.redo")}
            </button>
          </>
        ) : null}
        <span className="graph-editor__hint">{t("editor.shortcuts")}</span>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        edgesReconnectable={!readOnly}
        elementsSelectable
        deleteKeyCode={readOnly ? null : ["Delete", "Backspace"]}
        selectionOnDrag
        panOnDrag={[1, 2]}
        panOnScroll
        zoomOnScroll={false}
        zoomOnDoubleClick={false}
        fitView
        minZoom={0.2}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
