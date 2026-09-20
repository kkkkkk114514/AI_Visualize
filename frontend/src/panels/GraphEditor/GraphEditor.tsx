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
import { useCallback, useEffect, useRef, useState, type CSSProperties, type DragEvent, type MouseEvent } from "react";
import { useTranslation } from "react-i18next";

import type { NodeType } from "../../graph/ir";
import { specOf } from "../../graph/specs";
import type { FlowGraphNode } from "../../graph/flow";
import { useGraphStore } from "../../stores/graphStore";
import { GraphNodeCard } from "./GraphNodeCard";
import { NodeContextMenu } from "./NodeContextMenu";
import { NODE_DRAG_MIME } from "./NodePalette";

const nodeTypes = { graphNode: GraphNodeCard };

/** 落点相对节点中心的偏移，让光标落在卡片上部而不是左上角。 */
const DROP_OFFSET = { x: 104, y: 34 };
const MENU_WIDTH = 248;

export function GraphEditor() {
  const { t } = useTranslation();
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const readOnly = useGraphStore((state) => state.readOnly);
  const applyPositions = useGraphStore((state) => state.applyPositions);
  const connect = useGraphStore((state) => state.connect);
  const addNode = useGraphStore((state) => state.addNode);
  const deleteElements = useGraphStore((state) => state.deleteElements);
  const copySelection = useGraphStore((state) => state.copySelection);
  const pasteClipboard = useGraphStore((state) => state.pasteClipboard);
  const undo = useGraphStore((state) => state.undo);
  const redo = useGraphStore((state) => state.redo);
  const autoLayout = useGraphStore((state) => state.autoLayout);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [menu, setMenu] = useState<{ style: CSSProperties; point: { x: number; y: number } } | null>(null);
  const { fitView, screenToFlowPosition } = useReactFlow();

  const closeMenu = useCallback(() => setMenu(null), []);

  const createAt = useCallback(
    (type: NodeType, clientX: number, clientY: number) => {
      if (readOnly || !specOf(type)) return;
      const point = screenToFlowPosition({ x: clientX, y: clientY });
      addNode(type, { x: point.x - DROP_OFFSET.x, y: point.y - DROP_OFFSET.y });
    },
    [addNode, readOnly, screenToFlowPosition],
  );

  const onDragOver = useCallback(
    (event: DragEvent) => {
      if (readOnly) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    },
    [readOnly],
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      if (readOnly) return;
      const type = event.dataTransfer.getData(NODE_DRAG_MIME) as NodeType;
      if (!specOf(type)) return;
      event.preventDefault();
      createAt(type, event.clientX, event.clientY);
    },
    [createAt, readOnly],
  );

  const onPaneContextMenu = useCallback(
    (event: MouseEvent | globalThis.MouseEvent) => {
      event.preventDefault();
      if (readOnly) return;
      setMenu((current) => {
        if (current) return null;
        const rect = wrapperRef.current?.getBoundingClientRect();
        if (!rect) return null;
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        const style: CSSProperties = {
          left: Math.max(8, Math.min(localX, rect.width - MENU_WIDTH - 8)),
        };
        if (localY > rect.height / 2) style.bottom = Math.max(8, rect.height - localY);
        else style.top = localY;
        return { style, point: { x: event.clientX, y: event.clientY } };
      });
    },
    [readOnly],
  );

  const pickFromMenu = useCallback(
    (type: NodeType) => {
      if (!menu) return;
      createAt(type, menu.point.x, menu.point.y);
      closeMenu();
    },
    [closeMenu, createAt, menu],
  );

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
        closeMenu();
        applyPositions(nodes.map((node) => ({ ...node, selected: false })));
      }
    };
    // selectionOnDrag 打开时 xyflow 不派发 onPaneClick，左键关闭菜单只能自己监听。
    const onClick = (event: globalThis.MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest(".graph-menu")) return;
      setMenu(null);
    };
    element.addEventListener("keydown", onKeyDown);
    element.addEventListener("click", onClick);
    return () => {
      element.removeEventListener("keydown", onKeyDown);
      element.removeEventListener("click", onClick);
    };
  }, [applyPositions, closeMenu, copySelection, nodes, pasteClipboard, redo, undo]);

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
        onDragOver={onDragOver}
        onDrop={onDrop}
        onPaneContextMenu={onPaneContextMenu}
        onMoveStart={closeMenu}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        edgesReconnectable={!readOnly}
        elementsSelectable
        deleteKeyCode={readOnly ? null : ["Delete", "Backspace"]}
        selectionOnDrag
        panOnDrag={[1]}
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
      {menu ? <NodeContextMenu style={menu.style} onPick={pickFromMenu} /> : null}
    </div>
  );
}
