import { useCallback } from "react";
import { useReactFlow } from "@xyflow/react";
import { useTranslation } from "react-i18next";

import type { NodeType } from "../../graph/ir";
import { PALETTE_GROUPS } from "../../graph/specs";
import { useGraphStore } from "../../stores/graphStore";

export function NodePalette() {
  const { t } = useTranslation();
  const readOnly = useGraphStore((state) => state.readOnly);
  const addNode = useGraphStore((state) => state.addNode);
  const { screenToFlowPosition } = useReactFlow();

  const handleAdd = useCallback(
    (type: NodeType) => {
      const center = screenToFlowPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
      });
      addNode(type, { x: center.x - 100 + Math.random() * 40, y: center.y + Math.random() * 40 });
    },
    [addNode, screenToFlowPosition],
  );

  return (
    <div className="palette">
      {PALETTE_GROUPS.map((group) => (
        <div key={group.category} className="palette__group">
          <div className="palette__title">{t(`nodes.category.${group.category}`)}</div>
          <div className="palette__items">
            {group.types.map((type) => (
              <button
                key={type}
                type="button"
                className={`palette__item palette__item--${group.category}`}
                disabled={readOnly}
                onClick={() => handleAdd(type)}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      ))}
      {readOnly ? <div className="palette__hint">{t("editor.readOnlyPaletteHint")}</div> : null}
    </div>
  );
}
