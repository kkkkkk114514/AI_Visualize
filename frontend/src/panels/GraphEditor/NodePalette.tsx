import { useTranslation } from "react-i18next";

import { PALETTE_GROUPS } from "../../graph/specs";
import { useGraphStore } from "../../stores/graphStore";

export const NODE_DRAG_MIME = "application/x-ai-visualize-node";

export function NodePalette() {
  const { t } = useTranslation();
  const readOnly = useGraphStore((state) => state.readOnly);

  return (
    <div className={`palette${readOnly ? " palette--readonly" : ""}`}>
      {PALETTE_GROUPS.map((group) => (
        <div key={group.category} className="palette__group">
          <div className="palette__title">{t(`nodes.category.${group.category}`)}</div>
          <div className="palette__items">
            {group.types.map((type) => (
              <div
                key={type}
                className={`palette__item palette__item--${group.category}`}
                draggable={!readOnly}
                title={t(`nodes.desc.${type}`)}
                onDragStart={(event) => {
                  event.dataTransfer.setData(NODE_DRAG_MIME, type);
                  event.dataTransfer.effectAllowed = "copy";
                }}
              >
                {type}
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="palette__hint">
        {readOnly ? t("editor.readOnlyPaletteHint") : t("editor.paletteHint")}
      </div>
    </div>
  );
}
