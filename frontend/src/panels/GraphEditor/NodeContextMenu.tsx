import type { CSSProperties } from "react";
import { useTranslation } from "react-i18next";

import type { NodeType } from "../../graph/ir";
import { PALETTE_GROUPS } from "../../graph/specs";

interface NodeContextMenuProps {
  style: CSSProperties;
  onPick: (type: NodeType) => void;
}

export function NodeContextMenu({ style, onPick }: NodeContextMenuProps) {
  const { t } = useTranslation();
  return (
    <div className="graph-menu" style={style} role="menu">
      <div className="graph-menu__title">{t("editor.pickNode")}</div>
      <div className="graph-menu__body">
        {PALETTE_GROUPS.map((group) => (
          <div key={group.category} className="graph-menu__group">
            <div className="graph-menu__group-title">{t(`nodes.category.${group.category}`)}</div>
            <div className="graph-menu__items">
              {group.types.map((type) => (
                <button
                  key={type}
                  type="button"
                  className={`graph-menu__item graph-menu__item--${group.category}`}
                  title={t(`nodes.desc.${type}`)}
                  onClick={() => onPick(type)}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
