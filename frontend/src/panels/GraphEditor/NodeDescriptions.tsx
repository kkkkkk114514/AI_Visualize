import { useTranslation } from "react-i18next";

import { PALETTE_GROUPS } from "../../graph/specs";
import { useGraphStore } from "../../stores/graphStore";

export function NodeDescriptions() {
  const { t } = useTranslation();
  const selectedType = useGraphStore(
    (state) => state.nodes.find((node) => node.selected)?.data.nodeType ?? null,
  );

  return (
    <div className="node-help">
      {PALETTE_GROUPS.map((group) => (
        <div key={group.category} className="node-help__group">
          <div className="node-help__title">{t(`nodes.category.${group.category}`)}</div>
          <ul className="node-help__list">
            {group.types.map((type) => (
              <li
                key={type}
                className={`node-help__item node-help__item--${group.category}${type === selectedType ? " is-active" : ""}`}
              >
                <span className="node-help__type">{type}</span>
                <span className="node-help__desc">{t(`nodes.desc.${type}`)}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
