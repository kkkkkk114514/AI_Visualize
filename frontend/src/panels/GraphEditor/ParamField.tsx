import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { ParamSpecUI } from "../../graph/specs";

interface Props {
  spec: ParamSpecUI;
  value: unknown;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}

function shapeText(value: unknown): string {
  if (Array.isArray(value)) return value.map((v) => String(v)).join("×");
  return String(value ?? "—");
}

export function ParamField({ spec, value, readOnly, onChange }: Props) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<string>(String(value ?? ""));

  useEffect(() => {
    setDraft(String(value ?? ""));
  }, [value]);

  if (spec.kind === "shape") {
    return (
      <div className="param-field param-field--readonly" title={t("editor.datasetBound")}>
        <span className="param-field__label">{t(spec.labelKey)}</span>
        <span className="param-field__value">{shapeText(value)}</span>
      </div>
    );
  }

  if (spec.kind === "bool") {
    return (
      <label className="param-field param-field--bool nodrag">
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={readOnly}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="param-field__label">{t(spec.labelKey)}</span>
      </label>
    );
  }

  if (spec.kind === "enum") {
    return (
      <label className="param-field nodrag">
        <span className="param-field__label">{t(spec.labelKey)}</span>
        <select
          className="param-field__input"
          value={String(value ?? "")}
          disabled={readOnly}
          onChange={(event) => onChange(event.target.value)}
        >
          {(spec.choices ?? []).map((choice) => (
            <option key={choice.value} value={choice.value}>
              {t(choice.labelKey)}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const commit = () => {
    const parsed = spec.kind === "int" ? Number.parseInt(draft, 10) : Number.parseFloat(draft);
    if (Number.isNaN(parsed)) {
      setDraft(String(value ?? ""));
      return;
    }
    if (parsed !== value) onChange(parsed);
  };

  return (
    <label className="param-field nodrag" title={spec.zeroMeansKernel ? t("editor.zeroMeansKernel") : undefined}>
      <span className="param-field__label">{t(spec.labelKey)}</span>
      <input
        className="param-field__input"
        type="number"
        value={draft}
        min={spec.min}
        max={spec.max}
        step={spec.step ?? (spec.kind === "int" ? 1 : 0.01)}
        disabled={readOnly}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          } else if (event.key === "Escape") {
            setDraft(String(value ?? ""));
          }
        }}
      />
    </label>
  );
}
