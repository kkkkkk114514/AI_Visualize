import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { AlgoField } from "../../api/types";

interface NumberInputProps {
  value: number;
  integer: boolean;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  onCommit: (value: number) => void;
}

/** 数字输入：编辑期用本地草稿，失焦 / 回车才提交（与训练配置同惯例）。 */
function NumberInput({ value, integer, min, max, step, disabled, onCommit }: NumberInputProps) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    if (Number.isNaN(parsed)) {
      setDraft(String(value));
      return;
    }
    const next = integer ? Math.round(parsed) : parsed;
    if (next !== value) onCommit(next);
  };

  return (
    <input
      className="algo-param__value"
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={(event) => commit(event.currentTarget.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        else if (event.key === "Escape") setDraft(String(value));
      }}
    />
  );
}

interface ParamFieldsProps {
  fields: AlgoField[];
  params: Record<string, number | string>;
  disabled?: boolean;
  onChange: (name: string, value: number | string) => void;
}

/**
 * schema 驱动的算法参数表单（docs/02 §13.2）：`int` / `float` → 数字输入 + 滑杆，
 * `choice` → 下拉；与服务端校验共用同一份 `GET /api/algos` 表。
 */
export function ParamFields({ fields, params, disabled, onChange }: ParamFieldsProps) {
  const { t } = useTranslation();

  return (
    <div className="param-form">
      {fields.map((field) => {
        const label = t(field.label_key, { defaultValue: field.name });
        const value = params[field.name] ?? field.default;
        if (field.type === "choice") {
          return (
            <label key={field.name} className="config-field">
              <span className="config-field__label">{label}</span>
              <select
                className="config-field__input"
                value={String(value)}
                disabled={disabled}
                onChange={(event) => onChange(field.name, event.target.value)}
              >
                {(field.choices ?? []).map((choice) => (
                  <option key={choice} value={choice}>
                    {t(`algo.choice.${choice}`, { defaultValue: choice })}
                  </option>
                ))}
              </select>
            </label>
          );
        }
        const integer = field.type === "int";
        const min = field.min ?? undefined;
        const max = field.max ?? undefined;
        const step = field.step ?? (integer ? 1 : 0.01);
        const numeric = typeof value === "number" ? value : Number(field.default);
        return (
          <div key={field.name} className="algo-param" title={label}>
            <div className="algo-param__head">
              <span className="algo-param__label">{label}</span>
              <NumberInput
                value={numeric}
                integer={integer}
                min={min}
                max={max}
                step={step}
                disabled={disabled}
                onCommit={(next) => onChange(field.name, next)}
              />
            </div>
            <input
              className="algo-param__slider"
              type="range"
              min={min}
              max={max}
              step={step}
              value={numeric}
              disabled={disabled}
              onChange={(event) =>
                onChange(field.name, integer ? Math.round(Number(event.target.value)) : Number(event.target.value))
              }
            />
          </div>
        );
      })}
    </div>
  );
}
