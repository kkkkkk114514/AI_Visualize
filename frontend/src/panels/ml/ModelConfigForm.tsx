import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { useAlgoStore } from "../../stores/algoStore";
import { useRunStore } from "../../stores/runStore";
import { localizedText } from "../../i18n/localized";
import { ParamFields } from "../algo/ParamFields";

/** ML 训练配置（docs/02 §13.6）：数据集下拉（synth2d / csv2d，含上传的点集）+ schema 驱动的算法参数表单。 */
export function ModelConfigForm() {
  const { t, i18n } = useTranslation();
  const spec = useAlgoStore((state) => state.spec);
  const schemas = useAlgoStore((state) => state.schemas);
  const schemasError = useAlgoStore((state) => state.schemasError);
  const loadSchemas = useAlgoStore((state) => state.loadSchemas);
  const setParam = useAlgoStore((state) => state.setParam);
  const setDataset = useAlgoStore((state) => state.setDataset);
  const fromRunId = useAlgoStore((state) => state.fromRunId);
  const datasets = useRunStore((state) => state.datasets);
  const loadDatasets = useRunStore((state) => state.loadDatasets);

  useEffect(() => {
    void loadSchemas();
  }, [loadSchemas]);

  useEffect(() => {
    if (datasets === null) void loadDatasets();
  }, [datasets, loadDatasets]);

  if (!spec) return null;
  const locked = fromRunId !== null;
  const schema = schemas?.[spec.algo] ?? null;
  // 与后端 ML 的 loader 白名单一致（docs/02 §13.3）：合成点集 + 上传的 csv 点集
  const choices = (datasets ?? []).filter((item) =>
    item.loader === "synth2d" || item.loader === "csv2d",
  );
  const dataset = choices.find((item) => item.id === spec.dataset_id) ?? null;
  const everyN = spec.probe_defaults?.every_n_steps ?? null;

  return (
    <div className="config">
      <div className="config__section">
        <div className="config__title">{t("run.dataset.title")}</div>
        <label className="config-field">
          <span className="config-field__label">{t("run.dataset.pick")}</span>
          <select
            className="config-field__input"
            value={spec.dataset_id ?? ""}
            disabled={locked || choices.length === 0}
            onChange={(event) => setDataset(event.target.value)}
          >
            {choices.map((item) => (
              <option key={item.id} value={item.id}>
                {localizedText(item.name, i18n.language) || item.id}
              </option>
            ))}
          </select>
        </label>
        {dataset ? (
          <div className="config__meta">
            {dataset.input_shape.join("×")} · {dataset.num_classes} {t("algo.classes")}
          </div>
        ) : null}
        <div className="config__hint">{t("algo.ml.datasetHint")}</div>
      </div>

      <div className="config__section">
        <div className="config__title">{t("algo.params.title")}</div>
        {schema ? (
          <ParamFields
            fields={schema.fields}
            params={spec.params}
            disabled={locked}
            onChange={setParam}
          />
        ) : (
          <div className="config__meta">{schemasError ?? t("common.loading")}</div>
        )}
        {everyN ? <div className="config__hint">{t("algo.probeNote", { n: everyN })}</div> : null}
        {locked ? <div className="config__hint">{t("run.replayLockedHint")}</div> : null}
      </div>
    </div>
  );
}
