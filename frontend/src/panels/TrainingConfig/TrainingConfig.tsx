import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { probeCandidates } from "../../graph/probes";
import { useGraphStore } from "../../stores/graphStore";
import type { RunConfig } from "../../stores/runStore";
import { isActiveStatus, useRunStore } from "../../stores/runStore";

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

interface NumberFieldProps {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  hint?: string;
  onCommit: (value: number) => void;
}

function NumberField({ label, value, step, min, max, disabled, hint, onCommit }: NumberFieldProps) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  // 提交读事件里的实时值：同一 tick 内连续 input+blur 时，闭包里的 draft 还是旧值
  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    if (Number.isNaN(parsed)) {
      setDraft(String(value));
      return;
    }
    if (parsed !== value) onCommit(parsed);
  };

  return (
    <label className="config-field" title={hint}>
      <span className="config-field__label">{label}</span>
      <input
        className="config-field__input"
        type="number"
        value={draft}
        min={min}
        max={max}
        step={step ?? 0.001}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={(event) => commit(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
          else if (event.key === "Escape") setDraft(String(value));
        }}
      />
    </label>
  );
}

export function TrainingConfig() {
  const { t, i18n } = useTranslation();
  const datasets = useRunStore((state) => state.datasets);
  const datasetId = useRunStore((state) => state.datasetId);
  const datasetProgress = useRunStore((state) => state.datasetProgress);
  const config = useRunStore((state) => state.config);
  const status = useRunStore((state) => state.status);
  const runId = useRunStore((state) => state.current?.id ?? null);
  const setDataset = useRunStore((state) => state.setDataset);
  const downloadDataset = useRunStore((state) => state.downloadDataset);
  const setConfig = useRunStore((state) => state.setConfig);
  const applyLr = useRunStore((state) => state.applyLr);
  const applyBatchSize = useRunStore((state) => state.applyBatchSize);
  const probeChoices = useRunStore((state) => state.probeChoices);
  const probeEveryN = useRunStore((state) => state.probeEveryN);
  const toggleProbe = useRunStore((state) => state.toggleProbe);
  const setProbeKind = useRunStore((state) => state.setProbeKind);
  const setProbeEveryN = useRunStore((state) => state.setProbeEveryN);
  const nodes = useGraphStore((state) => state.nodes);

  const running = isActiveStatus(status);
  const dataset = datasets?.find((item) => item.id === datasetId) ?? null;
  const progress = datasetId ? datasetProgress[datasetId] : undefined;
  const downloading = progress !== undefined && progress.phase !== "done" && progress.phase !== "error";
  const candidates = useMemo(() => probeCandidates(nodes), [nodes]);

  const frozen = (key: keyof RunConfig) =>
    running && key !== "lr" && key !== "batch_size";

  return (
    <div className="config">
      <div className="config__section">
        <div className="config__title">{t("run.dataset.title")}</div>
        <label className="config-field">
          <span className="config-field__label">{t("run.dataset.pick")}</span>
          <select
            className="config-field__input"
            value={datasetId ?? ""}
            disabled={running || !datasets || datasets.length === 0}
            onChange={(event) => setDataset(event.target.value)}
          >
            {(datasets ?? []).map((item) => (
              <option key={item.id} value={item.id}>
                {typeof item.name === "object"
                  ? (i18n.language.startsWith("zh") ? item.name.zh : item.name.en) ?? item.id
                  : item.id}
              </option>
            ))}
          </select>
        </label>

        {dataset ? (
          <div className="config__dataset">
            <span className={`badge ${dataset.cached ? "badge--ok" : "badge--warn"}`}>
              {dataset.cached ? t("run.dataset.cached") : t("run.dataset.notCached")}
            </span>
            <span className="config__meta">
              {dataset.input_shape.join("×")} ·{" "}
              {dataset.loader === "text_char"
                ? t("run.dataset.vocab", { n: dataset.vocab_size ?? dataset.num_classes })
                : dataset.num_classes}{" "}
              · {dataset.cached ? formatBytes(dataset.size_bytes) : formatBytes(dataset.total_bytes)}
            </span>
          </div>
        ) : null}

        {dataset && !dataset.cached && !downloading ? (
          <div className="config__row">
            <button type="button" className="btn" onClick={() => void downloadDataset(dataset.id)}>
              {t("run.dataset.download")}
            </button>
            <span className="config__meta" title={`${dataset.path} · ${dataset.manual_dir}`}>
              {t("run.dataset.manualHint")}
            </span>
          </div>
        ) : null}

        {downloading && progress ? (
          <div className="config__progress" title={`${Math.round(progress.progress * 100)}%`}>
            <div className="config__progress-bar" style={{ width: `${Math.round(progress.progress * 100)}%` }} />
            <span className="config__progress-text">
              {t(`run.dataset.phase.${progress.phase}`, { defaultValue: progress.phase })} ·{" "}
              {Math.round(progress.progress * 100)}%
              {progress.file ? ` · ${progress.file}` : ""}
            </span>
          </div>
        ) : null}
      </div>

      <div className="config__section">
        <div className="config__title">{t("run.hyper.title")}</div>
        <div className="config__grid">
          <label className="config-field">
            <span className="config-field__label">{t("run.hyper.optimizer")}</span>
            <select
              className="config-field__input"
              value={config.optimizer}
              disabled={frozen("optimizer")}
              onChange={(event) => setConfig({ optimizer: event.target.value as RunConfig["optimizer"] })}
            >
              {(["adam", "adamw", "sgd"] as const).map((name) => (
                <option key={name} value={name}>
                  {t(`run.hyper.optimizerName.${name}`)}
                </option>
              ))}
            </select>
          </label>
          <label className="config-field">
            <span className="config-field__label">{t("run.hyper.loss")}</span>
            <select
              className="config-field__input"
              value={config.loss}
              disabled={frozen("loss")}
              onChange={(event) => setConfig({ loss: event.target.value as RunConfig["loss"] })}
            >
              {(["cross_entropy", "mse"] as const).map((name) => (
                <option key={name} value={name}>
                  {t(`run.hyper.lossName.${name}`)}
                </option>
              ))}
            </select>
          </label>
          <NumberField
            label={t("run.hyper.lr")}
            value={config.lr}
            step={0.0001}
            min={1e-6}
            max={1}
            hint={running ? t("run.hyper.lrLive") : undefined}
            onCommit={(value) => void applyLr(value)}
          />
          <NumberField
            label={t("run.hyper.batchSize")}
            value={config.batch_size}
            step={1}
            min={1}
            max={4096}
            hint={running ? t("run.hyper.batchLive") : undefined}
            onCommit={(value) => void applyBatchSize(value)}
          />
          <NumberField
            label={t("run.hyper.epochs")}
            value={config.epochs}
            step={1}
            min={1}
            max={50}
            disabled={frozen("epochs")}
            onCommit={(value) => setConfig({ epochs: Math.round(value) })}
          />
          <NumberField
            label={t("run.hyper.gradClip")}
            value={config.grad_clip}
            step={0.5}
            min={0}
            max={1000}
            disabled={frozen("grad_clip")}
            onCommit={(value) => setConfig({ grad_clip: value })}
          />
          <NumberField
            label={t("run.hyper.trainSize")}
            value={config.train_size}
            step={1000}
            min={0}
            max={10000000}
            disabled={frozen("train_size")}
            hint={t("run.hyper.sizeHint")}
            onCommit={(value) => setConfig({ train_size: Math.round(value) })}
          />
          <NumberField
            label={t("run.hyper.valSize")}
            value={config.val_size}
            step={1000}
            min={0}
            max={1000000}
            disabled={frozen("val_size")}
            onCommit={(value) => setConfig({ val_size: Math.round(value) })}
          />
        </div>
        {running ? (
          <div className="config__hint">{t("run.hyper.lockedHint", { run: runId ?? "" })}</div>
        ) : null}
      </div>

      <div className="config__section">
        <div className="config__title">{t("run.probe.title")}</div>
        <div className="probe-pick">
          {candidates.map((candidate) => {
            const choice = probeChoices.find((item) => item.nodeId === candidate.nodeId);
            return (
              <div key={candidate.nodeId} className="probe-pick__row">
                <label className="probe-pick__check" title={candidate.nodeType}>
                  <input
                    type="checkbox"
                    checked={Boolean(choice)}
                    disabled={running}
                    onChange={() => toggleProbe(candidate.nodeId, candidate.kinds[0])}
                  />
                  <span className="probe-pick__name">{candidate.nodeId}</span>
                  <span className="probe-pick__type">{candidate.nodeType}</span>
                </label>
                {choice ? (
                  <select
                    className="probe-pick__kind"
                    value={choice.kind}
                    disabled={running}
                    onChange={(event) =>
                      setProbeKind(candidate.nodeId, event.target.value as typeof choice.kind)
                    }
                  >
                    {candidate.kinds.map((kind) => (
                      <option key={kind} value={kind}>
                        {t(`run.probe.kind.${kind}`)}
                      </option>
                    ))}
                  </select>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="config__row">
          <NumberField
            label={t("run.probe.everyN")}
            value={probeEveryN}
            step={10}
            min={1}
            max={100000}
            disabled={running}
            hint={t("run.probe.everyNHint")}
            onCommit={setProbeEveryN}
          />
          <span className="config__meta">{t("run.probe.picked", { n: probeChoices.length })}</span>
        </div>
        <div className="config__hint">
          {running ? t("run.probe.lockedHint") : t("run.probe.hint")}
        </div>
      </div>
    </div>
  );
}
