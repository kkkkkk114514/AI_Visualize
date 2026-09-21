import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { deleteDataset, uploadDataset } from "../api/datasets";
import { ApiError, apiGet } from "../api/rest";
import type { DatasetInfo, ModelKind, ModelsResponse } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DeviceBadge } from "../components/DeviceBadge";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { localizedText } from "../i18n/localized";
import { useRunStore } from "../stores/runStore";

const SECTION_ORDER: ModelKind[] = ["ml", "dl", "rl"];
/** 上传入口接受的后缀，与后端 `FORMAT_EXT` 一致（docs/02 §7.5）。 */
const UPLOAD_ACCEPT = ".zip,.csv,.txt";

/** 待展示的错误：message_key 与 args 交给 t() 渲染双语，detail 是服务端给出的具体原因。 */
interface Notice {
  key: string;
  args: Record<string, unknown>;
  detail?: string;
}

function notice(cause: unknown, fallbackKey: string): Notice {
  if (cause instanceof ApiError) {
    const payload = cause.payload;
    return {
      key: payload?.message_key ?? fallbackKey,
      args: payload?.args ?? {},
      // 有 payload 时 detail 只认服务端字段：无 detail 的响应里 ApiError.message 就是 message_key 本身
      detail: payload ? payload.detail : cause.message,
    };
  }
  return { key: fallbackKey, args: {}, detail: cause instanceof Error ? cause.message : String(cause) };
}

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export default function LibraryPage() {
  const { t, i18n } = useTranslation();
  const [groups, setGroups] = useState<ModelsResponse["groups"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const datasets = useRunStore((state) => state.datasets);
  const datasetsError = useRunStore((state) => state.datasetsError);
  const loadDatasets = useRunStore((state) => state.loadDatasets);
  const [noticeError, setNoticeError] = useState<Notice | null>(null);
  const [uploading, setUploading] = useState(false);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const data = await apiGet<ModelsResponse>("/api/models");
      setGroups(data.groups);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  // 列表刷新后，指向已消失数据集的确认态自动失效
  useEffect(() => {
    if (confirm && datasets && !datasets.some((item) => item.id === confirm)) setConfirm(null);
  }, [datasets, confirm]);

  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    setNoticeError(null);
    try {
      await uploadDataset(file);
      await loadDatasets();
    } catch (cause) {
      setNoticeError(notice(cause, "errors.upload.failed"));
    } finally {
      setUploading(false);
      // 清空 input，同一个文件改完再传也能触发 change
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const onRemove = async (datasetId: string) => {
    setConfirm(null);
    setRemoving(datasetId);
    setNoticeError(null);
    try {
      await deleteDataset(datasetId);
      await loadDatasets();
    } catch (cause) {
      setNoticeError(notice(cause, "errors.dataset.deleteFailed"));
    } finally {
      setRemoving(null);
    }
  };

  const datasetMeta = (item: DatasetInfo) => {
    const classes =
      item.loader === "text_char"
        ? t("run.dataset.vocab", { n: item.vocab_size ?? item.num_classes })
        : `${item.num_classes} ${t("algo.classes")}`;
    const bytes = item.cached ? item.size_bytes : item.total_bytes;
    // gridworld 没有输入形状 / 类别 / 文件，空项一律略过（否则会留下「 · 0 类」这类残句）
    return [
      item.input_shape.length > 0 ? item.input_shape.join("×") : null,
      item.num_classes > 0 ? classes : null,
      bytes > 0 ? formatBytes(bytes) : null,
    ]
      .filter(Boolean)
      .join(" · ");
  };

  return (
    <div className="app-shell">
      <AppHeader
        left={
          <>
            <span className="app-header__title">{t("app.title")}</span>
            <span className="app-header__subtitle">{t("app.subtitle")}</span>
          </>
        }
        actions={
          <>
            <DeviceBadge />
            <ConnectionStatus />
            <LanguageSwitch />
          </>
        }
      />
      <main className="page">
        <div className="page__intro">
          <div>
            <h1>{t("library.title")}</h1>
            <p>{t("library.intro")}</p>
          </div>
          <Link className="btn" to="/lab/new">
            {t("library.newModel")}
          </Link>
        </div>

        {error ? (
          <div className="error-state">
            <span>
              {t("library.loadError")}：{error}
            </span>
            <button type="button" className="btn" onClick={() => void load()}>
              {t("library.retry")}
            </button>
          </div>
        ) : null}

        <div className="sections">
          {SECTION_ORDER.map((kind) => {
            const models = groups?.[kind] ?? [];
            return (
              <section key={kind} className="section">
                <div className="section__header">
                  <h2>{t(`library.sections.${kind}`)}</h2>
                  {models.length > 0 ? <span className="section__count">{models.length}</span> : null}
                </div>
                {models.length === 0 ? (
                  <p className="section__empty">{t("library.empty")}</p>
                ) : (
                  <div className="model-grid">
                    {models.map((model) => (
                      <Link key={model.id} className="model-card" to={`/lab/${encodeURIComponent(model.id)}`}>
                        <div className="model-card__name">{localizedText(model.name, i18n.language) || model.id}</div>
                        <div className="model-card__desc">{localizedText(model.desc, i18n.language)}</div>
                        <div className="model-card__meta">
                          <span className={`chip chip--${model.kind}`}>{model.kind.toUpperCase()}</span>
                          <span className="chip chip--muted">{t(`library.source.${model.source}`)}</span>
                        </div>
                      </Link>
                    ))}
                  </div>
                )}
              </section>
            );
          })}

          <section className="section">
            <div className="section__header">
              <h2>{t("library.datasets.title")}</h2>
              {datasets ? <span className="section__count">{datasets.length}</span> : null}
              <div className="section__actions">
                <button
                  type="button"
                  className="btn"
                  disabled={uploading}
                  onClick={() => fileRef.current?.click()}
                >
                  {uploading ? t("library.datasets.uploading") : t("library.datasets.upload")}
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept={UPLOAD_ACCEPT}
                  hidden
                  onChange={(event) => void onUpload(event.target.files?.[0])}
                />
              </div>
            </div>
            <p className="section__hint">{t("library.datasets.hint")}</p>

            {noticeError ? (
              <div className="error-state">
                <span>
                  {t(noticeError.key, { ...noticeError.args, defaultValue: noticeError.key })}
                  {noticeError.detail ? `：${noticeError.detail}` : ""}
                </span>
                <button type="button" className="btn" onClick={() => setNoticeError(null)}>
                  {t("library.datasets.dismiss")}
                </button>
              </div>
            ) : null}

            {datasetsError ? (
              <div className="error-state">
                <span>
                  {t("library.datasets.loadError")}：{datasetsError}
                </span>
                <button type="button" className="btn" onClick={() => void loadDatasets()}>
                  {t("library.retry")}
                </button>
              </div>
            ) : null}

            {datasets === null ? (
              <p className="section__empty">{t("common.loading")}</p>
            ) : datasets.length === 0 ? (
              <p className="section__empty">{t("library.datasets.empty")}</p>
            ) : (
              <div className="dataset-grid">
                {datasets.map((item) => {
                  const busy = removing === item.id;
                  return (
                    <div key={item.id} className="dataset-card">
                      <div className="dataset-card__head">
                        <span className="dataset-card__name">
                          {localizedText(item.name, i18n.language) || item.id}
                        </span>
                        <span className={`badge ${item.cached ? "badge--ok" : "badge--warn"}`}>
                          {item.cached ? t("run.dataset.cached") : t("run.dataset.notCached")}
                        </span>
                      </div>
                      <p className="dataset-card__note">{localizedText(item.note, i18n.language)}</p>
                      <div className="dataset-card__meta">
                        <span className="chip">{item.loader}</span>
                        <span className="dataset-card__specs">{datasetMeta(item)}</span>
                      </div>
                      <div className="dataset-card__foot">
                        <span
                          className="dataset-card__path"
                          title={item.cached ? item.path : `${item.path} · ${item.manual_dir}`}
                        >
                          {item.path}
                        </span>
                        {item.uploaded ? (
                          <span className="dataset-card__actions">
                            {confirm === item.id ? (
                              <>
                                <button
                                  type="button"
                                  className="btn btn--danger"
                                  disabled={busy}
                                  onClick={() => void onRemove(item.id)}
                                >
                                  {t("library.datasets.deleteConfirm")}
                                </button>
                                <button
                                  type="button"
                                  className="btn btn--ghost"
                                  disabled={busy}
                                  onClick={() => setConfirm(null)}
                                >
                                  {t("library.datasets.cancel")}
                                </button>
                              </>
                            ) : (
                              <button
                                type="button"
                                className="btn btn--ghost"
                                disabled={busy}
                                onClick={() => setConfirm(item.id)}
                              >
                                {t("library.datasets.delete")}
                              </button>
                            )}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
