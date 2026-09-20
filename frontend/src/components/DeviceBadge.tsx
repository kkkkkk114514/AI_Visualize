import { useTranslation } from "react-i18next";

import { useAppStore } from "../stores/appStore";

export function DeviceBadge() {
  const { t } = useTranslation();
  const health = useAppStore((state) => state.health);

  if (!health) {
    return <span className="badge">{t("device.loading")}</span>;
  }

  if (health.device === "cuda") {
    return (
      <span className="badge badge--cuda" title={health.device_name}>
        {t("device.cuda")} · {health.device_name}
      </span>
    );
  }

  return (
    <span className="badge" title={health.device_name}>
      {t("device.cpu")}
      {health.cpu_count ? ` · ${t("device.cpuCores", { count: health.cpu_count })}` : ""}
    </span>
  );
}
