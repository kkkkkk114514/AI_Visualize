import { useTranslation } from "react-i18next";

import { useAppStore } from "../stores/appStore";

export function ConnectionStatus() {
  const { t } = useTranslation();
  const wsStatus = useAppStore((state) => state.wsStatus);

  return (
    <span className={`conn conn--${wsStatus}`}>
      <span className="conn__dot" aria-hidden />
      {t(`connection.${wsStatus}`)}
    </span>
  );
}
