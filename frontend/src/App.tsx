import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Navigate, Route, Routes } from "react-router-dom";

import { wsClient } from "./api/ws";
import LabPage from "./pages/LabPage";
import LibraryPage from "./pages/LibraryPage";
import { useAppStore } from "./stores/appStore";
import { bindRunEvents } from "./stores/runStore";

export default function App() {
  const { t, i18n } = useTranslation();
  const loadHealth = useAppStore((state) => state.loadHealth);
  const setWsStatus = useAppStore((state) => state.setWsStatus);

  useEffect(() => {
    document.documentElement.lang = i18n.language;
    document.title = t("app.title");
  }, [t, i18n.language]);

  useEffect(() => {
    void loadHealth();
    const unsubscribe = wsClient.onStatus(setWsStatus);
    const unbindRuns = bindRunEvents();
    wsClient.connect();
    return () => {
      unsubscribe();
      unbindRuns();
    };
  }, [loadHealth, setWsStatus]);

  return (
    <Routes>
      <Route path="/" element={<LibraryPage />} />
      <Route path="/lab/:graphId" element={<LabPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
