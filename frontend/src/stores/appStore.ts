import { create } from "zustand";

import { apiGet } from "../api/rest";
import type { HealthInfo } from "../api/types";
import type { WsStatus } from "../api/ws";

interface AppState {
  health: HealthInfo | null;
  healthError: string | null;
  wsStatus: WsStatus;
  loadHealth: () => Promise<void>;
  setWsStatus: (status: WsStatus) => void;
}

export const useAppStore = create<AppState>((set) => ({
  health: null,
  healthError: null,
  wsStatus: "disconnected",
  loadHealth: async () => {
    try {
      const health = await apiGet<HealthInfo>("/api/health");
      set({ health, healthError: null });
    } catch (error) {
      set({ healthError: error instanceof Error ? error.message : String(error) });
    }
  },
  setWsStatus: (wsStatus) => set({ wsStatus }),
}));
