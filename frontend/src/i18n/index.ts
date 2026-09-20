import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en.json";
import zhCN from "./locales/zh-CN.json";

export const SUPPORTED_LANGUAGES = ["zh-CN", "en"] as const;
export type Language = (typeof SUPPORTED_LANGUAGES)[number];

const STORAGE_KEY = "ai-visualize.language";

function detectLanguage(): Language {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "zh-CN" || stored === "en") return stored;
  return navigator.language.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
}

void i18n.use(initReactI18next).init({
  resources: {
    "zh-CN": { translation: zhCN },
    en: { translation: en },
  },
  lng: detectLanguage(),
  fallbackLng: "zh-CN",
  interpolation: { escapeValue: false },
});

export function persistLanguage(language: Language): void {
  localStorage.setItem(STORAGE_KEY, language);
}

export default i18n;
