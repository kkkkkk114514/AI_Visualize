import { useTranslation } from "react-i18next";

import { persistLanguage, type Language } from "../i18n";

const OPTIONS: { value: Language; label: string }[] = [
  { value: "zh-CN", label: "中文" },
  { value: "en", label: "English" },
];

export function LanguageSwitch() {
  const { i18n } = useTranslation();

  const change = (language: Language) => {
    void i18n.changeLanguage(language);
    persistLanguage(language);
  };

  return (
    <div className="lang-switch" role="group" aria-label="Language">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`lang-switch__item${i18n.language === option.value ? " is-active" : ""}`}
          onClick={() => change(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
