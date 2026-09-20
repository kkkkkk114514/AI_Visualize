import type { LocalizedText } from "../api/types";

/** 取当前语言文案；纯字符串直接返回；缺失语言时回退另一种，最后省略。 */
export function localizedText(value: LocalizedText | null | undefined, language: string): string {
  if (!value) return "";
  if (typeof value === "string") return value;
  const primary = language.startsWith("zh") ? value.zh : value.en;
  if (primary) return primary;
  return value.zh ?? value.en ?? "";
}
