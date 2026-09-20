# 字符级语料（M3）

这两个文件随仓库分发，供「字符级小 GPT」预置离线训练使用（`loader = "text_char"`）。
已做清洗：去除页码/页眉/维基标记与目录，仅保留正文。

| 文件 | 来源 | 许可 |
| --- | --- | --- |
| `alice_en.txt` | Project Gutenberg eBook #11《Alice's Adventures in Wonderland》(Lewis Carroll, 1865) — https://www.gutenberg.org/ebooks/11 | 原文公有领域；Project Gutenberg License（去头尾的正文部分在美国不受版权保护） |
| `xiyouji_zh.txt` | 中文维基文库《西遊記》第 1–30 回 — https://zh.wikisource.org/wiki/西遊記/第001回 | 原文公有领域（明代，吴承恩）；维基文库转录文本 CC BY-SA 4.0 |

md5 记录在 `backend/app/datasets/registry.py` 的 `DatasetSpec.files` 中；若替换文件需同步更新 md5 与词表说明。
