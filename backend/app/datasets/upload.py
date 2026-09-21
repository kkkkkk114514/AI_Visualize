"""数据集上传（契约见 docs/02 §7.5）：zip 图片集 / csv 点集 / txt 语料 → 落盘 + 清单 + 注册表注入。

解码 / 缩放 / 切分 / 统计都在上传时完成，产物写进 `data/datasets/{id}/raw/`；此后与内置集
走同一条读取通路（registry.locate / is_cached / cache_state / loaders.build_loaders）。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import secrets
import shutil
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app import config
from app.datasets import csv2d, registry

log = logging.getLogger(__name__)

IMAGE_SIZE = 64
IMAGE_CLASS_MIN = 2
IMAGE_CLASS_MAX = 20
IMAGE_PER_CLASS_MIN = 8
IMAGE_TOTAL_MAX = 2000
ZIP_INFLATE_MAX = 128 << 20
TEXT_MIN_CHARS = 2000
TEXT_VOCAB_CAP = 1024
IMAGE_VAL_FRACTION = 0.1

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FORMAT_EXT = {".zip": "zip", ".csv": "csv", ".txt": "txt"}
PREPARED_FILE = {"zip": "data.npz", "csv": "points.csv", "txt": "corpus.txt"}

_lock = threading.Lock()


class UploadError(Exception):
    """REST 层可翻译的上传错误：HTTP 状态 + code + i18n key + detail（具体原因，英文调试文案）。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message_key: str,
        error_args: dict[str, Any] | None = None,
        detail: str = "",
    ):
        super().__init__(detail or message_key)
        self.status_code = status_code
        self.code = code
        self.message_key = message_key
        self.error_args = error_args or {}
        self.detail = detail


@dataclass
class Prepared:
    loader: str
    payload: bytes
    meta: dict[str, Any]
    note: dict[str, str]
    task: str
    input_shape: tuple[int, ...]
    num_classes: int
    mean: float = 0.0
    std: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 格式与清单

def resolve_format(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    fmt = FORMAT_EXT.get(ext)
    if fmt is None:
        raise UploadError(
            415,
            "upload_bad_type",
            "errors.upload.badType",
            {"ext": ext or Path(filename or "").name or "(none)"},
        )
    return fmt


def limit_mb(fmt: str) -> int:
    limits = {
        "zip": config.UPLOAD_ZIP_MAX_MB,
        "csv": config.UPLOAD_CSV_MAX_MB,
        "txt": config.UPLOAD_TXT_MAX_MB,
    }
    return limits[fmt]


def manifest_path() -> Path:
    return config.DATASETS_DIR / "uploads.json"


def read_manifest() -> list[dict[str, Any]]:
    path = manifest_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("上传清单无法解析，本次忽略全部条目：%s", exc)
        return []
    entries = payload.get("uploads") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    return [item for item in entries if isinstance(item, dict)]


def write_manifest(entries: list[dict[str, Any]]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"uploads": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def is_uploaded(dataset_id: str) -> bool:
    return dataset_id in registry.UPLOADED


def _spec_from_entry(entry: dict[str, Any]) -> registry.DatasetSpec | None:
    """清单条目 → DatasetSpec（启动注入与上传即时注册共用同一条构造路径）。"""
    try:
        dataset_id = str(entry["id"])
        loader = str(entry["loader"])
        file = entry["file"]
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        raw_name = entry.get("name")
        name = raw_name if isinstance(raw_name, dict) else {"zh": dataset_id, "en": dataset_id}
        raw_note = entry.get("note")
        note = raw_note if isinstance(raw_note, dict) else {"zh": "", "en": ""}
        dataset_file = registry.DatasetFile(
            str(file["name"]), str(file["md5"]), int(file["size"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("上传清单条目损坏，已跳过：%s（%s）", exc, entry.get("id"))
        return None

    if loader == "image_dir":
        shape = tuple(int(value) for value in meta.get("input_shape") or [3, IMAGE_SIZE, IMAGE_SIZE])
        return registry.DatasetSpec(
            id=dataset_id,
            name=name,
            task="image_classification",
            input_shape=shape,
            num_classes=int(meta.get("num_classes") or 2),
            mean=float(meta.get("mean") or 0.0),
            std=float(meta.get("std") or 1.0),
            splits=(),
            files=(dataset_file,),
            note=note,
            loader="image_dir",
        )
    if loader == "csv2d":
        return registry.DatasetSpec(
            id=dataset_id,
            name=name,
            task="binary_classification",
            input_shape=(2,),
            num_classes=2,
            mean=0.0,
            std=1.0,
            splits=(),
            files=(dataset_file,),
            note=note,
            loader="csv2d",
            corpus=dataset_file.name,
        )
    if loader == "text_char":
        return registry.DatasetSpec(
            id=dataset_id,
            name=name,
            task="text_lm",
            input_shape=(registry.TEXT_SEQ_LEN,),
            num_classes=0,
            mean=0.0,
            std=0.0,
            splits=(),
            files=(dataset_file,),
            note=note,
            loader="text_char",
            corpus=dataset_file.name,
            vocab_cap=int(meta.get("vocab_cap") or TEXT_VOCAB_CAP),
        )
    log.warning("上传清单条目的 loader 未知，已跳过：%s（%s）", loader, dataset_id)
    return None


def register_all() -> None:
    """启动时把清单里的上传集注入 registry.SPECS（内置 id 与损坏条目一律跳过）。"""
    for entry in read_manifest():
        dataset_id = entry.get("id")
        spec = _spec_from_entry(entry)
        if spec is None:
            continue
        if spec.id in registry.SPECS and spec.id not in registry.UPLOADED:
            log.warning("上传清单的 id 与内置数据集冲突，已跳过：%s", dataset_id)
            continue
        registry.SPECS[spec.id] = spec
        registry.UPLOADED.add(spec.id)
    if registry.UPLOADED:
        log.info(
            "已载入 %d 个上传数据集：%s",
            len(registry.UPLOADED),
            ", ".join(sorted(registry.UPLOADED)),
        )


# ---------------------------------------------------------------- 上传入口

def _slug(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:24].strip("-")
    return slug or "data"


def _new_id(slug: str) -> str:
    for _ in range(64):
        candidate = f"up-{slug}-{secrets.token_hex(2)}"
        if candidate not in registry.SPECS and not (config.DATASETS_DIR / candidate).exists():
            return candidate
    raise RuntimeError("无法分配上传数据集 id")


def handle_upload(filename: str, data: bytes) -> registry.DatasetSpec:
    """校验 + 处理 + 落盘 + 记清单 + 注入注册表；成功后返回新 spec。"""
    safe_name = Path(filename or "").name
    fmt = resolve_format(safe_name)
    cap = limit_mb(fmt)
    if len(data) > cap << 20:
        raise UploadError(
            413,
            "upload_too_large",
            "errors.upload.tooLarge",
            {"format": fmt, "limit_mb": cap},
            f"{len(data)} bytes exceeds {cap}MB",
        )
    if not data:
        raise UploadError(
            422, "upload_bad_content", "errors.upload.badContent", {"format": fmt}, "empty file"
        )

    prepared = _PROCESSORS[fmt](data)
    stem = safe_name[: -len(Path(safe_name).suffix)] or "data"
    dataset_id = _new_id(_slug(stem))
    raw_dir = config.DATASETS_DIR / dataset_id / "raw"
    display = stem[:48]

    with _lock:
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = raw_dir / PREPARED_FILE[fmt]
            target.write_bytes(prepared.payload)
            entry = {
                "id": dataset_id,
                "loader": prepared.loader,
                "name": {"zh": f"{display}（上传）", "en": f"{display} (upload)"},
                "file": {
                    "name": target.name,
                    "md5": hashlib.md5(prepared.payload).hexdigest(),
                    "size": len(prepared.payload),
                },
                "meta": prepared.meta,
                "note": prepared.note,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            write_manifest([*read_manifest(), entry])
        except OSError as exc:
            shutil.rmtree(config.DATASETS_DIR / dataset_id, ignore_errors=True)
            raise UploadError(
                500,
                "upload_write_failed",
                "errors.upload.writeFailed",
                {"id": dataset_id},
                str(exc),
            ) from exc

    spec = _spec_from_entry(entry)
    if spec is None:  # 构造路径与启动注入一致，正常不会发生
        raise RuntimeError(f"上传条目无法构造 spec：{dataset_id}")
    registry.SPECS[dataset_id] = spec
    registry.UPLOADED.add(dataset_id)
    log.info("数据集 %s 上传完成（%s，%d 字节）", dataset_id, prepared.loader, len(prepared.payload))
    return spec


def remove_uploaded(dataset_id: str) -> None:
    with _lock:
        write_manifest([item for item in read_manifest() if item.get("id") != dataset_id])
    shutil.rmtree(config.DATASETS_DIR / dataset_id, ignore_errors=True)
    registry.SPECS.pop(dataset_id, None)
    registry.UPLOADED.discard(dataset_id)
    log.info("数据集 %s 已删除", dataset_id)


# ---------------------------------------------------------------- 三种格式

def _process_zip(data: bytes) -> Prepared:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadError(
            422, "upload_bad_content", "errors.upload.badContent", {"format": "zip"}, f"not a zip archive: {exc}"
        ) from exc

    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        inflated = sum(info.file_size for info in infos)
        if inflated > ZIP_INFLATE_MAX:
            raise UploadError(
                422,
                "upload_bad_content",
                "errors.upload.badContent",
                {"format": "zip"},
                f"inflated size {inflated} exceeds {ZIP_INFLATE_MAX >> 20}MB",
            )
        buckets: dict[str, list[zipfile.ZipInfo]] = {}
        for info in infos:
            parts = [part for part in info.filename.replace("\\", "/").split("/") if part not in ("", ".")]
            if len(parts) < 2:  # 根下散图
                continue
            if parts[0] == "__MACOSX" or any(part.startswith(".") for part in parts):
                continue
            if Path(parts[-1]).suffix.lower() not in IMAGE_EXTS:
                continue
            buckets.setdefault(parts[0], []).append(info)

        classes = sorted(buckets)
        if not IMAGE_CLASS_MIN <= len(classes) <= IMAGE_CLASS_MAX:
            raise UploadError(
                422,
                "upload_bad_content",
                "errors.upload.badContent",
                {"format": "zip"},
                f"need {IMAGE_CLASS_MIN}..{IMAGE_CLASS_MAX} class folders, got {len(classes)}",
            )
        total = sum(len(members) for members in buckets.values())
        if total > IMAGE_TOTAL_MAX:
            raise UploadError(
                422,
                "upload_bad_content",
                "errors.upload.badContent",
                {"format": "zip"},
                f"too many images: {total} > {IMAGE_TOTAL_MAX}",
            )
        for label in classes:
            if len(buckets[label]) < IMAGE_PER_CLASS_MIN:
                raise UploadError(
                    422,
                    "upload_bad_content",
                    "errors.upload.badContent",
                    {"format": "zip"},
                    f"class '{label}' has {len(buckets[label])} images, need >= {IMAGE_PER_CLASS_MIN}",
                )

        rng = np.random.default_rng(config.DEFAULT_SEED)
        train: list[np.ndarray] = []
        train_labels: list[int] = []
        val: list[np.ndarray] = []
        val_labels: list[int] = []
        for label, name in enumerate(classes):
            members = sorted(buckets[name], key=lambda info: info.filename)
            n_val = max(1, int(round(len(members) * IMAGE_VAL_FRACTION)))
            for rank, slot in enumerate(rng.permutation(len(members))):
                array = _decode_image(archive, members[int(slot)])
                if rank < len(members) - n_val:
                    train.append(array)
                    train_labels.append(label)
                else:
                    val.append(array)
                    val_labels.append(label)

    train_x = np.stack(train)
    val_x = np.stack(val)
    both = np.concatenate([train_x, val_x]).astype(np.float32) / 255.0
    mean = round(float(both.mean()), 6)
    std = round(float(both.std()), 6) or 1.0

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        train_x=train_x,
        train_y=np.asarray(train_labels, dtype=np.int64),
        val_x=val_x,
        val_y=np.asarray(val_labels, dtype=np.int64),
    )
    shown = ", ".join(classes[:6]) + ("…" if len(classes) > 6 else "")
    return Prepared(
        loader="image_dir",
        payload=buffer.getvalue(),
        meta={
            "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
            "num_classes": len(classes),
            "mean": mean,
            "std": std,
            "classes": classes,
        },
        note={
            "zh": f"上传的图片集：{len(classes)} 类（{shown}），共 {total} 张 → 统一 {IMAGE_SIZE}×{IMAGE_SIZE} RGB，"
            f"90/10 切分（seed 42）；训练 {train_x.shape[0]} / 验证 {val_x.shape[0]}",
            "en": f"Uploaded image set: {len(classes)} classes ({shown}), {total} images → "
            f"{IMAGE_SIZE}×{IMAGE_SIZE} RGB, 90/10 split (seed 42); {train_x.shape[0]} train / {val_x.shape[0]} val",
        },
        task="image_classification",
        input_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        num_classes=len(classes),
        mean=mean,
        std=std,
        extra={"classes": classes},
    )


def _decode_image(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> np.ndarray:
    try:
        with archive.open(info) as handle:
            image = Image.open(io.BytesIO(handle.read()))
            image.load()
            rgb = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
            )
    except Exception as exc:  # Pillow 抛出的类型很杂（UnidentifiedImageError / OSError / ValueError…）
        raise UploadError(
            422,
            "upload_bad_content",
            "errors.upload.badContent",
            {"format": "zip"},
            f"cannot decode {info.filename}: {exc}",
        ) from exc
    return np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1)


def _process_csv(data: bytes) -> Prepared:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError(
            422, "upload_bad_content", "errors.upload.badContent", {"format": "csv"}, f"not valid UTF-8: {exc}"
        ) from exc
    try:
        _, labels = csv2d.parse_text(text)
    except ValueError as exc:
        raise UploadError(
            422, "upload_bad_content", "errors.upload.badContent", {"format": "csv"}, str(exc)
        ) from exc
    positives = int(labels.sum())
    rows = int(labels.shape[0])
    return Prepared(
        loader="csv2d",
        payload=data,
        meta={"input_shape": [2], "num_classes": 2, "rows": rows},
        note={
            "zh": f"上传的二维点集：{rows} 行（类 1 占 {positives} / 类 0 占 {rows - positives}）→ "
            f"seed 42 打乱后 80/20 切分；可直接训练或画决策边界",
            "en": f"Uploaded 2-D point set: {rows} rows (class 1: {positives} / class 0: {rows - positives}) → "
            f"shuffled with seed 42, 80/20 split; trainable with the boundary canvas",
        },
        task="binary_classification",
        input_shape=(2,),
        num_classes=2,
    )


def _process_txt(data: bytes) -> Prepared:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadError(
            422, "upload_bad_content", "errors.upload.badContent", {"format": "txt"}, f"not valid UTF-8: {exc}"
        ) from exc
    count = len(text)
    if count < TEXT_MIN_CHARS:
        raise UploadError(
            422,
            "upload_bad_content",
            "errors.upload.badContent",
            {"format": "txt"},
            f"need >= {TEXT_MIN_CHARS} characters, got {count}",
        )
    distinct = len(set(text))
    return Prepared(
        loader="text_char",
        payload=data,
        meta={"vocab_cap": TEXT_VOCAB_CAP, "chars": count},
        note={
            "zh": f"上传的字符级语料：{count} 字符 / {distinct} 种 → 词表上限 {TEXT_VOCAB_CAP}"
            f"（<unk> + 高频字），前 90% 训练 / 后 10% 验证",
            "en": f"Uploaded char-level corpus: {count} chars / {distinct} distinct → vocab cap {TEXT_VOCAB_CAP}"
            f" (<unk> + top chars), first 90% train / last 10% val",
        },
        task="text_lm",
        input_shape=(registry.TEXT_SEQ_LEN,),
        num_classes=0,
    )


_PROCESSORS = {"zip": _process_zip, "csv": _process_csv, "txt": _process_txt}
