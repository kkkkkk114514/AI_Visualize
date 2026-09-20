"""数据集注册表（契约见 docs/02 §7.1）：文件清单、md5 校验与缓存状态。"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import config


@dataclass(frozen=True)
class DatasetFile:
    name: str
    md5: str
    size: int


@dataclass(frozen=True)
class DatasetSplit:
    key: str
    images: str
    labels: str


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: dict[str, Any]
    task: str
    input_shape: tuple[int, ...]
    num_classes: int
    mean: float
    std: float
    splits: tuple[DatasetSplit, ...]
    files: tuple[DatasetFile, ...]
    note: dict[str, Any]

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    def split(self, key: str) -> DatasetSplit | None:
        for item in self.splits:
            if item.key == key:
                return item
        return None

    def file(self, name: str) -> DatasetFile | None:
        for item in self.files:
            if item.name == name:
                return item
        return None


MNIST = DatasetSpec(
    id="mnist",
    name={"zh": "MNIST 手写数字", "en": "MNIST Digits"},
    task="image_classification",
    input_shape=(1, 28, 28),
    num_classes=10,
    mean=0.1307,
    std=0.3081,
    splits=(
        DatasetSplit("train", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        DatasetSplit("val", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    ),
    files=(
        DatasetFile("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 9_912_422),
        DatasetFile("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 28_881),
        DatasetFile("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 1_648_877),
        DatasetFile("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 4_542),
    ),
    note={
        "zh": "训练 6 万 / 验证 1 万；缓存目录 data/datasets/mnist/raw，或把官方压缩包放到 data/datasets/manual",
        "en": "60k train / 10k val; cached in data/datasets/mnist/raw, or drop the official archives into data/datasets/manual",
    },
)

SPECS: dict[str, DatasetSpec] = {MNIST.id: MNIST}

_memo_lock = threading.Lock()
_md5_memo: dict[tuple[str, int, int], bool] = {}


def get_spec(dataset_id: str) -> DatasetSpec | None:
    return SPECS.get(dataset_id)


def dataset_dir(spec: DatasetSpec) -> Path:
    return config.DATASETS_DIR / spec.id


def raw_dir(spec: DatasetSpec) -> Path:
    return dataset_dir(spec) / "raw"


def manual_dir() -> Path:
    return config.DATASETS_DIR / "manual"


def search_dirs(spec: DatasetSpec) -> tuple[tuple[str, Path], ...]:
    return (("raw", raw_dir(spec)), ("manual", manual_dir()))


def _md5_matches(path: Path, expected: str) -> bool:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _memo_lock:
        cached = _md5_memo.get(key)
    if cached is not None:
        return cached
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    ok = digest.hexdigest() == expected
    with _memo_lock:
        _md5_memo[key] = ok
    return ok


def locate(spec: DatasetSpec, file: DatasetFile) -> Path | None:
    """按 raw → manual 顺序找到 md5 正确的文件。"""
    for _, directory in search_dirs(spec):
        candidate = directory / file.name
        if candidate.is_file() and _md5_matches(candidate, file.md5):
            return candidate
    return None


def missing_files(spec: DatasetSpec) -> list[DatasetFile]:
    return [item for item in spec.files if locate(spec, item) is None]


def is_cached(spec: DatasetSpec) -> bool:
    return not missing_files(spec)


def cache_state(spec: DatasetSpec) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    size_bytes = 0
    for item in spec.files:
        path = locate(spec, item)
        if path is None:
            raw_candidate = raw_dir(spec) / item.name
            present = raw_candidate.is_file()
            files.append(
                {"name": item.name, "present": present, "md5_ok": False, "bytes": item.size}
            )
            continue
        size_bytes += item.size
        files.append(
            {
                "name": item.name,
                "present": True,
                "md5_ok": True,
                "bytes": item.size,
                "source": "raw" if path.parent == raw_dir(spec) else "manual",
            }
        )
    cached = all(item["md5_ok"] for item in files)
    return {
        "id": spec.id,
        "name": spec.name,
        "task": spec.task,
        "input_shape": list(spec.input_shape),
        "num_classes": spec.num_classes,
        "cached": cached,
        "size_bytes": size_bytes if cached else 0,
        "total_bytes": spec.total_bytes,
        "path": str(raw_dir(spec)),
        "manual_dir": str(manual_dir()),
        "files": files,
        "note": spec.note,
    }


def list_payload() -> list[dict[str, Any]]:
    return [cache_state(spec) for spec in SPECS.values()]


def meta(spec: DatasetSpec) -> dict[str, Any]:
    """训练侧要用的元数据（docs/02 §4.1：dummy 输入与数据绑定参数一律来自数据集）。"""
    return {
        "id": spec.id,
        "task": spec.task,
        "input_shape": list(spec.input_shape),
        "num_classes": spec.num_classes,
        "mean": spec.mean,
        "std": spec.std,
        "vocab_size": None,
    }
