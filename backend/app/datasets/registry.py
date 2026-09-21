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
    # image = idx 图像（loaders.load_split）；text_char = 随仓库分发的字符级语料
    loader: str = "image"
    corpus: str | None = None
    vocab_cap: int = 0

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

# 上传集 id（docs/02 §7.5）：upload 模块在注入 / 删除时维护，列表据此标记可否删除
UPLOADED: set[str] = set()

TEXT_SEQ_LEN = 32

ALICE = DatasetSpec(
    id="alice",
    name={"zh": "《爱丽丝梦游仙境》字符级（英）", "en": "Alice in Wonderland (char-level)"},
    task="text_lm",
    input_shape=(TEXT_SEQ_LEN,),
    num_classes=0,  # 文本任务由词表决定（见 meta()）
    mean=0.0,
    std=0.0,
    splits=(),
    files=(DatasetFile("alice_en.txt", "b63ee23704b7b5c40833526e10ebf892", 154_475),),
    note={
        "zh": "Project Gutenberg #11（公版）字符级语料，<unk> + 约 80 个字符；随仓库分发，无需下载",
        "en": "Char-level corpus from Project Gutenberg #11 (public domain): <unk> + ~80 chars; shipped with the repo",
    },
    loader="text_char",
    corpus="alice_en.txt",
    vocab_cap=128,
)

XIYOUJI = DatasetSpec(
    id="xiyouji",
    name={"zh": "《西游记》前三十回字符级（繁）", "en": "Journey to the West ch.1–30 (char-level)"},
    task="text_lm",
    input_shape=(TEXT_SEQ_LEN,),
    num_classes=0,
    mean=0.0,
    std=0.0,
    splits=(),
    files=(DatasetFile("xiyouji_zh.txt", "512497e317e5ec058a20923821d56181", 637_553),),
    note={
        "zh": "中文维基文库《西遊記》第 1–30 回（繁体，公版）字符级语料，词表取高频 1023 字 + <unk>；随仓库分发，无需下载",
        "en": "Char-level corpus of Journey to the West ch.1–30 from Chinese Wikisource (traditional, public domain): top 1023 chars + <unk>; shipped with the repo",
    },
    loader="text_char",
    corpus="xiyouji_zh.txt",
    vocab_cap=1024,
)

SPECS.update({ALICE.id: ALICE, XIYOUJI.id: XIYOUJI})


def _synth(
    dataset_id: str, name: dict[str, str], note: dict[str, str]
) -> DatasetSpec:
    """二维合成集：文件为空 ⇒ is_cached 恒真，点集由 synth2d 按固定种子程序生成（§13.3）。"""
    return DatasetSpec(
        id=dataset_id,
        name=name,
        task="binary_classification",
        input_shape=(2,),
        num_classes=2,
        mean=0.0,
        std=1.0,
        splits=(),
        files=(),
        note=note,
        loader="synth2d",
    )


MOONS = _synth(
    "moons",
    {"zh": "双月（moons）", "en": "Two Moons"},
    {
        "zh": "程序生成 300 训练 + 200 验证（噪声 0.18 / 种子 7）；无需下载",
        "en": "Generated on the fly: 300 train + 200 val (noise 0.18 / seed 7); nothing to download",
    },
)

CIRCLES = _synth(
    "circles",
    {"zh": "同心圆（circles）", "en": "Concentric Circles"},
    {
        "zh": "程序生成 300 训练 + 200 验证（噪声 0.12 / 种子 11）；无需下载",
        "en": "Generated on the fly: 300 train + 200 val (noise 0.12 / seed 11); nothing to download",
    },
)

BLOBS = _synth(
    "blobs",
    {"zh": "双高斯团（blobs）", "en": "Gaussian Blobs"},
    {
        "zh": "程序生成 300 训练 + 200 验证（离散度 0.40 / 种子 13）；线性可分，作线性模型基线；无需下载",
        "en": "Generated on the fly: 300 train + 200 val (spread 0.40 / seed 13); linearly separable baseline; nothing to download",
    },
)

SPIRAL = _synth(
    "spiral",
    {"zh": "双螺旋（spiral）", "en": "Two Spirals"},
    {
        "zh": "程序生成 300 训练 + 200 验证（噪声 0.06 / 种子 17）；强非线性，核方法与树明显占优；无需下载",
        "en": "Generated on the fly: 300 train + 200 val (noise 0.06 / seed 17); strongly non-linear; nothing to download",
    },
)

GRIDWORLD = DatasetSpec(
    id="gridworld",
    name={"zh": "网格世界（12×8）", "en": "GridWorld (12×8)"},
    task="control",
    input_shape=(),
    num_classes=0,
    mean=0.0,
    std=1.0,
    splits=(),
    files=(),
    note={
        "zh": "无数据文件：环境（起点 / 终点 / 障碍 / 奖励）定义在模型里，可编辑后重训",
        "en": "No data files: the environment (start / goal / obstacles / rewards) lives in the model spec",
    },
    loader="gridworld",
)

SPECS.update(
    {spec.id: spec for spec in (MOONS, CIRCLES, BLOBS, SPIRAL, GRIDWORLD)}
)

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


def corpora_dir() -> Path:
    return config.CORPORA_DIR


def search_dirs(spec: DatasetSpec) -> tuple[tuple[str, Path], ...]:
    return (("raw", raw_dir(spec)), ("manual", manual_dir()), ("corpora", corpora_dir()))


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
    """按 raw → manual → corpora 顺序找到 md5 正确的文件。"""
    for _, directory in search_dirs(spec):
        candidate = directory / file.name
        if candidate.is_file() and _md5_matches(candidate, file.md5):
            return candidate
    return None


def missing_files(spec: DatasetSpec) -> list[DatasetFile]:
    return [item for item in spec.files if locate(spec, item) is None]


def is_cached(spec: DatasetSpec) -> bool:
    return not missing_files(spec)


def vocab_size(spec: DatasetSpec) -> int | None:
    """字符级语料的词表大小（含 <unk>）；非文本数据集返回 None。"""
    if spec.loader != "text_char" or not spec.corpus:
        return None
    from app.datasets import loaders  # 延迟导入：loaders 反向依赖本模块

    return loaders.text_vocab_size(spec)


def cache_state(spec: DatasetSpec) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    size_bytes = 0
    labels = {str(directory): label for label, directory in search_dirs(spec)}
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
                "source": labels.get(str(path.parent), path.parent.name),
            }
        )
    cached = all(item["md5_ok"] for item in files)
    vocab = vocab_size(spec)
    return {
        "id": spec.id,
        "name": spec.name,
        "task": spec.task,
        "loader": spec.loader,
        "uploaded": spec.id in UPLOADED,
        "input_shape": list(spec.input_shape),
        "num_classes": vocab if vocab else spec.num_classes,
        "vocab_size": vocab,
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
    vocab = vocab_size(spec)
    return {
        "id": spec.id,
        "task": spec.task,
        "loader": spec.loader,
        "input_shape": list(spec.input_shape),
        "num_classes": vocab if vocab else spec.num_classes,
        "mean": spec.mean,
        "std": spec.std,
        "vocab_size": vocab,
    }
