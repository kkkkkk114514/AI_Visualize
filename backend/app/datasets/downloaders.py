"""数据集下载（契约见 docs/02 §7.1 / §8）：多镜像回退、.part 临时文件、md5 校验、进度回调。"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from app import config
from app.datasets import registry

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, dict[str, Any]], None]

_active: set[str] = set()
_active_lock = threading.Lock()

_last_report: dict[str, float] = {}


class DatasetDownloadError(Exception):
    def __init__(
        self, code: str, message_key: str, detail: str = "", error_args: dict[str, Any] | None = None
    ):
        super().__init__(detail or message_key)
        self.code = code
        self.message_key = message_key
        self.detail = detail
        self.error_args = error_args or {}


def is_downloading(dataset_id: str) -> bool:
    with _active_lock:
        return dataset_id in _active


def _claim(dataset_id: str) -> bool:
    with _active_lock:
        if dataset_id in _active:
            return False
        _active.add(dataset_id)
        return True


def _release(dataset_id: str) -> None:
    with _active_lock:
        _active.discard(dataset_id)


def _report(dataset_id: str, on_progress: ProgressCallback | None, phase: str, progress: float, extra: dict[str, Any] | None = None, force: bool = False) -> None:
    if on_progress is None:
        return
    now = time.monotonic()
    if not force and now - _last_report.get(dataset_id, 0.0) < 0.2:
        return
    _last_report[dataset_id] = now
    on_progress(phase, max(0.0, min(1.0, progress)), extra or {})


def _download_file(url: str, dest: Path, expected_md5: str, on_chunk: Callable[[int], None]) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    request = urllib.request.Request(url, headers={"User-Agent": config.SERVICE_NAME})
    try:
        with urllib.request.urlopen(request, timeout=config.DATASET_TIMEOUT_S) as response, part.open("wb") as fh:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                digest.update(chunk)
                on_chunk(len(chunk))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        raise DatasetDownloadError(
            "download_failed",
            "errors.dataset.downloadFailed",
            detail=f"{url}: {type(exc).__name__}: {exc}",
        ) from exc
    if digest.hexdigest() != expected_md5:
        part.unlink(missing_ok=True)
        raise DatasetDownloadError(
            "checksum_failed",
            "errors.dataset.checksum",
            detail=f"{dest.name}: md5={digest.hexdigest()} expected={expected_md5}",
            error_args={"file": dest.name},
        )
    part.replace(dest)


def download(spec: registry.DatasetSpec, on_progress: ProgressCallback | None = None) -> None:
    """同步下载缺失文件（幂等）；调用方负责线程调度与事件广播。"""
    pending = registry.missing_files(spec)
    if not pending:
        _report(spec.id, on_progress, "done", 1.0, {"detail": "cached"}, force=True)
        return

    total = sum(item.size for item in pending)
    done_bytes = 0
    for index, item in enumerate(pending):
        _report(
            spec.id,
            on_progress,
            "download",
            done_bytes / total if total else 0.0,
            {"file": item.name, "index": index + 1, "count": len(pending)},
            force=True,
        )
        local_bytes = [0]
        errors: list[DatasetDownloadError] = []
        for mirror in config.DATASET_MIRRORS:
            url = f"{mirror}{item.name}"

            def on_chunk(size: int, _item: registry.DatasetFile = item) -> None:
                local_bytes[0] += size
                _report(
                    spec.id,
                    on_progress,
                    "download",
                    (done_bytes + local_bytes[0]) / total if total else 0.0,
                    {"file": _item.name, "index": index + 1, "count": len(pending)},
                )

            try:
                _download_file(url, registry.raw_dir(spec) / item.name, item.md5, on_chunk)
                errors.clear()
                break
            except DatasetDownloadError as exc:
                errors.append(exc)
                log.warning("镜像失败 %s：%s", url, exc.detail or exc.message_key)
                local_bytes[0] = 0
        if errors:
            raise errors[-1]
        done_bytes += item.size

    _report(spec.id, on_progress, "verify", 1.0, {}, force=True)
    if not registry.is_cached(spec):
        raise DatasetDownloadError(
            "checksum_failed",
            "errors.dataset.checksum",
            detail="下载完成后校验未通过",
        )
    _report(spec.id, on_progress, "done", 1.0, {}, force=True)


def start_download(dataset_id: str, on_progress: ProgressCallback | None = None) -> bool:
    """在后台线程里启动下载；已在下载则返回 False（由调用方回 409）。"""
    spec = registry.get_spec(dataset_id)
    if spec is None or registry.is_cached(spec):
        return False
    if not _claim(dataset_id):
        return False

    def worker() -> None:
        try:
            download(spec, on_progress)
        except DatasetDownloadError as exc:
            log.warning("数据集 %s 下载失败：%s", dataset_id, exc.detail or exc.message_key)
            if on_progress is not None:
                on_progress(
                    "error",
                    0.0,
                    {"message_key": exc.message_key, "args": exc.error_args, "detail": exc.detail},
                )
        finally:
            _release(dataset_id)

    threading.Thread(target=worker, name=f"dataset-{dataset_id}", daemon=True).start()
    return True
