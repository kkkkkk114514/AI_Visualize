from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
from functools import lru_cache

from fastapi import APIRouter

from app import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


def _total_memory_gb() -> float | None:
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


@lru_cache(maxsize=1)
def device_info() -> dict:
    info: dict = {
        "device": "cpu",
        "device_name": platform.processor() or platform.machine() or "CPU",
        "cpu_count": os.cpu_count(),
        "memory_gb": _total_memory_gb(),
        "torch": None,
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        log.warning("未安装 PyTorch，训练能力不可用")
        return info

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info.update(
            device="cuda",
            device_name=torch.cuda.get_device_name(0),
            cuda_available=True,
            cuda_version=torch.version.cuda,
        )
    return info


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
        "python": sys.version.split()[0],
        **device_info(),
    }
