"""跨模块隔离：App 启动（lifespan）会把「当前数据目录」里的上传清单注入进程级注册表。

各训练类测试模块的目录补丁只覆盖 DATA_DIR / RUNS_DIR / DB_PATH，DATASETS_DIR 仍指向仓库的真实
data/datasets（test_runs 需要真实 MNIST 缓存，不能整体改指），于是别的模块启动 App 时会把开发机上
的真实上传集带进 registry.SPECS / registry.UPLOADED，残留到后续用例（表现为 test_upload 的清单
重载断言多出条目）。这里在每个测试模块前后取还快照，让模块之间互不污染。
"""

from __future__ import annotations

import pytest

from app.datasets import registry


@pytest.fixture(autouse=True, scope="module")
def isolate_registry():
    specs = dict(registry.SPECS)
    uploaded = set(registry.UPLOADED)
    yield
    registry.SPECS.clear()
    registry.SPECS.update(specs)
    registry.UPLOADED.clear()
    registry.UPLOADED.update(uploaded)
