"""病例加载缓存。

病例加载的瓶颈是 skeletonize(lee) 与 distance_transform_edt（在 147×512×512
体积上约 20 秒），而这两步的结果只依赖掩膜文件内容。

缓存策略：
  - 指纹 = 参与计算的 nii 文件的 (名称, 大小, mtime_ns) 摘要
  - 命中则反序列化派生状态，跳过骨架化与建图，实测 22.8s -> 0.4s
  - 掩膜是稀疏二值体，整型标签降位宽后 zlib 压缩，实测 270MB -> 1.3MB
  - 任何相关文件变化都会让指纹失效，不存在脏读

注意：ct_image（SimpleITK 对象）不入缓存，加载时按需重新读取（约 1s）。
"""
from __future__ import annotations

import hashlib
import pickle
import zlib
from pathlib import Path
from typing import Any

CACHE_SCHEMA = 4
COMPRESS_LEVEL = 1  # 压缩 0.3s / 解压 0.25s，远快于重算 20s

_RELEVANT_SUFFIX = ".nii.gz"
_EXT = ".pklz"

# 可以安全降位宽的整型数组
_SHRINKABLE = ("int32", "int64")


def cache_root() -> Path:
    from .navbridge import NAV_ROOT

    return NAV_ROOT / ".cache" / "agent"


def fingerprint(case_dir: Path) -> str:
    """对病例目录内所有 nii 文件生成指纹。"""
    digest = hashlib.sha256()
    digest.update(f"schema={CACHE_SCHEMA}".encode())
    for path in sorted(Path(case_dir).glob(f"*{_RELEVANT_SUFFIX}")):
        stat = path.stat()
        digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:32]


def cache_path(case_dir: Path, extra: str = "") -> Path:
    case_dir = Path(case_dir).resolve()
    slug = case_dir.parent.name if case_dir.name == "stage4_package" else case_dir.name
    return cache_root() / f"{slug}__{fingerprint(case_dir)}{extra}{_EXT}"


def _shrink(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """把大整型数组降到能装下的最小位宽，记录原 dtype 以便还原。"""
    import numpy as np

    out = dict(state)
    dtypes: dict[str, str] = {}
    for key, value in state.items():
        if not isinstance(value, np.ndarray) or value.dtype.name not in _SHRINKABLE:
            continue
        if value.size == 0:
            continue
        peak = int(value.max())
        if 0 <= peak <= 127:
            out[key] = value.astype(np.int8)
            dtypes[key] = value.dtype.name
        elif 0 <= peak <= 32767:
            out[key] = value.astype(np.int16)
            dtypes[key] = value.dtype.name
    return out, dtypes


def _restore(state: dict[str, Any], dtypes: dict[str, str]) -> dict[str, Any]:
    import numpy as np

    for key, name in (dtypes or {}).items():
        value = state.get(key)
        if isinstance(value, np.ndarray):
            state[key] = value.astype(np.dtype(name))
    return state


def load(case_dir: Path, extra: str = "") -> dict[str, Any] | None:
    path = cache_path(case_dir, extra)
    if not path.is_file():
        return None
    try:
        raw = zlib.decompress(path.read_bytes())
        payload = pickle.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
        return None
    state = payload.get("state")
    if not isinstance(state, dict):
        return None
    return _restore(state, payload.get("dtypes", {}))


def save(case_dir: Path, state: dict[str, Any], extra: str = "") -> Path | None:
    path = cache_path(case_dir, extra)
    try:
        shrunk, dtypes = _shrink(state)
        payload = {"schema": CACHE_SCHEMA, "state": shrunk, "dtypes": dtypes}
        blob = zlib.compress(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), COMPRESS_LEVEL
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(path)
        return path
    except Exception:
        return None


def clear() -> int:
    """清空缓存，返回删除的文件数。"""
    root = cache_root()
    if not root.is_dir():
        return 0
    removed = 0
    for path in list(root.glob(f"*{_EXT}")) + list(root.glob("*.pkl")):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def info() -> dict[str, Any]:
    root = cache_root()
    if not root.is_dir():
        return {"dir": str(root), "entries": 0, "mb": 0.0, "items": []}
    files = list(root.glob(f"*{_EXT}"))
    items = [
        {"name": p.name, "mb": round(p.stat().st_size / 1024 / 1024, 3)}
        for p in sorted(files, key=lambda p: -p.stat().st_size)
    ]
    return {
        "dir": str(root),
        "entries": len(files),
        "mb": round(sum(p.stat().st_size for p in files) / 1024 / 1024, 3),
        "items": items,
    }
