"""审计日志：追加式 JSONL + 哈希链（见 `store.py` 的模块文档）。"""
from .store import (
    append,
    audit_dir,
    hash_file,
    iter_records,
    list_logs,
    log_path,
    query,
    record_auth,
    record_phi_scan,
    record_run,
    verify_chain,
)

__all__ = [
    "append",
    "audit_dir",
    "hash_file",
    "iter_records",
    "list_logs",
    "log_path",
    "query",
    "record_auth",
    "record_phi_scan",
    "record_run",
    "verify_chain",
]
