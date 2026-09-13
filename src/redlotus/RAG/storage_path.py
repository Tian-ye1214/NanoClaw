from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

from redlotus.infra import logger
from redlotus.config.app_config import get_env
from redlotus.infra.paths import APP_NAME, user_data_dir

_UNSUPPORTED_FS = frozenset({"EXFAT", "FAT32", "FAT"})


def volume_filesystem(path: Path) -> str | None:
    """返回路径所在卷的 filesystem 名称（如 NTFS、exFAT）；无法检测时返回 None。"""
    resolved = path.resolve()
    if sys.platform == "win32":
        root = os.path.splitdrive(str(resolved))[0] + "\\"
        if len(root) < 3:
            return None
        import ctypes

        buf = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            None,
            0,
            None,
            None,
            None,
            buf,
            len(buf),
        )
        return buf.value if ok else None
    return None


def _is_unsupported_fs(path: Path) -> bool:
    fs = volume_filesystem(path)
    return bool(fs and fs.upper() in _UNSUPPORTED_FS)


def _fit_windows_path(path: Path, table_name: str) -> str:
    # Lance adds table, data/index directories and generated filenames. Its Windows
    # writer currently drops the extended-path prefix before committing a file.
    if (
        not table_name
        or sys.platform != "win32"
        or len(str(path)) + len(table_name) + 110 < 248
    ):
        return str(path)
    digest = hashlib.sha256(str(path).casefold().encode()).hexdigest()[:16]
    local_root = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    fallback = local_root / APP_NAME / "rag_lancedb" / digest
    logger.info(
        "LanceDB: 路径过深，向量索引改存 %s；原始记忆与配置保持原位。", fallback
    )
    return str(fallback)


def resolve_lancedb_dir(configured_path: str, *, table_name: str = "") -> str:
    """将配置的 LanceDB 目录解析为绝对路径；exFAT/FAT 上自动改到 NTFS（RAG_DB_PATH 或 LOCALAPPDATA）。"""
    p = Path(configured_path)
    if not p.is_absolute():
        p = (user_data_dir() / p).resolve()
    else:
        p = p.resolve()

    if not _is_unsupported_fs(p):
        return _fit_windows_path(p, table_name)

    override = get_env("RAG_DB_PATH", warn=False)
    if override:
        op = Path(override)
        if not op.is_absolute():
            op = (user_data_dir() / op).resolve()
        else:
            op = op.resolve()
        if not _is_unsupported_fs(op):
            logger.info(
                "LanceDB: 配置路径 %s 位于 %s，不支持 Lance 事务；改用 RAG_DB_PATH=%s",
                p,
                volume_filesystem(p) or "?",
                op,
            )
            return _fit_windows_path(op, table_name)

    local_root = os.environ.get("LOCALAPPDATA") or str(Path.home())
    fallback = Path(local_root) / APP_NAME / "rag_lancedb" / p.name
    logger.info(
        "LanceDB: 配置路径 %s 位于 %s（不支持硬链接）；已改用 %s。"
        "可在 config 的 short_term_memory.db_path 或环境变量 RAG_DB_PATH 中指定 NTFS 目录。",
        p,
        volume_filesystem(p) or "?",
        fallback,
    )
    return _fit_windows_path(fallback.resolve(), table_name)
