from __future__ import annotations

import sys
import os
from pathlib import Path

import platformdirs

APP_NAME = "RedLotus"


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """随包只读资源根（redlotus 包目录）。

    - 正常 / editable 安装：本文件位于 redlotus/infra/paths.py，上溯两级即包根。
    - PyInstaller 冻结：优先 _MEIPASS/redlotus，回退 exe 目录。
    """
    if _frozen():
        base = getattr(sys, "_MEIPASS", None)
        root = Path(base) if base else Path(sys.executable).resolve().parent
        pkg = root / "redlotus"
        return pkg if pkg.is_dir() else root
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """全局可写状态根：日志 / 向量库 / 长期记忆 / 运行时技能 overlay。"""
    return Path(
        os.environ.get("REDLOTUS_DATA_DIR")
        or platformdirs.user_data_dir(APP_NAME, appauthor=False)
    )


def user_config_dir() -> Path:
    """用户配置根：config.json / .env / bot config.yaml。"""
    return Path(
        os.environ.get("REDLOTUS_CONFIG_DIR")
        or platformdirs.user_config_dir(APP_NAME, appauthor=False)
    )


# ---- 工作产物：跟随当前工作目录 ----
# ---- 配置 / 密钥：用户配置目录 ----
def config_file() -> Path:
    if path := os.environ.get("REDLOTUS_CONFIG_FILE"):
        return Path(path).resolve()
    if directory := os.environ.get("REDLOTUS_CONFIG_DIR"):
        return Path(directory).resolve() / "config.json"
    return resource_root() / "config.json"


def default_config_file() -> Path:
    """随包默认 config.json（首次运行 seed 用）。"""
    return resource_root() / "config.json"


def dotenv_file() -> Path:
    return user_config_dir() / ".env"


# ---- 只读随包资源 ----
def prompts_dir() -> Path:
    return resource_root() / "prompts"


def skills_dir() -> Path:
    """随包基线技能（只读）。"""
    return resource_root() / "skills"


# ---- 全局可写状态 ----
def logs_dir() -> Path:
    return user_data_dir() / "logs"


def memory_dir() -> Path:
    """个人全局 MEMORY.md 及旧 SOUL/USER 文档的迁移备份目录。"""
    return user_data_dir() / "LongTermMemory"


def user_skills_dir() -> Path:
    """运行时安装的技能 overlay（可写）；与随包基线技能合并加载。"""
    return user_data_dir() / "skills"
