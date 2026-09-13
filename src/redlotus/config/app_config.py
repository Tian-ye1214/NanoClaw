from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redlotus.infra.persist_utils import save_locked_json
from redlotus.workspace.workspace import current_workspace

if TYPE_CHECKING:
    from pydantic_ai.usage import UsageLimits as _UsageLimits

from redlotus.infra import logger
from dotenv import dotenv_values
from redlotus.infra.paths import config_file, default_config_file, dotenv_file
from pydantic_ai.usage import UsageLimits
from redlotus.runtime.runtime_state import AgentRunPolicy

CONFIG_FILE, DOTENV_FILE = config_file(), dotenv_file()

_CONFIG: dict[str, Any] | None = None
_DOTENV_CACHE: dict[tuple, dict[str, str]] | None = None
_API_CONFIG_KEYS = {"BASE_URL", "API_KEY", "SILICONFLOW_BASE", "SILICONFLOW_KEY"}
THINKING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def _seed_config_if_missing() -> None:
    if CONFIG_FILE.exists():
        return
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_config_file(), CONFIG_FILE)


def load_config() -> dict[str, Any]:
    global _CONFIG
    _seed_config_if_missing()
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def reload_config() -> dict[str, Any]:
    global _CONFIG, _DOTENV_CACHE
    _CONFIG = None
    _DOTENV_CACHE = None
    return load_config()


def settings() -> dict[str, Any]:
    if _CONFIG is None:
        load_config()
    return _CONFIG  # type: ignore[return-value]


def _dotenv_files() -> list[Path]:
    """.env 来源：用户配置目录优先，当前工作目录（项目本地）覆盖之。"""
    out: list[Path] = []
    for p in (DOTENV_FILE, current_workspace() / ".env"):
        if p not in out:
            out.append(p)
    return out


def _dotenv_values() -> dict[str, str]:
    """解析并缓存 .env（进程内静态）：键值均 strip，空值丢弃；cwd/.env 覆盖用户目录 .env。"""
    global _DOTENV_CACHE
    files = _dotenv_files()
    key = tuple(
        (str(path), path.stat().st_mtime_ns if path.is_file() else None)
        for path in files
    )
    if _DOTENV_CACHE is None:
        _DOTENV_CACHE = {}
    if key not in _DOTENV_CACHE:
        values = {}
        for path in files:
            if path.is_file():
                values.update(
                    {
                        str(k): str(v).strip()
                        for k, v in dotenv_values(path).items()
                        if v and str(v).strip()
                    }
                )
        _DOTENV_CACHE[key] = values
    return _DOTENV_CACHE[key]


def _config_scalar(key: str) -> str:
    raw = settings().get(key)
    if raw is not None and not isinstance(raw, (dict, list)):
        return raw.strip() if isinstance(raw, str) else str(raw).strip()
    return ""


def get_env(key: str, *, warn: bool = True, default: str = "") -> str:
    """配置读取唯一入口；/api 管理的 key 让 config.json 优先于 .env。"""
    if env_val := (os.environ.get(key) or "").strip():
        return env_val
    configured, dotenv = _config_scalar(key), _dotenv_values().get(key, "")
    value = (
        (configured or dotenv) if key in _API_CONFIG_KEYS else (dotenv or configured)
    )
    if not value and warn and not default:
        logger.warning("未配置 %r，请在 .env 或 config.json 根中填写。", key)
    return value or default


def _missing_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(key for key in keys if not get_env(key, warn=False).strip())


def missing_main_api_keys() -> tuple[str, ...]:
    return _missing_keys(("BASE_URL", "API_KEY"))


def missing_rag_api_keys() -> tuple[str, ...]:
    return _missing_keys(("SILICONFLOW_BASE", "SILICONFLOW_KEY"))


def save_config(cfg: dict[str, Any] | None = None) -> None:
    global _CONFIG
    cfg = settings() if cfg is None else cfg
    _CONFIG = cfg
    save_locked_json(CONFIG_FILE, cfg)


def get_agent_usage_limits() -> "_UsageLimits":
    """单次 Agent 运行对模型请求次数上限"""
    cfg = settings()
    raw = cfg["request_limit"]
    if raw is None or str(raw).strip().lower() in ("none", "unlimited", "null", ""):
        return UsageLimits(request_limit=None)
    return UsageLimits(request_limit=int(raw))


def get_agent_run_policy() -> AgentRunPolicy:
    return AgentRunPolicy.from_config(settings())


def supported_thinking_efforts(model_name: str | None) -> tuple[str, ...]:
    from redlotus.ModelGateway.ModelChecker import _lookup_openrouter_meta

    meta = _lookup_openrouter_meta(model_name) if model_name else None
    available = (meta or {}).get("supported_efforts") or THINKING_EFFORTS
    return tuple(value for value in THINKING_EFFORTS if value in available)


def role_supported_thinking_efforts(role: str) -> tuple[str, ...]:
    return supported_thinking_efforts(settings()["models"][role]["name"])


def apply_thinking_config(model_params, *, model_name=None):
    """Translate config thinking fields to Pydantic AI's common model settings."""
    params = deepcopy(model_params)
    thinking = str(params.pop("thinking", "")).strip().lower()
    effort = str(params.pop("reasoning_effort", "")).strip().lower()
    if thinking in ("disabled", "off", "false"):
        params["extra_body"] = {
            **params.get("extra_body", {}),
            "thinking": {"type": "disabled"},
        }
    elif thinking == "enabled":
        supported = supported_thinking_efforts(model_name)
        if supported:
            params["thinking"] = effort if effort in supported else supported[-1]
    return params


def get_model_and_params(role: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    raw: dict[str, Any] = deepcopy(settings()["models"][role])
    name = str(raw.pop("name")).strip()
    raw.update(deepcopy(kwargs))
    return name, raw


def set_model_name(role: str, model_name: str) -> None:
    cfg = settings()
    cfg["models"][role]["name"] = model_name.strip()
    save_config(cfg)


def set_api(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
) -> None:
    values = {
        "BASE_URL": base_url,
        "API_KEY": api_key,
        "SILICONFLOW_BASE": embedding_url,
        "SILICONFLOW_KEY": embedding_key,
    }
    if all(v is None for v in values.values()):
        return
    cfg = settings()
    for key, value in values.items():
        if value is not None:
            cfg[key] = value.strip()
    save_config(cfg)


def get_agent_roles(*, cfg=None) -> tuple[str, ...]:
    return tuple((settings() if cfg is None else cfg)["models"])


def get_context_profile_roles() -> tuple[str, ...]:
    return tuple(role for role in get_agent_roles() if role in settings()["context"])


def get_context_config(role: str) -> dict[str, Any]:
    raw = settings()["context"]
    if role not in get_agent_roles():
        raise ValueError(f"Unknown Agent role: {role}")
    if not any(key in raw for key in get_agent_roles()):
        return dict(raw)  # Legacy shared context configuration.
    return {**raw.get("defaults", {}), **raw.get(role, {})}
