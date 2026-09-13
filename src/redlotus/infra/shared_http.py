from __future__ import annotations

from collections.abc import Callable
import asyncio
import threading
import httpx
from urllib.parse import urlsplit

from redlotus.infra import logger

_local = threading.local()


def openai_base_url(value: str | None) -> str | None:
    """Append /v1 for a bare host; preserve provider-specific API prefixes."""
    if not value:
        return None
    base = value.rstrip("/")
    return base if urlsplit(base).path else base + "/v1"


def _clients() -> dict[str, httpx.AsyncClient]:
    """A connection pool belongs to one event loop in one thread."""
    loop = asyncio.get_running_loop()
    pools = getattr(_local, "pools", None)
    if pools is None:
        pools = _local.pools = {}
    return pools.setdefault(loop, {})


def get_client(
    key: str,
    factory: Callable[[], httpx.AsyncClient],
) -> httpx.AsyncClient:
    """Return a named AsyncClient owned by the current event loop."""
    clients = _clients()
    client = clients.get(key)
    if client is None or client.is_closed:
        client = factory()
        clients[key] = client
    return client


async def close_client(key: str) -> None:
    client = _clients().pop(key, None)
    if client is not None and not client.is_closed:
        try:
            await client.aclose()
        except Exception as e:
            logger.debug("关闭 HTTP 客户端 %r 时忽略异常: %s", key, e)


async def close_all_clients() -> None:
    """Close this loop's clients without touching other Agent threads."""
    keys = list(_clients())
    for key in keys:
        await close_client(key)
    _local.pools.pop(asyncio.get_running_loop(), None)
