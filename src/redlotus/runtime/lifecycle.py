from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any

from redlotus.infra import logger
from redlotus.config.app_config import settings
from redlotus.runtime.runtime_state import TRACE_STORE, agent_context, turn_context

_invocation_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "lifecycle_invocation_stack", default=()
)


class AgentInstanceState(Enum):
    IDLE = "idle"
    RUNNING = "running"


class AgentInvocationState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def make_agent_id(session_key: str, role: str, suffix: str | None = None) -> str:
    if suffix:
        return f"{session_key}:{role}:{suffix}"
    return f"{session_key}:{role}"


def _invocation_history_limit() -> int:
    lc = settings().get("lifecycle")
    if not isinstance(lc, dict):
        raise KeyError("config.json 缺少 lifecycle 配置块")
    n = lc.get("invocation_history_per_session")
    if not isinstance(n, int) or n < 1:
        raise ValueError("lifecycle.invocation_history_per_session 须为正整数")
    return n


@dataclass
class AgentInstance:
    agent_id: str
    role: str
    session_key: str
    state: AgentInstanceState
    current_invocation_id: str | None = None


@dataclass
class AgentInvocation:
    invocation_id: str
    agent_id: str
    role: str
    session_key: str
    parent_invocation_id: str | None
    turn_id: str | None
    state: AgentInvocationState
    started_at: float
    finished_at: float | None
    task_ref: asyncio.Task[Any] | None


@dataclass
class SessionLifecycleView:
    session_key: str
    agents: list[AgentInstance]
    active_invocations: list[AgentInvocation]
    recent_invocations: list[AgentInvocation]


class AgentRegistry:
    """Lifecycle state owned by the application loop; child threads use their owner bridge."""

    def __init__(self):
        self._agents = {}
        self._invocations = {}
        self._history = {}

    @staticmethod
    def _prefix_matches(keys, prefix):
        return (
            [prefix]
            if prefix in keys
            else [key for key in keys if key.startswith(prefix.strip())]
        )

    async def ensure_agent(self, session_key, role, suffix=None):
        identity = make_agent_id(session_key, role, suffix)
        self._agents.setdefault(
            identity,
            AgentInstance(identity, role, session_key, AgentInstanceState.IDLE),
        )
        return identity

    async def list_agents(self, session_key=None):
        return [
            row
            for row in self._agents.values()
            if session_key is None or row.session_key == session_key
        ]

    async def list_active_invocations(self, session_key=None):
        return [
            row
            for row in self._invocations.values()
            if session_key is None or row.session_key == session_key
        ]

    async def list_recent_invocations(self, session_key):
        return list(self._history.get(session_key, ()))

    async def remove_session(self, session_key):
        self._agents = {
            key: row
            for key, row in self._agents.items()
            if row.session_key != session_key
        }
        self._history.pop(session_key, None)

    async def resolve_active_invocation_id(self, prefix):
        matches = self._prefix_matches(self._invocations, prefix)
        return matches[0] if len(matches) == 1 else None

    async def count_active_invocation_prefix_matches(self, prefix):
        return len(self._prefix_matches(self._invocations, prefix))

    async def find_recent_invocation_by_prefix(self, session_key, prefix):
        rows = {row.invocation_id: row for row in self._history.get(session_key, ())}
        matches = self._prefix_matches(rows, prefix)
        return rows[matches[0]] if len(matches) == 1 else None

    async def cancel(self, invocation_id):
        identity = await self.resolve_active_invocation_id(invocation_id)
        if identity is None:
            return False
        task = self._invocations[identity].task_ref
        if task and not task.done():
            task.cancel()
        return True

    async def _cancel_where(self, predicate):
        ids = [
            row.invocation_id for row in self._invocations.values() if predicate(row)
        ]
        for identity in ids:
            await self.cancel(identity)
        return len(ids)

    async def cancel_agent(self, agent_id):
        return await self._cancel_where(lambda row: row.agent_id == agent_id)

    async def cancel_turn(self, turn_id):
        return await self._cancel_where(lambda row: row.turn_id == turn_id)

    async def cancel_session(self, session_key):
        return await self._cancel_where(lambda row: row.session_key == session_key)

    async def cancel_all(self):
        return await self._cancel_where(lambda row: True)

    async def run(self, factory, *, agent_id, turn_id=None):
        agent = self._agents[agent_id]
        stack = _invocation_stack.get()
        invocation = AgentInvocation(
            uuid.uuid4().hex,
            agent_id,
            agent.role,
            agent.session_key,
            stack[-1] if stack else None,
            turn_id,
            AgentInvocationState.RUNNING,
            time.monotonic(),
            None,
            asyncio.current_task(),
        )
        self._invocations[invocation.invocation_id] = invocation
        agent.state, agent.current_invocation_id = (
            AgentInstanceState.RUNNING,
            invocation.invocation_id,
        )
        token = _invocation_stack.set((*stack, invocation.invocation_id))
        fields = dict(
            invocation_id=invocation.invocation_id,
            agent_id=agent_id,
            role=agent.role,
            parent=invocation.parent_invocation_id,
        )
        TRACE_STORE.record(turn_id, "invocation_start", **fields)
        error = ""
        try:
            with turn_context(turn_id), agent_context(agent_id):
                result = await factory()
            invocation.state = AgentInvocationState.COMPLETED
            return result
        except asyncio.CancelledError:
            invocation.state = AgentInvocationState.CANCELLED
            raise
        except BaseException as exc:
            invocation.state, error = AgentInvocationState.FAILED, str(exc)
            raise
        finally:
            invocation.finished_at = time.monotonic()
            invocation.task_ref = None
            _invocation_stack.reset(token)
            self._invocations.pop(invocation.invocation_id, None)
            self._history.setdefault(
                agent.session_key, deque(maxlen=_invocation_history_limit())
            ).append(invocation)
            if agent.current_invocation_id == invocation.invocation_id:
                agent.state, agent.current_invocation_id = AgentInstanceState.IDLE, None
            if agent.role == "worker":
                self._agents.pop(agent_id, None)
            TRACE_STORE.record(
                turn_id, "invocation_" + invocation.state.value, **fields, error=error
            )
            logger.debug(
                "[lifecycle] role=%s state=%s invocation=%s",
                agent.role,
                invocation.state.value,
                invocation.invocation_id,
            )
