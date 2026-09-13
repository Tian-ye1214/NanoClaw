from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from redlotus.ModelGateway.agent_factory import create_agent, create_function_toolset
from redlotus.config.app_config import get_model_and_params
from redlotus.prompt import get_coordinator_system_prompt
from redlotus.skills.SkillsManager import SkillsManager


async def create_coordinator_agent(
    skills_manager: SkillsManager,
    memory_injection: str,
    routing_tools: Sequence[Any],
    worker_tools: Sequence[Any],
):
    model, params = get_model_and_params("coordinator")
    instructions = await asyncio.to_thread(
        get_coordinator_system_prompt, skills_manager, memory_injection
    )
    toolsets = [
        create_function_toolset(list(tools), toolset_id=name)
        for name, tools in (("delegation", routing_tools), ("execution", worker_tools))
        if tools
    ]
    return create_agent(model, params, instructions=instructions, toolsets=toolsets)
