from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from redlotus.infra import logger

from redlotus.ModelGateway.agent_factory import create_agent
from redlotus.config.app_config import get_agent_usage_limits, get_model_and_params
from redlotus.infra.shared_http import close_all_clients


async def complete_text(
    role: str, system_prompt: str, user_text: str, **parameters
) -> str:
    """Auxiliary calls use the foreground Agent's provider routing."""
    name, params = get_model_and_params(role)
    params.update(parameters)
    agent = create_agent(name, params, instructions=system_prompt)
    result = await agent.run(user_text, usage_limits=get_agent_usage_limits())
    logger.info_file_only(
        "[model_usage] %s",
        json.dumps(
            {"role": role, "model": name, **asdict(result.usage)}, ensure_ascii=False
        ),
    )
    return str(result.output or "")


def complete_text_sync(
    role: str, system_prompt: str, user_text: str, **parameters
) -> str:
    """Compression workers own and close their event-loop resources."""

    async def run() -> str:
        try:
            return await complete_text(role, system_prompt, user_text, **parameters)
        finally:
            await close_all_clients()

    return asyncio.run(run())
