import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel, DeltaToolCall
from pydantic_ai.usage import UsageLimits

from redlotus.agent_core.runner import AgentRunner
from redlotus.ModelGateway import ModelChecker as checker
from redlotus.runtime.context import WorkspaceContext, workspace_context
from redlotus.tools.conversation_log import (
    ConversationLog,
    read_saved_model_messages_file,
)
from redlotus.tools.memory.chat_history import messages_safe_for_new_prompt
from redlotus.runtime.runtime_state import AgentRunPolicy
from redlotus.runtime.tool_telemetry import _model_result, tool_result_succeeded


async def run_test_agent(model, *, tools=(), **kwargs):
    kwargs.update(prompt="test", message_history=[], usage_limits=UsageLimits())
    return await AgentRunner().run(
        agent=Agent(FunctionModel(stream_function=model), tools=tools), **kwargs
    )


async def test_large_tool_batch_compacts_before_request_and_retains_original(
    tmp_path, monkeypatch
):
    async def limit(**kwargs):
        return 700

    monkeypatch.setattr(checker, "get_effective_max_context_async", limit)
    monkeypatch.setattr(checker, "get_effective_max_context", lambda **kwargs: 700)
    monkeypatch.setattr(
        checker,
        "get_context_config",
        lambda role: dict(auto_compress_ratio=0.7, head_turns=2, tail_turns=2),
    )
    monkeypatch.setattr(
        checker, "_save_compress_debug_artifacts", lambda **kwargs: None
    )
    monkeypatch.setattr(
        checker,
        "_call_compressor_llm",
        lambda **kwargs: "\n\n".join(
            h + "\n保留当前目标与失败证据" for h in checker._COMPRESS_REQUIRED_HEADINGS
        ),
    )
    requests = []

    async def work() -> str:
        return "原始工具结果" * 600

    async def model(messages, info):
        requests.append(list(messages))
        if len(requests) == 1:
            yield {0: DeltaToolCall(name="work", json_args="{}", tool_call_id="a")}
        else:
            yield "done"

    with workspace_context(WorkspaceContext.from_path(tmp_path)):
        log = ConversationLog("coordinator", "today", "compact")

        async def save(run):
            await log.save(run.all_messages(), extra={"turn_id": "one"})

        result = await run_test_agent(
            model,
            tools=[work],
            on_node=save,
            before_request=lambda run, node: checker.prepare_model_request(
                run, node, role="coordinator"
            ),
        )
        assert result.output == "done"
        assert checker.estimate_context_tokens(requests[1]) < 700
        assert messages_safe_for_new_prompt(requests[1]) == requests[1]
        journal = next((tmp_path / ".redlotus").glob("*.jsonl")).read_text(
            encoding="utf-8"
        )
        assert "原始工具结果" * 600 in journal
        messages, _ = read_saved_model_messages_file(log.model_messages_path())
        assert messages_safe_for_new_prompt(messages) == messages


async def test_cancel_closes_outstanding_call_and_saves_partial_results():
    started = asyncio.Event()
    captured = []

    async def work() -> str:
        started.set()
        await asyncio.sleep(60)
        return "should not return"

    async def model(messages, info):
        yield {0: DeltaToolCall(name="work", json_args="{}", tool_call_id="cancel-me")}

    async def save(run):
        captured[:] = run.all_messages()

    task = asyncio.create_task(run_test_agent(model, tools=[work], on_node=save))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert messages_safe_for_new_prompt(captured) == captured
    returns = [
        part
        for message in captured
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(returns) == 1 and returns[0].tool_call_id == "cancel-me"
    assert returns[0].content["status"] == "cancelled"


def test_full_tool_output_is_retained_and_failure_survives_preview(tmp_path):
    with workspace_context(WorkspaceContext.from_path(tmp_path)):
        original = "x" * 1000 + "\nExit code: 1"
        preview = _model_result(original, AgentRunPolicy(3, 100, 60))
        assert not tool_result_succeeded(preview)
        path = next((tmp_path / ".redlotus" / "tool_results").glob("*.txt"))
        assert path.read_text(encoding="utf-8") == original


def test_auxiliary_model_uses_shared_fallback_without_changing_model(monkeypatch):
    looked_up = []
    monkeypatch.setattr(
        checker,
        "get_context_config",
        lambda role: (
            {"default_context_tokens": 393216} if role == "coordinator" else {}
        ),
    )
    monkeypatch.setattr(
        checker, "get_model_and_params", lambda role: ("configured-compressor", {})
    )
    monkeypatch.setattr(
        checker, "lookup_model_context", lambda name: looked_up.append(name)
    )
    assert checker.get_effective_max_context(role="compressor") == 393216
    assert looked_up == ["configured-compressor"]
