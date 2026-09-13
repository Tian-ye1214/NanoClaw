import asyncio
import json
import threading

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, DeltaToolCall

from redlotus.agent_core.input_messages import UserMessage
from redlotus.agent_core.system import AgentSystem
from redlotus.runtime.context import WorkspaceContext
from redlotus.tools.memory import ChatHistory
from redlotus.tools.memory.ltm import LongTermMemory
from redlotus.tools.ManagementTools import TaskManager, TaskStatus
from redlotus.runtime.worker_result import SubagentResult


async def noop(*args, **kwargs):
    pass


def configured_system(tmp_path, monkeypatch):
    system = AgentSystem(workspace=WorkspaceContext.from_path(tmp_path))
    system._memory.long_term = LongTermMemory(tmp_path / "global")
    system._context_prewarmed = True
    monkeypatch.setattr(system, "_sync_skills_for_user_turn", noop)
    monkeypatch.setattr(system._memory, "process_pending", noop)
    monkeypatch.setattr("redlotus.agent_core.system.prepare_model_request", noop)
    return system


async def test_system_serializes_turns_refreshes_memory_and_records_raw(
    tmp_path, monkeypatch
):
    system = configured_system(tmp_path, monkeypatch)
    inputs, injections = [], []
    active = 0

    async def create(skills, memory, routing, tools):
        injections.append(memory)

        async def model(messages, info):
            nonlocal active
            active += 1
            assert active == 1
            inputs.append(messages[-1].parts[0].content)
            await asyncio.sleep(0.02)
            if len(inputs) == 1:
                system._memory.long_term.path.write_text(
                    system._memory.long_term.read() + "\n偏好中文", encoding="utf-8"
                )
            yield "done"
            active -= 1

        return Agent(FunctionModel(stream_function=model))

    monkeypatch.setattr("redlotus.agent_core.system.create_coordinator_agent", create)
    history = ChatHistory()
    await asyncio.gather(
        *(
            system.run_agent_system(UserMessage(text=text), history)
            for text in ("第一条", "第二条", "第三条")
        )
    )
    assert inputs == ["第一条", "第二条", "第三条"]
    assert "偏好中文" not in injections[0] and "偏好中文" in injections[1]
    assert len(system._memory.observations.order()) == 3
    journals = list((tmp_path / ".redlotus").glob("*.jsonl"))
    rows = [
        json.loads(line)
        for line in journals[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [
        p["content"]
        for row in rows
        for p in row["message"]["parts"]
        if p["part_kind"] == "user-prompt"
    ] == inputs
    await system.shutdown()


async def test_subagent_creation_and_model_run_are_inside_child_thread(
    tmp_path, monkeypatch
):
    system = configured_system(tmp_path, monkeypatch)
    await system.bind_session("session")
    system._session_logs.ensure("child")
    parent_thread = threading.get_ident()
    threads = []

    async def model(messages, info):
        threads.append(threading.get_ident())
        tool = info.output_tools[0]
        yield {
            0: DeltaToolCall(
                name=tool.name,
                json_args='{"status":"success","summary":"verified test"}',
                tool_call_id="out",
            )
        }

    def create(name, params, **kwargs):
        assert threading.get_ident() != parent_thread
        return Agent(
            FunctionModel(stream_function=model), output_type=kwargs["output_type"]
        )

    import importlib

    monkeypatch.setattr(
        importlib.import_module("redlotus.tools.WorkerOrchestrator"),
        "create_agent",
        create,
    )
    monkeypatch.setattr(
        "redlotus.ModelGateway.ModelChecker.prepare_model_request", noop
    )
    success, output = await system._orchestrator.execute_task_with_worker(
        "test", turn_id="turn"
    )
    assert success and json.loads(output)["status"] == "success"
    assert all(t != parent_thread for t in threads)
    assert not system._orchestrator.factory.handles
    await system.shutdown()


async def test_cli_fifo_does_not_merge_inputs(tmp_path, monkeypatch):
    system = configured_system(tmp_path, monkeypatch)
    cli = system._cli_controller
    state = cli.new_session_state()
    calls = []

    async def start(text, state, **kwargs):
        calls.append(text)

    monkeypatch.setattr(cli, "_start_user_turn_from_raw_input", start)
    monkeypatch.setattr(
        "redlotus.agent_core.cli_controller.app_config.missing_main_api_keys",
        lambda: (),
    )
    await cli.process_line("first", state, wait_for_turn=False)
    await cli.process_line("second", state, wait_for_turn=False)
    await system._session.queue.join()
    assert calls == ["first", "second"]
    await system.shutdown()


def test_todo_updates_preserve_results_and_reject_invalid_batch():
    manager = TaskManager()
    manager.create_todo_list('[{"id":"a","description":"first"}]')
    manager.mark_task_complete("a", "verified result")
    manager.create_todo_list(
        '[{"id":"a","description":"first"},{"id":"b","description":"second","dependencies":["a"]}]'
    )
    assert manager.tasks["a"].status == TaskStatus.COMPLETED
    assert manager.tasks["a"].result == "verified result"
    assert [t.id for t in manager.get_all_ready_tasks()] == ["b"]
    assert manager.create_todo_list(
        '[{"id":"c","description":"third","dependencies":["missing"]}]'
    ).startswith("Error:")
    assert set(manager.tasks) == {"a", "b"}


def test_worker_output_fails_closed():
    from pydantic import ValidationError
    import pytest

    for value in (
        "",
        "I could not complete the task",
        '{"status":"cancelled"}',
        '{"status":"success","summary":"   "}',
    ):
        with pytest.raises(ValidationError):
            SubagentResult.model_validate_json(value)
    assert not SubagentResult(status="cancelled", summary="cancelled by user").success
    assert SubagentResult(status="success", summary="verified").success


async def test_loading_session_restores_completed_tasks_and_dependencies(
    tmp_path, monkeypatch
):
    from redlotus.tools.conversation_log import read_saved_model_messages_file

    system = configured_system(tmp_path, monkeypatch)
    system._task_manager.create_todo_list(
        '[{"id":"a","description":"done"},{"id":"b","description":"next","dependencies":["a"]}]'
    )
    system._task_manager.mark_task_complete("a", "verified artifact")
    system._task_manager.tasks["b"].status = TaskStatus.IN_PROGRESS

    async def create(*args):
        async def model(messages, info):
            yield "saved"

        return Agent(FunctionModel(stream_function=model))

    monkeypatch.setattr("redlotus.agent_core.system.create_coordinator_agent", create)
    await system.run_agent_system(UserMessage(text="保存当前状态"), ChatHistory())
    path = system._session_logs.for_agent("coordinator").model_messages_path()
    messages, meta = read_saved_model_messages_file(path)
    system._task_manager.reset()
    system.bind_loaded_snapshot("coordinator", path, meta)
    assert system._task_manager.tasks["a"].result == "verified artifact"
    assert system._task_manager.tasks["a"].status == TaskStatus.COMPLETED
    assert [task.id for task in system._task_manager.get_all_ready_tasks()] == ["b"]
    assert messages
    await system.shutdown()
