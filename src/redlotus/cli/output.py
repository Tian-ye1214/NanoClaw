"""Output routing for legacy console and Textual UI modes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from typing import Any, Protocol

from rich.console import Console
from rich.text import Text


@dataclass(frozen=True)
class ContextUsageItem:
    role_label: str
    used_tokens: int
    max_tokens: int
    percent: float


class OutputSink(Protocol):
    supports_model_stream: bool

    def emit(self, renderable: Any) -> None: ...
    def update(self, action: str, *args) -> None: ...


class LegacyOutputSink:
    supports_model_stream = False

    def __init__(self, console: Console) -> None:
        self.console = console

    def emit(self, renderable: Any) -> None:
        if isinstance(renderable, str) and "\x1b[" in renderable:
            renderable = Text.from_ansi(renderable)
        self.console.print(renderable)

    def update(self, action: str, *args) -> None:
        if action == "rule":
            self.console.rule(*args)
        elif action == "set_status":
            self.console.print(Text(args[0], style="dim"))


_console = Console(highlight=False, legacy_windows=sys.platform == "win32")
_sink: OutputSink = LegacyOutputSink(_console)


def set_output_sink(sink: OutputSink | None) -> None:
    global _sink
    _sink = sink if sink is not None else LegacyOutputSink(_console)


def supports_model_stream() -> bool:
    return _sink.supports_model_stream


def emit_renderable(renderable: Any) -> None:
    _sink.emit(renderable)


def emit_rule(title: str) -> None:
    _sink.update("rule", title)


def set_status(message: str) -> None:
    _sink.update("set_status", message)


def clear_status() -> None:
    _sink.update("clear_status")


def set_context_usage(items: list[ContextUsageItem]) -> None:
    _sink.update("set_context_usage", items)


def clear_context_usage() -> None:
    _sink.update("clear_context_usage")


def begin_model_stream(title: str) -> None:
    _sink.update("begin_model_stream", title)


def append_model_stream_delta(text: str) -> None:
    _sink.update("append_model_stream_delta", text)


def clear_model_stream() -> None:
    _sink.update("clear_model_stream")


@contextmanager
def status_message(message: str):
    set_status(message)
    try:
        yield
    finally:
        clear_status()
