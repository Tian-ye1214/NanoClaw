from __future__ import annotations

import difflib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _opcodes(baseline: str, current: str):
    a = baseline.splitlines(keepends=True)
    b = current.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return a, b, sm.get_opcodes()


@dataclass(frozen=True)
class Hunk:
    """一处连续改动（baseline→current 中的一个非 equal 区块）。"""

    index: int
    old_start: int
    old_lines: list[str]
    new_start: int
    new_lines: list[str]

    @property
    def location(self) -> str:
        return f"L{self.new_start}" if self.new_lines else f"L{self.old_start}"


def compute_hunks(baseline: str, current: str) -> list[Hunk]:
    """把 baseline→current 的差异切成逐块 Hunk 列表（equal 区块跳过）。"""
    a, b, ops = _opcodes(baseline, current)
    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        hunks.append(Hunk(len(hunks), i1 + 1, a[i1:i2], j1 + 1, b[j1:j2]))
    return hunks


def reconstruct(baseline: str, current: str, rejected: set[int]) -> str:
    """按逐块决定重建文件内容：rejected 的块取 baseline 侧，其余取 current 侧。

    rejected 为空 → 完全等于 current；rejected 含全部块 → 完全等于 baseline。
    """
    a, b, ops = _opcodes(baseline, current)
    out: list[str] = []
    idx = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out.extend(b[j1:j2])
            continue
        out.extend(a[i1:i2] if idx in rejected else b[j1:j2])
        idx += 1
    return "".join(out)


@dataclass
class ReviewEntry:
    path: Path
    name: str
    baseline: str
    snapshot: str
    decisions: dict[int, bool] = field(default_factory=dict)
    existed: bool = True

    @property
    def hunks(self):
        return compute_hunks(self.baseline, self.snapshot)


class PendingReviewStore:
    """跨线程共享的待审查暂存区。复用 toolkit 的 file_lock，避免与 agent 写盘竞争。"""

    def __init__(self, file_lock: threading.Lock) -> None:
        self._lock = file_lock
        self._entries: dict[str, ReviewEntry] = {}
        self._on_change: Callable[[], None] | None = None

    def activate(self, on_change: Callable[[], None]) -> None:
        with self._lock:
            self._on_change = on_change

    def deactivate(self) -> None:
        with self._lock:
            self._on_change = None
            self._entries.clear()

    def clear(self) -> None:
        """Discard the previous project's reviews while retaining the UI subscription."""
        with self._lock:
            self._entries.clear()
            cb = self._on_change
        self._notify(cb)

    def _register_locked(self, path, name, baseline, snapshot):
        if baseline == snapshot or self._on_change is None:
            return
        old = self._entries.get(str(path))
        self._entries[str(path)] = ReviewEntry(
            path,
            name,
            old.baseline if old else baseline or "",
            snapshot,
            existed=old.existed if old else baseline is not None,
        )

    def register(self, path: Path, *, name: str, baseline: str, snapshot: str) -> None:
        with self._lock:
            self._register_locked(path, name, baseline, snapshot)
            callback = self._on_change
        self._notify(callback)

    def write(self, path: Path, name: str, update):
        """Publish file contents and their review snapshot as one locked operation."""
        with self._lock:
            previous = path.read_text(encoding="utf-8") if path.exists() else None
            content = update(previous)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._register_locked(path, name, previous, content)
            callback = self._on_change
        self._notify(callback)
        return previous or "", content

    def entries(self) -> list[ReviewEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, key: str) -> ReviewEntry | None:
        with self._lock:
            return self._entries.get(key)

    def decide(self, entry: ReviewEntry, index: int, reject: bool) -> bool:
        """Apply a decision only to the exact version displayed by the UI."""
        with self._lock:
            if self._entries.get(str(entry.path)) is not entry:
                return False
            previous_rejections = {
                key for key, value in entry.decisions.items() if value
            }
            expected = reconstruct(entry.baseline, entry.snapshot, previous_rejections)
            current = (
                entry.path.read_text(encoding="utf-8") if entry.path.exists() else ""
            )
            if current != expected:
                raise ValueError(
                    "文件已在审查界面之外被修改；为保留这些改动，本次决定未应用。"
                )
            decisions = {**entry.decisions, index: reject}
            rejected = {key for key, value in decisions.items() if value}
            if not entry.existed and len(rejected) == len(entry.hunks):
                entry.path.unlink(missing_ok=True)
            else:
                entry.path.write_text(
                    reconstruct(entry.baseline, entry.snapshot, rejected),
                    encoding="utf-8",
                )
            entry.decisions = decisions
        return True

    def finish_decided(self):
        for entry in self.entries():
            if all(hunk.index in entry.decisions for hunk in entry.hunks):
                self.finish(str(entry.path))

    def finish(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
            cb = self._on_change
        self._notify(cb)

    def _notify(self, cb: Callable[[], None] | None) -> None:
        if cb is not None:
            cb()
