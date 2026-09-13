from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path


from redlotus.infra.paths import memory_dir
from redlotus.infra.persist_utils import atomic_write_text, file_lock

SECTIONS = ("用户画像", "可复用经验")
EMPTY_MEMORY = "# MEMORY\n\n## 用户画像\n\n## 可复用经验\n"
CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:\b(?:api[_ -]?key|access[_ -]?token|password|secret|密码|密钥)\s*[:=：]\s*\S+"
    r"|\bsk-[A-Za-z0-9_-]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9_.-]{12,})"
)


class LongTermMemory:
    """Editable core profile; complete semantic records belong to LanceDB."""

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else memory_dir()
        self.path = self.directory / "MEMORY.md"

    def read(self):
        with file_lock(self.path):
            if not self.path.exists():
                sections = {}
                for name, heading in (("USER.md", "用户偏好"), ("SOUL.md", "经验")):
                    old = self.directory / name
                    if old.exists():
                        backup = self.directory / "migration_backup" / name
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        if not backup.exists():
                            shutil.copy2(old, backup)
                        sections[heading] = re.sub(
                            r"^#.*\n", "", old.read_text(encoding="utf-8"), count=1
                        ).strip()
                atomic_write_text(
                    self.path,
                    self._render("# MEMORY", sections)
                    if any(sections.values())
                    else EMPTY_MEMORY,
                )
            return self.path.read_text(encoding="utf-8")

    @staticmethod
    def _parse(body):
        parts = re.split(r"^## ([^\n]+)$", body, flags=re.M)
        if len(parts) == 1:
            raise ValueError("MEMORY.md requires section headings")
        aliases = {"用户偏好": "用户画像", "经验": "可复用经验"}
        sections = {
            aliases.get(name.strip(), name.strip()): text.strip()
            for name, text in zip(parts[1::2], parts[2::2])
        }
        for name in SECTIONS:
            sections.setdefault(name, "")
        return parts[0].rstrip(), sections

    @staticmethod
    def _render(prefix, sections):
        return (
            prefix.rstrip()
            + "\n\n"
            + "\n\n".join(
                f"## {name}\n\n{text}".rstrip() for name, text in sections.items()
            )
            + "\n"
        )

    def get_injection(self):
        return (
            "MEMORY.md 是常用画像、环境、约束与通用经验；详细资料用 search_memory/read_memory 召回。\n<core_memory>\n"
            + self.read()
            + "</core_memory>"
        )

    def apply_record(self, record, previous=None, *, core_old_text=""):
        self.read()
        with file_lock(self.path):
            original = self.path.read_text(encoding="utf-8")
            prefix, sections = self._parse(original)
            marker = re.compile(
                r"<!-- memory:"
                + re.escape(record.id)
                + r" -->\n(.*?)\n<!-- /memory -->",
                re.S,
            )
            match = marker.search(original)
            content = record.content or record.result or record.goal
            if match and previous and record.origin != "explicit":
                old = previous.content or previous.result or previous.goal
                if match.group(1).strip() not in (old.strip(), content.strip()):
                    return False
            sections = {
                name: marker.sub("", text).strip() for name, text in sections.items()
            }
            if record.state == "active" and record.projection != "none":
                if CREDENTIAL_PATTERN.search(content):
                    raise ValueError("Credentials cannot enter core memory")
                name = "用户画像" if record.projection == "profile" else "可复用经验"
                text = sections[name]
                block = f"<!-- memory:{record.id} -->\n{content}\n<!-- /memory -->"
                if core_old_text and text.count(core_old_text) == 1:
                    sections[name] = text.replace(core_old_text, block, 1)
                elif core_old_text and not match and content not in text:
                    raise ValueError("Core memory changed; old text no longer matches")
                elif content not in text:
                    sections[name] = (text + "\n\n" + block).strip()
            elif core_old_text and not match:
                if sum(text.count(core_old_text) for text in sections.values()) > 1:
                    raise ValueError("Core memory text to remove is ambiguous")
                sections = {
                    name: text.replace(core_old_text, "", 1).strip()
                    for name, text in sections.items()
                }
            updated = self._render(prefix, sections)
            if updated != original:
                atomic_write_text(self.path, updated)
            return True

    def legacy_content(self):
        body = self.read()
        if not re.search(r"^## (用户偏好|项目简况|经验)$", body, re.M):
            return ""
        backup = self.directory / "migration_backup/MEMORY-v1.md"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(body, encoding="utf-8")
        return body

    def finish_legacy_migration(self, expected):
        with file_lock(self.path):
            prefix, sections = self._parse(self.path.read_text(encoding="utf-8"))
            _, old = self._parse(expected)
            if sections.get("项目简况") == old.get("项目简况"):
                sections.pop("项目简况", None)
            atomic_write_text(self.path, self._render(prefix, sections))

    async def list_memory(self):
        return await asyncio.to_thread(self.read)

    async def snapshot(self):
        body = await self.list_memory()
        return {
            "memory": dict(
                path=self.path,
                body=body,
                chars=len(body),
                empty=body.strip() == EMPTY_MEMORY.strip(),
            )
        }

    async def clear_all(self):
        def clear():
            with file_lock(self.path):
                atomic_write_text(self.path, EMPTY_MEMORY)

        await asyncio.to_thread(clear)
