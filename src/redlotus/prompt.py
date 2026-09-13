from __future__ import annotations

import datetime
import os
import platform
from typing import TYPE_CHECKING

from redlotus.infra.paths import prompts_dir

if TYPE_CHECKING:
    from redlotus.skills.SkillsManager import SkillsManager


def get_skills_summary(skills_manager: SkillsManager) -> str:
    return skills_manager.get_skills_summary()


def format_system_info() -> str:
    shell = (
        "cmd.exe (use type to read files, or explicitly invoke PowerShell)"
        if os.name == "nt"
        else "/bin/sh"
    )
    return (
        f"## System Environment\nOS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"Python: {platform.python_version()}; CPU cores: {os.cpu_count()}\n"
        f"run_command shell: {shell}\n"
    )


def format_prompt_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_prompt(filename: str) -> str:
    filepath = prompts_dir() / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_skills_layout_text(skills_manager: SkillsManager) -> str:
    root = skills_manager.skills_dir.resolve()
    skills_root_path = f"Local absolute path: `{root}`"
    return load_prompt("skills_layout.md").format(skills_root_path=skills_root_path)


def format_long_term_memory_for_prompt(memory_injection: str) -> str:
    text = (memory_injection or "").strip()
    if not text:
        return ""
    return f"## Persistent memory (MEMORY.md)\n\n{text}\n"


def get_skills_as_in_system_prompt(skills_manager: SkillsManager) -> str:
    layout = get_skills_layout_text(skills_manager).rstrip()
    summary = get_skills_summary(skills_manager).rstrip()
    if not layout:
        return summary
    if not summary:
        return layout
    return f"{layout}\n\n{summary}"


def get_common_conduct() -> str:
    return load_prompt("common_conduct.md")


def _build_role_prompt(
    template_name: str,
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    template = load_prompt(template_name)
    fields = {
        "current_time": format_prompt_current_time(),
        "skills_layout": get_skills_layout_text(skills_manager),
        "skills_summary": get_skills_summary(skills_manager),
        "long_term_memory": format_long_term_memory_for_prompt(memory_injection),
        "common_conduct": get_common_conduct(),
        "system_info": format_system_info(),
    }
    from redlotus.workspace.workspace import current_workspace

    return (
        template.format(**fields)
        + f"\nCurrent project: {current_workspace()}\nDeliverables: {current_workspace() / 'WorkDatabase'}\n"
    )


def get_manager_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt("manager_system.md", skills_manager, memory_injection)


def get_worker_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt("worker_system.md", skills_manager, memory_injection)


def get_coordinator_system_prompt(
    skills_manager: SkillsManager,
    memory_injection: str = "",
) -> str:
    return _build_role_prompt("coordinator_system.md", skills_manager, memory_injection)
