from __future__ import annotations

from pathlib import Path

from redlotus.infra.paths import (
    resource_root,
    skills_dir,
    user_skills_dir,
)


def runtime_repo_root() -> Path:
    """随包资源根；供 @file 引用等作为 cwd 之外的回退根。"""
    return resource_root()


def readable_roots(*, work_base: Path) -> tuple[Path, ...]:
    """Agent 可读根：当前项目 + 随包基线技能 + 运行时技能 overlay。"""
    roots = [work_base.resolve()]
    for d in (skills_dir(), user_skills_dir()):
        try:
            roots.append(d.resolve())
        except OSError:
            pass
    return tuple(roots)


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_readable_path(path: Path, *, work_base: Path) -> Path:
    """解析后的路径必须落在 当前项目 或技能目录（基线 / overlay）下。"""
    resolved = path.resolve()
    for root in readable_roots(work_base=work_base):
        if is_under_root(resolved, root):
            return resolved
    roots = ", ".join(str(r) for r in readable_roots(work_base=work_base))
    raise ValueError(f"Path not allowed (must be under: {roots}): {resolved}")


def resolve_readable_path(name: str, *, work_base: Path) -> Path:
    """相对路径：技能路径锚定到基线/overlay，其余锚定到 当前项目；绝对路径须落在可读根内。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("Path name must not be empty")
    work = work_base.resolve()

    p_in = Path(name).expanduser()
    if p_in.is_absolute():
        return assert_readable_path(p_in, work_base=work)

    norm = name.replace("\\", "/").strip("/")
    low = norm.lower()
    # 兼容旧写法 src/skills；归一到 skills/...
    if low == "src/skills" or low.startswith("src/skills/"):
        norm = norm[len("src/") :]
        low = norm.lower()
    if low == "skills" or low.startswith("skills/"):
        rel = norm[len("skills") :].lstrip("/")
        for base in (skills_dir(), user_skills_dir()):
            cand = (base / rel).resolve() if rel else base.resolve()
            if cand.exists():
                return assert_readable_path(cand, work_base=work)
        # 默认落在可写 overlay（供新建 / 安装技能）
        cand = (
            (user_skills_dir() / rel).resolve() if rel else user_skills_dir().resolve()
        )
        return assert_readable_path(cand, work_base=work)

    return assert_readable_path((work / name).resolve(), work_base=work)
