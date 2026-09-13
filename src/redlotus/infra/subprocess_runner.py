# -*- coding: utf-8 -*-
"""可取消的子进程执行原语。

被 tools/ 与 skills/ 共用的叶子模块：只依赖标准库，不依赖任何项目模块，
因此可被任意层 import 而不引入循环依赖。run_command /
execute_skill_script 共用同一套 杀进程树 / 超时 / 取消 语义。
"""

import asyncio
import os
import signal
import subprocess
import platform as _platform


async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """杀掉子进程及其后代（无 psutil 依赖），并收尸。"""
    if proc.returncode is not None:
        return
    try:
        if _platform.system() == "Windows":
            # /T 杀整棵树：shell 会经 cmd.exe 再起真正的子进程。
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
            if killer.returncode and proc.returncode is None:
                raise PermissionError(
                    f"taskkill could not terminate process tree {proc.pid} (exit {killer.returncode})"
                )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise  # Do not report successful tree cleanup when the OS denied it.
    await asyncio.wait_for(proc.wait(), timeout=5)


async def run_subprocess(
    args, *, shell: bool, cwd: str, env: dict | None = None, timeout: float
) -> tuple[str, str, int | None]:
    """跑子进程并返回 (stdout, stderr, returncode)；取消或超时都会杀掉整棵进程树。

    取代 asyncio.to_thread(subprocess.run, ...)——后者在任务被取消时既不中断阻塞线程、
    也不杀子进程，会留下孤儿进程与卡死的线程池槽位。
    """
    kwargs: dict = dict(
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env
    )
    if _platform.system() == "Windows":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True  # 独立进程组，便于 killpg

    if shell:
        proc = await asyncio.create_subprocess_shell(args, **kwargs)
    else:
        proc = await asyncio.create_subprocess_exec(*args, **kwargs)

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _terminate_process_tree(proc)
        raise subprocess.TimeoutExpired(args, timeout)
    except asyncio.CancelledError:
        await _terminate_process_tree(proc)
        raise
    return (
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        proc.returncode,
    )
