"""Tiny helpers for coroutine / executor handling."""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import Executor
from typing import Any, Awaitable, Callable


def is_coro(fn: Callable[..., Any]) -> bool:
    """Return True if *fn* is a coroutine function."""
    return inspect.iscoroutinefunction(fn)


async def run_async(
    fn: Callable[..., Any],
    *args: Any,
    executor: Executor | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: Any
) -> Any:
    """Run *fn* in the current loop; off-load to *executor* if provided."""
    loop = loop or asyncio.get_running_loop()
    if executor:
        return await loop.run_in_executor(executor, fn, *args)
    if is_coro(fn):
        return await fn(*args)
    # synchronous call in event-loop thread
    return fn(*args)
