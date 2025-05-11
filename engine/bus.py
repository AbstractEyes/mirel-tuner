# engine/bus.py
##############################################
from __future__ import annotations
import asyncio
import fnmatch
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Protocol

##############################################
# optional accelerate integration
try:
    from accelerate import Accelerator      # noqa
    _ACCEL = Accelerator()
    _WORLD = _ACCEL.num_processes
    _RANK  = _ACCEL.process_index
    _ACCEL_OK = True
except Exception:
    _ACCEL = None
    _WORLD = 1
    _RANK  = 0
    _ACCEL_OK = False

##############################################
# callback signature protocol
class _CallbackSig(Protocol):
    async def __call__(self, *, event: str, payload: Dict[str, Any]) -> None: ...

##############################################
# subscription handle
@dataclass
class _SubHandle:
    pattern: str
    fn:      _CallbackSig
    bus:     "EngineEventBus"
    def close(self) -> None:
        self.bus._remove(self)

##############################################
# main bus class
class EngineEventBus:
    def __init__(self) -> None:
        self._subs: List[_SubHandle] = []
        self._pool = ThreadPoolExecutor(max_workers=4)

    ##########################################
    # subscribe (sync or async listener)
    def subscribe(self, pattern: str, fn: Callable) -> _SubHandle:
        if asyncio.iscoroutinefunction(fn):
            async_fn = fn                                   # type: ignore
        else:
            async def wrapper(*, event: str, payload: Dict[str, Any]) -> None:  # noqa
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._pool, fn, event, payload)
            async_fn = wrapper                              # type: ignore

        h = _SubHandle(pattern, async_fn, self)
        self._subs.append(h)
        return h

    ##########################################
    # decorator sugar
    def on(self, pattern: str) -> Callable[[Callable], Callable]:
        def deco(fn: Callable) -> Callable:
            self.subscribe(pattern, fn)
            return fn
        return deco

    ##########################################
    # emit event (optional broadcast via accelerate)
    def emit(self, event: str, /, broadcast: bool = False, **payload: Any) -> None:
        if broadcast and _ACCEL_OK:
            if _RANK == 0:
                _ACCEL.broadcast_object_list([event, payload], src=0)
            else:
                event, payload = _ACCEL.broadcast_object_list([None, None], src=0)  # noqa

        if not self._subs:
            return

        loop = asyncio.get_running_loop()
        for h in self._subs:
            if fnmatch.fnmatch(event, h.pattern):
                loop.create_task(h.fn(event=event, payload=payload))

    ##########################################
    # remove a listener
    def _remove(self, h: _SubHandle) -> None:
        self._subs = [s for s in self._subs if s is not h]

##############################################
# per-process singleton via ContextVar
_bus_var: contextvars.ContextVar[EngineEventBus] = contextvars.ContextVar("bus")

def get_bus() -> EngineEventBus:
    bus = _bus_var.get(None)        # type: ignore
    if bus is None:
        bus = EngineEventBus()
        _bus_var.set(bus)
    return bus

##############################################
# convenience aliases
bus       = get_bus()
subscribe = bus.subscribe
on        = bus.on
emit      = bus.emit
##############################################
