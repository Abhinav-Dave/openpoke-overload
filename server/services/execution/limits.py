"""Process-wide limits on execution agents (shared by chat and trigger paths)."""

import asyncio
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Dict, Optional, Set


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


MAX_PARALLEL_AGENTS = _env_int("OPENPOKE_MAX_PARALLEL_AGENTS", 5)
MAX_AGENTS_PER_TURN = _env_int("OPENPOKE_MAX_AGENTS_PER_TURN", 4)

_sem: Optional[asyncio.Semaphore] = None
_sem_loop: Optional[asyncio.AbstractEventLoop] = None
_running = _waiting = _peak = 0


def reset() -> None:
    """Drop the semaphore and zero the counters (tests)."""
    global _sem, _sem_loop, _running, _waiting, _peak
    _sem = _sem_loop = None
    _running = _waiting = _peak = 0


def _semaphore() -> asyncio.Semaphore:
    # Created lazily inside the running loop; a new loop gets a new semaphore.
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem, _sem_loop = asyncio.Semaphore(MAX_PARALLEL_AGENTS), loop
    return _sem


@asynccontextmanager
async def agent_slot():
    """Hold one of MAX_PARALLEL_AGENTS slots; counters survive errors and cancellation."""
    global _running, _waiting, _peak
    sem = _semaphore()
    _waiting += 1
    try:
        await sem.acquire()
    finally:
        _waiting -= 1
    _running += 1
    _peak = max(_peak, _running)
    try:
        yield
    finally:
        _running -= 1
        sem.release()


def get_load() -> Dict[str, int]:
    return {
        "running": _running,
        "waiting": _waiting,
        "max_parallel": MAX_PARALLEL_AGENTS,
        "peak_running": _peak,
    }


# Agents started in the current interaction turn. Set by the interaction runtime for the
# duration of one tool loop; None outside a turn (no cap).
_turn_agents: ContextVar[Optional[Set[str]]] = ContextVar("turn_agents", default=None)


# Set (non-empty) once find_agents ran this turn; None outside a turn.
_turn_found: ContextVar[Optional[Set[str]]] = ContextVar("turn_found", default=None)


# Marked before the turn started (the prompt builder runs first); consumed by start_turn.
_find_pending: ContextVar[bool] = ContextVar("find_pending", default=False)


def start_turn():
    pending = _find_pending.get()
    _find_pending.set(False)
    return _turn_agents.set(set()), _turn_found.set({"find_agents"} if pending else set())


def end_turn(token) -> None:
    _turn_agents.reset(token[0])
    _turn_found.reset(token[1])


def mark_find_agents_called() -> None:
    found = _turn_found.get()
    if found is not None:
        found.add("find_agents")
    else:
        _find_pending.set(True)


def find_agents_missing() -> bool:
    """True inside a turn where find_agents has not been called yet."""
    found = _turn_found.get()
    return found is not None and not found


def try_start_agent(agent_name: str) -> bool:
    """False if this turn already started MAX_AGENTS_PER_TURN other agents."""
    started = _turn_agents.get()
    if started is None or agent_name in started:
        return True
    if len(started) >= MAX_AGENTS_PER_TURN:
        return False
    started.add(agent_name)
    return True
