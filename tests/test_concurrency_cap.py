import asyncio
import importlib
import json

import pytest

from server.agents.execution_agent import batch_manager as bm_mod
from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent import tools as tools_mod
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.services.execution import get_agent_roster, limits


@pytest.fixture(autouse=True)
def fresh_limits(monkeypatch):
    monkeypatch.setattr(limits, "MAX_PARALLEL_AGENTS", 5)
    monkeypatch.setattr(limits, "MAX_AGENTS_PER_TURN", 4)
    limits.reset()
    yield
    limits.reset()


class FakeRuntime:
    """Stands in for ExecutionAgentRuntime; behaviour is set per test via `behaviour`."""

    behaviour = None

    def __init__(self, agent_name):
        self.agent_name = agent_name

    async def execute(self, instructions):
        await FakeRuntime.behaviour(self.agent_name)
        return ExecutionResult(agent_name=self.agent_name, success=True, response="done")


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(bm_mod, "ExecutionAgentRuntime", FakeRuntime)

    async def no_dispatch(self, payload):
        return None

    monkeypatch.setattr(ExecutionBatchManager, "_dispatch_to_interaction_agent", no_dispatch)
    return FakeRuntime


async def _sleep_run(name):
    await asyncio.sleep(0.02)


async def test_20_runs_never_exceed_5(fake_runtime):
    live = peak = 0

    async def counted(name):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1

    fake_runtime.behaviour = counted
    mgr = ExecutionBatchManager()

    results = await asyncio.gather(*(mgr.execute_agent(f"a{i}", "x") for i in range(20)))

    assert all(r.success for r in results) and len(results) == 20
    assert peak == 5
    assert limits.get_load() == {"running": 0, "waiting": 0, "max_parallel": 5, "peak_running": 5}


async def test_two_managers_share_one_limit(fake_runtime):
    fake_runtime.behaviour = _sleep_run
    m1, m2 = ExecutionBatchManager(), ExecutionBatchManager()

    await asyncio.gather(
        *(m1.execute_agent(f"a{i}", "x") for i in range(10)),
        *(m2.execute_agent(f"b{i}", "x") for i in range(10)),
    )

    assert limits.get_load()["peak_running"] == 5


async def test_load_during_burst(fake_runtime):
    gate = asyncio.Event()

    async def blocked(name):
        await gate.wait()

    fake_runtime.behaviour = blocked
    mgr = ExecutionBatchManager()
    tasks = [asyncio.create_task(mgr.execute_agent(f"a{i}", "x")) for i in range(8)]
    await asyncio.sleep(0.05)

    assert limits.get_load() == {"running": 5, "waiting": 3, "max_parallel": 5, "peak_running": 5}
    gate.set()
    await asyncio.gather(*tasks)
    assert limits.get_load() == {"running": 0, "waiting": 0, "max_parallel": 5, "peak_running": 5}


async def test_raising_run_releases_slot(fake_runtime):
    async def boom(name):
        raise RuntimeError("kaboom")

    fake_runtime.behaviour = boom
    mgr = ExecutionBatchManager()
    for i in range(7):  # more than the 5 slots: leaked slots would hang
        result = await asyncio.wait_for(mgr.execute_agent(f"a{i}", "x"), timeout=2)
        assert not result.success

    assert limits.get_load()["running"] == 0


async def test_timed_out_run_releases_slot(fake_runtime):
    async def forever(name):
        await asyncio.sleep(10)

    fake_runtime.behaviour = forever
    mgr = ExecutionBatchManager(timeout_seconds=0.05)
    for i in range(7):
        result = await asyncio.wait_for(mgr.execute_agent(f"a{i}", "x"), timeout=2)
        assert result.error == "Timeout"

    assert limits.get_load()["running"] == 0


async def test_cancelled_run_releases_slot_and_waiter(fake_runtime):
    async def forever(name):
        await asyncio.sleep(10)

    fake_runtime.behaviour = forever
    mgr = ExecutionBatchManager()
    tasks = [asyncio.create_task(mgr.execute_agent(f"a{i}", "x")) for i in range(7)]
    await asyncio.sleep(0.05)
    assert limits.get_load()["running"] == 5 and limits.get_load()["waiting"] == 2

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert limits.get_load()["running"] == 0 and limits.get_load()["waiting"] == 0


async def test_timeout_starts_after_slot_acquired(fake_runtime):
    # 1 slot, 0.15s timeout, each run takes 0.1s: the second run waits 0.1s in the queue,
    # which would time out if the clock started at queueing.
    limits.MAX_PARALLEL_AGENTS = 1
    limits.reset()

    async def short(name):
        await asyncio.sleep(0.1)

    fake_runtime.behaviour = short
    mgr = ExecutionBatchManager(timeout_seconds=0.15)

    results = await asyncio.gather(mgr.execute_agent("a", "x"), mgr.execute_agent("b", "x"))

    assert [r.success for r in results] == [True, True]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENPOKE_MAX_PARALLEL_AGENTS", "9")
    monkeypatch.setenv("OPENPOKE_MAX_AGENTS_PER_TURN", "2")
    importlib.reload(limits)
    try:
        assert (limits.MAX_PARALLEL_AGENTS, limits.MAX_AGENTS_PER_TURN) == (9, 2)
    finally:
        monkeypatch.undo()
        importlib.reload(limits)
    assert (limits.MAX_PARALLEL_AGENTS, limits.MAX_AGENTS_PER_TURN) == (5, 4)


# ---- per-turn cap ---------------------------------------------------------

def _calls(names):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "send_message_to_agent",
         "arguments": json.dumps({"agent_name": n, "instructions": "do it"})}}
        for i, n in enumerate(names)]}}]}


class StubManager:
    def __init__(self):
        self.started = []

    async def execute_agent(self, agent_name, instructions, request_id=None):
        self.started.append(agent_name)
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")


@pytest.fixture
def stub_manager(monkeypatch):
    stub = StubManager()
    monkeypatch.setattr(tools_mod, "_EXECUTION_BATCH_MANAGER", stub)
    return stub


async def test_turn_cap_rejects_fifth_distinct_agent(fake_llm, stub_manager):
    fake_llm.script = [_calls(["a", "b", "c", "d", "e"]), {"choices": [{"message": {"content": "ok"}}]}]

    result = await InteractionAgentRuntime().execute("go")
    await asyncio.sleep(0.01)

    assert result.execution_agents_used == 4
    assert stub_manager.started == ["a", "b", "c", "d"]
    assert "e" not in get_agent_roster().get_agents()
    tool_msgs = [m for m in fake_llm.calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[4]["content"])["status"] == "error"
    assert "too many agents" in tool_msgs[4]["content"].lower()
    assert all(json.loads(m["content"])["status"] == "success" for m in tool_msgs[:4])


async def test_repeat_message_to_same_agent_counts_once(fake_llm, stub_manager):
    fake_llm.script = [_calls(["a", "a", "a", "b", "c", "d"]), {"choices": [{"message": {"content": "ok"}}]}]

    await InteractionAgentRuntime().execute("go")
    await asyncio.sleep(0.01)

    assert stub_manager.started == ["a", "a", "a", "b", "c", "d"]


async def test_counter_resets_next_turn(fake_llm, stub_manager):
    done = {"choices": [{"message": {"content": "ok"}}]}
    fake_llm.script = [_calls(["a", "b", "c", "d"]), done, _calls(["e", "f", "g", "h"]), done]

    await InteractionAgentRuntime().execute("first")
    await InteractionAgentRuntime().execute("second")
    await asyncio.sleep(0.01)

    assert stub_manager.started == list("abcdefgh")


def test_tool_outside_a_turn_is_not_capped(stub_manager):
    # No turn context (e.g. direct call): behaves as before.
    async def run():
        for n in "abcdef":
            assert tools_mod.send_message_to_agent(n, "x").success
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert len(stub_manager.started) == 6
