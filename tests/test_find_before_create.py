import asyncio
import json

import pytest

from server.agents.interaction_agent import tools as tools_mod
from server.agents.interaction_agent.agent import prepare_message_with_history
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import handle_tool_call
from server.agents.execution_agent.runtime import ExecutionResult
from server.services.execution import get_agent_roster, get_execution_agent_logs, limits
from server.services.execution import roster_view


@pytest.fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    monkeypatch.setattr(roster_view, "BOUND_ABOVE", 11)


class StubManager:
    def __init__(self):
        self.started = []

    async def execute_agent(self, agent_name, instructions, request_id=None):
        self.started.append(agent_name)
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")


@pytest.fixture
def stub(monkeypatch):
    s = StubManager()
    monkeypatch.setattr(tools_mod, "_EXECUTION_BATCH_MANAGER", s)
    return s


def seed(n):
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    for i in range(n):
        roster.add_agent(f"agent-{i}")
        logs.record_request(f"agent-{i}", f"routine task number {i}")
    roster.add_agent("misc-helper")
    logs.record_request("misc-helper", "Keep an eye on my sourdough starter feeding schedule")


def send(name):
    return handle_tool_call("send_message_to_agent", {"agent_name": name, "instructions": "go"})


@pytest.fixture
def turn():
    token = limits.start_turn()
    yield
    limits.end_turn(token)


async def test_new_name_without_find_is_rejected_and_changes_nothing(stub, turn):
    seed(20)
    before = list(get_agent_roster().get_agents())
    res = send("brand-new")
    await asyncio.sleep(0.01)
    assert not res.success
    assert "find_agents" in res.payload["error"]
    assert get_agent_roster().get_agents() == before
    assert stub.started == []
    for i in range(limits.MAX_AGENTS_PER_TURN):  # cap untouched
        assert limits.try_start_agent(f"x{i}")


async def test_new_name_succeeds_after_find_even_when_empty(stub, turn):
    seed(20)
    assert handle_tool_call("find_agents", {"keywords": "zzzqqq"}).payload["agents"] == []
    res = send("brand-new")
    await asyncio.sleep(0.01)
    assert res.success and res.payload["new_agent_created"]
    assert "brand-new" in get_agent_roster().get_agents()
    assert stub.started == ["brand-new"]


async def test_existing_name_is_not_gated(stub, turn):
    seed(20)
    assert send("misc-helper").success
    await asyncio.sleep(0.01)
    assert stub.started == ["misc-helper"]


async def test_small_roster_is_not_gated(stub, turn):
    seed(3)
    assert send("brand-new").success


async def test_outside_a_turn_is_not_gated(stub):
    seed(20)
    assert send("brand-new").success


async def test_flag_resets_on_the_next_turn(stub):
    seed(20)
    token = limits.start_turn()
    handle_tool_call("find_agents", {"keywords": "sourdough"})
    assert send("one").success
    limits.end_turn(token)
    token = limits.start_turn()
    try:
        assert not send("two").success
    finally:
        limits.end_turn(token)


async def test_rejection_does_not_count_toward_the_cap(stub, turn):
    seed(20)
    for _ in range(limits.MAX_AGENTS_PER_TURN + 2):
        assert not send("brand-new").success
    handle_tool_call("find_agents", {"keywords": "x"})
    for i in range(limits.MAX_AGENTS_PER_TURN):
        assert send(f"n{i}").success
    assert not send("one-too-many").success  # the cap still counts only started agents


def test_hint_says_find_is_required():
    seed(20)
    content = prepare_message_with_history("hi", "", message_type="user")[0]["content"]
    assert "required" in content.split("<active_agents>")[1].lower()


def _tool(name, args, i):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _msg(*calls):
    return {"choices": [{"message": {"content": "", "tool_calls": list(calls)}}]}


async def test_runtime_rejected_create_then_find_then_reuse(fake_llm, stub):
    seed(20)
    size = len(get_agent_roster().get_agents())
    fake_llm.script = [
        _msg(_tool("send_message_to_agent", {"agent_name": "bread-culture", "instructions": "status"}, 1)),
        _msg(_tool("find_agents", {"keywords": "sourdough starter"}, 2)),
        _msg(_tool("send_message_to_agent", {"agent_name": "misc-helper", "instructions": "status"}, 3)),
        {"choices": [{"message": {"content": "done"}}]},
    ]
    result = await InteractionAgentRuntime().execute("sourdough starter feeding schedule")  # confident match: no auto-search, the gate is the backstop
    await asyncio.sleep(0.01)
    assert result.success
    assert stub.started == ["misc-helper"]
    assert len(get_agent_roster().get_agents()) == size


async def test_runtime_find_nothing_then_create_succeeds(fake_llm, stub):
    seed(20)
    size = len(get_agent_roster().get_agents())
    fake_llm.script = [
        _msg(_tool("send_message_to_agent", {"agent_name": "kayak-trip", "instructions": "plan"}, 1)),
        _msg(_tool("find_agents", {"keywords": "zzzqqq"}, 2)),
        _msg(_tool("send_message_to_agent", {"agent_name": "kayak-trip", "instructions": "plan"}, 3)),
        {"choices": [{"message": {"content": "done"}}]},
    ]
    result = await InteractionAgentRuntime().execute("sourdough starter feeding kayak trip")  # confident match: no auto-search, the gate is the backstop
    await asyncio.sleep(0.01)
    assert result.success
    assert stub.started == ["kayak-trip"]
    assert len(get_agent_roster().get_agents()) == size + 1
