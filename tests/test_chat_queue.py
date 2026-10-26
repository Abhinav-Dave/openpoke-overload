import asyncio
import json
import re

import pytest

from server.agents.interaction_agent import runtime as rt
from server.agents.interaction_agent import tools as tools_mod
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.execution_agent.runtime import ExecutionResult
from server.models import ChatMessage, ChatRequest
from server.services.conversation import get_conversation_log
from server.services.conversation.chat_handler import handle_chat_request
from server.services.execution import limits

DONE = {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture(autouse=True)
def fresh_queue(monkeypatch):
    monkeypatch.setattr(rt, "MAX_QUEUED_CHAT", 10)
    monkeypatch.setattr(limits, "MAX_AGENTS_PER_TURN", 4)
    rt.reset_chat_queue()
    yield
    rt.reset_chat_queue()


class SlowLLM:
    """Async stand-in for request_chat_completion that tracks overlap."""

    def __init__(self, delay=0.01, script=None):
        self.delay, self.script, self.live, self.peak, self.calls = delay, script, 0, 0, 0

    async def __call__(self, *args, **kwargs):
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
            if self.script:
                item = self.script[min(self.calls, len(self.script)) - 1]
                if isinstance(item, Exception):
                    raise item
                return item
            return DONE
        finally:
            self.live -= 1


@pytest.fixture
def llm(monkeypatch):
    fake = SlowLLM()
    monkeypatch.setattr(rt, "request_chat_completion", fake)
    return fake


def _req(text):
    return ChatRequest(messages=[ChatMessage(role="user", content=text)])


def _tool(name, **args):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


async def _drain():
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending, return_exceptions=True)


def _replies():
    return re.findall(r"<poke_reply[^>]*>(.*?)</poke_reply>", get_conversation_log().load_transcript())


def _users():
    return re.findall(r"<user_message[^>]*>(.*?)</user_message>", get_conversation_log().load_transcript())


async def test_five_messages_run_one_at_a_time_in_order(llm):
    peak_running = 0

    async def watch():
        nonlocal peak_running
        while True:
            peak_running = max(peak_running, rt.get_chat_load()["running"])
            await asyncio.sleep(0)

    watcher = asyncio.create_task(watch())
    resps = [await handle_chat_request(_req(f"m{i}")) for i in range(5)]
    assert [r.status_code for r in resps] == [202] * 5
    await asyncio.gather(*[t for t in asyncio.all_tasks() if t not in (asyncio.current_task(), watcher)])
    watcher.cancel()

    assert llm.peak == 1 and peak_running == 1
    assert _users() == [f"m{i}" for i in range(5)]
    assert rt.get_chat_load() == {"running": 0, "waiting": 0, "max_queued": 10}


async def test_queue_full_rejects_new_user_message(llm):
    ok = [await handle_chat_request(_req(f"m{i}")) for i in range(10)]
    assert all(r.status_code == 202 for r in ok)
    assert rt.get_chat_load()["waiting"] == 10

    rejected = await handle_chat_request(_req("m10"))

    assert rejected.status_code == 429
    body = json.loads(rejected.body)
    assert body["ok"] is False and "still working" in body["error"]
    await _drain()
    assert "m10" not in _users() and len(_users()) == 10
    assert llm.calls == 10


async def test_agent_results_queue_and_are_never_rejected(llm):
    for i in range(10):
        await handle_chat_request(_req(f"m{i}"))
    agent_runs = [asyncio.create_task(InteractionAgentRuntime().handle_agent_message(f"a{i}")) for i in range(3)]
    await asyncio.sleep(0)
    load = rt.get_chat_load()
    assert load["waiting"] + load["running"] == 13

    results = await asyncio.gather(*agent_runs)

    assert all(r.success for r in results)
    await _drain()
    assert llm.peak == 1
    assert len(re.findall(r"<agent_message", get_conversation_log().load_transcript())) == 3


async def test_lock_released_when_run_raises(monkeypatch):
    llm = SlowLLM(script=[RuntimeError("boom"), DONE])
    monkeypatch.setattr(rt, "request_chat_completion", llm)

    first = await InteractionAgentRuntime().execute("one")
    second = await InteractionAgentRuntime().execute("two")

    assert not first.success and second.success
    assert rt.get_chat_load()["running"] == 0


async def test_lock_released_when_run_cancelled(llm):
    llm.delay = 5
    task = asyncio.create_task(InteractionAgentRuntime().execute("one"))
    await asyncio.sleep(0.01)
    assert rt.get_chat_load()["running"] == 1
    waiter = asyncio.create_task(InteractionAgentRuntime().execute("two"))
    await asyncio.sleep(0.01)
    assert rt.get_chat_load()["waiting"] == 1

    waiter.cancel()
    task.cancel()
    await asyncio.gather(task, waiter, return_exceptions=True)
    assert rt.get_chat_load() == {"running": 0, "waiting": 0, "max_queued": 10}

    llm.delay = 0
    assert (await InteractionAgentRuntime().execute("three")).success


async def test_iteration_limit_tells_the_user(monkeypatch):
    monkeypatch.setattr(rt, "request_chat_completion", SlowLLM(delay=0, script=[_tool("bogus_tool")]))

    result = await InteractionAgentRuntime().execute("loop forever")

    assert not result.success
    assert len(_replies()) == 1 and "could not finish" in _replies()[0]


async def test_exception_tells_the_user_without_exception_text(monkeypatch):
    monkeypatch.setattr(rt, "request_chat_completion", SlowLLM(delay=0, script=[RuntimeError("secret-db-password")]))

    result = await InteractionAgentRuntime().execute("hi")

    assert not result.success and "secret-db-password" in result.error
    assert len(_replies()) == 1 and "secret-db-password" not in _replies()[0]


async def test_agent_message_failure_also_tells_the_user(monkeypatch):
    monkeypatch.setattr(rt, "request_chat_completion", SlowLLM(delay=0, script=[RuntimeError("x")]))

    await InteractionAgentRuntime().handle_agent_message("result")

    assert len(_replies()) == 1


async def test_no_duplicate_error_after_reply_already_sent(monkeypatch):
    # Model replied to the user, then the loop never finished: the user already got an answer.
    script = [_tool("send_message_to_user", message="hello there"), _tool("bogus_tool")]
    monkeypatch.setattr(rt, "request_chat_completion", SlowLLM(delay=0, script=script))

    result = await InteractionAgentRuntime().execute("hi")

    assert not result.success
    assert _replies() == ["hello there"]


async def test_turn_agent_cap_still_applies_inside_lock(monkeypatch):
    started = []

    class Stub:
        async def execute_agent(self, agent_name, instructions, request_id=None):
            started.append(agent_name)
            return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_mod, "_EXECUTION_BATCH_MANAGER", Stub())
    calls = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "send_message_to_agent",
         "arguments": json.dumps({"agent_name": n, "instructions": "x"})}} for i, n in enumerate("abcde")]}}]}
    monkeypatch.setattr(rt, "request_chat_completion", SlowLLM(delay=0, script=[calls, DONE]))

    result = await InteractionAgentRuntime().execute("go")
    await asyncio.sleep(0.01)

    assert result.execution_agents_used == 4 and started == list("abcd")


async def test_single_message_unchanged(llm):
    resp = await handle_chat_request(_req("hello"))
    await _drain()

    assert resp.status_code == 202
    assert _users() == ["hello"] and _replies() == ["ok"]
    assert rt.get_chat_load() == {"running": 0, "waiting": 0, "max_queued": 10}


async def test_lock_is_recreated_for_a_new_loop(llm):
    await InteractionAgentRuntime().execute("a")
    first = rt._lock
    rt._lock_loop = object()  # pretend the loop changed
    await InteractionAgentRuntime().execute("b")
    assert rt._lock is not first
