import asyncio

import pytest

from server.agents.interaction_agent import runtime as rt
from server.models import ChatMessage, ChatRequest
from server.services.conversation.chat_handler import handle_chat_request

DONE = {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture(autouse=True)
def fresh_queue(monkeypatch):
    monkeypatch.setattr(rt, "MAX_QUEUED_CHAT", 10)
    rt.reset_chat_queue()
    yield
    rt.reset_chat_queue()


def _req(text="hi"):
    return ChatRequest(messages=[ChatMessage(role="user", content=text)])


def _spawned_tasks():
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def test_task_cancelled_before_start_releases_slot():
    resp = await handle_chat_request(_req())
    assert resp.status_code == 202
    assert rt.get_chat_load()["waiting"] == 1
    tasks = _spawned_tasks()
    assert len(tasks) == 1
    tasks[0].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert rt.get_chat_load()["waiting"] == 0


async def test_normal_run_releases_slot_once(monkeypatch):
    async def fake(*a, **k):
        return DONE

    monkeypatch.setattr(rt, "request_chat_completion", fake)
    await handle_chat_request(_req())
    await asyncio.gather(*_spawned_tasks(), return_exceptions=True)
    assert rt.get_chat_load() == {"running": 0, "waiting": 0, "max_queued": 10}


async def test_cancelled_after_start_is_not_released_twice(monkeypatch):
    started = asyncio.Event()

    async def hang(*a, **k):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(rt, "request_chat_completion", hang)
    await handle_chat_request(_req("one"))
    await handle_chat_request(_req("two"))
    await started.wait()
    assert rt.get_chat_load()["waiting"] == 1  # "two" waits behind "one"
    for t in _spawned_tasks():
        t.cancel()
    await asyncio.gather(*_spawned_tasks(), return_exceptions=True)
    assert rt.get_chat_load()["waiting"] == 0
    assert rt.get_chat_load()["running"] == 0


async def test_release_helper_never_goes_negative():
    rt.release_chat_slot()
    assert rt.get_chat_load()["waiting"] == 0


async def test_failed_task_releases_once(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(rt, "request_chat_completion", boom)
    await handle_chat_request(_req())
    await asyncio.gather(*_spawned_tasks(), return_exceptions=True)
    assert rt.get_chat_load()["waiting"] == 0
