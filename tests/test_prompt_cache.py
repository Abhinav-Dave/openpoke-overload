import json

import httpx
import pytest

from server.openrouter_client import client as oc
from server.services import usage

RealClient = httpx.AsyncClient
OK = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
CACHE = {"type": "ephemeral"}


@pytest.fixture
def net(monkeypatch):
    state = type("Net", (), {})()
    state.script, state.bodies = [], []

    def handler(request):
        state.bodies.append(json.loads(request.content))
        return state.script.pop(0)

    async def fake_sleep(s):
        pass

    monkeypatch.setattr(oc.httpx, "AsyncClient", lambda **kw: RealClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(oc, "_sleep", fake_sleep)
    monkeypatch.delenv("OPENPOKE_PROMPT_CACHE", raising=False)
    monkeypatch.setattr(oc, "_cache_unsupported", False)
    usage.reset()
    yield state
    usage.reset()


def resp(status, body=None):
    return httpx.Response(status, json=body if body is not None else OK)


async def call(model="anthropic/claude-sonnet-5.5"):
    return await oc.request_chat_completion(model=model, messages=[{"role": "user", "content": "hi"}], api_key="k")


async def test_anthropic_model_gets_cache_control(net):
    net.script = [resp(200)]
    await call()
    assert net.bodies[0]["cache_control"] == CACHE


async def test_other_model_has_no_cache_field(net):
    net.script = [resp(200)]
    await call("openai/gpt-5")
    assert "cache_control" not in net.bodies[0]


async def test_env_zero_disables(net, monkeypatch):
    monkeypatch.setenv("OPENPOKE_PROMPT_CACHE", "0")
    net.script = [resp(200)]
    await call()
    assert "cache_control" not in net.bodies[0]


async def test_400_mentioning_cache_control_retries_once_then_skips(net):
    net.script = [resp(400, {"error": "unknown field cache_control"}), resp(200), resp(200)]
    await call()
    assert len(net.bodies) == 2
    assert "cache_control" in net.bodies[0] and "cache_control" not in net.bodies[1]
    await call()
    assert len(net.bodies) == 3 and "cache_control" not in net.bodies[2]


async def test_unrelated_400_not_retried(net):
    net.script = [resp(400, {"error": "bad messages"})]
    with pytest.raises(oc.OpenRouterError):
        await call()
    assert len(net.bodies) == 1 and oc._cache_unsupported is False


def test_usage_accumulates_cache_tokens_and_tolerates_missing():
    usage.reset()
    usage.record_success("m", {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 5}})
    usage.record_success("m", {"prompt_tokens": 10})
    usage.record_success("m", {"prompt_tokens_details": None})
    u = usage.get_usage()
    assert (u["cached_tokens"], u["cache_write_tokens"]) == (80, 5)
    assert u["models"]["m"]["cached_tokens"] == 80
    usage.reset()
