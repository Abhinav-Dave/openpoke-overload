import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.openrouter_client import client as oc
from server.routes.meta import router as meta_router
from server.services import usage

KEY = "sk-secret-key-123"
OK = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002}}
RealClient = httpx.AsyncClient


@pytest.fixture
def net(monkeypatch):
    """Fake HTTP: net.script is a list of httpx.Response / Exception, consumed in order."""
    state = type("Net", (), {})()
    state.script, state.calls, state.sleeps = [], 0, []

    def handler(request):
        state.calls += 1
        item = state.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_sleep(s):
        state.sleeps.append(s)

    monkeypatch.setattr(oc.httpx, "AsyncClient", lambda **kw: RealClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(oc, "_sleep", fake_sleep)
    monkeypatch.delenv("OPENPOKE_LLM_MAX_ATTEMPTS", raising=False)
    usage.reset()
    yield state
    usage.reset()


def resp(status, body=None, headers=None):
    return httpx.Response(status, json=body if body is not None else {"error": "x"}, headers=headers)


async def call(model="m1"):
    return await oc.request_chat_completion(model=model, messages=[{"role": "user", "content": "SECRET PROMPT"}], api_key=KEY)


async def test_success_first_try(net):
    net.script = [resp(200, OK)]
    assert await call() == OK
    assert net.calls == 1 and net.sleeps == []
    u = usage.get_usage()
    assert (u["calls"], u["retries"], u["failed_calls"]) == (1, 0, 0)
    assert (u["prompt_tokens"], u["completion_tokens"]) == (10, 5)


async def test_429_then_200_retries_once(net):
    net.script = [resp(429), resp(200, OK)]
    await call()
    assert net.calls == 2
    u = usage.get_usage()
    assert u["retries"] == 1 and u["calls"] == 1


async def test_retry_after_honoured_and_capped(net):
    net.script = [resp(429, headers={"Retry-After": "3"}), resp(429, headers={"Retry-After": "999"}), resp(200, OK)]
    await call()
    assert net.sleeps == [3, 30]


async def test_503_three_times_raises(net):
    net.script = [resp(503), resp(503), resp(503)]
    with pytest.raises(oc.OpenRouterError) as ei:
        await call()
    assert net.calls == 3
    assert usage.get_usage()["failed_calls"] == 1
    assert KEY not in str(ei.value) and "SECRET PROMPT" not in str(ei.value)


@pytest.mark.parametrize("status", [401, 402, 404])
async def test_other_4xx_not_retried(net, status):
    net.script = [resp(status)]
    with pytest.raises(oc.OpenRouterError):
        await call()
    assert net.calls == 1
    assert usage.get_usage()["failed_calls"] == 1


async def test_timeout_retried(net):
    net.script = [httpx.ReadTimeout("slow"), resp(200, OK)]
    await call()
    assert net.calls == 2


async def test_backoff_sequence(net, monkeypatch):
    monkeypatch.setenv("OPENPOKE_LLM_MAX_ATTEMPTS", "4")
    net.script = [resp(500), resp(502), resp(504), resp(200, OK)]
    await call()
    assert len(net.sleeps) == 3
    for got, base in zip(net.sleeps, [0.5, 1.0, 2.0]):
        assert base <= got <= base * 1.2


async def test_env_override_attempts(net, monkeypatch):
    monkeypatch.setenv("OPENPOKE_LLM_MAX_ATTEMPTS", "1")
    net.script = [resp(429), resp(200, OK)]
    with pytest.raises(oc.OpenRouterError):
        await call()
    assert net.calls == 1


async def test_usage_totals_per_model_and_cost(net):
    net.script = [resp(200, OK), resp(200, OK), resp(200, {"usage": {"prompt_tokens": 1, "completion_tokens": 2}})]
    await call("a")
    await call("a")
    await call("b")
    u = usage.get_usage()
    assert u["calls"] == 3 and u["prompt_tokens"] == 21 and u["completion_tokens"] == 12
    assert u["models"]["a"]["calls"] == 2 and u["models"]["a"]["cost"] == pytest.approx(0.004)
    assert u["models"]["b"]["prompt_tokens"] == 1 and u["models"]["b"]["cost"] is None


async def test_missing_usage_tolerated(net):
    net.script = [resp(200, {"choices": []})]
    await call()
    u = usage.get_usage()
    assert u["calls"] == 1 and u["prompt_tokens"] == 0 and u["cost"] is None


async def test_reset_and_no_secrets_in_usage(net):
    net.script = [resp(200, OK)]
    await call()
    text = str(usage.get_usage())
    assert KEY not in text and "SECRET PROMPT" not in text and "Bearer" not in text
    usage.reset()
    assert usage.get_usage()["calls"] == 0 and usage.get_usage()["models"] == {}


def test_load_endpoint(net):
    app = FastAPI()
    app.include_router(meta_router, prefix="/api/v1")
    usage.record_success("m", {"prompt_tokens": 3, "completion_tokens": 4})
    body = TestClient(app).get("/api/v1/meta/load").json()
    assert {"running", "waiting", "max_parallel", "peak_running"} <= set(body["agents"])
    assert body["usage"]["prompt_tokens"] == 3
