"""Shared fixtures. OPENPOKE_DATA_DIR must be set before any `server` import."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_DIR = ROOT / "server" / "data"
# Taken before `server` is imported, so tests can prove the real folder was never touched.
REAL_DATA_BEFORE = sorted(REAL_DATA_DIR.rglob("*")) if REAL_DATA_DIR.exists() else []

DATA_DIR = Path(tempfile.mkdtemp(prefix="openpoke_test_data_"))
os.environ["OPENPOKE_DATA_DIR"] = str(DATA_DIR)

import server.app  # noqa: E402,F401  (loads every module so the fixtures below can patch them)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(DATA_DIR, ignore_errors=True)


def _loaded_server_modules():
    return [m for n, m in list(sys.modules.items()) if n.startswith("server") and m is not None]


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Wipe the temp data folder and reset module-level singletons between tests."""
    from server.agents.execution_agent.batch_manager import ExecutionBatchManager
    from server.agents.interaction_agent import tools as interaction_tools
    from server.config import get_settings
    from server.services.conversation.summarization import scheduler as summ_scheduler
    from server.services.execution import get_agent_roster
    from server.services.gmail import client as gmail_client
    from server.services.gmail import importance_watcher
    from server.services import timezone_store, trigger_scheduler
    from server.services.triggers import get_trigger_service

    # triggers.db stays: TriggerStore leaves sqlite handles open, which Windows will not let us delete.
    for child in DATA_DIR.iterdir():
        if child.name.startswith("triggers.db"):
            continue
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    (DATA_DIR / "execution_agents").mkdir()
    (DATA_DIR / "conversation").mkdir()
    get_trigger_service().clear_all()

    get_agent_roster()._agents = []
    interaction_tools._EXECUTION_BATCH_MANAGER = ExecutionBatchManager()
    summ_scheduler._pending = summ_scheduler._running = False
    importance_watcher._watcher_instance = None
    trigger_scheduler._scheduler_instance = None
    timezone_store._timezone_store._cached = None
    gmail_client._CLIENT = None
    gmail_client._ACTIVE_USER_ID = None
    gmail_client._PROFILE_CACHE.clear()

    # Settings are read at import time; never read .env, never call the network.
    monkeypatch.setattr(get_settings(), "openrouter_api_key", "test-key")
    yield


class FakeLLM:
    """Scripted stand-in for request_chat_completion. `script` is a list of responses
    (dict) or callables(kwargs) -> dict, consumed in order; the last one repeats."""

    def __init__(self):
        self.script = [{"choices": [{"message": {"content": "ok"}}]}]
        self.calls = []

    @property
    def count(self):
        return len(self.calls)

    async def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        item = self.script[min(len(self.calls), len(self.script)) - 1]
        return item(kwargs) if callable(item) else item


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    for mod in _loaded_server_modules():
        if hasattr(mod, "request_chat_completion") and not mod.__name__.startswith("server.openrouter_client"):
            monkeypatch.setattr(mod, "request_chat_completion", fake)
    return fake


class FakeGmail:
    def __init__(self):
        self.result = {"data": {"messages": []}}
        self.calls = []
        self.user_id = "fake-user"

    def execute(self, tool, user_id, arguments=None, **kwargs):
        self.calls.append((tool, user_id, arguments))
        return self.result


@pytest.fixture
def fake_gmail(monkeypatch):
    fake = FakeGmail()
    for mod in _loaded_server_modules():
        if hasattr(mod, "execute_gmail_tool"):
            monkeypatch.setattr(mod, "execute_gmail_tool", fake.execute)
        if hasattr(mod, "get_active_gmail_user_id"):
            monkeypatch.setattr(mod, "get_active_gmail_user_id", lambda *a, **k: fake.user_id)
    return fake
