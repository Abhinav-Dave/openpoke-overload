import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from server.services.execution import get_agent_roster, get_execution_agent_logs
from server.services.execution.log_store import _slugify

from conftest import DATA_DIR, REAL_DATA_BEFORE, REAL_DATA_DIR, ROOT


def test_every_store_lives_under_data_dir():
    from server.services.conversation import get_conversation_log, get_working_memory_log
    from server.services.gmail.importance_watcher import get_important_email_watcher
    from server.services.timezone_store import get_timezone_store
    from server.services.triggers import get_trigger_service

    paths = [
        get_agent_roster()._roster_path,
        get_execution_agent_logs()._base_dir,
        get_conversation_log()._path,
        get_working_memory_log()._path,
        get_trigger_service()._store._db_path,
        get_important_email_watcher()._seen_store._path,
        get_timezone_store()._path,
    ]
    for p in paths:
        assert DATA_DIR in Path(p).parents, p


def test_writes_land_in_temp_dir_not_real_data():
    from server.services.conversation import get_conversation_log
    from server.services.timezone_store import get_timezone_store

    get_agent_roster().add_agent("alpha")
    get_execution_agent_logs().record_request("alpha", "hi")
    get_conversation_log().record_user_message("hello")
    get_timezone_store().set_timezone("UTC")

    assert (DATA_DIR / "execution_agents" / "roster.json").exists()
    assert (DATA_DIR / "execution_agents" / "alpha.log").exists()
    assert (DATA_DIR / "conversation" / "poke_conversation.log").exists()
    assert (DATA_DIR / "timezone.txt").exists()
    assert (DATA_DIR / "triggers.db").exists()
    after = sorted(REAL_DATA_DIR.rglob("*")) if REAL_DATA_DIR.exists() else []
    assert after == REAL_DATA_BEFORE


def test_slug_is_safe_on_windows():
    bad = set('<>:"/\\|?*')
    for name in ['a:b', 'what?', 'x*y', 'say "hi"', '<tag>', 'a|b', 'a/b', 'a\\b', ':?*"<>|/\\', 'CON']:
        slug = _slugify(name)
        assert slug and not (set(slug) & bad), (name, slug)
    logs = get_execution_agent_logs()
    logs.record_request('inv:oice? "q3"', "x")
    assert logs.load_recent('inv:oice? "q3"')


def test_slug_unchanged_for_normal_names():
    assert _slugify("Invoice Followups-ACME") == "invoice-followups-acme"
    assert _slugify("flight_booking 2") == "flight-booking-2"
    assert _slugify("   ") == "agent"


def _settings(**env):
    code = "from server.config import get_settings as g; s=g(); print(s.interaction_agent_model, s.summarizer_model, s.data_dir)"
    full = {**{k: v for k, v in os.environ.items() if not k.startswith("OPENPOKE_") or k == "OPENPOKE_DATA_DIR"}, **env}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=full, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def test_model_overrides():
    assert _settings(OPENPOKE_MODEL="vendor/all")[:2] == ["vendor/all", "vendor/all"]
    got = _settings(OPENPOKE_MODEL="vendor/all", OPENPOKE_SUMMARIZER_MODEL="vendor/sum")
    assert got[:2] == ["vendor/all", "vendor/sum"]


def test_data_dir_setting_and_default(monkeypatch):
    assert _settings()[2] == str(DATA_DIR)
    # config.py has no package imports, so load a private copy with the variable unset
    monkeypatch.delenv("OPENPOKE_DATA_DIR")
    spec = importlib.util.spec_from_file_location("config_default_check", ROOT / "server" / "config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    assert cfg.DATA_DIR == REAL_DATA_DIR


@pytest.mark.asyncio
async def test_fake_llm_and_gmail_fixtures_patch_by_name(fake_llm, fake_gmail):
    from server.agents.interaction_agent import runtime

    resp = await runtime.request_chat_completion(model="m", messages=[])
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert fake_llm.count == 1
