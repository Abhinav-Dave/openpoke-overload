import json
import logging
import os
import subprocess
import sys
import threading

from server.services.execution import roster as roster_mod
from server.services.execution.roster import AgentRoster

from conftest import ROOT


def test_reads_legacy_list_format(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")

    assert AgentRoster(path).get_agents() == ["alpha", "beta"]


def test_save_writes_names_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("alpha")
    roster.add_agent("beta")

    assert json.loads(path.read_text(encoding="utf-8")) == ["alpha", "beta"]
    assert [p.name for p in tmp_path.iterdir()] == ["roster.json"]


def test_crash_during_replace_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("alpha")
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    roster.add_agent("beta")

    assert path.read_bytes() == before


def test_50_threads_lose_no_name(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    names = [f"agent-{i}" for i in range(50)]
    threads = [threading.Thread(target=roster.add_agent, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(roster.get_agents()) == sorted(names)
    assert sorted(json.loads(path.read_text(encoding="utf-8"))) == sorted(names)


def test_corrupt_file_kept_as_bad_not_overwritten(tmp_path, caplog):
    path = tmp_path / "roster.json"
    path.write_text('["alpha", "be', encoding="utf-8")
    bad = tmp_path / "roster.json.bad"
    bad.write_text("older junk", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        roster = AgentRoster(path)

    assert roster.get_agents() == []
    assert bad.read_text(encoding="utf-8") == '["alpha", "be'
    assert not path.exists()
    assert any(r.levelno >= logging.ERROR for r in caplog.records)

    roster.add_agent("gamma")  # first real add writes the file again
    assert json.loads(path.read_text(encoding="utf-8")) == ["gamma"]
    assert bad.read_text(encoding="utf-8") == '["alpha", "be'


def test_import_works_without_fcntl():
    code = "import sys; sys.modules['fcntl'] = None; import server.services.execution.roster"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_atomic_write_json_gives_up_after_permission_errors(tmp_path, monkeypatch):
    path = tmp_path / "roster.json"
    path.write_text('["keep"]', encoding="utf-8")
    calls = []

    def locked(*a, **k):
        calls.append(1)
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(roster_mod.time, "sleep", lambda s: None)
    roster_mod.atomic_write_json(path, ["new"])  # must not raise

    assert len(calls) > 1
    assert path.read_text(encoding="utf-8") == '["keep"]'
