import re
import subprocess
import sys

import pytest

from server.services.execution.log_store import ExecutionAgentLogStore, _file_slug

COLLIDING = ["a:b", "a/b", "a b", "A-B"]
CANONICAL = ["a-b", "email-agent", "agent1", "x", "a-b-2"]
WINDOWS_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@pytest.fixture
def store(tmp_path):
    return ExecutionAgentLogStore(tmp_path)


def test_colliding_names_get_distinct_files(store):
    paths = {store._log_path(n) for n in COLLIDING}
    assert len(paths) == 4


def test_canonical_names_keep_old_slug():
    for name in CANONICAL:
        assert _file_slug(name) == name


def test_noncanonical_is_old_slug_plus_hash():
    s = _file_slug("a:b")
    assert re.fullmatch(r"a-b-[0-9a-f]{6}", s)


def test_slug_stable_across_calls_and_processes():
    code = "from server.services.execution.log_store import _file_slug;print(_file_slug('A:B'))"
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        for _ in range(2)
    }
    assert outs == {_file_slug("A:B")}


def test_legacy_file_is_picked_up(store, tmp_path):
    (tmp_path / "a-b.log").write_text('<agent_request timestamp="t">old</agent_request>\n', encoding="utf-8")
    assert store._log_path("a:b") == tmp_path / "a-b.log"
    assert [e[2] for e in store.iter_entries("a:b")] == ["old"]
    store.record_action("a:b", "more")
    assert len(list(store.iter_entries("a:b"))) == 2
    # the other colliding names do not get the legacy file once it exists... they share it by design
    assert store._log_path("a/b") == tmp_path / "a-b.log"


def test_new_file_wins_over_legacy_when_both_exist(store, tmp_path):
    (tmp_path / "a-b.log").write_text("", encoding="utf-8")
    new = tmp_path / f"{_file_slug('a:b')}.log"
    new.write_text("", encoding="utf-8")
    assert store._log_path("a:b") == new


def test_two_writes_land_in_one_file(store, tmp_path):
    store.record_request("a:b", "one")
    store.record_request("a:b", "two")
    store.record_request("a/b", "other")
    files = sorted(p.name for p in tmp_path.glob("*.log"))
    assert len(files) == 2
    assert [e[2] for e in store.iter_entries("a:b")] == ["one", "two"]
    assert [e[2] for e in store.iter_entries("a/b")] == ["other"]


@pytest.mark.parametrize("name", ["", "   ", "!!!", "café", "日本語", "😀 agent", "con", "CON"])
def test_odd_names_make_valid_windows_names(name):
    s = _file_slug(name)
    assert s and not WINDOWS_BAD.search(s)
    assert not s.endswith((".", " "))
    assert s.split(".")[0].lower() not in {"con", "prn", "aux", "nul"}
    assert len(s) < 200


def test_distinct_unicode_names_distinct_slugs():
    assert _file_slug("日本語") != _file_slug("中文字")
    assert _file_slug("") != _file_slug("!!!")


def test_reserved_device_name_keeps_prefix():
    assert _file_slug("con") == "agent-con"
    assert _file_slug("nul") == "agent-nul"


def test_all_slugs_valid_on_disk(store, tmp_path):
    for n in COLLIDING + ["", "日本語", "con", "😀"]:
        store.record_request(n, "x")
    assert len(list(tmp_path.glob("*.log"))) >= 7


@pytest.mark.parametrize("other", ["A:B", "a b", "A-B", "a/b"])
def test_canonical_agent_log_is_not_adopted(store, other):
    from server.services.execution import get_agent_roster

    get_agent_roster().add_agent("a-b")
    store.record_request("a-b", "canonical")
    store.record_request(other, "other")
    assert store._log_path(other) != store._log_path("a-b")
    assert [e[2] for e in store.iter_entries("a-b")] == ["canonical"]
    assert [e[2] for e in store.iter_entries(other)] == ["other"]


def test_legacy_adoption_only_without_canonical_roster_entry(store, tmp_path):
    from server.services.execution import get_agent_roster

    get_agent_roster().add_agent("A:B")  # roster has no canonical "a-b"
    (tmp_path / "a-b.log").write_text('<agent_request timestamp="t">old</agent_request>\n', encoding="utf-8")
    assert store._log_path("A:B") == tmp_path / "a-b.log"


def test_roster_failure_means_no_adoption(store, tmp_path, monkeypatch):
    from server.services.execution import get_agent_roster

    def boom():
        raise RuntimeError("roster down")

    monkeypatch.setattr(get_agent_roster(), "get_agents", boom)
    (tmp_path / "a-b.log").write_text("", encoding="utf-8")
    assert store._log_path("A:B") == tmp_path / f"{_file_slug('A:B')}.log"
