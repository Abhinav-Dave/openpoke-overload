import os
import subprocess
import sys

import pytest

from server.agents.execution_agent.agent import ExecutionAgent
from server.services.execution import log_store as ls
from server.services.execution.log_store import ExecutionAgentLogStore

from conftest import ROOT

NAME = "invoice-followups"


@pytest.fixture
def store(tmp_path):
    return ExecutionAgentLogStore(tmp_path)


def fill(store, n, text=None):
    for i in range(n):
        body = text or f"entry {i}: " + "invoice detail text " * 15
        (store.record_request if i % 2 == 0 else store.record_agent_response)(NAME, body)


def parse_all(transcript):
    """Every non-header line must be one complete, parseable entry."""
    lines = transcript.split("\n")
    s = ExecutionAgentLogStore.__new__(ExecutionAgentLogStore)
    return [s._parse_line(l) for l in lines if not l.startswith("[Earlier")]


def test_small_log_is_byte_identical(store):
    fill(store, 5)
    assert store.load_transcript_tail(NAME) == store.load_transcript(NAME)


def test_empty_log(store):
    assert store.load_transcript_tail(NAME) == ""


def test_200_entries_bounded_with_header_and_newest(store):
    fill(store, 200)
    out = store.load_transcript_tail(NAME)
    assert len(out) <= ls.EXEC_LOG_MAX_CHARS
    assert "entry 199:" in out and "entry 179:" not in out
    assert out.startswith("[Earlier history omitted: 180 entries.")


def test_cut_is_on_entry_boundary(store):
    fill(store, 200)
    parsed = parse_all(store.load_transcript_tail(NAME))
    assert len(parsed) == 20 and all(p is not None for p in parsed)


def test_order_preserved(store):
    fill(store, 50)
    out = store.load_transcript_tail(NAME)
    idx = [out.index(f"entry {i}:") for i in range(30, 50)]
    assert idx == sorted(idx)


def test_char_cap_drops_oldest_whole_entries(store):
    fill(store, 30, text="x" * 500)
    out = store.load_transcript_tail(NAME, max_entries=30, max_chars=3000)
    parsed = parse_all(out)
    assert len(out) <= 3000 and 0 < len(parsed) < 30
    assert all(p is not None for p in parsed)


def test_huge_newest_entry_truncated_not_dropped(store):
    fill(store, 3)
    store.record_agent_response(NAME, "Z" * 50000)
    out = store.load_transcript_tail(NAME, max_chars=1000)
    assert len(out) <= 1000
    assert out.endswith("[truncated]") and "ZZZZ" in out
    assert out.startswith("[Earlier history omitted: 3 entries.")


def test_header_contains_first_request(store):
    store.record_request(NAME, "Chase the Acme invoice\nplease " + "y" * 500)
    fill(store, 40)
    header = store.load_transcript_tail(NAME).split("\n")[0]
    assert "First request was: Chase the Acme invoice please yyy" in header
    assert len(header) < 400


def test_two_tail_sizes(store):
    fill(store, 100)
    five = store.load_transcript_tail(NAME, max_entries=5)
    ten = store.load_transcript_tail(NAME, max_entries=10)
    assert len(parse_all(five)) == 5 and len(parse_all(ten)) == 10
    assert "omitted: 95 entries" in five and "omitted: 90 entries" in ten


def test_disk_file_untouched(store):
    fill(store, 100)
    path = store._log_path(NAME)
    before = path.read_bytes()
    store.load_transcript_tail(NAME)
    assert path.read_bytes() == before


def test_env_overrides_read_at_import():
    env = {**os.environ, "OPENPOKE_EXEC_LOG_TAIL_ENTRIES": "7", "OPENPOKE_EXEC_LOG_MAX_CHARS": "999"}
    code = "from server.services.execution import log_store as l; print(l.EXEC_LOG_TAIL_ENTRIES, l.EXEC_LOG_MAX_CHARS)"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.split() == ["7", "999"], out.stderr


def test_agent_prompt_uses_bounded_history(monkeypatch):
    monkeypatch.setattr(ls, "EXEC_LOG_TAIL_ENTRIES", 3)
    from server.services.execution import get_execution_agent_logs

    logs = get_execution_agent_logs()
    fill(logs, 10)
    prompt = ExecutionAgent(NAME).build_system_prompt_with_history()
    assert "entry 9:" in prompt and "entry 6:" not in prompt
    assert "[Earlier history omitted: 7 entries." in prompt
