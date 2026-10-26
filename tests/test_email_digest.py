import datetime as dt
import importlib
import json
import logging
import os

import pytest

from server.services.gmail import importance_classifier as classifier_mod
from server.services.gmail import importance_watcher as watcher_mod
from server.services.gmail.importance_classifier import ClassificationFailed, EmailVerdict
from server.services.gmail.importance_watcher import ImportantEmailWatcher
from server.services.gmail.seen_store import GmailSeenStore


def _raw(i):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {"messageId": f"m{i}", "threadId": f"t{i}", "subject": f"Subject {i}",
            "sender": f"p{i}@example.com", "to": "me@example.com", "messageTimestamp": now,
            "messageText": f"body {i}", "labelIds": ["INBOX"]}


class Harness:
    """Watcher wired to a scripted classifier, a fake inbox and a recording interaction runtime."""

    def __init__(self, tmp_path, monkeypatch):
        self.inbox = []
        self.verdicts = {}      # email id -> EmailVerdict | None | "fail"
        self.classified = []
        self.sent = []
        self.watcher = ImportantEmailWatcher(seen_store=GmailSeenStore(tmp_path / "seen.json", 300))
        self.watcher._has_seeded_initial_snapshot = True

        async def classify(email):
            self.classified.append(email.id)
            v = self.verdicts.get(email.id, EmailVerdict("needs a reply"))
            if v == "fail":
                raise ClassificationFailed("boom")
            return v

        outer = self

        class Runtime:
            async def handle_agent_message(self, text):
                outer.sent.append(text)

        monkeypatch.setattr(watcher_mod, "classify_email_importance", classify)
        monkeypatch.setattr(watcher_mod, "_resolve_interaction_runtime", lambda: Runtime())
        monkeypatch.setattr(watcher_mod, "get_active_gmail_user_id", lambda *a, **k: "u")
        monkeypatch.setattr(watcher_mod, "execute_gmail_tool",
                            lambda *a, **k: {"data": {"messages": [dict(m) for m in self.inbox]}})

    def fill(self, n, start=0):
        self.inbox = [_raw(i) for i in range(start, start + n)]

    async def poll(self):
        await self.watcher._poll_once()


@pytest.fixture
def h(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


async def test_50_important_emails_make_one_digest(h):
    h.fill(50)
    await h.poll()
    assert len(h.classified) == 50
    assert len(h.sent) == 1
    assert "Subject 0" in h.sent[0]


async def test_urgent_go_immediately_and_rest_in_one_digest(h):
    h.fill(12)
    h.verdicts = {"m0": EmailVerdict("code 123", urgent=True), "m1": EmailVerdict("call me", urgent=True)}
    await h.poll()
    assert len(h.sent) == 3
    assert "code 123" in h.sent[0] and "call me" in h.sent[1]
    assert h.sent[2].count("\n- ") == 10          # 10 normal items in the digest


async def test_urgent_overflow_goes_into_digest(h):
    h.fill(5)
    h.verdicts = {f"m{i}": EmailVerdict(f"urgent {i}", urgent=True) for i in range(5)}
    await h.poll()
    assert len(h.sent) == watcher_mod.MAX_URGENT + 1
    digest = h.sent[-1]
    assert "urgent 3" in digest and "urgent 4" in digest
    assert "urgent 0" not in digest


async def test_digest_item_cap_and_and_n_more(h):
    h.fill(25)
    await h.poll()
    assert len(h.sent) == 1
    assert h.sent[0].count("\n- ") == watcher_mod.MAX_ITEMS == 20
    assert h.sent[0].rstrip().endswith("and 5 more")


async def test_single_non_urgent_keeps_plain_format(h):
    h.fill(1)
    await h.poll()
    assert h.sent == ["Important email watcher notification:\nneeds a reply"]


async def test_no_important_mail_dispatches_nothing(h):
    h.fill(4)
    h.verdicts = {f"m{i}": None for i in range(4)}
    await h.poll()
    assert h.sent == []
    assert all(h.watcher._seen_store.is_seen(f"m{i}") for i in range(4))


async def test_classifier_failure_leaves_unseen_and_retries(h):
    h.fill(1)
    h.verdicts = {"m0": "fail"}
    await h.poll()
    assert not h.watcher._seen_store.is_seen("m0")
    assert h.sent == []

    h.verdicts = {}
    await h.poll()                                  # retried even though it is now older than the poll cutoff
    assert h.classified == ["m0", "m0"]
    assert len(h.sent) == 1
    assert h.watcher._seen_store.is_seen("m0")


async def test_gives_up_after_max_attempts_and_warns(h, caplog):
    h.fill(1)
    h.verdicts = {"m0": "fail"}
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            await h.poll()
    assert len(h.classified) == 3
    assert h.watcher._seen_store.is_seen("m0")
    assert any("giving up" in r.getMessage().lower() for r in caplog.records)
    await h.poll()
    assert len(h.classified) == 3                   # no infinite retry


async def test_mixed_poll_failures_do_not_block_the_rest(h):
    h.fill(6)
    h.verdicts = {"m1": "fail", "m4": "fail"}
    await h.poll()
    assert len(h.sent) == 1 and h.sent[0].count("\n- ") == 4
    seen = h.watcher._seen_store
    assert [seen.is_seen(f"m{i}") for i in range(6)] == [True, False, True, True, False, True]


async def test_cap_hit_warning(h, caplog):
    h.fill(watcher_mod.DEFAULT_MAX_RESULTS)
    with caplog.at_level(logging.WARNING):
        await h.poll()
    assert any("max_results" in r.getMessage() for r in caplog.records)


async def test_no_cap_warning_below_cap(h, caplog):
    h.fill(3)
    with caplog.at_level(logging.WARNING):
        await h.poll()
    assert not any("max_results" in r.getMessage() for r in caplog.records)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENPOKE_DIGEST_MAX_URGENT", "1")
    monkeypatch.setenv("OPENPOKE_DIGEST_MAX_ITEMS", "7")
    monkeypatch.setenv("OPENPOKE_CLASSIFY_MAX_ATTEMPTS", "5")
    try:
        mod = importlib.reload(watcher_mod)
        assert (mod.MAX_URGENT, mod.MAX_ITEMS, mod.CLASSIFY_MAX_ATTEMPTS) == (1, 7, 5)
    finally:
        monkeypatch.undo()
        importlib.reload(watcher_mod)


# ---- classifier ------------------------------------------------------------
def _email():
    from server.services.gmail.processing import ProcessedEmail
    return ProcessedEmail("x", None, "q", "s", "a@b.c", "me", dt.datetime.now(dt.timezone.utc),
                          [], "text", False, 0, [])


def _tool_response(args):
    return {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "mark_email_importance", "arguments": json.dumps(args)}}]}}]}


async def test_classifier_reads_urgent_flag(fake_llm):
    fake_llm.script = [_tool_response({"important": True, "summary": "S", "urgent": True})]
    assert await classifier_mod.classify_email_importance(_email()) == EmailVerdict("S", True)
    fake_llm.script = [_tool_response({"important": True, "summary": "S"})]
    assert await classifier_mod.classify_email_importance(_email()) == EmailVerdict("S", False)
    fake_llm.script = [_tool_response({"important": False})]
    assert await classifier_mod.classify_email_importance(_email()) is None


async def test_classifier_raises_on_api_error_and_missing_tool_call(fake_llm):
    from server.openrouter_client import OpenRouterError

    def err(kwargs):
        raise OpenRouterError("down")

    fake_llm.script = [err]
    with pytest.raises(ClassificationFailed):
        await classifier_mod.classify_email_importance(_email())
    fake_llm.script = [{"choices": [{"message": {"content": "hi"}}]}]
    with pytest.raises(ClassificationFailed):
        await classifier_mod.classify_email_importance(_email())


# ---- seen store ------------------------------------------------------------
def test_seen_store_crash_during_replace_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "seen.json"
    store = GmailSeenStore(path, 300)
    store.mark_seen(["a"])
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    store.mark_seen(["b"])
    assert path.read_bytes() == before


def test_seen_store_leaves_no_tmp_and_keeps_list_format_and_cap(tmp_path):
    path = tmp_path / "seen.json"
    store = GmailSeenStore(path, 300)
    store.mark_seen([f"id{i}" for i in range(350)])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == [f"id{i}" for i in range(50, 350)]
    assert [p.name for p in tmp_path.iterdir()] == ["seen.json"]
    assert GmailSeenStore(path, 300).snapshot() == data
