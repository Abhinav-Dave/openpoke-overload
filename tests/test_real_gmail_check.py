import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import real_gmail_check as rg  # noqa: E402


@pytest.mark.parametrize("tool", ["GMAIL_SEND_EMAIL", "GMAIL_CREATE_EMAIL_DRAFT", "GMAIL_REPLY_TO_THREAD",
                                  "GMAIL_FORWARD_MESSAGE", "GMAIL_DELETE_MESSAGE", "GMAIL_ADD_LABEL_TO_EMAIL"])
def test_write_tools_are_blocked_before_any_call(tool):
    called = []
    with pytest.raises(PermissionError):
        rg.read_only(lambda *a, **k: called.append(a), tool, "user")
    assert called == []


def test_read_tools_pass_through():
    assert rg.read_only(lambda t, u, arguments=None: (t, u), "GMAIL_FETCH_EMAILS", "u") == ("GMAIL_FETCH_EMAILS", "u")


def test_runs_old_vs_new():
    assert rg.runs_old_new(10, 0) == (10, 1)
    assert rg.runs_old_new(10, 2) == (10, 3)    # 2 urgent alone + 1 digest
    assert rg.runs_old_new(10, 10) == (10, 4)   # 3 urgent alone + 1 digest of the other 7
    assert rg.runs_old_new(3, 3) == (3, 3)      # nothing left for a digest
    assert rg.runs_old_new(0, 0) == (0, 0)


def test_messy_queries_skip_repeated_senders():
    emails = [{"sender": "Ann <a@x.com>", "subject": "Re: Hello", "clean_text": "hi"},
              {"sender": "Bob <b@x.com>", "subject": "Invoice 7", "clean_text": "pay"},
              {"sender": "Bob <b@x.com>", "subject": "Invoice 8", "clean_text": "pay"}]
    agents = rg.messy_agents(emails)
    assert len(agents) == 3
    labelled = {q["expected"] for q in rg.messy_queries(agents) if q["expected"]}
    assert labelled == {"Ann - Hello"}
