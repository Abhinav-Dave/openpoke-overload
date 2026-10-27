"""Tests for eval/live_eval.py. No network: the model is a scripted httpx.MockTransport."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import live_eval as le  # noqa: E402

SECRET = "sk-or-SECRET-do-not-print-123"
SEEDED = {"a-agent", "b-agent"}


def tc(tool, agent=None):
    return (tool, agent)


def row(**kw):
    base = dict(tree="new", size=150, repeat=1, qid=0, category="direct", expected="a", outcome="reused_expected",
                agent="a", find_called=False, find_before_choice=False, calls=2, prompt_tokens=1000,
                completion_tokens=50, roster_size=150, error=None)
    return {**base, **kw}


# ---- classification --------------------------------------------------------
def test_classify_four_outcomes_and_edge_cases():
    c = lambda calls, exp="a-agent": le.classify(calls, SEEDED, exp)["outcome"]
    assert c([tc("send_message_to_agent", "a-agent")]) == "reused_expected"
    assert c([tc("send_message_to_agent", "b-agent")]) == "reused_other"
    assert c([tc("send_message_to_agent", "brand-new")]) == "created_new"
    assert c([tc("send_message_to_user")]) == "no_agent_call"
    assert c([]) == "no_agent_call"
    # expected=None: reusing an existing agent is the "other" outcome
    assert c([tc("send_message_to_agent", "a-agent")], None) == "reused_other"
    # first agent call is the decision; an invalid call (no name) is ignored
    assert c([tc("send_message_to_agent", None), tc("send_message_to_agent", "a-agent")]) == "reused_expected"
    assert c([tc("send_message_to_agent", "b-agent"), tc("send_message_to_agent", "a-agent")]) == "reused_other"


def test_clone_of_expected_agent_counts_as_reused_expected():
    seeded = SEEDED | {"a-agent-2", "a-agent-3", "a-agent-extra"}
    c = lambda name: le.classify([tc("send_message_to_agent", name)], seeded, "a-agent")["outcome"]
    assert c("a-agent-2") == "reused_expected" and c("a-agent-3") == "reused_expected"
    assert c("a-agent-extra") == "reused_other"      # not a clone suffix


def test_followup_history_has_user_request_and_agent_report():
    from server.services.conversation import get_conversation_log

    log = get_conversation_log()
    le.seed_followup_history(log, "a-agent")
    transcript = log.load_transcript()
    assert "[SUCCESS] a-agent: Draft ready for your review." in transcript
    assert transcript.index("user_message") < transcript.index("agent_message")


def test_classify_find_agents_order():
    r = le.classify([tc("find_agents"), tc("send_message_to_agent", "a-agent")], SEEDED, "a-agent")
    assert r["find_called"] and r["find_before_choice"]
    r = le.classify([tc("send_message_to_agent", "a-agent"), tc("find_agents")], SEEDED, "a-agent")
    assert r["find_called"] and not r["find_before_choice"]
    r = le.classify([tc("find_agents")], SEEDED, "a-agent")
    assert r["find_called"] and not r["find_before_choice"] and r["outcome"] == "no_agent_call"


# ---- summary arithmetic ----------------------------------------------------
def test_percentile_nearest_rank():
    assert le.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert le.percentile([3], 95) == 3
    assert le.percentile([], 95) is None


def test_summarize_rates():
    rows = [
        row(qid=0, outcome="reused_expected", calls=2, prompt_tokens=1000),
        row(qid=1, outcome="created_new", agent="x", calls=2, prompt_tokens=2000),     # duplicate
        row(qid=2, outcome="reused_other", agent="z", calls=4, prompt_tokens=3000, find_called=True,
            find_before_choice=True),
        row(qid=3, outcome="no_agent_call", expected=None, agent=None, category="offtopic", calls=1, prompt_tokens=400),
        row(qid=4, outcome="created_new", expected=None, agent="y", category="offtopic", calls=3, prompt_tokens=600),
        row(qid=5, outcome="reused_other", expected=None, agent="z", category="offtopic", calls=2, prompt_tokens=0),
    ]
    s = le.summarize(rows)["overall"]
    assert s["n"] == 6
    assert s["counts"] == {"reused_expected": 1, "reused_other": 2, "created_new": 2, "no_agent_call": 1, "error": 0}
    assert s["correct_rate"] == round(3 / 6, 3)          # 1 expected + 2 correct negatives
    assert s["duplicate_rate"] == round(1 / 3, 3)        # over the 3 rows with an expected agent
    assert s["wrong_reuse_rate"] == round(2 / 6, 3)
    assert s["prompt_tokens_mean"] == round(7000 / 6)
    assert s["calls_mean"] == round(14 / 6, 2)
    assert s["loop_p95"] == 4
    assert s["find_called_rate"] == round(1 / 6, 3)
    cat = le.summarize(rows)["by_category"]
    assert cat["offtopic"]["n"] == 3 and cat["offtopic"]["duplicate_rate"] is None
    assert cat["offtopic"]["correct_rate"] == round(2 / 3, 3)


def test_error_rows_count_as_not_correct():
    s = le.summarize([row(outcome="error", agent=None, error="boom"), row()])["overall"]
    assert s["counts"]["error"] == 1 and s["correct_rate"] == 0.5


def test_consistency_across_repeats():
    rows = [row(qid=0, repeat=1), row(qid=0, repeat=2),
            row(qid=1, repeat=1), row(qid=1, repeat=2, outcome="created_new")]
    assert le.consistency(rows) == 0.5
    assert le.consistency([row(qid=0, repeat=1)]) is None


# ---- selection and plan ----------------------------------------------------
def test_select_queries_limit_and_split():
    agents, queries = le.load_data()
    allq = le.select_queries(queries, agents, "all", None)
    test = le.select_queries(queries, agents, "test", None)
    assert 0 < len(test) < len(allq)
    assert all(q["split"] == "test" for q in test)
    one = le.select_queries(queries, agents, "test", 1)
    assert len(one) == len({q["category"] for q in one}) == 6
    assert all("qid" in q for q in one)


def test_dry_run_makes_no_request_and_prints_plan(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("network/subprocess used in dry run")
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    assert le.main([]) == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert "DRY RUN" in text and "worst-case" in text and "USD" in text
    assert "orig" in text and "new" in text and SECRET not in text


def test_run_requires_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_EVAL_FAKE_HTTP", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("spawned without a key"))
    assert le.main(["--run", "--out", str(tmp_path / "o.json")]) == 2


# ---- original tree ---------------------------------------------------------
def test_extract_original_tree(tmp_path):
    le.extract_original(tmp_path)
    assert (tmp_path / "server" / "agents" / "interaction_agent" / "runtime.py").exists()
    assert not (tmp_path / "server" / "services" / "execution" / "roster_view.py").exists()  # new code absent
    assert "fcntl" in (tmp_path / "server" / "services" / "execution" / "roster.py").read_text(encoding="utf-8")


# ---- stubs, in process, with the real runtime ------------------------------
def make_policy(steps):
    return {"steps": steps}


def test_stubs_count_usage_block_gmail_and_start_nothing(monkeypatch, tmp_path):
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.services.execution import get_agent_roster
    import server.services.gmail.client as gmail_client

    roster = get_agent_roster()
    roster.add_agent("a-agent")
    policy = make_policy({"hello there": [{"tool": "send_message_to_agent",
                                           "args": {"agent_name": "a-agent", "instructions": "do it"}}]})
    transport = le.fake_transport(policy)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(*a, transport=transport, **k))
    state = le.RunState()
    undo = le.install_stubs(state)
    try:
        res = asyncio.run(_run(InteractionAgentRuntime(), "hello there"))
        assert res.success
        assert state.calls == 2                              # tool call, then final text
        assert state.prompt_tokens > 0 and state.completion_tokens > 0
        assert state.tool_calls == [("send_message_to_agent", "a-agent")]
        assert state.starts == ["a-agent"]                   # recorded, nothing really started
        assert state.usage_missing == 0
        with pytest.raises(RuntimeError):
            gmail_client.execute_gmail_tool("GMAIL_FETCH_EMAILS", "u", {})
        assert state.gmail_calls == 1
    finally:
        undo()
    assert not hasattr(gmail_client.execute_gmail_tool, "_live_eval")


async def _run(rt, text):
    res = await rt.execute(text)
    await asyncio.sleep(0.05)
    return res


# ---- end to end through real subprocess workers ---------------------------
def e2e_policy(monkeypatch, tmp_path, tree_note=""):
    agents, queries = le.load_data()
    names = [a["name"] for a in agents]
    chosen = le.select_queries(queries, agents, "test", 1)
    steps = {}
    for q in chosen:
        exp, cat = q["expected"], q["category"]
        other = next(n for n in names if n != exp)
        if cat == "direct":
            s = [{"tool": "send_message_to_agent", "args": {"agent_name": exp, "instructions": "x"}}]
        elif cat == "paraphrase":
            s = [{"tool": "send_message_to_agent", "args": {"agent_name": "brand-new-agent", "instructions": "x"}}]
        elif cat == "disambiguation":
            s = [{"tool": "send_message_to_agent", "args": {"agent_name": other, "instructions": "x"}}]
        elif cat == "followup":
            s = [{"tool": "find_agents", "args": {"keywords": "k"}},
                 {"tool": "send_message_to_agent", "args": {"agent_name": exp, "instructions": "x"}}]
        elif cat == "no_match":
            s = []
        else:
            s = [{"tool": "send_message_to_agent", "args": {"agent_name": "another-new-agent", "instructions": "x"}}]
        steps[q["text"]] = s
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"steps": steps, "key_usage_start": 1.0, "key_usage_step": 0.25}), encoding="utf-8")
    monkeypatch.setenv("LIVE_EVAL_FAKE_HTTP", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    return chosen, len(names)


def test_end_to_end_smoke_both_trees(monkeypatch, tmp_path, capfd):
    chosen, n_agents = e2e_policy(monkeypatch, tmp_path)
    out = tmp_path / "live.json"
    code = le.main(["--run", "--limit", "1", "--repeats", "2", "--trees", "orig,new", "--out", str(out)])
    cap = capfd.readouterr()
    assert code == 0, cap.out + cap.err
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert SECRET not in text and SECRET not in cap.out and SECRET not in cap.err
    assert "account spend" in cap.out.lower()
    assert data["stopped_early"] is False
    for tree in ("orig", "new"):
        rows = [r for r in data["rows"] if r["tree"] == tree]
        assert len(rows) == 6 * 2
        # roster restored between queries: always the seeded size, even after created_new rows
        assert {r["roster_size"] for r in rows} == {n_agents}
        got = {(r["category"]): r["outcome"] for r in rows if r["repeat"] == 1}
        assert got == {"direct": "reused_expected", "paraphrase": "created_new", "disambiguation": "reused_other",
                       "followup": "reused_expected", "no_match": "no_agent_call", "offtopic": "created_new"}
        fu = next(r for r in rows if r["category"] == "followup")
        assert fu["find_called"] and fu["find_before_choice"]
        s = data["summary"][tree]["150"]
        assert s["overall"]["n"] == 12 and s["consistency"] == 1.0
        assert s["overall"]["correct_rate"] == round(4 / 6, 3)   # direct, followup, no_match, offtopic
    assert data["gmail_calls"] == 0
    assert data["spent_estimate_usd"] > 0
    assert data["account_spend_delta_usd"] == 0.25


def test_budget_stop_saves_partial(monkeypatch, tmp_path, capfd):
    e2e_policy(monkeypatch, tmp_path)
    out = tmp_path / "live.json"
    code = le.main(["--run", "--limit", "1", "--repeats", "2", "--trees", "new", "--max-usd", "0.01",
                    "--price-in", "50", "--price-out", "50", "--out", str(out)])
    cap = capfd.readouterr()
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stopped_early"] is True
    assert 1 <= len(data["rows"]) < 12
    assert "STOPPED" in cap.out and SECRET not in cap.out + cap.err + out.read_text(encoding="utf-8")
