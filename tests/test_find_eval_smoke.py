"""Smoke tests for eval/find_eval.py and its 30 reworded questions. No network: the embedder is the fake."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import find_eval as fe  # noqa: E402
import routing_eval as ev  # noqa: E402

from server.services.execution import get_agent_roster, get_execution_agent_logs, roster_view  # noqa: E402


@pytest.fixture(scope="module")
def new_queries():
    return json.loads(fe.NEW_QUERIES.read_text(encoding="utf-8"))["queries"]


def test_the_30_new_questions_follow_the_rules(new_queries):
    agents, old = ev.load_data()
    names = {a["name"] for a in agents}
    old_paraphrase_agents = {q["expected"] for q in old if q["category"] == "paraphrase"}
    texts = [q["text"] for q in new_queries]
    assert len(new_queries) == 30
    assert len(set(texts)) == 30 and not set(texts) & {q["text"] for q in old}
    assert [q["split"] for q in new_queries] == ["dev" if i % 5 < 2 else "test" for i in range(30)]
    for q in new_queries:
        assert q["category"] == "paraphrase" and q["expected"] in names
        assert q["expected"] not in old_paraphrase_agents
        assert not roster_view._content_tokens(q["text"]) & set(roster_view.tokenize(q["expected"])), q


def test_question_groups_and_counts():
    qs = fe.questions()
    assert not [q for q in qs if q["category"] == "followup"]
    count = lambda g, split: sum(fe.group(q) == g and q["split"] == split for q in qs)
    assert count("paraphrase_old", "test") + count("paraphrase_new", "test") == 31
    assert count("direct+disambiguation", "test") == 32
    assert count("no_match+offtopic", "test") == 21


@pytest.mark.parametrize("name, expected, hit", [("a-b", "a-b", True), ("a-b-12", "a-b", True),
                                                 ("a-b-c", "a-b", False), ("a-bc", "a-b", False)])
def test_is_hit_accepts_numbered_clones_only(name, expected, hit):
    assert fe.is_hit(name, expected) is hit


def small_roster():
    agents, _ = ev.load_data()
    keep = {"therapy-session-scheduling", "invoice-followups-acme", "passport-renewal"}
    picked = [a for a in agents if a["name"] in keep] + [a for a in agents if a["name"] not in keep][:17]
    ev.build(get_agent_roster(), get_execution_agent_logs(), picked)
    return get_agent_roster().get_agents()


QUESTIONS = [
    {"text": "Set up my weekly chat with my counsellor", "expected": "therapy-session-scheduling",
     "category": "paraphrase", "split": "test", "source": "new"},
    {"text": "Did Acme pay invoice 1042?", "expected": "invoice-followups-acme", "category": "direct",
     "split": "test", "source": "old"},
    {"text": "zzzqqq", "expected": None, "category": "no_match", "split": "dev", "source": "old"},
]


def test_small_run_with_the_fake_reports_both_modes(fake_embedder, monkeypatch):
    names = small_roster()
    monkeypatch.setattr(roster_view, "FIND_EMBED_ABOVE", 0)
    bm25 = fe.run(QUESTIONS, names, hybrid=False)
    hybrid = fe.run(QUESTIONS, names, hybrid=True)
    for rows in (bm25, hybrid):
        assert [set(r) for r in rows] == [{"group", "split", "hit1", "hit5", "empty", "ms", "fell_back"}] * 3
        assert rows[2]["hit1"] is None and rows[2]["empty"] is True
        assert rows[1]["hit1"] is True
    assert not any(r["fell_back"] for r in bm25 + hybrid)
    assert hybrid[0]["hit1"] is True  # the fake knows counsellor ~ therapy
    s = fe.summarize(hybrid)
    assert s["fallbacks"] == 0 and s["paraphrase_new"]["test"]["hit1"] == 1 and s["paraphrase_all"]["test"]["n"] == 1
    assert fake_embedder.count == 3


def test_a_run_whose_embedder_fails_is_invalid_and_not_written(monkeypatch, tmp_path):
    names = small_roster()
    monkeypatch.setattr(roster_view, "FIND_EMBED_ABOVE", 0)
    rows = fe.run(QUESTIONS, names, hybrid=True)  # conftest blocks the network, so every call fails
    assert all(r["fell_back"] for r in rows)
    out = tmp_path / "find_eval.json"
    assert fe.save({"x": 1}, rows, out) is False and not out.exists()
    assert fe.save({"x": 1}, [dict(r, fell_back=False) for r in rows], out) is True and out.exists()


def test_hybrid_below_the_threshold_counts_as_a_fallback(fake_embedder):
    names = small_roster()  # 20 agents, default threshold 1,000: the embedder never runs
    assert all(r["fell_back"] for r in fe.run(QUESTIONS, names, hybrid=True))
    assert fake_embedder.count == 0


def s(v, para, direct, empty):
    return {"min_sim": v, "para_hit5": para, "direct_hit1": direct, "empty": empty}


BM25 = {"para_hit5": 0.5, "direct_hit1": 0.9, "empty": 0.4}


@pytest.mark.parametrize("sweep, chosen", [
    ([s(0.2, 0.8, 0.9, 0.4), s(0.3, 0.7, 0.9, 0.4)], 0.2),  # best paraphrase recall wins
    ([s(0.2, 0.8, 0.9, 0.4), s(0.3, 0.8, 0.9, 0.4)], 0.3),  # tie: the higher MIN_SIM
    ([s(0.2, 0.9, 0.85, 0.4), s(0.3, 0.6, 0.9, 0.4)], 0.3),  # 0.2 loses direct hit@1
    ([s(0.2, 0.9, 0.9, 0.29), s(0.3, 0.6, 0.9, 0.3)], 0.3),  # 0.2 loses more than 10 points of empty rate
    ([s(0.2, 0.9, 0.8, 0.4), s(0.3, 0.9, 0.9, 0.2)], None),  # nothing qualifies
    ([], None),
])
def test_pick_min_sim(sweep, chosen):
    assert fe.pick_min_sim(sweep, BM25) == chosen


@pytest.mark.parametrize("hybrid, on", [
    ({"para_hit5": 23, "direct_hit1": 29, "empty": 4}, True),  # +3, -1, -2: every limit exactly met
    ({"para_hit5": 22, "direct_hit1": 30, "empty": 6}, False),  # only +2 on paraphrase
    ({"para_hit5": 25, "direct_hit1": 28, "empty": 6}, False),  # loses 2 direct hits
    ({"para_hit5": 25, "direct_hit1": 30, "empty": 3}, False),  # 3 fewer empty no-match answers
])
def test_keep_on_applies_the_default_rule(hybrid, on):
    assert fe.keep_on({"para_hit5": 20, "direct_hit1": 30, "empty": 6}, hybrid) is on
