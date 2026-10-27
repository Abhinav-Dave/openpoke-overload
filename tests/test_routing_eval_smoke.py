"""Smoke tests for eval/routing_eval.py: data is valid, metrics are shaped right, a random chooser scores worse."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import routing_eval as ev  # noqa: E402


@pytest.fixture
def env():
    from server.agents.interaction_agent import agent as agent_mod
    from server.services.execution import get_agent_roster, get_execution_agent_logs, roster_view
    return agent_mod, roster_view, get_agent_roster(), get_execution_agent_logs()


@pytest.fixture(scope="module")
def data():
    return ev.load_data()


def small_set(agents, queries):
    """10 direct queries with known agents, padded to 20 agents."""
    qs = [q for q in queries if q["category"] == "direct"][:10]
    wanted = {q["expected"] for q in qs}
    rest = [a for a in agents if a["name"] not in wanted]
    return [a for a in agents if a["name"] in wanted] + rest[: 20 - len(wanted)], qs


def test_data_files_are_valid(data):
    agents, queries = data
    names = [a["name"] for a in agents]
    assert len(names) == len(set(names)) >= 150
    assert all(a["requests"] and 2 <= len(a["requests"]) <= 5 for a in agents)
    assert all(q["expected"] in names for q in queries if q["expected"])
    assert all(q["expected"] is None for q in queries if q["category"] in ("no_match", "offtopic"))
    counts = Counter(q["category"] for q in queries)
    assert set(counts) == set(ev.CATEGORIES) and min(counts.values()) >= 12
    for raw in ("routing_agents.json", "routing_queries.json"):
        assert json.loads((ev.DATA / raw).read_text(encoding="utf-8"))["synthetic"] is True


def test_splits_are_disjoint_and_sized(data):
    _, queries = data
    texts = [q["text"] for q in queries]
    assert len(texts) == len(set(texts))  # same text can never sit in both splits
    assert {q["split"] for q in queries} == {"dev", "test"}
    assert 0.3 < sum(q["split"] == "dev" for q in queries) / len(queries) < 0.5


def test_followup_recency_is_labelled(data):
    _, queries = data
    assert all(q["last_used"] == q["expected"] for q in queries if q["category"] == "followup")


def test_small_run_has_expected_keys(env, data):
    agents, queries = small_set(*data)
    ev.build(env[2], env[3], agents)
    m = ev.metrics(ev.evaluate(env, agents, queries))
    assert set(m) == {"n", "visible_recall", "false_no_match_rate", "no_match_correct_rate",
                      "shown_agents_mean", "block_chars_mean", "select_ms_mean"}
    assert m["n"] == 10 and m["block_chars_mean"] > 0
    base = ev.metrics(ev.evaluate(env, agents, queries, names_only=True))
    assert base["visible_recall"] == 1.0 and base["false_no_match_rate"] == 0.0


def test_random_chooser_scores_lower_than_real(env, data, monkeypatch):
    agents, queries = small_set(*data)
    ev.build(env[2], env[3], agents)
    real = ev.metrics(ev.evaluate(env, agents, queries))["visible_recall"]

    rng = random.Random(1)

    def random_pick(query, names, **kw):
        picked = rng.sample(names, 3)
        return [env[1].AgentView(n) for n in picked], len(names) - 3, False

    monkeypatch.setattr(env[1], "select_agents", random_pick)
    rand = ev.metrics(ev.evaluate(env, agents, queries))["visible_recall"]
    assert real - rand >= 0.3, (real, rand)
