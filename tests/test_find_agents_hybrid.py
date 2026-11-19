"""find_agents blends BM25 with embedding search by reciprocal rank fusion, only above FIND_EMBED_ABOVE agents."""
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from server.agents.interaction_agent.agent import prepare_message_with_history
from server.agents.interaction_agent.tools import handle_tool_call
from server.openrouter_client import OpenRouterError
from server.services.execution import embeddings, get_agent_roster, get_execution_agent_logs, roster_view

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ["invoices", "flights", "hotel", "taxes", "recipes", "visa", "podcast", "newsletter", "landlord", "gym"]
CLIENTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark"]
ABOVE = 11


@pytest.fixture(autouse=True)
def _embed_above_eleven(monkeypatch):
    # production blends above 1,000 agents; these tests use small rosters
    monkeypatch.setattr(roster_view, "FIND_EMBED_ABOVE", ABOVE)


def seed(n):
    """n topic agents plus two whose words differ from their names (n + 2 agents in total)."""
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    for i in range(n):
        name = f"{TOPICS[i % 10]}-{CLIENTS[i // 10]}"
        roster.add_agent(name)
        logs.record_request(name, f"Follow up on {TOPICS[i % 10]} for {CLIENTS[i // 10]}. Check the inbox.")
    roster.add_agent("misc-helper")
    logs.record_request("misc-helper", "Keep an eye on my sourdough starter feeding schedule")
    roster.add_agent("wellbeing-sessions")
    logs.record_request("wellbeing-sessions", "Book my weekly therapy sessions and keep notes")


def find(keywords):
    return [a.name for a in roster_view.find_agents(keywords, get_agent_roster().get_agents())]


def bm25_only(monkeypatch, keywords):
    monkeypatch.setattr(embeddings, "ENABLED", False)
    try:
        return roster_view.find_agents(keywords, get_agent_roster().get_agents())
    finally:
        monkeypatch.setattr(embeddings, "ENABLED", True)


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_first_seen_order():
    assert roster_view._rrf([0, 1, 2], [2, 0]) == [0, 2, 1]
    assert roster_view._rrf([0, 1], [1, 0]) == [0, 1]
    assert roster_view._rrf([1, 0], [0, 1]) == [1, 0]
    assert roster_view._rrf([], [3, 4]) == [3, 4]
    assert roster_view._rrf([5], []) == [5]
    assert roster_view._rrf([], []) == []


def test_no_embedding_call_at_the_threshold_and_one_call_above_it(fake_embedder):
    seed(ABOVE - 2)
    assert len(get_agent_roster().get_agents()) == ABOVE
    find("bread culture")
    assert fake_embedder.count == 0
    get_agent_roster().add_agent("one-more")
    find("bread culture")
    assert fake_embedder.count == 1


@pytest.mark.parametrize("keywords, expected", [("bread culture", "misc-helper"),
                                                ("chat with my counsellor", "wellbeing-sessions")])
def test_hybrid_finds_an_agent_that_shares_no_word_with_the_request(fake_embedder, monkeypatch, keywords, expected):
    seed(60)
    assert bm25_only(monkeypatch, keywords) == []
    assert find(keywords)[0] == expected


def test_a_tie_keeps_the_bm25_order(fake_embedder, monkeypatch):
    seed(60)
    bm25 = [a.name for a in bm25_only(monkeypatch, "invoices acme")]
    names = get_agent_roster().get_agents()
    top_two = [names.index(n) for n in bm25[:2]]
    monkeypatch.setattr(embeddings, "similar", lambda *a: list(reversed(top_two)))
    assert find("invoices acme") == bm25  # the top two tie on fused score; BM25 decides


@pytest.mark.parametrize("boom", ["openrouter", "timeout", "short"])
@pytest.mark.parametrize("keywords", ["bread culture", "invoices acme", "follow up inbox", "zzzqqq"])
def test_any_embedder_failure_gives_exactly_the_bm25_result(fake_embedder, monkeypatch, boom, keywords):
    seed(60)
    expected = bm25_only(monkeypatch, keywords)
    if boom == "short":
        monkeypatch.setattr(embeddings, "request_embeddings", lambda **kw: [[1.0]])
    else:
        fake_embedder.fail = OpenRouterError("x") if boom == "openrouter" else httpx.TimeoutException("slow")
    assert roster_view.find_agents(keywords, get_agent_roster().get_agents()) == expected


def test_nonsense_still_finds_nothing(fake_embedder):
    seed(60)
    res = handle_tool_call("find_agents", {"keywords": "zzzqqq"})
    assert res.success and res.payload["agents"] == []
    assert "no" in res.payload["message"].lower()
    assert fake_embedder.count == 1


def test_payload_shape_and_cap_of_five_are_unchanged(fake_embedder):
    seed(60)
    res = handle_tool_call("find_agents", {"keywords": "follow up inbox bread"})
    agents = res.payload["agents"]
    assert res.success and len(agents) == 5
    assert all(set(a) == {"name", "description"} for a in agents)
    assert "misc-helper" in [a["name"] for a in agents]  # the semantic hit made it into the five


def test_the_bounded_view_never_calls_the_embedder(fake_embedder, monkeypatch):
    monkeypatch.setattr(roster_view, "BOUND_ABOVE", ABOVE)
    seed(60)
    shown, hidden, _ = roster_view.select_agents("bread culture", get_agent_roster().get_agents())
    assert hidden > 0
    prepare_message_with_history("how is my bread culture doing", "", message_type="user")
    assert fake_embedder.count == 0


def test_without_the_fake_the_network_is_blocked_and_bm25_is_used(monkeypatch):
    monkeypatch.setattr(embeddings, "ENABLED", True)
    assert embeddings.similar("bread", ["a"], ["a: bread"]) is None
    embeddings._down_until = 0.0
    seed(60)
    assert find("sourdough starter") == ["misc-helper"]
    assert embeddings._down_until > 0  # the blocked call was really made, and it failed


def test_settings_come_from_the_environment(tmp_path):
    env = dict(os.environ, OPENPOKE_DATA_DIR=str(tmp_path), OPENPOKE_FIND_EMBED_ABOVE="7", OPENPOKE_EMBED="0",
               OPENPOKE_EMBED_MODEL="some/model", OPENPOKE_EMBED_MIN_SIM="0.42")
    code = ("import json; from server.services.execution import embeddings as e, roster_view as r; "
            "print(json.dumps([r.FIND_EMBED_ABOVE, e.ENABLED, e.MODEL, e.MIN_SIM, str(e._cache_path())]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    above, enabled, model, min_sim, path = json.loads(out.stdout.strip().splitlines()[-1])
    assert (above, enabled, model, min_sim) == (7, False, "some/model", 0.42)
    assert Path(path).is_relative_to(tmp_path)
