import pytest

from server.agents.interaction_agent.agent import prepare_message_with_history
from server.services.execution import get_agent_roster, get_execution_agent_logs, limits, roster_view

LABEL = "Closest existing agents (searched for you, none is a confident match)"
TOPICS = ["invoices", "flights", "hotel", "taxes", "recipes", "visa", "podcast", "newsletter", "landlord", "gym"]
CLIENTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka", "soylent", "tyrell"]


@pytest.fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    monkeypatch.setattr(roster_view, "BOUND_ABOVE", 11)


@pytest.fixture
def turn():
    token = limits.start_turn()
    yield
    limits.end_turn(token)


def seed(n):
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    for i in range(n):
        name = f"{TOPICS[i % 10]}-{CLIENTS[(i // 10) % 10]}" + (f"-{i // 100 + 1}" if i >= 100 else "")
        roster.add_agent(name)
        logs.record_request(name, f"Follow up on {TOPICS[i % 10]} for {CLIENTS[(i // 10) % 10]}. Check the inbox.")


def prompt(text):
    return prepare_message_with_history(text, "")[0]["content"]


# no_match (best score under MIN_SCORE) that still has weak hits on "inbox"
NEAR_MISS = "inbox zzzunseen"


def test_no_match_adds_labelled_hits_and_marks_find_flag(turn):
    seed(30)
    content = prompt(NEAR_MISS)
    assert "No listed agent fits" in content
    line = next(l for l in content.splitlines() if l.startswith(LABEL))
    assert 1 <= line.count(" (Follow up") <= 5
    assert not limits.find_agents_missing()


def test_flag_marked_even_when_prompt_is_built_before_the_turn_starts():
    seed(30)
    assert LABEL in prompt(NEAR_MISS)  # runtime builds the prompt, then calls start_turn
    token = limits.start_turn()
    try:
        assert not limits.find_agents_missing()
    finally:
        limits.end_turn(token)


def test_confident_match_has_no_block_and_flag_unmarked(turn):
    seed(30)
    assert LABEL not in prompt("invoices for acme")
    assert limits.find_agents_missing()


def test_small_roster_prompt_unchanged(turn, monkeypatch):
    seed(8)
    monkeypatch.setattr(roster_view, "BOUND_ABOVE", 200)
    content = prompt(NEAR_MISS)
    assert LABEL not in content and "No listed agent fits" not in content
    assert limits.find_agents_missing()


def test_vague_message_does_not_search(turn):
    seed(30)
    assert LABEL not in prompt("send it")
    assert limits.find_agents_missing()


def test_no_match_with_zero_hits_has_no_empty_block_but_marks_flag(turn):
    seed(30)
    content = prompt("tell me a joke about whales")
    assert "No listed agent fits" in content
    assert LABEL not in content
    assert not limits.find_agents_missing()
