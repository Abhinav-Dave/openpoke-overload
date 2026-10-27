from html import escape

from server.agents.interaction_agent.agent import prepare_message_with_history
from server.services.execution import get_agent_roster, get_execution_agent_logs

TOPICS = ["invoices", "flights", "hotel", "taxes", "recipes", "visa", "podcast", "newsletter", "landlord", "gym"]
CLIENTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka", "soylent", "tyrell"]


def seed(n):
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    made = []
    for i in range(n):
        name = f"{TOPICS[i % 10]}-{CLIENTS[(i // 10) % 10]}" + (f"-{i // 100 + 1}" if i >= 100 else "")
        roster.add_agent(name)
        logs.record_request(name, f"Follow up on {TOPICS[i % 10]} for {CLIENTS[(i // 10) % 10]}. Check the inbox and report back.")
        made.append(name)
    return made


def agents_block(content):
    return content.split("<active_agents>\n", 1)[1].split("\n</active_agents>", 1)[0]


def prompt(text, kind="user"):
    return prepare_message_with_history(text, "", message_type=kind)[0]["content"]


def test_small_roster_block_is_identical_to_legacy_format():
    made = seed(11)
    legacy = "\n".join(f'<agent name="{escape(n, quote=True)}" />' for n in made)
    assert agents_block(prompt("hello")) == legacy


def test_empty_roster_says_none():
    assert agents_block(prompt("hello")) == "None"


def test_big_roster_block_is_much_smaller_than_legacy(capsys):
    made = seed(300)
    legacy = "\n".join(f'<agent name="{escape(n, quote=True)}" />' for n in made)
    block = agents_block(prompt("any news on the invoices for acme"))
    print(f"\nlegacy block {len(legacy)} chars, new block {len(block)} chars, ratio {len(block) / len(legacy):.2f}")
    assert len(block) < 0.6 * len(legacy)  # every name stays visible in the compact tail
    assert "more agents are not shown" in block
    assert "invoices-acme" in block


def test_no_match_tells_the_model_to_create_a_new_agent():
    seed(300)
    block = agents_block(prompt("tell me a joke about whales"))
    assert "No listed agent fits" in block
    assert "more agents are not shown" in block


def test_matching_query_has_no_no_match_line():
    seed(300)
    assert "No listed agent fits" not in agents_block(prompt("invoices for acme"))


def test_agent_result_turn_shows_the_agents_named_in_the_payload():
    made = seed(300)
    named = made[57]
    block = agents_block(prompt(f"[SUCCESS] {named}: done", kind="agent"))
    assert f'name="{named}"' in block


@__import__("pytest").fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    # these tests exercise the bounded view on small rosters; production activates it above 200 agents
    from server.services.execution import roster_view as _rv

    monkeypatch.setattr(_rv, "BOUND_ABOVE", 11)
