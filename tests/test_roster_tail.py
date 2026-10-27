import os
import subprocess
import sys
from html import escape

from server.agents.interaction_agent.agent import prepare_message_with_history
from server.agents.interaction_agent.tools import TOOL_SCHEMAS, get_tool_schemas
from server.services.execution import get_agent_roster, get_execution_agent_logs, roster_view

BOUND = 200


def seed(n, names=None):
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    made = []
    for i in range(n):
        name = names[i] if names else f"topic{i}-client{i}"
        roster.add_agent(name)
        logs.record_request(name, f"Handle topic{i} for client{i}.")
        os.utime(logs._log_path(name), (1_000_000 + i, 1_000_000 + i))  # higher i = more recent
        made.append(name)
    return made


def block(text="hello"):
    c = prepare_message_with_history(text, "", message_type="user")[0]["content"]
    return c.split("<active_agents>\n", 1)[1].split("\n</active_agents>", 1)[0]


def tail_of(b):
    return next(l for l in b.split("\n") if l.startswith("Other agents"))


def shown_names(b):
    return {l.split('"')[1] for l in b.split("\n") if l.startswith("<agent name=")}


def tool_names():
    return [s["function"]["name"] for s in get_tool_schemas()]


def test_defaults():
    assert roster_view.BOUND_ABOVE == 200
    assert roster_view.TAIL_NAMES == 200


def test_up_to_threshold_is_byte_identical_to_original_block():
    made = seed(BOUND)
    legacy = "\n".join(f'<agent name="{escape(n, quote=True)}" />' for n in made)
    assert block("topic3 client3") == legacy
    assert get_tool_schemas() == TOOL_SCHEMAS


def test_above_threshold_is_bounded_and_tail_names_every_other_agent():
    made = seed(BOUND + 1)
    b = block("topic3 client3")
    shown = shown_names(b)
    tail = tail_of(b)
    assert "topic3-client3" in shown and len(shown) < len(made)
    assert "find_agents" in b
    for n in set(made) - shown:
        assert n in tail
    assert "more (use find_agents)" not in tail  # nothing truncated at 201


def test_tail_is_most_recent_first():
    made = seed(BOUND + 1)
    b = block("zebra giraffe")
    tail = tail_of(b)
    rest = [n for n in made if n not in shown_names(b)]
    expected = sorted(rest, key=lambda n: -int(n.split("client")[1]))
    assert tail.split(": ", 1)[1].split(", ") == expected


def test_tail_cap_adds_and_n_more(monkeypatch):
    seed(BOUND + 1)
    monkeypatch.setattr(roster_view, "TAIL_NAMES", 5)
    b = block("zebra giraffe")
    tail = tail_of(b)
    remaining = BOUND + 1 - len(shown_names(b)) - 5
    assert tail.count(",") == 4
    assert tail.endswith(f"and {remaining} more (use find_agents)")


def test_names_are_escaped_in_tail_and_tags():
    names = [f'q"<b>&{i}' for i in range(BOUND + 1)]
    seed(BOUND + 1, names)
    b = block("hello")
    assert "<b>" not in b and 'q"' not in b
    assert escape(names[0], quote=True) in b


def test_find_agents_exposed_only_above_threshold(monkeypatch):
    seed(BOUND + 1)
    assert "find_agents" in tool_names()
    monkeypatch.setattr(roster_view, "BOUND_ABOVE", BOUND + 1)
    assert get_tool_schemas() == TOOL_SCHEMAS
    assert "more agents are not shown" not in block()


def test_env_overrides():
    env = dict(os.environ, OPENPOKE_ROSTER_BOUND_ABOVE="12", OPENPOKE_ROSTER_TAIL_NAMES="7")
    code = "from server.services.execution import roster_view as r; print(r.BOUND_ABOVE, r.TAIL_NAMES)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.split() == ["12", "7"], out.stderr
