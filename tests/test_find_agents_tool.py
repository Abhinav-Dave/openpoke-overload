import asyncio
import json

import pytest

from server.agents.interaction_agent import tools as tools_mod
from server.agents.interaction_agent.agent import prepare_message_with_history
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import TOOL_SCHEMAS, get_tool_schemas, handle_tool_call
from server.agents.execution_agent.runtime import ExecutionResult
from server.services.execution import get_agent_roster, get_execution_agent_logs, limits

OLD_TOOL_NAMES = ["send_message_to_agent", "send_message_to_user", "send_draft", "wait"]
TOPICS = ["invoices", "flights", "hotel", "taxes", "recipes", "visa", "podcast", "newsletter", "landlord", "gym"]
CLIENTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark"]


def seed(n):
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    for i in range(n):
        name = f"{TOPICS[i % 10]}-{CLIENTS[i // 10]}"
        roster.add_agent(name)
        logs.record_request(name, f"Follow up on {TOPICS[i % 10]} for {CLIENTS[i // 10]}. Check the inbox.")
    # one agent whose words differ from its name
    roster.add_agent("misc-helper")
    logs.record_request("misc-helper", "Keep an eye on my sourdough starter feeding schedule")


def names_of(schemas):
    return [s["function"]["name"] for s in schemas]


def test_small_roster_tool_list_is_unchanged():
    seed(5)
    assert names_of(get_tool_schemas()) == OLD_TOOL_NAMES
    assert get_tool_schemas() == TOOL_SCHEMAS


def test_empty_roster_tool_list_is_unchanged():
    assert names_of(get_tool_schemas()) == OLD_TOOL_NAMES


def test_big_roster_adds_find_agents():
    seed(60)
    assert names_of(get_tool_schemas()) == OLD_TOOL_NAMES + ["find_agents"]
    assert names_of(TOOL_SCHEMAS) == OLD_TOOL_NAMES  # module list is not mutated


def test_find_agents_returns_agent_by_its_words():
    seed(60)
    res = handle_tool_call("find_agents", {"keywords": "sourdough starter"})
    assert res.success
    agents = res.payload["agents"]
    assert agents[0]["name"] == "misc-helper"
    assert "sourdough" in agents[0]["description"]
    assert len(agents) <= 5


def test_find_agents_is_best_effort_without_a_min_score():
    seed(60)
    res = handle_tool_call("find_agents", json.dumps({"keywords": "sourdough"}))
    assert [a["name"] for a in res.payload["agents"]] == ["misc-helper"]  # a single weak hit still comes back


def test_find_agents_caps_at_five():
    seed(60)
    assert len(handle_tool_call("find_agents", {"keywords": "follow up inbox"}).payload["agents"]) == 5


def test_find_agents_empty_result_says_so():
    seed(60)
    res = handle_tool_call("find_agents", {"keywords": "zzzqqq"})
    assert res.success
    assert res.payload["agents"] == []
    assert "no" in res.payload["message"].lower()


def test_find_agents_missing_keywords_is_an_error_not_a_crash():
    seed(60)
    assert not handle_tool_call("find_agents", {}).success


def test_find_agents_does_not_touch_roster_or_turn_cap():
    seed(60)
    before = list(get_agent_roster().get_agents())
    token = limits.start_turn()
    try:
        for _ in range(limits.MAX_AGENTS_PER_TURN + 3):
            assert handle_tool_call("find_agents", {"keywords": "invoices acme"}).success
        assert limits.try_start_agent("someone")  # cap still has room
    finally:
        limits.end_turn(token)
    assert get_agent_roster().get_agents() == before


def _block(text):
    content = prepare_message_with_history(text, "", message_type="user")[0]["content"]
    return content.split("<active_agents>\n", 1)[1].split("\n</active_agents>", 1)[0]


def test_bounded_block_mentions_find_agents_on_both_lines():
    seed(60)
    b = _block("tell me a joke about whales")
    assert "No listed agent fits" in b and "more agents are not shown" in b
    assert b.count("find_agents") == 2


def test_small_roster_prompt_is_byte_identical_to_legacy():
    seed(5)
    legacy = "\n".join(f'<agent name="{n}" />' for n in get_agent_roster().get_agents())
    assert _block("anything") == legacy


class StubManager:
    def __init__(self):
        self.started = []

    async def execute_agent(self, agent_name, instructions, request_id=None):
        self.started.append(agent_name)
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")


def _tool(name, args, i):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _msg(*calls):
    return {"choices": [{"message": {"content": "", "tool_calls": list(calls)}}]}


async def test_model_finds_older_agent_instead_of_creating_a_duplicate(fake_llm, monkeypatch):
    seed(60)
    stub = StubManager()
    monkeypatch.setattr(tools_mod, "_EXECUTION_BATCH_MANAGER", stub)
    size = len(get_agent_roster().get_agents())
    fake_llm.script = [
        _msg(_tool("find_agents", {"keywords": "sourdough starter"}, 1)),
        _msg(_tool("send_message_to_agent", {"agent_name": "misc-helper", "instructions": "how is it going"}, 2)),
        {"choices": [{"message": {"content": "done"}}]},
    ]

    result = await InteractionAgentRuntime().execute("how is my bread culture doing")
    await asyncio.sleep(0.01)

    assert result.success
    tool_msgs = [m for m in fake_llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "misc-helper" in tool_msgs[0]["content"]
    assert [t["function"]["name"] for t in fake_llm.calls[0]["tools"]][-1] == "find_agents"
    assert stub.started == ["misc-helper"]
    assert len(get_agent_roster().get_agents()) == size
    assert result.execution_agents_used == 1


@__import__("pytest").fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    # these tests exercise the bounded view on small rosters; production activates it above 200 agents
    from server.services.execution import roster_view as _rv

    monkeypatch.setattr(_rv, "BOUND_ABOVE", 11)
