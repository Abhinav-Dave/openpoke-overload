import pytest

from server.agents.interaction_agent import runtime as rt
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.services.conversation import get_conversation_log


def _tell(exc):
    InteractionAgentRuntime()._tell_user_of_failure(exc)
    return get_conversation_log().load_transcript()


def test_iteration_limit_error_gets_limit_wording():
    out = _tell(rt.IterationLimitError("Reached tool iteration limit without final response"))
    assert rt.ITERATION_LIMIT_REPLY in out
    assert rt.FAILURE_REPLY not in out


def test_other_exception_gets_generic_wording():
    out = _tell(RuntimeError("boom"))
    assert rt.FAILURE_REPLY in out
    assert rt.ITERATION_LIMIT_REPLY not in out


def test_generic_exception_mentioning_iteration_limit_stays_generic():
    out = _tell(RuntimeError("Reached tool iteration limit without final response"))
    assert rt.FAILURE_REPLY in out
    assert rt.ITERATION_LIMIT_REPLY not in out


def test_error_is_a_runtime_error():
    assert issubclass(rt.IterationLimitError, RuntimeError)


async def test_loop_raises_typed_error_at_limit(monkeypatch):
    tool_call = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "nope", "arguments": "{}"}}]}}]}

    async def fake(*a, **k):
        return tool_call

    monkeypatch.setattr(rt, "request_chat_completion", fake)
    with pytest.raises(rt.IterationLimitError):
        await InteractionAgentRuntime()._run_turn("sys", [])
