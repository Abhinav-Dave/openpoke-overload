import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import baseline as b  # noqa: E402


def _reply(pt, ct, tools=()):
    return {"usage": {"prompt_tokens": pt, "completion_tokens": ct},
            "choices": [{"message": {"tool_calls": [{"function": {"name": n}} for n in tools]}}]}


def test_live_wrapper_counts_spend_and_stops_at_cap():
    b.S.reset()
    b.LIVE.update(on=True, usd=0.0, cap=0.0002, tokens_in=0, tokens_out=0, requested=0, calls=0)

    async def real(*a, **k):
        return _reply(1000, 100, ["send_message_to_agent"])

    live = b.make_live("agents.interaction_agent.runtime", real)
    asyncio.run(live(messages=[{"role": "user", "content": "hi"}], system="s"))
    assert (b.LIVE["tokens_in"], b.LIVE["tokens_out"], b.LIVE["requested"], b.S.runs) == (1000, 100, 1, 1)
    assert round(b.LIVE["usd"], 6) == 0.00015
    asyncio.run(live(messages=[{"role": "user", "content": "hi"}], system="s"))   # 0.0003 total now >= cap
    try:
        asyncio.run(live(messages=[{"role": "user", "content": "hi"}], system="s"))
        raise AssertionError("cap not enforced")
    except RuntimeError as e:
        assert "spend cap" in str(e)
    b.LIVE["on"] = False
