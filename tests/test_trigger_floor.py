"""G3: recurrence floor and trigger count caps, enforced in TriggerService."""

import os
import subprocess
import sys

import pytest

from conftest import ROOT
from server.services.triggers import limits
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore

START = "2030-01-01T09:01:00"


@pytest.fixture
def svc(tmp_path):
    return TriggerService(TriggerStore(tmp_path / "t.db"))


def make(svc, rule=None, agent="a", start=START):
    return svc.create_trigger(agent_name=agent, payload="p", recurrence_rule=rule, start_time=start, timezone_name="UTC")


OK_RULES = [
    "FREQ=DAILY",
    "FREQ=WEEKLY;BYDAY=MO",
    "FREQ=HOURLY",
    "FREQ=MINUTELY;INTERVAL=15",
    "FREQ=MINUTELY;INTERVAL=5",
    "FREQ=MONTHLY;BYMONTHDAY=1,2",
]
BAD_RULES = [
    "FREQ=MINUTELY",
    "FREQ=MINUTELY;INTERVAL=2",
    "FREQ=SECONDLY;INTERVAL=30",
    "FREQ=SECONDLY",
    "FREQ=DAILY;BYHOUR=9;BYMINUTE=0;BYSECOND=0,10,20",
    "FREQ=HOURLY;BYMINUTE=0,1,2",
    # first gap is 6 hours, later gaps are 2 minutes
    "FREQ=DAILY;BYHOUR=9,15;BYMINUTE=0,2",
]


@pytest.mark.parametrize("rule", OK_RULES)
def test_safe_rules_accepted(svc, rule):
    assert make(svc, rule).recurrence_rule


@pytest.mark.parametrize("rule", BAD_RULES)
def test_unsafe_rules_rejected_on_create(svc, rule):
    with pytest.raises(limits.TriggerLimitError) as e:
        make(svc, rule)
    assert "5 minutes" in str(e.value)
    assert svc.list_triggers(agent_name="a") == []


def test_message_names_actual_interval(svc):
    with pytest.raises(limits.TriggerLimitError, match="every 30 seconds.*shortest repeat allowed is 5 minutes"):
        make(svc, "FREQ=SECONDLY;INTERVAL=30")


def test_floor_is_a_module_constant(svc, monkeypatch):
    monkeypatch.setattr(limits, "MIN_TRIGGER_INTERVAL_SECONDS", 60)
    assert make(svc, "FREQ=MINUTELY;INTERVAL=2")
    with pytest.raises(limits.TriggerLimitError, match="1 minute"):
        make(svc, "FREQ=SECONDLY;INTERVAL=30")


def test_one_shot_not_subject_to_floor(svc):
    assert make(svc, None).recurrence_rule is None


def test_garbage_rrule_rejected(svc):
    with pytest.raises(limits.TriggerLimitError, match="repeat rule"):
        make(svc, "FREQ=SOMETIMES;BANANA")
    assert svc.list_triggers(agent_name="a") == []


def test_update_tightening_rule_rejected_and_stored_unchanged(svc):
    t = make(svc, "FREQ=DAILY")
    with pytest.raises(limits.TriggerLimitError, match="5 minutes"):
        svc.update_trigger(t.id, agent_name="a", recurrence_rule="FREQ=MINUTELY", timezone_name="UTC")
    again = svc.list_triggers(agent_name="a")[0]
    assert again.recurrence_rule == t.recurrence_rule and again.next_trigger == t.next_trigger


def test_update_to_safe_rule_and_pause_of_legacy_rule_still_work(svc):
    t = make(svc, "FREQ=DAILY")
    assert svc.update_trigger(t.id, agent_name="a", recurrence_rule="FREQ=HOURLY", timezone_name="UTC")
    # a legacy fast trigger already in the store can still be paused
    legacy = svc._store.insert({
        "agent_name": "a", "payload": "p", "start_time": None, "next_trigger": None,
        "recurrence_rule": "DTSTART:20300101T000000Z\nRRULE:FREQ=MINUTELY", "timezone": "UTC",
        "status": "active", "last_error": None, "created_at": "x", "updated_at": "x",
    })
    paused = svc.update_trigger(legacy, agent_name="a", status="paused", timezone_name="UTC")
    assert paused.status == "paused"


def test_per_agent_cap_rejects_21st(svc):
    for _ in range(20):
        make(svc, "FREQ=DAILY")
    with pytest.raises(limits.TriggerLimitError, match="20"):
        make(svc, "FREQ=DAILY")
    assert make(svc, "FREQ=DAILY", agent="other")


def test_completed_trigger_frees_a_slot(svc):
    first = make(svc, "FREQ=DAILY")
    for _ in range(19):
        make(svc, "FREQ=DAILY")
    svc.mark_as_completed(first.id, agent_name="a")
    assert make(svc, "FREQ=DAILY")


def test_total_cap_rejects_101st(svc):
    for n in range(5):
        for _ in range(20):
            make(svc, "FREQ=DAILY", agent=f"agent{n}")
    with pytest.raises(limits.TriggerLimitError, match="100"):
        make(svc, "FREQ=DAILY", agent="agent9")


def test_env_overrides_and_defaults():
    code = (
        "from server.services.triggers import limits as l; "
        "print(l.MIN_TRIGGER_INTERVAL_SECONDS, l.MAX_TRIGGERS_PER_AGENT, l.MAX_TRIGGERS_TOTAL)"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith("OPENPOKE_") or k == "OPENPOKE_DATA_DIR"}

    def run(extra):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**base, **extra}, capture_output=True, text=True)
        return out.stdout.split()

    assert run({}) == ["300", "20", "100"]
    assert run({
        "OPENPOKE_MIN_TRIGGER_INTERVAL_SECONDS": "60",
        "OPENPOKE_MAX_TRIGGERS_PER_AGENT": "3",
        "OPENPOKE_MAX_TRIGGERS_TOTAL": "7",
    }) == ["60", "3", "7"]
    assert run({"OPENPOKE_MAX_TRIGGERS_TOTAL": "junk"})[2] == "100"
