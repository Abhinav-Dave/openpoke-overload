"""Safety floor for reminders: minimum repeat interval and trigger count caps."""

from __future__ import annotations

import os
from itertools import islice
from typing import Optional

from .utils import load_rrule


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


MIN_TRIGGER_INTERVAL_SECONDS = _env_int("OPENPOKE_MIN_TRIGGER_INTERVAL_SECONDS", 300)
MAX_TRIGGERS_PER_AGENT = _env_int("OPENPOKE_MAX_TRIGGERS_PER_AGENT", 20)
MAX_TRIGGERS_TOTAL = _env_int("OPENPOKE_MAX_TRIGGERS_TOTAL", 100)

# ponytail: only the next few repeats are checked; a rule that turns fast after that is not caught.
_SAMPLED_OCCURRENCES = 6


class TriggerLimitError(ValueError):
    """Raised with a plain message the agent can act on."""


def _human(seconds: int) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size and seconds % size == 0:
            n = seconds // size
            return f"{n} {unit}{'' if n == 1 else 's'}"
    return f"{seconds} second{'' if seconds == 1 else 's'}"


def check_recurrence(stored_recurrence: Optional[str]) -> None:
    """Reject rules that are unparseable or repeat faster than the floor."""

    if not stored_recurrence:
        return
    try:
        times = list(islice(load_rrule(stored_recurrence), _SAMPLED_OCCURRENCES))
    except Exception as exc:
        raise TriggerLimitError(
            "That repeat rule could not be understood. Use a standard RRULE such as FREQ=DAILY."
        ) from exc
    gaps = [int((b - a).total_seconds()) for a, b in zip(times, times[1:])]
    if gaps and min(gaps) < MIN_TRIGGER_INTERVAL_SECONDS:
        raise TriggerLimitError(
            f"That repeats every {_human(min(gaps))}. "
            f"The shortest repeat allowed is {_human(MIN_TRIGGER_INTERVAL_SECONDS)}."
        )


def check_counts(agent_count: int, total_count: int) -> None:
    """Reject a new trigger when the open-trigger counts are already at the cap."""

    if agent_count >= MAX_TRIGGERS_PER_AGENT:
        raise TriggerLimitError(
            f"This agent already has {agent_count} open triggers (limit {MAX_TRIGGERS_PER_AGENT}). "
            "Complete or reuse an existing trigger instead of creating another."
        )
    if total_count >= MAX_TRIGGERS_TOTAL:
        raise TriggerLimitError(
            f"There are already {total_count} open triggers in total (limit {MAX_TRIGGERS_TOTAL}). "
            "Complete an existing trigger before creating another."
        )
