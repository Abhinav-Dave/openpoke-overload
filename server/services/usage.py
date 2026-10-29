"""In-process model-call counters: tokens, optional cost, retries, failures (per model too)."""

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_total: Dict[str, Any] = {}
_models: Dict[str, Dict[str, Any]] = {}
_retries = _failed = 0


def _blank() -> Dict[str, Any]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0, "cost": None}


def reset() -> None:
    global _total, _models, _retries, _failed
    with _lock:
        _total, _models, _retries, _failed = _blank(), {}, 0, 0


reset()


def _num(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def record_success(model: str, usage: Optional[Dict[str, Any]]) -> None:
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(_num(usage.get("prompt_tokens")) or 0)
    completion = int(_num(usage.get("completion_tokens")) or 0)
    cost = _num(usage.get("cost"))
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    cached = int(_num(details.get("cached_tokens")) or 0)
    written = int(_num(details.get("cache_write_tokens")) or 0)
    with _lock:
        for bucket in (_total, _models.setdefault(model, _blank())):
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["cached_tokens"] += cached
            bucket["cache_write_tokens"] += written
            if cost is not None:
                bucket["cost"] = (bucket["cost"] or 0) + cost  # stays None if the API never reports cost


def record_retry() -> None:
    global _retries
    with _lock:
        _retries += 1


def record_failure() -> None:
    global _failed
    with _lock:
        _failed += 1


def get_usage() -> Dict[str, Any]:
    with _lock:
        return {
            **_total,
            "retries": _retries,
            "failed_calls": _failed,
            "models": {name: dict(b) for name, b in _models.items()},
        }
