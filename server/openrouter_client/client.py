from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings
from ..services import usage

OpenRouterBaseURL = "https://openrouter.ai/api/v1"


_RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_S = 30.0
_sleep = asyncio.sleep  # tests replace this


def _max_attempts() -> int:
    try:
        return max(1, int(os.getenv("OPENPOKE_LLM_MAX_ATTEMPTS", 3)))
    except ValueError:
        return 3


def _retry_after(response: httpx.Response) -> Optional[float]:
    try:
        return min(max(float(response.headers["Retry-After"]), 0.0), MAX_RETRY_AFTER_S)
    except (KeyError, ValueError):
        return None


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload."""

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _headers(api_key=api_key)
    attempts = _max_attempts()

    async with httpx.AsyncClient() as client:
        for attempt in range(1, attempts + 1):
            retry_after: Optional[float] = None
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                if response.status_code < 400:
                    data = response.json()
                    usage.record_success(model, data.get("usage") if isinstance(data, dict) else None)
                    return data
                if response.status_code not in _RETRY_STATUSES:
                    usage.record_failure()
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        _handle_response_error(exc)
                retry_after = _retry_after(response)
                error = f"HTTP {response.status_code}"
            except httpx.TransportError as exc:  # network errors and timeouts
                error = type(exc).__name__
            if attempt == attempts:
                usage.record_failure()
                raise OpenRouterError(f"OpenRouter request failed after {attempts} attempts ({error})")
            usage.record_retry()
            delay = retry_after if retry_after is not None else 0.5 * 2 ** (attempt - 1) * (1 + random.random() * 0.1)
            await _sleep(delay)

    raise OpenRouterError("OpenRouter request failed: unknown error")  # unreachable; keeps type checkers calm


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL"]
