import asyncio
from typing import Optional, Union

from fastapi import status
from fastapi.responses import JSONResponse, PlainTextResponse

from ...agents.interaction_agent.runtime import InteractionAgentRuntime, release_chat_slot, reserve_chat_slot
from ...logging_config import logger
from ...models import ChatMessage, ChatRequest
from ...utils import error_response


# Extract the most recent user message from the chat request payload
def _extract_latest_user_message(payload: ChatRequest) -> Optional[ChatMessage]:
    for message in reversed(payload.messages):
        if message.role.lower().strip() == "user" and message.content.strip():
            return message
    return None


# Process incoming chat requests by routing them to the interaction agent runtime
async def handle_chat_request(payload: ChatRequest) -> Union[PlainTextResponse, JSONResponse]:
    """Handle a chat request using the InteractionAgentRuntime."""

    # Extract user message
    user_message = _extract_latest_user_message(payload)
    if user_message is None:
        return error_response("Missing user message", status_code=status.HTTP_400_BAD_REQUEST)

    user_content = user_message.content.strip()  # Already checked in _extract_latest_user_message

    logger.info("chat request", extra={"message_length": len(user_content)})

    try:
        runtime = InteractionAgentRuntime()
    except ValueError as ve:
        # Missing API key error
        logger.error("configuration error", extra={"error": str(ve)})
        return error_response(str(ve), status_code=status.HTTP_400_BAD_REQUEST)

    if not reserve_chat_slot():
        return error_response(
            "I am still working on earlier messages. Try again in a moment.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    started = False

    async def _run_interaction() -> None:
        nonlocal started
        # No await between here and execute() taking over the slot (_run_slot), so
        # started=True means _run_slot releases it exactly once, even on cancellation.
        started = True
        try:
            await runtime.execute(user_message=user_content, reserved=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("chat task failed", extra={"error": str(exc)})

    task = asyncio.create_task(_run_interaction())
    # Cancelled before its first step: the coroutine never ran, so nobody else frees the slot.
    task.add_done_callback(lambda _t: None if started else release_chat_slot())

    return PlainTextResponse("", status_code=status.HTTP_202_ACCEPTED)
