"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List

from ...services.execution import get_agent_roster, limits
from ...services.execution.roster_view import find_agents, select_agents, tail_names

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()

_FIND_HINT = "A find_agents call (a few keywords, try synonyms) is required before creating a new agent; without it the creation is refused. If an older agent fits, reuse it."


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_active_agents(latest_text, message_type)}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Format the execution agents likely to fit this turn into XML tags for LLM awareness
def _render_active_agents(query: str = "", message_type: str = "user") -> str:
    roster = get_agent_roster()
    roster.load()
    agents = roster.get_agents()

    if not agents:
        return "None"

    is_result = message_type == "agent"
    named = [a for a in agents if a and a in query] if is_result else []
    shown, hidden, no_match = select_agents(query, agents, include=named)
    no_match = no_match and not is_result  # a result payload is not a new request

    rendered: List[str] = []
    for view in shown:
        name = escape(view.name or "agent", quote=True)
        if view.description:
            rendered.append(f'<agent name="{name}">{escape(view.description, quote=False)}</agent>')
        else:
            rendered.append(f'<agent name="{name}" />')
    if hidden:
        rendered.append(f"{hidden} more agents are not shown. You can still reuse any agent by its exact name. {_FIND_HINT}")
        tail, cut = tail_names(agents, [v.name for v in shown])
        line = "Other agents (most recent first): " + ", ".join(escape(n, quote=True) for n in tail)
        rendered.append(line + (f" and {cut} more (use find_agents)" if cut else ""))
    if no_match:
        rendered.append(
            "No listed agent fits this message. Create a new agent with a short descriptive name, "
            "unless the user clearly means an older agent (then reuse it by its exact name). " + _FIND_HINT
        )
        # Run the find_agents search here (saves a model round trip) and satisfy the find-before-create gate.
        close = find_agents(query, agents)
        limits.mark_find_agents_called()
        if close:
            items = "; ".join(
                escape(a.name, quote=True) + (f" ({escape(a.description, quote=False)})" if a.description else "")
                for a in close
            )
            rendered.append(f"Closest existing agents (searched for you, none is a confident match): {items}")

    return "\n".join(rendered)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
