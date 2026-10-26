"""Simple agent roster management - just a list of agent names."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from ...config import DATA_DIR
from ...logging_config import logger

_REPLACE_RETRIES = 5
_REPLACE_DELAY = 0.05


def atomic_write_json(path: Path, data: Any) -> bool:
    """Write JSON next to `path` and swap it in, so a crash never leaves a half-written file.

    On Windows os.replace can raise PermissionError while another handle is open; retry a few
    times, then log and give up (the old file stays intact). Returns True on success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError as exc:
                if attempt == _REPLACE_RETRIES - 1:
                    logger.error(f"Could not replace {path.name} after retries: {exc}")
                else:
                    time.sleep(_REPLACE_DELAY)
    except Exception as exc:
        logger.error(f"Failed to write {path.name}: {exc}")
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return False


class AgentRoster:
    """Simple roster that stores agent names in a JSON file."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._agents: list[str] = []
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        """Load agent names from roster.json (a JSON list of names)."""
        if not self._roster_path.exists():
            self._agents = []
            self.save()
            return
        try:
            with open(self._roster_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("roster.json is not a list")
            self._agents = [str(name) for name in data]
        except Exception as exc:
            bad = self._roster_path.with_name(self._roster_path.name + ".bad")
            logger.error(f"roster.json is unreadable ({exc}); keeping it as {bad.name} and starting empty")
            try:
                os.replace(self._roster_path, bad)
            except OSError as move_exc:
                logger.error(f"Could not move corrupt roster aside: {move_exc}")
            self._agents = []

    def save(self) -> None:
        """Save agent names to roster.json atomically."""
        with self._lock:
            atomic_write_json(self._roster_path, self._agents)

    def add_agent(self, agent_name: str) -> None:
        """Add an agent to the roster if not already present."""
        with self._lock:
            if agent_name in self._agents:
                return
            self._agents.append(agent_name)
            atomic_write_json(self._roster_path, self._agents)

    def get_agents(self) -> list[str]:
        """Get list of all agent names."""
        return list(self._agents)

    def clear(self) -> None:
        """Clear the agent roster."""
        self._agents = []
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")


_ROSTER_PATH = DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
