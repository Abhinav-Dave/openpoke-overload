"""Embedding search for find_agents on large rosters (see roster_view.find_agents). Stdlib only.

One call per search: the query plus up to MAX_NEW agents whose text is new or changed. Vectors are kept
in a disk cache, one JSON file per model. Any failure returns None, and the caller then uses word search
alone. The query and the agents' request text are sent to the embedding provider.
"""

import base64
import hashlib
import json
import math
import operator
import os
import time
from array import array
from pathlib import Path
from typing import Optional

from ...config import DATA_DIR
from ...logging_config import logger
from ...openrouter_client import request_embeddings
from .log_store import _file_slug
from .roster import atomic_write_json

ENABLED = os.getenv("OPENPOKE_EMBED", "1") != "0"
MODEL = os.getenv("OPENPOKE_EMBED_MODEL", "openai/text-embedding-3-small")
MIN_SIM = float(os.getenv("OPENPOKE_EMBED_MIN_SIM", "0.4"))  # tuned on the dev split, see eval/find_eval.py
MAX_NEW = 128  # measured on OpenRouter: 128 inputs took 1.6 to 2.3 s, 256 took 3.1 to 4.0 s
TIMEOUT_S = 5.0  # about twice the slowest 128-input call
COOLDOWN_S = 60.0  # after a failure, skip embedding search for this long
QUERY_CHARS = 2000

_cache: Optional[dict] = None  # model -> {agent name: (sha1 of embedded text, unit vector)}; loaded once per process
_down_until = 0.0
_now = time.monotonic
_told = False  # the data-flow notice is logged once per process


def _cache_path() -> Path:
    # ponytail: one JSON file per model, rewritten whole on every save (about 12 MB at 1,500 agents of
    # 1,536 dimensions), and a pure-Python cosine loop (about 130 ms at 1,500 agents); move to one file per
    # agent and ask for fewer dimensions (or use numpy) if rosters grow far past that.
    return DATA_DIR / "embeddings" / f"{_file_slug(MODEL)}.json"


def _unit(vec: list[float]) -> array:
    norm = math.sqrt(sum(x * x for x in vec))
    return array("f", (x / norm for x in vec) if norm else vec)  # a zero vector stays zero


def _decode(b64: str) -> array:
    vec = array("f")
    vec.frombytes(base64.b64decode(b64, validate=True))
    return vec


def _load() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {name: (sha, _decode(b64)) for name, (sha, b64) in raw.items()}
    except Exception as exc:
        logger.warning(f"Ignoring unreadable embedding cache {path.name} ({type(exc).__name__})")
        return {}


def _save(entries: dict) -> None:
    data = {name: [sha, base64.b64encode(vec.tobytes()).decode("ascii")] for name, (sha, vec) in entries.items()}
    atomic_write_json(_cache_path(), data)


def _rank(query: str, names: list[str], texts: list[str]) -> list[int]:
    global _cache, _told
    if _cache is None:
        _cache = {}
    if MODEL not in _cache:
        _cache[MODEL] = _load()
    entries = _cache[MODEL]
    shas = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [i for i, name in enumerate(names) if entries.get(name, ("",))[0] != shas[i]][:MAX_NEW]

    if not _told:
        _told = True
        logger.info(f"find_agents sends its keywords and agent request text to {MODEL} for embedding search "
                    "(OPENPOKE_EMBED=0 turns this off)")
    vectors = request_embeddings(model=MODEL, inputs=[query] + [texts[i] for i in todo], timeout=TIMEOUT_S)
    if len(vectors) != 1 + len(todo) or len({len(v) for v in vectors}) != 1:
        raise ValueError("embedding count or size mismatch")
    q = _unit(vectors[0])
    if todo or any(len(vec) != len(q) for _, vec in entries.values()):
        # Drop agents that left the roster and vectors of another size (they are embedded again next time).
        current = set(names)
        entries = {name: e for name, e in entries.items() if name in current and len(e[1]) == len(q)}
        entries.update({names[i]: (shas[i], _unit(v)) for i, v in zip(todo, vectors[1:])})
        _save(entries)
        _cache[MODEL] = entries

    scores = {}
    for i, name in enumerate(names):
        entry = entries.get(name)
        if entry and entry[0] == shas[i] and len(entry[1]) == len(q):
            scores[i] = sum(map(operator.mul, q, entry[1]))
    return sorted((i for i, s in scores.items() if s > 0 and s >= MIN_SIM), key=lambda i: -scores[i])


def similar(query: str, names: list[str], texts: list[str]) -> Optional[list[int]]:
    """Indices into `names`, best cosine first, keeping cosine >= MIN_SIM. None means unavailable:
    disabled, blank query, cooling down after a failure, or any failure now."""
    global _down_until
    if not ENABLED or not query.strip() or _now() < _down_until:
        return None
    try:
        return _rank(query[:QUERY_CHARS], names, texts)
    except Exception as exc:
        _down_until = _now() + COOLDOWN_S
        # The type only: the message could carry the query or agent text.
        logger.warning(f"Embedding search off for {COOLDOWN_S:.0f} s, word search only ({type(exc).__name__})")
        return None
