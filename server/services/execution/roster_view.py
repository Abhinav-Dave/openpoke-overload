"""Pick which execution agents the interaction agent should see for a message.

Stdlib only: BM25-style word scoring over name + first request + recent requests.
"""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from .log_store import get_execution_agent_logs

K = int(os.getenv("OPENPOKE_ROSTER_VIEW_K", "8"))
HOT_N = int(os.getenv("OPENPOKE_ROSTER_HOT_N", "3"))
MIN_SCORE = float(os.getenv("OPENPOKE_ROSTER_MIN_SCORE", "2.0"))
MIN_CONTENT_TOKENS = int(os.getenv("OPENPOKE_ROSTER_MIN_CONTENT_TOKENS", "2"))
MIN_COVERAGE = float(os.getenv("OPENPOKE_ROSTER_MIN_COVERAGE", "0.0"))
BOUND_ABOVE = int(os.getenv("OPENPOKE_ROSTER_BOUND_ABOVE", "200"))  # at most this many agents: list every name (original)
TAIL_NAMES = int(os.getenv("OPENPOKE_ROSTER_TAIL_NAMES", "200"))  # names in the compact tail of the bounded view
FIND_LIMIT = 5

DESC_CHARS = 100
RECENT_REQUESTS = 3
RECENT_CHARS = 300
NAME_BOOST = 3  # name tokens count this many times in the searchable text
_K1, _B = 1.5, 0.75
# Query words that say nothing about a topic; without this "about the" matches half the roster.
_STOP = frozenset(
    "the and for you your about with that this from have has what when where how any all can "
    "did does was are but not out get got new its our there their they them then than into "
    "like some more much going still yet too very just please me my we us is in of to it be on at as or an do if so "
    "tell give let make show send check find look want need".split()
)


@dataclass(frozen=True)
class AgentView:
    name: str
    description: str = ""


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def _describe(first_request: str) -> str:
    one_line = " ".join(first_request.split())
    return one_line if len(one_line) <= DESC_CHARS else one_line[:DESC_CHARS] + "..."


def _read_agent(name: str) -> tuple[str, str, float]:
    """(description, searchable text, log mtime) for one agent."""
    logs = get_execution_agent_logs()
    requests = [p for tag, _, p in logs.iter_entries(name) if tag == "agent_request"]
    description = _describe(requests[0]) if requests else ""
    recent = " ".join(r[:RECENT_CHARS] for r in requests[1:][-RECENT_REQUESTS:])  # first one is the description
    try:
        mtime = logs._log_path(name).stat().st_mtime
    except OSError:
        mtime = 0.0
    return description, f"{description} {recent}", mtime


def _content_tokens(query: str) -> set[str]:
    return set(tokenize(query)) - _STOP


def _scores(query: str, docs: list[list[str]]) -> list[float]:
    """BM25 score of each tokenised doc against the query."""
    q = _content_tokens(query)
    n = len(docs)
    avg = (sum(len(d) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in set(d) if t in q)
    out = []
    for d in docs:
        tf, score = Counter(d), 0.0
        for t in q & tf.keys():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (_K1 + 1) / (tf[t] + _K1 * (1 - _B + _B * len(d) / avg))
        out.append(score)
    return out


def select_agents(
    query: str,
    agent_names: list[str],
    *,
    k: Optional[int] = None,
    hot_n: Optional[int] = None,
    min_score: Optional[float] = None,
    min_coverage: Optional[float] = None,
    include: Iterable[str] = (),
) -> tuple[list[AgentView], int, bool]:
    """Return (agents to show, number hidden, no_match).

    Rosters of BOUND_ABOVE or fewer are returned whole, names only (the legacy prompt format).
    Otherwise: top-k by score, the hot_n most recently used, and anything in `include`.
    """
    k = K if k is None else k
    hot_n = HOT_N if hot_n is None else hot_n
    min_score = MIN_SCORE if min_score is None else min_score
    min_coverage = MIN_COVERAGE if min_coverage is None else min_coverage
    if len(agent_names) <= BOUND_ABOVE:
        return [AgentView(n) for n in agent_names], 0, False

    info = [_read_agent(n) for n in agent_names]
    # Name tokens repeated NAME_BOOST times so a name hit outweighs one stray word in a request.
    docs = [tokenize(n) * NAME_BOOST + tokenize(text) for n, (_, text, _) in zip(agent_names, info)]
    scores = _scores(query, docs)
    order = sorted(range(len(agent_names)), key=lambda i: -scores[i])
    q = _content_tokens(query)
    covered = len(q & set(docs[order[0]])) / len(q) if q else 0.0
    # Too few content words ("send it") means we cannot judge fit, so never claim "no match".
    no_match = len(q) >= MIN_CONTENT_TOKENS and (scores[order[0]] < min_score or covered < min_coverage)

    picked: list[int] = [] if no_match else [i for i in order[:k] if scores[i] > 0]
    hot = sorted(range(len(agent_names)), key=lambda i: -info[i][2])[:hot_n]
    wanted = set(include)
    picked += [i for i in hot if i not in picked]
    picked += [i for i, n in enumerate(agent_names) if n in wanted and i not in picked]

    shown = [AgentView(agent_names[i], info[i][0]) for i in picked]
    return shown, len(agent_names) - len(shown), no_match


def tail_names(agent_names: list[str], shown: Iterable[str], cap: Optional[int] = None) -> tuple[list[str], int]:
    """Names not in `shown`, most recently used first, capped; plus how many were cut."""
    cap = TAIL_NAMES if cap is None else cap
    skip = set(shown)

    def mtime(n: str) -> float:
        try:
            return get_execution_agent_logs()._log_path(n).stat().st_mtime
        except OSError:
            return 0.0

    rest = sorted((n for n in agent_names if n not in skip), key=lambda n: -mtime(n))
    return rest[:cap], max(0, len(rest) - cap)


def find_agents(keywords: str, agent_names: list[str], limit: int = FIND_LIMIT) -> list[AgentView]:
    """Second-chance lookup over the whole roster: best-effort BM25, no minimum score."""
    if not agent_names:
        return []
    info = [_read_agent(n) for n in agent_names]
    docs = [tokenize(n) * NAME_BOOST + tokenize(text) for n, (_, text, _) in zip(agent_names, info)]
    scores = _scores(keywords, docs)
    order = [i for i in sorted(range(len(agent_names)), key=lambda i: -scores[i]) if scores[i] > 0]
    return [AgentView(agent_names[i], info[i][0]) for i in order[:limit]]
