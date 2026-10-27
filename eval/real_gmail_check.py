"""READ-ONLY check on a real Gmail inbox: snapshot, importance replay, messy-roster file.

  python eval/real_gmail_check.py snapshot --user-id ID [--days 14] [--max 40]   # one Composio read, saves a private snapshot
  python eval/real_gmail_check.py classify [--max-usd 0.3]                       # real model labels each saved email, counts runs
  python eval/real_gmail_check.py build                                          # makes private agents+queries for live_eval --real-file

Safety: every Gmail call goes through `read_only`, which only lets fetch and profile calls through. Send, draft,
reply, forward, delete and label calls raise before any network call. Email text is private: it is written only under
session_context/private/ (git-excluded), never printed, never put in eval/results. The key is read from the process
environment only; no .env file is read here. The text of each email also goes to the model provider during `classify`.
"""
import argparse, asyncio, json, os, re, sys
from dataclasses import asdict
from email.utils import parseaddr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "session_context" / "private"
SNAPSHOT, REAL_FILE = PRIVATE / "inbox_snapshot.json", PRIVATE / "real_roster.json"
ALLOWED = {"GMAIL_FETCH_EMAILS", "GMAIL_GET_PROFILE"}
COST_IN, COST_OUT = 0.10, 0.50   # USD per million tokens, haiku-5.5


def read_only(execute, tool_name, *args, **kwargs):
    """Call `execute` only for allow-listed read tools."""
    if tool_name not in ALLOWED:
        raise PermissionError(f"read-only check: {tool_name} is blocked")
    return execute(tool_name, *args, **kwargs)


def first_name(sender):
    name = parseaddr(sender)[0].strip().strip('"')
    return (name or parseaddr(sender)[1].split("@")[0]).strip()


def clean_subject(subject):
    return re.sub(r"^(\s*(re|fwd?)\s*:\s*)+", "", subject or "", flags=re.I).strip() or "(no subject)"


def messy_agents(emails):
    """One agent per email, named the way a rushed user or model would: mixed case, spaces, punctuation, numbers."""
    agents, seen = [], set()
    for e in emails:
        name = f"{first_name(e['sender'])} - {clean_subject(e['subject'])}"[:60].strip()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        body = re.sub(r"\s+", " ", e.get("clean_text") or "")[:150]
        agents.append({"name": name, "requests": [f"Deal with the email from {e['sender']}: {clean_subject(e['subject'])}. {body}"],
                       "sender": first_name(e["sender"])})
    return agents


NO_MATCH = ["Book me a table for four on Friday evening", "What is the weather like in Lisbon this weekend?",
            "Set up a monthly reminder to water the plants", "Find me a good running route near the lake",
            "Write a short birthday message for my cousin", "Compare two standing desks under 400 dollars"]


def messy_queries(agents):
    """Queries a person would type. Only senders that appear once get a labelled query, so the label is unambiguous."""
    by_sender = {}
    for a in agents:
        by_sender.setdefault(a["sender"].lower(), []).append(a)
    qs = []
    for group in by_sender.values():
        if len(group) != 1:
            continue
        a = group[0]
        qs.append({"text": f"Did {a['sender']} get back to me yet?", "expected": a["name"], "category": "direct", "split": "test"})
        qs.append({"text": f"Remind me what {a['sender']} wanted and draft a reply", "expected": a["name"], "category": "paraphrase", "split": "test"})
    qs += [{"text": t, "expected": None, "category": "no_match", "split": "test"} for t in NO_MATCH]
    return qs


def cmd_snapshot(a):
    sys.path.insert(0, str(ROOT))
    from server.services.gmail.client import execute_gmail_tool
    from server.services.gmail.processing import EmailTextCleaner, parse_gmail_fetch_response
    query = f"label:INBOX newer_than:{a.days}d"
    raw = read_only(execute_gmail_tool, "GMAIL_FETCH_EMAILS", a.user_id,
                    arguments={"query": query, "include_payload": True, "max_results": a.max})
    emails, _ = parse_gmail_fetch_response(raw, query=query, cleaner=EmailTextCleaner(max_url_length=60))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(e), "timestamp": e.timestamp.isoformat()} for e in emails]
    SNAPSHOT.write_text(json.dumps(rows), encoding="utf-8")
    print(f"saved {len(rows)} emails to {SNAPSHOT.relative_to(ROOT)} (private, git-excluded)")


async def classify_all(rows, max_usd):
    from server.services.gmail import importance_classifier as ic
    from server.services.gmail.importance_classifier import ClassificationFailed, classify_email_importance
    from server.services.gmail.processing import ProcessedEmail
    from datetime import datetime
    tokens, real = [0, 0], ic.request_chat_completion

    async def counted(*x, **k):
        r = await real(*x, **k)
        u = r.get("usage") or {}
        tokens[0] += u.get("prompt_tokens") or 0
        tokens[1] += u.get("completion_tokens") or 0
        return r
    ic.request_chat_completion = counted
    important = urgent = failed = 0
    for row in rows:
        if (tokens[0] * COST_IN + tokens[1] * COST_OUT) / 1e6 >= max_usd:
            print("stopped: spend cap reached")
            break
        e = ProcessedEmail(**{**row, "timestamp": datetime.fromisoformat(row["timestamp"])})
        try:
            v = await classify_email_importance(e)
        except ClassificationFailed:
            failed += 1
            continue
        important += v is not None
        urgent += bool(v and v.urgent)
    return important, urgent, failed, tokens


def runs_old_new(important, urgent, max_urgent=3):
    """Interaction-agent runs one poll causes. Old: one per important email. New: urgent ones (capped) plus one digest."""
    sent_alone = min(urgent, max_urgent)
    digest = (important - urgent) + (urgent - sent_alone)
    return important, sent_alone + (1 if digest else 0)


def cmd_classify(a):
    sys.path.insert(0, str(ROOT))
    rows = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    os.environ.setdefault("OPENPOKE_MODEL", "anthropic/claude-haiku-5.5")
    important, urgent, failed, tokens = asyncio.run(classify_all(rows, a.max_usd))
    old, new = runs_old_new(important, urgent)
    usd = (tokens[0] * COST_IN + tokens[1] * COST_OUT) / 1e6
    print(f"emails {len(rows)} | important {important} | urgent {urgent} | classifier failures {failed}")
    print(f"if all arrived in ONE poll: interaction runs old {old}, new {new}")
    print(f"classifier tokens in {tokens[0]}, out {tokens[1]}, about ${usd:.4f}")


def cmd_build(a):
    rows = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    agents = messy_agents(rows)
    REAL_FILE.write_text(json.dumps({"agents": agents, "queries": messy_queries(agents)}), encoding="utf-8")
    q = json.loads(REAL_FILE.read_text(encoding="utf-8"))["queries"]
    print(f"{len(agents)} messy agents, {len(q)} queries ({sum(1 for x in q if x['expected'])} labelled) -> {REAL_FILE.relative_to(ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--user-id", required=True)
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--max", type=int, default=40)
    c = sub.add_parser("classify")
    c.add_argument("--max-usd", type=float, default=0.3)
    sub.add_parser("build")
    a = p.parse_args(argv)
    {"snapshot": cmd_snapshot, "classify": cmd_classify, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
