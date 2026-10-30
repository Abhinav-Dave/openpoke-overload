"""Offline overload baseline for OpenPoke: fake model + fake Gmail, no network.

Usage: python eval/baseline.py [--src <dir containing server/>] [--label <name>]
Writes eval/results/<label>.json and prints a table. Run it on the original code
and on the improved code, then compare the two JSON files.
"""
import argparse, asyncio, collections, datetime as dt, importlib, json, os, platform
import shutil, subprocess, sys, tempfile, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATENCY = 0.05      # simulated seconds per fake model call
EXEC_SLEEP = 0.2    # simulated seconds per fake execution-agent model call (burst test)
NA = "n/a"

# Synthetic seed data: names and log text are made up, not taken from any real user.
TOPICS = ["invoice-followups", "flight-booking", "dentist-reschedule", "lease-renewal", "tax-documents",
          "conference-travel", "newsletter-cleanup", "gym-membership", "visa-appointment", "insurance-claim",
          "wedding-rsvp", "laptop-warranty"]
CLIENTS = ["acme", "march", "q3", "berlin", "mom", "landlord", "payroll", "airline", "school", "bank",
           "team-offsite", "freelance"]


def agent_names(n):
    base = len(TOPICS) * len(CLIENTS)
    return [f"{TOPICS[i % len(TOPICS)]}-{CLIENTS[(i // len(TOPICS)) % len(CLIENTS)]}"
            + (f"-{i // base + 1}" if i >= base else "") for i in range(n)]


# ---- fake model -----------------------------------------------------------
class State:
    def reset(self):
        self.calls = collections.Counter()      # fake model calls by module
        self.inflight = collections.Counter()
        self.peak = collections.Counter()
        self.runs = 0                           # interaction runs started (first call of a run)
        self.prompt = {}                        # label -> (chars, system chars) of last first-call
        self.latency = {}

S = State(); S.reset()


def _tool_call(name, args):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, (n, a) in enumerate([(name, args)] if not isinstance(name, list) else name)]}}]}


def _text(t):
    return {"choices": [{"message": {"content": t}}]}


def make_fake(label):
    async def fake(*args, **kw):
        messages = kw.get("messages") or []
        system = kw.get("system") or ""
        S.calls[label] += 1
        S.inflight[label] += 1
        S.peak[label] = max(S.peak[label], S.inflight[label])
        try:
            first = len(messages) == 1
            if "interaction_agent" in label:
                if first:
                    S.runs += 1
                    S.prompt[label] = (len(system) + sum(len(str(m.get("content", ""))) for m in messages), len(system))
                await asyncio.sleep(S.latency.get(label, LATENCY))
                tail = (messages[0]["content"].rsplit("<new_user_message>", 1)[-1]
                        if first and "<new_user_message>" in messages[0]["content"] else "")
                if first and "SPAWN8" in tail:
                    calls = [("send_message_to_agent", {"agent_name": f"burst-agent-{i}", "instructions": f"task {i}"})
                             for i in range(8)]
                    return _tool_call(calls, None)  # a list of (name, args) = parallel tool calls
                return _text("ok")
            if "execution_agent.runtime" in label:
                if first:
                    S.prompt[label] = (len(system) + sum(len(str(m.get("content", ""))) for m in messages), len(system))
                await asyncio.sleep(S.latency.get(label, LATENCY))
                return _text("Done. Task completed.")
            if "importance_classifier" in label:
                await asyncio.sleep(S.latency.get(label, LATENCY))
                return _tool_call("mark_email_importance", {"important": True, "summary": "Important: needs a reply today."})
            await asyncio.sleep(S.latency.get(label, LATENCY))   # summarizer, search_email
            return _text("summary")
        finally:
            S.inflight[label] -= 1
    return fake


# ---- setup ----------------------------------------------------------------
def install_fcntl_stub():
    # Only because the original roster.py does `import fcntl` (POSIX only); flock is a no-op here.
    if "fcntl" in sys.modules:
        return
    m = types.ModuleType("fcntl")
    m.LOCK_EX, m.LOCK_NB, m.LOCK_UN = 2, 4, 8
    m.flock = lambda *a, **k: None
    sys.modules["fcntl"] = m


def patch_all():
    """Patch seams by name in every loaded server module. Returns names of seams not found."""
    found = collections.Counter()
    gmail_fetch = None

    def fake_gmail(tool, user_id, arguments=None, **kw):
        return {"data": {"messages": [dict(m) for m in INBOX_FACTORY()]}}

    for name, mod in list(sys.modules.items()):
        if not name.startswith("server") or mod is None:
            continue
        if hasattr(mod, "request_chat_completion") and not name.startswith("server.openrouter_client"):
            mod.request_chat_completion = make_fake(name.replace("server.", ""))
            found["request_chat_completion"] += 1
        if hasattr(mod, "execute_gmail_tool"):
            mod.execute_gmail_tool = fake_gmail
            found["execute_gmail_tool"] += 1
        if hasattr(mod, "get_active_gmail_user_id"):
            mod.get_active_gmail_user_id = lambda *a, **k: "fake-user"
            found["get_active_gmail_user_id"] += 1
    return found


INBOX_FACTORY = lambda: []


def make_inbox(n):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [{"messageId": f"msg{i}", "threadId": f"t{i}", "subject": f"Action needed #{i}",
             "sender": f"person{i}@example.com", "to": "me@example.com", "messageTimestamp": now,
             "messageText": f"Hi, please confirm item {i} by end of day. " * 5, "labelIds": ["INBOX"]}
            for i in range(n)]


async def drain(timeout=60):
    """Wait for every background task (agents, interaction runs) to finish."""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=1)


def seed(n, pairs=6):
    from server.services.execution import get_agent_roster, get_execution_agent_logs
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    roster.clear(); logs.clear_all()
    for name in agent_names(n):
        roster.add_agent(name)
        for p in range(pairs):
            logs.record_request(name, f"Follow up on item {p} for {name}. Check the inbox and report back.")
            logs.record_agent_response(name, f"Checked the inbox for {name}, item {p}: found 2 related threads, drafted a reply.")


def reset_conversation():
    from server.services.conversation import get_conversation_log, get_working_memory_log
    for getter in (get_conversation_log, get_working_memory_log):
        try:
            getter().clear()
        except Exception:
            pass


def size(label_part):
    for k, v in S.prompt.items():
        if label_part in k:
            return {"chars": v[0], "approx_tokens": v[0] // 4, "system_chars": v[1]}
    return None


# ---- metrics --------------------------------------------------------------
async def m_interaction_prompt():
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    out = {}
    for n in (10, 50, 100, 300):
        seed(n); reset_conversation(); S.reset()
        await InteractionAgentRuntime().execute("What is on my plate today?")
        await drain()
        out[str(n)] = size("interaction_agent")
    return out


async def m_execution_prompt():
    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.services.execution import get_execution_agent_logs
    logs = get_execution_agent_logs()
    out = {}
    for k in (5, 50, 200):
        seed(0); S.reset()
        name = "invoice-followups-acme"
        kinds = [logs.record_request, lambda a, t: logs.record_action(a, t),
                 lambda a, t: logs.record_tool_response(a, "gmail_fetch", t), logs.record_agent_response]
        for i in range(k):
            kinds[i % 4](name, f"entry {i}: " + "invoice detail text " * 15)
        await ExecutionAgentRuntime(name).execute("Send a reminder about invoice 42.")
        out[str(k)] = size("execution_agent.runtime")
    return out


async def m_flood():
    from server.services.gmail.importance_watcher import ImportantEmailWatcher
    from server.services.gmail.seen_store import GmailSeenStore
    global INBOX_FACTORY
    out = {}
    for f in (10, 50):
        seed(10); reset_conversation(); S.reset()
        watcher = ImportantEmailWatcher(seen_store=GmailSeenStore(Path(tempfile.mkdtemp(dir=TMP)) / "seen.json", 300))
        INBOX_FACTORY = lambda: []
        await watcher._poll_once()                 # first poll only warms up
        INBOX_FACTORY = lambda f=f: make_inbox(f)
        t0 = time.perf_counter()
        await watcher._poll_once()
        await drain()
        wall = time.perf_counter() - t0
        out[str(f)] = {"classifier_calls": sum(v for k, v in S.calls.items() if "importance_classifier" in k),
                       "interaction_runs": S.runs, "total_model_calls": sum(S.calls.values()),
                       "wall_s": round(wall, 2)}
    INBOX_FACTORY = lambda: []
    return out


def _exec_peak():
    return sum(v for k, v in S.peak.items() if "execution_agent.runtime" in k) and \
        max(v for k, v in S.peak.items() if "execution_agent.runtime" in k)


def _interaction_peak():
    return max([v for k, v in S.peak.items() if "interaction_agent" in k] or [0])


async def m_burst_triggers():
    from server.services.trigger_scheduler import TriggerScheduler
    from server.services.triggers import TriggerRecord
    seed(0); reset_conversation(); S.reset()
    S.latency["agents.execution_agent.runtime"] = EXEC_SLEEP
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    recs = [TriggerRecord(id=i + 1, agent_name=f"trigger-agent-{i}", payload="Check the inbox.", status="active",
                          next_trigger=now, created_at=now, updated_at=now) for i in range(20)]

    class FakeService:                       # only the calls the scheduler makes
        def get_due_triggers(self, **kw): return list(recs)
        def schedule_next_occurrence(self, *a, **k): pass
        def record_failure(self, *a, **k): pass
        def clear_next_fire(self, *a, **k): pass

    sched = TriggerScheduler()
    sched._service = FakeService()
    t0 = time.perf_counter()
    await sched._poll_once()
    await drain()
    return {"due_triggers": 20, "peak_parallel_agents": _exec_peak(), "interaction_runs": S.runs,
            "total_model_calls": sum(S.calls.values()), "wall_s": round(time.perf_counter() - t0, 2)}


async def m_burst_agents():
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    seed(0); reset_conversation(); S.reset()
    S.latency["agents.execution_agent.runtime"] = EXEC_SLEEP
    t0 = time.perf_counter()
    await InteractionAgentRuntime().execute("SPAWN8 do eight things")
    await drain()
    return {"agents_requested": 8, "peak_parallel_agents": _exec_peak(), "interaction_runs": S.runs,
            "total_model_calls": sum(S.calls.values()), "wall_s": round(time.perf_counter() - t0, 2)}


async def m_burst_chat():
    from server.models import ChatMessage, ChatRequest
    from server.services.conversation.chat_handler import handle_chat_request
    seed(0); reset_conversation(); S.reset()
    for k in list(S.latency): del S.latency[k]
    S.latency["agents.interaction_agent.runtime"] = 0.1
    reqs = [ChatRequest(messages=[ChatMessage(role="user", content=f"message number {i}")]) for i in range(5)]
    t0 = time.perf_counter()
    await asyncio.gather(*(handle_chat_request(r) for r in reqs))
    await drain()
    return {"chat_messages": 5, "peak_overlapping_interaction_runs": _interaction_peak(), "interaction_runs": S.runs,
            "wall_s": round(time.perf_counter() - t0, 2)}


async def run_metric(fn):
    try:
        return await fn()
    except Exception as exc:       # a missing seam must not crash the script
        return {"error": f"{type(exc).__name__}: {exc}", "value": NA}


# ---- main -----------------------------------------------------------------
def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return NA


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return NA
        d = d[p]
    return d


def table(r):
    rows = [("metric", "value")]
    for n in (10, 50, 100, 300):
        rows.append((f"interaction prompt, {n} agents (chars / ~tokens)",
                     f"{get(r['interaction_prompt'], str(n), 'chars')} / {get(r['interaction_prompt'], str(n), 'approx_tokens')}"))
    for k in (5, 50, 200):
        rows.append((f"execution prompt, {k} log entries (chars / ~tokens)",
                     f"{get(r['execution_prompt'], str(k), 'chars')} / {get(r['execution_prompt'], str(k), 'approx_tokens')}"))
    for f in (10, 50):
        d = lambda key: get(r["email_flood"], str(f), key)
        rows.append((f"flood {f} emails: classifier / interaction runs / total calls / wall s",
                     f"{d('classifier_calls')} / {d('interaction_runs')} / {d('total_model_calls')} / {d('wall_s')}"))
    rows.append(("burst 20 triggers: peak parallel agents (interaction runs)",
                 f"{get(r['burst_triggers'], 'peak_parallel_agents')} ({get(r['burst_triggers'], 'interaction_runs')})"))
    rows.append(("burst 8 agents in one turn: peak parallel agents",
                 f"{get(r['burst_agents'], 'peak_parallel_agents')}"))
    rows.append(("burst 5 chat messages: peak overlapping interaction runs",
                 f"{get(r['burst_chat'], 'peak_overlapping_interaction_runs')}"))
    w = max(len(a) for a, _ in rows)
    return "\n".join(f"{a.ljust(w)}  {b}" for a, b in rows)


async def amain():
    results = {}
    for key, fn in (("interaction_prompt", m_interaction_prompt), ("execution_prompt", m_execution_prompt),
                    ("email_flood", m_flood), ("burst_triggers", m_burst_triggers),
                    ("burst_agents", m_burst_agents), ("burst_chat", m_burst_chat)):
        results[key] = await run_metric(fn)
    return results


def main():
    global TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()
    if args.src is None:
        sys.path.insert(0, str(ROOT / "eval"))
        from regression_eval import default_orig
        args.src = str(default_orig())
    src = Path(args.src).resolve()
    TMP = tempfile.mkdtemp(prefix="openpoke_eval_")
    try:
        shutil.copytree(src / "server", Path(TMP) / "server", ignore=shutil.ignore_patterns("__pycache__", "data"))
        sys.path.insert(0, TMP)
        os.environ["OPENROUTER_API_KEY"] = "fake-key"   # read at import; no call ever leaves the process
        install_fcntl_stub()
        for mod in ("server.agents.interaction_agent.runtime", "server.agents.execution_agent.runtime",
                    "server.services.conversation.summarization.summarizer", "server.services.gmail.importance_classifier",
                    "server.agents.execution_agent.tasks.search_email.tool", "server.services.gmail.importance_watcher",
                    "server.services.trigger_scheduler", "server.services.conversation.chat_handler"):
            try:
                importlib.import_module(mod)
            except Exception as exc:
                print(f"warning: could not import {mod}: {exc}", file=sys.stderr)
        import logging; logging.disable(logging.CRITICAL)
        seams = patch_all()
        results = asyncio.run(amain())
        results["meta"] = {"label": args.label, "python": platform.python_version(), "git_head": git_head(),
                           "src": str(src), "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                           "seams_patched": dict(seams), "fake_latency_s": LATENCY, "exec_sleep_s": EXEC_SLEEP}
        out = ROOT / "eval" / "results" / f"{args.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(table(results)); print(f"\nwrote {out}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
