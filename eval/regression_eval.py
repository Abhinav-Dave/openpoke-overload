"""Regression + gap evaluator for OpenPoke (offline: fake model, fake Gmail, no network).

Usage: python eval/regression_eval.py [--orig <dir with server/>] [--new <dir with server/>]
                                      [--checks 1,2,..7] [--gaps all|none|quick]
Each scenario runs in its own Python process on a fresh temp copy of a source tree, once for the
ORIGINAL tree and once for the NEW tree. Part A (regression) must match; part B (gaps) only reports.
Exit code 1 if any regression check fails. Writes eval/results/regression_eval.json.
"""
import argparse, asyncio, collections, datetime as dt, hashlib, json, os, re, shutil, subprocess
import sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LATENCY = 0.01   # simulated seconds per fake model call
MARK = "RESULT:"


def default_orig():
    """The original code: a private copy if present, else the first commit, pulled out of git into a temp dir."""
    for p in [ROOT, *ROOT.parents]:
        if (p / "session_context" / "baseline_src" / "server").is_dir():
            return p / "session_context" / "baseline_src"
    import io, subprocess, tarfile
    dest = Path(tempfile.mkdtemp(prefix="openpoke_orig_"))
    out = subprocess.run(["git", "archive", "9788701", "server"], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        tar.extractall(dest, members=[m for m in tar.getmembers() if m.name.startswith("server") and ".." not in Path(m.name).parts])
    return dest


# =============================== WORKER SIDE ===============================
# Runs inside a subprocess: `regression_eval.py --worker NAME --tree T --work W [--data D] [--params JSON]`
class M:
    """Scripted fake model state."""
    calls, handlers = [], {}
    inflight, peak = collections.Counter(), collections.Counter()
    latency = {}

    @classmethod
    def reset(cls):
        cls.calls, cls.handlers = [], {}
        cls.inflight, cls.peak, cls.latency = collections.Counter(), collections.Counter(), {}

    @classmethod
    def n(cls, role, first_only=False):
        return sum(1 for c in cls.calls if c["role"] == role and (not first_only or c["n"] == 1))


def tc(*calls):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}}]}


def txt(t):
    return {"choices": [{"message": {"content": t}}]}


def make_fake(role):
    async def fake(*args, **kw):
        msgs = kw.get("messages") or []
        rec = {"role": role, "n": len(msgs), "model": kw.get("model"), "system": kw.get("system") or "",
               "first": msgs[0]["content"] if msgs else "", "msgs": msgs, "tools": []}
        M.calls.append(rec)
        M.inflight[role] += 1
        M.peak[role] = max(M.peak[role], M.inflight[role])
        try:
            await asyncio.sleep(M.latency.get(role, LATENCY))
            handler = M.handlers.get(role)
            resp = handler(rec) if handler else (
                tc(("mark_email_importance", {"important": True, "summary": "Important."}))
                if role == "classifier" else txt("ok"))
            rec["tools"] = [t["function"]["name"] for t in resp["choices"][0]["message"].get("tool_calls", [])]
            return resp
        finally:
            M.inflight[role] -= 1
    return fake


def role_of(modname):
    for key, role in (("interaction_agent", "interaction"), ("execution_agent.runtime", "execution"),
                      ("importance_classifier", "classifier"), ("search_email", "search")):
        if key in modname:
            return role
    return "summarizer"


def patch_seams(tmp_gmail):
    for name, mod in list(sys.modules.items()):
        if not name.startswith("server") or mod is None:
            continue
        if hasattr(mod, "request_chat_completion") and not name.startswith("server.openrouter_client"):
            mod.request_chat_completion = make_fake(role_of(name))
        if hasattr(mod, "execute_gmail_tool"):
            mod.execute_gmail_tool = tmp_gmail
        if hasattr(mod, "get_active_gmail_user_id"):
            mod.get_active_gmail_user_id = lambda *a, **k: "fake-user"


INBOX = []   # what the fake Gmail returns


def fake_gmail(tool, user_id, arguments=None, **kw):
    return {"data": {"messages": [dict(m) for m in INBOX]}}


def worker_setup(tree, work, data):
    shutil.copytree(Path(tree) / "server", Path(work) / "server", ignore=shutil.ignore_patterns("__pycache__", "data"))
    sys.path.insert(0, str(work))
    sys.path.insert(0, str(HERE))
    os.environ["OPENROUTER_API_KEY"] = "fake-key"
    os.environ["OPENPOKE_DATA_DIR"] = str(data or Path(work) / "server" / "data")   # ignored by the original code
    from baseline import install_fcntl_stub
    install_fcntl_stub()   # the ORIGINAL roster imports fcntl (POSIX only)
    import importlib, logging
    importlib.import_module("server.app")
    for mod in ("server.services.trigger_scheduler", "server.services.gmail.importance_watcher",
                "server.services.conversation.chat_handler"):
        importlib.import_module(mod)
    logging.disable(logging.CRITICAL)
    patch_seams(fake_gmail)


def reset_all():
    """Clear data and reset module singletons (guards: the original has no such helpers)."""
    from server.agents.execution_agent.batch_manager import ExecutionBatchManager
    from server.agents.interaction_agent import runtime as irt, tools as itools
    from server.services.conversation import get_conversation_log
    from server.services.execution import get_agent_roster, get_execution_agent_logs
    get_agent_roster().clear()
    get_execution_agent_logs().clear_all()
    get_conversation_log().clear()
    itools._EXECUTION_BATCH_MANAGER = ExecutionBatchManager()
    if hasattr(irt, "reset_chat_queue"):
        irt.reset_chat_queue()
    try:
        from server.services.execution import limits
        limits.reset()
    except ImportError:
        pass
    M.reset()


async def drain(timeout=60):
    from baseline import drain as d
    await d(timeout)


def norm(s):
    return re.sub(r'timestamp="[^"]*"', 'timestamp="T"', s)


def replies():
    from server.services.conversation import get_conversation_log
    return [p for t, _, p in get_conversation_log().iter_entries() if t == "poke_reply"]


def conv_entries():
    from server.services.conversation import get_conversation_log
    return [[t, p] for t, _, p in get_conversation_log().iter_entries()]


def seed_agents(names, pairs=2):
    from server.services.execution import get_agent_roster, get_execution_agent_logs
    for n in names:
        get_agent_roster().add_agent(n)
        for p in range(pairs):
            get_execution_agent_logs().record_request(n, f"Follow up on item {p} for {n}.")
            get_execution_agent_logs().record_agent_response(n, f"Checked the inbox for {n}, item {p}.")


def tail(first):
    return first.rsplit("<new_user_message>", 1)[-1]


# ---- scenarios -----------------------------------------------------------
async def w_prompts(p):
    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.services.execution import get_execution_agent_logs
    reset_all()
    names = ["invoice-followups-acme", "flight-booking-march", "dentist-reschedule-q3"]
    seed_agents(names)
    await InteractionAgentRuntime().execute("What is on my plate today?")
    await drain()
    ia = next(c for c in M.calls if c["role"] == "interaction")
    logs = get_execution_agent_logs()
    reset_all()
    n = "invoice-followups-acme"
    for tag, text in (("request", "Send a reminder about invoice 42."), ("action", "Calling gmail_send with: {}"),
                      ("tool", "gmail_send: ok"), ("agent_response", "Sent the reminder."),
                      ("request", "Check for a reply <now> & report.")):
        {"request": logs.record_request, "action": logs.record_action,
         "agent_response": logs.record_agent_response}.get(tag, lambda a, t: logs.record_tool_response(a, "gmail_send", t))(n, text)
    await ExecutionAgentRuntime(n).execute("Send a reminder about invoice 42.")
    ex = next(c for c in M.calls if c["role"] == "execution")
    return {"interaction": ia["first"], "interaction_system": ia["system"], "interaction_model": ia["model"],
            "execution": norm(ex["system"]), "execution_model": ex["model"],
            "names_in_roster_block": all(f'name="{x}"' in ia["first"] for x in names),
            "log_entries_in_exec_prompt": ex["system"].count("</")}


def snapshot():
    from server.services.conversation import get_conversation_log, get_working_memory_log
    from server.services.execution import get_agent_roster, get_execution_agent_logs
    from server.services.gmail.importance_watcher import ImportantEmailWatcher
    from server.services.triggers import get_trigger_service
    roster, logs, svc = get_agent_roster(), get_execution_agent_logs(), get_trigger_service()
    names = roster.get_agents()
    raw = json.loads(roster._roster_path.read_text(encoding="utf-8"))
    return {"roster": names, "roster_file": raw, "transcripts": {n: logs.load_transcript(n) for n in names},
            "triggers": {n: [t.model_dump() for t in svc.list_triggers(agent_name=n)] for n in names},
            "seen": ImportantEmailWatcher()._seen_store.snapshot(),
            "conversation": get_conversation_log().load_transcript(),
            "working_memory": get_working_memory_log().render_transcript()}


async def w_legacy_seed(p):
    from server.services.conversation import get_conversation_log
    from server.services.execution import get_agent_roster, get_execution_agent_logs
    from server.services.gmail.seen_store import GmailSeenStore
    from server.services.triggers import get_trigger_service
    reset_all()
    names = ["Vercel Job Offer", "Email to Sharanjeet", "gym <membership> & more"]
    logs, svc = get_execution_agent_logs(), get_trigger_service()
    for n in names:
        get_agent_roster().add_agent(n)
        logs.record_request(n, f"Handle {n}.\nSecond line with <tag> & ampersand.")
        logs.record_action(n, "Calling gmail_fetch_emails with: {}")
        logs.record_tool_response(n, "gmail_fetch_emails", "found 2 threads")
        logs.record_agent_response(n, "Done.\nAll set.")
    svc.clear_all()
    svc.create_trigger(agent_name=names[0], payload="Daily digest", recurrence_rule="FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
                       timezone_name="UTC")
    svc.create_trigger(agent_name=names[1], payload="One-off nudge", start_time="2030-06-01T10:00:00",
                       timezone_name="UTC")
    GmailSeenStore(get_agent_roster()._roster_path.parent.parent / "gmail_seen.json", 300).mark_seen(["m1", "m2", "m3"])
    c = get_conversation_log()
    c.record_user_message("Plan my trip\nplease")
    c.record_agent_message("[SUCCESS] Vercel Job Offer: ok")
    c.record_reply("On it & done <3")
    return snapshot()


async def w_legacy_load(p):
    return snapshot()


def due_past(svc, rec):
    from server.services.triggers.utils import to_storage_timestamp, utc_now
    svc._store.update(rec.id, rec.agent_name, {"next_trigger": to_storage_timestamp(utc_now() - dt.timedelta(minutes=1))})


async def w_reminders(p):
    from server.services.trigger_scheduler import TriggerScheduler
    from server.services.triggers import get_trigger_service
    from server.services.triggers.utils import parse_iso, utc_now
    svc, out = get_trigger_service(), {}
    for key, rule in (("daily_9am", "FREQ=DAILY;BYHOUR=9;BYMINUTE=0"), ("hourly", "FREQ=HOURLY")):
        reset_all()
        svc.clear_all()
        agent = f"reminder-{key}"
        try:
            rec = svc.create_trigger(agent_name=agent, payload="Remind me.", recurrence_rule=rule, timezone_name="UTC")
        except Exception as exc:
            out[key] = {"accepted": False, "error": str(exc)}
            continue
        due_past(svc, rec)
        await TriggerScheduler()._poll_once()
        await drain()
        after = svc.list_triggers(agent_name=agent)[0]
        out[key] = {"accepted": True, "fired": sum(1 for c in M.calls if c["role"] == "execution" and f"Agent Name: {agent}" in c["system"]),
                    "rescheduled_to_future": bool(after.next_trigger) and parse_iso(after.next_trigger) > utc_now(),
                    "last_error": after.last_error}
    return out


async def w_chat_script(p):
    from server.models import ChatMessage, ChatRequest
    from server.services.conversation.chat_handler import handle_chat_request

    def interaction(rec):
        if rec["n"] > 1:
            return txt("Okay.")
        if "<new_agent_message>" in rec["first"]:
            return tc(("send_message_to_user", {"message": "All done: trip plan ready."}))
        return tc(("send_message_to_agent", {"agent_name": "trip-planner", "instructions": "Plan the trip"}),
                  ("send_message_to_user", {"message": "On it."}))
    reset_all()
    M.handlers["interaction"] = interaction
    resp = await handle_chat_request(ChatRequest(messages=[ChatMessage(role="user", content="Plan my trip")]))
    await drain()
    return {"status": resp.status_code, "sequence": [[c["role"], c["n"], c["tools"]] for c in M.calls if c["role"] != "summarizer"],
            "conversation": [[t, norm(x)] for t, x in conv_entries()]}


async def w_one_email(p):
    global INBOX
    from baseline import make_inbox
    from server.services.gmail.importance_watcher import ImportantEmailWatcher
    from server.services.gmail.seen_store import GmailSeenStore
    from server.services.conversation import get_conversation_log
    reset_all()
    summary = "Alice (person0@example.com) needs your sign-off on item 0 today."
    M.handlers["classifier"] = lambda rec: tc(("mark_email_importance", {"important": True, "summary": summary}))
    M.handlers["interaction"] = lambda rec: txt("ok") if rec["n"] > 1 else tc(("send_message_to_user", {"message": "Heads up: " + summary}))
    w = ImportantEmailWatcher(seen_store=GmailSeenStore(Path(tempfile.mkdtemp()) / "seen.json", 300))
    INBOX = []
    await w._poll_once()
    INBOX = make_inbox(1)
    await w._poll_once()
    await drain()
    agent_msgs = [x for t, _, x in get_conversation_log().iter_entries() if t == "agent_message"]
    return {"agent_messages": agent_msgs, "replies": replies(), "summary": summary, "sender": "person0@example.com",
            "interaction_runs": M.n("interaction", True)}


async def w_agents_n(p):
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.services.execution import get_agent_roster
    out = {}
    for n in range(1, 5):
        reset_all()
        M.handlers["interaction"] = lambda rec, n=n: txt("ok") if rec["n"] > 1 or "<new_agent_message>" in rec["first"] else tc(
            *[("send_message_to_agent", {"agent_name": f"worker-{i}", "instructions": f"task {i}"}) for i in range(n)])
        await InteractionAgentRuntime().execute("do several things")
        await drain()
        started = {m.group(1) for c in M.calls if c["role"] == "execution" for m in [re.search(r"Agent Name: (\S+)", c["system"])] if m}
        rejected = sum(1 for c in M.calls if c["role"] == "interaction" and c["n"] > 1
                       for m in c["msgs"] if m.get("role") == "tool" and '"status": "error"' in m["content"])
        out[str(n)] = {"requested": n, "started": len(started), "rejected": rejected, "roster": len(get_agent_roster().get_agents())}
    return out


# ---- gap scenarios ---------------------------------------------------------
def user_visible_handler(rec):
    if rec["n"] > 1:
        return txt("")
    body = rec["first"].rsplit("<new_agent_message>", 1)[-1].strip().splitlines()
    return tc(("send_message_to_user", {"message": "Heads up: " + (body[1] if len(body) > 1 else "update")}))


async def w_flood(p):
    global INBOX
    from baseline import make_inbox
    from server.services.gmail.importance_watcher import ImportantEmailWatcher
    from server.services.gmail.seen_store import GmailSeenStore
    reset_all()
    M.handlers["interaction"] = user_visible_handler
    M.handlers["classifier"] = lambda rec: tc(("mark_email_importance", {"important": True, "summary": "Reply needed " + re.search(r"#\d+", rec["first"]).group(0)}))
    w = ImportantEmailWatcher(seen_store=GmailSeenStore(Path(tempfile.mkdtemp()) / "seen.json", 300))
    INBOX = []
    await w._poll_once()
    INBOX = make_inbox(p["f"])
    t0 = time.perf_counter()
    await w._poll_once()
    await drain()
    from server.services.conversation import get_conversation_log
    named = {m for t, _, x in get_conversation_log().iter_entries() if t == "agent_message" for m in re.findall(r"#\d+", x)}
    return {"model_calls": len(M.calls), "classifier_calls": M.n("classifier"), "interaction_runs": M.n("interaction", True),
            "user_visible": len(replies()), "emails_named_in_text": len(named), "wall_s": round(time.perf_counter() - t0, 2)}


async def w_burst_triggers(p):
    from server.services.trigger_scheduler import TriggerScheduler
    from server.services.triggers import TriggerRecord
    reset_all()
    M.latency["execution"] = 0.2
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    recs = [TriggerRecord(id=i + 1, agent_name=f"trigger-agent-{i}", payload="Check the inbox.", status="active",
                          next_trigger=now, created_at=now, updated_at=now) for i in range(20)]

    class Svc:
        def get_due_triggers(self, **kw): return list(recs)
        def schedule_next_occurrence(self, *a, **k): pass
        def record_failure(self, *a, **k): pass
        def clear_next_fire(self, *a, **k): pass
    s = TriggerScheduler()
    s._service = Svc()
    t0 = time.perf_counter()
    await s._poll_once()
    await drain()
    return {"due": 20, "peak_parallel_agents": M.peak["execution"], "agents_run": M.n("execution"),
            "wall_s": round(time.perf_counter() - t0, 2)}


async def w_burst_agents(p):
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    reset_all()
    M.latency["execution"] = 0.2
    M.handlers["interaction"] = lambda rec: txt("ok") if rec["n"] > 1 or "<new_agent_message>" in rec["first"] else tc(
        *[("send_message_to_agent", {"agent_name": f"burst-agent-{i}", "instructions": f"task {i}"}) for i in range(8)])
    await InteractionAgentRuntime().execute("do eight things")
    await drain()
    return {"requested": 8, "agents_run": M.n("execution"), "peak_parallel_agents": M.peak["execution"]}


async def w_chat(p):
    from server.models import ChatMessage, ChatRequest
    from server.services.conversation.chat_handler import handle_chat_request
    reset_all()
    M.latency["interaction"] = 0.1
    rs = await asyncio.gather(*(handle_chat_request(ChatRequest(messages=[ChatMessage(role="user", content=f"message {i}")])) for i in range(p["n"])))
    await drain()
    codes = collections.Counter(r.status_code for r in rs)
    return {"messages": p["n"], "accepted_202": codes[202], "rejected_429": codes[429],
            "peak_overlapping_runs": M.peak["interaction"], "runs": M.n("interaction", True)}


async def w_chat_fail(p):
    from server.models import ChatMessage, ChatRequest
    from server.services.conversation.chat_handler import handle_chat_request
    reset_all()
    M.handlers["interaction"] = lambda rec: tc(("wait", {"reason": f"still thinking {rec['n']}"}))
    await handle_chat_request(ChatRequest(messages=[ChatMessage(role="user", content="loop forever")]))
    await drain()
    return {"visible_replies": replies(), "model_calls": M.n("interaction")}


async def w_classifier_fail(p):
    global INBOX
    from baseline import make_inbox
    from server.openrouter_client import OpenRouterError
    from server.services.gmail.importance_watcher import ImportantEmailWatcher
    from server.services.gmail.seen_store import GmailSeenStore
    reset_all()
    ok, tried = set(), set()

    def classify(rec):
        i = int(re.search(r"#(\d+)", rec["first"]).group(1))
        if i % 10 < 3 and i not in tried:       # 30 percent fail on their first attempt only
            tried.add(i)
            raise OpenRouterError("fake classifier outage")
        ok.add(i)
        return tc(("mark_email_importance", {"important": True, "summary": f"Reply needed #{i}"}))
    M.handlers["classifier"] = classify
    M.handlers["interaction"] = user_visible_handler
    store = GmailSeenStore(Path(tempfile.mkdtemp()) / "seen.json", 300)
    w = ImportantEmailWatcher(seen_store=store)
    INBOX = []
    await w._poll_once()
    INBOX = make_inbox(50)
    for _ in range(4):
        await w._poll_once()
        await drain()
    seen = {i for i in range(50) if store.is_seen(f"msg{i}")}
    return {"emails": 50, "first_attempt_failures": len(tried), "surfaced_after_4_polls": len(ok),
            "lost_seen_but_never_surfaced": len(seen - ok), "still_unseen": 50 - len(seen)}


RULES = ["FREQ=DAILY;BYHOUR=9;BYMINUTE=0", "FREQ=WEEKLY;BYDAY=MO", "FREQ=HOURLY", "FREQ=MINUTELY;INTERVAL=15",
         "FREQ=MINUTELY;INTERVAL=5", "FREQ=MINUTELY;INTERVAL=2", "FREQ=MINUTELY", "FREQ=SECONDLY;INTERVAL=30",
         "FREQ=SECONDLY", "FREQ=DAILY;BYHOUR=9;BYMINUTE=0;BYSECOND=0,10,20", "FREQ=HOURLY;BYMINUTE=0,1,2",
         "FREQ=SOMETIMES;BANANA"]


async def w_floor(p):
    from server.services.triggers import get_trigger_service
    svc = get_trigger_service()
    svc.clear_all()
    out = []
    for i, rule in enumerate(RULES):
        try:
            svc.create_trigger(agent_name=f"floor-{i}", payload="p", recurrence_rule=rule,
                               start_time="2030-01-01T09:01:00", timezone_name="UTC")
            out.append({"rule": rule, "accepted": True, "message": ""})
        except Exception as exc:
            out.append({"rule": rule, "accepted": False, "message": f"{type(exc).__name__}: {exc}"[:110]})
    return out


async def w_retry(p):
    import httpx
    from server.openrouter_client import client as oc
    try:
        from server.services import usage
    except ImportError:
        usage = None
    real = httpx.AsyncClient
    state = {"calls": 0, "script": []}
    ok = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002}}

    def handler(request):
        state["calls"] += 1
        return state["script"].pop(0) if state["script"] else httpx.Response(500, json={"error": "script exhausted"})

    async def no_sleep(s):
        pass
    oc.httpx.AsyncClient = lambda **kw: real(transport=httpx.MockTransport(handler))
    if hasattr(oc, "_sleep"):
        oc._sleep = no_sleep
    out = {}
    for key, script in (("429_then_200", [429, 200]), ("503_x3", [503, 503, 503]), ("401", [401])):
        state["calls"], state["script"] = 0, [httpx.Response(s, json=ok if s == 200 else {"error": "x"}) for s in script]
        if usage:
            usage.reset()
        try:
            await oc.request_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}], api_key="k")
            outcome = "ok"
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {str(exc)[:60]}"
        u = usage.get_usage() if usage else None
        out[key] = {"attempts": state["calls"], "outcome": outcome,
                    "usage": ({k: u[k] for k in ("calls", "retries", "failed_calls", "prompt_tokens", "completion_tokens")} if u else "not available")}
    return out


WORKERS = {k[2:]: v for k, v in globals().items() if k.startswith("w_")}


def worker_main(args):
    worker_setup(args.tree, args.work, args.data)
    params = json.loads(args.params or "{}")
    res = asyncio.run(WORKERS[args.worker](params))
    print(MARK + json.dumps(res), flush=True)


# =============================== ORCHESTRATOR ===============================
class Runner:
    def __init__(self, orig, new):
        self.trees = {"orig": Path(orig), "new": Path(new)}
        self.temps, self.cache = [], {}

    def tmp(self):
        d = Path(tempfile.mkdtemp(prefix="regeval_"))
        self.temps.append(d)
        return d

    def run(self, side, worker, params=None, work=None, data=None, timeout=150):
        key = (side, worker, json.dumps(params or {}, sort_keys=True), str(data))
        if key in self.cache and work is None:
            return self.cache[key]
        work = work or self.tmp()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", worker, "--tree", str(self.trees[side]),
               "--work", str(work), "--params", json.dumps(params or {})] + (["--data", str(data)] if data else [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"_error": f"timeout after {timeout}s"}
        line = next((l for l in r.stdout.splitlines()[::-1] if l.startswith(MARK)), None)
        res = json.loads(line[len(MARK):]) if line else {"_error": (r.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        self.cache[key] = res
        return res

    def pair(self, worker, params=None):
        return self.run("orig", worker, params), self.run("new", worker, params)

    def cleanup(self):
        for d in self.temps:
            shutil.rmtree(d, ignore_errors=True)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def verdict(ok, reason, **extra):
    return {"status": "PASS" if ok else "FAIL", "reason": reason, **extra}


def errs(*rs):
    e = [r["_error"] for r in rs if isinstance(r, dict) and "_error" in r]
    return verdict(False, "scenario crashed: " + "; ".join(e)) if e else None


def chk_small_roster(R):
    o, n = R.pair("prompts")
    if errs(o, n): return errs(o, n)
    same, real = o["interaction"] == n["interaction"] and o["interaction_system"] == n["interaction_system"], o["names_in_roster_block"] and n["names_in_roster_block"]
    return verdict(same and real, f"3-agent interaction prompt {'byte-identical' if same else 'DIFFERS'} ({len(o['interaction'])} chars, sha {sha(o['interaction'])}/{sha(n['interaction'])})"
                   + ("" if real else "; roster block missing the names"))


def chk_small_log(R):
    o, n = R.pair("prompts")
    if errs(o, n): return errs(o, n)
    same = o["execution"] == n["execution"]
    return verdict(same and o["log_entries_in_exec_prompt"] >= 5, f"5-entry execution system prompt {'byte-identical' if same else 'DIFFERS'} (timestamps normalised; {len(o['execution'])} chars, sha {sha(o['execution'])}/{sha(n['execution'])})")


def chk_legacy(R):
    work_o = R.tmp()
    o = R.run("orig", "legacy_seed", work=work_o)
    n = R.run("new", "legacy_load", work=R.tmp(), data=work_o / "server" / "data")
    if errs(o, n): return errs(o, n)
    parts = {k: o[k] == n[k] for k in ("roster", "roster_file", "transcripts", "triggers", "seen", "conversation", "working_memory")}
    bad = [k for k, v in parts.items() if not v]
    solid = len(o["roster"]) == 3 and len(sum(o["triggers"].values(), [])) == 2 and o["seen"] == ["m1", "m2", "m3"] and o["conversation"]
    return verdict(not bad and solid, "legacy roster list, agent logs, 2 triggers, seen ids, conversation all load identically" if not bad and solid else f"mismatch in {bad or 'seed (empty)'}")


def chk_reminders(R):
    o, n = R.pair("reminders")
    if errs(o, n): return errs(o, n)
    good = all(r.get("accepted") and r["fired"] >= 1 and r["rescheduled_to_future"] and not r["last_error"] for r in n.values())
    return verdict(good and o == n, f"daily 9am and hourly: accepted, fired {[n[k].get('fired') for k in n]}x, rescheduled; same as original" if good and o == n else f"orig={o} new={n}")


def chk_chat(R):
    o, n = R.pair("chat_script")
    if errs(o, n): return errs(o, n)
    ok = o == n and len(o["sequence"]) >= 3 and any(t == "poke_reply" for t, _ in o["conversation"])
    return verdict(ok, f"same {len(o['sequence'])} model calls, same tool sequence, same recorded replies" if ok else f"orig={o['sequence']} new={n['sequence']}")


def chk_email(R):
    o, n = R.pair("one_email")
    if errs(o, n): return errs(o, n)
    text = " ".join(n["agent_messages"])
    ok = o["agent_messages"] == n["agent_messages"] and len(n["agent_messages"]) == 1 and n["sender"] in text and n["summary"] in text and o["replies"] == n["replies"] and len(n["replies"]) == 1
    return verdict(ok, "one important email: delivered once, identical text (sender + summary present), user reply recorded" if ok else f"orig={o} new={n}")


def chk_agents(R):
    o, n = R.pair("agents_n")
    if errs(o, n): return errs(o, n)
    ok = all(n[k]["started"] == int(k) and n[k]["rejected"] == 0 and o[k]["started"] == int(k) for k in n)
    return verdict(ok, "1..4 agents in one turn: all started, none rejected (same as original)" if ok else f"orig={o} new={n}")


CHECKS = {1: ("small roster prompt identical", chk_small_roster), 2: ("small log execution prompt identical", chk_small_log),
          3: ("legacy data loads", chk_legacy), 4: ("valid reminders accepted and fire", chk_reminders),
          5: ("single chat message: same tool calls and reply", chk_chat), 6: ("one important email still reaches user", chk_email),
          7: ("1-4 agents per turn all start", chk_agents)}


def intended(R):
    p_o, p_n = R.pair("prompts")
    model = f"{p_o.get('interaction_model')} -> {p_n.get('interaction_model')}"
    return [
        ("default model name", f"{model}; the old model is no longer served; env overridable (OPENPOKE_MODEL)"),
        ("roster larger than 11 agents", "prompt shows top-K + recent agents instead of all (bounded view, the overload fix)"),
        ("execution log over 20 entries / 12000 chars", "only newest whole entries are injected, with a header noting omitted ones"),
        ("2 or more important emails in one poll", "one digest message instead of one interaction run each; single email keeps old format"),
        ("more than 4 agents in one turn / more than 5 at once", "extra starts rejected with a message; extra runs wait for a slot"),
        ("user chat while 10 are already waiting", "HTTP 429 with a plain message; chat runs one at a time"),
        ("recurrence faster than 5 minutes, or unparseable", "rejected when created"),
        ("model call fails with 429/5xx/timeout", "retried with backoff (3 attempts)"),
        ("tool loop hits the step limit", "user gets a plain failure reply instead of silence"),
        ("classifier error", "email is retried (up to 3 attempts) instead of marked seen"),
    ]


def run_gaps(R, mode):
    g = {}
    sizes = (10,) if mode == "quick" else (10, 50, 200)
    for f in sizes:
        g[f"flood_{f}"] = R.pair("flood", {"f": f})
    g["floor"] = R.pair("floor")
    if mode == "quick":
        return g
    g["burst_triggers"] = R.pair("burst_triggers")
    g["burst_agents"] = R.pair("burst_agents")
    for c in (5, 20):
        g[f"chat_{c}"] = R.pair("chat", {"n": c})
    g["chat_fail"] = R.pair("chat_fail")
    g["classifier_fail"] = R.pair("classifier_fail")
    g["retry"] = R.pair("retry")
    return g


def gap_rows(g):
    rows = []

    def v(r, key, default="crashed"):
        return r.get(key, r.get("_error", default)) if isinstance(r, dict) else default
    for k, (o, n) in g.items():
        if k.startswith("flood_"):
            f = k.split("_")[1]
            for key in ("model_calls", "interaction_runs", "user_visible", "emails_named_in_text", "wall_s"):
                rows.append((f"email flood F={f}: {key}", v(o, key), v(n, key)))
        elif k == "burst_triggers":
            rows += [("20 due triggers: peak parallel agents", v(o, "peak_parallel_agents"), v(n, "peak_parallel_agents")),
                     ("20 due triggers: wall s", v(o, "wall_s"), v(n, "wall_s"))]
        elif k == "burst_agents":
            rows += [("8 agents in one turn: peak parallel", v(o, "peak_parallel_agents"), v(n, "peak_parallel_agents")),
                     ("8 agents in one turn: agents actually run", v(o, "agents_run"), v(n, "agents_run"))]
        elif k.startswith("chat_") and k != "chat_fail":
            c = k.split("_")[1]
            rows += [(f"{c} chat messages: peak overlapping runs", v(o, "peak_overlapping_runs"), v(n, "peak_overlapping_runs")),
                     (f"{c} chat messages: rejected with 429", "0 (no cap)", v(n, "rejected_429"))]
        elif k == "chat_fail":
            rows.append(("endless tool loop: visible reply to user", "none" if not o.get("visible_replies") else o["visible_replies"][0],
                         (n.get("visible_replies") or ["none"])[0]))
        elif k == "classifier_fail":
            for key in ("first_attempt_failures", "surfaced_after_4_polls", "lost_seen_but_never_surfaced"):
                rows.append((f"classifier fails 30% of 50 emails: {key}", v(o, key), v(n, key)))
        elif k == "retry":
            for case in ("429_then_200", "503_x3", "401"):
                rows.append((f"HTTP {case}: attempts / outcome", f"{o[case]['attempts']} / {o[case]['outcome']}", f"{n[case]['attempts']} / {n[case]['outcome']}"))
                u = n[case]["usage"]
                rows.append((f"HTTP {case}: usage counters", "not available", u if isinstance(u, str) else f"calls={u['calls']} retries={u['retries']} failed={u['failed_calls']} tokens={u['prompt_tokens']}+{u['completion_tokens']}"))
        elif k == "floor":
            for a, b in zip(o, n):
                rows.append((f"trigger rule {a['rule']}", "accepted" if a["accepted"] else "rejected: " + a["message"][:50],
                             "accepted" if b["accepted"] else "rejected: " + b["message"][:100]))
    return rows


def table(rows, head):
    w0, w1 = max(len(str(r[0])) for r in rows), min(max(len(str(r[1])) for r in rows), 56)
    lines = [f"{head[0].ljust(w0)}  {head[1].ljust(w1)}  {head[2]}"]
    lines += [f"{str(a).ljust(w0)}  {str(b).ljust(w1)[:max(w1, 0)]}  {c}" for a, b, c in rows]
    return "\n".join(lines)


def evaluate(orig, new, checks=tuple(CHECKS), gaps="all", out=None, quiet=False):
    t0 = time.time()
    R = Runner(orig, new)
    try:
        reg = []
        for i in checks:
            name, fn = CHECKS[i]
            try:
                res = fn(R)
            except Exception as exc:
                res = verdict(False, f"checker crashed: {type(exc).__name__}: {exc}")
            reg.append({"id": i, "name": name, **res})
        diffs = intended(R) if checks and set(checks) & {1} else []
        g = run_gaps(R, gaps) if gaps != "none" else {}
        rows = gap_rows(g) if g else []
    finally:
        R.cleanup()
    result = {"meta": {"orig": str(orig), "new": str(new), "seconds": round(time.time() - t0, 1),
                       "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "fake_latency_s": LATENCY},
              "regression": reg, "intended_differences": [{"when": a, "what": b} for a, b in diffs],
              "gaps": {k: {"original": o, "new": n} for k, (o, n) in g.items()}}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not quiet:
        print("A. REGRESSION (original vs improved; must match)")
        for r in reg:
            print(f"  [{r['status']}] {r['id']}. {r['name']}: {r['reason']}")
        if diffs:
            print("\n  intended differences (not failures):")
            for a, b in diffs:
                print(f"    - {a}: {b}")
        if rows:
            print("\nB. GAPS (same scenario on both trees; reported, never failed)")
            print(table(rows, ("scenario", "original", "new")))
        print(f"\nregression: {sum(r['status'] == 'PASS' for r in reg)}/{len(reg)} PASS   total {result['meta']['seconds']}s")
        if out:
            print(f"wrote {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=str(default_orig()))
    ap.add_argument("--new", default=str(ROOT))
    ap.add_argument("--checks", default=",".join(map(str, CHECKS)))
    ap.add_argument("--gaps", default="all", choices=["all", "quick", "none"])
    ap.add_argument("--out", default=str(HERE / "results" / "regression_eval.json"))
    ap.add_argument("--worker"), ap.add_argument("--tree"), ap.add_argument("--work")
    ap.add_argument("--data"), ap.add_argument("--params")
    a = ap.parse_args()
    if a.worker:
        return worker_main(a)
    res = evaluate(a.orig, a.new, [int(x) for x in a.checks.split(",") if x], a.gaps, a.out)
    sys.exit(1 if any(r["status"] != "PASS" for r in res["regression"]) else 0)


if __name__ == "__main__":
    main()
