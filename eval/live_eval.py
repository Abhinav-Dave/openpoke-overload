"""LIVE routing check: ORIGINAL logic (full roster of names) vs NEW logic (bounded view + find_agents).

Dry run by default: prints the plan and a worst-case dollar estimate, makes NO network call.
  python eval/live_eval.py                      # plan only
  python eval/live_eval.py --run --max-usd 3    # calls the REAL model (needs OPENROUTER_API_KEY in the environment)

Both trees run the real interaction-agent turn with the real request_chat_completion. Only side effects are stubbed:
execution agents never start (the start is recorded), Gmail raises if touched. The original tree is
`git archive 9788701 server` (original code + the model-name fix). Each (tree, size, repeat) runs in its own
subprocess with its own data dir. The key is read from the process environment only; no .env file is read here.
Output: eval/results/live_eval.json (rows + summary). Never prints the key, prompts or email text.
"""
import argparse, asyncio, io, json, math, os, random, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_eval import CATEGORIES, SEED, answerable, build, load_data, with_clones  # noqa: E402

ORIG_COMMIT = "9788701"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
OUTCOMES = ("reused_expected", "reused_other", "created_new", "no_agent_call", "error")
MAX_ITER = 8          # InteractionAgentRuntime.MAX_TOOL_ITERATIONS
FAKE_ENV = "LIVE_EVAL_FAKE_HTTP"   # tests only: path to a JSON policy; model and /key calls go to a scripted MockTransport


def scrub(text):
    key = os.environ.get("OPENROUTER_API_KEY")
    return str(text).replace(key, "***") if key else str(text)


def say(*parts):
    print(scrub(" ".join(str(p) for p in parts)), flush=True)


# ---- pure helpers ----------------------------------------------------------
def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def classify(tool_calls, seeded, expected, starts=None):
    """tool_calls: [(tool_name, agent_name_or_None)] in order.
    The decision is the first send_message_to_agent that was ACCEPTED. When `starts` (agents the stub really started)
    is given it decides, so a call refused by the find-before-create gate is not scored as a created agent."""
    find_idx = [i for i, (t, _) in enumerate(tool_calls) if t == "find_agents"]
    sends = [i for i, (t, n) in enumerate(tool_calls) if t == "send_message_to_agent" and isinstance(n, str) and n]
    if starts is None:
        agent = tool_calls[sends[0]][1] if sends else None
        choice_idx = sends[0] if sends else None
    else:
        agent = starts[0] if starts else None
        picked = [i for i in sends if tool_calls[i][1] == agent]
        choice_idx = picked[-1] if picked else None
    if agent is None:
        outcome = "no_agent_call"
    else:
        is_expected = expected and (agent == expected or re.fullmatch(re.escape(expected) + r"-\d+", agent))  # with_clones suffix
        outcome = "reused_expected" if is_expected else "reused_other" if agent in seeded else "created_new"
    return {"outcome": outcome, "agent": agent, "find_called": bool(find_idx),
            "find_before_choice": bool(find_idx and choice_idx is not None and find_idx[0] < choice_idx)}


def seed_followup_history(conversation_log, agent_name):
    """A realistic earlier exchange, recorded the way the runtime records it, so "send it" has something to refer to."""
    conversation_log.record_user_message("Please prepare that for me.")
    conversation_log.record_agent_message(f"[SUCCESS] {agent_name}: Draft ready for your review.")


def _rate(num, den):
    return round(num / den, 3) if den else None


def _correct(r):
    if r["expected"]:
        return r["outcome"] == "reused_expected"
    return r["outcome"] in ("created_new", "no_agent_call")


def metrics(rows):
    n = len(rows)
    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    exp = [r for r in rows if r["expected"]]
    calls = [r["calls"] for r in rows]
    secs = [r["seconds"] for r in rows]
    found = [r for r in rows if r["find_called"]]
    return {"n": n, "counts": counts,
            "correct_rate": _rate(sum(_correct(r) for r in rows), n),
            "duplicate_rate": _rate(sum(r["outcome"] == "created_new" for r in exp), len(exp)),  # expected agent existed
            "wrong_reuse_rate": _rate(counts["reused_other"], n),
            "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rows) / n) if n else None,
            "completion_tokens_mean": round(sum(r["completion_tokens"] for r in rows) / n) if n else None,
            "calls_mean": round(sum(calls) / n, 2) if n else None,
            "latency_mean": round(sum(secs) / n, 3) if n else None,
            "latency_p50": percentile(secs, 50), "latency_p95": percentile(secs, 95),
            "model_seconds_mean": round(sum(r["model_seconds"] for r in rows) / n, 3) if n else None,
            "loop_p95": percentile(calls, 95),
            "find_called_rate": _rate(len(found), n),
            "correct_rate_when_find_called": _rate(sum(_correct(r) for r in found), len(found))}


def summarize(rows):
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in rows)]
    return {"overall": metrics(rows), "by_category": {c: metrics([r for r in rows if r["category"] == c]) for c in cats}}


def consistency(rows):
    """Share of (size, query) groups with >=2 repeats whose outcome label is identical across repeats."""
    groups = {}
    for r in rows:
        groups.setdefault((r["size"], r["qid"]), []).append(r["outcome"])
    multi = [g for g in groups.values() if len(g) > 1]
    return _rate(sum(len(set(g)) == 1 for g in multi), len(multi))


def select_queries(queries, agents, split, limit):
    """Answerable queries of the split, first `limit` per category; each gets a stable qid (index in the file)."""
    tagged = [dict(q, qid=i) for i, q in enumerate(queries)]
    tagged = answerable(tagged, agents)
    if split != "all":
        tagged = [q for q in tagged if q["split"] == split]
    if limit:
        seen, out = {}, []
        for q in tagged:
            seen[q["category"]] = seen.get(q["category"], 0) + 1
            if seen[q["category"]] <= limit:
                out.append(q)
        tagged = out
    return tagged


def worst_case(tree, size):
    """Token upper bound for one query: 8 model calls, prompt growing ~300 tokens per call, 400 tokens out per call.
    Base prompt from session_context/benchmarks.md (chars/4): original ~2600 + 10 per agent, new ~3500 flat."""
    base = 2600 + 10 * size if tree == "orig" else 3500
    return MAX_ITER * base + 300 * MAX_ITER * (MAX_ITER - 1) // 2, MAX_ITER * 400


def cost(tin, tout, price_in, price_out):
    return (tin * price_in + tout * price_out) / 1e6


# ---- original tree ---------------------------------------------------------
def extract_original(dest):
    out = subprocess.run(["git", "archive", ORIG_COMMIT, "server"], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        safe = [m for m in tar.getmembers() if m.name.startswith("server") and ".." not in Path(m.name).parts
                and not os.path.isabs(m.name)]
        tar.extractall(dest, members=safe)


def copy_tree(src, dest):
    shutil.copytree(Path(src) / "server", Path(dest) / "server", ignore=shutil.ignore_patterns("__pycache__", "data"))


# ---- fake HTTP (tests only) -------------------------------------------------
_FAKE = None


def fake_transport(policy):
    import httpx
    import re
    key_calls = {"n": 0}

    def handler(request):
        if request.url.path.endswith("/key"):
            n, key_calls["n"] = key_calls["n"], key_calls["n"] + 1
            return httpx.Response(200, json={"data": {"usage": policy.get("key_usage_start", 0.0)
                                                      + n * policy.get("key_usage_step", 0.0)}})
        msgs = json.loads(request.content)["messages"]
        first = next(m["content"] for m in msgs if m.get("role") == "user")
        m = re.search(r"<new_user_message>\n(.*?)\n</new_user_message>", first, re.S)
        steps = policy["steps"].get(m.group(1) if m else "", [])
        i = sum(1 for x in msgs if x.get("role") == "assistant")
        msg = {"role": "assistant", "content": "ok"}
        if i < len(steps):
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": steps[i]["tool"], "arguments": json.dumps(steps[i]["args"])}}]}
        return httpx.Response(200, json={"choices": [{"message": msg}],
                                         "usage": {"prompt_tokens": len(request.content) // 4, "completion_tokens": 20}})
    return httpx.MockTransport(handler)


def _transport():
    global _FAKE
    path = os.environ.get(FAKE_ENV)
    if not path:
        return None
    if _FAKE is None:
        _FAKE = fake_transport(json.loads(Path(path).read_text(encoding="utf-8")))
    return _FAKE


def key_usage():
    """Account spend (USD) from OpenRouter GET /key, or None. Only used when the key is in the environment."""
    import httpx
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        with httpx.Client(transport=_transport(), timeout=15) as c:
            r = c.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
        v = r.json()["data"]["usage"] if r.status_code == 200 else None
        return float(v) if v is not None else None
    except Exception:
        return None


# ---- worker: one tree, one size, one repeat, in its own process ---------------
class RunState:
    def __init__(self):
        self.usage_missing = self.gmail_calls = 0
        self.reset()

    def reset(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.model_seconds, self.call_seconds = 0.0, []
        self.tool_calls, self.starts = [], []


def install_stubs(state):
    """Count every model call (by name, in every server module), record agent starts, forbid Gmail. Returns undo()."""
    from server.agents.execution_agent.batch_manager import ExecutionBatchManager
    from server.openrouter_client import client as oc
    real = oc.request_chat_completion
    undo = []

    def patch(obj, name, value):
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    async def counted(*a, **kw):
        t0 = time.perf_counter()
        resp = await real(*a, **kw)
        dt = time.perf_counter() - t0
        state.call_seconds.append(dt)
        state.model_seconds += dt
        usage = resp.get("usage") or {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(pt, int) or not isinstance(ct, int):   # ponytail: chars/4 guess if the provider omits usage
            state.usage_missing += 1
            chars = len(kw.get("system") or "") + sum(len(str(m.get("content", ""))) for m in kw.get("messages", []))
            pt, ct = chars // 4, 0
        state.calls += 1
        state.prompt_tokens += pt
        state.completion_tokens += ct
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        for t in msg.get("tool_calls") or []:
            fn = t.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            name = args.get("agent_name") if isinstance(args, dict) else None
            state.tool_calls.append((fn.get("name"), name if isinstance(name, str) else None))
        return resp

    async def no_start(self, agent_name, instructions, request_id=None):
        state.starts.append(agent_name)
        return SimpleNamespace(success=True, agent_name=agent_name, response="", error=None)

    def no_gmail(*a, **k):
        state.gmail_calls += 1
        raise RuntimeError("Gmail must never be called in the live routing check")
    no_gmail._live_eval = True

    patch(ExecutionBatchManager, "execute_agent", no_start)
    for name, mod in list(sys.modules.items()):
        if not name.startswith("server") or mod is None:
            continue
        if hasattr(mod, "request_chat_completion") and not name.startswith("server.openrouter_client"):
            patch(mod, "request_chat_completion", counted)
        if hasattr(mod, "execute_gmail_tool"):
            patch(mod, "execute_gmail_tool", no_gmail)

    def undo_all():
        for obj, name, old in reversed(undo):
            setattr(obj, name, old)
    return undo_all


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)
    rest = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if rest:
        await asyncio.wait(rest, timeout=5)


def worker_main(a):
    src = Path(a.src).resolve()
    sys.path.insert(0, str(src))
    os.environ.setdefault("OPENPOKE_MODEL", DEFAULT_MODEL)
    if os.environ.get(FAKE_ENV):
        os.environ.setdefault("OPENROUTER_API_KEY", "fake-key")
    os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke_live_")   # ignored by the original tree
    from baseline import install_fcntl_stub
    install_fcntl_stub()   # the original roster.py imports fcntl (POSIX only)
    import importlib, logging
    for mod in ("server.agents.interaction_agent.runtime", "server.agents.execution_agent.runtime",
                "server.services.conversation.summarization.summarizer", "server.services.gmail.importance_classifier",
                "server.agents.execution_agent.tasks.search_email.tool", "server.services.gmail.importance_watcher",
                "server.services.trigger_scheduler", "server.services.conversation.chat_handler"):
        try:
            importlib.import_module(mod)
        except Exception:
            pass
    logging.disable(logging.CRITICAL)
    if _transport() is not None:
        import httpx
        real_client, tr = httpx.AsyncClient, _transport()
        httpx.AsyncClient = lambda *x, **k: real_client(*x, transport=tr, **k)
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.services.conversation import get_conversation_log, get_working_memory_log
    from server.services.execution import get_agent_roster, get_execution_agent_logs

    agents, _ = load_data()
    queries = json.loads(Path(a.queries_file).read_text(encoding="utf-8"))
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    real = json.loads(Path(a.real_file).read_text(encoding="utf-8"))["agents"] if a.real_file else []
    seeded_agents = roster_for(agents, a.size, real)
    build(roster, logs, seeded_agents)
    seeded = {x["name"] for x in seeded_agents}
    pristine = Path(tempfile.mkdtemp(prefix="openpoke_pristine_")) / "snap"
    base_dir = logs._base_dir
    shutil.copytree(base_dir, pristine)   # copy2 keeps log mtimes, which decide which agents are "hot"

    state = RunState()
    install_stubs(state)
    rows, spent, stopped = [], 0.0, False
    for i, q in enumerate(queries):
        if spent >= a.budget_left:
            stopped = True
            break
        shutil.rmtree(base_dir)           # undo anything the previous query created or appended
        shutil.copytree(pristine, base_dir)
        roster.load()
        for log in (get_conversation_log(), get_working_memory_log()):
            try:
                log.clear()
            except Exception:
                pass
        if q.get("last_used"):            # follow-up: the user just used this agent
            now = time.time()
            os.utime(logs._log_path(q["last_used"]), (now, now))
            seed_followup_history(get_conversation_log(), q["last_used"])
        state.reset()
        roster_size = len(roster.get_agents())
        t0 = time.perf_counter()
        res = asyncio.run(_one(InteractionAgentRuntime, q["text"]))
        seconds = time.perf_counter() - t0
        c = classify(state.tool_calls, seeded, q["expected"], state.starts)
        error = None if res.success else scrub(res.error or "failed")[:120]
        if error and not state.tool_calls:
            c["outcome"] = "error"
        spent += cost(state.prompt_tokens, state.completion_tokens, a.price_in, a.price_out)
        rows.append({"tree": a.tree, "size": a.size, "repeat": a.repeat, "qid": q["qid"], "category": q["category"],
                     "expected": q["expected"], **c, "calls": state.calls, "prompt_tokens": state.prompt_tokens,
                     "completion_tokens": state.completion_tokens, "roster_size": roster_size, "error": error,
                     "seconds": round(seconds, 3), "model_seconds": round(state.model_seconds, 3),
                     "slowest_call_seconds": round(max(state.call_seconds, default=0.0), 3)})
        say(f"[{a.tree} n={a.size} r={a.repeat}] {i + 1}/{len(queries)} {q['category']} -> {c['outcome']}")
        _write_json(a.rows_out, {"rows": rows, "spent_usd": spent, "stopped": False,
                                 "usage_missing": state.usage_missing, "gmail_calls": state.gmail_calls})
    _write_json(a.rows_out, {"rows": rows, "spent_usd": spent, "stopped": stopped,
                             "usage_missing": state.usage_missing, "gmail_calls": state.gmail_calls})
    return 0


async def _one(runtime_cls, text):
    res = await runtime_cls().execute(text)
    await _settle()
    return res


# ---- parent -----------------------------------------------------------------
def roster_for(agents, size, real):
    """Synthetic agents (and clones) padded to `size`, plus any real-inbox agents that are always included."""
    out = with_clones(agents, max(size - len(real), 0), random.Random(SEED)) + [dict(r, cloned=False) for r in real]
    random.Random(SEED).shuffle(out)
    return out


def size_queries(queries, agents, size, split, limit):
    roster = with_clones(agents, size, random.Random(SEED))
    return answerable(select_queries(queries, agents, split, limit), roster)


def print_plan(a, sizes, per_size):
    trees = a.trees
    runs = sum(len(per_size[s]) for s in sizes) * len(trees) * a.repeats
    tin = tout = 0
    for t in trees:
        for s in sizes:
            i, o = worst_case(t, s)
            tin += i * len(per_size[s]) * a.repeats
            tout += o * len(per_size[s]) * a.repeats
    worst = cost(tin, tout, a.price_in, a.price_out)
    say("DRY RUN: no network call, no model call. Add --run to execute.")
    say(f"model: {os.environ.get('OPENPOKE_MODEL', DEFAULT_MODEL)} | trees: {','.join(trees)} | sizes: {sizes} "
        f"| split: {a.split} | repeats: {a.repeats} | limit: {a.limit or 'none'}")
    say("queries per size: " + ", ".join(f"{s}: {len(per_size[s])}" for s in sizes))
    say(f"query runs: {runs}  (queries x sizes x {len(trees)} trees x {a.repeats} repeats)")
    say(f"worst-case tokens: {tin:,} in, {tout:,} out ({MAX_ITER} model calls per query, growing prompt)")
    say(f"worst-case cost: ${worst:.2f} USD at ${a.price_in}/M in and ${a.price_out}/M out; "
        f"max-usd {a.max_usd} -> {'within' if worst <= a.max_usd else 'EXCEEDS (run will stop early at max-usd)'}")
    say(f"typical cost (about 2 calls per query, short prompts) is roughly {worst / 10:.2f}-{worst / 4:.2f} USD")


def run_all(a, sizes, per_size, agents):
    start_key = key_usage()
    result = {"meta": {"model": os.environ.get("OPENPOKE_MODEL", DEFAULT_MODEL), "trees": a.trees, "sizes": sizes,
                       "split": a.split, "repeats": a.repeats, "limit": a.limit, "max_usd": a.max_usd,
                       "price_in": a.price_in, "price_out": a.price_out, "orig_commit": ORIG_COMMIT,
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              "rows": [], "summary": {}, "stopped_early": False, "spent_estimate_usd": 0.0,
              "account_spend_delta_usd": None, "gmail_calls": 0, "usage_missing": 0}
    tmp = Path(tempfile.mkdtemp(prefix="openpoke_live_trees_"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        for t in a.trees:
            for s in sizes:
                rs = [r for r in result["rows"] if r["tree"] == t and r["size"] == s]
                if rs:
                    result["summary"].setdefault(t, {})[str(s)] = {**summarize(rs), "consistency": consistency(rs)}
        _write_json(out, result)
    try:
        roots = {}
        if "orig" in a.trees:
            roots["orig"] = tmp / "orig"
            extract_original(roots["orig"])
        if "new" in a.trees:
            roots["new"] = tmp / "new"
            copy_tree(a.new, roots["new"])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.setdefault("OPENPOKE_MODEL", DEFAULT_MODEL)
        done = False
        for s in sizes:
            qfile = tmp / f"queries_{s}.json"
            qfile.write_text(json.dumps(per_size[s]), encoding="utf-8")
            for rep in range(1, a.repeats + 1):
                for t in a.trees:
                    left = a.max_usd - result["spent_estimate_usd"]
                    if left <= 0:
                        done = True
                        break
                    rows_file = tmp / f"rows_{t}_{s}_{rep}.json"
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--src", str(roots[t]),
                           "--tree", t, "--size", str(s), "--repeat", str(rep), "--queries-file", str(qfile),
                           "--rows-out", str(rows_file), "--budget-left", str(left),
                           "--price-in", str(a.price_in), "--price-out", str(a.price_out)]
                    if a.real_file:
                        cmd += ["--real-file", a.real_file]
                    code = subprocess.run(cmd, env=env, cwd=str(tmp)).returncode
                    if rows_file.exists():
                        d = json.loads(rows_file.read_text(encoding="utf-8"))
                        result["rows"] += d["rows"]
                        result["spent_estimate_usd"] = round(result["spent_estimate_usd"] + d["spent_usd"], 6)
                        result["gmail_calls"] += d["gmail_calls"]
                        result["usage_missing"] += d["usage_missing"]
                        done = done or d["stopped"]
                    save()
                    if code != 0:
                        say(f"worker for {t} (size {s}, repeat {rep}) exited with code {code}; partial results saved")
                        result["worker_failed"] = True
                        done = True
                    if done:
                        break
                if done:
                    break
            if done:
                break
        result["stopped_early"] = bool(done and not result.get("worker_failed")
                                       and result["spent_estimate_usd"] >= a.max_usd)
        end_key = key_usage()
        if start_key is not None and end_key is not None:
            result["account_spend_delta_usd"] = round(end_key - start_key, 4)
        save()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def print_summary(result):
    for t, by_size in result["summary"].items():
        for s, sm in by_size.items():
            say(f"\n{t.upper()} tree, roster {s} agents (consistency across repeats: {sm['consistency']})")
            say(f"{'category':15}{'n':>4}{'correct':>9}{'dup':>7}{'wrong':>7}{'tok_in':>8}{'calls':>7}{'p95':>5}{'find':>6}{'s_mean':>8}{'s_p50':>7}{'s_p95':>7}{'model_s':>8}")
            for name, m in list(sm["by_category"].items()) + [("OVERALL", sm["overall"])]:
                f = lambda v: "  n/a" if v is None else f"{v:5.2f}"
                say(f"{name:15}{m['n']:>4}{f(m['correct_rate']):>9}{f(m['duplicate_rate']):>7}{f(m['wrong_reuse_rate']):>7}"
                    f"{m['prompt_tokens_mean']:>8}{m['calls_mean']:>7}{m['loop_p95']:>5}{f(m['find_called_rate']):>6}"
                    f"{m['latency_mean']:>8.2f}{m['latency_p50']:>7.2f}{m['latency_p95']:>7.2f}{m['model_seconds_mean']:>8.2f}")
    say(f"\nEstimated spend: ${result['spent_estimate_usd']:.4f} (usage tokens x given prices); "
        f"gmail calls: {result['gmail_calls']}; responses without usage: {result['usage_missing']}")
    d = result["account_spend_delta_usd"]
    say("Real account spend delta (OpenRouter /key): " + (f"${d:.4f}" if d is not None else "unavailable"))
    if result["stopped_early"]:
        say(f"STOPPED EARLY: estimated spend reached max-usd {result['meta']['max_usd']}. Partial results saved.")


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", action="store_true", help="call the real model (default: dry run)")
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--price-in", type=float, default=0.10, help="USD per million prompt tokens")
    p.add_argument("--price-out", type=float, default=0.50, help="USD per million completion tokens")
    p.add_argument("--split", choices=("test", "dev", "all"), default="test")
    p.add_argument("--sizes", default="150", help="comma list, e.g. 150,300")
    p.add_argument("--trees", default="orig,new")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="first N queries per category (smoke run)")
    p.add_argument("--new", default=".", help="path of the new-logic tree (contains server/)")
    p.add_argument("--real-file", help="private JSON {agents, queries} from real_gmail_check.py build; replaces the "
                   "labelled queries and adds those agents to every roster. Use --out outside eval/results.")
    p.add_argument("--out", default=str(ROOT / "eval" / "results" / "live_eval.json"))
    w = p.add_argument_group("internal worker options")
    w.add_argument("--worker", action="store_true")
    for name in ("--src", "--tree", "--queries-file", "--rows-out"):
        w.add_argument(name)
    w.add_argument("--size", type=int)
    w.add_argument("--repeat", type=int)
    w.add_argument("--budget-left", type=float, default=0.0)
    a = p.parse_args(argv)
    a.trees = [t for t in a.trees.split(",") if t]
    return a


def main(argv=None):
    a = parse(argv)
    if a.worker:
        return worker_main(a)
    if not set(a.trees) <= {"orig", "new"} or not a.trees:
        say("--trees must be orig, new or orig,new")
        return 2
    sizes = [int(s) for s in a.sizes.split(",") if s]
    agents, queries = load_data()
    if a.real_file:
        real = json.loads(Path(a.real_file).read_text(encoding="utf-8"))
        per_size = {s: [dict(q, qid=i) for i, q in enumerate(real["queries"])] for s in sizes}
    else:
        per_size = {s: size_queries(queries, agents, s, a.split, a.limit) for s in sizes}
    if not a.run:
        print_plan(a, sizes, per_size)
        return 0
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get(FAKE_ENV):
        say("--run needs OPENROUTER_API_KEY in the environment (no .env file is read).")
        return 2
    if "new" in a.trees and not (Path(a.new) / "server").is_dir():
        say(f"--new {a.new}: no server/ folder there")
        return 2
    result = run_all(a, sizes, per_size, agents)
    print_summary(result)
    say(f"wrote {a.out}")
    return 0 if not result.get("worker_failed") else 1


if __name__ == "__main__":
    sys.exit(main())
