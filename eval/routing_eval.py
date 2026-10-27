"""Offline routing evaluator: does the bounded roster view still show the agent the user means?

Usage: python eval/routing_eval.py
Builds a temp data dir, creates synthetic agents through the real log store, then runs the real
select_agents and the real <active_agents> renderer for every labelled query. No network, no model.
Writes eval/results/routing_eval.json. Always exits 0 (it reports, it does not gate).
"""
import json, logging, os, random, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "datasets"
OUT = ROOT / "eval" / "results" / "routing_eval.json"
SEED = 7
BASE_AGE_S = 10 * 86400   # baseline log age; each agent a little older than the previous one
KS, MIN_SCORES = (4, 8, 12, 20), (1, 2, 3, 4)
SIZES = (10, 50, 100, 300, 600)
CATEGORIES = ("direct", "paraphrase", "disambiguation", "followup", "no_match", "offtopic")


def _import_server():
    """Import the server pointed at a temp data dir (unless a test harness already did)."""
    if "server.config" not in sys.modules:
        os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke_routing_eval_")
        sys.path.insert(0, str(ROOT))
    import server.config as cfg
    if Path(cfg.DATA_DIR).resolve() == (ROOT / "server" / "data").resolve():
        raise SystemExit("refusing to run against the real server/data folder")
    from server.agents.interaction_agent import agent as agent_mod
    from server.services.execution import get_agent_roster, get_execution_agent_logs, roster_view
    return agent_mod, roster_view, get_agent_roster(), get_execution_agent_logs()


def load_data():
    agents = json.loads((DATA / "routing_agents.json").read_text(encoding="utf-8"))["agents"]
    queries = json.loads((DATA / "routing_queries.json").read_text(encoding="utf-8"))["queries"]
    return agents, queries


def with_clones(agents, n, rng):
    """First n agents of a seeded shuffle; beyond len(agents), clones with a numeric suffix (marked cloned)."""
    pool = list(agents)
    rng.shuffle(pool)
    out = [dict(a, cloned=False) for a in pool[:n]]
    i = 2
    while len(out) < n:  # ponytail: clones share text with their original, so they tie on score; ties are shuffled below
        for a in pool:
            if len(out) >= n:
                break
            out.append({"name": f"{a['name']}-{i}", "requests": a["requests"], "cloned": True})
        i += 1
    rng.shuffle(out)
    return out


def build(roster, logs, agents):
    """Create agents through the real log store; older agents get older log mtimes."""
    roster.clear(); logs.clear_all()
    now = time.time()
    for i, a in enumerate(agents):
        roster.add_agent(a["name"])
        for r in a["requests"]:
            logs.record_request(a["name"], r)
        t = now - BASE_AGE_S - i * 60
        os.utime(logs._log_path(a["name"]), (t, t))


def run_queries(agent_mod, roster_view, logs, queries, *, k=None, min_score=None, names_only=False):
    """One row per query: real select_agents, real <active_agents> rendering."""
    captured = {}
    real = roster_view.select_agents

    def wrapper(query, names, **kw):
        if names_only:  # original behaviour: every agent, name only
            res = ([roster_view.AgentView(n) for n in names], 0, False)
        else:
            res = real(query, names, k=k, min_score=min_score, **kw)
        captured["res"] = res
        return res

    orig = agent_mod.select_agents
    agent_mod.select_agents = wrapper
    rows = []
    try:
        for q in queries:
            path = restore = None
            if q.get("last_used"):  # the user just used this agent: make it the most recent
                path = logs._log_path(q["last_used"])
                restore = path.stat().st_mtime
                now = time.time()
                os.utime(path, (now, now))
            t0 = time.perf_counter()
            try:
                block = agent_mod._render_active_agents(q["text"], "user")
            finally:
                ms = (time.perf_counter() - t0) * 1000
                if path is not None:
                    os.utime(path, (restore, restore))
            shown, _, no_match = captured["res"]
            rows.append({"category": q["category"], "expected": q["expected"],
                         "visible": q["expected"] in {v.name for v in shown} if q["expected"] else None,
                         "no_match": no_match, "shown": len(shown), "chars": len(block), "ms": ms})
    finally:
        agent_mod.select_agents = orig
    return rows


def _rate(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def metrics(rows):
    pos = [r for r in rows if r["expected"]]
    neg = [r for r in rows if not r["expected"]]
    return {"n": len(rows),
            "visible_recall": _rate([r["visible"] for r in pos]),
            "false_no_match_rate": _rate([r["no_match"] for r in pos]),
            "no_match_correct_rate": _rate([r["no_match"] for r in neg]),
            "shown_agents_mean": round(sum(r["shown"] for r in rows) / len(rows), 2) if rows else None,
            "block_chars_mean": round(sum(r["chars"] for r in rows) / len(rows)) if rows else None,
            "select_ms_mean": round(sum(r["ms"] for r in rows) / len(rows), 1) if rows else None}


def by_category(rows):
    return {c: metrics([r for r in rows if r["category"] == c]) for c in CATEGORIES}


def evaluate(env, agents, queries, **kw):
    agent_mod, roster_view, roster, logs = env
    return run_queries(agent_mod, roster_view, logs, queries, **kw)


def answerable(queries, agents):
    names = {a["name"] for a in agents}
    return [q for q in queries if q["expected"] is None or q["expected"] in names]


def fmt(v):
    return "  n/a" if v is None else f"{v:5.2f}" if isinstance(v, float) and v <= 1 else f"{v:5}"


def scorecard(title, cats, overall):
    print(f"\n{title}")
    print(f"{'category':15}{'n':>4} {'vis_recall':>10} {'false_nomatch':>13} {'nomatch_ok':>10} {'shown':>6} {'chars':>7}")
    for name, m in list(cats.items()) + [("OVERALL", overall)]:
        print(f"{name:15}{m['n']:>4} {fmt(m['visible_recall']):>10} {fmt(m['false_no_match_rate']):>13} "
              f"{fmt(m['no_match_correct_rate']):>10} {m['shown_agents_mean']:>6} {m['block_chars_mean']:>7}")


def warnings(cats):
    out = []
    for c, m in cats.items():
        if m["visible_recall"] is not None and m["visible_recall"] < 0.8:
            out.append(f"WARNING: {c} visible_recall {m['visible_recall']} < 0.8")
        if m["false_no_match_rate"] is not None and m["false_no_match_rate"] > 0.1:
            out.append(f"WARNING: {c} false_no_match_rate {m['false_no_match_rate']} > 0.1")
    return out


def main():
    t0 = time.time()
    env = _import_server()
    logging.disable(logging.CRITICAL)  # the real roster/log store log every clear
    agent_mod, roster_view, roster, logs = env
    agents, queries = load_data()
    rng = random.Random(SEED)
    dev = [q for q in queries if q["split"] == "dev"]
    test = [q for q in queries if q["split"] == "test"]
    result = {"defaults": {"K": roster_view.K, "HOT_N": roster_view.HOT_N, "MIN_SCORE": roster_view.MIN_SCORE},
              "synthetic_data": True, "n_agents": len(agents), "n_dev": len(dev), "n_test": len(test)}

    build(roster, logs, with_clones(agents, len(agents), rng))
    rows =evaluate(env, agents, test)
    cats, overall = by_category(rows), metrics(rows)
    result["headline_test"] = {"overall": overall, "by_category": cats}
    base = evaluate(env, agents, test, names_only=True)
    result["baseline_names_only_test"] = {"overall": metrics(base), "by_category": by_category(base)}

    print(f"Routing eval (SYNTHETIC data): {len(agents)} agents, test split {len(test)} queries, "
          f"defaults K={roster_view.K} HOT_N={roster_view.HOT_N} MIN_SCORE={roster_view.MIN_SCORE}")
    scorecard("HEADLINE: current product, test split", cats, overall)
    b = metrics(base)
    print(f"\nBASELINE (original behaviour, every agent listed by name only): visible_recall {b['visible_recall']} "
          f"and false_no_match 0.0 BY CONSTRUCTION; mean block {b['block_chars_mean']} chars.\n"
          "  Caveat: names-only gives the model no description to choose with. This offline test cannot measure that; the live check will.")

    sweep = []
    print("\nSWEEP on DEV split (K x MIN_SCORE; HOT_N fixed)")
    print(f"{'K':>4}{'MIN':>5}{'vis_recall':>11}{'false_nomatch':>14}{'nomatch_ok':>11}{'block_chars':>12}")
    for k in KS:
        for ms in MIN_SCORES:
            m = metrics(evaluate(env, agents, dev, k=k, min_score=float(ms)))
            sweep.append({"K": k, "MIN_SCORE": ms, **m})
            print(f"{k:>4}{ms:>5}{fmt(m['visible_recall']):>11}{fmt(m['false_no_match_rate']):>14}"
                  f"{fmt(m['no_match_correct_rate']):>11}{m['block_chars_mean']:>12}")
    result["sweep_dev"] = sweep

    curve = []
    print("\nROSTER-SIZE CURVE (test split, current defaults; queries whose expected agent exists in the roster)")
    print(f"{'N':>5}{'cloned':>8}{'n_q':>5}{'vis_recall':>11}{'false_nomatch':>14}{'block_chars':>12}{'baseline_chars':>15}{'ms/query':>10}")
    for n in SIZES:
        ro = with_clones(agents, n, random.Random(SEED))
        build(roster, logs, ro)
        qs = answerable(test, ro)
        m = metrics(evaluate(env, ro, qs))
        mb = metrics(evaluate(env, ro, qs, names_only=True))
        cl = sum(a["cloned"] for a in ro)
        curve.append({"N": n, "cloned": cl, **m, "baseline_block_chars_mean": mb["block_chars_mean"]})
        print(f"{n:>5}{cl:>8}{m['n']:>5}{fmt(m['visible_recall']):>11}{fmt(m['false_no_match_rate']):>14}"
              f"{m['block_chars_mean']:>12}{mb['block_chars_mean']:>15}{m['select_ms_mean']:>10}")
    result["size_curve_test"] = curve

    print("\nNote: clones copy an original's text, so with N>150 they tie with it on score; roster order is shuffled.")
    warns = warnings(cats)
    result["warnings"] = warns
    print()
    for w in warns:
        print(w)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
