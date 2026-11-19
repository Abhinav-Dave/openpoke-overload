"""Offline find_agents check: BM25 alone against BM25 fused with embedding search (hybrid).

Usage: python eval/find_eval.py
Needs OPENROUTER_API_KEY (environment or the repo .env). It calls the embedding model only, never a chat model.
It builds made-up agents through the real log store in a temp folder and calls the real roster_view.find_agents
with each labelled question as the keywords. Writes eval/results/find_eval.json.

Questions: eval/datasets/routing_queries.json without the follow-ups (they need chat history), plus the 30
reworded questions in eval/datasets/routing_paraphrase_queries.json. Rows are tagged by source file.
Groups: paraphrase_old, paraphrase_new, direct+disambiguation, no_match+offtopic.
hit@1 / hit@5: the expected agent, or a numbered clone of it, is first / in the first five results.
empty: find_agents returned no agent (the right answer for no_match and offtopic questions).

Conditions:
  (a) the 150 made-up agents, with FIND_EMBED_ABOVE=0 so that the hybrid runs at this size;
  (b) 1,500 agents (the 150 plus numbered clones) at the default threshold of 1,000.
Modes: BM25 (embeddings.ENABLED = False) and hybrid. The embedding cache is warmed before any hybrid row.

Pre-registered rules, written before the first real run:
  1. Tune MIN_SIM on the dev split at 1,500 agents over MIN_SIMS. A value qualifies when direct+disambiguation
     hit@1 is at least BM25's and the no-match empty rate is at least BM25's minus 10 points. Take the qualifying
     value with the highest paraphrase hit@5 (old and new together); ties go to the higher MIN_SIM. If no value
     qualifies, the OPENPOKE_EMBED default becomes 0.
  2. If direct+disambiguation hit@1 falls below BM25's at any swept value, repeat the dev sweep once with a pin:
     the BM25 top hit stays first when its BM25 score is at least roster_view.MIN_SCORE. Use the pin only if it
     lets a value with a higher paraphrase hit@5 qualify under rule 1. Dev only.
  3. Run the test split once, with the chosen MIN_SIM.
  4. Keep OPENPOKE_EMBED=1 as the default only if, at 1,500 agents on the test split, paraphrase hit@5 gains at
     least 3 of 31, direct+disambiguation hit@1 loses at most 1 of 32, and the no-match empty count drops by at
     most 2 of 21. Otherwise the default is 0: built, measured, off by default.
The script refuses to write a result if any hybrid row fell back to BM25 (the embedder failed or did not run).
"""
import hashlib, json, logging, math, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_eval import SEED, _import_server, build, load_data, with_clones  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NEW_QUERIES = ROOT / "eval" / "datasets" / "routing_paraphrase_queries.json"
OUT = ROOT / "eval" / "results" / "find_eval.json"
MIN_SIMS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
BIG = 1500
GROUPS = ("paraphrase_old", "paraphrase_new", "paraphrase_all", "direct+disambiguation", "no_match+offtopic")


def questions():
    _, old = load_data()
    new = json.loads(NEW_QUERIES.read_text(encoding="utf-8"))["queries"]
    return [dict(q, source="old") for q in old if q["category"] != "followup"] + [dict(q, source="new") for q in new]


def group(q):
    if q["category"] == "paraphrase":
        return f"paraphrase_{q['source']}"
    return "direct+disambiguation" if q["category"] in ("direct", "disambiguation") else "no_match+offtopic"


def is_hit(name, expected):
    return name == expected or re.fullmatch(re.escape(expected) + r"-\d+", name) is not None  # with_clones suffix


def run(queries, names, *, hybrid):
    """One row per question through the real find_agents. fell_back: hybrid was asked for but did not answer."""
    from server.services.execution import embeddings, roster_view
    real_similar, was_enabled = embeddings.similar, embeddings.ENABLED
    answered = []

    def watched(*args):
        out = real_similar(*args)
        answered.append(out is not None)
        return out

    embeddings.similar, embeddings.ENABLED = watched, hybrid
    rows = []
    try:
        for q in queries:
            answered.clear()
            t0 = time.perf_counter()
            found = [a.name for a in roster_view.find_agents(q["text"], names)]
            ms = (time.perf_counter() - t0) * 1000
            exp = q["expected"]
            rows.append({"group": group(q), "split": q["split"],
                         "hit1": (bool(found) and is_hit(found[0], exp)) if exp else None,
                         "hit5": any(is_hit(n, exp) for n in found[:5]) if exp else None,
                         "empty": not found, "ms": round(ms, 1),
                         "fell_back": hybrid and not (answered and answered[-1])})
    finally:
        embeddings.similar, embeddings.ENABLED = real_similar, was_enabled
    return rows


def _stats(rows):
    pos = [r for r in rows if r["hit1"] is not None]
    rate = lambda k, rs: round(sum(r[k] for r in rs) / len(rs), 3) if rs else None
    return {"n": len(rows), "hit1": sum(r["hit1"] for r in pos), "hit5": sum(r["hit5"] for r in pos),
            "empty": sum(r["empty"] for r in rows), "hit1_rate": rate("hit1", pos), "hit5_rate": rate("hit5", pos),
            "empty_rate": rate("empty", rows),
            "ms_mean": round(sum(r["ms"] for r in rows) / len(rows), 1) if rows else None}


def summarize(rows):
    out = {}
    for g in GROUPS:
        in_group = [r for r in rows if r["group"] == g or (g == "paraphrase_all" and r["group"].startswith("paraphrase"))]
        out[g] = {split: _stats([r for r in in_group if r["split"] == split]) for split in ("dev", "test")}
    out["fallbacks"] = sum(r["fell_back"] for r in rows)
    return out


def key_numbers(summary, split, *, counts=False):
    """The three numbers the rules use: paraphrase hit@5, direct+disambiguation hit@1, no-match empty."""
    s = lambda g: summary[g][split]
    if counts:
        return {"para_hit5": s("paraphrase_all")["hit5"], "direct_hit1": s("direct+disambiguation")["hit1"],
                "empty": s("no_match+offtopic")["empty"]}
    return {"para_hit5": s("paraphrase_all")["hit5_rate"], "direct_hit1": s("direct+disambiguation")["hit1_rate"],
            "empty": s("no_match+offtopic")["empty_rate"]}


def pick_min_sim(sweep, bm25):
    """Rule 1. sweep: [{min_sim, para_hit5, direct_hit1, empty}] as dev rates; bm25: the same keys. None: none qualifies."""
    ok = [s for s in sweep if s["direct_hit1"] >= bm25["direct_hit1"] and s["empty"] >= bm25["empty"] - 0.10 - 1e-9]
    return max(ok, key=lambda s: (s["para_hit5"], s["min_sim"]))["min_sim"] if ok else None


def keep_on(bm25, hybrid):
    """Rule 4, on test-split counts at 1,500 agents."""
    return (hybrid["para_hit5"] - bm25["para_hit5"] >= 3 and bm25["direct_hit1"] - hybrid["direct_hit1"] <= 1
            and bm25["empty"] - hybrid["empty"] <= 2)


def warm(names):
    """Fill the embedding cache through the real find_agents path. Returns (calls, seconds, inputs per call)."""
    from server.services.execution import embeddings, roster_view
    real, was_enabled, sizes = embeddings.request_embeddings, embeddings.ENABLED, []

    def counted(**kw):
        sizes.append(len(kw["inputs"]))
        return real(**kw)

    embeddings.request_embeddings, embeddings.ENABLED = counted, True
    t0 = time.perf_counter()
    try:
        for _ in range(math.ceil(len(names) / embeddings.MAX_NEW) + 1):
            roster_view.find_agents("warm up", names)
            if sizes and sizes[-1] == 1:  # only the query was embedded: every agent is cached
                break
    finally:
        embeddings.request_embeddings, embeddings.ENABLED = real, was_enabled
    return len(sizes), round(time.perf_counter() - t0, 2), sizes


def memo_queries():
    """Reuse query vectors during the sweep (only MIN_SIM changes there). Returns undo()."""
    from server.services.execution import embeddings
    real, memo = embeddings.request_embeddings, {}

    def memoized(**kw):
        if len(kw["inputs"]) != 1:
            return real(**kw)
        key = (kw["model"], kw["inputs"][0])
        if key not in memo:
            memo[key] = real(**kw)
        return memo[key]

    embeddings.request_embeddings = memoized
    return lambda: setattr(embeddings, "request_embeddings", real)


def save(result, hybrid_rows, path=OUT):
    """Write the result unless a hybrid row fell back. Returns True if written."""
    bad = sum(r["fell_back"] for r in hybrid_rows)
    if bad:
        print(f"INVALID RUN: {bad} hybrid rows fell back to BM25 (embedder failed or did not run). Nothing written.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return True


def table(title, bm25, hybrid, split):
    print(f"\n{title} ({split} split)")
    print(f"{'group':24}{'n':>4} {'hit@1 bm25':>11}{'hybrid':>8} {'hit@5 bm25':>11}{'hybrid':>8} {'empty bm25':>11}{'hybrid':>8}")
    for g in GROUPS:
        b, h = bm25[g][split], hybrid[g][split]
        f = lambda v: "  n/a" if v is None else f"{v:5.2f}"
        print(f"{g:24}{b['n']:>4} {f(b['hit1_rate']):>11}{f(h['hit1_rate']):>8} {f(b['hit5_rate']):>11}"
              f"{f(h['hit5_rate']):>8} {f(b['empty_rate']):>11}{f(h['empty_rate']):>8}")


def main():
    t_start = time.time()
    _, roster_view, roster, logs = _import_server()
    from server.services import usage
    from server.services.execution import embeddings
    logging.disable(logging.CRITICAL)  # the real roster and log store log every clear
    agents, _ = load_data()
    qs = questions()
    dev, test = [q for q in qs if q["split"] == "dev"], [q for q in qs if q["split"] == "test"]
    usage.reset()
    placeholder = embeddings.MIN_SIM
    result = {"synthetic_data": True, "embed_model": embeddings.MODEL,
              "paraphrase_file_sha256": hashlib.sha256(NEW_QUERIES.read_bytes()).hexdigest(),
              "n_questions": {"dev": len(dev), "test": len(test)}, "rules": __doc__.split("Pre-registered rules")[1]}
    hybrid_rows = []

    # (b) 1,500 agents at the default threshold
    build(roster, logs, with_clones(agents, BIG, random.Random(SEED)))
    names = roster.get_agents()
    calls, seconds, sizes = warm(names)
    result["warm_1500"] = {"calls": calls, "seconds": seconds, "inputs_per_call": sizes,
                           "cache_file_bytes": embeddings._cache_path().stat().st_size}
    print(f"warmed {len(names)} agents in {calls} calls, {seconds} s; cache file {result['warm_1500']['cache_file_bytes']:,} bytes")
    bm25_big = summarize(run(qs, names, hybrid=False))

    sweep, undo = [], memo_queries()
    try:
        for v in MIN_SIMS:
            embeddings.MIN_SIM = v
            rows = run(dev, names, hybrid=True)
            hybrid_rows += rows
            sweep.append({"min_sim": v, **key_numbers(summarize(rows), "dev"), "fallbacks": summarize(rows)["fallbacks"]})
    finally:
        undo()
    bm25_dev = key_numbers(bm25_big, "dev")
    chosen = pick_min_sim(sweep, bm25_dev)
    result["sweep_dev_1500"] = {"bm25": bm25_dev, "hybrid": sweep, "chosen_min_sim": chosen}
    print("\nDEV SWEEP at 1,500 agents (rates): bm25", bm25_dev)
    for s in sweep:
        print(f"  MIN_SIM {s['min_sim']:.2f}: para_hit5 {s['para_hit5']}  direct_hit1 {s['direct_hit1']}  empty {s['empty']}")
    print(f"chosen MIN_SIM: {chosen}" + ("" if chosen is not None else f" (none qualifies; test runs at {placeholder})"))

    embeddings.MIN_SIM = chosen if chosen is not None else placeholder
    rows = run(test, names, hybrid=True)  # rule 3: the test split, once
    hybrid_rows += rows
    hybrid_big = summarize(rows)
    result["agents_1500"] = {"bm25": bm25_big, "hybrid_test": hybrid_big}
    table("1,500 agents, default threshold", bm25_big, hybrid_big, "test")

    # (a) the 150 made-up agents, hybrid forced on
    build(roster, logs, with_clones(agents, len(agents), random.Random(SEED)))
    small = roster.get_agents()
    threshold, roster_view.FIND_EMBED_ABOVE = roster_view.FIND_EMBED_ABOVE, 0
    try:
        warm(small)
        bm25_small = summarize(run(qs, small, hybrid=False))
        rows = run(qs, small, hybrid=True)
    finally:
        roster_view.FIND_EMBED_ABOVE = threshold
    hybrid_rows += rows
    hybrid_small = summarize(rows)
    result["agents_150"] = {"bm25": bm25_small, "hybrid": hybrid_small}
    table("150 agents, FIND_EMBED_ABOVE=0", bm25_small, hybrid_small, "test")
    table("150 agents, FIND_EMBED_ABOVE=0", bm25_small, hybrid_small, "dev")

    b, h = key_numbers(bm25_big, "test", counts=True), key_numbers(hybrid_big, "test", counts=True)
    on = chosen is not None and keep_on(b, h)
    result["decision"] = {"bm25_test_counts": b, "hybrid_test_counts": h, "min_sim": embeddings.MIN_SIM,
                          "openpoke_embed_default": 1 if on else 0}
    result["fallbacks"] = sum(r["fell_back"] for r in hybrid_rows)
    u = usage.get_usage()
    result["embedding_usage"] = {"failed_calls": u["failed_calls"], **u["models"].get(embeddings.MODEL, {})}
    result["seconds"] = round(time.time() - t_start)
    print(f"\ntest counts at 1,500: bm25 {b}  hybrid {h}  ->  OPENPOKE_EMBED default {1 if on else 0}")
    print(f"embedding usage: {result['embedding_usage']}; hybrid fallbacks: {result['fallbacks']}")
    if not save(result, hybrid_rows):
        return 1
    print(f"wrote {OUT} in {result['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
