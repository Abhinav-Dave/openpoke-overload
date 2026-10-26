"""Print eval/results/*.json side by side: python eval/compare.py baseline u0 c3 ... [--md]"""
import json
import sys
from pathlib import Path

ROWS = [
    ("interaction prompt chars, 10 agents", ("interaction_prompt", "10", "chars")),
    ("interaction prompt chars, 100 agents", ("interaction_prompt", "100", "chars")),
    ("interaction prompt chars, 300 agents", ("interaction_prompt", "300", "chars")),
    ("execution prompt chars, 5 log entries", ("execution_prompt", "5", "chars")),
    ("execution prompt chars, 200 log entries", ("execution_prompt", "200", "chars")),
    ("model calls, 10-email flood", ("email_flood", "10", "total_model_calls")),
    ("model calls, 50-email flood", ("email_flood", "50", "total_model_calls")),
    ("peak parallel agents, 20 triggers", ("burst_triggers", "peak_parallel_agents")),
    ("peak parallel agents, 8 in one turn", ("burst_agents", "peak_parallel_agents")),
    ("overlapping runs, 5 chat messages", ("burst_chat", "peak_overlapping_interaction_runs")),
]


def pick(data: dict, path: tuple):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return "n/a"
        data = data[key]
    return data


def main(argv: list[str]) -> None:
    names = [a for a in argv if not a.startswith("--")]
    results = [json.loads((Path(__file__).parent / "results" / f"{n}.json").read_text()) for n in names]
    header = ["metric", *names]
    rows = [[label, *[str(pick(r, path)) for r in results]] for label, path in ROWS]
    if "--md" in argv:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main(sys.argv[1:])
