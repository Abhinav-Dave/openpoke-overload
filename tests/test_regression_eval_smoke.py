"""Smoke test for eval/regression_eval.py: reduced scenarios, plus a break-on-purpose case."""

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("regression_eval", ROOT / "eval" / "regression_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ORIG = ev.default_orig()
pytestmark = pytest.mark.skipif(not (ORIG / "server").is_dir(), reason="original source tree not available")


def run(new, checks, gaps="none"):
    return ev.evaluate(ORIG, new, checks=checks, gaps=gaps, out=None, quiet=True)


def test_prompt_checks_pass_on_current_tree():
    res = run(ROOT, [1, 2])
    assert [r["status"] for r in res["regression"]] == ["PASS", "PASS"], res["regression"]


def test_break_on_purpose_roster_format_fails_check_1(tmp_path):
    shutil.copytree(ROOT / "server", tmp_path / "server", ignore=shutil.ignore_patterns("__pycache__", "data"))
    f = tmp_path / "server" / "agents" / "interaction_agent" / "agent.py"
    text = f.read_text(encoding="utf-8")
    assert '" />' in text
    f.write_text(text.replace('" />', '"/>'), encoding="utf-8")   # one character of roster format
    res = run(tmp_path, [1])
    assert res["regression"][0]["status"] == "FAIL", res["regression"]


def test_gap_floor_table_shows_the_difference():
    g = run(ROOT, [], gaps="quick")["gaps"]["floor"]
    rule = lambda side, r: next(x for x in g[side] if x["rule"] == r)
    assert rule("original", "FREQ=SECONDLY")["accepted"] is True
    assert rule("new", "FREQ=SECONDLY")["accepted"] is False
    assert rule("new", "FREQ=DAILY;BYHOUR=9;BYMINUTE=0")["accepted"] is True
