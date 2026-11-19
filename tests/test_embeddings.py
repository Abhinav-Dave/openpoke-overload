"""embeddings.similar: cosine ranking over a disk cache, one call per search, None on any failure."""
import base64
import hashlib
import json
import math
from array import array

import httpx
import pytest

from server.openrouter_client import OpenRouterError
from server.services.execution import embeddings

NAMES = ["a-therapy", "b-bread", "c-invoice", "d-none"]
TEXTS = ["a-therapy: weekly therapist notes", "b-bread: sourdough starter schedule",
         "c-invoice: payment reminders", "d-none: misc stuff"]


def sim(query="counsellor visit", names=NAMES, texts=TEXTS):
    return embeddings.similar(query, names, texts)


def saved():
    return json.loads(embeddings._cache_path().read_text(encoding="utf-8"))


def decode(b64):
    vec = array("f")
    vec.frombytes(base64.b64decode(b64))
    return vec


def test_disabled_returns_none_without_a_call(fake_embedder, monkeypatch):
    monkeypatch.setattr(embeddings, "ENABLED", False)
    assert sim() is None and fake_embedder.count == 0


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_returns_none_without_a_call(fake_embedder, query):
    assert sim(query) is None and fake_embedder.count == 0


def test_ranks_by_cosine_and_drops_below_min_sim(fake_embedder, monkeypatch):
    assert sim("counsellor visit") == [0]  # only the therapy agent shares a concept
    assert sim("bread culture and invoice") == [1, 2]  # equal cosines (0.71) keep roster order; d-none is a zero vector
    assert sim("nothing relevant at all") == []  # a zero query vector matches nothing and does not crash
    monkeypatch.setattr(embeddings, "MIN_SIM", 0.8)
    assert sim("bread culture and invoice") == []  # 0.71 is below 0.8
    assert sim("loaf") == [1]


def test_cache_holds_a_sha_and_a_unit_float32_vector_per_agent(fake_embedder):
    sim()
    data = saved()
    assert set(data) == set(NAMES)
    sha, b64 = data["b-bread"]
    assert sha == hashlib.sha1(TEXTS[1].encode("utf-8")).hexdigest()
    vec = decode(b64)
    assert len(vec) == 64 and math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)
    assert not any(decode(data["d-none"][1]))  # a zero vector stays zero


def test_second_call_embeds_only_the_query(fake_embedder):
    sim()
    sim("loaf")
    first, second = fake_embedder.calls
    assert first["inputs"] == ["counsellor visit"] + TEXTS
    assert second["inputs"] == ["loaf"]
    assert first["model"] == embeddings.MODEL and first["timeout"] == embeddings.TIMEOUT_S


def test_long_query_is_cut(fake_embedder):
    sim("bread " * 1000)
    assert len(fake_embedder.calls[0]["inputs"][0]) == embeddings.QUERY_CHARS


def test_cache_survives_a_restart(fake_embedder):
    sim()
    embeddings._cache = None  # a new process
    assert sim("loaf") == [1]
    assert fake_embedder.calls[1]["inputs"] == ["loaf"]


def test_changed_text_re_embeds_only_that_agent(fake_embedder):
    sim()
    texts = list(TEXTS)
    texts[3] = "d-none: now about the bill I owe"
    assert sim("payment", texts=texts) == [2, 3]
    assert fake_embedder.calls[1]["inputs"] == ["payment", texts[3]]


def test_model_change_starts_a_fresh_cache(fake_embedder, monkeypatch):
    sim()
    old = embeddings._cache_path()
    monkeypatch.setattr(embeddings, "MODEL", "other/model")
    sim()
    assert fake_embedder.calls[1]["inputs"] == ["counsellor visit"] + TEXTS
    assert fake_embedder.calls[1]["model"] == "other/model"
    assert embeddings._cache_path() != old and old.exists() and embeddings._cache_path().exists()


def test_cold_cache_embeds_at_most_max_new_per_call(fake_embedder, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_NEW", 2)
    assert sim() == [0]  # agents not embedded yet are simply not ranked
    assert sim("payment") == [2]
    assert sim("payment") == [2]
    assert [c["inputs"] for c in fake_embedder.calls] == [
        ["counsellor visit"] + TEXTS[:2], ["payment"] + TEXTS[2:], ["payment"]]


@pytest.mark.parametrize("boom", ["openrouter", "timeout", "short", "ragged"])
def test_any_failure_returns_none_and_leaves_the_cache_alone(fake_embedder, monkeypatch, boom):
    sim()
    file_before = embeddings._cache_path().read_bytes()
    memory_before = {n: e[0] for n, e in embeddings._cache[embeddings.MODEL].items()}
    texts = list(TEXTS)
    texts[0] = "a-therapy: changed text"
    if boom == "short":
        monkeypatch.setattr(embeddings, "request_embeddings", lambda **kw: [[1.0, 0.0]])
    elif boom == "ragged":
        monkeypatch.setattr(embeddings, "request_embeddings", lambda **kw: [[1.0, 0.0], [1.0]])
    else:
        fake_embedder.fail = OpenRouterError("x") if boom == "openrouter" else httpx.TimeoutException("slow")
    assert sim(texts=texts) is None
    assert embeddings._cache_path().read_bytes() == file_before
    assert {n: e[0] for n, e in embeddings._cache[embeddings.MODEL].items()} == memory_before


def test_failure_starts_a_cooldown_then_retries(fake_embedder, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(embeddings, "_now", lambda: clock[0])
    fake_embedder.fail = OpenRouterError("x")
    assert sim() is None
    fake_embedder.fail = None
    clock[0] += embeddings.COOLDOWN_S - 1
    assert sim() is None and fake_embedder.count == 1  # still cooling down: no call at all
    clock[0] += 2
    assert sim() == [0] and fake_embedder.count == 2


def test_failure_log_names_the_error_type_only(fake_embedder, caplog):
    fake_embedder.fail = OpenRouterError("SECRET-ERROR-TEXT")
    with caplog.at_level("WARNING"):
        sim("SECRET-QUERY counsellor", texts=["SECRET-AGENT-TEXT"] * 4)
    assert "OpenRouterError" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"a-therapy": 5}', '{"a-therapy": ["x", "%%%"]}'])
def test_unreadable_cache_file_is_ignored_with_a_warning(fake_embedder, caplog, content):
    path = embeddings._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert sim() == [0]
    assert "cache" in caplog.text.lower()
    embeddings._cache = None
    assert sim("loaf") == [1]  # the file was rewritten and now loads
    assert fake_embedder.calls[1]["inputs"] == ["loaf"]


def test_removed_agents_are_pruned_when_the_cache_is_saved(fake_embedder):
    sim()
    sim("x", names=NAMES[:2], texts=TEXTS[:2])  # nothing new: no save, so nothing pruned yet
    assert set(saved()) == set(NAMES)
    sim("x", names=NAMES[:2] + ["e-new"], texts=TEXTS[:2] + ["e-new: sourdough"])
    assert set(saved()) == {"a-therapy", "b-bread", "e-new"}


def test_cached_vectors_of_another_size_are_never_compared(fake_embedder):
    path = embeddings._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    three = base64.b64encode(array("f", [1.0, 0.0, 0.0]).tobytes()).decode()
    path.write_text(json.dumps({n: [hashlib.sha1(t.encode()).hexdigest(), three] for n, t in zip(NAMES, TEXTS)}))
    assert sim() == []  # the 64-dim query is not compared with 3-dim vectors
    assert fake_embedder.calls[0]["inputs"] == ["counsellor visit"]
    assert sim() == [0]  # the 3-dim vectors were dropped, so this call embeds those agents again
    assert fake_embedder.calls[1]["inputs"] == ["counsellor visit"] + TEXTS


def test_the_data_flow_notice_is_logged_once_without_any_text(fake_embedder, monkeypatch, caplog):
    monkeypatch.setattr(embeddings, "_told", False)
    with caplog.at_level("INFO"):
        sim("SECRET-QUERY counsellor")
        sim("SECRET-QUERY loaf")
    notices = [r for r in caplog.records if "OPENPOKE_EMBED=0" in r.getMessage()]
    assert len(notices) == 1 and embeddings.MODEL in notices[0].getMessage()
    assert "SECRET" not in caplog.text
