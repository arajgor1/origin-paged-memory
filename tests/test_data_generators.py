"""P0.5 dataset generators: gold correctness, determinism, split guard. No network: SyntheticFiller + whitespace tokenizer."""
import json

import numpy as np
import pytest

from opa.data import SplitRefused, check_split, drift, lbmhqa, mtlbmhqa, thrash
from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
from opa.data.names import hamming1
from opa.data.schema import late_bound_flags, validate_item

L32 = 32768


@pytest.fixture(scope="module")
def filler():
    return SyntheticFiller(n_books=40, words_per_book=30000)


@pytest.fixture(scope="module")
def tok():
    return WhitespaceTokenizer()


def _lb(filler, tok, seed=0, hops=3, placement="late", alpha=0.0, length=L32):
    return lbmhqa.make_item(seed, hops, length, placement, alpha, filler, tok)


def _mt(filler, tok, seed=0, cross=True, n_threads=8, hops=3, **kw):
    return mtlbmhqa.make_item(seed, n_threads, hops, 32768, cross, 0.0, filler, tok, **kw)


# ---------------------------------------------------------------- Definition 6
def test_late_bound_definition():
    assert late_bound_flags([5, 9, 12]) == [False, False, False]       # forward
    assert late_bound_flags([12, 9, 5]) == [False, True, True]         # reversed
    assert late_bound_flags([9, 12, 5]) == [False, False, True]
    assert late_bound_flags([0, 0]) == [False, False]                  # same origin is not "before"


# ---------------------------------------------------------------- LB-MHQA
@pytest.mark.parametrize("hops", [2, 3, 4])
def test_lbmhqa_h_late_per_placement(filler, tok, hops):
    for seed in range(3):
        fw = _lb(filler, tok, seed, hops, "forward")
        lt = _lb(filler, tok, seed, hops, "late")
        mx = _lb(filler, tok, seed, hops, "mixed")
        for it in (fw, lt, mx):
            validate_item(it)
        assert fw["gold"]["h_late"] == 0
        assert lt["gold"]["h_late"] == hops - 1
        assert all(f["late_bound"] for f in lt["gold"]["facts"][1:]) and not lt["gold"]["facts"][0]["late_bound"]
        if hops >= 3:
            assert 0 < mx["gold"]["h_late"] < hops - 1
        fo = lt["gold"]["fact_origins"]
        assert fo == sorted(fo, reverse=True) and len({o // 4 for o in fo}) == hops


def test_lbmhqa_gold_fields(filler, tok):
    it = _lb(filler, tok, 7, 3, "late")
    g = it["gold"]
    assert len(it["origins"]) == L32 // 512 and it["question_thread"] == 0
    assert g["entities"][0] in it["question"] and it["answer"] in g["facts"][-1]["text"]
    for j, f in enumerate(g["facts"]):
        o = it["origins"][f["origin_id"]]
        assert o["text"][slice(*f["char_span"])] == f["text"]
        assert f["subject"] == g["entities"][j]
        if j < 2:
            assert f["reveals"] == g["entities"][j + 1]
    assert g["confusers"] == [] and g["cross_thread_hop"] is None
    assert len(g["distractors"]) == 2 * 3
    assert not set(d["subject"] for d in g["distractors"]) & set(g["entities"])
    assert it["answer"] not in g["distractor_values"]
    for o in it["origins"]:
        assert 400 < o["n_tokens"] < 700


def test_lbmhqa_alias_rate(filler, tok):
    n_alias = sum(len(v) for s in range(10) for v in _lb(filler, tok, s, 3, "late", alpha=0.3)["gold"]["aliases"].values())
    assert n_alias > 0
    for s in range(3):
        it = _lb(filler, tok, s, 3, "late", alpha=0.3)
        validate_item(it)
        for f in it["gold"]["facts"]:
            if f["subject_surface"] != f["subject"]:
                assert f["subject_surface"] in it["gold"]["aliases"][f["subject"]]
                assert f["subject"].split()[0].lower() == f["subject_surface"].split()[0].lower()
    assert all(not _lb(filler, tok, s, 3, "late", alpha=0.0)["gold"]["aliases"] for s in range(3))


# ---------------------------------------------------------------- MT-LB-MHQA
def test_mt_cross_thread_hop(filler, tok):
    for seed in range(4):
        it = _mt(filler, tok, seed, cross=True)
        validate_item(it)
        g, qt = it["gold"], it["question_thread"]
        assert qt == g["thread_of"][-1]
        last = g["facts"][-1]
        assert g["cross_thread_hop"] == 2 and last["cross_thread"]
        assert g["thread_of"][last["origin_id"]] != qt
        assert last["late_bound"] and last["origin_id"] < g["facts"][-2]["origin_id"]
        assert all(g["thread_of"][f["origin_id"]] == qt for f in g["facts"][:-1])
        assert g["h_late"] == 2


def test_mt_nocross(filler, tok):
    for seed in range(4):
        it = _mt(filler, tok, seed, cross=False)
        validate_item(it)
        g, qt = it["gold"], it["question_thread"]
        assert g["cross_thread_hop"] is None
        assert all(g["thread_of"][o] == qt for o in g["fact_origins"])
        assert g["fact_origins"] == sorted(g["fact_origins"], reverse=True)
        assert g["h_late"] == 2


def test_mt_threads_interleaved_and_aligned(filler, tok):
    it = _mt(filler, tok, 3)
    th = it["gold"]["thread_of"]
    assert all(len(set(th[a:a + 4])) == 1 for a in range(0, len(th), 4))
    runs = it["gold"]["run_schedule"]
    assert sorted(set(runs)) == list(range(8))
    assert [runs.index(t) for t in range(8)] == sorted(runs.index(t) for t in range(8))   # ids by first appearance
    assert any(runs[i] != runs[i + 1] and runs[i] in runs[i + 2:] for i in range(len(runs) - 2))  # threads resume
    books = {}
    for o in it["origins"]:
        books.setdefault(o["thread_id"], set()).add(o["book_id"])
    firsts = [sorted(b)[0] for b in books.values()]
    assert len(set(firsts)) == 8                                   # each thread its own book


def test_mt_confusers(filler, tok):
    for seed in range(3):
        it = _mt(filler, tok, seed)
        g, qt = it["gold"], it["question_thread"]
        final = g["entities"][-1]
        assert len(g["confusers"]) == 7
        assert sorted(c["thread_id"] for c in g["confusers"]) == [t for t in range(8) if t != qt]
        vals = [c["value"] for c in g["confusers"]]
        assert len(set(vals)) == 7 and it["answer"] not in vals
        for c in g["confusers"]:
            assert hamming1(c["entity"], final) and c["entity"] != final
            assert it["origins"][c["origin_id"]]["thread_id"] == c["thread_id"]
            assert it["origins"][c["origin_id"]]["text"][slice(*c["char_span"])] == c["text"]
    none = _mt(filler, tok, 0, confuser_rate=0.0)
    validate_item(none)
    assert none["gold"]["confusers"] == []


# ---------------------------------------------------------------- thrash
@pytest.mark.parametrize("r", [2, 3, 4, 6])
def test_thrash_needed_origins(filler, tok, r):
    for seed in range(3):
        it = thrash.make_item(seed, r, filler, tok)
        validate_item(it)
        g = it["gold"]
        assert len(it["origins"]) == 32
        assert len(set(g["needed_origins"])) == r == len(g["quotes"])
        prices = [float(q["price"].split()[0]) for q in g["quotes"]]
        assert len(set(prices)) == r
        assert it["answer"] == g["quotes"][int(np.argmin(prices))]["supplier"]
        for q in g["quotes"]:
            assert q["supplier"] in it["question"] and g["part"] in q["text"]


# ---------------------------------------------------------------- drift
def test_drift_probe_gold():
    for seed in range(20):
        s = drift.make_session(seed)
        validate_item(s)
        assert len(s["turns"]) == 50 and len(s["constraints"]) == 10
        upd = [(c, u) for c in s["constraints"] for u in c["updates"]]
        assert len(upd) == 3
        assert sorted(u["turn"] for _, u in upd)[0] <= 10 and 11 <= sorted(u["turn"] for _, u in upd)[1] <= 25
        for c, u in upd:
            assert u["thread"] != c["initial"]["thread"] == "design"
            for p in s["probes"]:
                assert p["gold"][c["key"]] == (u["value"] if p["turn"] >= u["turn"] else c["initial"]["value"])
        for c in s["constraints"]:
            for n in c["near_misses"]:
                assert all(p["gold"][c["key"]] != n["value"] for p in s["probes"])
                assert n["text"] in s["turns"][n["turn"] - 1]["text"]
            if not c["updates"]:
                assert all(p["gold"][c["key"]] == c["initial"]["value"] for p in s["probes"])
        assert sum(len(c["near_misses"]) for c in s["constraints"]) == 6
        kinds = {c["kind"] for c in s["constraints"]}
        assert {"temperature", "tolerance", "part", "supplier", "date"} <= kinds
        assert any("±" in c["initial"]["value"] for c in s["constraints"] if c["kind"] == "tolerance")


def test_drift_near_miss_after_update_does_not_revert():
    found = False
    for seed in range(40):
        s = drift.make_session(seed)
        for c in s["constraints"]:
            for u in c["updates"]:
                for n in c["near_misses"]:
                    if n["turn"] > u["turn"]:
                        found = True
                        assert s["probes"][-1]["gold"][c["key"]] == u["value"]
    assert found


# ---------------------------------------------------------------- determinism and split guard
def test_determinism(filler, tok):
    a = json.dumps(_lb(filler, tok, 5, 3, "mixed"), sort_keys=True)
    b = json.dumps(_lb(filler, tok, 5, 3, "mixed"), sort_keys=True)
    assert a == b
    m1, m2 = _mt(filler, tok, 2), _mt(filler, tok, 2)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert json.dumps(drift.make_session(3)) == json.dumps(drift.make_session(3))
    assert json.dumps(thrash.make_item(4, 3, filler, tok)) == json.dumps(thrash.make_item(4, 3, filler, tok))
    # different seeds and different cells give different items
    assert _lb(filler, tok, 6, 3, "mixed")["gold"]["entities"] != _lb(filler, tok, 5, 3, "mixed")["gold"]["entities"]
    assert _lb(filler, tok, 5, 3, "late")["gold"]["entities"] != _lb(filler, tok, 5, 3, "forward")["gold"]["entities"]
    assert _mt(filler, tok, 2, cross=False)["gold"]["entities"] != m1["gold"]["entities"]


def test_test_split_refused(filler, tok, monkeypatch):
    from opa import launch
    monkeypatch.setattr(launch, "prereg_tagged", lambda: False)
    with pytest.raises(SplitRefused):
        lbmhqa.make_item(100000, 3, L32, "late", 0.0, filler, tok, split="test")
    with pytest.raises(SplitRefused):
        mtlbmhqa.make_item(100000, 8, 3, 32768, True, 0.0, filler, tok, split="test")
    with pytest.raises(SplitRefused):
        thrash.make_item(100000, 3, filler, tok, split="test")
    with pytest.raises(SplitRefused):
        drift.make_session(100000, split="test")
    with pytest.raises(ValueError):
        check_split("dev", 100000)                 # dev seeds are 0..999
    with pytest.raises(ValueError):
        check_split("train", 0)
    monkeypatch.setattr(launch, "prereg_tagged", lambda: True)
    with pytest.raises(ValueError):
        check_split("test", 5)                     # test seeds start at 100000
    check_split("test", 100000)


def test_pg19_filler_split_guard(filler, tok):
    class FakePG19(SyntheticFiller):
        pg19_split = "test"
    with pytest.raises(ValueError):
        lbmhqa.make_item(0, 3, L32, "late", 0.0, FakePG19(n_books=12), tok, split="dev")


@pytest.mark.parametrize("T,rb", [(4, 4), (8, 4), (16, 2)])
def test_mt_thread_grid_at_32k(filler, tok, T, rb):
    for cross in (False, True):
        for seed in range(3):
            it = _mt(filler, tok, seed, cross=cross, n_threads=T)
            validate_item(it)
            g, qt = it["gold"], it["question_thread"]
            assert it["params"]["run_blocks"] == rb and sorted(set(g["thread_of"])) == list(range(T))
            assert g["thread_of"].count(qt) >= 4 * rb and g["h_late"] == 2


def test_mt_rejects_impossible_grids(filler, tok):
    with pytest.raises(ValueError):
        mtlbmhqa.make_item(0, 40, 3, 8192, False, 0.0, filler, tok)          # 16 origins, 40 threads
    with pytest.raises(ValueError):
        mtlbmhqa.make_item(0, 16, 3, 8192, False, 0.0, filler, tok)          # 8K: too few runs for T=16 and h=3
