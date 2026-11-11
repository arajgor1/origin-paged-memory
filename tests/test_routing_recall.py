"""H8 routing-recall rows on synthetic items with precomputed (hash) vectors; plus the H10 ITT contrast."""
from __future__ import annotations

import numpy as np
import pytest

from opa import claims as cl
from opa.analysis import routing_recall as rr
from opa.analysis import tables as tb
from opa.tracks.track_a.embed import compute_item_embeddings, embeddings_to_bytes
from opa.tracks.track_a.index import HashEmbedder
from opa.tracks.track_a.systems import SYSTEMS, session_cfg
from opa.tracks.track_a.thrash import CONTRAST_KEY, H10_EXTRA_COLUMNS, h10_contrast, write_h10_contrast

VOCAB = [f"w{chr(97 + i // 26)}{chr(97 + i % 26)}x" for i in range(300)]


def make_items(n_items=24, n_orig=40, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for k in range(n_items):
        texts = [" ".join(rng.choice(VOCAB, 12, replace=False)) for _ in range(n_orig)]
        gold = sorted(int(x) for x in rng.choice(n_orig, 2, replace=False))
        q = " ".join(w for g in gold for w in texts[g].split()[:5])
        items.append({"item_id": f"it{k:02d}", "dataset": "lbmhqa", "split": "dev", "question": q,
                      "origins": [{"origin_id": i, "text": t, "thread_id": 0, "n_tokens": 20} for i, t in enumerate(texts)],
                      "gold": {"facts": [{"origin_id": g, "hop": j} for j, g in enumerate(gold)], "fact_origins": gold,
                               "aliases": {}, "confusers": [], "distractors": []}})
    return items


def make_skeletons(items, version="sllm-v2", status="ok"):
    """S-llm skeleton payloads as build_skeletons writes them; entities = the first words of each origin."""
    return {it["item_id"]: {"item_id": it["item_id"], "prompt_version": version, "status": status,
                            "entities": {str(o["origin_id"]): o["text"].split()[:3] for o in it["origins"]}}
            for it in items}


@pytest.fixture(scope="module")
def world():
    items = make_items()
    return items, embeddings_to_bytes(compute_item_embeddings(items, HashEmbedder()))


def test_rows_primary_first_and_consistent(world):
    items, npz = world
    rows = rr.routing_recall_rows(items, npz, make_skeletons(items), M=4, split="dev", n_boot=500, run_ids=["r1"])
    assert [r["key"] for r in rows] == ["descent_beam4_vs_flat_recall_at_M", "descent_beam1_vs_flat_recall_at_M",
                                        "descent_beam2_vs_flat_recall_at_M"]
    p = rows[0]
    assert p["n"] == 24 and p["split"] == "dev" and p["M"] == 4 and p["beam"] == 4 and p["n_excluded"] == 0
    d = np.array([v["descent"] - v["flat"] for v in p["per_item"].values()]) * 100.0
    assert p["value"] == pytest.approx(d.mean()) and p["sd_diff"] == pytest.approx(d.std(ddof=1))
    assert p["n10"] == int((d > 0).sum()) and p["n01"] == int((d < 0).sum())
    assert p["flat_recall"] > 0.0
    assert p["scores_flat"] == 40 and rows[1]["scores_descent"] < rows[0]["scores_descent"] < p["scores_flat"] * 2
    assert rows[1]["scores_descent"] <= rows[2]["scores_descent"] <= rows[0]["scores_descent"]


def test_matches_session_routines(world):
    from opa.tracks.track_a.embed import PrecomputedEmbedder
    from opa.tracks.track_a.index import Session
    items, npz = world
    pre = PrecomputedEmbedder(npz)
    it = items[3]
    bound = pre.for_item(it)
    sess = Session(it, session_cfg("S12", {}), bound, skeleton_llm=make_skeletons([it])[it["item_id"]]["entities"])
    q = bound.embed_query([it["question"]])[0]
    out = rr.route_item(it, bound, 4, make_skeletons([it])[it["item_id"]]["entities"])
    assert out["flat"]["cands"] == sess.flat_scan(it["question"], q, 4)[0]
    assert out["beam4"]["cands"] == sess.descend(it["question"], q, beam=4, top=4)[0]


def test_cfg_is_the_s12_cfg(world, monkeypatch):
    items, npz = world
    assert rr.s12_cfg() == session_cfg("S12", {}) and rr.s12_cfg({"b": 4})["b"] == 4
    assert rr.s12_cfg()["skeleton"] == SYSTEMS["S12"][0] == "hyb" and rr.s12_cfg()["graph"] is True
    from opa.tracks.track_a import index
    seen = []
    orig = index.Session.__init__

    def spy(self, item, cfg, embedder, skeleton_llm=None):
        seen.append((dict(cfg), skeleton_llm))
        orig(self, item, cfg, embedder, skeleton_llm)

    monkeypatch.setattr(index.Session, "__init__", spy)
    rr.routing_recall_rows(items[:2], npz, make_skeletons(items[:2]), M=4, split="dev", n_boot=50)
    assert seen and all(c == session_cfg("S12", {}) and sk for c, sk in seen)


def test_skeleton_checks(world, tmp_path):
    import json
    items, npz = world
    for bad, msg in ((make_skeletons(items, version="sllm-v1"), "prompt_version"),
                     (make_skeletons(items, status="error"), "quality check"),
                     ({}, "no S-llm skeleton")):
        with pytest.raises(RuntimeError, match=msg):
            rr.routing_recall_rows(items, npz, bad, M=4, split="dev", n_boot=50)
    d = tmp_path / "skeleton_llm"
    d.mkdir()
    for iid, payload in make_skeletons(items).items():
        (d / f"{iid}.json").write_text(json.dumps(payload))
    rows = rr.routing_recall_rows(items, npz, str(d), M=4, split="dev", n_boot=50)     # directory form
    assert rows[0]["n"] == 24


def test_exclusions_and_split(world):
    items, npz = world
    extra = make_items(2, seed=9)
    extra[0]["item_id"] = "unembedded"
    extra[1]["gold"] = {"facts": [], "fact_origins": []}
    rows = rr.routing_recall_rows(items + extra, npz, make_skeletons(items + extra), M=4, split="test", n_boot=200)
    why = {e["item_id"]: e["reason"] for e in rows[0]["excluded"]}
    assert set(why) == {"unembedded", extra[1]["item_id"]} and why[extra[1]["item_id"]] == "no gold-fact origins"
    assert why["unembedded"].startswith("no usable embeddings") and rows[0]["n"] == 24 and rows[0]["n_excluded"] == 2
    with pytest.raises(ValueError, match="split"):
        rr.routing_recall_rows(items, npz, make_skeletons(items), M=4, split="n/a")
    with pytest.raises(ValueError):
        rr.routing_recall_rows(items, npz, make_skeletons(items), M=4, split="dev", beams=(1, 2))


def test_table_blanks_p_raw_for_primary_only(world, tmp_path):
    items, npz = world
    rows = rr.routing_recall_rows(items, npz, make_skeletons(items), M=4, split="dev", n_boot=300, run_ids=["r1"])
    rr.write_routing_table(tmp_path / "t.csv", rows)
    t = {r["key"]: r for r in tb.read_table(tmp_path / "t.csv")}
    prim = t["descent_beam4_vs_flat_recall_at_M"]
    assert prim["p_raw"] == "" and prim["test"] == tb.NI_TEST_LABEL and prim["sd_diff"] != ""
    assert t["descent_beam1_vs_flat_recall_at_M"]["p_raw"] != ""
    res = cl.result_from_table(tmp_path, "t.csv", "descent_beam4_vs_flat_recall_at_M", level="agent", primary=True)
    assert res["p_raw"] is None and res["n"] == 24 and res["ci95"] is not None


# ---------------------------------------------------------------- H10 intention-to-treat
def rec(i, M, r, rp, status="ok"):
    return {"item_id": f"i{i}", "M": M, "r": r, "repeated_page_ins": rp, "status": status}


def test_h10_itt_imputes_worst_value_and_counts(tmp_path):
    # cell (r=4, M=2): non-failed repeated page-ins are 6, 5, 3, 0, 0, 0, 0 -> worst = 6; i9 is pinned-only
    unp = [rec(0, 2, 4, 6), rec(1, 2, 4, 5), rec(2, 2, 4, 7, "error"), rec(3, 2, 4, 4, "budget"), rec(4, 2, 4, 3),
           rec(5, 2, 2, 0)]
    pin = [rec(0, 2, 4, 0), rec(1, 2, 4, 1, "does_not_fit"), rec(2, 2, 4, 0), rec(3, 2, 4, 0), rec(4, 2, 4, 0),
           rec(5, 2, 2, 0), rec(9, 2, 4, 0)]
    c = h10_contrast(unp, pin, split="dev", n_boot=200)
    assert c["key"] == CONTRAST_KEY and c["n"] == 5 and c["split"] == "dev"      # r > M pairs present in both arms
    assert c["imputed_worst"] == {"r4-M2": 6}
    # unpinned [6, 5, 6 (error), 6 (budget), 3]; pinned [0, 6 (does_not_fit), 0, 0, 0]
    assert c["value"] == pytest.approx(np.mean([6, -1, 6, 6, 3]))
    assert c["sd_diff"] == pytest.approx(np.std([6, -1, 6, 6, 3], ddof=1))
    assert (c["n_error_unpinned"], c["n_budget_unpinned"], c["n_not_fit_unpinned"]) == (1, 1, 0)
    assert (c["n_error_pinned"], c["n_budget_pinned"], c["n_not_fit_pinned"]) == (0, 0, 1)
    assert c["excluded"] == [{"item_id": "i9", "M": 2, "reason": "missing unpinned"}] and c["n_excluded"] == 1
    write_h10_contrast(tmp_path / "h10.csv", c)
    row = tb.read_table(tmp_path / "h10.csv")[0]
    assert row["n_budget_unpinned"] == "1" and row["n_not_fit_pinned"] == "1" and set(H10_EXTRA_COLUMNS) <= set(row)


def test_h10_pinned_failure_cannot_increase_the_reduction():
    unp = [rec(i, 2, 4, v) for i, v in enumerate([5, 6, 4, 7])]
    pin = [rec(i, 2, 4, v) for i, v in enumerate([1, 0, 2, 1])]
    base = h10_contrast(unp, pin, split="dev", n_boot=100)["value"]
    for status in ("error", "budget", "does_not_fit"):
        pin_f = [dict(pin[0], status=status)] + pin[1:]
        assert h10_contrast(unp, pin_f, split="dev", n_boot=100)["value"] <= base
    pin_f = [dict(pin[0], status="error")] + pin[1:]
    assert h10_contrast(unp, pin_f, split="dev", n_boot=100)["value"] < base          # imputed 7 > observed 1


def test_h10_unpinned_failure_gets_the_cell_max():
    unp = [rec(0, 2, 4, 5), rec(1, 2, 4, 9, "error"), rec(2, 2, 4, 3)]
    pin = [rec(0, 2, 4, 1), rec(1, 2, 4, 2), rec(2, 2, 4, 0)]
    c = h10_contrast(unp, pin, split="test", n_boot=100)
    assert c["imputed_worst"] == {"r4-M2": 5}                 # max over non-failed records of the cell, both arms
    assert c["value"] == pytest.approx(np.mean([5 - 1, 5 - 2, 3 - 0]))


def test_h10_cell_with_only_failures_excludes_the_pair():
    unp = [rec(0, 2, 4, 5), rec(1, 2, 6, 3, "error"), rec(2, 2, 6, 3, "budget")]
    pin = [rec(0, 2, 4, 1), rec(1, 2, 6, 0, "error"), rec(2, 2, 6, 0, "does_not_fit")]
    c = h10_contrast(unp, pin, split="dev", n_boot=100)
    assert c["n"] == 1 and c["n_excluded"] == 2
    assert {e["reason"] for e in c["excluded"]} == {"no observed value in cell"}
    with pytest.raises(ValueError, match="observed value"):
        h10_contrast(unp[1:], pin[1:], split="dev")


def test_h10_split_required():
    unp, pin = [rec(0, 2, 4, 3)], [rec(0, 2, 4, 1)]
    with pytest.raises(TypeError):
        h10_contrast(unp, pin)
    for bad in ("n/a", None, "train"):
        with pytest.raises(ValueError, match="split"):
            h10_contrast(unp, pin, split=bad)


def test_h10_duplicate_records_raise():
    with pytest.raises(ValueError, match="duplicate"):
        h10_contrast([rec(0, 2, 4, 1), rec(0, 2, 4, 2)], [rec(0, 2, 4, 0)], split="dev")
