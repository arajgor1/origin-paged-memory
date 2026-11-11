"""H5 distortion check (opa.analysis.distortion): layouts come from the harness Session."""
import csv
import gzip
import json

import numpy as np
import pytest

from opa.analysis import distortion as dz
from opa.data import mtlbmhqa
from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
from opa.geometry import TreeLayout


@pytest.fixture(scope="module")
def items():
    filler, tok = SyntheticFiller(n_books=8, words_per_book=30000), WhitespaceTokenizer()
    return [mtlbmhqa.make_item(s, 4, 3, 8192, True, 0.0, filler, tok) for s in (0, 1)]


def test_distortion_metric_known_cases():
    dT = np.array([1.0, 2.0, 3.0])
    r = dz.distortion(5.0 * dT, dT)                       # pure rescaling is free
    assert abs(r["worst"] - 1.0) < 1e-12 and r["mean_rel"] < 1e-12 and abs(r["scale"] - 0.2) < 1e-12
    r = dz.distortion(np.array([1.0, 1.0, 3.0]), dT)      # contraction of the middle pair
    assert r["worst"] > 1.5 and r["mean_rel"] > 0
    assert dz.distortion(np.array([0.0, 1.0, 1.0]), dT)["worst"] == float("inf")


def test_path_on_a_line_has_euclidean_distortion_one():
    lay = TreeLayout(m=4, delta=1.0)
    u = np.array([1.0, 0, 0, 0])
    for c in range(1, 12):
        lay.add_node(c, c - 1, u)
    pairs = [(i, j) for i in range(12) for j in range(i)]
    d = np.array([lay.distance(i, j, "A1") for i, j in pairs])
    dT = np.array([lay.distance(i, j, "A3") for i, j in pairs])
    r = dz.distortion(d, dT)
    assert abs(r["worst"] - 1.0) < 1e-9 and r["mean_rel"] < 1e-9


def test_branching_tree_euclidean_above_one_hyperbolic_near_one():
    # a binary tree (not the harness shape, a pure metric sanity check): large delta, below the float64 horizon
    n = 61
    parent = [None] + [(i - 1) // 2 if i > 1 else 0 for i in range(1, n)]
    lay = TreeLayout(m=32, delta=6.0, d_clamp=1e9)
    from opa.geometry import spread_directions
    kids = {}
    for c in range(1, n):
        kids.setdefault(parent[c], []).append(c)
    for p in sorted(kids):
        for c, u in zip(kids[p], spread_directions(len(kids[p]), 32, seed=p)):
            lay.add_node(c, p, u)
    pairs = [(i, j) for i in range(n) for j in range(i)]
    dT = np.array([lay.distance(i, j, "A3") for i, j in pairs]) / 6.0
    a1 = dz.distortion([lay.distance(i, j, "A1") for i, j in pairs], dT)
    a2 = dz.distortion([lay.distance(i, j, "A2") for i, j in pairs], dT)
    assert a1["worst"] > 1.5
    assert a2["worst"] < 1.25 and a2["mean_rel"] < 0.03


def test_module_distances_equal_session_distance(items):
    it = items[0]
    for gold in (True, False):
        sess = dz.make_session(it, gold)
        pairs = dz.pair_list(sess.N, max_pairs=10 ** 6)
        assert len(pairs) == sess.N * (sess.N - 1) // 2
        a1, a2, dT = dz.session_distances(sess, pairs)
        for k in (0, 7, len(pairs) // 2, len(pairs) - 1):
            i, j = pairs[k]
            assert a1[k] == sess.distance(i, j, "A1")
            assert a2[k] == sess.distance(i, j, "A2")
            assert dT[k] * sess.layout.delta == sess.distance(i, j, "A3")
        # threads are chains: one direction per thread, parent = previous origin of the same thread
        for o in range(sess.N):
            par = sess.parent_origin[o]
            assert par is None or (par < o and sess.thread_of[par] == sess.thread_of[o])
    assert dz.make_session(it, True).n_threads == 4


def test_q_anchored_pairs_and_determinism(items):
    assert dz.pair_list(16, q=15) == [(15, c) for c in range(15)]
    a = dz.item_metrics(items[0], True)
    b = dz.item_metrics(items[0], True)
    assert {k: v for k, v in a.items() if k != "dist_sec"} == {k: v for k, v in b.items() if k != "dist_sec"}
    assert a["q_pairs"] == a["n_origins"] - 1
    assert 0 <= a["q_clamp_frac"] <= 1 and a["q_clamp_count"] == a["q_clamp_frac"] * a["q_pairs"]


def test_row_csv_and_sidecar(tmp_path, items):
    rows = [dz.cell_row("c_synth_gold", "T4-h3-8k-cross", "synth", items, True),
            dz.cell_row("c_synth_inferred", "T4-h3-8k-cross", "synth", items, False)]
    assert rows[0]["T"] == 4 and rows[0]["h"] == 3 and rows[0]["length_tokens"] == 8192
    assert rows[0][dz.PRIMARY_RATIO] == rows[0]["q_A1_worst"] / rows[0]["q_A2_worst"] > 0
    out = tmp_path / "t" / "h5.csv"
    rec = {"uri": "gs://b/opa/data/x/items.jsonl.gz", "sha256": "ab" * 32}
    side = dz.write_outputs(rows, out, [rec], dz.synthetic_provenance())
    text = out.read_bytes()
    assert b"\r" not in text and b"e-" not in text
    got = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(got[0].keys()) == dz.COLUMNS and [r["key"] for r in got] == ["c_synth_gold", "c_synth_inferred"]
    assert float(got[0][dz.PRIMARY_RATIO]) == rows[0][dz.PRIMARY_RATIO]
    s = json.loads(side.read_text(encoding="utf-8"))
    assert s["script"] == "code/opa/analysis/distortion.py" and s["inputs"] == [rec] and s["created_at"].endswith("Z")
    assert s["synthetic"]["seeds"] == [0, 1, 2] and s["synthetic"]["filler"]["class"] == "SyntheticFiller"
    out2 = tmp_path / "t" / "again.csv"
    dz.write_outputs(rows, out2, [rec])
    assert out2.read_bytes() == text


def test_read_cell_dir(tmp_path, items):
    d = tmp_path / "v9" / "T4-h3-8k-cross"
    d.mkdir(parents=True)
    (d / "items.jsonl.gz").write_bytes(gzip.compress("\n".join(json.dumps(i) for i in items).encode("utf-8")))
    (d / "manifest.json").write_text(json.dumps({"cell": "T4-h3-8k-cross"}), encoding="utf-8")
    cell, got, rec = dz.read_cell_dir(d)
    assert cell == "T4-h3-8k-cross" and len(got) == 2
    assert rec["uri"].startswith("gs://") and rec["uri"].endswith("/opa/data/" + "/".join(d.parts[-4:]) + "/items.jsonl.gz")
    assert len(rec["sha256"]) == 64


def test_default_clamp_none_vs_36_and_labels(items):
    prim = dz.cell_row("a", "T4-h3-8k-cross", "synth", items, True)                       # harness default
    sens = dz.cell_row("b", "T4-h3-8k-cross", "synth", items, True, d_clamp=36.0)
    assert prim["d_clamp"] == "none" and sens["d_clamp"] == "36"
    assert prim["q_A1_worst"] == sens["q_A1_worst"]                                        # Euclidean arm unaffected
    inf = dz.cell_row("c", "T4-h3-8k-cross", "synth", items, False)
    assert inf["embedder"] == "hash-bow-512" and prim["embedder"] == ""
    assert dz.make_session(items[0], True).layout.d_clamp is None                          # Track A default
    assert prim["q_A1_affine"] >= 0 and prim["q_A2_affine"] >= 0


def test_affine_residual_known():
    dT = np.arange(1.0, 9.0)
    assert dz.affine_residual(3.0 * dT + 2.0, dT) < 1e-12
    assert dz.affine_residual(dT ** 2, dT) > 0.05


def test_largest_ratio_only_real_gold_default_rows():
    P = dz.PRIMARY_RATIO
    v = dz.GENERATOR_VERSION
    rows = [{"source": "synth", "threads": "gold", "d_clamp": "none", P: 9.0, "key": "s"},
            {"source": "v1", "threads": "gold", "d_clamp": "none", P: 8.0, "key": "o"},
            {"source": v, "threads": "inferred", "d_clamp": "none", P: 7.0, "key": "i"},
            {"source": v, "threads": "gold", "d_clamp": "36", P: 6.0, "key": "c"},
            {"source": v, "threads": "gold", "d_clamp": "none", P: 1.3, "key": "a"},
            {"source": v, "threads": "gold", "d_clamp": "none", P: 1.1, "key": "b"}]
    assert dz.largest_ratio_row(rows)["key"] == "a"
    assert dz.largest_ratio_row(rows[:4]) is None


def test_flagged_pairs_counted_and_columns(items):
    r = dz.cell_row("a", "T4-h3-8k-cross", "synth", items, True)
    for k in ("q_beyond_safe_count", "q_beyond_safe_frac", "q_exact_fallback", "all_exact_fallback"):
        assert k in r and k in dz.COLUMNS
    assert "dist_sec" in r and "dist_sec" not in dz.COLUMNS
    assert r["q_beyond_safe_count"] == 0 and r["q_exact_fallback"] == 0 and r["q_clamp_count"] == 0   # 16-origin chains


def test_long_chain_uses_exact_fallback_and_agrees_with_tree(items):
    # one thread of 45 origins: pairs beyond the safe bound go to the exact path; delta=1 chain distance = tree distance
    it = dict(items[0])
    n = 48
    it["origins"] = [{"origin_id": o, "text": f"origin {o} alpha beta", "thread_id": 0, "n_tokens": 512} for o in range(n)]
    sess = dz.make_session(it, True)
    d = [sess.distance(n - 1, c, "A2") for c in range(n - 1)]
    t = [sess.distance(n - 1, c, "A3") for c in range(n - 1)]
    assert sess.geometry_counts()["a2_exact_pairs"] > 0 and sess.geometry_counts()["a2_capped_pairs"] == 0
    assert np.allclose(d, t, atol=0.01)
    m = dz.item_metrics(it, True)
    assert m["q_beyond_safe_count"] > 0 and m["q_exact_fallback"] == m["q_beyond_safe_count"]
    assert m["q_A2_worst"] < 1.01
