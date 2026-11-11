"""tau_thread calibration (opa.analysis.tau_calibration)."""
import copy
import csv
import gzip
import json

import numpy as np
import pytest

from opa.analysis import tau_calibration as tc
from opa.data import mtlbmhqa
from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
from opa.tracks.track_a.embed import PrecomputedEmbedder, compute_item_embeddings, embeddings_to_bytes
from opa.tracks.track_a.index import HashEmbedder


@pytest.fixture(scope="module")
def items():
    filler, tok = SyntheticFiller(n_books=8, words_per_book=30000), WhitespaceTokenizer()
    return [mtlbmhqa.make_item(s, 4, 3, 8192, True, 0.0, filler, tok) for s in (0, 1)]


@pytest.fixture(scope="module")
def npz_bytes(items):
    return embeddings_to_bytes(compute_item_embeddings(items, HashEmbedder()))


def test_ari_known_labelings():
    a = [0, 0, 1, 1, 2, 2, 2]
    assert tc.adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert tc.adjusted_rand_index(a, [5, 5, 9, 9, 1, 1, 1]) == pytest.approx(1.0)      # renamed labels
    rng = np.random.default_rng(0)
    vals = [tc.adjusted_rand_index(rng.integers(0, 5, 400).tolist(), rng.integers(0, 5, 400).tolist())
            for _ in range(20)]
    assert abs(float(np.mean(vals))) < 0.02
    assert tc.adjusted_rand_index([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    assert tc.adjusted_rand_index([0, 1, 2, 3], [0, 0, 0, 0]) == pytest.approx(0.0)      # all-split vs merged
    # sklearn reference value: ARI([0,0,1,1],[0,0,1,2]) = 0.5714285714
    assert tc.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)


def test_purity_and_inverse():
    gold = [0, 0, 1, 1]
    over = [0, 1, 2, 3]                      # over-split: purity 1, inverse purity 0.5
    assert tc.purity(over, gold) == 1.0 and tc.inverse_purity(over, gold) == 0.5
    merged = [0, 0, 0, 0]                    # merged: purity 0.5, inverse purity 1
    assert tc.purity(merged, gold) == 0.5 and tc.inverse_purity(merged, gold) == 1.0


def test_tie_rule_prefers_larger_tau():
    assert tc.choose_tau({0.5: 0.40, 0.6: 0.60, 0.7: 0.598, 0.8: 0.50}) == 0.7        # within 0.005 -> larger
    assert tc.choose_tau({0.5: 0.40, 0.6: 0.60, 0.7: 0.594}) == 0.6                   # outside tolerance
    assert tc.choose_tau({0.5: 0.7, 0.9: 0.7}) == 0.9
    assert tc.TAUS[0] == 0.5 and tc.TAUS[-1] == 0.95 and len(tc.TAUS) == 10


def test_refuses_non_dev_items(items, npz_bytes):
    bad = copy.deepcopy(items)
    bad[0]["split"] = "test"
    with pytest.raises(ValueError, match="not 'dev'"):
        tc.calibrate([("c", bad)], [(PrecomputedEmbedder(npz_bytes), {})], taus=(0.7,))
    with pytest.raises(ValueError):
        tc.check_dev({"item_id": "x"})


def _write_cell(root, name, its, split="dev"):
    d = root / "mtlbmhqa" / "dev" / "v2" / name
    d.mkdir(parents=True)
    (d / "items.jsonl.gz").write_bytes(gzip.compress(chr(10).join(json.dumps(i) for i in its).encode()))
    (d / "manifest.json").write_text(json.dumps({"cell": name, "split": split}))
    return d


def _run(tmp_path, npz, extra=(), out="o/tau.csv"):
    out = tmp_path / out
    out.parent.mkdir(exist_ok=True)
    chosen = tc.main(["--npz-root", str(npz), "--cells-dir", str(tmp_path / "cells"), "--out", str(out),
                      "--expect-model", "hash-bow-512", *extra])
    return chosen, out


def test_end_to_end_and_determinism(tmp_path, items, npz_bytes):
    _write_cell(tmp_path / "cells", "T4-h3-8k-cross", items)
    _write_cell(tmp_path / "cells", "T4-h3-8k-cross-conf", items)       # skipped by default
    (tmp_path / "emb").mkdir()
    (tmp_path / "emb" / "e.npz").write_bytes(npz_bytes)
    outs = []
    for k in range(2):
        chosen, out = _run(tmp_path, tmp_path / "emb", out=f"o{k}/tau.csv")
        outs.append(out.read_bytes())
        assert chosen in tc.TAUS
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode().splitlines()))
    assert len(rows) == 10 * 2
    pooled = [r for r in rows if r["level"] == "pooled"]
    assert sum(r["selected"] == "true" for r in pooled) == 1
    assert all(r["split"] == "dev" and float(r["n_threads_gold"]) == 4 for r in pooled)
    n = [float(r["n_threads_inferred"]) for r in sorted(pooled, key=lambda r: float(r["tau"]))]
    assert n == sorted(n)
    side = json.loads((tmp_path / "o0" / "tau.json").read_text())
    assert side["script"] == "code/opa/analysis/tau_calibration.py" and side["chosen_tau_thread"] == chosen
    assert side["n_items"] == 10 and side["offset"] == 0 and side["embedding_meta"]["model"] == "hash-bow-512"
    assert all(i["sha256"] for i in side["inputs"])
    assert any(i.get("uri", "").endswith("/opa/data/mtlbmhqa/dev/v2/T4-h3-8k-cross/items.jsonl.gz") for i in side["inputs"])


def test_npz_covers_fewer_items_than_cell(tmp_path, items):
    filler, tok = SyntheticFiller(n_books=8, words_per_book=30000), WhitespaceTokenizer()
    four = [mtlbmhqa.make_item(s, 4, 3, 8192, True, 0.0, filler, tok) for s in range(4)]
    _write_cell(tmp_path / "cells", "T4-h3-8k-cross", four)
    npz = tmp_path / "e.npz"
    npz.write_bytes(embeddings_to_bytes(compute_item_embeddings(four[:2], HashEmbedder())))
    with pytest.raises(KeyError):                                     # default n_items 10 reaches items 2, 3
        _run(tmp_path, npz)
    chosen, out = _run(tmp_path, npz, ["--n-items", "2"])
    assert chosen in tc.TAUS
    assert json.loads(out.with_suffix(".json").read_text())["n_items"] == 2
    pooled = [r for r in csv.DictReader(out.read_text().splitlines()) if r["level"] == "pooled"]
    assert all(r["n_items"] == "2" for r in pooled)
    with pytest.raises(KeyError):                                     # offset 1 + 2 items reaches item 2
        _run(tmp_path, npz, ["--n-items", "2", "--offset", "1"], out="o2/tau.csv")


def test_meta_must_match_and_item_in_one_file(tmp_path, items):
    _write_cell(tmp_path / "cells", "T4-h3-8k-cross", items)
    arrays = compute_item_embeddings(items, HashEmbedder())
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    a.write_bytes(embeddings_to_bytes(arrays))
    b.write_bytes(embeddings_to_bytes(arrays))
    with pytest.raises(ValueError, match="exactly one"):             # same items in both files
        _run(tmp_path, f"{a},{b}")
    meta = json.loads(str(arrays["meta"]))
    meta["revision"] = "other"
    b.write_bytes(embeddings_to_bytes({**arrays, "meta": np.array(json.dumps(meta))}))
    with pytest.raises(ValueError, match="disagree"):
        _run(tmp_path, f"{a},{b}", out="o2/tau.csv")
    with pytest.raises(ValueError, match="expected"):                # wrong model
        tc.main(["--npz-root", str(a), "--cells-dir", str(tmp_path / "cells"), "--out", str(tmp_path / "x.csv")])


def test_refuses_non_dev_cell_dir(tmp_path, items, npz_bytes):
    d = _write_cell(tmp_path / "cells", "T4-h3-8k-cross", items, split="test")
    with pytest.raises(ValueError, match="manifest split"):
        tc.read_cell(d)
    d2 = tmp_path / "cells" / "mtlbmhqa" / "testsplit" / "v2" / "c"
    d2.mkdir(parents=True)
    (d2 / "manifest.json").write_text(json.dumps({"cell": "c", "split": "dev"}))
    with pytest.raises(ValueError, match="'dev' path segment"):
        tc.read_cell(d2)


def test_real_model_meta_must_be_complete(items):
    arrays = compute_item_embeddings(items, HashEmbedder())
    emb = [(PrecomputedEmbedder(embeddings_to_bytes(arrays)), {})]
    assert tc.check_meta(emb, "hash-bow-512")["model"] == "hash-bow-512"          # hash files are exempt
    meta = json.loads(str(arrays["meta"]))
    meta["model"] = tc.EXPECT_MODEL                                              # looks real but lacks revision etc.
    emb = [(PrecomputedEmbedder(embeddings_to_bytes({**arrays, "meta": np.array(json.dumps(meta))})), {})]
    with pytest.raises(ValueError, match="lacks"):
        tc.check_meta(emb)
    meta.update(revision="abc", max_length=1024, dtype="float32", B=512, version="emb-v2")
    emb = [(PrecomputedEmbedder(embeddings_to_bytes({**arrays, "meta": np.array(json.dumps(meta))})), {})]
    assert tc.check_meta(emb)["revision"] == "abc"
    meta["revision"] = None
    emb = [(PrecomputedEmbedder(embeddings_to_bytes({**arrays, "meta": np.array(json.dumps(meta))})), {})]
    with pytest.raises(ValueError, match="revision"):
        tc.check_meta(emb)


def test_n_items_zero_refused_and_short_flag(tmp_path, items, npz_bytes, capsys):
    _write_cell(tmp_path / "cells", "T4-h3-8k-cross", items)             # 2 items
    npz = tmp_path / "e.npz"
    npz.write_bytes(npz_bytes)
    with pytest.raises(SystemExit):
        _run(tmp_path, npz, ["--n-items", "0"])
    _, out = _run(tmp_path, npz)                                          # requested 10, cell has 2
    assert "WARNING" in capsys.readouterr().out
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert all(r["n_items_short"] == "true" for r in rows)
    assert json.loads(out.with_suffix(".json").read_text())["n_items_short"] == ["T4-h3-8k-cross"]
    _, out = _run(tmp_path, npz, ["--n-items", "2"], out="o2/tau.csv")
    assert all(r["n_items_short"] == "false" for r in csv.DictReader(out.read_text().splitlines()))
