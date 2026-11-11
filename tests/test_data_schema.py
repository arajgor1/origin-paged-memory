"""P0.5 item schema: validate_item catches errors, rechunk round-trips gold, io, names, filler, build."""
import copy
import json

import numpy as np
import pytest

from opa.data import GENERATOR_VERSION, lbmhqa, mtlbmhqa, thrash
from opa.data.build import build, parse_cell
from opa.data.filler import SyntheticFiller, WhitespaceTokenizer, insert_sentences, sentence_starts
from opa.data.io import read_items, write_items
from opa.data.names import NameGen, hamming1
from opa.data.schema import rechunk, validate_item


@pytest.fixture(scope="module")
def filler():
    return SyntheticFiller(n_books=12, words_per_book=30000)


@pytest.fixture(scope="module")
def tok():
    return WhitespaceTokenizer()


@pytest.fixture(scope="module")
def lb_item(filler, tok):
    return lbmhqa.make_item(1, 3, 32768, "late", 0.3, filler, tok)


@pytest.fixture(scope="module")
def mt_item(filler, tok):
    return mtlbmhqa.make_item(1, 8, 3, 32768, True, 0.0, filler, tok)


# ---------------------------------------------------------------- rechunk
@pytest.mark.parametrize("B", [1024, 2048])
def test_rechunk_roundtrips_gold(lb_item, mt_item, filler, tok, B):
    th = thrash.make_item(2, 4, filler, tok)
    for it in (lb_item, mt_item, th):
        k = B // 512
        rc = rechunk(it, B)
        validate_item(rc)
        assert len(rc["origins"]) == len(it["origins"]) // k
        assert rc["params"]["block_tokens"] == B
        assert rc["gold"]["thread_of"] == [o["thread_id"] for o in rc["origins"]]
        for o in rc["origins"]:
            assert o["merged_from"] == list(range(o["origin_id"] * k, o["origin_id"] * k + k))
        if it["dataset"] == "thrash":
            for q0, q1 in zip(it["gold"]["quotes"], rc["gold"]["quotes"]):
                assert q1["origin_id"] == q0["origin_id"] // k
                assert rc["origins"][q1["origin_id"]]["text"][slice(*q1["char_span"])] == q0["text"]
            continue
        assert rc["gold"]["h_late"] == it["gold"]["h_late"]          # facts lie in distinct 4-block groups
        for f0, f1 in zip(it["gold"]["facts"], rc["gold"]["facts"]):
            assert f1["origin_id"] == f0["origin_id"] // k
            assert rc["origins"][f1["origin_id"]]["text"][slice(*f1["char_span"])] == f0["text"]
        for c0, c1 in zip(it["gold"]["confusers"], rc["gold"]["confusers"]):
            assert rc["origins"][c1["origin_id"]]["text"][slice(*c1["char_span"])] == c0["text"]
        for d0, d1 in zip(it["gold"]["distractors"], rc["gold"]["distractors"]):
            assert rc["origins"][d1["origin_id"]]["text"][slice(*d1["char_span"])] == d0["text"]
        assert sum(o["n_tokens"] for o in rc["origins"]) == sum(o["n_tokens"] for o in it["origins"])
    # merging twice equals merging once
    assert rechunk(rechunk(lb_item, 1024), 2048)["gold"] == rechunk(lb_item, 2048)["gold"]


def test_rechunk_identity_and_errors(lb_item, mt_item, tok):
    assert rechunk(lb_item, 512) == lb_item and rechunk(lb_item, 512) is not lb_item
    with pytest.raises(ValueError):
        rechunk(lb_item, 768)
    bad = copy.deepcopy(mt_item)
    bad["origins"][1]["thread_id"] = (bad["origins"][0]["thread_id"] + 1) % 8
    with pytest.raises(ValueError):
        rechunk(bad, 1024)
    rc = rechunk(lb_item, 1024, tokenizer=tok)
    assert all(o["n_tokens"] == tok.count(o["text"]) for o in rc["origins"])


# ---------------------------------------------------------------- validate_item catches errors
def _expect_invalid(item, mutate):
    bad = copy.deepcopy(item)
    mutate(bad)
    with pytest.raises(ValueError):
        validate_item(bad)


def test_validate_catches_errors(lb_item, mt_item):
    validate_item(lb_item)
    validate_item(mt_item)

    def shift_span(it):
        it["gold"]["facts"][1]["char_span"][0] += 1
    _expect_invalid(lb_item, shift_span)
    _expect_invalid(lb_item, lambda it: it["gold"].__setitem__("h_late", 0))
    _expect_invalid(lb_item, lambda it: it["gold"]["facts"][2].__setitem__("late_bound", False))
    _expect_invalid(lb_item, lambda it: it["origins"][3].__setitem__("origin_id", 4))
    _expect_invalid(lb_item, lambda it: it["gold"]["fact_origins"].reverse())
    _expect_invalid(lb_item, lambda it: it.__setitem__("answer", "1 °C"))

    def leak_name(it):                          # an entity name appearing in filler text is caught
        o = it["origins"][0]
        o["text"] = o["text"] + " " + it["gold"]["entities"][1]
    _expect_invalid(lb_item, leak_name)
    _expect_invalid(mt_item, lambda it: it.__setitem__("question_thread", (it["question_thread"] + 1) % 8))
    _expect_invalid(mt_item, lambda it: it["gold"].__setitem__("cross_thread_hop", None))
    _expect_invalid(mt_item, lambda it: it["gold"]["confusers"][0].__setitem__("value", it["answer"]))
    _expect_invalid(mt_item, lambda it: it["gold"]["confusers"][0].__setitem__("entity", it["gold"]["entities"][-1]))
    _expect_invalid(mt_item, lambda it: it["gold"]["thread_of"].__setitem__(0, 99))


# ---------------------------------------------------------------- pieces
def test_names_twins_and_vocab_check():
    rng = np.random.default_rng(0)
    first = NameGen(np.random.default_rng(0)).stem().lower()
    ng = NameGen(np.random.default_rng(0), vocab={first})
    assert ng.stem().lower() != first and ng.n_rejected >= 1
    ng = NameGen(np.random.default_rng(0), text_lower=f"once upon a time {first}s lived here")
    assert ng.stem().lower() != first
    ng = NameGen(rng)
    names = [ng.entity() for _ in range(30)]
    stems = [n.stem.lower() for n in names]
    from opa.data.names import levenshtein
    assert all(levenshtein(a, b) >= 3 for i, a in enumerate(stems) for b in stems[i + 1:])
    for n in names[:10]:
        t = ng.twin(n)
        assert hamming1(t.full, n.full) and t.suffix == n.suffix
        assert n.stem.upper() in n.alias_options()


def test_insert_sentences_spans():
    text = "One two three. Four five six. Seven eight."
    starts = sentence_starts(text)
    assert starts == [15, 30]
    new, spans = insert_sentences(text, [(15, "Alpha beta."), (0, "Gamma."), (len(text), "Delta."), (15, "Eps.")])
    assert [new[s:e] for s, e in spans] == ["Alpha beta.", "Gamma.", "Delta.", "Eps."]
    assert sentence_starts("Dr. Gould came. Then J. H. Smith left.") == [16]


def test_filler_blocks(filler, tok):
    rng = np.random.default_rng(0)
    blocks = filler.take_blocks(rng, 70, tok, 512, books=[3, 4, 5])
    assert len(blocks) == 70 and {b["book_id"] for b in blocks} >= {"synthetic-003", "synthetic-004"}
    sizes = [tok.count(b["text"]) for b in blocks]
    assert 470 <= np.mean(sizes) <= 530 and min(sizes) > 400 and max(sizes) < 600


def test_io_roundtrip(tmp_path, lb_item, mt_item):
    p = tmp_path / "items.jsonl.gz"
    a = write_items([lb_item, mt_item], p)
    assert read_items(p) == [json.loads(json.dumps(lb_item)), json.loads(json.dumps(mt_item))]
    b = write_items([lb_item, mt_item], tmp_path / "again.jsonl.gz")
    assert a["sha256"] == b["sha256"] and a["n_items"] == 2


def test_cell_names_roundtrip():
    assert parse_cell("lbmhqa", "late-h3-32k-a0") == {"placement": "late", "hops": 3, "length_tokens": 32768,
                                                     "alpha": 0.0}
    assert parse_cell("lbmhqa", "mixed-h2-8k-a03")["alpha"] == 0.3
    assert parse_cell("lbmhqa", "forward-h3-16k-a0")["length_tokens"] == 16384
    assert lbmhqa.cell_name("forward", 2, 32768, 0.3) == "forward-h2-32k-a03"
    assert parse_cell("mtlbmhqa", "T8-h3-32k-nocross") == {"n_threads": 8, "hops": 3, "length_tokens": 32768,
                                                           "cross_thread": False, "alpha": 0.0, "decoys": False}
    assert parse_cell("mtlbmhqa", "T4-h3-32k-cross-conf")["decoys"] is True
    assert mtlbmhqa.cell_name(16, 2, 32768, True) == "T16-h2-32k-cross"
    assert mtlbmhqa.cell_name(4, 3, 32768, True, 0.0, True) == "T4-h3-32k-cross-conf"
    assert parse_cell("thrash", "r6") == {"r": 6}
    with pytest.raises(ValueError):
        parse_cell("lbmhqa", "backward-h3-32k-a0")


def test_build_synthetic(tmp_path):
    out = build("thrash", "dev", "r3", 3, out_root=tmp_path, synthetic=True, quiet=True)
    m = out["manifest"]
    assert m["seeds"] == [0, 1, 2] and m["counts"]["n_items"] == 3 and m["validated"]
    items = read_items(tmp_path / "thrash" / "dev" / GENERATOR_VERSION / "r3" / "items.jsonl.gz")
    assert [it["item_id"] for it in items] == ["thrash-dev-r3-000000", "thrash-dev-r3-000001", "thrash-dev-r3-000002"]
    assert (tmp_path / "thrash" / "dev" / GENERATOR_VERSION / "r3" / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        build("thrash", "dev", "r3", 3, out_root=tmp_path, synthetic=True, quiet=True)
    d = build("drift", "dev", "s50", 2, out_root=tmp_path, synthetic=True, quiet=True)
    assert d["manifest"]["counts"]["updates_per_session"] == [3]


# ---------------------------------------------------------------- v3: config-driven lengths, presets, token tolerance
def test_config_defaults_drive_generators():
    from opa.data import check_length, data_config, default_block_tokens
    cfg = data_config()
    assert cfg["context_lengths"] == [8192, 16384, 32768] and cfg["block_tokens"] == 512
    assert cfg["tokenizer"] == "Qwen/Qwen3-4B-Instruct-2507" and default_block_tokens() == 512
    check_length(32768)
    with pytest.raises(ValueError):
        check_length(131072)


def test_build_refuses_unconfigured_length_unless_overridden(tmp_path):
    with pytest.raises(ValueError, match="configured context lengths"):
        build("lbmhqa", "dev", "late-h3-64k-a0", 1, out_root=tmp_path, synthetic=True, quiet=True)
    out = build("lbmhqa", "dev", "late-h3-8k-a0", 2, out_root=tmp_path, synthetic=True, quiet=True)
    m = out["manifest"]
    assert m["params"]["length_tokens"] == 8192 and m["counts"]["max_abs_length_error_pct"] < 2
    assert m["block_tokens"] == 512 and all(len(it["origins"]) == 16 for it in out["items"])
    assert all(it["params"]["block_tokens"] == 512 for it in out["items"])


def test_v3_dev_preset_cells():
    from opa.data.build import v3_dev_cells
    cells = v3_dev_cells()
    names = {(d, c) for d, c, _ in cells}
    assert len(names) == len(cells) and all(n >= 60 for d, c, n in cells if d in ("lbmhqa", "mtlbmhqa"))
    for pl in ("forward", "late", "mixed"):
        for h in (3, 2):
            assert ("lbmhqa", f"{pl}-h{h}-32k-a0") in names
    assert ("lbmhqa", "late-h3-8k-a0") in names and ("lbmhqa", "late-h3-16k-a0") in names
    for t in (4, 8, 16):
        for x in ("cross", "nocross"):
            assert ("mtlbmhqa", f"T{t}-h3-32k-{x}") in names and ("mtlbmhqa", f"T{t}-h3-32k-{x}-conf") in names
    assert all("128k" not in c for _, c, _ in cells)
    for d, c, _ in cells:                                      # every preset cell parses and is canonical
        parse_cell(d, c)
    assert v3_dev_cells([4096, 8192])[0][1] == "forward-h3-8k-a0"       # lengths follow the argument


def test_token_length_check_rejects_off_target_items(lb_item):
    from opa.data.build import check_token_lengths, item_tokens
    assert check_token_lengths("lbmhqa", [lb_item]) < 0.02
    bad = copy.deepcopy(lb_item)
    bad["params"]["length_tokens"] = int(item_tokens(bad) * 1.05)
    with pytest.raises(ValueError, match="off"):
        check_token_lengths("lbmhqa", [bad])
