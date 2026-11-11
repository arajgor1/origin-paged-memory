"""H12 thread-confusion task (MT-LB-MHQA with look-alike decoys): gold fields, kinds, determinism, scoring helper.
No network: SyntheticFiller + whitespace tokenizer."""
import copy
import json

import pytest

from opa.data import confusion, mtlbmhqa
from opa.data.build import build, parse_cell
from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
from opa.data.io import read_items
from opa.data.names import hamming1
from opa.data.schema import rechunk, validate_item

L = 32768
LEGAL = ("Ltd.", "GmbH", "Corp.", "Group", "Inc.", "AG", "plc", "LLC", "S.A.")


@pytest.fixture(scope="module")
def filler():
    return SyntheticFiller(n_books=40, words_per_book=30000)


@pytest.fixture(scope="module")
def tok():
    return WhitespaceTokenizer()


def conf(filler, tok, seed=0, T=8, cross=True):
    return mtlbmhqa.make_item(seed, T, 3, L, cross, 0.0, filler, tok, decoys=True)


def test_cell_name_and_params(filler, tok):
    assert parse_cell("mtlbmhqa", "T4-h3-32k-cross-conf")["decoys"] is True
    it = conf(filler, tok, 1, 4, True)
    assert it["cell"] == "T4-h3-32k-cross-conf" and it["item_id"].startswith("mtlbmhqa-dev-T4-h3-32k-cross-conf-")
    assert it["params"]["decoys"] is True and it["params"]["cross_thread"] is True
    assert "thread_cue" not in it["params"]
    assert mtlbmhqa.make_item(1, 4, 3, L, True, 0.0, filler, tok)["params"]["decoys"] is False


@pytest.mark.parametrize("T", [4, 8, 16])
@pytest.mark.parametrize("cross", [True, False])
def test_decoy_gold_fields(filler, tok, T, cross):
    for seed in range(6):
        it = conf(filler, tok, seed, T, cross)
        validate_item(it)
        g, qt = it["gold"], it["question_thread"]
        final = g["entities"][-1]
        decs = g["confusers"]
        assert len(decs) == T - 1
        assert sorted(d["thread_id"] for d in decs) == [t for t in range(T) if t != qt]    # none in the question thread
        vals = [d["value"] for d in decs]
        assert len(set(vals)) == T - 1 and it["answer"] not in vals
        for d in decs:                                           # gold per decoy: thread, origin, value, span, text
            for k in ("thread_id", "origin_id", "value", "entity", "kind", "hop", "char_span", "text", "value_aliases"):
                assert k in d
            assert it["origins"][d["origin_id"]]["thread_id"] == d["thread_id"] != qt
            assert it["origins"][d["origin_id"]]["text"][slice(*d["char_span"])] == d["text"]
            assert d["entity"] in d["text"] and d["value"] in d["text"] and d["hop"] == 2
            assert confusion.decoy_entity_ok(d["kind"], d["entity"], final)
            assert d["kind"] in ("twin", "suffix")
            assert d["entity"] not in confusion.final_aliases(final)
        kinds = {d["kind"] for d in decs}
        if confusion.suffix_allowed(final):
            assert kinds == {"twin", "suffix"}
        else:
            assert kinds == {"twin"}
        if cross:
            assert g["thread_of"][g["fact_origins"][-1]] != qt                 # the true final fact is cross-thread
        assert "Use only the facts" not in it["question"]


def test_suffix_decoys_never_legal_forms_and_only_on_suffixed_finals(filler, tok):
    n_suffix = n_plain = 0
    for seed in range(40):
        it = conf(filler, tok, seed, 8, True)
        final = it["gold"]["entities"][-1]
        if not confusion.suffix_allowed(final):
            n_plain += 1
            assert all(d["kind"] == "twin" for d in it["gold"]["confusers"])
            continue
        n_suffix += 1
        assert not final.endswith(LEGAL)
        for d in it["gold"]["confusers"]:
            if d["kind"] == "suffix":
                assert d["entity"] != final and d["entity"].split()[0] == final.split()[0]
                assert not d["entity"].endswith(LEGAL)
    assert n_suffix > 0 and n_plain > 0
    assert not any(s.strip() in LEGAL for s in confusion.PRODUCT_SUFFIXES)


def test_decoy_kinds_surface_forms(filler, tok):
    seen = {}
    for seed in range(30):
        it = conf(filler, tok, seed, 8, False)
        final = it["gold"]["entities"][-1]
        for d in it["gold"]["confusers"]:
            seen[d["kind"]] = seen.get(d["kind"], 0) + 1
            if d["kind"] == "twin":
                assert hamming1(d["entity"], final)
    assert set(seen) == {"twin", "suffix"}


def test_decoy_same_template_different_value(filler, tok):
    it = conf(filler, tok, 3, 8, True)
    final_fact = it["gold"]["facts"][-1]["text"]
    ans = it["answer"]
    surf = it["gold"]["facts"][-1]["subject_surface"]
    for d in it["gold"]["confusers"]:
        # putting the true entity and value into the decoy sentence reproduces the true final fact's wording
        assert d["text"].replace(d["entity"], surf).replace(d["value"], ans) == final_fact


def test_classify_prediction(filler, tok):
    it = conf(filler, tok, 2, 4, True)
    d = it["gold"]["confusers"][1]
    assert confusion.classify_prediction(it, it["answer"]) == {"outcome": "correct", "decoy_thread": None}
    assert confusion.classify_prediction(it, d["value"]) == {"outcome": "wrong_thread", "decoy_thread": d["thread_id"]}
    num = d["value"].split()[0]
    assert confusion.classify_prediction(it, num)["outcome"] == "wrong_thread"           # unit-less alias
    assert confusion.classify_prediction(it, "1 parsec")["outcome"] == "other"
    # the F8 check in opa.scoring reads the same gold field
    from opa.scoring import thread_confused
    assert thread_confused(it, {"prediction": d["value"]}) is True
    assert thread_confused(it, {"prediction": num}) is True
    assert thread_confused(it, {"prediction": it["answer"]}) is False


def test_non_conf_confusers_carry_value_aliases(filler, tok):
    it = mtlbmhqa.make_item(1, 4, 3, L, True, 0.0, filler, tok)
    validate_item(it)
    assert it["gold"]["confusers"] and all(c["value"] in c["value_aliases"] for c in it["gold"]["confusers"])
    c = it["gold"]["confusers"][0]
    from opa.scoring import thread_confused
    assert thread_confused(it, {"prediction": c["value"].split()[0]}) is True


def test_validate_catches_bad_decoys(filler, tok):
    it = next(x for x in (conf(filler, tok, s, 4, True) for s in range(40))
              if confusion.suffix_allowed(x["gold"]["entities"][-1]))

    def bad(mutate):
        x = copy.deepcopy(it)
        mutate(x)
        with pytest.raises(ValueError):
            validate_item(x)
    final = it["gold"]["entities"][-1]
    stem = final.split()[0]
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("kind", "samename"))
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("entity", final))
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("entity", stem))                  # alias of the final entity
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("entity", stem.upper()))
    ti = next(i for i, d in enumerate(it["gold"]["confusers"]) if d["kind"] == "twin")
    bad(lambda x: x["gold"]["confusers"][ti].__setitem__("kind", "suffix"))               # a twin labelled suffix
    bad(lambda x: x["gold"]["confusers"][0].pop("kind"))
    bad(lambda x: x["gold"]["confusers"].pop())                                      # one decoy per other thread
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("value", x["answer"]))
    bad(lambda x: x["gold"]["confusers"][0].__setitem__("thread_id", x["question_thread"]))
    assert confusion.decoy_entity_ok("suffix", stem + " Works", stem + " Pumps")
    assert not confusion.decoy_entity_ok("suffix", stem + " GmbH", stem + " Pumps")        # legal form on one side
    assert not confusion.decoy_entity_ok("suffix", stem + " Pumps", stem + " Ltd.")
    assert not confusion.decoy_entity_ok("suffix", stem + " Pumps", stem)                  # final without a suffix


def test_rechunk_keeps_decoys(filler, tok):
    it = conf(filler, tok, 5, 8, False)
    for B in (1024, 2048):
        rc = rechunk(it, B)
        validate_item(rc)
        for d0, d1 in zip(it["gold"]["confusers"], rc["gold"]["confusers"]):
            assert d1["origin_id"] == d0["origin_id"] // (B // 512) and d1["kind"] == d0["kind"]
            assert rc["origins"][d1["origin_id"]]["text"][slice(*d1["char_span"])] == d0["text"]


def test_rechunk_composition_titles_and_books(filler, tok):
    it = conf(filler, tok, 5, 4, False)
    a = rechunk(rechunk(it, 1024), 2048)["origins"]
    b = rechunk(it, 2048)["origins"]
    assert [(o["text"], o["doc_title"], o["book_id"]) for o in a] == [(o["text"], o["doc_title"], o["book_id"]) for o in b]


def test_determinism_and_independence(filler, tok):
    a = json.dumps(conf(filler, tok, 7, 8, True), sort_keys=True)
    assert a == json.dumps(conf(filler, tok, 7, 8, True), sort_keys=True)
    assert conf(filler, tok, 7, 8, True)["gold"]["entities"] != conf(filler, tok, 7, 8, False)["gold"]["entities"]
    plain = mtlbmhqa.make_item(7, 8, 3, L, True, 0.0, filler, tok)
    assert plain["gold"]["entities"] != conf(filler, tok, 7, 8, True)["gold"]["entities"]     # different cell stream


def test_build_conf_cell_synthetic_rebuilds_identically(tmp_path):
    cell = "T4-h3-32k-cross-conf"
    a = build("mtlbmhqa", "dev", cell, 5, out_root=tmp_path / "a", synthetic=True, quiet=True)
    b = build("mtlbmhqa", "dev", cell, 5, out_root=tmp_path / "b", synthetic=True, quiet=True)
    assert a["manifest"]["items_sha256"] == b["manifest"]["items_sha256"]
    assert a["manifest"]["counts"]["decoy_kinds"] and a["manifest"]["counts"]["confusers_per_item"] == [3]
    items = read_items(tmp_path / "a" / "mtlbmhqa" / "dev" / a["manifest"]["version"] / cell / "items.jsonl.gz")
    assert len(items) == 5 and all(it["gold"]["confusers"] for it in items)
