"""Offline end-to-end tests of the Track A harness with a scripted model (no API, no money)."""
import gzip
import json
import re

import numpy as np
import pytest

from opa.tracks.track_a.index import HashEmbedder, Session
from opa.tracks.track_a.models import FakeChat, parse_json
from opa.tracks.track_a.run import LocalCtx, run_system
from opa.tracks.track_a.systems import BASELINES, SYSTEMS, Ctx, Fold, question_entities, session_cfg
from opa.tracks.track_a.text import find_exact, lexical_entities, normalize_key

NAMES = ["Varrel Pumps", "Ostrand Ltd.", "Kelvane", "Dorrin Valves"]
FILLER = ["The rain had stopped by the time Margaret reached the old mill, and the river ran high and brown. ",
          "He spoke of the harvest and of the long winter that had come before it, though no one listened. ",
          "There was a letter on the table, sealed with grey wax, which nobody in the house would open. ",
          "Across the valley the bells of Saint Aldric rang for evening, slow and uneven in the wind. "]


def make_item(late=True, n=24, threads=1, seed=0):
    rng = np.random.default_rng(seed)
    texts = [FILLER[i % 4] + FILLER[(i + 1) % 4] for i in range(n)]
    facts = ["Varrel Pumps sources seals from Ostrand Ltd.", "Ostrand Ltd. makes its seals from Kelvane.",
             "Kelvane seals degrade above 83 °C in ester coolant."]
    slots = sorted(rng.choice(np.arange(2, n - 2), size=3, replace=False).tolist())
    if late:
        slots = slots[::-1]
    gold_facts = []
    for j, (f, s) in enumerate(zip(facts, slots)):
        start = len(texts[s]) + 1
        texts[s] = texts[s] + " " + f
        gold_facts.append({"hop": j, "origin_id": s, "char_span": [start, start + len(f)], "text": f,
                           "subject": NAMES[j], "reveals": NAMES[j + 1] if j < 2 else None})
    texts[1] += " Dorrin Valves sources seals from Hallum Works."          # distractor
    relevant_at = {NAMES[0]: 0}
    for j in range(2):
        relevant_at.setdefault(NAMES[j + 1], slots[j])
    for j, f in enumerate(gold_facts):
        f["late_bound"] = slots[j] < relevant_at[NAMES[j]]
    thread_of = [(i * threads) // n for i in range(n)]
    return {"item_id": f"t-{seed}-{int(late)}", "dataset": "lbmhqa", "split": "dev", "seed": seed, "generator_version": "test",
            "origins": [{"origin_id": i, "text": t, "thread_id": thread_of[i], "n_tokens": len(t) // 4} for i, t in enumerate(texts)],
            "question": "Above what temperature do the seals in Varrel Pumps's supply chain degrade?",
            "question_thread": thread_of[-1], "answer": "83 °C", "answer_aliases": ["83 °C", "83 C"],
            "gold": {"entities": NAMES[:3], "aliases": {}, "facts": gold_facts, "fact_origins": slots,
                     "h_late": sum(f["late_bound"] for f in gold_facts), "thread_of": thread_of, "cross_thread_hop": None,
                     "confusers": [], "distractors": []}}


def policy(system, user):
    """A careful scripted reader: pins sentences naming a relevant entity; answers with the first temperature."""
    if "reading module" in system:
        rel = re.search(r"Relevant entities: (.*)", user).group(1)
        rel_keys = [normalize_key(x) for x in rel.split(";") if x.strip() and "(" not in x]
        pins, found = [], False
        for oid, body in re.findall(r'<origin id="(\d+)">\n(.*?)\n</origin>', user, re.S):
            for sent in re.split(r"(?<!Ltd\.)(?<=[.!?])\s+", body):
                ks = [k for k in rel_keys if k and k in normalize_key(sent)]
                if ks:
                    others = [nm for nm in NAMES + ["Hallum Works"] if normalize_key(nm) not in ks and nm.split()[0] in sent]
                    key = "quote" if "Copy each quote" in system else "note"
                    pins.append({"origin_id": int(oid), key: sent.strip(), "entity": ks[0], "reveals": others})
                    found |= "°C" in sent
        return json.dumps({"pins" if "quote" in system and "Copy each quote" in system else "notes": pins, "answer_found": found})
    if system.startswith("Answer the question"):
        m = re.search(r"(\d+ °C)", user)
        return json.dumps({"answer": m.group(1) if m else "unknown", "cited_pin_ids": ["p1"]})
    if "List the named entities" in system:
        ids = re.findall(r'<origin id="(\d+)">', user)
        return json.dumps({"entities": {i: [nm for nm in NAMES if nm.split()[0] in user] for i in ids}})
    if "gist" in system:
        return json.dumps({"gist": user[:300]})
    if "Choose up to" in system:
        return json.dumps({"pages": [0, 1, 2, 3, 4]})
    if "one worker in a chain" in system:
        keep = [s for s in re.split(r"(?<=[.!?])\s+", user) if any(nm.split()[0] in s for nm in NAMES)]
        return json.dumps({"notes": " ".join(keep)[-1500:]})
    if "reasoning over retrieved" in system:
        m = re.search(r"(\d+ °C)", user)
        return json.dumps({"thought": "keep looking", "answer": m.group(1) if m else None})
    if "Summarize the passages" in system:
        return json.dumps({"summary": user[:400]})
    return "{}"


@pytest.fixture
def item():
    return make_item()


def test_text_utils():
    assert find_exact("sources  seals\nfrom", "X sources seals from Y") == (2, 20)
    assert find_exact("not there", "abc") is None
    ents = lexical_entities("Varrel Pumps sources seals from Ostrand Ltd. under QX-2291 at 83 °C on 2026-11-06.")
    assert {"varrel pumps", "ostrand ltd", "qx-2291", "83 c", "2026-11-06"} <= ents
    assert question_entities("Above what temperature do the seals in Varrel Pumps's supply chain degrade?") == ["varrel pumps"]
    assert parse_json('sure ```json\n{"a": 1}\n```') == {"a": 1} and parse_json("x {\"b\": [1]} y") == {"b": [1]}


def test_session_structures(item):
    sess = Session(item, {"skeleton": "hyb", "graph": True, "use_gold_threads": True}, HashEmbedder(),
                   skeleton_llm={str(i): [] for i in range(24)})
    assert sess.lookup("Ostrand Ltd.") == [item["gold"]["fact_origins"][0], item["gold"]["fact_origins"][1]] or \
        set(sess.lookup("Ostrand Ltd.")) == {item["gold"]["fact_origins"][0], item["gold"]["fact_origins"][1]}
    assert set(sess.lookup("OSTRAND")) == set(sess.lookup("Ostrand Ltd."))          # alias key
    f0, f1 = item["gold"]["fact_origins"][:2]
    assert f1 in sess.neighbors(f0, kinds=("ent",))                              # shared entity edge
    cands, n_scores = sess.descend(item["question"], HashEmbedder().embed([item["question"]])[0], beam=2)
    assert cands and n_scores > 0
    rows = sess.db.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
    assert rows == 24
    for arm in ("A0", "A1", "A2", "A3"):
        assert sess.distance(23, 23, arm) == 0.0
    assert sess.distance(23, 0, "A3") == 23.0 and abs(sess.distance(23, 0, "A2") - 23.0) < 1e-6


def _run(system, item, opts=None, tmp=None):
    emb = HashEmbedder()
    chat = FakeChat(policy)
    cfg = session_cfg(system, opts or {})
    sess = Session(item, cfg, emb, skeleton_llm={str(o["origin_id"]): [nm for nm in NAMES if nm.split()[0] in o["text"]]
                                               for o in item["origins"]})
    events = []
    c = Ctx(item=item, sess=sess, chat=chat, opts=dict(opts or {}, _q_emb=emb.embed_query([item["question"]])[0], _embedder=emb),
            log_fn=events.append)
    from opa.tracks.track_a.systems import run_item
    return run_item(system, c), events


@pytest.mark.parametrize("system", list(SYSTEMS) + list(BASELINES))
def test_every_system_runs(system, item):
    rec, events = _run(system, item, {"lambda": 1.0})
    assert rec["status"] in ("ok", "budget", "does_not_fit"), rec.get("error")
    assert isinstance(rec["prediction"], str)
    assert events, "every system logs events"


def test_fold_recovers_late_bound_chain_with_skeleton(item):
    assert item["gold"]["h_late"] == 2
    rec, events = _run("S8", item)
    assert rec["prediction"] == "83 °C"
    assert rec["faults"], "late-bound hops need page faults"
    assert all(p["quote"] in item["origins"][p["origin_id"]]["text"] for p in rec["pins"])     # exact pins only


def test_fold_without_skeleton_fails_on_late_binding(item):
    rec, _ = _run("S6", item)
    assert rec["prediction"] != "83 °C" and not rec["faults"]


def test_forward_placement_needs_no_faults():
    rec, _ = _run("S8", make_item(late=False))
    assert rec["prediction"] == "83 °C"


def test_exact_pin_rule_rejects_paraphrase(item):
    def liar(system, user):
        if "reading module" in system:
            ids = re.findall(r'<origin id="(\d+)">', user)
            return json.dumps({"pins": [{"origin_id": int(ids[0]), "quote": "Varrel Pumps buys from someone", "entity": "x"}],
                               "answer_found": False})
        return policy(system, user)
    sess = Session(item, session_cfg("S8", {}), HashEmbedder(), skeleton_llm={})
    emb = HashEmbedder()
    events = []
    c = Ctx(item=item, sess=sess, chat=FakeChat(liar), opts={"_q_emb": emb.embed_query([item["question"]])[0], "_embedder": emb},
            log_fn=events.append)
    rec = Fold(c, "S8").run()
    assert rec["pins"] == [] and any(e["action"] == "pin_rejected" for e in events)


def test_slots_and_pin_budget(item):
    rec, events = _run("S8", item, {"M": 2, "read_window": 1, "P_pin": 12})
    assert any(e["action"] == "evict" for e in events)
    assert any(e["action"] == "evict_pin" for e in events)


def test_h5_arms_share_candidates_and_differ_only_in_distance(item):
    recs = {s: _run(s, item, {"lambda": 2.0})[0] for s in ("S12", "S14", "S15", "S16")}
    cands = {s: tuple(r["descend_candidates"]) for s, r in recs.items()}
    assert len(set(cands.values())) == 1, cands
    assert {r["arm"] for r in recs.values()} == {"A0", "A1", "A2", "A3"}


def test_run_system_entrypoint_offline(tmp_path, item):
    items_path = tmp_path / "items.jsonl.gz"
    with gzip.open(items_path, "wt") as f:
        for it in (item, make_item(seed=1), make_item(late=False, seed=2)):
            f.write(json.dumps(it) + "\n")
    sk_dir = tmp_path / "sk"
    sk_dir.mkdir()
    for it in (item, make_item(seed=1), make_item(late=False, seed=2)):
        (sk_dir / f"{it['item_id']}.json").write_text(json.dumps({"prompt_version": "sllm-v2", "entities": {str(o["origin_id"]): [] for o in it["origins"]}}))
    ctx = LocalCtx(str(tmp_path / "out"))
    out = run_system({"items_uri": str(items_path), "system": "S11", "embedder": "hash", "workers": 2,
                      "skeleton_llm_uri": str(sk_dir), "model": {"kind": "fake", "chat": FakeChat(policy)}}, ctx)
    assert out["metrics"]["n"] == 3
    recs = [json.loads(l) for l in gzip.decompress((tmp_path / "out" / "records.jsonl.gz").read_bytes()).decode().splitlines()]
    assert {r["item_id"] for r in recs} == {"t-0-1", "t-1-1", "t-2-0"}
    assert all("edges_from_resident" in r for r in recs)
    assert ctx.events and all("run_id" in e for e in ctx.events)
