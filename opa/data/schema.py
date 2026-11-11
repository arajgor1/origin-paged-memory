"""validate_item (all four datasets) and rechunk (B in {512, 1024, 2048}) for the item contract in SCHEMA.md."""
from __future__ import annotations

import copy

from opa.data import BASE_BLOCK_TOKENS, confusion
from opa.data.names import hamming1

QA_DATASETS = ("lbmhqa", "mtlbmhqa")
ORIGIN_DATASETS = ("lbmhqa", "mtlbmhqa", "thrash")
INSERT_KINDS = {"fact", "distractor", "confuser", "alias", "quote"}
ALLOWED_B = (512, 1024, 2048)


def _fail(item: dict, msg: str):
    raise ValueError(f"{item.get('item_id', '<no item_id>')}: {msg}")


def _require(item: dict, obj: dict, keys, where: str):
    for k in keys:
        if k not in obj:
            _fail(item, f"missing field {where}{k}")


def late_bound_flags(fact_origins: list[int]) -> list[bool]:
    """Definition 6: entity 0 relevant from origin 0; entity j >= 1 becomes relevant where hop j-1's fact reveals it."""
    relevant_at = [0] + [int(o) for o in fact_origins[:-1]]
    return [int(fact_origins[j]) < relevant_at[j] for j in range(len(fact_origins))]


# ---------------------------------------------------------------- common
def _validate_header(item: dict):
    _require(item, item, ["item_id", "dataset", "split", "seed", "generator_version", "params"], "")
    if item["split"] not in ("dev", "test"):
        _fail(item, f"bad split {item['split']!r}")
    seed = item["seed"]
    if not isinstance(seed, int):
        _fail(item, "seed must be an int")
    if item["split"] == "dev" and not 0 <= seed <= 999:
        _fail(item, f"dev seed {seed} outside 0..999")
    if item["split"] == "test" and seed < 100000:
        _fail(item, f"test seed {seed} below 100000")
    if not item["item_id"].startswith(f"{item['dataset']}-{item['split']}-") or not item["item_id"].endswith(f"{seed:06d}"):
        _fail(item, "item_id does not match dataset/split/seed")


def _validate_origins(item: dict) -> list[dict]:
    origins = item.get("origins")
    if not isinstance(origins, list) or not origins:
        _fail(item, "origins must be a non-empty list")
    for i, o in enumerate(origins):
        _require(item, o, ["origin_id", "text", "thread_id", "n_tokens", "inserted"], f"origins[{i}].")
        if o["origin_id"] != i:
            _fail(item, f"origins[{i}].origin_id is {o['origin_id']}, must equal its index")
        if not isinstance(o["text"], str) or not o["text"]:
            _fail(item, f"origin {i} has empty text")
        if not isinstance(o["n_tokens"], int) or o["n_tokens"] <= 0:
            _fail(item, f"origin {i} has bad n_tokens")
        last = -1
        for ins in sorted(o["inserted"], key=lambda d: d["char_span"][0]):
            if ins.get("kind") not in INSERT_KINDS:
                _fail(item, f"origin {i}: unknown inserted kind {ins.get('kind')!r}")
            s, e = ins["char_span"]
            if not 0 <= s < e <= len(o["text"]):
                _fail(item, f"origin {i}: inserted span {ins['char_span']} out of range")
            if s < last:
                _fail(item, f"origin {i}: overlapping inserted spans")
            last = e
    return origins


def _span_text(item: dict, origins: list[dict], oid: int, span, expect: str, what: str):
    if not isinstance(oid, int) or not 0 <= oid < len(origins):
        _fail(item, f"{what}: origin_id {oid} out of range")
    s, e = span
    t = origins[oid]["text"]
    if not 0 <= s < e <= len(t):
        _fail(item, f"{what}: char_span {span} out of range for origin {oid}")
    if t[s:e] != expect:
        _fail(item, f"{what}: text at origin {oid} span {span} is {t[s:e]!r}, expected {expect!r}")


def _inserted_set(origins: list[dict], kind: str) -> list[tuple]:
    return sorted((o["origin_id"], tuple(ins["char_span"])) for o in origins for ins in o["inserted"]
                  if ins["kind"] == kind)


def _names_only_in_inserts(item: dict, origins: list[dict], names: list[str]):
    """Invented names must not occur in the filler: every occurrence of a stem lies inside an inserted span."""
    stems = sorted({n.split()[0].lower() for n in names if n})
    for o in origins:
        low = o["text"].lower()
        spans = [ins["char_span"] for ins in o["inserted"]]
        for st in stems:
            p = low.find(st)
            while p >= 0:
                if not any(s <= p and p + len(st) <= e for s, e in spans):
                    _fail(item, f"name {st!r} occurs in filler text of origin {o['origin_id']} at {p}")
                p = low.find(st, p + 1)


# ---------------------------------------------------------------- LB-MHQA / MT-LB-MHQA
def _validate_qa(item: dict):
    ds = item["dataset"]
    _require(item, item, ["cell", "origins", "question", "question_thread", "answer", "answer_aliases", "gold"], "")
    p = item["params"]
    _require(item, p, ["hops", "length_tokens", "block_tokens"], "params.")
    hops, B = p["hops"], p["block_tokens"]
    origins = _validate_origins(item)
    if len(origins) != p["length_tokens"] // B:
        _fail(item, f"{len(origins)} origins, expected length_tokens/block_tokens = {p['length_tokens'] // B}")
    g = item["gold"]
    _require(item, g, ["entities", "aliases", "facts", "fact_origins", "h_late", "thread_of", "cross_thread_hop",
                       "confusers", "distractors"], "gold.")
    ents, facts = g["entities"], g["facts"]
    if len(ents) != hops or len(set(ents)) != hops:
        _fail(item, "gold.entities must hold `hops` distinct names")
    if len(facts) != hops:
        _fail(item, "gold.facts must hold one fact per hop")
    if item["answer"] not in item["answer_aliases"]:
        _fail(item, "answer missing from answer_aliases")
    if ents[0] not in item["question"]:
        _fail(item, "question does not name entity 0")
    thread_of = g["thread_of"]
    if thread_of != [o["thread_id"] for o in origins]:
        _fail(item, "gold.thread_of disagrees with origins[].thread_id")
    qt = item["question_thread"]
    for j, f in enumerate(facts):
        _require(item, f, ["hop", "origin_id", "char_span", "text", "subject", "reveals", "late_bound", "cross_thread"],
                 f"gold.facts[{j}].")
        if f["hop"] != j:
            _fail(item, f"facts[{j}].hop is {f['hop']}")
        _span_text(item, origins, f["origin_id"], f["char_span"], f["text"], f"fact hop {j}")
        if f["subject"] != ents[j]:
            _fail(item, f"fact hop {j}: subject {f['subject']!r} is not entity {j} {ents[j]!r}")
        surf = f.get("subject_surface", f["subject"])
        if surf != f["subject"] and surf not in g["aliases"].get(f["subject"], []):
            _fail(item, f"fact hop {j}: alias {surf!r} not listed in gold.aliases")
        if surf not in f["text"]:
            _fail(item, f"fact hop {j}: subject mention {surf!r} not in fact text")
        if j < hops - 1:
            if f["reveals"] != ents[j + 1] or ents[j + 1] not in f["text"]:
                _fail(item, f"fact hop {j}: must reveal entity {j + 1}")
        else:
            if f["reveals"] != item["answer"] or item["answer"] not in f["text"]:
                _fail(item, "last fact must state the answer")
        if "thread_id" in f and f["thread_id"] != thread_of[f["origin_id"]]:
            _fail(item, f"fact hop {j}: thread_id disagrees with thread_of")
        if f["cross_thread"] != (thread_of[f["origin_id"]] != qt):
            _fail(item, f"fact hop {j}: cross_thread flag inconsistent with thread labels")
    fo = [f["origin_id"] for f in facts]
    if g["fact_origins"] != fo:
        _fail(item, "gold.fact_origins disagrees with facts[].origin_id")
    lb = late_bound_flags(fo)
    if [f["late_bound"] for f in facts] != lb:
        _fail(item, f"late_bound flags {[f['late_bound'] for f in facts]} disagree with Definition 6 {lb}")
    if g["h_late"] != sum(lb):
        _fail(item, f"h_late {g['h_late']} != {sum(lb)} (Definition 6)")
    if _inserted_set(origins, "fact") != sorted((f["origin_id"], tuple(f["char_span"])) for f in facts):
        _fail(item, "inserted 'fact' spans disagree with gold.facts")

    if ds == "lbmhqa":
        if qt != 0 or any(t != 0 for t in thread_of):
            _fail(item, "LB-MHQA has one thread (0)")
        if g["cross_thread_hop"] is not None or g["confusers"]:
            _fail(item, "LB-MHQA has no cross-thread hop and no confusers")
        placement = p.get("placement")
        if placement == "forward" and g["h_late"] != 0:
            _fail(item, "forward placement must have h_late 0")
        if placement == "late" and g["h_late"] != hops - 1:
            _fail(item, "late placement must have h_late = hops - 1")
        if placement == "mixed" and hops >= 3 and not 0 < g["h_late"] < hops - 1:
            _fail(item, "mixed placement must have 0 < h_late < hops - 1")
    else:
        nt = p.get("n_threads")
        if qt != thread_of[-1]:
            _fail(item, "question_thread must be the thread of the last origin")
        if sorted(set(thread_of)) != list(range(nt)):
            _fail(item, f"thread ids must be 0..{nt - 1}")
        rb = p.get("run_blocks")
        if rb:
            unit = max(1, rb * BASE_BLOCK_TOKENS // B)
            for a in range(0, len(thread_of), unit):
                if len(set(thread_of[a:a + unit])) != 1:
                    _fail(item, f"thread runs not aligned to {unit} origins at origin {a}")
        cross = [f["hop"] for f in facts if f["cross_thread"]]
        if len(cross) > 1:
            _fail(item, "at most one cross-thread hop")
        want = cross[0] if cross else None
        if g["cross_thread_hop"] != want:
            _fail(item, f"cross_thread_hop {g['cross_thread_hop']} disagrees with fact threads ({want})")
        if p.get("cross_thread") and want != hops - 1:
            _fail(item, "cross_thread item must have its last hop in another thread")
        if not p.get("cross_thread") and want is not None:
            _fail(item, "no-cross item has a cross-thread hop")
        in_thread = [f["origin_id"] for f in facts if not f["cross_thread"]]
        if any(a < b for a, b in zip(in_thread, in_thread[1:])):
            _fail(item, "question-thread facts must be in reverse order (late-bound)")
        if g["h_late"] != hops - 1:
            _fail(item, "MT items must have every hop after the first late-bound")
        final = ents[-1]
        decoys = bool(p.get("decoys"))
        if decoys:
            if len(g["confusers"]) != nt - 1:
                _fail(item, "decoy items need one decoy in every other thread")
        seen_threads, vals = set(), set()
        for c in g["confusers"]:
            _require(item, c, ["entity", "value", "origin_id", "thread_id", "char_span"], "gold.confusers[].")
            if decoys:
                _require(item, c, ["kind", "hop", "value_aliases", "text"], "gold.confusers[].")
                if c["kind"] not in confusion.allowed_kinds(final):
                    _fail(item, f"decoy kind {c['kind']!r} not allowed in this cell")
                if not confusion.decoy_entity_ok(c["kind"], c["entity"], final):
                    _fail(item, f"decoy {c['entity']!r} is not a {c['kind']} of {final!r}")
                if c["hop"] != hops - 1 or c["value"] not in c["value_aliases"]:
                    _fail(item, "decoy hop/value_aliases inconsistent")
                if c["value_aliases"] and set(c["value_aliases"]) & set(item["answer_aliases"]):
                    _fail(item, "a decoy value alias coincides with an accepted answer")
            elif not hamming1(c["entity"], final):
                _fail(item, f"confuser {c['entity']!r} does not differ from {final!r} by exactly one character")
            if c["value"] == item["answer"]:
                _fail(item, "confuser value equals the answer")
            if c["value"] in vals:
                _fail(item, "confuser values must be distinct")
            vals.add(c["value"])
            if c["thread_id"] != thread_of[c["origin_id"]] or c["thread_id"] == qt:
                _fail(item, "confuser must sit in a thread other than the question's, with a correct label")
            if c["thread_id"] in seen_threads:
                _fail(item, "more than one confuser in a thread")
            seen_threads.add(c["thread_id"])
            if "text" in c:
                _span_text(item, origins, c["origin_id"], c["char_span"], c["text"], "confuser")
                if c["entity"] not in c["text"] or c["value"] not in c["text"]:
                    _fail(item, "confuser text must name the twin and its value")
        if p.get("confuser_rate", 1.0) >= 1.0 and len(g["confusers"]) != nt - 1:
            _fail(item, "confuser_rate 1.0 needs one confuser in every other thread")
    if _inserted_set(origins, "confuser") != sorted((c["origin_id"], tuple(c["char_span"])) for c in g["confusers"]):
        _fail(item, "inserted 'confuser' spans disagree with gold.confusers")
    for d in g["distractors"]:
        _require(item, d, ["origin_id", "char_span", "subject"], "gold.distractors[].")
        if d["subject"] in ents:
            _fail(item, "a distractor's subject is a chain entity")
        if "text" in d:
            _span_text(item, origins, d["origin_id"], d["char_span"], d["text"], "distractor")
    if _inserted_set(origins, "distractor") != sorted((d["origin_id"], tuple(d["char_span"])) for d in g["distractors"]):
        _fail(item, "inserted 'distractor' spans disagree with gold.distractors")
    names = list(ents) + [c["entity"] for c in g["confusers"]] + [d["subject"] for d in g["distractors"]]
    _names_only_in_inserts(item, origins, names)


# ---------------------------------------------------------------- thrash
def _validate_thrash(item: dict):
    _require(item, item, ["cell", "origins", "question", "answer", "answer_aliases", "gold"], "")
    p = item["params"]
    _require(item, p, ["r", "block_tokens", "n_origins"], "params.")
    origins = _validate_origins(item)
    if len(origins) != p["n_origins"]:
        _fail(item, "len(origins) != params.n_origins")
    g = item["gold"]
    _require(item, g, ["needed_origins", "quotes"], "gold.")
    q = g["quotes"]
    if len(q) != p["r"]:
        _fail(item, "need r quotes")
    for x in q:
        _require(item, x, ["supplier", "price", "origin_id", "char_span"], "gold.quotes[].")
        if "text" in x:
            _span_text(item, origins, x["origin_id"], x["char_span"], x["text"], "quote")
            if x["supplier"] not in x["text"] or x["price"] not in x["text"]:
                _fail(item, "quote text must name supplier and price")
        if x["supplier"] not in item["question"]:
            _fail(item, f"supplier {x['supplier']!r} not listed in the question")
    if g["needed_origins"] != sorted({x["origin_id"] for x in q}):
        _fail(item, "needed_origins must be the sorted distinct quote origins")
    if p["block_tokens"] == BASE_BLOCK_TOKENS and len(g["needed_origins"]) != p["r"]:
        _fail(item, "each needed quote must sit in a different origin")
    prices = [float(x["price"].split()[0]) for x in q]
    if len(set(prices)) != len(prices):
        _fail(item, "prices must be distinct")
    best = q[prices.index(min(prices))]["supplier"]
    if item["answer"] != best or best not in item["answer_aliases"]:
        _fail(item, "answer must be the supplier with the lowest price")
    if _inserted_set(origins, "quote") != sorted((x["origin_id"], tuple(x["char_span"])) for x in q):
        _fail(item, "inserted 'quote' spans disagree with gold.quotes")
    _names_only_in_inserts(item, origins, [x["supplier"] for x in q])


# ---------------------------------------------------------------- drift
def _validate_drift(item: dict):
    from opa.data.drift import THREADS, current_values, value_text
    _require(item, item, ["turns", "constraints", "probes"], "")
    p = item["params"]
    _require(item, p, ["n_turns", "n_constraints", "n_updates", "threads"], "params.")
    turns = item["turns"]
    if [t["turn"] for t in turns] != list(range(1, p["n_turns"] + 1)):
        _fail(item, "turns must be numbered 1..n_turns")
    for t in turns:
        _require(item, t, ["turn", "thread", "speaker", "text"], "turns[].")
        if t["thread"] not in THREADS:
            _fail(item, f"turn {t['turn']}: unknown thread {t['thread']!r}")
    if turns[0]["thread"] != "design":
        _fail(item, "turn 1 is in the design thread")
    cons = item["constraints"]
    if len(cons) != p["n_constraints"] or len({c["key"] for c in cons}) != len(cons):
        _fail(item, "need n_constraints distinct constraints")
    kinds = {c.get("kind") for c in cons}
    for k in ("temperature", "tolerance", "part", "supplier", "date"):
        if k not in kinds:
            _fail(item, f"no {k} constraint")
    n_upd = 0
    for c in cons:
        _require(item, c, ["key", "label", "unit", "initial", "updates", "near_misses"], "constraints[].")
        ini = c["initial"]
        if ini["turn"] != 1 or ini["thread"] != "design":
            _fail(item, f"{c['key']}: initial value must be stated in turn 1 (design)")
        if value_text(ini["value"], c["unit"]) not in turns[0]["text"]:
            _fail(item, f"{c['key']}: initial value not in turn 1 text")
        seen = [ini["value"]]
        for u in c["updates"]:
            n_upd += 1
            if u["thread"] == ini["thread"]:
                _fail(item, f"{c['key']}: update in the same thread as the original statement")
            t = turns[u["turn"] - 1]
            if t["thread"] != u["thread"]:
                _fail(item, f"{c['key']}: update thread label disagrees with the turn")
            if value_text(u["value"], c["unit"]) not in t["text"] or c["label"] not in t["text"]:
                _fail(item, f"{c['key']}: update value/label not in turn {u['turn']} text")
            if u["value"] == seen[-1]:
                _fail(item, f"{c['key']}: update does not change the value")
            seen.append(u["value"])
        for n in c["near_misses"]:
            t = turns[n["turn"] - 1]
            if n["value"] in seen:
                _fail(item, f"{c['key']}: near-miss value equals a real value")
            if n["text"] not in t["text"] or value_text(n["value"], c["unit"]) not in n["text"]:
                _fail(item, f"{c['key']}: near-miss text not in turn {n['turn']}")
            if t["thread"] != n["thread"]:
                _fail(item, f"{c['key']}: near-miss thread label disagrees with the turn")
    if n_upd != p["n_updates"]:
        _fail(item, f"{n_upd} updates, expected {p['n_updates']}")
    if [pr["turn"] for pr in item["probes"]] != [10, 25, 50]:
        _fail(item, "probes at turns 10, 25 and 50")
    for pr in item["probes"]:
        if pr["gold"] != current_values(cons, pr["turn"]):
            _fail(item, f"probe {pr['turn']}: gold disagrees with the constraint history")


# ---------------------------------------------------------------- entry points
def validate_item(item: dict) -> None:
    """Raise ValueError naming the first problem found."""
    if not isinstance(item, dict):
        raise ValueError("item must be a dict")
    _validate_header(item)
    ds = item["dataset"]
    if ds in QA_DATASETS:
        _validate_qa(item)
    elif ds == "thrash":
        _validate_thrash(item)
    elif ds == "drift":
        _validate_drift(item)
    else:
        _fail(item, f"unknown dataset {ds!r}")


def rechunk(item: dict, B: int, tokenizer=None) -> dict:
    """Copy of `item` with every k = B / block_tokens consecutive origins merged into one (B in {512, 1024, 2048}).
    Origin ids, char spans, thread_of, fact origins, confusers, distractors and quotes are remapped; late_bound and
    h_late are recomputed with Definition 6 on the merged origins (two hops in one merged origin are not late-bound
    with respect to each other). n_tokens is the sum of the parts unless a tokenizer is given to recount."""
    if item["dataset"] not in ORIGIN_DATASETS:
        raise ValueError(f"rechunk does not apply to {item['dataset']}")
    if B not in ALLOWED_B:
        raise ValueError(f"B must be one of {ALLOWED_B}")
    out = copy.deepcopy(item)
    b0 = item["params"]["block_tokens"]
    if B == b0:
        return out
    if B % b0:
        raise ValueError(f"cannot rechunk B={b0} into B={B}")
    k = B // b0
    old = item["origins"]
    if len(old) % k:
        raise ValueError(f"{len(old)} origins do not divide into groups of {k}")
    new_origins, base = [], {}
    for g in range(len(old) // k):
        members = old[g * k:(g + 1) * k]
        threads = {m["thread_id"] for m in members}
        if len(threads) != 1:
            raise ValueError(f"origins {g * k}..{g * k + k - 1} span several threads; thread runs are not aligned")
        text, ins = "", []
        for i, m in enumerate(members):
            if i:
                # compare the last title of the previous member with the first of this one, so merging an already
                # merged item (titles "A / B") separates exactly as merging the 512-token blocks would
                prev_t = members[i - 1].get("doc_title", "").split(" / ")[-1]
                text += " " if m.get("doc_title", "").split(" / ")[0] == prev_t else "\n\n"
            base[m["origin_id"]] = (g, len(text))
            ins += [{**d, "char_span": [d["char_span"][0] + len(text), d["char_span"][1] + len(text)]}
                    for d in m["inserted"]]
            text += m["text"]
        # flatten "A / B" parts so composing rechunks gives the same title and book lists as one rechunk
        titles = list(dict.fromkeys(t for m in members for t in m.get("doc_title", "").split(" / ")))
        books = list(dict.fromkeys(b for m in members for b in m.get("book_id", "").split(" / ")))
        o = {"origin_id": g, "text": text, "thread_id": members[0]["thread_id"], "doc_title": " / ".join(titles),
             "book_id": " / ".join(books), "n_tokens": sum(m["n_tokens"] for m in members), "inserted": ins,
             "merged_from": [m["origin_id"] for m in members]}
        new_origins.append(o)
    if tokenizer is not None:
        from opa.data.filler import as_tokenizer
        counts = as_tokenizer(tokenizer).count_many([o["text"] for o in new_origins])
        for o, c in zip(new_origins, counts):
            o["n_tokens"] = int(c)
    out["origins"] = new_origins

    def remap(rec: dict):
        g, off = base[rec["origin_id"]]
        rec["origin_id"] = g
        rec["char_span"] = [rec["char_span"][0] + off, rec["char_span"][1] + off]

    gold = out["gold"]
    gold["thread_of"] = [o["thread_id"] for o in new_origins]
    if item["dataset"] in QA_DATASETS:
        for f in gold["facts"]:
            remap(f)
        for c in gold["confusers"]:
            remap(c)
        for d in gold["distractors"]:
            remap(d)
        gold["fact_origins"] = [f["origin_id"] for f in gold["facts"]]
        lb = late_bound_flags(gold["fact_origins"])
        for f, x in zip(gold["facts"], lb):
            f["late_bound"] = bool(x)
        gold["h_late"] = int(sum(lb))
    else:
        for q in gold["quotes"]:
            remap(q)
        gold["needed_origins"] = sorted({q["origin_id"] for q in gold["quotes"]})
        out["params"]["n_origins"] = len(new_origins)
    out["params"]["block_tokens"] = B
    out["params"]["rechunked_from"] = b0
    return out

