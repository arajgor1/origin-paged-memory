"""Offline tests of the H10 thrashing controller (protocol 3.8) with scripted models (no API, no money)."""
import gzip
import json
import re
from pathlib import Path

import pytest

from opa.tracks.track_a import run_thrash
from opa.tracks.track_a.models import FakeChat
from opa.tracks.track_a.run import LocalCtx, load_items
from opa.tracks.track_a.thrash import (CONTRAST_KEY, ThrashController, build_index, h10_contrast, question_names,
                                       run_thrash_item)

DATA = Path("C:/Idea/Technical/opa_data/thrash/dev/v1")


def items_for(r: int, n: int = 6) -> list[dict]:
    p = DATA / f"r{r}" / "items.jsonl.gz"
    if p.exists():
        return load_items(str(p))[:n]
    from opa.data import thrash
    from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
    f, t = SyntheticFiller(), WhitespaceTokenizer()
    return [thrash.make_item(s, r, f, t) for s in range(n)]


def items_uri(r: int, tmp_path, n: int = 6) -> str:
    p = DATA / f"r{r}" / "items.jsonl.gz"
    if p.exists():
        return str(p)
    out = tmp_path / f"r{r}.jsonl.gz"
    out.write_bytes(gzip.compress("\n".join(json.dumps(it) for it in items_for(r, n)).encode()))
    return str(out)


# ---------------------------------------------------------------- prompt parsing for scripted models


def parse_prompt(user: str) -> dict:
    idx_block = re.search(r"Index \(name -> origins that mention it\):\n(.*?)\n\n", user, re.S).group(1)
    index = {m.group(1): json.loads(m.group(2)) for m in re.finditer(r"^- (.+): (\[.*\])$", idx_block, re.M)}
    slots = json.loads(re.search(r"Slots \(\d+/\d+ used[^)]*\): (\[.*?\])", user).group(1))
    resident = {int(o): t for o, t in re.findall(r'<origin id="(\d+)">\n(.*?)\n</origin>', user, re.S)}
    pins = re.findall(r'^\[p\d+\] \(origin (\d+)\) "(.*)"$', user, re.M)
    return {"index": index, "slots": slots, "resident": resident, "pins": [(int(o), q) for o, q in pins],
            "final": "final step" in user}


PRICE = re.compile(r"(\d+\.\d{2}) USD")


def quote_for(name: str, text: str) -> str | None:
    """Exact substring from the supplier name to the end of the first price after it (every template puts the
    supplier before the price)."""
    i = text.find(name)
    if i < 0:
        return None
    m = PRICE.search(text, i)
    return text[i:m.end()] if m else None


def cheapest(prices: dict) -> str:
    return min(prices, key=lambda n: float(prices[n])) if prices else ""


def joint_reader(system, user):
    """Unpinned reader that needs every quote resident at once (the joint-reasoning step of Proposition 15): pages in
    the first candidate that is not resident; answers once all are resident (or at the final step, from what it sees)."""
    s = parse_prompt(user)
    prices = {}
    for n in s["index"]:
        for o in s["slots"]:
            q = quote_for(n, s["resident"][o])
            if q:
                prices[n] = PRICE.search(q).group(1)
    if len(prices) == len(s["index"]) or s["final"]:
        return json.dumps({"action": "answer", "answer": cheapest(prices)})
    want = [o for ids in s["index"].values() for o in ids if o not in s["slots"]]
    return json.dumps({"action": "page_in", "origin_id": want[0]})


def pinner(system, user):
    """Pinning reader: pins each quote it sees, pages in the next unpinned candidate, answers from the pins."""
    s = parse_prompt(user)
    pinned = {}
    for _, q in s["pins"]:
        for n in s["index"]:
            if q.startswith(n):
                pinned[n] = PRICE.search(q).group(1)
    if len(pinned) == len(s["index"]) or s["final"]:
        return json.dumps({"action": "answer", "answer": cheapest(pinned)})
    for n in s["index"]:
        if n in pinned:
            continue
        for o in s["slots"]:
            q = quote_for(n, s["resident"][o])
            if q:
                return json.dumps({"action": "pin", "origin_id": o, "quote": q})
    for n, ids in s["index"].items():
        if n not in pinned:
            return json.dumps({"action": "page_in", "origin_id": [o for o in ids if o not in s["slots"]][0]})
    return json.dumps({"action": "answer", "answer": cheapest(pinned)})


def run_one(item, policy, M, pinning, opts=None):
    events = []
    rec = run_thrash_item(item, FakeChat(policy), M, pinning, opts or {}, events.append)
    return rec, events


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("r", [2, 3, 4, 6])
def test_index_is_fair_and_correct(r):
    for it in items_for(r, 4):
        assert question_names(it["question"]) == it["gold"]["suppliers"]
        idx = build_index(it)
        assert sorted(o for ids in idx.values() for o in ids) == it["gold"]["needed_origins"]
        by_supplier = {q["supplier"]: [q["origin_id"]] for q in it["gold"]["quotes"]}
        assert idx == {n: by_supplier[n] for n in it["gold"]["suppliers"]}


def test_unpinned_repeats_rise_when_r_exceeds_M():
    M = 2
    mean_rep = {}
    for r in (2, 3, 4, 6):
        recs = [run_one(it, joint_reader, M, False)[0] for it in items_for(r, 4)]
        assert all(x["status"] in ("ok", "step_cap") for x in recs), recs[0].get("error")
        mean_rep[r] = sum(x["repeated_page_ins"] for x in recs) / len(recs)
        if r <= M:
            assert all(x["repeated_page_ins"] == 0 and x["em"] == 1.0 and x["status"] == "ok" for x in recs)
        else:
            assert all(x["used_final_step"] and x["steps"] == 4 * r + 4 for x in recs)   # forced answer at the cap
            assert all(x["needed_visible_max"] == M for x in recs)
    assert mean_rep[2] == 0
    assert mean_rep[3] >= 8 and mean_rep[4] > mean_rep[3] and mean_rep[6] > mean_rep[4]


def test_unpinned_M3_no_repeats_at_r3_but_repeats_at_r4():
    rec3, _ = run_one(items_for(3, 1)[0], joint_reader, 3, False)
    rec4, _ = run_one(items_for(4, 1)[0], joint_reader, 3, False)
    assert rec3["repeated_page_ins"] == 0 and rec3["em"] == 1.0
    assert rec4["repeated_page_ins"] > 0 and rec4["used_final_step"]


@pytest.mark.parametrize("r,M", [(3, 2), (4, 2), (6, 2), (6, 3), (2, 2)])
def test_pinned_arm_has_no_repeats(r, M):
    for it in items_for(r, 4):
        rec, ev = run_one(it, pinner, M, True)
        assert rec["status"] == "ok", rec.get("error")
        assert rec["repeated_page_ins"] == 0
        assert rec["em"] == 1.0
        assert rec["page_ins"] == r and rec["steps"] == 2 * r + 1
        assert len([p for p in rec["pins"] if not p["evicted"]]) == r
        assert rec["needed_visible_max"] == r                    # all r quotes visible at once (as pins)
        for p in rec["pins"]:                                    # exact-pin rule: spans index the origin text
            assert it["origins"][p["origin_id"]]["text"][p["char_span"][0]:p["char_span"][1]] == p["quote"]


def test_lru_eviction_logged_and_touch_is_not_a_page_in():
    it = items_for(4, 1)[0]
    a, b, c = it["gold"]["needed_origins"][:3]
    script = iter([{"action": "page_in", "origin_id": a}, {"action": "page_in", "origin_id": b},
                   {"action": "page_in", "origin_id": a},             # touch: a becomes most recently used
                   {"action": "page_in", "origin_id": c},             # evicts b (LRU), not a
                   {"action": "page_in", "origin_id": b},             # repeat load of b, evicts a
                   {"action": "answer", "answer": "x"}])
    rec, ev = run_one(it, lambda s, u: json.dumps(next(script)), 2, False)
    evicts = [e for e in ev if e["action"] == "evict"]
    assert [(e["origin_id"], e["reason"]) for e in evicts] == [(b, "lru"), (a, "lru")]
    assert [e["origin_id"] for e in ev if e["action"] == "touch"] == [a]
    assert rec["paged_in"] == [a, b, c, b] and rec["touches"] == [a]
    assert rec["repeated_page_ins"] == 1 and rec["repeated_by_origin"] == {str(b): 1}
    assert [e["origin_id"] for e in ev if e["action"] == "repeat_page_in"] == [b]
    assert rec["evicted"] == [{"origin_id": b, "reason": "lru"}, {"origin_id": a, "reason": "lru"}]
    assert rec["trace"][-2]["slots"] == [c, b]
    assert all(e["item_id"] == it["item_id"] for e in ev)
    steps = [e for e in ev if e["action"] == "llm_step"]
    assert len(steps) == 6 and "prompt" in steps[0] and steps[0]["reply"].startswith("{")


def test_step_cap_enforced():
    it = items_for(2, 1)[0]
    n_calls = []

    def stubborn(system, user):
        n_calls.append(1)
        return json.dumps({"action": "page_in", "origin_id": len(n_calls) % 3})

    rec, ev = run_one(it, stubborn, 2, False)
    assert rec["max_steps"] == 4 * 2 + 4 and rec["steps"] == 12 and len(n_calls) == 12
    assert rec["status"] == "step_cap" and rec["prediction"] == "" and rec["em"] == 0.0
    assert any(e["action"] == "step_cap" for e in ev)
    n_calls.clear()
    rec, _ = run_one(it, stubborn, 2, False, {"max_steps": 5})
    assert rec["steps"] == 5 and len(n_calls) == 5
    final = []
    run_one(it, lambda s, u: final.append("final step" in u) or json.dumps({"action": "page_in", "origin_id": 0}), 2,
            False, {"max_steps": 3})
    assert final == [False, False, True]                        # the last prompt asks for an answer


def test_pin_rules():
    it = items_for(3, 1)[0]
    q0 = it["gold"]["quotes"][0]
    o, other = q0["origin_id"], it["gold"]["quotes"][1]["origin_id"]
    script = iter([{"action": "pin", "origin_id": o, "quote": q0["text"]},          # not resident
                   {"action": "page_in", "origin_id": o},
                   {"action": "pin", "origin_id": o, "quote": q0["text"] + " invented"},  # not exact
                   {"action": "pin", "origin_id": o, "quote": it["origins"][o]["text"][:1000]},  # too long
                   {"action": "pin", "origin_id": o, "quote": q0["text"]},          # ok
                   {"action": "pin", "origin_id": o, "quote": q0["text"]},          # duplicate
                   {"action": "page_in", "origin_id": other},
                   "not json at all", "still not json",
                   {"action": "fly"},
                   {"action": "answer", "answer": q0["supplier"]}])

    def pol(s, u):
        x = next(script)
        return x if isinstance(x, str) else json.dumps(x)

    rec, ev = run_one(it, pol, 2, True)
    assert rec["pins_rejected"] == 4 and len(rec["pins"]) == 1 and rec["pins"][0]["quote"] == q0["text"]
    reasons = [e.get("reason") for e in ev if e["action"] == "pin_rejected"]
    assert reasons == ["origin not resident", None, "quote too long", "duplicate"]   # None: Fold's exact-pin log
    assert rec["invalid_actions"] == 2 and rec["steps"] == 10 and rec["n_llm_calls"] == 11
    # pinning disabled in the unpinned arm
    rec2, ev2 = run_one(it, lambda s, u: json.dumps({"action": "pin", "origin_id": o, "quote": q0["text"]}), 2, False,
                        {"max_steps": 2})
    assert rec2["invalid_actions"] == 2 and not rec2["pins"]
    assert "pin" not in ThrashController(it, FakeChat(pol), 2, False, {}, lambda e: None).system_prompt.split("one of:")[1]


def test_pin_budget_drops_oldest():
    it = items_for(4, 1)[0]
    rec, ev = run_one(it, pinner, 2, True, {"P_pin": 30})                  # room for ~1 quote of ~70 chars
    assert any(e["action"] == "evict_pin" for e in ev)
    assert sum(not p["evicted"] for p in rec["pins"]) < 4


def test_run_thrash_entrypoint_and_h10_contrast(tmp_path):
    uris = [items_uri(r, tmp_path) for r in (2, 4)]
    out = {}
    for pinning, pol in ((False, joint_reader), (True, pinner)):
        ctx = LocalCtx(str(tmp_path / f"pin{int(pinning)}"), run_id=f"run-{int(pinning)}")
        res = run_thrash({"items_uri": uris, "n_items": 6, "M": 2, "pinning": pinning,
                          "model": {"kind": "fake", "chat": FakeChat(pol)}, "workers": 4}, ctx)
        m = res["metrics"]
        assert m["n"] == 12 and m["errors"] == 0
        assert set(m["by_cell"]) == {f"r2-M2-{'pin' if pinning else 'nopin'}", f"r4-M2-{'pin' if pinning else 'nopin'}"}
        recs = [json.loads(x) for x in gzip.decompress((ctx.out / "records.jsonl.gz").read_bytes()).decode().splitlines()]
        assert len(recs) == 12 and all(r["run_id"] == ctx.run_id for r in recs)
        assert json.loads((ctx.out / "metrics.json").read_text())["M"] == 2
        assert any(e["action"] == "thrash_config" for e in ctx.events)
        assert m["tokens_in_per_item"] > 0 and m["calls_per_item"] > 0
        out[pinning] = (m, recs)
    nopin, pin = out[False][0]["by_cell"], out[True][0]["by_cell"]
    assert nopin["r4-M2-nopin"]["repeated_page_ins_mean"] > 10 and nopin["r2-M2-nopin"]["repeated_page_ins_mean"] == 0
    assert pin["r4-M2-pin"]["repeated_page_ins_mean"] == 0 and pin["r4-M2-pin"]["em"] == 1.0
    c = h10_contrast(out[False][1], out[True][1], split="dev", n_boot=2000)
    assert c["key"] == CONTRAST_KEY and c["n"] == 6 and c["value"] > 10 and c["ci_lo"] > 0 and c["p_raw"] < 0.05
    assert "Wilcoxon" in c["test"]
    assert h10_contrast(out[False][1], out[True][1], split="dev", cells="all", n_boot=500)["n"] == 12


def test_item_error_is_recorded_not_raised():
    it = items_for(2, 1)[0]

    def boom(s, u):
        raise RuntimeError("model down")

    rec, _ = run_one(it, boom, 2, False)
    assert rec["status"] == "error" and "model down" in rec["error"] and rec["em"] == 0.0


def test_failure_record_carries_versions_and_h10_still_runs():
    """A controller that raises produces a record with prompt_version / tuning_iteration; h10_contrast then imputes it
    at the worst value instead of raising "mixes prompt versions" (also when an old failure record has no version)."""
    from opa.tracks.track_a import thrash
    its = items_for(4, 3)

    def boom(s, u):
        raise RuntimeError("model down")

    bad = run_thrash_item(its[0], FakeChat(boom), 2, True, {}, lambda ev: None)
    assert bad["status"] == "error" and bad["prompt_version"] == thrash.PROMPT_VERSION and bad["tuning_iteration"] == 1
    unp = [{"item_id": it["item_id"], "r": 4, "M": 2, "status": "ok", "repeated_page_ins": 3 + k, "prompt_version": thrash.PROMPT_VERSION}
           for k, it in enumerate(its)]
    pin = [dict(r, repeated_page_ins=1) for r in unp]
    pin[0] = {**bad, "r": 4, "M": 2}
    row = h10_contrast(unp, pin, split="dev", n_boot=200)
    assert row["n"] == 3 and row["n_error_pinned"] == 1 and row["imputed_worst"] == {"r4-M2": 5}
    assert row["pinned_mean"] == pytest.approx((5 + 1 + 1) / 3)
    old = {k: v for k, v in pin[0].items() if k not in ("prompt_version", "tuning_iteration")}     # old failure record, no version
    pin[0] = old
    assert h10_contrast(unp, pin, split="dev", n_boot=200)["prompt_version"] == thrash.PROMPT_VERSION


def test_answer_contract_asks_for_the_name_only_in_both_arms():
    """Pilot finding: the v1 prompt said "<short answer>" and Qwen3-4B answered with sentences, so EM was ~0 for correct picks."""
    from opa.tracks.track_a import thrash
    it = items_for(3, 1)[0]
    for pinning in (False, True):
        c = ThrashController(it, FakeChat(lambda s, u: "{}"), 2, pinning, {}, lambda ev: None)
        assert "<the supplier name only, copied from the question, no other words>" in c.system_prompt
        assert "short answer" not in c.system_prompt
    assert thrash.PROMPT_VERSION == "thrash-v2"


def test_arms_share_the_system_prompt_except_the_pin_lines():
    import difflib
    from opa.tracks.track_a import thrash
    for M, S in ((2, 16), (3, 20)):
        pin = thrash.build_system_prompt(M, S, True, 300, 512)
        nopin = thrash.build_system_prompt(M, S, False, 300, 512)
        pin_block = thrash.PIN_RULES.format(C=300, P=512) + thrash.ACTION_PIN
        # removing exactly the two pin-specific pieces from the pin prompt gives the nopin prompt, byte for byte
        stripped = pin.replace(thrash.PIN_RULES.format(C=300, P=512), "", 1).replace(thrash.ACTION_PIN, "", 1)
        assert stripped == nopin
        # and a line diff shows nothing but the pin lines (rules paragraph + pin action), no other line changed or removed
        d = [x for x in difflib.ndiff(nopin.splitlines(), pin.splitlines()) if x[:1] in "+-"]
        assert all(x[0] == "+" for x in d) and "\n".join(x[2:] for x in d) == pin_block.rstrip("\n")
    assert "as soon as you see it, before you page in the next origin" in thrash.PIN_RULES
    assert "answer from the pinned quotes" in thrash.PIN_RULES
    assert "do not page in" not in thrash.PIN_RULES.lower()                  # the re-paging instruction is shared, not pin-only
    shared = "Do not page in an origin again if the text you need from it is still visible to you (in a slot or, if you have them, in your pinned quotes)."
    assert shared in pin and shared in nopin


def test_user_prompts_differ_only_by_the_pinned_quotes_section():
    it = items_for(3, 1)[0]
    pin = ThrashController(it, FakeChat(lambda s, u: "{}"), 2, True, {}, lambda ev: None)
    nopin = ThrashController(it, FakeChat(lambda s, u: "{}"), 2, False, {}, lambda ev: None)
    a, b = pin.user_prompt(1), nopin.user_prompt(1)
    assert a.replace("Pinned quotes:\n(none)\n\n", "", 1) == b


def test_tuning_iteration_is_recorded(tmp_path):
    from opa.tracks.track_a import thrash
    assert thrash.TUNING_ITERATION == {"pin": 1, "nopin": 1}
    for pinning, want in ((True, 1), (False, 1)):
        recs = [run_thrash_item(it, FakeChat(lambda s, u: json.dumps({"action": "answer", "answer": "x"})), 2, pinning, {}, lambda ev: None)
                for it in items_for(3, 2)]
        assert all(r["tuning_iteration"] == want for r in recs)
        m = thrash.summarize_thrash(recs, 2, pinning, "fake")
        assert m["tuning_iteration"] == want and m["prompt_version"] == "thrash-v2"


def _h10_rows(version=None, tuning=None, n=3, pinned_repeats=0, unpinned_repeats=2):
    unp = [{"item_id": f"i{k}", "r": 4, "M": 2, "status": "ok", "repeated_page_ins": unpinned_repeats + k % 2,
            **({"prompt_version": version} if version else {}), **({"tuning_iteration": tuning} if tuning is not None else {})} for k in range(n)]
    pin = [dict(r, repeated_page_ins=pinned_repeats) for r in unp]
    return unp, pin


def test_h10_contrast_refuses_mixed_prompt_versions_and_reports_version_and_tuning():
    unp, pin = _h10_rows("thrash-v2", 1)
    row = h10_contrast(unp, pin, split="dev", n_boot=200)
    assert row["prompt_version"] == "thrash-v2" and row["tuning_iteration_unpinned"] == 1 and row["tuning_iteration_pinned"] == 1
    old_unp, old_pin = _h10_rows(None, None)
    row_old = h10_contrast(old_unp, old_pin, split="dev", n_boot=200)                # records without a version count as v1
    assert row_old["prompt_version"] == "thrash-v1" and row_old["tuning_iteration_pinned"] is None
    with pytest.raises(ValueError, match="mixes prompt versions"):
        h10_contrast(unp, old_pin, split="dev", n_boot=200)                          # v2 unpinned vs unversioned (v1) pinned
    unp2, pin2 = _h10_rows("thrash-v2", 1)
    pin2[1]["prompt_version"] = "thrash-v3"
    with pytest.raises(ValueError, match="mixes prompt versions"):
        h10_contrast(unp2, pin2, split="dev", n_boot=200)
