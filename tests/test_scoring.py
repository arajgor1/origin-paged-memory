"""P0.6: opa.scoring against hand-built items and records (opa/data/SCHEMA.md formats)."""
import copy
import math

import pytest

from opa import scoring as sc


# ------------------------------------------------------------------ normalization, EM, F1
@pytest.mark.parametrize("form", ["83 °C", "83 C", "83 degC", "83 degrees C", "83°C", "83.0 °C", "83 ℃",
                                  "83 deg. C", "83 degrees Celsius", "83 c", "83.00°C"])
def test_temperature_forms_normalize_equal(form):
    assert sc.normalize_answer(form) == "83 degc"


@pytest.mark.parametrize("a,b", [("83.0", "83"), ("1.50", "1.5"), ("1,000", "1000"), ("The Ostrand Ltd.", "ostrand ltd"),
                                 ("  Kelvane   seals ", "kelvane seals"), ("QX−2291", "QX-2291"),
                                 ("100 °F", "100 degrees Fahrenheit")])
def test_normalize_equal(a, b):
    assert sc.normalize_answer(a) == sc.normalize_answer(b)


@pytest.mark.parametrize("a,b", [("83.5", "835"), ("-5 °C", "5 °C"), ("007", "7"), ("80 °C", "80 °F"), ("0.05", "5")])
def test_normalize_keeps_distinctions(a, b):
    assert sc.normalize_answer(a) != sc.normalize_answer(b)


def test_normalize_examples():
    assert sc.normalize_answer("The answer is 83.50!") == "answer is 83.5"
    assert sc.normalize_answer("an apple, a pear") == "apple pear"
    assert sc.normalize_answer(None) == ""


def test_exact_match_and_f1():
    answers = ["83 °C", "83 C", "83 degrees C", "83"]
    assert sc.exact_match("83 degC", answers) == 1.0
    assert sc.exact_match("83", answers) == 1.0
    assert sc.exact_match("84 °C", answers) == 0.0
    assert sc.exact_match("", answers) == 0.0
    assert sc.exact_match("83.0", "83") == 1.0
    # SQuAD token F1 by hand: pred {cat, sat}, gold {cat, sat, on, mat}: P = 1, R = 0.5, F1 = 2/3
    assert sc.f1_score("the cat sat", ["cat sat on the mat"]) == pytest.approx(2 / 3)
    assert sc.f1_score("dog", ["cat sat", "dog"]) == 1.0
    assert sc.f1_score("x", []) == 0.0


# ------------------------------------------------------------------ fixtures
def make_item():
    facts = [
        {"hop": 0, "origin_id": 10, "char_span": [40, 98], "text": "Varrel Pumps sources seals from Ostrand Ltd.",
         "subject": "Varrel Pumps", "reveals": "Ostrand Ltd.", "late_bound": False, "cross_thread": False},
        {"hop": 1, "origin_id": 5, "char_span": [10, 60], "text": "OSTRAND makes its seals from Kelvane.",
         "subject": "Ostrand Ltd.", "reveals": "Kelvane", "late_bound": True, "cross_thread": False},
        {"hop": 2, "origin_id": 2, "char_span": [0, 50], "text": "Kelvane seals degrade above 83 °C in ester coolant.",
         "subject": "Kelvane", "reveals": None, "late_bound": True, "cross_thread": True},
    ]
    return {
        "item_id": "it-1", "dataset": "mtlbmhqa", "answer": "83 °C", "answer_aliases": ["83 °C", "83 C", "83"],
        "gold": {"entities": ["Varrel Pumps", "Ostrand Ltd.", "Kelvane"], "aliases": {"Ostrand Ltd.": ["OSTRAND"]},
                 "facts": facts, "fact_origins": [10, 5, 2], "h_late": 2,
                 "confusers": [{"entity": "Kelvanf", "value": "71 °C", "origin_id": 30, "thread_id": 3,
                                "char_span": [0, 40]}]},
    }


def pin(pid, origin, span, quote="", evicted=False):
    return {"pin_id": pid, "origin_id": origin, "char_span": span, "quote": quote, "evicted": evicted}


P0 = pin("p0", 10, [40, 98])
P1 = pin("p1", 5, [10, 60])
P2 = pin("p2", 2, [0, 50])


def rec(**kw):
    r = {"run_id": "r", "item_id": "it-1", "system": "S11", "prediction": "90 °C", "cited_pin_ids": [],
         "pins": [], "paged_in": [], "evicted": [], "faults": [], "descend_candidates": [], "skeleton": {},
         "edges_from_resident": {}, "tokens_in": 100, "tokens_out": 10, "api_usd": 0.01, "wall_ms": 5,
         "n_llm_calls": 2, "status": "ok", "error": None}
    r.update(kw)
    return r


def code(item, r, **kw):
    return sc.score_item(item, r, **kw)["failure_code"]


# ------------------------------------------------------------------ coverage
def test_pin_coverage_rules():
    f = make_item()["gold"]["facts"][0]                     # span [40, 98], length 58
    assert sc.pin_covers_fact(pin("a", 10, [40, 98]), f)
    assert sc.pin_covers_fact(pin("a", 10, [69, 200]), f)    # overlap 29 = exactly 50%
    assert not sc.pin_covers_fact(pin("a", 10, [70, 200]), f)  # overlap 28 < 50%
    assert not sc.pin_covers_fact(pin("a", 11, [40, 98]), f)   # other origin, span route needs same origin
    assert sc.pin_covers_fact(pin("a", 11, [0, 5], quote="...so VARREL pumps sources seals from Ostrand Ltd, and"), f)
    assert not sc.pin_covers_fact(pin("a", 10, [0, 5], quote="Varrel Pumps sources seals"), f)


def test_evidence_precision_recall():
    item = make_item()
    r = rec(pins=[P0, pin("px", 7, [0, 10], "noise")], cited_pin_ids=["p0"])
    ev = sc.evidence_scores(item, r)
    assert ev["evidence_precision"] == 0.5 and ev["evidence_recall"] == pytest.approx(1 / 3)
    assert ev["cited_precision"] == 1.0 and ev["cited_recall"] == pytest.approx(1 / 3)
    assert math.isnan(sc.evidence_scores(item, rec())["evidence_precision"])


# ------------------------------------------------------------------ failure codes, one test per branch
def test_correct_has_no_code():
    s = sc.score_item(make_item(), rec(prediction="83 C"))
    assert s["em"] == 1.0 and s["failure_code"] is None


def test_f8_wrong_thread_takes_precedence():
    r = rec(prediction="71 C")                              # nothing pinned or paged: F8 still wins
    assert code(make_item(), r) == "F8"
    assert sc.thread_confused(make_item(), r)


def test_f7_reasoning_error_all_pinned():
    assert code(make_item(), rec(pins=[P0, P1, P2], paged_in=[10, 5, 2])) == "F7"


def test_f6_evicted_pin():
    r = rec(pins=[P0, pin("p1", 5, [10, 60], evicted=True), P2], paged_in=[10, 5, 2])
    assert code(make_item(), r) == "F6"
    assert sc.score_item(make_item(), r)["failure_hop"] == 1


def test_first_uncovered_hop_decides():
    # hop 1 is uncovered (never paged) and hop 2 is also uncovered (paged, unpinned): hop 1 decides -> F1
    r = rec(pins=[P0], paged_in=[10, 2])
    s = sc.score_item(make_item(), r)
    assert s["failure_code"] == "F1" and s["failure_hop"] == 1


def test_f9_descent_wrong_turn():
    r = rec(pins=[P0], paged_in=[10], descend_candidates=[99, 10])
    assert code(make_item(), r) == "F9"
    r2 = rec(pins=[P0], paged_in=[10], descend_candidates=[99, 5])   # origin was a candidate: not F9
    assert code(make_item(), r2) == "F1"


def test_f5_alias_mismatch_fact_uses_alias():
    # hop 1's fact says "OSTRAND"; skeleton of origin 5 lacks the key the agent looks up ("ostrand ltd")
    r = rec(pins=[P0], paged_in=[10], skeleton={"5": ["kelvane"]})
    assert code(make_item(), r) == "F5"
    r2 = rec(pins=[P0], paged_in=[10], skeleton={"5": ["ostrand", "kelvane"]})  # alias key only: still F5
    assert code(make_item(), r2) == "F5"


def test_f5_vice_versa_and_f4():
    item = make_item()
    item["gold"]["facts"][1]["text"] = "Ostrand Ltd. makes its seals from Kelvane."   # canonical form in the fact
    assert code(item, rec(pins=[P0], paged_in=[10], skeleton={"5": ["ostrand"]})) == "F5"  # skeleton has alias only
    assert code(item, rec(pins=[P0], paged_in=[10], skeleton={"5": ["kelvane"]})) == "F4"  # no alias involved
    # hop 2 (no aliases): skeleton miss
    assert code(make_item(), rec(pins=[P0, P1], paged_in=[10, 5], skeleton={"2": ["coolant"]})) == "F4"


def test_skeleton_hit_falls_through_to_f10_or_f1():
    base = dict(pins=[P0, P1], paged_in=[10, 5], skeleton={"2": ["kelvane"]})
    assert code(make_item(), rec(**base, edges_from_resident={"2": False})) == "F10"
    assert code(make_item(), rec(**base, edges_from_resident={"5": True})) == "F10"      # absent -> false
    assert code(make_item(), rec(**base, edges_from_resident={"2": True})) == "F1"
    assert code(make_item(), rec(**base)) == "F1"                                       # no graph
    # explicit system flags override inference from the record
    assert code(make_item(), rec(**base), has_graph=True) == "F10"
    assert code(make_item(), rec(pins=[P0, P1], paged_in=[10, 5]), has_skeleton=True) == "F4"


def test_f2_and_f3():
    assert code(make_item(), rec(pins=[P0, P1], paged_in=[10, 5, 2])) == "F2"
    r = rec(pins=[P0, P1, pin("p9", 2, [200, 260], "unrelated words")], paged_in=[10, 5, 2])
    assert code(make_item(), r) == "F3"
    # an evicted non-covering pin from the origin still means "pinned but misread"
    r = rec(pins=[P0, P1, pin("p9", 2, [200, 260], "x", evicted=True)], paged_in=[10, 5, 2])
    assert code(make_item(), r) == "F3"


def test_error_record_gets_no_code():
    s = sc.score_item(make_item(), rec(status="error", prediction=None))
    assert s["error"] and s["failure_code"] is None and s["em"] == 0.0


def test_exactly_one_code_per_wrong_answer():
    variants = [rec(), rec(prediction="71 C"), rec(pins=[P0, P1, P2], paged_in=[10, 5, 2]),
                rec(pins=[P0], paged_in=[10], descend_candidates=[1]), rec(pins=[P0, P1], paged_in=[10, 5, 2])]
    for r in variants:
        c = code(make_item(), r)
        assert c in sc.FAILURE_CODES


# ------------------------------------------------------------------ gamma, faults, aggregate
def test_coverage_factors():
    r = rec(skeleton={"5": ["ostrand ltd"], "2": []}, edges_from_resident={"5": False, "2": True})
    cov = sc.coverage_factors(make_item(), r)
    assert cov["h_late"] == 2 and cov["gamma_s"] == 0.5 and cov["gamma_g"] == 0.5
    assert cov["gamma_eff"] == 0.75 and cov["predicted"] == pytest.approx(0.5625)
    assert cov["skeleton_miss"] == [False, True] and cov["graph_miss"] == [True, False]
    no_struct = sc.coverage_factors(make_item(), rec())
    assert no_struct["gamma_eff"] == 0.0 and no_struct["predicted"] == 0.0
    item0 = make_item()
    for f in item0["gold"]["facts"]:
        f["late_bound"] = False
    cov0 = sc.coverage_factors(item0, r)
    assert cov0["predicted"] == 1.0 and math.isnan(cov0["gamma_s"])


def test_fault_counts():
    r = rec(faults=[{"origin_id": 5}, {"origin_id": 77}, {"origin_id": 2}, {"origin_id": 78}])
    assert sc.fault_counts(make_item(), r) == {"n_faults": 4, "necessary_faults": 2, "spurious_faults": 2}


def test_aggregate():
    item = make_item()
    item2 = copy.deepcopy(item)
    item2["item_id"] = "it-2"
    item2["gold"]["confusers"] = []
    records = [
        rec(system="S11", prediction="83 C", faults=[{"origin_id": 5}], tokens_in=200, api_usd=0.02),
        rec(system="S11", item_id="it-2", prediction="1", pins=[P0, P1, P2], paged_in=[10, 5, 2],
            faults=[{"origin_id": 99}], tokens_in=100, api_usd=0.04),
        rec(system="S2", prediction="71 °C"),
        rec(system="S2", item_id="it-2", status="error", prediction=None),
    ]
    agg = sc.aggregate([item, item2], records)
    s11, s2 = agg["S11"], agg["S2"]
    assert s11["n"] == 2 and s11["em"] == 0.5 and s11["failure_codes"]["F7"] == 1
    assert s11["mean_tokens_in"] == 150 and s11["cost_usd"] == pytest.approx(0.06)
    assert s11["spurious_fault_rate"] == 0.5 and s11["mean_necessary_faults"] == 0.5
    assert s11["thread_confusion_rate"] == 0.0                 # only it-1 has confusers
    assert s2["thread_confusion_rate"] == 1.0 and s2["failure_codes"]["F8"] == 1
    assert s2["error_rate"] == 0.5 and s2["n_error"] == 1 and sum(s2["failure_codes"].values()) == 1
    assert math.isnan(s2["spurious_fault_rate"])
    by2 = sc.aggregate([item, item2], records, by=("system", "item_id"))
    assert by2[("S11", "it-2")]["em"] == 0.0


# ------------------------------------------------------------------ drift
def make_session():
    return {
        "item_id": "drift-dev-000001", "dataset": "drift",
        "turns": [{"turn": 1, "thread": "design", "text": "Max coolant temperature is 80 °C. Supplier is Brennick Supply."},
                  {"turn": 22, "thread": "testing", "text": "Change the max coolant temperature to 75 °C."},
                  {"turn": 30, "thread": "supplier", "text": "The old spec said 70 °C. Also we bought 12 pallets."}],
        "constraints": [
            {"key": "max_coolant_temp", "unit": "°C", "initial": {"value": "80", "turn": 1},
             "updates": [{"value": "75", "turn": 22}], "near_misses": [{"value": "70", "turn": 30}]},
            {"key": "supplier", "unit": None, "initial": {"value": "Brennick Supply", "turn": 1}, "updates": [],
             "near_misses": []},
        ],
        "probes": [{"turn": 10, "gold": {}}, {"turn": 25, "gold": {}}, {"turn": 50, "gold": {}}],
    }


def test_drift_outcomes():
    s = make_session()
    r = {"item_id": s["item_id"], "condition": "D2", "probes": [
        {"turn": 10, "predicted": {"max_coolant_temp": "80", "supplier": "Brennick Supply"}},
        {"turn": 25, "predicted": {"max_coolant_temp": "80 °C", "supplier": "Brennick supply"}},
        {"turn": 50, "predicted": {"max_coolant_temp": "70 C", "supplier": ""}},
    ]}
    assert sc.drift_probe_outcomes(s, r, 10) == {"max_coolant_temp": "correct", "supplier": "correct"}
    assert sc.drift_probe_outcomes(s, r, 25) == {"max_coolant_temp": "stale", "supplier": "correct"}
    assert sc.drift_probe_outcomes(s, r, 50) == {"max_coolant_temp": "near_miss", "supplier": "missing"}
    r["probes"][2]["predicted"] = {"max_coolant_temp": "99 °C", "supplier": "12 pallets"}
    assert sc.drift_probe_outcomes(s, r, 50) == {"max_coolant_temp": "fabricated", "supplier": "other"}
    r["probes"][2]["predicted"] = {"max_coolant_temp": "75 degrees C", "supplier": "Brennick Supply"}
    rows = sc.score_drift_session(s, r)
    assert [x["accuracy"] for x in rows] == [1.0, 0.5, 1.0]
    assert rows[1]["stale_rate"] == 0.5 and rows[1]["fabricated_rate"] == 0.0
    # 70 is a near-miss only from turn 30; at turn 25 it had not appeared -> fabricated
    r["probes"][1]["predicted"]["max_coolant_temp"] = "70"
    assert sc.drift_probe_outcomes(s, r, 25)["max_coolant_temp"] == "fabricated"


def test_drift_plus_minus_tolerance():
    s = make_session()
    s["constraints"].append({"key": "flatness", "unit": "mm", "initial": {"value": "±0.16", "turn": 1}, "updates": [],
                             "near_misses": []})
    s["turns"][0]["text"] += " Flatness tolerance ±0.16 mm."
    r = {"item_id": s["item_id"], "probes": [{"turn": 10, "predicted": {"flatness": "0.16 mm"}}]}
    assert sc.drift_probe_outcomes(s, r, 10)["flatness"] == "correct"
    r["probes"][0]["predicted"]["flatness"] = "0.61 mm"
    assert sc.drift_probe_outcomes(s, r, 10)["flatness"] == "fabricated"


def test_drift_on_generated_session():
    drift = pytest.importorskip("opa.data.drift")
    try:
        s = drift.make_session(3)
    except Exception as e:                                   # generator needs resources not present here
        pytest.skip(f"drift generator unavailable: {e}")
    perfect = {"item_id": s["item_id"], "condition": "D1",
               "probes": [{"turn": p["turn"], "predicted": dict(p["gold"])} for p in s["probes"]]}
    for row in sc.score_drift_session(s, perfect):
        assert row["accuracy"] == 1.0
    frozen = {"item_id": s["item_id"], "condition": "D2", "probes": [
        {"turn": p["turn"], "predicted": {c["key"]: c["initial"]["value"] for c in s["constraints"]}}
        for p in s["probes"]]}
    last = sc.score_drift_session(s, frozen)[-1]
    n_upd = sum(1 for c in s["constraints"] if c["updates"])
    assert last["stale_rate"] == pytest.approx(n_upd / len(s["constraints"]))
    assert last["accuracy"] == pytest.approx(1 - n_upd / len(s["constraints"]))


def test_drift_aggregate_and_binary():
    s = make_session()
    good = {"item_id": s["item_id"], "condition": "D4", "probes": [
        {"turn": t, "predicted": {"max_coolant_temp": v, "supplier": "Brennick Supply"}}
        for t, v in ((10, "80"), (25, "75"), (50, "75"))]}
    bad = {"item_id": s["item_id"], "condition": "D2", "probes": [
        {"turn": t, "predicted": {"max_coolant_temp": "80", "supplier": "Brennick Supply"}} for t in (10, 25, 50)]}
    agg = sc.aggregate_drift([s], [good, bad])
    assert agg["D4"][50]["accuracy"] == 1.0 and agg["D2"][50]["accuracy"] == 0.5
    assert agg["D2"][50]["stale_rate"] == 0.5 and agg["D2"][10]["accuracy"] == 1.0
    assert sc.drift_binary_outcomes([s], [bad], 50) == {(s["item_id"], "max_coolant_temp"): 0,
                                                        (s["item_id"], "supplier"): 1}


def test_thread_confused_uses_decoy_value_aliases():
    from opa.data.lbmhqa import answer_aliases
    it = make_item()
    # a decoy imitating a service-life / pressure value: bare numbers and unit variants must count
    it["gold"]["confusers"] = [
        {"entity": "Kelvanf", "value": "4300 hours", "value_aliases": answer_aliases("service_life", 4300),
         "origin_id": 30, "thread_id": 3, "char_span": [0, 40]},
        {"entity": "Kelvanf Pumps", "value": "22 bar", "value_aliases": answer_aliases("pressure", 22),
         "origin_id": 31, "thread_id": 4, "char_span": [0, 40]}]
    for pred in ("4300", "4300 h", "4,300 hours", "4300 operating hours", "22bar", "22", "22 bar"):
        assert sc.thread_confused(it, rec(prediction=pred)), pred
    for pred in ("4301", "23 bar", "83 C", "", None):
        assert not sc.thread_confused(it, rec(prediction=pred)), pred
    del it["gold"]["confusers"][0]["value_aliases"]                      # older items: fall back to [value]
    assert sc.thread_confused(it, rec(prediction="4300 hours")) and not sc.thread_confused(it, rec(prediction="4300"))
