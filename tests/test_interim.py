"""Interim Track A package (opa.analysis.interim, interim_figures, paper_input_interim) on synthetic data: no GCS, no ledger writes."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from opa import claims as cl
from opa import PROJECT_DIR
from opa.analysis import interim as itm
from opa.analysis import interim_figures as ifig
from opa.analysis import paper_input_interim as pii
from opa.analysis.tables import write_contrast_table, write_table

RULES, MIN_N = cl.load_prereg()
K = {"s2": "S11_vs_S2_lbmhqa_late_32k_primary", "s4": "S11_vs_S4_lbmhqa_late_32k_primary", "s8": "S11_vs_S8_late_bound",
     "a21": "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross", "tost": "A2_vs_A3_agent_equivalence", "h8": "descent_beam4_vs_flat_recall_at_M",
     "h10": "pinned_vs_unpinned_repeat_pagein_reduction", "pages": "A2_vs_A1_pages_read_before_gold_hit"}


def row(key, value, lo, hi, n, p=None, split="test", **kw):
    r = {"key": key, "value": value, "ci_lo": lo, "ci_hi": hi, "n": n, "p_raw": p, "split": split, "test": "paired bootstrap",
         "run_ids": ["R1"], "em_a": 0.7, "em_b": 0.6, "systems": "A vs B", "n_itt_imputed": 0, "n_excluded_missing": 0}
    r.update(kw)
    return r


def strings(r):
    return {k: ("" if v is None else str(v)) for k, v in r.items()}


def contrasts(split="test", strong=True, n_scale=1):
    n = lambda k: int(MIN_N[k] * n_scale)             # noqa: E731
    rows = [row(K["s2"], 7.0, 3.0, 11.0, n(K["s2"]), 0.001, split), row(K["s4"], 40.0, 30.0, 50.0, n(K["s4"]), None, split),
            row(K["s8"], 3.0, 1.0, 5.0, n(K["s8"]), 0.01, split), row(K["a21"], 6.0, 2.0, 10.0, n(K["a21"]), 0.004, split),
            row(K["tost"], 1.0, -1.0, 3.0, 191, 0.02, split), row(K["pages"], -0.5, -0.9, -0.1, 249, 0.03, split),
            row(K["h8"], 0.0, -1.0, 1.0, n(K["h8"]), None, split),
            row(K["h10"], 3.5, 2.5, 4.5, n(K["h10"]), 1e-9, split)]
    return {r["key"]: strings(r) for r in rows}


UNITS = {"runs": {"u1_u5": "RU15", "u6_u8_h7": "RU68"}, "h7": {"reanchored_error_growth_ratio_depth40_vs_depth1_bf16": 0.96,
                                                               "global_fail_depth_bf16": 4.0, "global_fail_depth_fp16": 7.0,
                                                               "global_fail_depth_fp32": 16.0},
         "U": {f"U{i}": {"run_id": "RU", "code_flag": True, "summary": {}, "test": {"passed": True, "criterion": "c", "metrics": {}}}
               for i in range(1, 9)}}
UNITS["U"]["U8"]["code_flag"] = False
UNITS["U"]["U8"]["test"] = {"passed": False, "criterion": "c", "metrics": {"tv_vs_evicted_router_mass_max_abs_err": 3.3e-16,
                                                                         "paged_sdpa_output_max_abs_err": 2.08e-05, "tolerance": 1e-5}}


def verdict_map(split="test", **kw):
    v = itm.build_verdicts(contrasts(split, **kw), UNITS, RULES, MIN_N, split)
    return v, {r["key"]: r for r in v["rows"]}


# ------------------------------------------------------------------------------------ verdicts
def test_family_is_the_frozen_holm_family():
    assert itm.family(RULES) == ["H1", "H2", "H3", "H4", "H5", "H6", "H10", "H12"]


def test_bonferroni_and_family_size():
    v, rows = verdict_map()
    r = rows["H1:" + K["s2"]]
    assert r["p_raw"] == pytest.approx(0.001) and r["p_bonferroni"] == pytest.approx(0.008) and r["family_m"] == 8
    assert r["holm_status"] == itm.HOLM_PENDING
    fam = {x["key"]: x for x in v["family"]}
    assert set(fam) == {"H1", "H2", "H3", "H4", "H5", "H6", "H10", "H12"} and fam["H1"]["family_m"] == 8
    assert fam["H2"]["p_raw"] is None and fam["H2"]["status"] == "inconclusive (not funded)"
    assert itm.bonferroni(0.2, 8) == 1.0 and itm.bonferroni(None, 8) is None


def test_h1_supported_when_both_leaves_pass_with_enough_n():
    _, rows = verdict_map()
    assert rows["H1"]["verdict_interim"] == "supported"
    assert rows["H1:" + K["s2"]]["verdict_interim"] == "supported" and rows["H1:" + K["s4"]]["verdict_interim"] == "supported"


def test_below_min_n_is_inconclusive_whatever_the_estimate():
    _, rows = verdict_map(n_scale=0.5)
    assert rows["H1:" + K["s2"]]["verdict_interim"] == "inconclusive"
    assert rows["H1"]["verdict_interim"] == "inconclusive"
    assert "min_n" in rows["H1:" + K["s2"]]["verdict_basis"]
    assert rows["H1"]["verdict_ignoring_min_n"] == "supported"           # informational column


def test_h3_is_never_supported_and_the_leg_is_reported_separately():
    _, rows = verdict_map()
    # leg: raw p 0.01 -> Bonferroni 0.08 > 0.05: refuted under the stand-in, but Holm-sensitive (raw p would pass)
    leg = rows["H3:" + K["s8"]]
    assert leg["verdict_interim"] == "pending_final_holm" and leg["holm_sensitive"] is True
    assert "refuted at the Bonferroni" in leg["verdict_basis"] and "supported at raw p" in leg["verdict_basis"]
    assert rows["H3"]["verdict_interim"] == "pending_final_holm"        # sensitive too: all_of with a Holm-sensitive refuted leg
    # a refutation that holds at raw p as well is a plain refutation
    c0 = contrasts()
    c0[K["s8"]] = strings(row(K["s8"], 3.0, 1.0, 5.0, MIN_N[K["s8"]], 0.5))
    r0 = {r["key"]: r for r in itm.build_verdicts(c0, UNITS, RULES, MIN_N, "test")["rows"]}
    assert r0["H3:" + K["s8"]]["verdict_interim"] == "refuted" and r0["H3"]["verdict_interim"] == "refuted"
    assert "R2" in rows["H3:R2_predicted_vs_observed_success"]["notes"]
    assert rows["H3:R2_predicted_vs_observed_success"]["verdict_interim"] == "inconclusive"
    # a passing leg leaves H3 inconclusive, never supported
    c = contrasts()
    c[K["s8"]] = strings(row(K["s8"], 3.0, 1.0, 5.0, MIN_N[K["s8"]], 1e-5))
    rows2 = {r["key"]: r for r in itm.build_verdicts(c, UNITS, RULES, MIN_N, "test")["rows"]}
    assert rows2["H3:" + K["s8"]]["verdict_interim"] == "supported" and rows2["H3"]["verdict_interim"] == "inconclusive"


def test_h5_model_leaf_pending_keeps_agent_refutation_inconclusive():
    c = contrasts()
    c[K["a21"]] = strings(row(K["a21"], -2.0, -5.0, 1.0, MIN_N[K["a21"]], 0.3))
    rows = {r["key"]: r for r in itm.build_verdicts(c, UNITS, RULES, MIN_N, "test")["rows"]}
    assert rows["H5:" + K["a21"]]["verdict_interim"] == "refuted"
    assert rows["H5"]["verdict_interim"] == "inconclusive"
    assert rows["H5:A2_vs_A1_model_thread_confusion_task"]["verdict_interim"] == "inconclusive"
    _, ok = verdict_map()
    assert ok["H5"]["verdict_interim"] == "supported"                    # agent arm alone supports the any_of


def test_h7_from_the_measurements_and_h8_h10():
    _, rows = verdict_map()
    assert rows["H7"]["verdict_interim"] == "supported" and rows["H7"]["data_split"] == "n/a"
    assert rows["H8"]["verdict_interim"] == "supported" and rows["H8:" + K["h8"]]["p_raw"] is None
    assert rows["H10"]["verdict_interim"] == "supported"
    bad = dict(UNITS, h7={**UNITS["h7"], "global_fail_depth_fp16": 9.0})
    rows2 = {r["key"]: r for r in itm.build_verdicts(contrasts(), bad, RULES, MIN_N, "test")["rows"]}
    assert rows2["H7"]["verdict_interim"] == "refuted"
    none = {r["key"]: r for r in itm.build_verdicts(contrasts(), {"h7": {}}, RULES, MIN_N, "test")["rows"]}
    assert none["H7"]["verdict_interim"] == "inconclusive"               # a missing measurement is not invented


def test_non_scope_hypotheses_get_status_not_verdict():
    _, rows = verdict_map()
    for h, s in (("H2", "inconclusive (not funded)"), ("H4", "running on the test split (gate G1)"), ("H6", "running on the test split (gate G1)"),
                 ("H9", "conditional_on_g2"), ("H11", "pending_track_b"), ("H13", "inconclusive (not funded)")):
        assert rows[h]["verdict_interim"] == s
    assert rows["H12"]["holm_status"] == itm.HOLM_PENDING


def test_dev_verdicts_are_prefixed_and_use_the_same_evaluator():
    _, rows = verdict_map("dev")
    assert rows["H1"]["verdict_interim"] == "dev:supported" and rows["H1"]["data_split"] == "dev"
    assert rows["H7"]["verdict_interim"] == "dev:supported"
    assert rows["H2"]["verdict_interim"] == "inconclusive (not funded)"


def test_verdict_table_columns_and_secondary_rows():
    v, rows = verdict_map()
    assert all(set(r) <= set(itm.VERDICT_COLUMNS) for r in v["rows"])
    assert rows["H5:" + K["tost"]]["verdict_interim"] == "reported" and rows["H5:" + K["tost"]].get("p_bonferroni") is None


# ------------------------------------------------------------------------------------ names / split hygiene
def test_check_name_rules():
    assert itm.check_name("dev", "interim_verdicts_dev.csv")
    with pytest.raises(itm.InterimError):
        itm.check_name("dev", "interim_verdicts.csv")
    with pytest.raises(itm.InterimError):
        itm.check_name("test", "interim_verdicts_dev.csv")
    assert itm.check_name("test", "v3_learning_curve_dev.csv", dev_data=True)
    with pytest.raises(itm.InterimError):
        itm.check_name("dev", "p2_contrasts_test_dev.csv")
    with pytest.raises(itm.InterimError):
        itm.check_split("prod")


def test_runs_yaml_both_forms(tmp_path):
    flat = tmp_path / "a.yaml"
    flat.write_text("h1-lb: [r1, r2]\n")
    nested = tmp_path / "b.yaml"
    nested.write_text("p2:\n  h1-lb: [r1]\ninterim:\n  learning_curve: [x]\n")
    assert itm.load_runs_yaml(flat) == ({"h1-lb": ["r1", "r2"]}, {})
    assert itm.load_runs_yaml(nested) == ({"h1-lb": ["r1"]}, {"learning_curve": ["x"]})


# ------------------------------------------------------------------------------------ costs / gates / deviations
LEDGER = [
    {"ts_utc": "2026-11-07T01:00:00Z", "envelope": "E2", "job": "p2-h1-lb-c01", "kind": "reserve", "amount_usd": "2.0", "provider": "modal", "run_id": "r", "note": ""},
    {"ts_utc": "2026-11-07T01:10:00Z", "envelope": "E2", "job": "p2-h1-lb-c01", "kind": "actual", "amount_usd": "0.9", "provider": "modal", "run_id": "r", "note": ""},
    {"ts_utc": "2026-11-07T01:10:00Z", "envelope": "E2", "job": "p2-h1-lb-c01", "kind": "release", "amount_usd": "2.0", "provider": "modal", "run_id": "r", "note": ""},
    {"ts_utc": "2026-11-07T02:00:00Z", "envelope": "E3", "job": "p3-units-u6-u8-h7", "kind": "actual", "amount_usd": "0.015", "provider": "modal", "run_id": "q", "note": ""},
    {"ts_utc": "2026-11-07T03:00:00Z", "envelope": "E2", "job": "p2-h10-pin-c01", "kind": "reserve", "amount_usd": "1.5", "provider": "modal", "run_id": "s", "note": ""},
    {"ts_utc": "2026-11-07T03:00:00Z", "envelope": "E0", "job": "p0-g0-s1-rep1", "kind": "actual", "amount_usd": "0.5", "provider": "modal", "run_id": "t", "note": ""},
]
APPR = {"total_cap_usd": 148, "envelopes": [{"id": e, "approved_usd": a, "approved_by": "AR", "approved_on": "2026-11-07"}
                                             for e, a in (("E0", 7.2), ("E2", 38.94), ("E3", 11.92), ("E5", 2.62))]}


def test_costs_tables_from_a_synthetic_ledger():
    c = itm.costs_tables(APPR, LEDGER)
    env = {r["key"]: r for r in c["envelope"]}
    assert env["E2"]["spent_usd"] == pytest.approx(0.9) and env["E2"]["reserved_usd"] == pytest.approx(1.5)
    assert env["TOTAL"]["approved_usd"] == 148 and env["TOTAL"]["spent_usd"] == pytest.approx(1.415)
    assert env["TOTAL"]["remaining_usd"] == pytest.approx(148 - 1.415 - 1.5)
    grp = {r["key"]: r for r in c["group"]}
    assert grp["P2 H1 (LB-MHQA): embeddings, skeletons, S11/S2/S8/S4"]["spent_usd"] == pytest.approx(0.9)
    assert grp["Units U6-U8 and H7"]["hypotheses"] == "H7;U6-U8" and "P0 setup: smoke, determinism, G0, benchmarks" in grp
    assert c["as_of"] == "2026-11-07T03:00:00Z"
    assert itm.group_of("p2-h5-mt-c03")[1] == "H5;H8" and itm.group_of("p1-thrash-pin")[1] == "H10;H2"


def test_gates_tables_pending_and_history(tmp_path):
    g = tmp_path / "gates"
    (g / "history").mkdir(parents=True)
    crit = [{"name": "H3 R2", "threshold": ">= 0.5", "observed": {"value": -7.38}, "met": False}, {"name": "H1", "threshold": ">=5", "observed": 10.0, "met": True}]
    (g / "G0.json").write_text(json.dumps({"id": "G0", "passed": True, "decided_at": "t0", "criteria": crit, "notes": "n"}))
    (g / "G1.json").write_text(json.dumps({"id": "G1", "passed": True, "decided_at": "t1", "criteria": crit[1:], "notes": "override"}))
    (g / "history" / "G1-original.json").write_text(json.dumps({"id": "G1", "passed": False, "decided_at": "t", "criteria": crit, "notes": "x"}))
    t = itm.gates_tables(tmp_path)
    st = {r["key"]: r["status"] for r in t["gates"]}
    assert st == {"G0": "passed", "G1": "passed", "G2": "pending", "G3": "pending", "G4": "pending"}
    assert t["history"][0]["status"] == "failed" and "H3 R2" in t["history"][0]["failed_criteria"]
    obs = {r["key"]: r["observed_value"] for r in t["criteria"]}
    assert obs["G1-original:0"] == -7.38 and obs["G0:1"] == 10.0


def test_deviation_rows_cover_files_and_disclosures(tmp_path):
    rows = {r["key"]: r for r in itm.deviation_rows(PROJECT_DIR, tmp_path)}
    for k in ("DEV-U8", "DEV-TRAIN-PIN", "DEV-U1", "DEV-G0-noise", "G1-OVERRIDE", "H3-R2-NOT-TESTABLE"):
        assert k in rows
    assert rows["DEV-U8"]["date"] == "2026-11-08" and rows["DEV-U8"]["source"].startswith("prereg/deviations/")
    (tmp_path / "deviations.md").write_text("### DEV-X1\n- Date: 2026-11-09\n- Change: a thing\n- Reason: r\n- Impact: i\n")
    assert "DEV-X1" in {r["key"] for r in itm.deviation_rows(PROJECT_DIR, tmp_path)}


def test_trackc_and_learning_curve_rows():
    t = itm.trackc_rows([{"key": "E5", "approved_usd": 2.62, "spent_usd": 0.0}])
    assert {r["key"] for r in t} == {"TrackC", "H9", "H13"} and t[0]["status"] == "not started"
    doc = {"curve": [{"arm": "V3", "opt_step": s, "train_loss": 3.0 - s / 1000, "attrib_em": 0.1, "attrib_rank_acc": 0.1, "attrib_chance": 0.17,
                      "passkey_em": 0.5, "probe": {"own_origin_mass_fraction": f, "own_origin_mass_ratio_to_uniform": 2.0}}
                     for s, f in ((0, 0.14), (200, 0.25), (400, 0.31))]}
    later = {"curve": [{"arm": "V3", "opt_step": 400, "probe": {"own_origin_mass_fraction": 0.5}}, {"arm": "V3", "opt_step": 600, "probe": {"own_origin_mass_fraction": 0.6}}]}
    rows = itm.learning_curve_rows([("run1", doc), ("run2", later)])
    by = {r["key"]: r for r in rows}
    assert list(by) == ["V3_step0", "V3_step200", "V3_step400", "V3_step600"]
    assert by["V3_step400"]["run_id"] == "run2" and by["V3_step400"]["own_origin_mass_fraction"] == 0.5       # later run replaces
    st = {r["key"]: r for r in itm.trackb_status_rows(itm.learning_curve_rows([("run1", doc)]))}
    assert st["V3_latest_own_origin_mass_fraction"]["status"] == "below G2 threshold"
    assert st["V3_rise_last_two_evaluations"]["value"] == pytest.approx(0.06) and "continues" in st["V3_rise_last_two_evaluations"]["status"]
    assert itm.trackb_status_rows([])[0]["status"] == "no curve"


def test_units_rows_u8_reports_the_identity_and_the_flag_separately():
    rows = {r["key"]: r for r in itm.units_rows(UNITS)}
    assert rows["U8"]["status"] == "pass" and rows["U8"]["code_flag"] is False and rows["U8"]["deviation"] == "DEV-U8"
    assert rows["U8"]["aux_observed"] == pytest.approx(2.08e-05) and rows["U8"]["observed"] == pytest.approx(3.3e-16)
    assert itm.units_rows({})[0]["status"] == "no result"


# ------------------------------------------------------------------------------------ end to end on a synthetic bundle
def make_bundle(tmp_path, split="test", n_scale=1, g2=None, trackb=None, own=None):
    out = tmp_path / "bundle"
    (out / "tables").mkdir(parents=True)
    (out / "gates").mkdir()
    for g in ("G0", "G1"):
        shutil.copy(PROJECT_DIR / "bundle" / "gates" / f"{g}.json", out / "gates" / f"{g}.json")
    shutil.copytree(PROJECT_DIR / "bundle" / "gates" / "history", out / "gates" / "history")
    if g2 is not None:
        (out / "gates" / "G2.json").write_text(json.dumps(g2))
    c = contrasts(split, n_scale=n_scale)
    write_contrast_table(out / "tables" / f"p2_contrasts_{split}.csv", [{**v, "key": k} for k, v in c.items()],
                         extra_columns=["split", "systems", "em_a", "em_b", "n_itt_imputed", "n_excluded_missing", "note"])
    dev = {k: {**v, "split": "dev"} for k, v in contrasts("dev").items()}
    dist = [{"key": f"T{T}-h3-32k-cross_v2_gold_clampnone", "tree_distortion_euclidean_over_hyperbolic": r, "n_items": 100}
            for T, r in ((4, 1.36), (8, 1.29), (16, 1.25))]
    write_table(out / "tables" / "h5_distortion_dev.csv", dist, columns=["key", "tree_distortion_euclidean_over_hyperbolic", "n_items"])
    curve = {"curve": [{"arm": "V3", "opt_step": s, "train_loss": 3.0 - s / 500, "attrib_em": 0.1 + s / 2000, "attrib_rank_acc": 0.1, "attrib_chance": 0.1667,
                        "passkey_em": 0.9, "probe": {"own_origin_mass_fraction": 0.14 + s / 3000, "own_origin_mass_ratio_to_uniform": 1.2}}
                       for s in (0, 50, 100, 200, 400)]}
    if own:
        for e in curve["curve"]:
            if e["opt_step"] in own:
                e["probe"]["own_origin_mass_fraction"] = own[e["opt_step"]]
        curve["curve"] += [{"arm": "V3", "opt_step": st, "train_loss": 2.3, "attrib_em": 0.3, "attrib_rank_acc": 0.2, "attrib_chance": 0.1667,
                            "passkey_em": 0.95, "probe": {"own_origin_mass_fraction": v, "own_origin_mass_ratio_to_uniform": 2.0}}
                           for st, v in own.items() if st not in (0, 50, 100, 200, 400)]
    if trackb:
        (out / "tables" / "trackb").mkdir()
        write_contrast_table(out / "tables" / "trackb" / "trackb_contrasts_test.csv", [{**v, "key": k} for k, v in trackb.items()],
                             extra_columns=["split", "note"])
        c = {**c, **trackb}
    itm.write_all_tables(out, split, contrasts=c, units=UNITS, rules=RULES, min_n=MIN_N, curves=[("RUNLC", curve)], dev_rows=dev,
                         repo=PROJECT_DIR, approvals=APPR, ledger=LEDGER)
    return out, c, dev


def test_end_to_end_test_split_tables_figures_paper_input_and_refs(tmp_path):
    out, c, dev = make_bundle(tmp_path, "test")
    names = {p.name for p in (out / "tables").iterdir()}
    assert {"interim_verdicts.csv", "interim_family.csv", "interim_units.csv", "interim_costs_envelope.csv", "interim_gates.csv",
            "interim_deviations.csv", "interim_trackc.csv", "v3_learning_curve_dev.csv", "trackb_status_dev.csv"} <= names
    assert not any(n.startswith("interim_") and "_dev." in n for n in names)           # no dev name in the test-mode verdict files
    head = (out / "tables" / "interim_verdicts.csv").read_text().splitlines()[0]
    assert head.startswith("key,hypothesis,kind,rule_key,value")
    figs = ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    for n in ("fig_h1_depth_context_test_with_dev_pilot", "fig_h5_thread_depth_test_with_dev_pilot"):
        for ext in ("png", "pdf", "json", "csv"):
            assert (out / "figures" / f"{n}.{ext}").is_file()
    for ext in ("png", "pdf", "json", "csv"):
        assert (out / "figures" / f"fig_v3_learning_curve_dev.{ext}").is_file()
    side = json.loads((out / "figures" / "fig_h1_depth_context_test_with_dev_pilot.json").read_text())
    assert side["script"] == "code/interim_figures.py" and side["inputs"][0]["path"] == "tables/fig_h1_depth_context_test_with_dev_pilot.csv"
    src = (out / "tables" / "fig_h1_depth_context_test_with_dev_pilot.csv").read_text()
    assert "test|late-h2-32k" in src and "dev|late-h2-32k" in src                    # both splits, labelled by the split column
    assert len(figs) >= 12
    md = pii.write_paper_input_interim(out, "test")
    assert md.name == "PAPER_INPUT_interim.md"
    assert pii.check_refs(md, out) == []
    text = md.read_text()
    for h in range(1, 14):
        assert f"### H{h} " in text
    assert "VERDICT: supported" in text and "Bonferroni" in text and "owner override" in text
    for dev_id in ("DEV-U8", "DEV-TRAIN-PIN", "DEV-U1", "DEV-G0-noise"):
        assert dev_id in text
    assert "## Track B status" in text and "## Track C status" in text and "learning curve" in text


def test_dev_mode_names_and_banner(tmp_path):
    out, c, dev = make_bundle(tmp_path, "dev")
    for p in (out / "tables").iterdir():
        assert "dev" in p.name, p.name
    ifig.make_figures(out, "dev", {}, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "dev")
    assert md.name == "PAPER_INPUT_interim_dev.md" and "DEV DRY RUN" in md.read_text()
    assert pii.check_refs(md, out) == []
    for p in (out / "figures").iterdir():
        assert "dev" in p.name


def test_check_refs_catches_wrong_missing_and_unresolvable(tmp_path):
    out, c, dev = make_bundle(tmp_path, "test")
    ref = "tables/interim_verdicts.csv#row=H1:" + K["s2"] + "&cols=value"
    base = "# t\n\n## Results\n\n"
    good = tmp_path / "good.md"
    good.write_text(base + f"- value 7.0 [[ref:{ref}]] points\n")
    assert pii.check_refs(good, out) == []
    bad = tmp_path / "bad.md"
    bad.write_text(base + f"- value 7.5 [[ref:{ref}]] points\n")
    assert any("does not match" in p for p in pii.check_refs(bad, out))
    naked = tmp_path / "naked.md"
    naked.write_text(base + "- value 7.0 points\n")
    assert any("has no [[ref" in p for p in pii.check_refs(naked, out))
    gone = tmp_path / "gone.md"
    gone.write_text(base + "- value 7.0 [[ref:tables/nope.csv#row=a&cols=value]]\n")
    assert pii.check_refs(gone, out)
    # a section the verifier itself skips is checked in strict mode only
    other = tmp_path / "other.md"
    other.write_text("# t\n\n## Gates\n\n- value 7.0 points\n")
    assert pii.check_refs(other, out) and pii.check_refs(other, out, strict=False) == []
    # backticked numbers are quotes
    quote = tmp_path / "quote.md"
    quote.write_text(base + "- paper said `0.913` and 95% CI\n")
    assert pii.check_refs(quote, out) == []


def test_defang_leaves_backticks_and_ci_alone():
    assert pii.defang("limit 2.0 and 20% and $3.50") == "limit `2.0` and `20%` and `$3.50`"
    assert pii.defang("`0.5` kept, 0.7 not; 95% CI") == "`0.5` kept, `0.7` not; 95% CI"
    assert pii.defang("depth 4 and run 20261107") == "depth 4 and run 20261107"


def test_split_guard_in_run_requires_valid_split(tmp_path):
    import argparse
    a = argparse.Namespace(split="prod", runs="x", out=str(tmp_path), runs_dir=None, creds=None, reuse_p2=True, dry_run=True,
                           no_dev_context=True)
    with pytest.raises(itm.InterimError):
        itm.run(a)


def test_discover_series_latest_ok_run_per_job_name(tmp_path):
    runs = tmp_path / "runs"

    def mk(rid, name, split, status):
        d = runs / rid
        d.mkdir(parents=True)
        import yaml as _y
        (d / "job.yaml").write_text(_y.safe_dump({"name": name, "split": split}))
        if status:
            (d / "result.json").write_text(json.dumps({"status": status}))

    mk("A-x-test-s1-20261108010101-abcdef0-0001", "p2-h1-lb-c02", "test", "ok")
    mk("A-x-test-s1-20261108020202-abcdef1-0002", "p2-h1-lb-c02", "test", "ok")          # later ok run wins
    mk("A-x-test-s1-20261108010101-abcdef0-0003", "p2-h1-lb-c01", "test", "ok")
    mk("A-x-test-s1-20261108030303-abcdef2-0004", "p2-h1-lb-c03", "test", None)          # still running: left out, listed
    mk("A-x-test-s1-20261108010101-abcdef0-0005", "p2-h10-pin-c01", "test", "failed")
    mk("A-x-dev-s0-20261108010101-abcdef0-0006", "p2-h1-lb-c04", "dev", "ok")           # wrong split for a P2 name: ignored
    mk("B-units-dev-s0-20261107010101-abcdef0-0007", "p0-trackb-units", "dev", "ok")
    mk("B-units-dev-s0-20261107020202-abcdef0-0008", "p0-trackb-units", "dev", "ok")
    mk("B-V3-dev-s0-20261107223205-abcdef0-0009", "p3-learning-curve-v3", "dev", "ok")
    mk("B-V3-dev-s0-20261108050505-abcdef0-0010", "p4-v3-c01", "dev", "ok")
    d = itm.discover_series(runs, "test")
    assert d["p2"] == {"h1-lb": ["A-x-test-s1-20261108010101-abcdef0-0003", "A-x-test-s1-20261108020202-abcdef1-0002"]}
    assert d["interim"]["units_u1_u5"].endswith("0008")
    assert d["interim"]["learning_curve"] == ["B-V3-dev-s0-20261107223205-abcdef0-0009", "B-V3-dev-s0-20261108050505-abcdef0-0010"]
    assert any("p2-h1-lb-c03" in x for x in d["not_ok"]) and any("p2-h10-pin-c01" in x for x in d["not_ok"])
    out = tmp_path / "s.yaml"
    assert itm.main(["--discover", str(out), "--runs-dir", str(runs)]) == 0
    assert itm.load_runs_yaml(out)[0]["h1-lb"][0].endswith("0003")


def test_reuse_p2_requires_tag_and_matching_split(tmp_path):
    import argparse
    out = tmp_path / "b"
    (out / "tables").mkdir(parents=True)
    write_contrast_table(out / "tables" / "p2_contrasts_test.csv", [{**v, "key": k, "split": "dev"} for k, v in contrasts("dev").items()],
                         extra_columns=["split"])
    root = tmp_path / "root"
    a = argparse.Namespace(split="test", runs=str(tmp_path / "r.yaml"), out=str(out), runs_dir=None, creds=None, reuse_p2=True, dry_run=True,
                           no_dev_context=True, project_root=str(root))
    (tmp_path / "r.yaml").write_text("h1-lb: []\n")
    (root / "prereg").mkdir(parents=True)
    for f in ("rules.yaml", "sample_sizes.yaml"):
        shutil.copy(PROJECT_DIR / "prereg" / f, root / "prereg" / f)
    with pytest.raises(PermissionError):                       # no tag record in this root
        itm.run(a, store=object())
    (root / "prereg" / "PREREG_TAG.json").write_text(json.dumps({"tag": "prereg-v2", "commit": "c", "committed_at": "2026-11-07T18:59:43-05:00"}))
    with pytest.raises(itm.InterimError, match="split"):       # dev rows in the test file
        itm.run(a, store=object())


def test_started_after_tag_uses_the_run_id_timestamp(tmp_path):
    tag = tmp_path / "t.json"
    tag.write_text(json.dumps({"committed_at": "2026-11-07T18:59:43-05:00"}))        # 23:59:43Z
    n, bad = itm.started_after_tag({"h1-lb": ["A-x-test-s1-20261108000000-abcdef0-0001"]}, tag)
    assert (n, bad) == (1, [])
    n, bad = itm.started_after_tag({"h1-lb": ["A-x-test-s1-20261107235943-abcdef0-0001", "A-x-test-s1-20261107200000-abcdef0-0002", "nostamp"]}, tag)
    assert n == 3 and len(bad) == 3


def test_discover_reports_repeats(tmp_path):
    runs = tmp_path / "runs"
    import yaml as _y
    ids = ["A-x-test-s1-20261108010101-abcdef0-0001", "A-x-test-s1-20261108020202-abcdef1-0002"]
    for rid in ids:
        d = runs / rid
        d.mkdir(parents=True)
        (d / "job.yaml").write_text(_y.safe_dump({"name": "p2-h10-pin-c01", "split": "test"}))
        (d / "result.json").write_text(json.dumps({"status": "ok"}))
    d = itm.discover_series(runs, "test")
    assert d["repeats"] == {"p2-h10-pin-c01": sorted(ids)} and d["p2"]["h10-pin"] == [ids[1]]


def test_no_truncation_of_gate_text_and_design_table(tmp_path):
    g = tmp_path / "gates"
    g.mkdir()
    long_note = "word " * 100
    obs = {"em": [0.38, 0.4], "note": "x" * 300}
    (g / "G0.json").write_text(json.dumps({"id": "G0", "passed": True, "decided_at": "t", "criteria": [{"name": "n", "threshold": "t", "observed": obs, "met": True}], "notes": long_note}))
    t = itm.gates_tables(tmp_path)
    assert t["gates"][0]["note"] == long_note and json.loads(t["criteria"][0]["observed_text"]) == obs
    d = {r["key"]: r["value"] for r in itm.design_rows(RULES)}
    assert d["h7_reanchored_growth_limit"] == 2.0 and d["g2_own_origin_threshold"] == 0.9 and d["h8_ni_margin_points"] == -3.0


def test_tiny_p_visible_text_and_design_numbers_are_referenced(tmp_path):
    out, c, dev = make_bundle(tmp_path, "test")
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "test", args_runs={"h1-lb": ["A-x-test-s1-20261108000000-abcdef0-0001"]})
    text = md.read_text()
    assert "`(< 0.000001)`" in text and "1e-6" not in text
    assert "pending_final_holm" in text and "marked pending the final Holm" in text
    assert "all 1 test-split runs (1 Track A + 0 Track B) started after it" in text
    assert "`0.52`" not in text and "design power 0.52 [[ref:tables/interim_design.csv" in text
    assert pii.check_refs(md, out) == []


G2_FAILED = {"id": "G2", "passed": False, "decided_at": "2026-11-08T09:00:00Z", "evidence": ["runs/x"], "notes": "n",
             "criteria": [{"name": "V3 own-origin mass fraction on fresh dev seeds", "threshold": ">= 0.90", "observed": 0.314, "met": False},
                          {"name": "V3 minus B1 attribution", "threshold": "> 0", "observed": {"value": 0.05, "n": 128}, "met": True}]}
G2_PASSED = {**G2_FAILED, "passed": True}


def tb_rows(split="test", v3b1=(6.0, 2.0, 10.0, 0.001), n=2451, own=0.95):
    r = {}
    r["V3_vs_B1_source_attribution"] = strings(row("V3_vs_B1_source_attribution", v3b1[0], v3b1[1], v3b1[2], n, v3b1[3], split))
    r["V3_own_origin_mass_fraction"] = strings(row("V3_own_origin_mass_fraction", own, None, None, 10, None, split))
    r["V0_own_origin_mass_ratio_to_uniform"] = strings(row("V0_own_origin_mass_ratio_to_uniform", 1.0, None, None, 10, None, split))
    r["R5_vs_R1_accuracy"] = strings(row("R5_vs_R1_accuracy", 4.0, 1.0, 7.0, 400, 0.001, split))
    r["best_remedy_recovery_fraction"] = strings(row("best_remedy_recovery_fraction", 0.95, 0.9, 1.0, 400, None, split))
    return r


def test_outside_status_follows_the_g2_record():
    none, failed, passed = itm.outside_status(None), itm.outside_status(G2_FAILED), itm.outside_status(G2_PASSED)
    assert itm.g2_state(None) is None and itm.g2_state(G2_FAILED) == "failed" and itm.g2_state(G2_PASSED) == "passed"
    for h in ("H11", "H12"):
        assert failed[h][0] == "not_run_gate_failed (G2)"
    assert failed["H13"][0] == "not_run_gate_failed (G2)" and "also not funded in the approved package" in failed["H13"][1]
    assert failed["H9"][0] == "not_run_gate_failed (G2); owner request budget/requests/H9_exploratory.md"
    assert failed["H4"][0] == failed["H6"][0] == "running on the test split (gate G1)"
    assert none["H9"][0] == "conditional_on_g2" and none["H11"][0] == "pending_track_b" and none["H13"][0] == "inconclusive (not funded)"
    assert passed["H9"][0].startswith("pending_track_c") and passed["H12"][0].startswith("pending_track_b")
    assert failed["H2"][0] == "inconclusive (not funded)"


def test_g2_failed_shows_in_verdict_table_h5_model_leaf_and_trackc():
    v = itm.build_verdicts(contrasts(), UNITS, RULES, MIN_N, "test", g2=G2_FAILED)
    rows = {r["key"]: r for r in v["rows"]}
    for h in ("H11", "H12"):
        assert rows[h]["verdict_interim"] == "not_run_gate_failed (G2)"
    assert rows["H13"]["verdict_interim"] == "not_run_gate_failed (G2)"
    assert rows["H9"]["verdict_interim"].startswith("not_run_gate_failed (G2); owner request")
    leaf = rows["H5:A2_vs_A1_model_thread_confusion_task"]
    assert "gate G2 failed" in leaf["verdict_reason"] and "not_run_gate_failed (G2)" in leaf["notes"]
    assert "not_run_gate_failed (G2)" in rows["H5"]["notes"]
    fam = {r["key"]: r for r in v["family"]}
    assert fam["H12"]["status"] == "not_run_gate_failed (G2)"
    tc = {r["key"]: r for r in itm.trackc_rows([{"key": "E5", "approved_usd": 2.62, "spent_usd": 0.0}], G2_FAILED)}
    assert tc["TrackC"]["status"].startswith("not run") and tc["H9"]["status"].startswith("not_run_gate_failed (G2)")
    assert "H9_exploratory" in tc["TrackC"]["note"]
    assert itm.trackc_rows([], None)[0]["status"] == "not started" and itm.trackc_rows([], G2_PASSED)[0]["status"].startswith("may start")


def test_trackb_tables_absent_and_present_in_the_verdict_table():
    _, absent = verdict_map()
    assert absent["H4"]["verdict_interim"].startswith("running on the test split") and "H4:V3_vs_B1_source_attribution" not in absent
    c = {**contrasts(), **tb_rows()}
    rows = {r["key"]: r for r in itm.build_verdicts(c, UNITS, RULES, MIN_N, "test")["rows"]}
    assert rows["H4"]["verdict_interim"] == "supported" and rows["H6"]["verdict_interim"] == "supported"
    leaf = rows["H4:V3_vs_B1_source_attribution"]
    assert leaf["p_bonferroni"] == pytest.approx(0.008) and leaf["family_m"] == 8 and leaf["holm_status"] == itm.HOLM_PENDING
    assert rows["H4:V3_own_origin_mass_fraction"]["data_split"] == "n/a" and rows["H4:V0_own_origin_mass_ratio_to_uniform"]["verdict_interim"] == "supported"
    # H4's own-origin leaf below 0.90 is a refuted measurement; Holm-sensitive attribution is pending_final_holm
    c2 = {**contrasts(), **tb_rows(v3b1=(6.0, 2.0, 10.0, 0.01))}
    r2 = {r["key"]: r for r in itm.build_verdicts(c2, UNITS, RULES, MIN_N, "test")["rows"]}
    assert r2["H4:V3_vs_B1_source_attribution"]["verdict_interim"] == "pending_final_holm"
    low = {r["key"]: r for r in itm.build_verdicts({**contrasts(), **tb_rows(own=0.31)}, UNITS, RULES, MIN_N, "test")["rows"]}
    assert low["H4:V3_own_origin_mass_fraction"]["verdict_interim"] == "refuted" and low["H4"]["verdict_interim"] == "refuted"
    fam = {r["key"]: r for r in itm.build_verdicts(c, UNITS, RULES, MIN_N, "test")["family"]}
    assert fam["H4"]["status"] == "available" and fam["H6"]["p_raw"] == pytest.approx(0.001) and fam["H4"]["p_bonferroni"] == pytest.approx(0.008)


def test_stopping_rule_pinned_checkpoints_and_status_rows():
    def curve(vals):
        return itm.learning_curve_rows([("r", {"curve": [{"arm": "V3", "opt_step": s, "probe": {"own_origin_mass_fraction": f}} for s, f in vals.items()]})])
    lc = curve({0: 0.14, 200: 0.249, 400: 0.307, 600: 0.3204})
    sr = itm.stopping_rule(lc)
    assert sr["fired"] and sr["step"] == 600 and sr["previous_step"] == 400 and sr["rise"] == pytest.approx(0.0134, abs=1e-4)
    st = {r["key"]: r for r in itm.trackb_status_rows(lc)}
    assert st["V3_stopping_rule_fired_step"]["value"] == 600.0 and "fires" in st["V3_rise_last_two_evaluations"]["status"]
    assert not itm.stopping_rule(curve({200: 0.249, 400: 0.307}))["fired"]
    assert not itm.stopping_rule(curve({0: 0.1, 200: 0.1}))["fired"]                           # never before 400
    pins = {r["key"]: r for r in itm.pinned_checkpoint_rows(PROJECT_DIR)}
    if pins:                                                                                  # DEV-TRAIN-PIN of this repository
        assert pins["V3"]["final_step"] == 600 and len(pins["V3"]["sha256"]) == 64 and pins["B1"]["run_id"].startswith("B-B1-")


def test_end_to_end_g2_failed_and_trackb_present_paper_input(tmp_path):
    out, c, dev = make_bundle(tmp_path, "test", g2=G2_FAILED, trackb=tb_rows(), own={0: 0.14, 200: 0.249, 400: 0.307, 600: 0.3204})
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "test", args_runs={"h1-lb": ["A-x-test-s1-20261108000000-abcdef0-0001"]})
    text = md.read_text()
    assert pii.check_refs(md, out) == []
    assert "not yet decided" not in text and "it is pending" not in text and "learning curve in progress" not in text
    assert "Stopping rule: fired at step 600" in text and "G2: failed" in text and "G2 criteria, with the observed values (G2 failed)" in text
    assert "not_run_gate_failed (G2); owner request budget/requests/H9_exploratory.md" in text
    assert "### H4 " in text and "`V3_vs_B1_source_attribution`" in text and "G2 failed, so the model-level arm" in text
    gates = {r["key"]: r for r in read_rows(out / "tables" / "interim_gates.csv")}
    assert gates["G2"]["status"] == "failed"
    crit = [r for r in read_rows(out / "tables" / "interim_gate_criteria.csv") if r["gate"] == "G2"]
    assert len(crit) == 2 and crit[0]["observed_value"] == "0.314"
    tc = {r["key"]: r for r in read_rows(out / "tables" / "interim_trackc.csv")}
    assert tc["H9"]["status"].startswith("not_run_gate_failed")
    assert {r["key"] for r in read_rows(out / "tables" / "trackb_pinned_checkpoints_dev.csv")} >= set()


def test_end_to_end_g2_absent_and_passed_wording(tmp_path):
    out, c, dev = make_bundle(tmp_path / "a", "test")
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    text = pii.write_paper_input_interim(out, "test").read_text()
    assert "no G2 record exists yet" in text and "Stopping rule: not fired" in text
    out2, c2, dev2 = make_bundle(tmp_path / "p", "test", g2=G2_PASSED)
    ifig.make_figures(out2, "test", c2, dev2, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out2, "test")
    t2 = md.read_text()
    assert "G2: passed" in t2 and "pending_track_b (G2 passed)" in t2 and pii.check_refs(md, out2) == []


def test_trackb_rows_with_wrong_split_or_key_clash_are_refused(tmp_path):
    import argparse
    out = tmp_path / "b"
    (out / "tables" / "trackb").mkdir(parents=True)
    write_contrast_table(out / "tables" / "p2_contrasts_test.csv", [{**v, "key": k} for k, v in contrasts().items()], extra_columns=["split"])
    bad = tb_rows("dev")
    write_contrast_table(out / "tables" / "trackb" / "trackb_contrasts_test.csv", [{**v, "key": k} for k, v in bad.items()], extra_columns=["split"])
    root = tmp_path / "root"
    (root / "prereg").mkdir(parents=True)
    for f in ("rules.yaml", "sample_sizes.yaml"):
        shutil.copy(PROJECT_DIR / "prereg" / f, root / "prereg" / f)
    (root / "prereg" / "PREREG_TAG.json").write_text(json.dumps({"tag": "t", "commit": "c", "committed_at": "2026-11-07T18:59:43-05:00"}))
    (tmp_path / "r.yaml").write_text("h1-lb: []\n")
    a = argparse.Namespace(split="test", runs=str(tmp_path / "r.yaml"), out=str(out), runs_dir=None, creds=None, reuse_p2=True, dry_run=True,
                           no_dev_context=True, project_root=str(root))
    with pytest.raises(itm.InterimError, match="split"):
        itm.run(a, store=object())


def read_rows(path):
    from opa.analysis.tables import read_table
    return read_table(path)


def test_rounding_is_half_away_from_zero_and_exact_for_gate_values():
    assert pii.fmt_away(-65.625, 2) == "-65.63" and pii.fmt_away(65.625, 2) == "65.63" and pii.fmt_away(-0.004, 2) == "0.00"
    assert pii.fmt_away(-6.25, 1) == "-6.3" and pii.fmt_away(0.0, 3) == "0.000"
    assert pii.exact_decimals(-65.625) == 3 and pii.exact_decimals(0.314) == 3 and pii.exact_decimals(2.0) == 2 and pii.exact_decimals(1.23456789) == 6
    g2 = {**G2_FAILED, "criteria": [{"name": "difference", "threshold": "> 0", "observed": {"difference_points": -65.625}, "met": False}]}
    out, c, dev = make_bundle(tmp_path_factory_dir(), "test", g2=g2)
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "test")
    assert "-65.625 [[ref:gates/G2.json#criteria.0.observed.difference_points]]" in md.read_text()
    assert pii.check_refs(md, out) == []


def tmp_path_factory_dir():
    import tempfile
    return Path(tempfile.mkdtemp(prefix="interim_t_"))


def test_p_values_below_display_precision_and_exact_zero(tmp_path):
    t = tmp_path / "tables"
    t.mkdir()
    write_table(t / "x.csv", [{"key": "a", "p": 0.0}, {"key": "b", "p": 1e-9}, {"key": "c", "p": 0.0123}, {"key": "d", "p": 6e-7}], columns=["key", "p"])
    d = pii.Doc(tmp_path, "test")
    assert d.pnum("x.csv", "a", "p").startswith("0.000000 [[ref:tables/x.csv#row=a&cols=p]]") and "`(< 0.000001)`" in d.pnum("x.csv", "a", "p")
    assert "`(< 0.000001)`" in d.pnum("x.csv", "b", "p")
    assert "< 0.000001" not in d.pnum("x.csv", "c", "p") and d.pnum("x.csv", "c", "p").startswith("0.012")
    assert "< 0.000001" not in d.pnum("x.csv", "d", "p")


def test_stopping_rule_reports_the_cap():
    def curve(vals):
        return itm.learning_curve_rows([("r", {"curve": [{"arm": "V3", "opt_step": s, "probe": {"own_origin_mass_fraction": f}} for s, f in vals.items()]})])
    vals = {s: 0.1 + 0.05 * i for i, s in enumerate(range(0, 2001, 200))}                    # always rising by 0.05
    sr = itm.stopping_rule(curve(vals))
    assert not sr["fired"] and sr["capped"] and sr["step"] == 2000
    st = {r["key"]: r for r in itm.trackb_status_rows(curve(vals))}
    assert st["V3_stopped_at_cap"]["status"] == "stopped at the 2,000-step cap" and "cap" in st["V3_rise_last_two_evaluations"]["status"]
    assert not itm.stopping_rule(curve({s: v for s, v in vals.items() if s <= 1000}))["capped"]
    out, c, dev = make_bundle(tmp_path_factory_dir(), "test", own=vals)
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "test")
    assert "training stopped at the 2,000-step cap" in md.read_text() and pii.check_refs(md, out) == []


# ------------------------------------------------------------------------------------ final pre-delivery fixes
def make_full(tmp_path, h8_rates=(0.0, 0.004), s8_p=0.5):
    c = contrasts()
    c[K["h8"]] = strings(row(K["h8"], -0.4, -1.2, 0.0, MIN_N[K["h8"]], None, "test", em_a=h8_rates[0], em_b=h8_rates[1], note="M=4; recall"))
    c[K["s8"]] = strings(row(K["s8"], 3.0, 1.0, 5.0, MIN_N[K["s8"]], s8_p, "test"))
    return c


def build_doc(tmp_path, c, g2=G2_FAILED, trackb=None, series=None):
    out, _, dev = make_bundle(tmp_path, "test", g2=g2, trackb=trackb)
    write_contrast_table(out / "tables" / "p2_contrasts_test.csv", [{**v, "key": k} for k, v in c.items()],
                         extra_columns=["split", "systems", "em_a", "em_b", "n_itt_imputed", "n_excluded_missing", "note"])
    (out / "tables" / "p2_dev_check").mkdir()
    write_contrast_table(out / "tables" / "p2_dev_check" / "p2_contrasts_dev.csv",
                         [{**row(K["h8"], 0.0, 0.0, 0.0, 50, None, "dev", em_a=0.0, em_b=0.0), "key": K["h8"]}], extra_columns=["split", "em_a", "em_b"])
    allc = {**c, **(trackb or {})}
    itm.write_all_tables(out, "test", contrasts=allc, units=UNITS, rules=RULES, min_n=MIN_N, curves=[], dev_rows=dev, repo=PROJECT_DIR,
                         approvals=APPR, ledger=LEDGER)
    ifig.make_figures(out, "test", allc, dev, repo=PROJECT_DIR)
    md = pii.write_paper_input_interim(out, "test", args_runs=series or {"h1-lb": ["A-x-test-s1-20261108000000-abcdef0-0001"]})
    return out, md


def test_floor_disclosure_for_non_inferiority_at_floor(tmp_path):
    out, md = build_doc(tmp_path, make_full(tmp_path))
    t = md.read_text()
    assert "Floor disclosure" in t and "non-inferiority at floor and says nothing about routing quality" in t
    assert "the number of scores is not compared" in t and "dev pilot 0.000 [[ref:tables/p2_dev_check/p2_contrasts_dev.csv" in t
    assert "0 and 1 of" in t and "arm recalls descent" in t and pii.check_refs(md, out) == []
    rows = {r["key"]: r for r in read_rows(out / "tables" / "interim_verdicts.csv")}
    assert "at floor" in rows["H8:" + K["h8"]]["notes"] and "at floor" in rows["H8:" + K["h8"]]["verdict_reason"]
    out2, md2 = build_doc(tmp_path / "n", make_full(tmp_path, h8_rates=(0.6, 0.62)))          # arms well above the floor: no disclosure
    assert "Floor disclosure" not in md2.read_text()


def test_h3_reasoning_provisional_holm_and_not_funded_labels(tmp_path):
    out, md = build_doc(tmp_path, make_full(tmp_path))
    t = md.read_text(encoding="utf-8")
    assert "H3 cannot be supported because the R2 leaf has no result (closure item 9); the verdict is refuted because the S11 vs S8 leg is refuted (frozen all_of rule)" in t
    assert "VERDICT: inconclusive (not funded)" in t.split("### H2 — ")[1].split("\n")[0]
    assert "VERDICT: not_run_gate_failed (G2)" in t.split("### H13 — ")[1].split("\n")[0]
    fam = {r["key"]: r for r in read_rows(out / "tables" / "interim_family.csv")}
    from opa import stats
    ps = {h: float(fam[h]["p_raw"]) for h in fam if fam[h]["p_raw"]}
    assert "H2" not in ps and "H12" not in ps and {"H1", "H3", "H5", "H10"} <= set(ps)
    for h, v in stats.holm(ps).items():
        assert float(fam[h]["p_holm_provisional"]) == pytest.approx(v)
    assert fam["H2"]["p_holm_provisional"] == "" and fam["H12"]["p_holm_provisional"] == ""
    assert "provisional Holm" in t and "until the bundle is assembled" in t and "H2 (inconclusive (not funded))" in t
    assert "provisional Holm p = " in t and "final Holm pending Tracks B/C" in t


def test_h4_disclosure_v0_reason_run_counts_and_model_leaf_columns(tmp_path):
    tb = tb_rows()
    del tb["V0_own_origin_mass_ratio_to_uniform"]
    out, md = build_doc(tmp_path, make_full(tmp_path), trackb=tb)
    write_table(out / "tables" / "trackb" / "trackb_series_test.csv",
                [{"key": "h4-v3", "run_ids": "B-x-test-s1-20261108050000-abcdef0-0001;B-x-test-s1-20261108050001-abcdef0-0002"},
                 {"key": "h4-b1", "run_ids": "B-y-test-s1-20261108050002-abcdef0-0003"}], columns=["key", "run_ids"])
    md = pii.write_paper_input_interim(out, "test", args_runs={"h1-lb": ["A-x-test-s1-20261108000000-abcdef0-0001", "A-x-test-s1-20261108000001-abcdef0-0002"]})
    t = md.read_text()
    assert "all 5 test-split runs (2 Track A + 3 Track B) started after it" in t
    assert "text origin headers" in t and "registration Section 12 item 22" in t
    assert "training-free test run not yet complete" in t
    rows = {r["key"]: r for r in read_rows(out / "tables" / "interim_verdicts.csv")}
    leaf = rows["H5:A2_vs_A1_model_thread_confusion_task"]
    assert leaf["level"] == "model" and leaf["data_split"] == "n/a"
    assert rows["H4:V3_vs_B1_source_attribution"]["level"] == "model" and rows["H4:V3_vs_B1_source_attribution"]["data_split"] == "test"
    assert pii.check_refs(md, out) == []


def test_failed_runs_section_and_sensitivity(tmp_path, monkeypatch):
    import yaml as _y
    runs = tmp_path / "runs"
    first = "A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9"
    retry = "A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108055403-10a14e2-6c9a"
    err = "Traceback in upload\n...\n: ('Request failed with status code', 412, 'Expected one of')\n"
    for rid, st, e in ((first, "failed", err), (retry, "ok", None)):
        d = runs / rid
        d.mkdir(parents=True)
        (d / "job.yaml").write_text(_y.safe_dump({"name": "p2-h1-lb-c07", "split": "test", "args": {"systems": ["S11", "S2"]}}))
        (d / "result.json").write_text(json.dumps({"status": st, "error": e}))
    items = {f"i{k}": {"item_id": f"i{k}"} for k in range(10)}
    monkeypatch.setattr(itm.pl, "load_run_items", lambda store, job: items)
    monkeypatch.setattr(itm.p2, "_load_job", lambda rd, rid, split: {"args": {"systems": ["S11", "S2"]}})

    def recs(store, rid, system):
        return {f"i{k}": {"item_id": f"i{k}"} for k in range(10) if rid != first or k < 8}
    monkeypatch.setattr(itm, "_records_by_item", recs)
    monkeypatch.setattr(itm.p2, "_em", lambda item, rec: (0.0 if (rec["item_id"] == "i0" and False) else 1.0, None))
    rows, sens = itm.failed_run_rows(object(), runs, [{"run_id": first, "replaced_by": retry},
                                                      {"run_id": "B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108022824-e36fdc0-201d"}])
    r0 = rows[0]
    assert r0["status"] == "failed" and "HTTP 412" in r0["reason"] and "retry 2 of 3" in r0["handling"] and r0["attempt"] == 1
    assert "pre-fix code" in rows[1]["handling"]
    assert {x["system"] for x in sens} == {"S11", "S2"} and all(x["n_shared"] == 8 for x in sens)
    assert sens[0]["diff_points"] == pytest.approx(0.0)
    out, _, dev = make_bundle(tmp_path / "b", "test")
    itm.write_all_tables(out, "test", contrasts=contrasts(), units=UNITS, rules=RULES, min_n=MIN_N, curves=[], dev_rows=dev, repo=PROJECT_DIR,
                         approvals=APPR, ledger=LEDGER, failed=(rows, sens))
    md = pii.write_paper_input_interim(out, "test")
    t = md.read_text()
    assert "## Failed and excluded runs" in t and "Sensitivity S11" in t and "pre-fix code" in t
    assert pii.check_refs(md, out) == []


def test_discover_lists_failed_runs_with_their_replacement(tmp_path):
    import yaml as _y
    runs = tmp_path / "runs"
    for rid, st in (("A-x-test-s1-20261108043839-abcdef0-0001", "failed"), ("A-x-test-s1-20261108055403-abcdef1-0002", "ok")):
        d = runs / rid
        d.mkdir(parents=True)
        (d / "job.yaml").write_text(_y.safe_dump({"name": "p2-h1-lb-c07", "split": "test"}))
        (d / "result.json").write_text(json.dumps({"status": st}))
    d = itm.discover_series(runs, "test")
    assert d["failed"] == [{"run_id": "A-x-test-s1-20261108043839-abcdef0-0001", "job": "p2-h1-lb-c07",
                            "replaced_by": "A-x-test-s1-20261108055403-abcdef1-0002"}]
    assert d["interim"]["failed_runs"] == d["failed"]


def test_figure_sources_title_cost_group_and_citation(tmp_path):
    out, c, dev = make_bundle(tmp_path, "test", g2=G2_FAILED)
    ifig.make_figures(out, "test", c, dev, repo=PROJECT_DIR)
    src = (out / "tables" / "fig_h1_depth_context_test_with_dev_pilot.csv").read_text()
    assert "tables/p2_dev_check/p2_contrasts_dev.csv#row=" in src and "tables/p2_contrasts_test.csv#row=" in src
    assert "tables/h5_distortion_dev.csv#row=" in (out / "tables" / "fig_h5_thread_depth_test_with_dev_pilot.csv").read_text()
    import pandas as pd
    df = pd.DataFrame([{"arm": "V3", "opt_step": 0, "own_origin_mass_fraction": 0.1, "attrib_em": 0.1, "passkey_em": 0.1, "train_loss": 3.0,
                        "attrib_chance": 0.1}])
    assert "G2 failed" in ifig.plot_learning_curve(df, "failed")._suptitle.get_text()
    assert "G2 pending" in ifig.plot_learning_curve(df, None)._suptitle.get_text()
    assert itm.group_of("p4-h6-lb-c01")[0] == "Track B: training, G2 evals, H4/H6 test runs" and "H6" in itm.group_of("p4-h6-mt-c02")[1]
    assert itm.group_of("p4-g2-eval-v3")[0] == itm.group_of("p4-h4-test-b1-c01")[0]
    from opa.analysis import release as rel
    cff = rel.citation_cff("2026-11-08", "failed", True)
    assert "G2 failed" in cff and "H4 / H6 test results included" in cff and "Track C not run" in cff
    assert "learning curve so far" in rel.citation_cff("2026-11-08")
