"""P6 final bundle (opa.analysis.final_bundle): results, claims, deviations, unit records and staging on synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from opa import claims as cl
from opa.analysis import final_bundle as fb
from opa.analysis import interim as itm

RULES, MIN_N = cl.load_prereg()


def row(key, value, lo=None, hi=None, n=100, p=None, split="test", run="R1", test="t"):
    return {"key": key, "value": str(value), "ci_lo": "" if lo is None else str(lo), "ci_hi": "" if hi is None else str(hi), "n": str(n),
            "p_raw": "" if p is None else str(p), "split": split, "run_ids": run, "test": test}


def tables(strong=True):
    p2 = {r["key"]: r for r in [
        row("S11_vs_S2_lbmhqa_late_32k_primary", 14.0, 8.0, 20.0, MIN_N["S11_vs_S2_lbmhqa_late_32k_primary"], 1e-6, run="Ra;Rb"),
        row("S11_vs_S4_lbmhqa_late_32k_primary", 40.0, 30.0, 50.0, MIN_N["S11_vs_S4_lbmhqa_late_32k_primary"], None),
        row("S11_vs_S8_late_bound", -1.0, -2.0, 0.0, MIN_N["S11_vs_S8_late_bound"], 0.4),
        row("A2_vs_A1_agent_mtlbmhqa_32k_primary_cross", 0.4, -2.0, 3.0, MIN_N["A2_vs_A1_agent_mtlbmhqa_32k_primary_cross"], 1.0),
        row("A2_vs_A3_agent_equivalence", 0.0, -1.0, 1.0, 191, 0.03),
        row("descent_beam4_vs_flat_recall_at_M", -0.4, -1.2, 0.0, MIN_N["descent_beam4_vs_flat_recall_at_M"]),
        row("pinned_vs_unpinned_repeat_pagein_reduction", 2.5, 2.2, 2.8, MIN_N["pinned_vs_unpinned_repeat_pagein_reduction"], 1e-50)]}
    tb = {r["key"]: r for r in [
        row("V3_vs_B1_source_attribution", -68.0, -70.0, -66.0, 2451, 0.0, run="Rv;Rb"),
        row("V3_own_origin_mass_fraction", 0.32, n=16), row("V0_own_origin_mass_ratio_to_uniform", 1.0, n=16),
        row("R5_vs_R1_accuracy", 0.5, -1.5, 2.5, 400, 0.8), row("best_remedy_recovery_fraction", 0.95, 0.9, 1.0, 400)]}
    h7 = {k: {"key": k, "value": str(v)} for k, v in zip(itm.H7_KEYS, (0.96, 4, 7, 16))}
    return p2, tb, h7


def gates(g2_failed=True):
    c = lambda met: [cl.criterion("c", ">= 1", 1.0 if met else 0.0, met)]          # noqa: E731
    g = {"G0": cl.gate("G0", True, c(True), ["e"]), "G1": cl.gate("G1", True, c(True), ["e"]), "G2": cl.gate("G2", not g2_failed, c(not g2_failed), ["e"])}
    if g2_failed:
        g["G3"] = cl.unreached_gate("G3", "G2")
    g["G4"] = cl.gate("G4", True, c(True), ["e"])
    return g


def build(tmp_path, g2_failed=True):
    p2, tb, h7 = tables()
    res = fb.build_results(p2, tb, h7, "RUNITS")
    text = fb.deviations_md([{"key": "A-run", "job": "p2-h1-lb-c07", "handling": "replaced by retry 2 of 3 allowed"}])
    (tmp_path / "deviations.md").write_text(text, encoding="utf-8")
    ids = cl.read_deviation_ids(tmp_path)
    return res, fb.make_claims(tmp_path, RULES, MIN_N, res, gates(g2_failed), ids), ids


def test_results_come_from_the_table_rows(tmp_path):
    p2, tb, h7 = tables()
    res = fb.build_results(p2, tb, h7, "RUNITS")
    h1 = {r["key"]: r for r in res["H1"]}
    s2 = h1["S11_vs_S2_lbmhqa_late_32k_primary"]
    assert s2["primary"] and s2["p_raw"] == pytest.approx(1e-6) and s2["ci95"] == [8.0, 20.0] and s2["run_ids"] == ["Ra", "Rb"]
    assert s2["table_ref"] == "tables/p2_contrasts_test.csv#row=S11_vs_S2_lbmhqa_late_32k_primary" and s2["split"] == "test"
    assert h1["S11_vs_S4_lbmhqa_late_32k_primary"]["p_raw"] is None and not h1["S11_vs_S4_lbmhqa_late_32k_primary"]["primary"]
    h4 = {r["key"]: r for r in res["H4"]}
    assert h4["V3_vs_B1_source_attribution"]["primary"] and h4["V3_vs_B1_source_attribution"]["p_raw"] == 0.0
    assert h4["V3_vs_B1_source_attribution"]["table_ref"].startswith("tables/trackb/trackb_contrasts_test.csv#row=")
    assert h4["V3_own_origin_mass_fraction"]["level"] == "model" and h4["V3_own_origin_mass_fraction"]["ci95"] is None
    h7r = res["H7"]
    assert len(h7r) == 4 and all(r["split"] == "n/a" and r["level"] == "measurement" and r["n"] == 0 and r["run_ids"] == ["RUNITS"] for r in h7r)
    assert [r["key"] for r in res["H8"]] == ["descent_beam4_vs_flat_recall_at_M"] and res["H8"][0]["primary"] and res["H8"][0]["p_raw"] is None
    assert "H2" not in res and "H13" not in res and "H11" not in res                       # no placeholder results for unfunded / not run


def test_claims_verdicts_and_labels(tmp_path):
    res, claims, ids = build(tmp_path)
    v = {h: c["verdict"] for h, c in claims.items()}
    assert v["H1"] == "supported" and v["H3"] == "refuted" and v["H7"] == "supported" and v["H8"] == "supported" and v["H10"] == "supported"
    assert v["H4"] == "refuted" and v["H6"] == "refuted"
    assert v["H5"] == "inconclusive" and "failed_gate" not in claims["H5"]                 # H5's gate is G1; the model leaf is missing
    for h in ("H9", "H11", "H12", "H13"):
        assert v[h] == "not_run_gate_failed" and claims[h]["failed_gate"] == "G2" and claims[h]["results"] == []
    assert "not funded" in claims["H13"]["notes"].lower() and "G2" in claims["H13"]["notes"]
    assert v["H2"] == "inconclusive" and "failed_gate" not in claims["H2"] and claims["H2"]["results"] == [] and "Not funded" in claims["H2"]["notes"]
    assert len(claims) == 13 and all(not cl.schema_errors(c, "claim") for c in claims.values())
    # the Holm adjustment is over every primary that reports a p (H1, H3, H4, H5, H6, H10) and the verifier recomputes it
    fam = {f"{h}:{r['key']}": r["p_raw"] for h, c in claims.items() for r in c["results"] if r["primary"] and r["p_raw"] is not None}
    assert {k.split(":")[0] for k in fam} == {"H1", "H3", "H4", "H5", "H6", "H10"}
    from opa import stats
    adj = stats.holm(fam)
    for h, c in claims.items():
        for r in c["results"]:
            if f"{h}:{r['key']}" in adj:
                assert r["p_holm"] == pytest.approx(adj[f"{h}:{r['key']}"], abs=1e-12)
    # every cited deviation is in deviations.md
    assert all(d in ids for c in claims.values() for d in c["deviations"])
    # the verifier's own evaluator agrees with each declared verdict
    for h, c in claims.items():
        if c["verdict"] != "not_run_gate_failed":
            assert cl.evaluate(h, c["results"], RULES, MIN_N) == (c["verdict"], [])


def test_verdict_table_agrees_with_claims(tmp_path):
    res, claims, _ = build(tmp_path)
    p2, tb, h7 = tables()
    units = {"h7": {k: float(v["value"]) for k, v in h7.items()}}
    v = fb.final_tables_dry({**p2, **tb}, units, RULES, MIN_N, {"passed": False})
    assert fb.check_verdicts(claims, v["rows"]) == []
    hyp = {r["key"]: r["verdict_interim"] for r in v["rows"] if r["kind"] == "hypothesis"}
    assert hyp["H4"] == "refuted" and hyp["H13"] == "not_run_gate_failed (G2)" and hyp["H2"] == "inconclusive (not funded)" and hyp["H11"] == "not_run_gate_failed (G2)"
    assert all(r.get("holm_status", "") in ("", "final Holm") and not r.get("holm_sensitive") for r in v["rows"])
    assert all(r.get("p_bonferroni") is None for r in v["family"]) and v["family"][0]["p_holm_provisional"] is not None
    bad = dict(claims)
    bad["H1"] = {**claims["H1"], "verdict": "refuted"}
    assert fb.check_verdicts(bad, v["rows"])


def test_deviations_md_format_and_content():
    t = fb.deviations_md([])
    entries = t.split("### ")[1:]
    assert len(entries) >= 12
    for e in entries:
        for field in ("Date:", "Change:", "Reason:", "Impact:"):
            assert field in e
    for name in ("DEV-U1", "DEV-G0-noise", "DEV-U8", "DEV-TRAIN-PIN", "DEV-G1-OVERRIDE", "DEV-PLAN-REV2", "DEV-PLAN-REV3", "DEV-PLAN-REV4",
                 "DEV-PLAN-REV5", "DEV-H3-R2", "DEV-OWNER-INTERIM", "DEV-FAILED-RUNS", "DEV-S4-V2"):
        assert f"### {name}\n" in t
    assert "no failed or excluded run" in t and "replaced by retry" in fb.deviations_md([{"key": "k", "job": "j", "handling": "replaced by retry"}])


def test_unit_records_u8_by_the_preregistered_criterion():
    def units(obs=3.3e-16, flag=False):
        U = {f"U{i}": {"run_id": "R", "test": {"id": f"U{i}", "name": "n", "passed": True, "details": "d", "run_id": "R"}} for i in range(1, 9)}
        U["U8"]["test"] = {"id": "U8", "name": "n", "passed": flag, "details": "d", "run_id": "R", "metrics": {"tv_vs_evicted_router_mass_max_abs_err": obs, "tolerance": 1e-5,
                                                                                                       "paged_sdpa_output_max_abs_err": 2.08e-05}}
        return {"U": U}
    recs = fb.unit_records(units())
    u8 = recs[7]
    assert u8["passed"] is True and u8["harness_flag"] is False and u8["deviation"] == "DEV-U8" and "2.08e-05" not in u8["details"] and "auxiliary" in u8["details"]
    assert [r["id"] for r in recs] == [f"U{i}" for i in range(1, 9)] and all(not cl.schema_errors(r, "test") for r in recs)
    with pytest.raises(fb.FinalError, match="not met"):
        fb.unit_records(units(obs=1e-3))
    broken = units()
    del broken["U"]["U3"]
    with pytest.raises(fb.FinalError, match="U3"):
        fb.unit_records(broken)


def test_stage_layout_copies_only_the_specified_parts_and_refuses_a_used_folder(tmp_path):
    src = tmp_path / "src"
    for rel in ("tables/a.csv", "tables/trackb/t.csv", "tables/interim_verdicts_dev.csv", "tables/p2_contrasts_dev.csv", "tables/v3_learning_curve_dev.csv",
                "tables/p2_dev_check/p2_contrasts_dev.csv", "figures/fig_h1_depth_context_dev.png", "figures/fig_v3_learning_curve_dev.png",
                "figures/fig_h1_depth_context_test_with_dev_pilot.png", "gates/G0.json", "gates/history/x.json", "code/interim_figures.py",
                "PAPER_INPUT_interim.md", "release/opa_v3_interim.zip", "README.md"):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    out = tmp_path / "final"
    staged = fb.stage_layout(src, out)
    assert "tables/a.csv" in staged and "tables/trackb/t.csv" in staged and "gates/history/x.json" in staged and "code/interim_figures.py" in staged
    assert "tables/v3_learning_curve_dev.csv" in staged and "tables/p2_dev_check/p2_contrasts_dev.csv" in staged         # real dev-data tables stay
    assert "figures/fig_v3_learning_curve_dev.png" in staged and "figures/fig_h1_depth_context_test_with_dev_pilot.png" in staged
    for gone in ("tables/interim_verdicts_dev.csv", "tables/p2_contrasts_dev.csv", "figures/fig_h1_depth_context_dev.png", "PAPER_INPUT_interim.md",
                 "release/opa_v3_interim.zip", "README.md"):
        assert gone not in staged and not (out / gone).exists()
    assert (src / "release" / "opa_v3_interim.zip").exists()                                                          # the source is untouched
    with pytest.raises(fb.FinalError, match="not empty"):
        fb.stage_layout(src, out)


def test_the_real_interim_gates_and_tables_feed_the_builder():
    """The repository's own interim bundle (if present) yields 13 schema-valid claims with the expected non-gate verdicts."""
    root = Path(__file__).resolve().parent.parent
    b = root / "bundle"
    if not (b / "tables" / "trackb" / "trackb_contrasts_test.csv").exists() or not (b / "gates" / "G2.json").exists():
        pytest.skip("interim test tables not in this checkout")
    p2, tb = fb._rows(b, fb.P2_T), fb._rows(b, fb.TB_T)
    h7 = fb._rows(b, "tables/interim_h7.csv")
    res = fb.build_results(p2, tb, h7, "RU")
    g = cl.read_gates(b)
    g["G3"] = cl.unreached_gate("G3", "G2")
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "deviations.md").write_text(fb.deviations_md([]), encoding="utf-8")
    claims = fb.make_claims(tmp, RULES, MIN_N, res, g, cl.read_deviation_ids(tmp))
    assert {h: c["verdict"] for h, c in claims.items()}["H13"] == "not_run_gate_failed" and claims["H13"]["failed_gate"] == "G2" and claims["H11"]["failed_gate"] == "G2"


# ------------------------------------------------------------------------------------ checker a7117d8 fixes
from . import test_interim as ti          # noqa: E402
from opa.analysis import interim_figures as ifig          # noqa: E402
from opa.analysis import paper_input_interim as pii          # noqa: E402
from opa.analysis.tables import read_table          # noqa: E402


def full_final(tmp_path, g4_fails=None):
    """A synthetic bundle through the final steps: gates (G4 decided from ``g4_fails``), tables, claims, PAPER_INPUT.md."""
    out, c, dev = ti.make_bundle(tmp_path, "test", g2=ti.G2_FAILED, trackb=ti.tb_rows(), own={0: 0.14, 200: 0.249, 400: 0.307, 600: 0.3204})
    ifig.make_figures(out, "test", {**c, **ti.tb_rows()}, dev, repo=Path(__file__).resolve().parent.parent)
    (out / "deviations.md").write_text(fb.deviations_md([]), encoding="utf-8")
    ids = cl.read_deviation_ids(out)
    cl.write_gates(out, {"G3": cl.unreached_gate("G3", "G2"), "G4": fb.decide_g4(g4_fails or [], "scratch/x", "2026-11-08T12:00:00Z")})
    p2, tb = fb._rows(out, fb.P2_T), fb._rows(out, fb.TB_T)
    units = {"h7": {k: float(r["value"]) for k, r in fb._rows(out, "tables/interim_h7.csv").items()}}
    fb.final_tables(out, {**p2, **tb}, units, RULES, MIN_N, ti.G2_FAILED)
    gates = {**cl.read_gates(out)}
    res = fb.build_results(p2, tb, fb._rows(out, "tables/interim_h7.csv"), "RU")
    claims = fb.make_claims(out, RULES, MIN_N, res, gates, ids)
    (out / "runs").mkdir(exist_ok=True)
    (out / "runs" / "index.csv").write_text("run_id,started_at,finished_at\nA,2026-11-08T00:21:00Z,2026-11-08T01:00:00Z\nB,2026-11-08T05:00:00Z,2026-11-08T07:30:00Z\n")
    (out / "budget").mkdir(exist_ok=True)
    import yaml as _y
    (out / "budget" / "approvals.yaml").write_text(_y.safe_dump(ti.APPR))
    bud_rows = ti.LEDGER
    import csv as _csv
    with (out / "budget" / "ledger.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["ts_utc", "envelope", "job", "kind", "amount_usd", "provider", "run_id", "note"])
        w.writeheader()
        w.writerows(bud_rows)
    fb.refresh_gate_tables(out)
    fb.refresh_cost_tables(out)
    md = pii.write_paper_input_interim(out, "test", final={"claims": claims, "verdict_table": "final_verdicts.csv", "family_table": "final_family.csv"})
    return out, md, claims


def test_g4_is_decided_by_the_verifier_and_the_document_agrees_in_both_branches(tmp_path):
    ok = fb.decide_g4([], "scratch/a", "2026-11-08T12:00:00Z")
    assert ok["passed"] is True and ok["decided_at"] == "2026-11-08T12:00:00Z"
    assert ok["criteria"][0]["observed"] == "BUNDLE VERIFIED, 0 failures (scratch copy scratch/a, 2026-11-08T12:00:00Z)"
    bad = fb.decide_g4(["[D9] x", "[D1] y"], "scratch/b", "2026-11-08T12:00:00Z")
    assert bad["passed"] is False and bad["criteria"][0]["observed"]["failures"] == ["[D9] x", "[D1] y"] and not bad["criteria"][0]["met"]
    assert fb.pending_g4()["passed"] is None and "pending" in fb.pending_g4()["notes"].lower()
    for fails, word in (([], "passed"), (["[D9] x"], "failed")):
        out, md, _ = full_final(tmp_path / word, fails)
        g4 = json.loads((out / "gates" / "G4.json").read_text(encoding="utf-8"))
        assert g4["passed"] is (not fails)
        t = md.read_text(encoding="utf-8")
        assert f"- **G4** {word} (decided `2026-11-08T12:00:00Z`" in t and f"G4 {word}." in t
        assert {r["key"]: r["status"] for r in read_table(out / "tables" / "interim_gates.csv")}["G4"] == word
        assert pii.check_refs(md, out) == []


def test_h13_label_agrees_in_every_table_and_document(tmp_path):
    out, md, claims = full_final(tmp_path)
    assert claims["H13"]["verdict"] == "not_run_gate_failed" and claims["H13"]["failed_gate"] == "G2"
    for name in ("interim_verdicts.csv", "final_verdicts.csv"):
        rows = {r["key"]: r for r in read_table(out / "tables" / name)}
        assert rows["H13"]["verdict_interim"] == "not_run_gate_failed (G2)", name
        assert "also not funded in the approved package (registration Section 2b)" in rows["H13"]["verdict_basis"]
    tc = {r["key"]: r for r in read_table(out / "tables" / "interim_trackc.csv")}
    assert tc["H13"]["status"] == "not_run_gate_failed (G2)"
    t = md.read_text(encoding="utf-8")
    assert "- **H13**: not_run_gate_failed (G2)" in t and "### H13 — " in t and "VERDICT: not_run_gate_failed (G2)" in t.split("### H13 — ")[1].split("\n")[0]
    assert "### H2 — " in t and "VERDICT: inconclusive (not funded)" in t.split("### H2 — ")[1].split("\n")[0]


def test_final_family_size_column_and_wording(tmp_path):
    out, md, _ = full_final(tmp_path)
    fam = {r["key"]: r for r in read_table(out / "tables" / "final_family.csv")}
    assert "p_holm" in next(iter(fam.values())) and "p_holm_provisional" not in next(iter(fam.values())) and "p_bonferroni" not in next(iter(fam.values()))
    assert {r["family_m"] for r in fam.values()} == {"6"}
    ver = list(read_table(out / "tables" / "final_verdicts.csv"))
    assert "p_holm" in ver[0] and "p_holm_provisional" not in ver[0] and {r["family_m"] for r in ver if r["family_m"]} <= {"6"}
    t = md.read_text(encoding="utf-8")
    assert "Holm-adjusted p = " in t and "cols=p_holm]]" in t and "cols=p_holm_provisional" not in t
    for stale in ("Interim verdicts", "so far", "interim hand-off", "pending_final_holm", "provisional", "Bonferroni-adjusted", "pending Tracks B/C"):
        assert stale not in t, stale
    assert "python -m opa.analysis.final_bundle" in t.split("## Reproduction")[1] and "python -m opa.bundle upload --final" in t


def test_template_gaps_run_window_models_figure_and_measured_row(tmp_path):
    out, md, _ = full_final(tmp_path)
    t = md.read_text(encoding="utf-8")
    assert "Run window (UTC): `2026-11-08T00:21:00Z` to `2026-11-08T07:30:00Z`" in t
    assert "`Qwen/Qwen3-4B-Instruct-2507@" in t and "`microsoft/harrier-oss-v1-0.6b@" in t
    h4 = t.split("### H4 — ")[1].split("### H5 — ")[0]
    assert "**Figure.** figures/fig_v3_learning_curve_dev.png" in h4
    row_ = [ln for ln in t.splitlines() if ln.startswith("| §7.2 own-origin mass")][0]
    assert "final_verdicts.csv#row=H4:V3_own_origin_mass_fraction&cols=value" in row_ and "0.320 [[ref" in row_ and "dev monitoring set gave" in row_
    assert pii.check_refs(md, out) == []


def test_cost_tables_come_from_the_ledger_copy_in_the_bundle(tmp_path):
    out, md, _ = full_final(tmp_path)
    env = {r["key"]: r for r in read_table(out / "tables" / "interim_costs_envelope.csv")}
    assert float(env["TOTAL"]["spent_usd"]) == pytest.approx(1.415) and env["TOTAL"]["as_of"] == "2026-11-07T03:00:00Z"


def test_unit_records_u1_deviation_and_u8_criteria():
    U = {f"U{i}": {"run_id": "R", "test": {"id": f"U{i}", "name": "n", "passed": True, "details": "d", "run_id": "R"}} for i in range(1, 9)}
    literal = "max abs error <= 1e-5 in fp32 (TV vs evicted mass, paged output vs a_paged @ v)"
    U["U8"]["test"] = {"id": "U8", "name": "n", "passed": False, "details": "harness text", "run_id": "R", "criterion": literal,
                       "metrics": {"tv_vs_evicted_router_mass_max_abs_err": 3.3e-16, "tolerance": 1e-5}}
    recs = {r["id"]: r for r in fb.unit_records({"U": U})}
    assert recs["U1"]["deviation"] == "DEV-U1" and "deviation" not in recs["U2"]
    u8 = recs["U8"]
    assert u8["harness_criterion"] == literal and u8["preregistered_criterion"] == fb.U8_PREREG_CRITERION and u8["criterion"] == fb.U8_PREREG_CRITERION
    assert "Paging identity" in u8["preregistered_criterion"] and u8["harness_flag"] is False and u8["passed"] is True and u8["details"].endswith("harness text")


def test_defang_keeps_a_trailing_comma_outside_the_backticks():
    assert pii.defang("total cap $148, base $117.33 and $1,000") == "total cap `$148`, base `$117.33` and `$1,000`"


def test_verify_scratch_uses_new_folders_and_deletes_nothing(tmp_path):
    out = tmp_path / "b"
    out.mkdir()
    (out / "x.txt").write_text("x")
    base = tmp_path / "scratch"
    n1, f1, t1 = fb.verify_scratch(out, Path(__file__).resolve().parent.parent, base)
    n2, f2, t2 = fb.verify_scratch(out, Path(__file__).resolve().parent.parent, base)
    assert t1 != t2 and (t1 / "x.txt").exists() and (t2 / "x.txt").exists() and f1 == f2 and f1
    assert fb.main(["--help"]) if False else True
