"""Tests for tools/verify_bundle.py: a valid fixture passes; each single kind of tampering is caught with the right code."""
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import make_fixture  # noqa: E402
import verify_bundle  # noqa: E402


@pytest.fixture()
def fx(tmp_path):
    b = make_fixture.build(tmp_path / "repo")
    return b


def run(b):
    rep = verify_bundle.verify(b, b.parent)
    return rep.fails


def rehash(b):
    make_fixture.manifest(b)


def edit_json(p, fn):
    obj = json.loads(p.read_text())
    fn(obj)
    p.write_text(json.dumps(obj, indent=2))


def codes(fails):
    return {f.split("]")[0].strip("[") for f in fails}


def test_valid_fixture_passes(fx):
    assert run(fx) == []


def test_spun_verdict_is_caught(fx):
    edit_json(fx / "claims/H2.json", lambda o: o.update(verdict="supported"))
    rehash(fx)
    fails = run(fx)
    assert any("H2: declared verdict 'supported' but the frozen rule gives 'refuted'" in f for f in fails)


def test_underpowered_result_cannot_be_supported(fx):
    edit_json(fx / "claims/H3.json", lambda o: o.update(verdict="supported"))
    rehash(fx)
    assert any("frozen rule gives 'inconclusive'" in f for f in run(fx))


def test_dev_split_cannot_feed_a_verdict(fx):
    edit_json(fx / "claims/H10.json", lambda o: o["results"][0].update(split="dev"))
    rehash(fx)
    assert "D3" in codes(run(fx))


def test_wrong_holm_is_caught(fx):
    edit_json(fx / "claims/H1.json", lambda o: o["results"][0].update(p_holm=0.001))
    rehash(fx)
    assert any("p_holm" in f for f in run(fx))


def test_number_mismatch_in_paper_input(fx):
    p = fx / "PAPER_INPUT.md"
    p.write_text(p.read_text().replace("7.2 points", "7.9 points"))
    rehash(fx)
    assert any("'7.9' does not match" in f for f in run(fx))


def test_unreferenced_number_in_results(fx):
    p = fx / "PAPER_INPUT.md"
    p.write_text(p.read_text().replace("short of the 10-point threshold.", "short of the threshold by 4.0 points."))
    rehash(fx)
    assert any("'4.0' has no [[ref" in f for f in run(fx))


def test_missing_verdict_heading(fx):
    p = fx / "PAPER_INPUT.md"
    p.write_text(p.read_text().replace("### H2 — drift — VERDICT: refuted", "### H2 — drift"))
    rehash(fx)
    assert "D10" in codes(run(fx))


def test_test_run_before_prereg_tag(fx):
    rows = list(csv.DictReader((fx / "runs/index.csv").open()))
    rows[0]["started_at"] = "2026-11-20T00:00:00Z"
    make_fixture.write_csv(fx / "runs/index.csv", rows)
    rehash(fx)
    assert any("started before the prereg-v2 commit" in f for f in run(fx))


def test_rules_changed_after_tag(fx):
    p = fx / "prereg/rules.yaml"
    p.write_text(p.read_text().replace("threshold: 10.0", "threshold: 5.0"))
    rehash(fx)
    assert any("differs from the version in the tagged commit" in f for f in run(fx))


def test_budget_overrun(fx):
    rows = list(csv.DictReader((fx / "budget/ledger.csv").open()))
    rows.append({**rows[1], "amount_usd": "500"})
    make_fixture.write_csv(fx / "budget/ledger.csv", rows)
    rehash(fx)
    assert "D9" in codes(run(fx))


def test_unapproved_envelope(fx):
    rows = list(csv.DictReader((fx / "budget/ledger.csv").open()))
    rows.append({**rows[1], "envelope": "E9", "amount_usd": "1"})
    make_fixture.write_csv(fx / "budget/ledger.csv", rows)
    rehash(fx)
    assert any("unapproved envelope 'E9'" in f for f in run(fx))


def test_missing_reconciliation_week(fx):
    rows = list(csv.DictReader((fx / "budget/ledger.csv").open()))
    rows.append({**rows[1], "ts_utc": "2026-12-20T12:00:00Z", "amount_usd": "1"})
    make_fixture.write_csv(fx / "budget/ledger.csv", rows)
    rehash(fx)
    assert any("no Modal reconciliation covers ISO week" in f for f in run(fx))


def test_gate_failure_claim_needs_failed_gate(fx):
    edit_json(fx / "gates/G1.json", lambda o: o.update(passed=True))
    rehash(fx)
    assert any("does not show passed: false" in f for f in run(fx))


def test_tampered_file_detected(fx):
    (fx / "tables/main.csv").write_text((fx / "tables/main.csv").read_text().replace("7.2", "9.9"))
    assert "D12" in codes(run(fx))


def test_failed_run_cannot_be_cited(fx):
    rows = list(csv.DictReader((fx / "runs/index.csv").open()))
    rows[0]["status"] = "failed"
    make_fixture.write_csv(fx / "runs/index.csv", rows)
    rehash(fx)
    assert any("has status 'failed'" in f for f in run(fx))


def test_figure_input_hash(fx):
    edit_json(fx / "figures/h1.json", lambda o: o["inputs"][0].update(sha256="0" * 64))
    rehash(fx)
    assert "D8" in codes(run(fx))


def test_model_results_require_unit_tests(fx):
    def add(o):
        o.update(verdict="refuted", failed_gate=None)
        o["results"] = [{"key": "R5_vs_R1_accuracy", "level": "model", "value": -1.0, "ci95": [-3.0, 1.0], "n": 250, "p_raw": None,
                         "p_holm": None, "split": "test", "primary": False, "run_ids": [json.loads((fx / "claims/H1.json").read_text())["results"][0]["run_ids"][0]],
                         "table_ref": "tables/main.csv#row=S11_vs_S2_lbmhqa_late_32k_primary"}]
    edit_json(fx / "claims/H6.json", add)
    rehash(fx)
    assert any("tests/U1.json missing" in f for f in run(fx))


def test_deviation_fields(fx):
    (fx / "deviations.md").write_text("### DEV-1\n- Date: 2026-12-02\n- Change: x\n")
    rehash(fx)
    assert any("lacks 'Reason:'" in f for f in run(fx))


def test_percent_and_backticks(fx):
    # '62%' matches 0.62 x 100 and '`0.0`' is ignored; the valid fixture already contains both
    assert run(fx) == []


def test_rule_evaluator_logic():
    rep = verify_bundle.Report()
    rule = {"any_of": [{"type": "min_delta", "key": "a", "threshold": 0, "require_ci_excludes_zero": True},
                       {"type": "min_delta", "key": "b", "threshold": 0, "require_ci_excludes_zero": True}]}
    res = {"a": {"value": 1.0, "ci95": [-0.5, 2.5], "n": 10, "split": "test"}}
    assert verify_bundle.eval_rule(rule, res, {"a": 10, "b": 10}, rep, "H5") == "inconclusive"   # a refuted, b missing
    res["b"] = {"value": 2.0, "ci95": [0.5, 3.5], "n": 10, "split": "test"}
    assert verify_bundle.eval_rule(rule, res, {"a": 10, "b": 10}, rep, "H5") == "supported"
    nonInf = {"type": "min_delta", "key": "c", "threshold": -3.0, "require_ci_lower_above": -3.0}
    assert verify_bundle.eval_rule(nonInf, {"c": {"value": -1.0, "ci95": [-2.5, 0.4], "n": 5, "split": "test"}}, {"c": 5}, rep, "H8") == "supported"
    assert verify_bundle.eval_rule(nonInf, {"c": {"value": -1.0, "ci95": [-3.5, 0.4], "n": 5, "split": "test"}}, {"c": 5}, rep, "H8") == "refuted"
    within = {"type": "within", "key": "d", "lo": 3, "hi": 5, "measurement": True}
    assert verify_bundle.eval_rule(within, {"d": {"value": 4, "n": 0, "split": "n/a"}}, {}, rep, "H7") == "supported"
    assert rep.fails == []


# ---- v3 (PLAN_V3.md): hypotheses come from the frozen rules file, so H11-H13 are required too

def test_v3_hypotheses_are_required(fx):
    (fx / "claims/H12.json").unlink()
    rehash(fx)
    assert any("claims/H12.json missing" in f for f in run(fx))


def test_v3_non_inferiority_rule_is_recomputed(fx):
    rules = yaml.safe_load((fx / "prereg/rules.yaml").read_text())["hypotheses"]
    res = {"key": "V3_paged_vs_B1_full_accuracy_lbmhqa_32k", "level": "model", "value": -1.0, "ci95": [-2.5, 0.5],
           "n": 400, "p_raw": None, "p_holm": None, "test": "paired bootstrap", "split": "test", "primary": True,
           "run_ids": [], "table_ref": "tables/main.csv#row=S11_vs_S2_lbmhqa_late_32k_primary"}
    mem = dict(res, key="V3_resident_kv_fraction_of_full", level="measurement", value=0.125, ci95=None, n=0, primary=False,
               split="n/a", test="measurement")
    by_key = {r["key"]: r for r in (res, mem)}
    rep = verify_bundle.Report()
    min_n = {"V3_paged_vs_B1_full_accuracy_lbmhqa_32k": 300}
    # CI lower bound -2.5 is inside the -3.0 margin (rules.yaml version 2, PLAN_V3 revision 4): non-inferior
    assert verify_bundle.eval_rule(rules["H11"]["rule"], by_key, min_n, rep, "H11") == "supported"
    by_key["V3_paged_vs_B1_full_accuracy_lbmhqa_32k"]["ci95"] = [-3.4, 0.5]
    # CI lower bound -3.4 is below the margin: non-inferiority not shown
    assert verify_bundle.eval_rule(rules["H11"]["rule"], by_key, min_n, rep, "H11") == "refuted"
    by_key["V3_paged_vs_B1_full_accuracy_lbmhqa_32k"]["ci95"] = [-1.6, 0.4]
    assert verify_bundle.eval_rule(rules["H11"]["rule"], by_key, min_n, rep, "H11") == "supported"
    # too much KV resident: the memory half of the claim fails
    by_key["V3_resident_kv_fraction_of_full"]["value"] = 0.5
    assert verify_bundle.eval_rule(rules["H11"]["rule"], by_key, min_n, rep, "H11") == "refuted"
    assert rep.fails == []
