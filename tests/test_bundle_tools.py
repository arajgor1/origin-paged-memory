"""P6 tooling: opa.claims + opa.bundle build complete synthetic proof bundles that the human-owned verifier
(tools/verify_bundle.py, run as a subprocess and imported read-only) accepts, for an all-gates-passed run with a
mix of verdicts and for a G1-failure run (H4-H9 not_run_gate_failed). Nothing here spends money or touches GCS."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from opa import PROJECT_DIR
from opa import bundle as bd
from opa import claims as cl
from opa import stats as st
from opa.analysis import figures as fg
from opa.analysis import tables as tb

sys.path.insert(0, str(PROJECT_DIR / "tools" / "tests"))
import make_fixture  # noqa: E402  (read-only import: TAG_TIME and the git helper)

VERIFIER = PROJECT_DIR / "tools" / "verify_bundle.py"
TAG_TIME = make_fixture.TAG_TIME
MIN_N = {"S11_vs_S2_lbmhqa_late_32k_primary": 300, "S11_vs_S4_lbmhqa_late_32k_primary": 300, "D4_vs_D2_retention_t50": 150,
         "S11_vs_S8_late_bound": 300, "V3_vs_B1_source_attribution": 200, "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross": 900,
         "A2_vs_A1_model_thread_confusion_task": 400, "R5_vs_R1_accuracy": 200, "descent_beam4_vs_flat_recall_at_M": 500,
         "pinned_vs_unpinned_repeat_pagein_reduction": 100,
         "V3_paged_vs_B1_full_accuracy_lbmhqa_32k": 300, "V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k": 300,
         "copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k": 300}


def run_verifier(bundle: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(VERIFIER), str(bundle), *extra], capture_output=True, text=True)


# ====================================================================== synthetic project
def rid(track, system, ds, cell, split, seed, day):
    """Run id as opa.launch.make_run_id builds it ('/' replaced by '_', so split n/a appears as n_a)."""
    return f"{track}-{system}-{ds}-{cell}-{split.replace('/', '_')}-s{seed}-202612{day:02d}120000-abc1234-{seed % 65536:04x}"


def make_project(root: Path) -> Path:
    """A throwaway project tree: prereg/, budget/, runs/, requirements.txt (what opa.bundle.assemble copies)."""
    (root / "prereg").mkdir(parents=True)
    shutil.copy(PROJECT_DIR / "prereg" / "rules.yaml", root / "prereg" / "rules.yaml")
    sizes = {"pilot_discordant_rates": {"S11_vs_S2_lbmhqa_late_32k_primary": 0.2}, "min_n": MIN_N,
             "equivalence": {"A2_vs_A3_margin_points": 3.0, "A2_vs_A3_agent_min_n": 1428}}
    (root / "prereg" / "sample_sizes.yaml").write_text(yaml.safe_dump(sizes, sort_keys=False), encoding="utf-8")
    (root / "prereg" / "prereg-v2.md").write_text("# Pre-registration (synthetic)\n", encoding="utf-8")
    (root / "prereg" / "PREREG_TAG.json").write_text(json.dumps({"tag": "prereg-v2", "commit": "0" * 40,
                                                                 "committed_at": TAG_TIME}))
    (root / "requirements.txt").write_text("numpy\n")
    (root / "budget" / "reconciliation").mkdir(parents=True)
    shutil.copy(PROJECT_DIR / "budget" / "prices.yaml", root / "budget" / "prices.yaml")
    approvals = {"total_cap_usd": 3000, "envelopes": [
        {"id": e, "approved_usd": a, "approved_by": "AR", "approved_on": "2026-11-10"}
        for e, a in (("ORCH", 300), ("E0", 150), ("E1", 500), ("E2", 1500), ("E3", 300), ("E4", 800))]}
    (root / "budget" / "approvals.yaml").write_text(yaml.safe_dump(approvals), encoding="utf-8")
    ledger = [
        {"ts_utc": "2026-12-05T12:00:00Z", "envelope": "E2", "job": "j1", "kind": "reserve", "amount_usd": "50", "provider": "modal+anthropic", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "actual", "amount_usd": "12.5", "provider": "modal", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "actual", "amount_usd": "30", "provider": "anthropic", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "release", "amount_usd": "50", "provider": "modal+anthropic", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-14T13:00:00Z", "envelope": "E4", "job": "j2", "kind": "actual", "amount_usd": "40.25", "provider": "modal", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-15T00:00:00Z", "envelope": "ORCH", "job": "claude-session", "kind": "actual", "amount_usd": "0.5", "provider": "orchestrator", "run_id": "", "note": ""},
        {"ts_utc": "2026-12-16T00:00:00Z", "envelope": "RECON", "job": "reconcile", "kind": "reconcile", "amount_usd": "53.1", "provider": "modal", "run_id": "", "note": ""},
    ]
    make_fixture.write_csv(root / "budget" / "ledger.csv", ledger)
    (root / "budget" / "reconciliation" / "modal_20261130_20261216.json").write_text(json.dumps(
        {"provider": "modal", "period_start": "2026-11-30T00:00:00+00:00", "period_end": "2026-12-16T00:00:00+00:00",
         "provider_total_usd": 53.1, "ledger_total_usd": 52.75, "explanation": "", "rows": []}))
    (root / "runs").mkdir()
    return root


def add_runs(root: Path, runs: list[tuple[str, str, int]]) -> None:
    """runs: (run_id, split, seed). Writes runs/index.csv and one folder per run as opa.launch would."""
    rows = []
    for i, (r, split, seed) in enumerate(runs):
        day = 3 + i % 20
        rows.append({"run_id": r, "track": r.split("-")[0], "hypotheses": "", "system": "", "dataset": "", "cell": "",
                     "split": split, "seed": seed, "started_at": f"2026-12-{day:02d}T12:00:00Z",
                     "finished_at": f"2026-12-{day:02d}T13:00:00Z", "status": "ok", "cost_usd": "1.0", "envelope": "E2",
                     "gcs_uri": f"gs://b/opa/runs/{r}", "git_sha": "abc1234", "call_id": "fc-1"})
        d = root / "runs" / r
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"run_id": r, "git_sha": "abc1234" * 5, "lock_sha256": "a" * 64,
                                                    "models": {"api_model": "claude-sonnet-5-5"}, "seeds": {"seed": seed},
                                                    "image": {"python": "3.11"}}))
        (d / "metrics.json").write_text(json.dumps({"em": 0.5}))
        (d / "events_uri.json").write_text(json.dumps({"uri": f"gs://b/opa/runs/{r}/events.jsonl.gz", "sha256": "b" * 64}))
        (d / "job.yaml").write_text("name: x\n")
    # one failed, uncited run: must not end up in the bundle index
    rows.append({**rows[0], "run_id": "A-failed-run", "status": "failed"})
    make_fixture.write_csv(root / "runs" / "index.csv", rows)


def bern(rng, p, n):
    return (rng.random(n) < p).astype(int)


def contrasts_and_measurements(rng, *, track_b: bool, g2_failed: bool = False, h5_weak: bool = False):
    """(contrast rows, measurement rows, extra tables, run ids) for a run where every gate passed (track_b=True), G1
    failed (track_b=False: Track A only) or G2 failed (g2_failed=True: no H9, H11-H13, no model-level H5 result)."""
    R = {}
    C, M = [], []

    def con(key, a, b, run, *, test="mcnemar", scale=100.0):
        row = st.contrast(a, b, key, test=test, scale=scale, run_ids=[run], split="test")
        C.append(row)
        return row

    def meas(key, value, run, split="n/a", n=0):
        M.append({"key": key, "value": float(value), "n": n, "run_ids": [run], "split": split})

    R["a1"] = rid("A", "S11", "lbmhqa", "late-h3-128k", "test", 100000, 5)
    base = bern(rng, 0.55, 400)
    s11 = np.where(rng.random(400) < 0.35, 1, base)
    con("S11_vs_S2_lbmhqa_late_32k_primary", s11, base, R["a1"])
    con("S11_vs_S4_lbmhqa_late_32k_primary", s11, np.where(rng.random(400) < 0.15, 0, base), R["a1"])
    R["a2"] = rid("A2", "D4", "drift", "s50", "test", 100001, 6)
    d2 = rng.integers(3, 8, 200) / 10
    d4 = np.clip(d2 + rng.choice([0.0, 0.1], 200, p=[0.4, 0.6]), 0, 1)
    con("D4_vs_D2_retention_t50", d4, d2, R["a2"], test="wilcoxon")
    R["a3"] = rid("A", "S11", "lbmhqa", "items", "test", 100002, 7)
    pred = rng.random(400)
    obs = bern(rng, 1.0, 400) * (rng.random(400) < pred)
    r2 = st.r2_binned(pred, obs)
    meas("R2_predicted_vs_observed_success", r2["r2"], R["a3"], split="test", n=r2["n"])
    b8 = bern(rng, 0.5, 100)
    con("S11_vs_S8_late_bound", np.where(rng.random(100) < 0.3, 1, b8), b8, R["a3"])
    R["a10"] = rid("A", "pin", "thrash", "r4", "test", 100003, 8)
    unp = rng.poisson(2.0, 150)
    con("pinned_vs_unpinned_repeat_pagein_reduction", unp, rng.poisson(0.2, 150), R["a10"], test="wilcoxon", scale=1.0)
    items = pd.DataFrame({"key": [f"i{i}" for i in range(400)], "predicted": pred, "observed": obs})
    extra = {"H3_items": items}
    if not track_b:
        return C, M, extra, R
    R["b4"] = rid("B", "V3", "archive", "w8", "test", 100004, 9)
    meas("V0_own_origin_mass_ratio_to_uniform", 1.12, R["b4"])
    if not g2_failed:      # V3 (LoRA, P4) does not exist once G2 failed: H4 keeps only its P3 measurement
        meas("V3_own_origin_mass_fraction", 0.947, R["b4"])
        b1 = bern(rng, 0.4, 300)
        con("V3_vs_B1_source_attribution", np.where(rng.random(300) < 0.4, 1, b1), b1, R["b4"])
    R["a5"] = rid("A", "S12", "mtlbmhqa", "T8-h3-128k-cross", "test", 100005, 10)
    a1 = bern(rng, 0.6, 1000)
    a2 = a1.copy() if h5_weak else np.where(rng.random(1000) < 0.15, 1, a1)
    con("A2_vs_A1_agent_mtlbmhqa_32k_primary_cross", a2, a1, R["a5"])
    a3 = np.where(rng.random(1000) < 0.05, 1 - a2, a2)
    t = st.tost_paired_p(a2, a3, margin=3.0)
    C.append({"key": "A2_vs_A3_agent_equivalence", "value": t["diff"], "ci_lo": None, "ci_hi": None, "n": t["n"],
              "p_raw": t["p"], "p_holm": None, "test": "TOST, Tango score, margin +-3 EM", "run_ids": [R["a5"]],
              "split": "test"})
    R["b5"] = rid("B", "A2", "task4", "thread", "test", 100006, 11)
    m1 = bern(rng, 0.5, 500)
    if not g2_failed:
        con("A2_vs_A1_model_thread_confusion_task", np.where(rng.random(500) < 0.3, 1, m1), m1, R["b5"])
    R["b6"] = rid("B", "R5", "longbench", "16k", "test", 100007, 12)
    r1 = bern(rng, 0.5, 300)
    con("R5_vs_R1_accuracy", np.where(rng.random(300) < 0.3, 1, r1), r1, R["b6"])
    meas("best_remedy_recovery_fraction", 0.81, R["b6"])
    R["b7"] = rid("B", "kernel", "precision", "depth", "n/a", 7, 13)
    for k, v in (("reanchored_error_growth_ratio_depth40_vs_depth1_bf16", 1.04), ("global_fail_depth_bf16", 4.0),
                 ("global_fail_depth_fp16", 7.0), ("global_fail_depth_fp32", 16.0)):
        meas(k, v, R["b7"])
    R["a8"] = rid("A", "S12", "mtlbmhqa", "routing", "test", 100008, 14)
    flat = bern(rng, 0.9, 600)
    desc = flat.copy()
    desc[rng.random(600) < 0.01] = 0
    desc[(flat == 0) & (rng.random(600) < 0.05)] = 1
    con("descent_beam4_vs_flat_recall_at_M", desc, flat, R["a8"])
    if g2_failed:
        return C, M, extra, R
    R["c9"] = rid("C", "copy", "kv", "sweep", "n/a", 9, 15)
    meas("max_speedup_copy_over_prefill", 0.83, R["c9"])
    # H11-H13 (v3): non-inferiority, thread-confusion reduction, resident fraction, copy reuse
    R["b11"] = rid("B", "V3", "lbmhqa", "paged-32k", "test", 100009, 16)
    f1 = bern(rng, 0.55, 400)
    con("V3_paged_vs_B1_full_accuracy_lbmhqa_32k", np.where(rng.random(400) < 0.005, 1 - f1, f1), f1, R["b11"])
    meas("V3_resident_kv_fraction_of_full", 0.21, R["b11"])
    R["b12"] = rid("B", "V3", "mtlbmhqa", "confusion-32k", "test", 100010, 17)
    c1 = bern(rng, 0.45, 400)
    con("V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k", np.where(rng.random(400) < 0.2, 1, c1), c1, R["b12"])
    R["c13"] = rid("C", "copy", "lbmhqa", "reuse-32k", "test", 100011, 18)
    r1 = bern(rng, 0.55, 400)
    con("copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k", np.where(rng.random(400) < 0.005, 1 - r1, r1), r1, R["c13"])
    meas("reload_speedup_copy_over_recompute_B512", 3.4, R["c13"])
    return C, M, extra, R


LEVEL = {"V3_vs_B1_source_attribution": "model", "A2_vs_A1_model_thread_confusion_task": "model",
         "R5_vs_R1_accuracy": "model", "max_speedup_copy_over_prefill": "systems",
         "V3_paged_vs_B1_full_accuracy_lbmhqa_32k": "model", "V3_resident_kv_fraction_of_full": "model",
         "V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k": "model",
         "copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k": "systems", "reload_speedup_copy_over_recompute_B512": "systems"}
HYP = {"S11_vs_S2_lbmhqa_late_32k_primary": "H1", "S11_vs_S4_lbmhqa_late_32k_primary": "H1", "D4_vs_D2_retention_t50": "H2",
       "R2_predicted_vs_observed_success": "H3", "S11_vs_S8_late_bound": "H3",
       "V0_own_origin_mass_ratio_to_uniform": "H4", "V3_own_origin_mass_fraction": "H4", "V3_vs_B1_source_attribution": "H4",
       "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross": "H5", "A2_vs_A3_agent_equivalence": "H5",
       "A2_vs_A1_model_thread_confusion_task": "H5", "R5_vs_R1_accuracy": "H6", "best_remedy_recovery_fraction": "H6",
       "reanchored_error_growth_ratio_depth40_vs_depth1_bf16": "H7", "global_fail_depth_bf16": "H7",
       "global_fail_depth_fp16": "H7", "global_fail_depth_fp32": "H7", "descent_beam4_vs_flat_recall_at_M": "H8",
       "max_speedup_copy_over_prefill": "H9", "pinned_vs_unpinned_repeat_pagein_reduction": "H10",
       "V3_paged_vs_B1_full_accuracy_lbmhqa_32k": "H11", "V3_resident_kv_fraction_of_full": "H11",
       "V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k": "H12",
       "copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k": "H13", "reload_speedup_copy_over_recompute_B512": "H13"}


def build_bundle(root: Path, *, g1_failed: bool, g2_failed: bool = False, h5_weak: bool = False, seed: int = 0) -> tuple[Path, dict]:
    """The whole P6 sequence on a synthetic project, as the docstring of opa.claims describes it."""
    make_project(root)
    rng = np.random.default_rng(seed)
    C, M, extra, R = contrasts_and_measurements(rng, track_b=not g1_failed, g2_failed=g2_failed, h5_weak=h5_weak)
    add_runs(root, [(r, "n/a" if "-n_a-" in r else "test", 100000 + i) for i, r in enumerate(R.values())])
    B = root / "bundle"
    rules, min_n = cl.load_prereg(root / "prereg")

    # 1. tables (frozen from here on)
    tb.write_contrast_table(B / "tables" / "contrasts.csv", C, extra_columns=["n10", "n01", "split"],
                            ci_only_keys=cl.non_inferiority_keys(rules))
    tb.write_table(B / "tables" / "measurements.csv", M, columns=["key", "value", "n", "run_ids", "split"])
    tb.write_table(B / "tables" / "H3_items.csv", extra["H3_items"])
    # 2. code, then a figure whose sidecar names a script inside the bundle
    bd.copy_code(project=PROJECT_DIR, bundle=B)
    fig = fg.plot_observed_vs_predicted(extra["H3_items"], predicted="predicted", observed="observed")
    fg.save_figure(fig, "h3_observed_vs_predicted", B / "figures", "code/opa/analysis/figures.py",
                   ["tables/H3_items.csv"], created_at="2027-01-01T00:00:00Z")
    # 3. gates
    gates = {"G0": cl.gate("G0", True, [cl.criterion("oracle S18 EM on LB-MHQA pilot", 0.9, 0.94, True)],
                           ["runs/index.csv"], decided_at="2026-11-20T00:00:00Z")}
    if g1_failed:
        pilot = {"S11_vs_S2_lbmhqa_late_32k_primary": {"value": 2.0}, "S11_vs_S4_lbmhqa_late_32k_primary": {"value": 4.0}}
        v = cl.evaluate_pilot("H1", pilot, rules)
        gates["G1"] = cl.gate("G1", False, [cl.criterion("H1 passes on the pilot (point estimates)", "supported", v, v == "supported")],
                              ["tables/contrasts.csv"], decided_at="2026-11-28T00:00:00Z")
    elif g2_failed:
        gates["G1"] = cl.gate("G1", True, [cl.criterion("G1 criteria", "met", "met", True)], ["tables/contrasts.csv"])
        gates["G2"] = cl.gate("G2", False, [cl.criterion("H4 and H6 on the pilot", "supported", "refuted", False)],
                              ["tables/contrasts.csv"], decided_at="2026-12-10T00:00:00Z")
    else:
        for g in ("G1", "G2", "G3"):
            gates[g] = cl.gate(g, True, [cl.criterion(f"{g} criteria", "met", "met", True)], ["tables/contrasts.csv"])
    gates["G4"] = cl.gate("G4", True, [cl.criterion("all primary results on test splits with CIs", True, True, True)],
                          ["claims/"])
    gates = cl.complete_gates(gates)
    cl.write_gates(B, gates)
    # 4. claims
    res: dict[str, list] = {}
    for row in C:
        res.setdefault(HYP[row["key"]], []).append(cl.result_from_table(
            B, "tables/contrasts.csv", row["key"], level=LEVEL.get(row["key"], "agent"),
            primary=row["key"] == rules["hypotheses"][HYP[row["key"]]].get("primary_key")))
    for row in M:
        res.setdefault(HYP[row["key"]], []).append(cl.result_from_table(
            B, "tables/measurements.csv", row["key"], level=LEVEL.get(row["key"], "measurement")))
    # deviations and unit-test records come before the claims (P6 order)
    cl.write_deviations(B, [{"id": "DEV-1", "date": "2026-12-04", "change": "drift contrast tested at session level",
                             "reason": "constraints within a session are not independent", "impact": "wider CI"}])
    if not g1_failed:
        cl.write_unit_tests(B, [{"id": u, "name": f"unit test {u}", "passed": True, "details": "synthetic",
                                 "run_id": R["b7"], **({"bf16": {"passed": True}} if u in cl.BF16_UNIT_IDS else {})}
                                for u in cl.U_IDS], runs_index=root / "runs" / "index.csv")
    claims = cl.build_claims(res, gates, rules=rules, min_n=min_n, deviation_ids=cl.read_deviation_ids(B),
                             figures={"H3": ["figures/h3_observed_vs_predicted.png"]},
                             deviations={"H2": ["DEV-1"]})
    cl.write_claims(B, claims)
    # 5. copy prereg, budget, cited runs, env
    problems = bd.assemble(project=root, bundle=B)
    assert problems == []
    cl.write_budget_table(B)
    measured = [{"location": "§7.2 own-origin mass at λ = 5", "old": "0.913", "decimals": 3,
                 "ref": "claims/H4.json#results.1.value" if not g1_failed else "claims/H3.json#results.1.value",
                 "note": "synthetic"}]
    if not g1_failed:
        k4 = "V0_own_origin_mass_ratio_to_uniform" if g2_failed else "V3_own_origin_mass_fraction"
        idx = [r["key"] for r in claims["H4"]["results"]].index(k4)
        measured[0]["ref"] = f"claims/H4.json#results.{idx}.value"
    else:
        idx = [r["key"] for r in claims["H3"]["results"]].index("R2_predicted_vs_observed_success")
        measured[0]["ref"] = f"claims/H3.json#results.{idx}.value"
    cl.write_paper_input(B, claims, gates, measured=measured,
                         captions={"figures/h3_observed_vs_predicted.png": "binned observed vs predicted success"})
    bd.manifest(B)
    return B, claims


# ====================================================================== tests
def test_holm_matches_verifier():
    v = cl.verifier()
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = {f"k{i}": float(x) for i, x in enumerate(rng.random(rng.integers(1, 11)) ** 3)}
        a, b = st.holm(p), v.holm(p)
        assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)


def test_evaluate_uses_verifier_rules():
    rules, _ = cl.load_prereg()
    ok = {"key": "D4_vs_D2_retention_t50", "value": 12.0, "ci95": [3.0, 20.0], "n": 200, "split": "test", "p_holm": 0.01}
    assert cl.evaluate("H2", [ok], rules, {"D4_vs_D2_retention_t50": 150}) == ("supported", [])
    assert cl.evaluate("H2", [{**ok, "n": 100}], rules, {"D4_vs_D2_retention_t50": 150})[0] == "inconclusive"
    assert cl.evaluate("H2", [{**ok, "p_holm": 0.2}], rules, {"D4_vs_D2_retention_t50": 150})[0] == "refuted"
    v, fails = cl.evaluate("H2", [ok], rules, {})
    assert v == "inconclusive" and fails and "minimum n" in fails[0]
    # pilot mode: thresholds only, no n / split / CI / Holm conditions
    pilot = {"S11_vs_S2_lbmhqa_late_32k_primary": {"value": 6.0, "split": "dev", "n": 50},
             "S11_vs_S4_lbmhqa_late_32k_primary": {"value": 3.5, "split": "dev", "n": 50}}
    assert cl.evaluate_pilot("H1", pilot, rules) == "supported"
    assert cl.evaluate_pilot("H1", {**pilot, "S11_vs_S4_lbmhqa_late_32k_primary": {"value": 2.0}}, rules) == "refuted"
    assert "Holm p < 0.05" in cl.rule_text(rules["hypotheses"]["H1"]["rule"])


def test_full_bundle_verifies(tmp_path):
    B, claims = build_bundle(tmp_path / "proj", g1_failed=False)
    verdicts = {h: c["verdict"] for h, c in claims.items()}
    assert verdicts == {"H1": "supported", "H2": "refuted", "H3": "inconclusive", "H4": "supported", "H5": "supported",
                        "H6": "refuted", "H7": "supported", "H8": "supported", "H9": "refuted", "H10": "supported",
                        "H11": "supported", "H12": "supported", "H13": "supported"}
    # Holm family = primaries whose rule leaf has require_p_holm_below; H8, H11, H13 (CI-based) report p_raw null
    fam = {h for h, c in claims.items() for r in c["results"] if r["primary"] and r["p_raw"] is not None}
    assert fam == {"H1", "H2", "H3", "H4", "H5", "H6", "H10", "H12"}
    for h in ("H8", "H11", "H13"):
        pr = [r for r in claims[h]["results"] if r["primary"]]
        assert len(pr) == 1 and pr[0]["p_raw"] is None and pr[0]["p_holm"] is None, h
    assert cl.preflight(B) == []
    out = run_verifier(B, "--no-git")
    assert out.returncode == 0 and "BUNDLE VERIFIED" in out.stdout, out.stdout + out.stderr
    idx = list(csv.DictReader((B / "runs" / "index.csv").open(newline="")))
    assert "A-failed-run" not in {r["run_id"] for r in idx}
    man = json.loads((B / "manifest.json").read_text())
    assert man["nested_manifests"] and all(k.startswith("runs/") for k in man["nested_manifests"])
    # tampering after the manifest is caught
    (B / "tables" / "contrasts.csv").write_text((B / "tables" / "contrasts.csv").read_text() + "\n")
    assert "D12" in run_verifier(B, "--no-git").stdout


def test_gate_failure_bundle_verifies(tmp_path):
    B, claims = build_bundle(tmp_path / "proj", g1_failed=True)
    for h in ("H4", "H5", "H6", "H7", "H8", "H9", "H11", "H12", "H13"):
        assert claims[h]["verdict"] == "not_run_gate_failed" and claims[h]["failed_gate"] == "G1", h
        assert claims[h]["results"] == []
    assert sum(c["verdict"] == "not_run_gate_failed" for c in claims.values()) == 9
    gates = cl.read_gates(B)
    assert [gates[g]["passed"] for g in cl.G_IDS] == [True, False, None, None, True]
    assert not (B / "tests").exists()          # no Track B results -> unit-test records not required
    out = run_verifier(B, "--no-git")
    assert out.returncode == 0 and "BUNDLE VERIFIED" in out.stdout, out.stdout + out.stderr
    text = (B / "PAPER_INPUT.md").read_text(encoding="utf-8")
    assert "### H9 - " in text and "VERDICT: not_run_gate_failed" in text


def test_g2_failure_bundle_verifies(tmp_path):
    """G2 failed: H9, H11, H12, H13 are not_run_gate_failed (rules.yaml gate_dependency G2); H4-H8 and H10 still ran
    (dependency G1, which passed); H5 is evaluated on its agent-level leaf alone."""
    B, claims = build_bundle(tmp_path / "proj", g1_failed=False, g2_failed=True)
    for h in ("H9", "H11", "H12", "H13"):
        assert claims[h]["verdict"] == "not_run_gate_failed" and claims[h]["failed_gate"] == "G2", h
        assert claims[h]["results"] == []
    for h in ("H1", "H4", "H5", "H6", "H7", "H8", "H10"):
        assert claims[h]["verdict"] != "not_run_gate_failed" and claims[h]["results"], h
    assert claims["H4"]["verdict"] == "inconclusive"     # only the P3 measurement V0 exists; V3 needs P4
    assert [r["key"] for r in claims["H4"]["results"]] == ["V0_own_origin_mass_ratio_to_uniform"]
    assert claims["H5"]["verdict"] == "supported"
    assert not [r for r in claims["H5"]["results"] if r["level"] == "model"]
    gates = cl.read_gates(B)
    assert [gates[g]["passed"] for g in cl.G_IDS] == [True, True, False, None, True]
    assert (B / "tests" / "U8.json").exists()           # model-level H4/H6 results still need the unit tests
    out = run_verifier(B, "--no-git")
    assert out.returncode == 0 and "BUNDLE VERIFIED" in out.stdout, out.stdout + out.stderr
    text = (B / "PAPER_INPUT.md").read_text(encoding="utf-8")
    for h in ("H9", "H11", "H12", "H13"):
        assert f"### {h} - " in text
    assert text.count("VERDICT: not_run_gate_failed") == 4


def test_g2_failure_h5_agent_leaf_unsupported_is_inconclusive(tmp_path):
    B, claims = build_bundle(tmp_path / "proj", g1_failed=False, g2_failed=True, h5_weak=True)
    assert claims["H5"]["verdict"] == "inconclusive"
    out = run_verifier(B, "--no-git")
    assert out.returncode == 0 and "BUNDLE VERIFIED" in out.stdout, out.stdout + out.stderr


def test_g2_failed_rejects_h11_h13_results_without_deviation(tmp_path):
    rules, gates = _tiny(tmp_path)
    gates = {**gates, "G1": cl.gate("G1", True, [], []), "G2": cl.gate("G2", False, [], [])}
    key = "V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k"
    tb.write_contrast_table(tmp_path / "tables" / "u.csv", [
        {"key": key, "value": 5.0, "ci95": [1.0, 9.0], "n": 400, "p_raw": 0.001, "test": "x", "run_ids": ["r1"]}])
    r = cl.result_from_table(tmp_path, "tables/u.csv", key, level="model", primary=True)
    with pytest.raises(cl.ClaimError, match="G2 failed"):
        cl.build_claims({"H12": [r]}, gates, rules=rules, min_n=MIN_N)
    claims = cl.build_claims({"H12": [r]}, gates, rules=rules, min_n=MIN_N, deviations={"H12": ["DEV-1"]}, deviation_ids=["DEV-1"])
    assert claims["H12"]["verdict"] == "supported"


def test_non_inferiority_primary_with_p_raw_is_rejected(tmp_path):
    rules, gates = _tiny(tmp_path)
    key = "V3_paged_vs_B1_full_accuracy_lbmhqa_32k"
    tb.write_contrast_table(tmp_path / "tables" / "n.csv", [
        {"key": key, "value": -0.5, "ci95": [-1.5, 0.5], "n": 400, "p_raw": 0.4, "test": "x", "run_ids": ["r1"]}])
    assert not cl.holm_required(rules, "H11") and cl.holm_required(rules, "H12") and cl.holm_required(rules, "H1")
    mn = {key: 300}
    bad = cl.result_from_table(tmp_path, "tables/n.csv", key, level="model", primary=True)
    assert bad["p_raw"] == 0.4
    with pytest.raises(cl.ClaimError, match="no Holm condition"):
        cl.build_claims({"H11": [bad]}, gates, rules=rules, min_n=mn)
    ok = cl.result_from_table(tmp_path, "tables/n.csv", key, level="model", primary=True, p_raw=None)
    assert cl.build_claims({"H11": [ok]}, gates, rules=rules, min_n=mn)["H11"]["results"][0]["p_raw"] is None
    # converse: a Holm primary without p_raw
    nop = cl.result_from_table(tmp_path, "tables/t.csv", "S11_vs_S2_lbmhqa_late_32k_primary", level="agent",
                               primary=True, p_raw=None)
    with pytest.raises(cl.ClaimError, match="Holm condition but p_raw is None"):
        cl.build_claims({"H1": [nop]}, gates, rules=rules, min_n=MIN_N)


def test_results_after_failed_gate_need_a_deviation(tmp_path):
    rules, gates = _tiny(tmp_path)
    gates = {**gates, "G1": cl.gate("G1", False, [cl.criterion("x", 1, 0, False)], [])}
    key = "V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k"
    tb.write_contrast_table(tmp_path / "tables" / "u.csv", [
        {"key": key, "value": 5.0, "ci95": [1.0, 9.0], "n": 400, "p_raw": 0.001, "test": "x", "run_ids": ["r1"]}])
    r = cl.result_from_table(tmp_path, "tables/u.csv", key, level="model", primary=True)
    with pytest.raises(cl.ClaimError, match="gate G1 failed"):
        cl.build_claims({"H12": [r]}, gates, rules=rules, min_n=MIN_N)
    assert cl.build_claims({"H12": [r]}, gates, rules=rules, min_n=MIN_N,
                           deviations={"H12": ["DEV-1"]}, deviation_ids=["DEV-1"])["H12"]["verdict"] == "supported"


def test_h9_exempt_after_g2_failure_only(tmp_path):
    rules, gates = _tiny(tmp_path)
    gates = {**gates, "G1": cl.gate("G1", True, [], []), "G2": cl.gate("G2", False, [], [])}
    tb.write_table(tmp_path / "tables" / "m.csv", [{"key": "max_speedup_copy_over_prefill", "value": 1.4, "n": 0,
                                                    "run_ids": "r1", "split": "n/a"}],
                   columns=["key", "value", "n", "run_ids", "split"])
    r = cl.result_from_table(tmp_path, "tables/m.csv", "max_speedup_copy_over_prefill", level="systems")
    assert cl.build_claims({"H9": [r]}, gates, rules=rules, min_n=MIN_N)["H9"]["verdict"] == "supported"
    gates["G1"] = cl.gate("G1", False, [cl.criterion("x", 1, 0, False)], [])
    with pytest.raises(cl.ClaimError, match="gate G1 failed"):
        cl.build_claims({"H9": [r]}, gates, rules=rules, min_n=MIN_N)


def test_non_inferiority_table_rows_blank_p_raw(tmp_path):
    rules, _ = cl.load_prereg()
    assert sorted(cl.non_inferiority_keys(rules)) == sorted([
        "descent_beam4_vs_flat_recall_at_M", "V3_paged_vs_B1_full_accuracy_lbmhqa_32k",
        "copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k"])
    key = "V3_paged_vs_B1_full_accuracy_lbmhqa_32k"
    rows = [{"key": key, "value": -0.5, "ci95": [-1.5, 0.5], "n": 400, "p_raw": 0.4, "test": "mcnemar", "run_ids": ["r1"]},
            {"key": "D4_vs_D2_retention_t50", "value": 12.0, "ci95": [3.0, 20.0], "n": 200, "p_raw": 0.01,
             "test": "wilcoxon", "run_ids": ["r1"]}]
    tb.write_contrast_table(tmp_path / "c.csv", rows, ci_only_keys=cl.non_inferiority_keys(rules))
    ni = cl.read_table_row(tmp_path, "c.csv", key)
    assert ni["p_raw"] == "" and ni["test"] == cl.NI_TEST_LABEL == tb.NI_TEST_LABEL
    assert "paired bootstrap 95% CI" in ni["test"] and ni["test"].endswith("non-inferiority: CI lower bound only")
    r = cl.result_from_table(tmp_path, "c.csv", key, level="model", primary=True)
    assert r["p_raw"] is None and r["test"] == cl.NI_TEST_LABEL
    assert cl.read_table_row(tmp_path, "c.csv", "D4_vs_D2_retention_t50")["p_raw"] == "0.01"


def test_unknown_deviation_reference_rejected(tmp_path):
    rules, gates = _tiny(tmp_path)
    mk = lambda key, **kw: cl.result_from_table(tmp_path, "tables/t.csv", key, level="agent", **kw)
    res = {"H1": [mk("S11_vs_S2_lbmhqa_late_32k_primary", primary=True), mk("S11_vs_S4_lbmhqa_late_32k_primary")]}
    cl.build_claims(res, gates, rules=rules, min_n=MIN_N, deviations={"H1": ["DEV-1"]}, deviation_ids=["DEV-1"])
    with pytest.raises(cl.ClaimError, match="DEV-9"):
        cl.build_claims(res, gates, rules=rules, min_n=MIN_N, deviations={"H1": ["DEV-9"]}, deviation_ids=["DEV-1"])
    cl.write_deviations(tmp_path, [{"id": "DEV-1", "date": "d", "change": "c", "reason": "r", "impact": "i"}])
    assert cl.read_deviation_ids(tmp_path) == ["DEV-1"]


def _ut(uid="U1", **kw):
    return {"id": uid, "name": "n", "passed": True, "details": "d", "run_id": "UT-run", **kw}


def _ut_project(tmp_path, status="ok"):
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / "index.csv").write_text(f"run_id,status\nUT-run,{status}\nOther-run,ok\n")
    return tmp_path


def test_write_unit_tests_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "PROJECT_DIR", tmp_path)       # default runs_index = <project>/runs/index.csv
    b = _ut_project(tmp_path)
    ok_bf = {"passed": True}
    with pytest.raises(cl.ClaimError, match="fp32"):                      # (a)
        cl.write_unit_tests(b, [_ut(stage="fp32_only")])
    with pytest.raises(cl.ClaimError, match="fp32"):
        cl.write_unit_tests(b, [_ut(action="unit_test_fp32_stage")])
    with pytest.raises(cl.ClaimError, match="not in"):                    # (b)
        cl.write_unit_tests(b, [_ut("U9")])
    with pytest.raises(cl.ClaimError, match="duplicate id U1"):
        cl.write_unit_tests(b, [_ut(), _ut()])
    for bad in ({"passed": False}, {"passed": True, "error": "x"}, {"passed": True, "state_restored": ["buffer:x"]}):
        with pytest.raises(cl.ClaimError, match="passed: true"):          # (c)
            cl.write_unit_tests(b, [_ut("U2", bf16=bad)])
    with pytest.raises(cl.ClaimError, match="passed: true"):
        cl.write_unit_tests(b, [_ut("U2", fp32={"passed": False}, bf16=ok_bf)])
    with pytest.raises(cl.ClaimError, match="passed must equal"):
        cl.write_unit_tests(b, [_ut("U2", passed=False, fp32={"passed": True}, bf16=ok_bf)])
    with pytest.raises(cl.ClaimError, match="needs its bf16 block"):
        cl.write_unit_tests(b, [_ut("U3")])
    cl.write_unit_tests(b, [_ut("U3")], require_bf16=False)
    (p,) = cl.write_unit_tests(b, [_ut("U4", passed=False, bf16={"passed": False})])   # (d) consistent failure is written
    assert json.loads(p.read_text())["passed"] is False
    with pytest.raises(cl.ClaimError, match="not in"):                    # (e)
        cl.write_unit_tests(b, [_ut(run_id="ghost")])
    with pytest.raises(cl.ClaimError, match="status"):
        cl.write_unit_tests(_ut_project(tmp_path, status="failed"), [_ut()])
    _ut_project(tmp_path)
    with pytest.raises(cl.ClaimError, match="different runs"):
        cl.write_unit_tests(b, [_ut("U1"), _ut("U5", run_id="Other-run")])
    with pytest.raises(cl.ClaimError, match="DEV-2 is not"):
        cl.write_unit_tests(b, [_ut("U1"), _ut("U5", run_id="Other-run")], same_run_deviation="DEV-2")
    ps = cl.write_unit_tests(b, [_ut("U1"), _ut("U5", run_id="Other-run")], same_run_deviation="DEV-2",
                             deviation_ids=["DEV-2"])
    assert all(json.loads(p.read_text())["deviation"] == "DEV-2" for p in ps)


def test_assemble_copies_runs_cited_only_by_unit_tests(tmp_path):
    root = make_project(tmp_path / "proj")
    rid_ = rid("B", "units", "m", "c", "n/a", 7, 3)
    add_runs(root, [(rid_, "n/a", 7)])
    B = root / "bundle"
    (B / "claims").mkdir(parents=True)
    (B / "tests").mkdir()
    (B / "tests" / "U1.json").write_text(json.dumps({"id": "U1", "name": "n", "passed": True, "details": "d", "run_id": rid_}))
    assert bd.cited_runs(B) == {rid_}
    assert bd.assemble(project=root, bundle=B) == []
    assert (B / "runs" / rid_ / "manifest.json").exists()


def test_deviations_without_ids_rejected_and_backstop(tmp_path):
    rules, gates = _tiny(tmp_path)
    mk = lambda key, **kw: cl.result_from_table(tmp_path, "tables/t.csv", key, level="agent", **kw)
    res = {"H1": [mk("S11_vs_S2_lbmhqa_late_32k_primary", primary=True), mk("S11_vs_S4_lbmhqa_late_32k_primary")]}
    with pytest.raises(cl.ClaimError, match="deviation_ids is None"):
        cl.build_claims(res, gates, rules=rules, min_n=MIN_N, deviations={"H1": ["DEV-1"]})


def test_hypotheses_and_keys_come_from_rules():
    rules, _ = cl.load_prereg()
    assert cl.hypothesis_ids(rules) == [f"H{i}" for i in range(1, 14)]
    assert rules["hypotheses"]["H1"]["primary_key"].endswith("32k_primary")
    r11 = {"key": "V3_paged_vs_B1_full_accuracy_lbmhqa_32k", "value": -0.5, "ci95": [-1.5, 0.5], "n": 400, "split": "test"}
    m = {"key": "V3_resident_kv_fraction_of_full", "value": 0.22, "n": 0, "split": "n/a"}
    mn = {"V3_paged_vs_B1_full_accuracy_lbmhqa_32k": 300}
    assert cl.evaluate("H11", [r11, m], rules, mn) == ("supported", [])
    # PLAN_V3 revision 4: H11's non-inferiority margin is 3 points (CI lower bound > -3)
    assert cl.evaluate("H11", [{**r11, "ci95": [-2.5, 0.5]}, m], rules, mn)[0] == "supported"
    assert cl.evaluate("H11", [{**r11, "ci95": [-3.5, 0.5]}, m], rules, mn)[0] == "refuted"
    assert cl.evaluate("H11", [r11, {**m, "value": 0.3}], rules, mn)[0] == "refuted"


def test_bundle_with_git_tag_verifies(tmp_path):
    """D1 with git: commit prereg in a throwaway repo, tag it, write PREREG_TAG.json with opa.bundle."""
    root = tmp_path / "proj"
    make_project(root)
    for a in (("init", "-q"), ("config", "user.email", "t@example.com"), ("config", "user.name", "t"),
              ("config", "core.autocrlf", "false"), ("add", "prereg/rules.yaml", "prereg/sample_sizes.yaml", "prereg/prereg-v2.md")):
        make_fixture.sh(root, "git", *a)
    make_fixture.sh(root, "git", "commit", "-q", "-m", "prereg", env={"GIT_COMMITTER_DATE": TAG_TIME, "GIT_AUTHOR_DATE": TAG_TIME})
    make_fixture.sh(root, "git", "tag", "prereg-v2")
    bd.write_prereg_tag(root)
    tag = json.loads((root / "prereg" / "PREREG_TAG.json").read_text())
    assert len(tag["commit"]) == 40 and tag["committed_at"].startswith("2026-12-01T10:00:00")
    # rebuild the bundle on top of this project (make_project would refuse an existing tree, so reuse its pieces)
    rng = np.random.default_rng(0)
    C, M, extra, R = contrasts_and_measurements(rng, track_b=False)
    add_runs(root, [(r, "test", 100000 + i) for i, r in enumerate(R.values())])
    B = root / "bundle"
    rules, min_n = cl.load_prereg(root / "prereg")
    tb.write_contrast_table(B / "tables" / "contrasts.csv", C, extra_columns=["split"])
    tb.write_table(B / "tables" / "measurements.csv", M, columns=["key", "value", "n", "run_ids", "split"])
    gates = cl.complete_gates({
        "G0": cl.gate("G0", True, [], ["runs/index.csv"]),
        "G1": cl.gate("G1", False, [cl.criterion("H3 on pilot", "supported", "refuted", False)], ["x"]),
        "G4": cl.gate("G4", True, [], ["claims/"])})
    cl.write_gates(B, gates)
    res = {}
    for row in C:
        res.setdefault(HYP[row["key"]], []).append(cl.result_from_table(
            B, "tables/contrasts.csv", row["key"], level="agent",
            primary=row["key"] == rules["hypotheses"][HYP[row["key"]]].get("primary_key")))
    for row in M:
        res.setdefault(HYP[row["key"]], []).append(cl.result_from_table(B, "tables/measurements.csv", row["key"], level="measurement"))
    claims = cl.build_claims(res, gates, rules=rules, min_n=min_n)
    cl.write_claims(B, claims)
    assert bd.assemble(project=root, bundle=B) == []
    cl.write_budget_table(B)
    cl.write_deviations(B)
    cl.write_paper_input(B, claims, gates)
    bd.manifest(B)
    out = run_verifier(B, "--repo", str(root))
    assert out.returncode == 0 and "BUNDLE VERIFIED" in out.stdout, out.stdout + out.stderr


def _tiny(tmp_path):
    rules, _ = cl.load_prereg()
    (tmp_path / "tables").mkdir()
    tb.write_contrast_table(tmp_path / "tables" / "t.csv", [
        {"key": "S11_vs_S2_lbmhqa_late_32k_primary", "value": 7.0, "ci95": [3.0, 11.0], "n": 400, "p_raw": 0.001,
         "test": "x", "run_ids": ["r1"]},
        {"key": "S11_vs_S4_lbmhqa_late_32k_primary", "value": 4.0, "ci95": [1.0, 7.0], "n": 400, "p_raw": 0.01,
         "test": "x", "run_ids": ["r1"]}])
    gates = {"G0": cl.gate("G0", True, [], [])}
    return rules, gates


def test_build_claims_rejects_misuse(tmp_path):
    rules, gates = _tiny(tmp_path)
    mk = lambda key, **kw: cl.result_from_table(tmp_path, "tables/t.csv", key, level="agent", **kw)
    good = {"H1": [mk("S11_vs_S2_lbmhqa_late_32k_primary", primary=True), mk("S11_vs_S4_lbmhqa_late_32k_primary")]}
    claims = cl.build_claims(good, gates, rules=rules, min_n=MIN_N)
    assert claims["H1"]["verdict"] == "supported"
    with pytest.raises(cl.ClaimError, match="split is 'dev'"):
        cl.build_claims({"H1": [mk("S11_vs_S2_lbmhqa_late_32k_primary", primary=True, split="dev")]}, gates, rules=rules, min_n=MIN_N)
    with pytest.raises(cl.ClaimError, match="not the rule's primary_key"):
        cl.build_claims({"H1": [mk("S11_vs_S4_lbmhqa_late_32k_primary", primary=True)]}, gates, rules=rules, min_n=MIN_N)
    with pytest.raises(cl.ClaimError, match="no pre-registered minimum n"):
        cl.build_claims(good, gates, rules=rules, min_n={})
    with pytest.raises(cl.ClaimError, match="does not show passed: false"):
        cl.build_claims(good, gates, rules=rules, min_n=MIN_N, not_run={"H4": "G1"})
    with pytest.raises(cl.ClaimError, match="later gate"):
        cl.build_claims(good, {**gates, "G2": cl.gate("G2", False, [], [])}, rules=rules, min_n=MIN_N, not_run={"H5": "G2"})


def test_holm_family_in_claims(tmp_path):
    rules, gates = _tiny(tmp_path)
    mk = lambda key, **kw: cl.result_from_table(tmp_path, "tables/t.csv", key, level="agent", **kw)
    claims = cl.build_claims({"H1": [mk("S11_vs_S2_lbmhqa_late_32k_primary", primary=True), mk("S11_vs_S4_lbmhqa_late_32k_primary")]},
                             gates, rules=rules, min_n=MIN_N)
    r = claims["H1"]["results"]
    assert r[0]["p_holm"] == pytest.approx(0.001) and r[1]["p_holm"] is None   # one primary -> Holm = raw
    assert claims["H2"]["verdict"] == "inconclusive" and claims["H2"]["results"] == []


def test_paper_input_problems_catch_bad_prose(tmp_path):
    B, claims = build_bundle(tmp_path / "proj", g1_failed=True)
    p = B / "PAPER_INPUT.md"
    text = p.read_text(encoding="utf-8")
    assert cl.paper_input_problems(B) == []
    p.write_text(text.replace("## Measured numbers", "Unreferenced 3.5 here.\n\n## Measured numbers"), encoding="utf-8")
    assert any("3.5" in x and "[D7]" in x for x in cl.paper_input_problems(B))
    p.write_text(text + "\n”quoted”\n", encoding="utf-8")
    assert any("cp1252" in x for x in cl.paper_input_problems(B))
    p.write_text(text, encoding="utf-8")
    (B / "deviations.md").write_text("None.\n")          # backstop: H2 cites DEV-1, no longer in deviations.md
    assert any("[D11]" in x and "DEV-1" in x for x in cl.paper_input_problems(B))


def test_assemble_reports_missing_runs(tmp_path):
    root = make_project(tmp_path / "proj")
    B = root / "bundle"
    (B / "claims").mkdir(parents=True)
    (B / "claims" / "H1.json").write_text(json.dumps({"id": "H1", "results": [{"run_ids": ["ghost-run"]}]}))
    probs = bd.assemble(project=root, bundle=B)
    assert any("ghost-run" in x and "index.csv" in x for x in probs)
    with pytest.raises(SystemExit):
        bd.assemble(project=root, bundle=B, strict=True)
