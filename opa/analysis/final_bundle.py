"""P6: the final proof bundle (bundle_spec/BUNDLE_SPEC.md), built from the interim tables by one command.

  python -m opa.analysis.final_bundle --runs jobs/interim/series_test.yaml --trackb jobs/interim/trackb_test_runs.yaml \\
         --src bundle --out <NEW folder> [--dry-run] [--project-root .] [--creds KEY] [--scratch DIR]

``--src`` is the interim bundle (default ``bundle``); ``--out`` is the final bundle folder, which must not exist or be empty (the final layout
is staged there by copying only the specified parts of ``--src``: tables, figures, gates and the figure script; the dev dry-run files,
PAPER_INPUT_interim*.md, README and release/ are left out, and nothing in ``--src`` is touched).

Inputs that must already exist under <src>/tables (written by the interim steps): p2_contrasts_test.csv
(opa.analysis.p2_results / opa.analysis.interim) and trackb/trackb_contrasts_test.csv (opa.analysis.trackb_test), plus the gate records
<out>/gates/G0.json, G1.json (and history/) and G2.json. ``--runs`` is the series YAML of the interim run (its ``p2:`` block, the
``interim:`` block with the unit-test, learning-curve and failed-run entries); ``--trackb`` is the Track B series YAML (checked against
the run ids in the Track B table).

Steps, in this order (``--dry-run`` stops after step 4 and prints the verdicts; nothing is written):
 1. G3 (not reached: G2 failed) and a candidate G4 (decided at the end by the verifier's own result) next to the existing gates.
 2. deviations.md: every pre-registered deviation file, the plan revisions 2-5, the G1 override, the owner instructions, the failed and
    excluded runs.
 3. The interim package is regenerated over the real gates (opa.analysis.interim.run, --reuse-p2): tables, figures, PAPER_INPUT_interim.md.
 4. final_verdicts.csv / final_family.csv: the same verdict builder with the Holm adjustment over every primary that reports a p.
 5. claims/H1.json ... H13.json with opa.claims.build_claims from the test tables (H2 inconclusive, "not funded"; H9, H11, H12, H13
    not_run_gate_failed with failed_gate G2; H5 by its rule), the verdicts checked against the verdict table.
 6. tests/U1.json ... U8.json (read from the unit-test runs; U8 by its pre-registered criterion, DEV-U8).
 7. opa.bundle.assemble (prereg, budget, env, cited runs), copy_code, prereg/deviations, PAPER_INPUT.md (final), manifest.
 8. tools/verify_bundle.py's own checks (in process, as claims.verifier().verify) on a SCRATCH COPY of the bundle; the decision for G4 is
    its result. On success G4 is written passed: true and the manifest is rewritten.
Read-only on GCS, the ledger and the budget files; never uploads (``python -m opa.bundle upload --final`` is a separate command).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .. import PROJECT_DIR
from .. import budget as bud
from .. import bundle as bn
from .. import claims as cl
from . import interim as itm
from . import paper_input_interim as pii
from .tables import write_table

FIG = {"H1": ["figures/fig_h1_depth_context_test_with_dev_pilot.png"], "H5": ["figures/fig_h5_thread_depth_test_with_dev_pilot.png"],
       "H4": ["figures/fig_v3_learning_curve_dev.png"]}
P2_T, TB_T, H7_T = "tables/p2_contrasts_test.csv", "tables/trackb/trackb_contrasts_test.csv", "tables/interim_h7.csv"
G2_NOT_RUN = ("H9", "H11", "H12", "H13")             # no result, gate G2 failed: the claim is not_run_gate_failed (PLAN_V3 Section 6, G2 row)
DEV_NAMES = {"DEV-U1": "2026-11-07", "DEV-G0-noise": "2026-11-07"}


class FinalError(RuntimeError):
    pass


# ====================================================================== results and claims
def _rows(out: Path, rel: str) -> dict[str, dict]:
    p = out / rel
    if not p.exists():
        raise FinalError(f"{p} does not exist (run the interim steps first)")
    with p.open(newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def result_from_row(table_rel: str, rows: Mapping[str, Mapping], key: str, *, level: str, primary: bool = False, with_p: bool = False,
                    split: str | None = None, run_ids: Sequence[str] | None = None, description: str | None = None) -> dict:
    """A claim result from one table row (value, CI, n, p, run ids exactly as in the table)."""
    r = rows[key]
    lo, hi = itm.f(r.get("ci_lo")), itm.f(r.get("ci_hi"))
    ids = list(run_ids) if run_ids else [x for x in str(r.get("run_ids", "")).split(";") if x]
    return cl.result(key, itm.f(r["value"]), level=level, n=int(itm.f(r.get("n")) or 0), split=split or r.get("split") or "test", run_ids=ids,
                     table_ref=cl.table_ref(table_rel, key), ci95=[lo, hi] if lo is not None and hi is not None else None,
                     p_raw=itm.f(r.get("p_raw")) if with_p else None, primary=primary, test=(r.get("test") or None), description=description)


def build_results(p2: Mapping, tb: Mapping, h7: Mapping, units_run: str) -> dict[str, list[dict]]:
    """{hid: [results]} from the test tables. Keys absent from a table are left out (the rule then reports the leaf missing)."""
    R: dict[str, list[dict]] = {}

    def add(h, table, rows, key, **kw):
        if key in rows:
            R.setdefault(h, []).append(result_from_row(table, rows, key, **kw))

    add("H1", P2_T, p2, "S11_vs_S2_lbmhqa_late_32k_primary", level="agent", primary=True, with_p=True)
    add("H1", P2_T, p2, "S11_vs_S4_lbmhqa_late_32k_primary", level="agent")
    add("H3", P2_T, p2, "S11_vs_S8_late_bound", level="agent", primary=True, with_p=True)
    add("H5", P2_T, p2, "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross", level="agent", primary=True, with_p=True)
    add("H5", P2_T, p2, "A2_vs_A3_agent_equivalence", level="agent", with_p=True, description="TOST equivalence, margin 3 points; reported, no verdict")
    add("H5", P2_T, p2, "A2_vs_A1_pages_read_before_gold_hit", level="agent", with_p=True, description="pages read before the first gold page; reported, no verdict")
    add("H8", P2_T, p2, "descent_beam4_vs_flat_recall_at_M", level="agent", primary=True)
    add("H10", P2_T, p2, "pinned_vs_unpinned_repeat_pagein_reduction", level="agent", primary=True, with_p=True)
    add("H4", TB_T, tb, "V3_vs_B1_source_attribution", level="model", primary=True, with_p=True)
    add("H4", TB_T, tb, "V3_vs_B1_source_attribution_rank", level="model", with_p=True, description="rank-restricted accuracy; reported only")
    add("H4", TB_T, tb, "V3_own_origin_mass_fraction", level="model")
    add("H4", TB_T, tb, "V0_own_origin_mass_ratio_to_uniform", level="model")
    add("H6", TB_T, tb, "R5_vs_R1_accuracy", level="model", primary=True, with_p=True)
    add("H6", TB_T, tb, "best_remedy_recovery_fraction", level="model")
    add("H6", TB_T, tb, "best_remedy_recovery_fraction_incl_oracle", level="model",
        description="oracle upper bound (R2oracle, R7oracle included); reported separately, labelled oracle")
    for k in itm.H7_KEYS:
        if k in h7:
            R.setdefault("H7", []).append(cl.result(k, itm.f(h7[k]["value"]), level="measurement", n=0, split="n/a", run_ids=[units_run],
                                                    table_ref=cl.table_ref(H7_T, k)))
    return R


NOTES = {
    "H1": "Primary cell LB-MHQA late h=2, 32K (primary h = 2 after the G0 fallback). Sized for 7 points (design power 0.52 at the 5-point rule "
          "threshold, disclosed). S4 is the v2 baseline. A failed first attempt of one H1 container was replaced by its retry (DEV-FAILED-RUNS).",
    "H2": "Not funded (registration Section 2b): no result; inconclusive by the frozen rule.",
    "H3": "H3 cannot be supported: the R2 leaf has no result (not testable at 32K, DEV-H3-R2). The verdict follows the S11 vs S8 leg under the frozen all_of rule.",
    "H4": "V3 and B1 are the checkpoints pinned in DEV-TRAIN-PIN (stopping rule fired at step 600). B1's attribution reflects the text origin "
          "headers both models carry (registration Section 12 item 22). G2 failed on fresh dev seeds.",
    "H5": "Agent arm on MT-LB-MHQA T4 h=3 cross, 32K. The model-level arm was not run (G2 failed), so the any_of rule leaves a refuted agent arm inconclusive.",
    "H6": "Test data: LB-MHQA late h=2 and MT-LB-MHQA T4 h=2 at 16K (registration closure item 18).",
    "H7": "Deterministic measurements on synthetic chains from the post-tag units run (split n/a); the re-anchored ratio shows same-branch precision only.",
    "H8": "Both arms are at floor (recall near zero), so the verdict is non-inferiority at floor and says nothing about routing quality; the number of scores is not compared.",
    "H9": "Not run: gate G2 failed. The owner asked for an exploratory request with a cost estimate (budget/requests/H9_exploratory.md).",
    "H10": "Thrash runs with the worst observed value imputed for failures (intention to treat).",
    "H11": "Not run: gate G2 failed.",
    "H12": "Not run: gate G2 failed.",
    "H13": "Not run: gate G2 failed (PLAN_V3 Section 6). It was also not funded in the approved package (registration Section 2b).",
}
CLAIM_DEVIATIONS = {
    "H1": ["DEV-G0-noise", "DEV-PLAN-REV3", "DEV-G1-OVERRIDE", "DEV-S4-V2", "DEV-FAILED-RUNS", "DEV-OWNER-INTERIM"],
    "H3": ["DEV-H3-R2", "DEV-PLAN-REV3", "DEV-G1-OVERRIDE"],
    "H4": ["DEV-TRAIN-PIN", "DEV-PLAN-REV2", "DEV-U1", "DEV-FAILED-RUNS"],
    "H5": ["DEV-PLAN-REV2", "DEV-PLAN-REV5"],
    "H6": ["DEV-PLAN-REV5"],
    "H7": ["DEV-U8"],
    "H11": ["DEV-PLAN-REV4"],
    "H13": ["DEV-PLAN-REV5"],
}


def make_claims(out: Path, rules: dict, min_n: dict, results: Mapping, gates: Mapping, deviation_ids: Sequence[str]) -> dict[str, dict]:
    """build_claims over the real gates: H9, H11, H12 and H13 (no result, gate G2 failed) become not_run_gate_failed with failed_gate G2;
    H2 (gate G0 passed, no result) is inconclusive by the frozen rule."""
    figures = {h: [f for f in fs if (out / f).exists()] for h, fs in FIG.items()}
    return cl.build_claims(results, gates, rules=rules, min_n=min_n, figures=figures, notes=NOTES, deviation_ids=deviation_ids,
                           deviations={h: [d for d in ds if d in set(deviation_ids)] for h, ds in CLAIM_DEVIATIONS.items()})


# ====================================================================== deviations.md
def _entry(name: str, date: str, change: str, reason: str, impact: str) -> str:
    return f"### {name}\n- Date: {date}\n- Change: {change}\n- Reason: {reason}\n- Impact: {impact}\n"


def deviations_md(failed_rows: Sequence[Mapping] = ()) -> str:
    """Every deviation, plan revision, override and owner instruction (BUNDLE_SPEC "Deviations")."""
    failed_txt = "; ".join(f"{r['key']} ({r['job']}): {r['handling']}" for r in failed_rows) or "no failed or excluded run"
    E = [
        _entry("DEV-U1", "2026-11-07", "U1 is judged by a two-part definition: float64 error at most 1e-10 and fp32 error at most 4 ulps of the largest key.",
               "The protocol's absolute 1e-5 tolerance is below one fp32 ulp at real key magnitudes (the largest key is about 298).",
               "U1 passes; the literal criterion is reported and not met. Anticipated and recorded before the tag."),
        _entry("DEV-G0-noise", "2026-11-07", "G0's 'three S1 repeats agree within 2 EM points' counts as passed if every difference is a near-tie flip.",
               "Owner ruling recorded before the attempt-2 results: attempt 2 had ten near-tie flips and no harness error (EM spread 6 points at n = 50).",
               "G0 passed; the per-item flip rate is reported next to every primary contrast. Pre-tag."),
        _entry("DEV-G1-OVERRIDE", "2026-11-07", "G1 was first decided under the protocol's definition at 18:30 UTC and failed on H3's R2 criterion alone "
               "(H1 passed); the owner then overrode the gate in PLAN_V3 revision 3 and G1 was re-decided under the new definition and passed. "
               "The original decision is kept in gates/history/. The pilot estimates for H1, H2, H3, the H5 agent arm and H10 had been seen when the override was made; "
               "no result yet existed for trained Track B models, Track C, H7, H8 or H9.",
               "At 32K the H3 coverage predictor was 1.0 for 149 of 150 pilot items (verified), so its R2 test could not succeed for any theory.",
               "Without the override G1 would have failed and H4 to H9 and H11 to H13 would not have run. The new G1 needs a working S4 and H1 on the pilot."),
        _entry("DEV-PLAN-REV2", "2026-11-07", "G2 is judged after V3 and B1 reach a pre-set stopping rule on a fresh dev set (not on the 30-minute pilot); a 400-step "
               "learning-curve check is added; the H5 trend key is renamed A2_vs_A1_slope_per_thread_depth.",
               "Corrects a plan error (the pilot allowed 39 optimizer steps); the pilot results had been seen; thresholds unchanged. Before the tag.",
               "G2 judged on fresh dev seeds after the stopping rule fired at step 600."),
        _entry("DEV-PLAN-REV3", "2026-11-07", "G1 redefined (working S4 via a baseline audit, H1 on the pilot's point estimates; H3 removed from G1); H3 is tested on cells chosen by predictor variance alone.",
               "At 32K the H3 predictor is constant, so the original G1 could not pass for any theory.", "See DEV-G1-OVERRIDE. H3 keeps its frozen rule."),
        _entry("DEV-S4-V2", "2026-11-07", "S4 v2 replaced the first S4 (gist cap 300 with salvage, one-origin pages, parallel lookup of up to 5 pages, harness-failure flags).",
               "The S4 baseline audit failed: a 160-token gist cap emptied 23.5% of the gists, invisible to the failure codes.",
               "Dev tuning iteration 1 inside the protocol's tuning budget; the audit then passed; S4 v2 is the baseline on the test split."),
        _entry("DEV-PLAN-REV4", "2026-11-07", "H11's non-inferiority margin widened from 2 to 3 EM points in prereg/rules.yaml.",
               "A 2-point margin needs about 5,885 questions, which the budget cannot fund. Set before the tag and before any H11 accuracy was measured.",
               "H11 was not run (G2 failed); H13 keeps its 2-point margin."),
        _entry("DEV-PLAN-REV5", "2026-11-07", "The owner approved the 'fits $126' package: total cap $148, base envelopes above the plan's $100 limit by owner exception, "
               "BOOST $30 for more H1 test items (use b).",
               "Budget decision before the tag.", "H2 (deferred) and H13's accuracy test were not funded and are inconclusive; the robustness extras and the 64K check were not run."),
        _entry("DEV-H3-R2", "2026-11-07", "H3's R2 leaf is reported as not testable at 32K: no result is entered for it.",
               "Both candidate cells have a constant predicted coverage of 1.0, so the identity-line R2 cannot be estimated (registration Section 2b, closure item 9).",
               "H3 cannot be supported; by the frozen all_of rule it is refuted if the S11 vs S8 leg is refuted, otherwise inconclusive."),
        _entry("DEV-TRAIN-PIN", "2026-11-08", "The V3 and B1 checkpoints are pinned after the prereg-v2 tag, not before.",
               "The tag was placed before the training to the stopping rule had run. The configuration, the stopping rule and the G2 design were fixed before the tag.",
               "V3 and B1 are pinned by SHA-256 in prereg/deviations/DEV-TRAIN-PIN.md (step 600); no checkpoint was chosen by test results."),
        _entry("DEV-U8", "2026-11-08", "U8 is reported by its pre-registered criterion (TV distance equals the evicted router mass, tolerance 1e-5): met. "
               "The harness's own flag is false because an auxiliary SDPA-vs-(a_paged @ v) check exceeds 1e-5 absolute.",
               "The auxiliary check was added by the agent before the tag to tie U8 to the real kernel; it compares two fp32 implementations.",
               "tests/U8.json records passed: true by the pre-registered criterion with the harness flag and the auxiliary value stated literally; no re-run for a pass."),
        _entry("DEV-OWNER-INTERIM", "2026-11-08", "Owner instruction (dated 2026-11-07, received 2026-11-08): a preprint on Track A, an interim Track A package, Track A first on contested GPU "
               "capacity, vLLM max_containers 8, reservations at 1.5 times measured cost for measured job types, B1 started in parallel with V3, H9 not run if G2 fails "
               "(an exploratory request instead).",
               "Time to the preprint; execution stays as pre-registered.", "No rule, gate, threshold or approval changed; no money outside the approved envelopes."),
        _entry("DEV-FAILED-RUNS", "2026-11-08", "Failed and excluded runs: " + failed_txt + ".",
               "A storage precondition failure (HTTP 412) ended one H1 container's first attempt; a V3 chunk was launched on code from before a fix.",
               "The retry is used in place of the first H1 attempt (sensitivity in tables/interim_rerun_sensitivity.csv); the pre-fix V3 chunk is excluded; "
               "nothing was deleted; both stay in runs/index.csv history and the ledger."),
    ]
    return "\n".join(E)


# ====================================================================== unit tests
U8_PREREG_CRITERION = ("Paging identity: total-variation distance between full and paged attention equals the evicted router mass "
                       "(tolerance <= 1e-5 in fp32; protocol 5.1 table as adopted by the registration Section 9)")


def unit_records(units: Mapping) -> list[dict]:
    """tests/U<i>.json records from the unit-test runs; U8 by its pre-registered criterion (DEV-U8)."""
    recs = []
    for i in range(1, 9):
        u = (units.get("U") or {}).get(f"U{i}")
        if not u or not u.get("test"):
            raise FinalError(f"no unit-test record for U{i}")
        rec = dict(u["test"])
        if i == 1:
            rec["deviation"] = "DEV-U1"
        if i == 8:
            m = rec.get("metrics") or {}
            obs, tol = m.get("tv_vs_evicted_router_mass_max_abs_err"), m.get("tolerance", 1e-5)
            if obs is None or not obs <= tol:
                raise FinalError("U8: the pre-registered paging identity is not met")
            rec["harness_flag"] = bool(rec.get("passed"))
            rec["harness_criterion"] = rec.get("criterion", "")           # the literal harness text, unchanged
            rec["preregistered_criterion"] = U8_PREREG_CRITERION
            rec["criterion"] = U8_PREREG_CRITERION
            rec["passed"] = True
            rec["details"] = ("Pre-registered criterion met (max abs error between the TV distance and the evicted router mass is within the "
                              "tolerance, DEV-U8). The harness flag is false because the auxiliary SDPA-vs-(a_paged @ v) check exceeds 1e-5 absolute; "
                              "reported literally. " + str(rec.get("details", "")))
            rec["deviation"] = "DEV-U8"
        rec.setdefault("run_id", u["run_id"])
        recs.append(rec)
    return recs


# ====================================================================== the build
def final_tables(out: Path, contrasts: Mapping, units: Mapping, rules: dict, min_n: Mapping, g2: Mapping | None) -> dict:
    v = final_tables_dry(contrasts, units, rules, min_n, g2)
    ren = lambda r: {("p_holm" if k == "p_holm_provisional" else k): x for k, x in r.items()}          # noqa: E731
    write_table(out / "tables" / "final_verdicts.csv", [ren(r) for r in v["rows"]], columns=[("p_holm" if c == "p_holm_provisional" else c) for c in itm.VERDICT_COLUMNS])
    write_table(out / "tables" / "final_family.csv", [ren(r) for r in v["family"]],
                columns=["key", "hypothesis", "primary_key", "p_raw", "p_holm", "family_m", "status", "holm_status", "family"])
    return v


def check_verdicts(claims: Mapping[str, Mapping], rows: Sequence[Mapping]) -> list[str]:
    """The claim verdict against the verdict table's hypothesis row, for the hypotheses that have results."""
    by = {r["key"]: r for r in rows if r["kind"] == "hypothesis"}
    out = []
    for h, c in claims.items():
        if c["results"] and c["verdict"] != by[h]["verdict_interim"]:
            out.append(f"{h}: claim verdict {c['verdict']} but the verdict table says {by[h]['verdict_interim']}")
    return out


def verify_scratch(out: Path, repo: Path, scratch: Path | None = None) -> tuple[list[str], list[str], Path]:
    """The verifier's checks (tools/verify_bundle.py, loaded read-only) on a scratch COPY of the bundle. The copy always goes to a NEW folder
    (under ``scratch`` if given, else a temp dir); nothing is deleted."""
    import uuid
    base = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="opa_bundle_verify_"))
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"v{uuid.uuid4().hex[:6]}" / "bundle"           # short names: Windows paths over 260 characters fail in the copy
    shutil.copytree(out, target)
    rep = cl.verifier().verify(target, repo)
    return list(rep.notes), list(rep.fails), target


STAGE_DIRS = ("tables", "figures", "gates", "code")
_DRYRUN = [r"^tables/interim_[a-z_0-9]*_dev\.csv$", r"^tables/p2_(contrasts|series|exclusions)_dev\.csv$", r"^tables/fig_(h1|h5)_[a-z_]*_dev\.csv$",
           r"^figures/fig_(h1|h5)_[a-z_]*_dev\."]


def stage_layout(src: Path, out: Path) -> list[str]:
    """Copy the specified parts of the interim bundle into the new final folder (never deletes anything). Returns the staged paths."""
    import re
    if out.exists() and any(out.iterdir()):
        raise FinalError(f"{out} is not empty: choose a new folder for the final bundle")
    staged = []
    for d in STAGE_DIRS:
        for p in sorted((src / d).rglob("*")) if (src / d).exists() else []:
            if not p.is_file():
                continue
            rel = p.relative_to(src).as_posix()
            if any(re.search(pat, rel) for pat in _DRYRUN):
                continue
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            staged.append(rel)
    return staged


def run(args: argparse.Namespace, store=None) -> int:
    src = Path(getattr(args, "src", None) or "bundle").resolve()
    out = Path(args.out).resolve()
    root = Path(getattr(args, "project_root", None) or PROJECT_DIR).resolve()
    series, icfg = itm.load_runs_yaml(args.runs)
    rules, min_n = cl.load_prereg(root / "prereg")
    store = store if store is not None else itm.make_store(getattr(args, "creds", None), root)
    p2 = _rows(src, P2_T)
    tb = _rows(src, TB_T)
    tbs = _rows(src, "tables/trackb/trackb_series_test.csv")
    if getattr(args, "trackb", None):
        want = {i for ids in itm.p2.load_mapping(args.trackb).values() for i in ids}
        have = {i for r in tbs.values() for i in str(r.get("run_ids", "")).split(";") if i}
        if want != have:
            raise FinalError(f"Track B runs in {args.trackb} differ from the Track B table: only in yaml {sorted(want - have)[:3]}, only in table {sorted(have - want)[:3]}")
    runs_dir = root / "runs"
    units = itm.load_units(runs_dir, store, icfg)
    if not (units.get("runs") or {}).get("u6_u8_h7"):
        raise FinalError("the series yaml has no interim.units_u6_u8_h7 run")
    g2 = itm.load_g2(src)
    if g2 is None or g2.get("passed") is not False:
        raise FinalError("bundle/gates/G2.json must exist and show passed: false for this build (it decides the not-run hypotheses)")
    contrasts = {**p2, **tb}
    v = final_tables_dry(contrasts, units, rules, min_n, g2)
    for r in v["rows"]:
        if r["kind"] == "hypothesis":
            print(f"{r['key']:<4} {r['verdict_interim']}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    staged = stage_layout(src, out)
    print(f"staged {len(staged)} files from {src} into {out}")
    # 1. gates
    gates = cl.read_gates(out)
    g3 = cl.unreached_gate("G3", "G2")
    cand = pending_g4()
    cl.write_gates(out, {"G3": g3, "G4": cand})
    gates = {**gates, "G3": g3, "G4": cand}
    # 2. deviations
    failed_rows, sens_rows = itm.failed_run_rows(store, runs_dir, icfg.get("failed_runs") or [], "test")
    (out / "deviations.md").write_text(deviations_md(failed_rows), encoding="utf-8", newline="\n")
    dev_ids = cl.read_deviation_ids(out)
    # 3. the interim tables and figures over the real gates
    ns = argparse.Namespace(split="test", runs=args.runs, out=str(out), runs_dir=None, creds=getattr(args, "creds", None), reuse_p2=True,
                            dry_run=False, no_dev_context=False, project_root=str(root), skip_paper=True)
    if itm.run(ns, store=store) != 0:
        raise FinalError("the interim regeneration reported problems (see above)")
    # 4. final verdict tables, 5. claims
    final_tables(out, contrasts, units, rules, min_n, g2)
    results = build_results(p2, tb, _rows(out, "tables/interim_h7.csv"), units["runs"]["u6_u8_h7"])
    claims = make_claims(out, rules, min_n, results, gates, dev_ids)
    problems = check_verdicts(claims, final_tables_dry(contrasts, units, rules, min_n, g2)["rows"])
    if problems:
        raise FinalError("; ".join(problems))
    cl.write_claims(out, claims)
    # 6. tests
    cl.write_unit_tests(out, unit_records(units), runs_index=root / "runs" / "index.csv", deviation_ids=dev_ids, same_run_deviation=None)
    # 7. assemble, code, prereg deviations, PAPER_INPUT, manifest
    assemble_all(out, root)
    write_final_documents(out, root, rules, min_n, series, icfg, claims)          # G4 still pending in this first pass
    # 8. the verifier decides G4: verify a scratch copy, write the definitive G4, regenerate the documents and the manifest, verify the final bytes
    notes, fails, scratch = verify_scratch(out, root, getattr(args, "scratch", None))
    g4 = decide_g4(fails, scratch)
    cl.write_gates(out, {"G4": g4})
    write_final_documents(out, root, rules, min_n, series, icfg, claims)
    notes2, fails2, scratch2 = verify_scratch(out, root, getattr(args, "scratch", None))
    if sorted(fails2) != sorted(fails):
        raise FinalError("the verifier disagrees between the decision pass and the final bytes:\n  first: " + "; ".join(fails) + "\n  final: " + "; ".join(fails2))
    for n in notes2:
        print("note:", n)
    if fails2:
        print(f"BUNDLE NOT VERIFIED on the scratch copy {scratch2}: {len(fails2)} failure(s)")
        for f in fails2:
            print("  " + f)
        return 1
    print(f"BUNDLE VERIFIED (scratch copy {scratch2}); G4 passed")
    return 0


def final_tables_dry(contrasts, units, rules, min_n, g2) -> dict:
    return itm.build_verdicts(contrasts, units, rules, min_n, "test", g2=g2, holm="final")


G4_CRITERION = "tools/verify_bundle.py checks (D1-D13) on a scratch copy of the assembled bundle"


def pending_g4() -> dict:
    return cl.gate("G4", None, [], ["tools/verify_bundle.py"], "Decision pending: written after the verifier has run on a scratch copy of this bundle.")


def decide_g4(fails: Sequence[str], scratch: Path | str, ts: str | None = None) -> dict:
    """G4 (paper-ready) from the verifier's own result, never presumed: passed true iff it reported no failure."""
    ts = ts or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not fails:
        return cl.gate("G4", True, [cl.criterion(G4_CRITERION, "BUNDLE VERIFIED, 0 failures", f"BUNDLE VERIFIED, 0 failures (scratch copy {scratch}, {ts})", True)],
                       ["tools/verify_bundle.py"], "Paper-ready: the verifier's checks pass on the assembled bundle.", decided_at=ts)
    return cl.gate("G4", False, [cl.criterion(G4_CRITERION, "BUNDLE VERIFIED, 0 failures", {"failures": list(fails), "scratch": str(scratch), "at": ts}, False)],
                   ["tools/verify_bundle.py"], f"Not paper-ready: the verifier reported {len(fails)} failure(s) (listed in the criterion).", decided_at=ts)


def refresh_gate_tables(out: Path) -> None:
    """Rewrite the gate tables from bundle/gates (after G4 is decided) so the documents state the decided gates."""
    g = itm.gates_tables(out)
    t = out / "tables"
    write_table(t / "interim_gates.csv", g["gates"], columns=["key", "status", "decided_at", "n_criteria", "n_met", "note"])
    write_table(t / "interim_gate_criteria.csv", g["criteria"], columns=["key", "gate", "name", "threshold", "observed_value", "observed_text", "met"])
    if g["history"]:
        write_table(t / "interim_gate_history.csv", g["history"], columns=["key", "gate", "status", "decided_at", "n_criteria", "n_met",
                                                                            "failed_criteria", "superseded_by", "note"])


def refresh_cost_tables(out: Path) -> None:
    """Cost tables from the ledger copy inside the bundle (so the tables and budget/ledger.csv cannot drift)."""
    c = itm.costs_tables(bud.load_approvals(out / "budget" / "approvals.yaml"), bud.read_ledger(out / "budget" / "ledger.csv"))
    t = out / "tables"
    write_table(t / "interim_costs_envelope.csv", c["envelope"], columns=["key", "approved_usd", "spent_usd", "reserved_usd", "remaining_usd", "as_of"])
    write_table(t / "interim_costs_group.csv", c["group"], columns=["key", "hypotheses", "spent_usd", "n_jobs", "envelopes", "as_of"])


def write_final_documents(out: Path, root: Path, rules: dict, min_n: Mapping, series: Mapping, icfg: Mapping, claims: Mapping) -> Path:
    refresh_gate_tables(out)
    refresh_cost_tables(out)
    md = pii.write_paper_input_interim(out, "test", rules=rules, min_n=min_n, args_runs=series, icfg=icfg, root=root,
                                       final={"claims": claims, "verdict_table": "final_verdicts.csv", "family_table": "final_family.csv"})
    probs = pii.check_refs(md, out)
    if probs:
        raise FinalError("PAPER_INPUT.md reference problems:\n  " + "\n  ".join(probs))
    bn.manifest(out)
    return md


def assemble_all(out: Path, root: Path) -> list[str]:
    problems = bn.assemble(project=root, bundle=out)
    bn.copy_code(project=root, bundle=out)
    dd = root / "prereg" / "deviations"
    if dd.exists():
        (out / "prereg" / "deviations").mkdir(parents=True, exist_ok=True)
        for f in sorted(dd.glob("*.md")):
            shutil.copy2(f, out / "prereg" / "deviations" / f.name)
    if problems:
        print("assemble problems:", *problems, sep="\n  ")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--trackb", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="bundle", help="the interim bundle to read (default: bundle)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--creds", default=None)
    ap.add_argument("--scratch", default=None, help="directory for the scratch copy the verifier runs on (default: a temp dir)")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except (FinalError, itm.InterimError, cl.ClaimError, ValueError, PermissionError) as e:
        print(f"final_bundle FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
