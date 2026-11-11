"""Interim Track A package: one command from the P2 runs to the verdict, cost, gate, deviation and Track B tables,
the figures and the PAPER_INPUT_interim.md.

  python -m opa.analysis.interim --split test --runs <series yaml> --out bundle [--dry-run]
  python -m opa.analysis.interim --split dev  --runs jobs/interim/series_dev.yaml --out bundle [--dry-run]

``--runs`` is a YAML mapping. Either the flat form of opa.analysis.p2_results (series -> [run ids]) or

  p2:                    # series -> [run ids]  (embed-lb sk-lb h1-lb h1-s4 embed-mt sk-mt h5-mt h10-pin h10-nopin)
  interim:               # all optional
    units_u1_u5: <run id>            # the P0 U1-U5 run
    units_u6_u8_h7: <run id>         # the U6-U8 / H7 run
    learning_curve: [<run id>, ...]  # V3 (and B1) runs that wrote learning_curve.json, in step order

What it does, in this order (``--dry-run`` stops after the verdicts and prints them; nothing is written)
1. P2 results (opa.analysis.p2_results) into <out>/tables/p2_contrasts_<split>.csv etc. ``--reuse-p2`` reads those files
   instead of recomputing them. In test mode the dev pilot contrasts are also (re)computed into
   <out>/tables/p2_dev_check/ for the dev-labelled context in the figures (``--no-dev-context`` skips this).
2. Interim verdicts under the FROZEN rules (prereg/rules.yaml, evaluated by the verifier's own eval_rule through
   opa.claims) for H1, H3, H5, H7, H8, H10 -> tables/interim_verdicts[_dev].csv. Each primary p enters a Bonferroni
   adjustment over the full pre-registered family (the rules' Holm family = primaries that report p_raw: H1 H2 H3 H4 H5
   H6 H10 H12, m = 8): p_bonferroni = min(1, m * p_raw). The rule's ``require_p_holm_below`` leaf uses it as the
   conservative interim stand-in for p_holm (every Bonferroni p is >= its Holm p, so a "supported" stays supported at
   the final Holm); the final Holm is "pending Tracks B/C". H2, H4, H6, H9, H11, H12, H13 have no interim result:
   they are listed with their status, not given a verdict.
3. Units U1-U8 and H7 measurements, costs to date (budget ledger, read-only), gates, deviations, Track B learning curve
   and Track C status tables.
4. Figures (PDF + PNG + sidecar + source CSV), the release archive is a separate command (opa.analysis.release).

Split hygiene. ``--split dev`` writes only file names that contain "dev" and labels every verdict ``dev:...``; it
evaluates the dev pilot only to exercise the pipeline. ``--split test`` never reads dev rows into a verdict; dev rows
reach the figures only through files whose names contain "dev" and with a ``split`` column, drawn with a different marker.
Read-only on GCS and on the ledger; spends nothing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .. import PROJECT_DIR, stats
from .. import budget as bud
from .. import claims as cl
from . import p2_results as p2
from . import pilot as pl
from .tables import read_table, write_table

# ---------------------------------------------------------------------------------------------- constants
SCOPE = ["H1", "H3", "H5", "H7", "H8", "H10"]          # hypotheses that get an interim verdict
HOLM_PENDING = "pending Tracks B/C"
H7_KEYS = ["reanchored_error_growth_ratio_depth40_vs_depth1_bf16", "global_fail_depth_bf16",
           "global_fail_depth_fp16", "global_fail_depth_fp32"]
TRACKB_TEST_HYPS = ("H4", "H6")                 # get verdict rows when the opa.analysis.trackb_test tables exist
G2_FAILED_STATUS = "not_run_gate_failed (G2)"
NOT_FUNDED_STATUS = "inconclusive (not funded)"
MODEL_HYPS = ("H4", "H6", "H11", "H12", "H13")
H9_FAILED_STATUS = "not_run_gate_failed (G2); owner request budget/requests/H9_exploratory.md"


def g2_state(g2: Mapping | None) -> str | None:
    """"passed" / "failed" from the gate record bundle/gates/G2.json (None: no record yet)."""
    if not g2 or g2.get("passed") is None:
        return None
    return "passed" if g2.get("passed") else "failed"


def outside_status(g2: Mapping | None = None) -> dict:
    """hypothesis -> (status, note) for the hypotheses that get no verdict row here; data-driven from the G2 record."""
    st = g2_state(g2)
    run_note = "being evaluated on the test split (G1 passed); its verdict rows appear when bundle/tables/trackb/trackb_contrasts_test.csv exists"
    d = {
        "H2": (NOT_FUNDED_STATUS, "deferred, not in the funded package (prereg Section 2b): the frozen rule gives inconclusive "
                                  "(required result missing)"),
        "H4": ("running on the test split (gate G1)", "V3 vs B1 attribution and the own-origin measurements: " + run_note),
        "H6": ("running on the test split (gate G1)", "contextual-dependence remedies: " + run_note),
    }
    if st == "failed":
        why = "gate G2 failed (bundle/gates/G2.json): not run, by the pre-registered gate rule"
        d["H9"] = (H9_FAILED_STATUS, why + "; the owner asked for an exploratory request with a cost estimate instead")
        for h in ("H11", "H12"):
            d[h] = (G2_FAILED_STATUS, why)
        d["H13"] = (G2_FAILED_STATUS, "gate G2 failed (bundle/gates/G2.json): not run; also not funded in the approved package (registration Section 2b)")
    else:
        passed = st == "passed"
        d["H9"] = ("pending_track_c (G2 passed)" if passed else "conditional_on_g2",
                   "Track C; runs only if G2 passes and E5 is approved" + (" (G2 passed)" if passed else " (owner instruction: if G2 fails, no H9)"))
        for h in ("H11", "H12"):
            d[h] = ("pending_track_b (G2 passed)" if passed else "pending_track_b", "needs trained V3 / B1" + (" (G2 passed)" if passed else " and G2"))
        d["H13"] = (NOT_FUNDED_STATUS, "copy-reuse accuracy margin not funded (prereg Section 2b): inconclusive by the frozen rule")
    return d


STATUS_OUTSIDE_SCOPE = outside_status(None)      # kept for callers that want the default (no G2 record)
VERDICT_COLUMNS = ["key", "hypothesis", "kind", "rule_key", "value", "ci_lo", "ci_hi", "n", "min_n", "p_raw",
                   "p_bonferroni", "family_m", "holm_status", "verdict_interim", "verdict_basis", "verdict_ignoring_min_n",
                   "holm_sensitive", "data_split", "level", "rule", "notes", "verdict_reason", "p_holm_provisional"]
R2_KEY = "R2_predicted_vs_observed_success"
PENDING_HOLM = "pending_final_holm"
FLOOR = 0.05          # a non-inferiority leaf whose two arm rates are both below this is "at floor"


class InterimError(ValueError):
    """A problem that stops the package (nothing is written)."""


# ---------------------------------------------------------------------------------------------- helpers
def f(x) -> float | None:
    """Float of a CSV/YAML cell; None for empty / nan / None."""
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _g(x) -> str:
    """3 significant digits, scientific for small values: the notes never carry a plain decimal that would need a reference."""
    v = f(x)
    return "n/a" if v is None else f"{v:.3g}"


def suffix(split: str) -> str:
    return "_dev" if split == "dev" else ""


def check_split(split: str) -> None:
    if split not in ("dev", "test"):
        raise InterimError(f"split must be 'dev' or 'test', got {split!r}")


def check_name(split: str, name: str, dev_data: bool = False) -> str:
    """Refuse a file name that mixes the splits: dev mode writes only names containing 'dev', test mode no name that
    contains 'dev' except files made of dev-split data (``dev_data``), whose names carry it on purpose."""
    check_split(split)
    low = name.lower()
    has = lambda tok: re.search(rf"(^|[_./\-]){tok}($|[_./\-])", low) is not None      # a whole token ("deviations" is not "dev")  # noqa: E731
    if split == "dev" and not has("dev"):
        raise InterimError(f"split dev: refusing to write {name!r}: the name must contain 'dev'")
    if split == "test" and has("dev") and not dev_data:
        raise InterimError(f"split test: refusing to write {name!r}: it contains 'dev'")
    if split == "dev" and has("test"):
        raise InterimError(f"split dev: refusing to write {name!r}: it contains 'test'")
    return name


def load_runs_yaml(src: Any) -> tuple[dict, dict]:
    """(p2 series mapping, interim options) from the --runs file (flat p2_results form or p2: / interim: form)."""
    d = p2.load_mapping(src)
    if "p2" in d or "interim" in d:
        return dict(d.get("p2") or {}), dict(d.get("interim") or {})
    return d, {}


def bonferroni(p: float | None, m: int) -> float | None:
    return None if p is None else min(1.0, m * float(p))


def family(rules: dict) -> list[str]:
    """The Holm family of the frozen rules: hypotheses whose primary leaf has require_p_holm_below (= primaries that
    report p_raw), in numeric order."""
    return [h for h in cl.hypothesis_ids(rules) if cl.holm_required(rules, h)]


def make_store(creds: str | None = None, root: Path | str | None = None):
    """GcsStore with GCP_SA_JSON set in-process from the key file (never printed). None if no key is found."""
    cands = [creds, pl.credentials_path(root) if root else pl.credentials_path()]
    here = Path(PROJECT_DIR)
    cands.append(pl.credentials_path(here.parent / "opa_experiments"))
    for c in cands:
        if c and Path(c).exists():
            if not os.environ.get("GCP_SA_JSON"):
                os.environ["GCP_SA_JSON"] = Path(c).read_text(encoding="utf-8")
            return pl.GcsStore()
    if os.environ.get("GCP_SA_JSON"):
        return pl.GcsStore()
    return None


# ---------------------------------------------------------------------------------------------- series discovery
P2_JOB = re.compile(r"^p2-(embed-lb|sk-lb|embed-mt|sk-mt|h1-lb|h1-s4|h5-mt|h10-pin|h10-nopin)-c(\d+)$")
TRAIN_JOB = re.compile(r"^p4-(v3|b1)-c(\d+)$")
_STAMP = re.compile(r"-(\d{14})-[0-9a-f]{7}-[0-9a-f]{4}$")


def discover_series(runs_dir: Path | str, split: str = "test") -> dict:
    """Series YAML content from the collected runs in ``runs_dir`` (runs/<run id>/job.yaml + result.json): for each P2 job
    name the latest run of that name with status ok and the given split. Returns {"p2", "interim", "not_ok", "repeats"} (``repeats``: job names with more than one ok run, all ids listed;
    the latest is used);
    ``not_ok`` lists runs of those names that are not status ok (running, failed, lost), which are left out."""
    runs_dir = Path(runs_dir)
    best: dict[str, tuple[str, str]] = {}           # job name -> (stamp, run id) of the latest ok run
    all_ok: dict[str, list[str]] = {}
    not_ok = []
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        jp = d / "job.yaml"
        if not jp.exists():
            continue
        try:
            job = yaml.safe_load(jp.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        name = str(job.get("name", ""))
        interesting = P2_JOB.match(name) or TRAIN_JOB.match(name) or name in ("p0-trackb-units", "p3-units-u6-u8-h7", "p3-learning-curve-v3")
        if not interesting or job.get("split") != (split if (P2_JOB.match(name)) else job.get("split")):
            continue
        rp = d / "result.json"
        try:
            status = str(json.loads(rp.read_text(encoding="utf-8")).get("status")) if rp.exists() else "no result.json"
        except ValueError:
            status = "unreadable result.json"
        if status != "ok":
            not_ok.append(f"{name}: {d.name} status={status}")
            continue
        m = _STAMP.search(d.name)
        stamp = m.group(1) if m else ""
        all_ok.setdefault(name, []).append(d.name)
        if name not in best or stamp > best[name][0]:
            best[name] = (stamp, d.name)
    p2s: dict[str, list] = {}
    for name in sorted((n for n in best if P2_JOB.match(n)), key=lambda n: (P2_JOB.match(n).group(1), int(P2_JOB.match(n).group(2)))):
        p2s.setdefault(P2_JOB.match(name).group(1), []).append(best[name][1])
    lc = [best[n][1] for n in sorted((n for n in best if n == "p3-learning-curve-v3" or TRAIN_JOB.match(n)),
                                     key=lambda n: (n != "p3-learning-curve-v3", n))]
    interim = {}
    if "p0-trackb-units" in best:
        interim["units_u1_u5"] = best["p0-trackb-units"][1]
    if "p3-units-u6-u8-h7" in best:
        interim["units_u6_u8_h7"] = best["p3-units-u6-u8-h7"][1]
    if lc:
        interim["learning_curve"] = lc
    repeats = {n: sorted(ids) for n, ids in all_ok.items() if len(ids) > 1}
    failed = []
    for line in not_ok:
        name, rest = line.split(": ", 1)
        rid, status = rest.rsplit(" status=", 1)
        if status != "no result.json":                      # a run still running is not a failure
            failed.append({"run_id": rid, "job": name, "replaced_by": best[name][1] if name in best else None})
    if failed:
        interim["failed_runs"] = failed
    return {"p2": p2s, "interim": interim, "not_ok": not_ok, "repeats": repeats, "failed": failed}


# ---------------------------------------------------------------------------------------------- measurements (runs)
def _read_json_local_or_store(runs_dir: Path, store, rid: str, name: str) -> dict | None:
    p = runs_dir / rid / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if store is not None:
        try:
            return json.loads(store.get(pl.run_uri(store, rid) + "/" + name))
        except (KeyError, ValueError):
            return None
    return None


def load_units(runs_dir: Path | str, store, cfg: Mapping) -> dict:
    """{"U1": {...}, ..., "U8": {...}, "h7": {key: value}, "runs": {...}} from the unit-test runs. Pass flags come from
    each run's metrics.json (local runs/ copy, else GCS); details and the H7 values from the run's tests/U*.json and
    h7.json in GCS when a store is given. Missing runs / files give empty entries, never an invented value."""
    runs_dir = Path(runs_dir)
    out: dict = {"runs": {}, "h7": {}, "U": {}}
    for tag, rid_key, ids in (("u1_u5", "units_u1_u5", range(1, 6)), ("u6_u8_h7", "units_u6_u8_h7", range(6, 9))):
        rid = cfg.get(rid_key)
        if not rid:
            continue
        out["runs"][tag] = rid
        m = _read_json_local_or_store(runs_dir, store, rid, "metrics.json") or {}
        for i in ids:
            rec = {"run_id": rid, "code_flag": m.get(f"U{i}_passed"), "summary": m.get(f"U{i}_summary") or {}}
            t = store is not None and _read_store_json(store, rid, f"tests/U{i}.json")
            if t:
                rec["test"] = t
            out["U"][f"U{i}"] = rec
        if tag == "u6_u8_h7":
            for k in H7_KEYS:
                if m.get(k) is not None:
                    out["h7"][k] = float(m[k])
            out["h7_run"] = rid
    return out


def _read_store_json(store, rid: str, rel: str) -> dict | None:
    try:
        return json.loads(store.get(pl.run_uri(store, rid) + "/" + rel))
    except (KeyError, ValueError):
        return None


def units_rows(units: Mapping, deviations_dir: Path | None = None) -> list[dict]:
    """tables/interim_units: one row per U1-U8. ``passed`` follows the pre-registered criterion as the run's test record
    states it (U1 under DEV-U1); ``code_flag`` is the harness's own flag; ``observed`` / ``threshold`` are the numbers
    behind the criterion where the test record has them."""
    rows = []
    dev_u8 = (Path(deviations_dir) / "DEV-U8.md") if deviations_dir else None
    for i in range(1, 9):
        u = (units.get("U") or {}).get(f"U{i}")
        key = f"U{i}"
        if not u:
            rows.append({"key": key, "unit": key, "status": "no result", "passed": None, "code_flag": None,
                         "observed": None, "threshold": None, "criterion": "", "deviation": "", "run_id": "", "notes": "run not given"})
            continue
        t = u.get("test") or {}
        s = u.get("summary") or {}
        passed, flag = t.get("passed", u["code_flag"]), u["code_flag"]
        observed = threshold = None
        dev, notes = "", ""
        aux = aux_name = ulps = None
        if key == "U1":
            observed, threshold = s.get("fp32_max_abs_err"), s.get("tolerance")
            aux, aux_name, ulps = s.get("float64_max_abs_err"), "float64 max abs error", s.get("fp32_err_in_ulps_of_max_key")
            dev = "DEV-U1"
            notes = ("passes the two-part definition of DEV-U1 (float64 error and fp32 error in ulps of the largest key, see the aux columns); "
                     "the literal 1e-5 absolute criterion is reported, not met: the observed column is the fp32 error") if s else "DEV-U1"
        elif key == "U2":
            observed, notes = s.get("max_abs_logit_diff"), "criterion: logits identical (max abs diff exactly 0)"
        elif key == "U3":
            observed = (s.get("single_origin") if not isinstance(s.get("single_origin"), dict)
                        else s["single_origin"].get("max_abs_logit_diff"))
            notes = "criterion: logits identical (max abs diff exactly 0)"
        elif key == "U4":
            observed, notes = s.get("max_abs_logit_diff"), "criterion: logits identical; live check changes logits"
        elif key == "U5":
            observed, threshold = s.get("a_primitives_fp32_max_abs_err"), 1e-6
            notes = "primitives fp32 error (a); LCA distances float64 and fp32 port vs reference (b, c) in tests/U5.json"
        elif key == "U6":
            observed, threshold = (units.get("h7") or {}).get("reanchored_error_growth_ratio_depth40_vs_depth1_bf16"), 2.0
            notes = "re-anchored bf16 error growth ratio depth 40 vs 1; global fail depths within one step of the predicted ones"
        elif key == "U7":
            m = t.get("metrics") or {}
            observed, threshold = m.get("factorization_max_abs_err"), m.get("tolerance", 1e-5)
            aux, aux_name = m.get("sdpa_output_max_abs_err"), "SDPA output max abs error"
            notes = "attention = router mass x local softmax; fp32 tolerance in the threshold column"
        elif key == "U8":
            m = t.get("metrics") or {}
            observed, threshold = m.get("tv_vs_evicted_router_mass_max_abs_err"), m.get("tolerance", 1e-5)
            aux, aux_name = m.get("paged_sdpa_output_max_abs_err"), "auxiliary SDPA-vs-(a_paged @ v) max abs error"
            dev = "DEV-U8"
            # the pre-registered identity (TV vs evicted router mass) decides; the auxiliary check is reported literally
            passed = bool(observed is not None and threshold is not None and observed <= threshold)
            notes = (f"pre-registered paging identity met (observed column is within the threshold column); the harness flag is {flag} "
                     "because the auxiliary check (aux columns) exceeds the 1e-5 absolute tolerance "
                     "(DEV-U8: reported literally, no re-run for a pass)")
        rows.append({"key": key, "unit": key, "status": "pass" if passed else "fail", "passed": passed, "code_flag": flag,
                     "observed": f(observed), "threshold": f(threshold),
                     "aux_observed": f(aux), "aux_name": aux_name or "", "ulps": f(ulps),
                     "criterion": (t.get("criterion") or ""), "deviation": dev, "run_id": u["run_id"], "notes": notes})
    return rows


def aux_u8(units: Mapping) -> float | None:
    t = ((units.get("U") or {}).get("U8") or {}).get("test") or {}
    return f((t.get("metrics") or {}).get("paged_sdpa_output_max_abs_err"))


# ---------------------------------------------------------------------------------------------- verdicts
def _res_from_row(row: Mapping, p_holm: float | None, force_test: bool) -> dict:
    lo, hi = f(row.get("ci_lo")), f(row.get("ci_hi"))
    return {"key": row["key"], "value": f(row["value"]), "ci95": [lo, hi] if lo is not None and hi is not None else None,
            "n": int(f(row.get("n")) or 0), "p_holm": p_holm, "split": "test" if force_test else row.get("split")}


def _eval(rule: dict, results: Mapping, min_n: Mapping, hid: str) -> tuple[str, list[str]]:
    v = cl.verifier()
    rep = v.Report()
    out = v.eval_rule(rule, dict(results), dict(min_n), rep, hid)
    return out, list(rep.fails)


def _ignoring_min_n(rule: dict) -> dict:
    r = copy.deepcopy(rule)
    for leaf in cl.rule_leaves(r):
        leaf["measurement"] = True
    return r


def _fmt(x, d=2) -> str:
    return "n/a" if x is None else f"{x:.{d}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def explain(leaf: dict, res: Mapping | None, min_n: Mapping, note_missing: str = "", holm_label: str = "Bonferroni stand-in for Holm") -> str:
    """Why a leaf came out as it did (frozen thresholds, in words)."""
    rt = cl.rule_text(leaf)
    if res is None:
        return f"{leaf['key']}: no result ({note_missing or 'result missing'}) -> inconclusive"
    if not leaf.get("measurement"):
        need = (min_n or {}).get(leaf["key"])
        if need is not None and res["n"] < need:
            return f"{leaf['key']}: n {res['n']} < pre-registered min_n {need} -> inconclusive"
    v, t = res["value"], leaf.get("threshold")
    parts = []
    if leaf["type"] in ("min_delta", "min_value"):
        parts.append(f"value {_fmt(v, 4)} {'>=' if not leaf.get('strict') else '>'} {t}: {'met' if (v > t if leaf.get('strict') else v >= t) else 'not met'}")
    elif leaf["type"] in ("max_value", "max_delta"):
        parts.append(f"value {_fmt(v, 4)} {'<=' if not leaf.get('strict') else '<'} {t}: {'met' if (v < t if leaf.get('strict') else v <= t) else 'not met'}")
    elif leaf["type"] == "within":
        parts.append(f"value {_fmt(v, 4)} in [{leaf['lo']}, {leaf['hi']}]: {'met' if leaf['lo'] <= v <= leaf['hi'] else 'not met'}")
    ci = res.get("ci95")
    if leaf.get("require_ci_excludes_zero"):
        parts.append(f"CI lower bound {_fmt(ci[0], 4) if ci else 'n/a'} > 0: {'met' if ci and ci[0] > 0 else 'not met'}")
    if "require_ci_lower_above" in leaf:
        x = leaf["require_ci_lower_above"]
        parts.append(f"CI lower bound {_fmt(ci[0], 4) if ci else 'n/a'} > {x}: {'met' if ci and ci[0] > x else 'not met'}")
    if "require_p_holm_below" in leaf:
        ph = res.get("p_holm")
        parts.append(f"{holm_label} p {_fmt(ph, 6)} < {leaf['require_p_holm_below']}: "
                     f"{'met' if ph is not None and ph < leaf['require_p_holm_below'] else 'not met'}")
    return f"{leaf['key']}: " + "; ".join(parts)


def reason(leaf: dict, res: Mapping | None, min_n: Mapping, note_missing: str = "", holm_label: str = "Bonferroni stand-in") -> str:
    """Number-free reason for a leaf outcome (the numbers sit in their own columns, which PAPER_INPUT references)."""
    if res is None:
        return note_missing or "result missing"
    if not leaf.get("measurement"):
        need = (min_n or {}).get(leaf["key"])
        if need is not None and res["n"] < need:
            return "n is below the pre-registered minimum"
    v, t = res["value"], leaf.get("threshold")
    parts = []
    if leaf["type"] in ("min_delta", "min_value"):
        parts.append("value meets the threshold" if (v > t if leaf.get("strict") else v >= t) else "value misses the threshold")
    elif leaf["type"] in ("max_value", "max_delta"):
        parts.append("value meets the limit" if (v < t if leaf.get("strict") else v <= t) else "value exceeds the limit")
    elif leaf["type"] == "within":
        parts.append("value is inside the band" if leaf["lo"] <= v <= leaf["hi"] else "value is outside the band")
    ci = res.get("ci95")
    if leaf.get("require_ci_excludes_zero"):
        parts.append("CI lower bound above zero" if ci and ci[0] > 0 else "CI lower bound not above zero")
    if "require_ci_lower_above" in leaf:
        parts.append("CI lower bound above the margin" if ci and ci[0] > leaf["require_ci_lower_above"] else "CI lower bound not above the margin")
    if "require_p_holm_below" in leaf:
        ph = res.get("p_holm")
        parts.append(f"{holm_label} p below the limit" if ph is not None and ph < leaf["require_p_holm_below"]
                     else f"{holm_label} p not below the limit")
    return "; ".join(parts)


def build_verdicts(contrasts: Mapping[str, Mapping], units: Mapping, rules: dict, min_n: Mapping, split: str,
                   m: int | None = None, g2: Mapping | None = None, holm: str = "bonferroni") -> dict:
    """The interim verdict rows and the Bonferroni family table.

    ``contrasts``: {key: row of p2_contrasts_<split>.csv}. Returns {"rows", "family", "fails"} (``fails`` are the
    verifier's own complaints, e.g. a rule key without a minimum n)."""
    check_split(split)
    fam = family(rules)
    m = m if m is not None else len(fam)
    force_test = split == "dev"            # dev pilot rows go through the same evaluator (it insists on split test)
    pfx = "dev:" if split == "dev" else ""
    data_split = split
    fails: list[str] = []
    primary_of = {h: rules["hypotheses"][h].get("primary_key") for h in fam}
    p_by_key: dict[str, float | None] = {}
    for h in fam:
        k = primary_of[h]
        row = contrasts.get(k)
        p_by_key[k] = f(row.get("p_raw")) if row else None
    prov = stats.holm({k: p for k, p in p_by_key.items() if p is not None})        # provisional: only the primaries that report a p now
    final = holm == "final"        # the final package: the Holm p over the primaries that report one replaces the Bonferroni stand-in
    if final:
        m = len(prov)              # family size = the number of p-values that enter the Holm adjustment
    hp_label = "final Holm" if final else HOLM_PENDING
    holm_word = "Holm" if final else "Bonferroni stand-in"

    def results_for(p_mode: str) -> dict:
        out = {}
        for k, row in contrasts.items():
            p = p_by_key.get(k)
            ph = (prov.get(k) if final else bonferroni(p, m)) if p_mode == "bonferroni" else p
            out[k] = _res_from_row({**row, "key": k}, ph, force_test)
        for k, v in (units.get("h7") or {}).items():
            out[k] = {"key": k, "value": v, "ci95": None, "n": 0, "p_holm": None, "split": "n/a"}
        return out

    res_b = results_for("bonferroni")
    res_raw = res_b if final else results_for("raw")
    rows: list[dict] = []

    def verdict_cell(v: str) -> str:
        return pfx + v

    g2s = g2_state(g2)
    outside = outside_status(g2)
    scope = list(SCOPE)
    for th in TRACKB_TEST_HYPS:        # Track B test tables present: H4 / H6 are evaluated like Track A
        if th in rules["hypotheses"] and any(k in contrasts for k in cl.rule_keys(rules, th)):
            scope.append(th)
    for hid in scope:
        h = rules["hypotheses"][hid]
        leaves = cl.rule_leaves(h["rule"])
        hv, hf = _eval(h["rule"], res_b, min_n, hid)
        hv_raw, _ = _eval(h["rule"], res_raw, min_n, hid)
        hv_nomin, _ = _eval(_ignoring_min_n(h["rule"]), res_b, min_n, hid)
        fails += hf
        leaf_out = []
        for leaf in leaves:
            k = leaf["key"]
            res = res_b.get(k)
            lv, lf = _eval(leaf, res_b, min_n, hid)
            lv_raw, _ = _eval(leaf, res_raw, min_n, hid)
            lv_nomin, _ = _eval(_ignoring_min_n(leaf), res_b, min_n, hid)
            fails += lf
            row = contrasts.get(k) or {}
            note = ""
            miss = ""
            if res is None:
                if k == R2_KEY:
                    miss = "not testable at 32K, no result reported (prereg closure item 9)"
                    note = "R2 leaf: both candidate cells have constant predicted coverage 1.0; the leaf has no result"
                elif k == "V0_own_origin_mass_ratio_to_uniform":
                    miss = "training-free test run not yet complete"
                    note = "H4 V0 leaf: the training-free test run is not yet complete"
                elif k.startswith("A2_vs_A1_model"):
                    if g2s == "failed":
                        miss = "model-level arm not run: gate G2 failed"
                        note = "model arm of the any_of rule: " + G2_FAILED_STATUS
                    else:
                        miss = "model-level result pending Track B (G2 not decided)" if g2s is None else "model-level result pending Track B (G2 passed)"
                        note = "model arm of the any_of rule: pending Track B"
                else:
                    miss = "result missing"
            if row.get("note"):
                note = (note + "; " if note else "") + str(row["note"])
            p_raw = p_by_key.get(k)
            floor = ""
            if "require_ci_lower_above" in leaf and row and f(row.get("em_a")) is not None and f(row.get("em_b")) is not None \
                    and f(row["em_a"]) < FLOOR and f(row["em_b"]) < FLOOR:
                floor = ("both arms are at floor (arm rates below the floor limit), so the verdict is non-inferiority at floor and says "
                         "nothing about quality; the number of scores is not compared")
                note = (note + "; " if note else "") + floor
            model_leaf = hid in MODEL_HYPS or k.startswith("A2_vs_A1_model")
            sens = bool(lv != lv_raw) if "require_p_holm_below" in leaf else False
            shown = (pfx + PENDING_HOLM) if sens else verdict_cell(lv)
            basis_l = explain(leaf, res, min_n, miss, holm_word + (" for Holm" if not final else ""))
            if sens:
                basis_l += (f"; {lv} at the Bonferroni stand-in (m={m}), {lv_raw} at raw p: a refutation that holds only under the "
                            f"Bonferroni stand-in is marked pending the final Holm")
            leaf_out.append({"key": k, "verdict": shown, "res": res})
            rows.append({
                "key": f"{hid}:{k}", "hypothesis": hid, "kind": "leaf", "rule_key": k,
                "value": res["value"] if res else None,
                "ci_lo": res["ci95"][0] if res and res["ci95"] else None,
                "ci_hi": res["ci95"][1] if res and res["ci95"] else None,
                "n": res["n"] if res and res["n"] else None,
                "min_n": None if leaf.get("measurement") else (min_n or {}).get(k),
                "p_raw": p_raw, "p_bonferroni": None if final else bonferroni(p_raw, m), "family_m": m if k in primary_of.values() else None,
                "holm_status": hp_label if k in primary_of.values() else "",
                "verdict_interim": shown,
                "verdict_basis": basis_l,
                "verdict_ignoring_min_n": lv_nomin, "holm_sensitive": sens,
                "data_split": data_split if (res is not None and not leaf.get("measurement")) else "n/a",
                "level": "measurement" if leaf.get("measurement") else ("model" if model_leaf else "agent"),
                "rule": cl.rule_text(leaf), "notes": note, "p_holm_provisional": prov.get(k),
                "verdict_reason": reason(leaf, res, min_n, miss, holm_word) + ("; marked pending the final Holm (a refutation that holds only under the "
                                                                    "Bonferroni stand-in)" if sens else "")
                + ("; both arms at floor (non-inferiority at floor)" if floor else "")})
        basis = f"frozen rule `{cl.rule_text(h['rule'])}`; leaves: " + " | ".join(
            f"{lo['key']}={lo['verdict']}" for lo in leaf_out)
        hsens = bool(hv != hv_raw)
        if hsens:
            basis += (f"; {hv} at the Bonferroni stand-in (m={m}), {hv_raw} at raw p: marked pending the final Holm")
        if any("require_p_holm_below" in lf_ for lf_ in leaves):
            basis += ("; p_holm leaves use the Holm adjustment over the primaries that report a p" if final else
                      f"; p_holm leaves use Bonferroni(m={m}) as the interim stand-in, final Holm {HOLM_PENDING}")
        hnote = ""
        if hid == "H3":
            leg = next((lo for lo in leaf_out if lo["key"] == "S11_vs_S8_late_bound"), None)
            hnote = ("H3 cannot be supported because the R2 leaf has no result (closure item 9); the verdict is "
                     f"{(pfx + PENDING_HOLM) if hsens else hv} because the S11 vs S8 leg is {leg['verdict'] if leg else 'n/a'} (frozen all_of rule)")
        elif hid == "H5":
            hnote = ("any_of: the model-level leaf is " + (G2_FAILED_STATUS if g2s == "failed" else "pending Track B")
                     + "; an agent-arm refutation alone leaves the hypothesis inconclusive")
        elif hid == "H7":
            hnote = ("measurement on synthetic chains (deterministic, split n/a); the re-anchored ratio composes two offsets in "
                     "the LCA frame of one node, so it shows same-branch precision, not cross-branch growth")
        rows.insert(len(rows) - len(leaves), {
            "key": hid, "hypothesis": hid, "kind": "hypothesis", "rule_key": h.get("primary_key") or "",
            "verdict_interim": (pfx + PENDING_HOLM) if hsens else verdict_cell(hv), "verdict_basis": basis,
            "verdict_ignoring_min_n": hv_nomin, "holm_sensitive": hsens, "family_m": m if hid in fam else None,
            "holm_status": hp_label if hid in fam else "", "data_split": data_split if hid != "H7" else "n/a",
            "level": "measurement" if hid == "H7" else "agent", "rule": cl.rule_text(h["rule"]), "notes": hnote,
            "verdict_reason": "; ".join(f"{lo['key']}: {lo['verdict']}" for lo in leaf_out)})
        for sk in h.get("secondary_reported", []) or []:
            row = contrasts.get(sk)
            if not row:
                continue
            res = _res_from_row({**row, "key": sk}, None, force_test)
            rows.append({"key": f"{hid}:{sk}", "hypothesis": hid, "kind": "secondary", "rule_key": sk, "value": res["value"],
                         "ci_lo": res["ci95"][0] if res["ci95"] else None, "ci_hi": res["ci95"][1] if res["ci95"] else None,
                         "n": res["n"], "min_n": (min_n or {}).get(sk), "p_raw": f(row.get("p_raw")),
                         "verdict_interim": "reported", "verdict_basis": "secondary result: reported, no verdict, not in the Holm family",
                         "data_split": data_split, "level": "agent", "rule": "", "notes": str(row.get("note") or "")})
    for hid, (status, note) in outside.items():
        if hid not in rules["hypotheses"] or hid in scope:
            continue
        rows.append({"key": hid, "hypothesis": hid, "kind": "hypothesis", "rule_key": rules["hypotheses"][hid].get("primary_key") or "",
                     "verdict_interim": status, "verdict_basis": note, "family_m": m if hid in fam else None,
                     "holm_status": hp_label if hid in fam else "", "data_split": "n/a", "level": "",
                     "rule": cl.rule_text(rules["hypotheses"][hid]["rule"]), "notes": "", "verdict_reason": note})
    rows.sort(key=lambda r: (int(r["hypothesis"][1:]), 0 if r["kind"] == "hypothesis" else 1 if r["kind"] == "leaf" else 2))
    famrows = []
    for h in fam:
        k = primary_of[h]
        p = p_by_key[k]
        status = "available" if p is not None else (outside.get(h, ("pending", ""))[0] if h in outside else "pending")
        famrows.append({"key": h, "hypothesis": h, "primary_key": k, "p_raw": p, "p_bonferroni": None if final else bonferroni(p, m),
                        "p_holm_provisional": prov.get(k), "family_m": m, "status": status, "holm_status": hp_label,
                        "family": ";".join(fam)})
    return {"rows": rows, "family": famrows, "fails": fails}


def print_verdicts(rows: Sequence[Mapping], file=sys.stdout) -> None:
    for r in rows:
        if r["kind"] == "secondary":
            continue
        ci = f"[{_fmt(f(r.get('ci_lo')), 3)}, {_fmt(f(r.get('ci_hi')), 3)}]" if r.get("ci_lo") is not None else ""
        pr = r.get("p_raw")
        pb = r.get("p_bonferroni")
        ps = f" p_raw={_g(pr)} p_bonf={_g(pb)}" if pr is not None else ""
        print(f"{r['key']:<62} {r['kind']:<10} {r['verdict_interim']:<18} v={_fmt(r.get('value'), 4)} {ci} n={r.get('n')} "
              f"min_n={r.get('min_n')}{ps} if_min_n_met={r.get('verdict_ignoring_min_n')}", file=file)


# ---------------------------------------------------------------------------------------------- costs
GROUPS = [   # (regex on ledger job name, group, hypotheses/purpose)
    (r"^p2-(embed|sk)-lb|^p2-h1-", "P2 H1 (LB-MHQA): embeddings, skeletons, S11/S2/S8/S4", "H1;H3"),
    (r"^p2-(embed|sk)-mt|^p2-h5-", "P2 H5 (MT-LB-MHQA): embeddings, skeletons, S12/S14/S15", "H5;H8"),
    (r"^p2-h10-", "P2 H10 (thrash)", "H10"),
    (r"^p3-units", "Units U6-U8 and H7", "H7;U6-U8"),
    (r"^p3-(learning|trainfree)", "Track B dev: V3 learning curve, training-free smoke", "H4;H5;H11;H12"),
    (r"^p4-", "Track B: training, G2 evals, H4/H6 test runs", "H4;H5;H6;H11;H12"),
    (r"^(p1-(lb|s4v2|g0|skeleton-lb|skeleton-lateh)|h3cells)", "P1 pilot: LB-MHQA / H1 / H3", "H1;H3"),
    (r"^(p1-(mt|embed)|h5cell)", "P1 pilot: MT-LB-MHQA / H5", "H5;H8"),
    (r"^p1-(thrash|drift)", "P1 pilot: thrash / drift", "H10;H2"),
    (r"^p1-", "P1 pilot: other", "G1"),
    (r"^(p0-|smoke|bench)", "P0 setup: smoke, determinism, G0, benchmarks", "G0"),
]


def group_of(job: str) -> tuple[str, str]:
    for pat, g, h in GROUPS:
        if re.search(pat, job):
            return g, h
    return "other", ""


def costs_tables(approvals: dict | None = None, ledger: list[dict] | None = None) -> dict:
    """Costs to date from the ledger (read-only): per envelope and per group of jobs. ``as_of`` = timestamp of the last
    ledger line, so the tables are deterministic."""
    approvals = approvals if approvals is not None else bud.load_approvals()
    ledger = ledger if ledger is not None else bud.read_ledger()
    st = bud.envelope_states(approvals, ledger)
    cap = float(approvals.get("total_cap_usd") or 0)
    as_of = max((r["ts_utc"] for r in ledger), default="")
    env_rows = []
    for k in sorted(st):
        if k == "RECON":
            continue
        s = st[k]
        env_rows.append({"key": k, "approved_usd": s.approved, "spent_usd": round(s.spent, 6), "reserved_usd": round(s.reserved, 6),
                         "remaining_usd": round(s.remaining, 6), "as_of": as_of})
    spent = sum(r["spent_usd"] for r in env_rows)
    reserved = sum(r["reserved_usd"] for r in env_rows)
    env_rows.append({"key": "TOTAL", "approved_usd": cap, "spent_usd": round(spent, 6), "reserved_usd": round(reserved, 6),
                     "remaining_usd": round(cap - spent - reserved, 6), "as_of": as_of})
    groups: dict[str, dict] = {}
    for r in ledger:
        if r["kind"] != "actual":
            continue
        g, h = group_of(r["job"])
        e = groups.setdefault(g, {"key": g, "hypotheses": h, "spent_usd": 0.0, "n_jobs": 0, "envelopes": set()})
        e["spent_usd"] += float(r["amount_usd"] or 0)
        e["n_jobs"] += 1
        e["envelopes"].add(r["envelope"])
    grp_rows = []
    for g in sorted(groups):
        e = groups[g]
        grp_rows.append({"key": g, "hypotheses": e["hypotheses"], "spent_usd": round(e["spent_usd"], 6), "n_jobs": e["n_jobs"],
                         "envelopes": ";".join(sorted(e["envelopes"])), "as_of": as_of})
    return {"envelope": env_rows, "group": grp_rows, "as_of": as_of}


# ---------------------------------------------------------------------------------------------- gates
def _short_obs(o: Any) -> tuple[float | None, str]:
    if isinstance(o, bool):
        return None, str(o).lower()
    if isinstance(o, (int, float)):
        return float(o), ""
    if isinstance(o, dict):
        if isinstance(o.get("value"), (int, float)):
            return float(o["value"]), json.dumps(o, sort_keys=True)
        return None, json.dumps(o, sort_keys=True)
    return None, str(o)


def gates_tables(bundle: Path | str) -> dict:
    """gates / history / criteria tables from bundle/gates (G0, G1, ...) and bundle/gates/history/*.json."""
    gdir = Path(bundle) / "gates"
    rows, crit, hist = [], [], []
    seen = set()
    for gid in cl.G_IDS if hasattr(cl, "G_IDS") else ["G0", "G1", "G2", "G3", "G4"]:
        p = gdir / f"{gid}.json"
        if p.exists():
            g = json.loads(p.read_text(encoding="utf-8"))
            seen.add(gid)
            cr = g.get("criteria") or []
            rows.append({"key": gid, "status": "not reached" if g.get("passed") is None else ("passed" if g.get("passed") else "failed"),
                         "decided_at": g.get("decided_at"),
                         "n_criteria": len(cr), "n_met": sum(bool(c.get("met")) for c in cr), "note": (g.get("notes") or "")})
            for i, c in enumerate(cr):
                v, txt = _short_obs(c.get("observed"))
                crit.append({"key": f"{gid}:{i}", "gate": gid, "name": c.get("name", ""), "threshold": str(c.get("threshold", "")),
                             "observed_value": v, "observed_text": txt, "met": bool(c.get("met"))})
        else:
            note = {"G2": "pending: no G2 record (bundle/gates/G2.json) yet; judged after V3 and B1 reach the stopping rule",
                    "G3": "pending: decisive ablation, only if G2 passes", "G4": "pending: paper-ready bundle"}.get(gid, "")
            rows.append({"key": gid, "status": "pending", "decided_at": None, "n_criteria": 0, "n_met": 0, "note": note})
    hdir = gdir / "history"
    if hdir.exists():
        for p in sorted(hdir.glob("*.json")):
            g = json.loads(p.read_text(encoding="utf-8"))
            cr = g.get("criteria") or []
            failed = [c.get("name", "") for c in cr if not c.get("met")]
            hist.append({"key": p.stem, "gate": g.get("id"), "status": "passed" if g.get("passed") else "failed",
                         "decided_at": g.get("decided_at"), "n_criteria": len(cr), "n_met": len(cr) - len(failed),
                         "failed_criteria": " | ".join(failed), "superseded_by": "PLAN_V3 revision 3 (owner override): bundle/gates/G1.json",
                         "note": (g.get("notes") or "")})
            for i, c in enumerate(cr):
                v, txt = _short_obs(c.get("observed"))
                crit.append({"key": f"{p.stem}:{i}", "gate": f"{g.get('id')} (original)", "name": c.get("name", ""),
                             "threshold": str(c.get("threshold", "")), "observed_value": v, "observed_text": txt,
                             "met": bool(c.get("met"))})
    return {"gates": rows, "history": hist, "criteria": crit}


# ---------------------------------------------------------------------------------------------- deviations
def _dev_meta(p: Path) -> dict:
    text = p.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else ""
    m = re.match(r"#\s*(DEV-[\w-]+)\s*[-—–:]+\s*(.*)", first)
    did = m.group(1) if m else p.stem
    title = m.group(2).strip() if m else first.lstrip("# ").strip()
    d = re.search(r"\*\*Date:\*\*\s*([0-9-]+)", text)
    return {"key": did, "kind": "deviation", "date": d.group(1) if d else "", "title": title,
            "source": f"prereg/deviations/{p.name}"}


BUILTIN_DEVIATIONS = [      # named in the registration or the gate records, without a file in prereg/deviations/
    {"key": "DEV-U1", "kind": "deviation", "date": "2026-11-07",
     "title": "U1 two-part definition (float64 error <= 1e-10 and fp32 error <= 4 ulps of the largest key); the literal 1e-5 "
              "absolute criterion is below one fp32 ulp at real key magnitudes and is reported, not met",
     "source": "prereg draft Section 3 and 9; bundle/gates/G0.json; state/decisions.md 2026-11-07 (pre-tag)"},
    {"key": "DEV-G0-noise", "kind": "deviation", "date": "2026-11-07",
     "title": "G0 S1 repeat criterion passed under the owner's ruling recorded before the results: every difference was a near-tie "
              "flip, none a harness error; the measured flip rate and EM spread are in the G0 record and next to every primary contrast",
     "source": "prereg draft Section 3; bundle/gates/G0.json; state/decisions.md 2026-11-07 (pre-tag)"},
    {"key": "G1-OVERRIDE", "kind": "disclosure", "date": "2026-11-07",
     "title": "G1 first failed under the protocol's definition (H3 R2 alone), then the owner replaced the definition "
              "(PLAN_V3 revision 3); G1 re-decided and passed; the original record is kept in bundle/gates/history/",
     "source": "PLAN_V3 revision 3 Section 6b; prereg draft Section 4; bundle/gates/history/"},
    {"key": "H3-R2-NOT-TESTABLE", "kind": "disclosure", "date": "2026-11-07",
     "title": "H3's R2 leaf is not testable at 32K (the predicted coverage is at its maximum for almost every pilot item); no result is "
              "reported for it, so H3 cannot be supported (refuted if the S11 vs S8 leg is refuted, else inconclusive)",
     "source": "prereg draft Section 2b and closure item 9"},
    {"key": "S12-BEYOND-PROTOCOL", "kind": "disclosure", "date": "2026-11-07",
     "title": "33 items where the registration goes beyond the protocol (Holm family, TOST via Tango score, intention-to-treat "
              "scoring, k and lambda fixed, ...); each is listed in Section 12 of the registration",
     "source": "prereg draft Section 12"},
]


def deviation_rows(repo: Path | str, bundle: Path | str) -> list[dict]:
    repo, bundle = Path(repo), Path(bundle)
    rows = []
    ddir = repo / "prereg" / "deviations"
    if ddir.exists():
        for p in sorted(ddir.glob("DEV-*.md")):
            rows.append(_dev_meta(p))
    have = {r["key"] for r in rows}
    for r in BUILTIN_DEVIATIONS:
        if r["key"] not in have:
            rows.append(dict(r))
            have.add(r["key"])
    bd = bundle / "deviations.md"
    if bd.exists():
        text = bd.read_text(encoding="utf-8")
        for m in re.finditer(r"^###\s*(DEV-[\w-]+)\s*$(.*?)(?=^### |\Z)", text, re.S | re.M):
            if m.group(1) in have:
                continue
            body = m.group(2)
            d = re.search(r"Date:\s*([0-9-]+)", body)
            c = re.search(r"Change:\s*(.+)", body)
            rows.append({"key": m.group(1), "kind": "deviation", "date": d.group(1) if d else "",
                         "title": (c.group(1) if c else ""), "source": "bundle/deviations.md"})
            have.add(m.group(1))
    return rows


# ---------------------------------------------------------------------------------------------- Track B / C
KNOWN_EXCLUSIONS = {      # run id -> handling text (decisions.md); anything else is described from its own result.json
    "B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108022824-e36fdc0-201d":
        "failed (pre-fix code: the resume path was unresolved); excluded; V3 chunk 1 is the later run on the fixed deploy",
}
FAILED_COLUMNS = ["key", "job", "status", "reason", "handling", "replaced_by", "attempt", "max_attempts"]
SENS_COLUMNS = ["key", "first_run", "retry_run", "system", "n_shared", "em_first", "em_retry", "diff_points"]


def short_reason(error: str | None) -> str:
    """One line from a failure traceback, without paths or identifiers."""
    if not error:
        return "no error text in the run record"
    m = re.search(r"status code', (\d{3})", error) or re.search(r"\b(4\d\d|5\d\d)\b", error)
    if m and ("upload" in error or "storage" in error):
        return f"upload of a run file to storage failed with HTTP {m.group(1)}"
    last = [ln for ln in error.strip().splitlines() if ln.strip()][-1].strip()
    return re.sub(r"https?://\S+", "<url>", last)[:160]


def _records_by_item(store, rid: str, system: str) -> dict:
    try:
        return {r["item_id"]: r for r in pl.read_jsonl(store.get(pl.run_uri(store, rid) + f"/{system}/r0/records.jsonl.gz"))}
    except KeyError:
        return {}


def rerun_sensitivity(store, runs_dir: Path, first: str, retry: str, split: str = "test") -> list[dict]:
    """First (failed) attempt against its retry, per system, on the items both have records for: the ITT EM of each, and the
    difference first minus retry in points."""
    jr = p2._load_job(runs_dir, retry, split)
    items = pl.load_run_items(store, jr)
    rows = []
    for system in jr["args"]["systems"]:
        rf, rr_ = _records_by_item(store, first, system), _records_by_item(store, retry, system)
        shared = sorted(set(rf) & set(rr_) & set(items))
        if not shared:
            rows.append({"key": f"{first}|{system}", "first_run": first, "retry_run": retry, "system": system, "n_shared": 0})
            continue
        ef = float(np.mean([p2._em(items[i], rf[i])[0] for i in shared]))
        er = float(np.mean([p2._em(items[i], rr_[i])[0] for i in shared]))
        rows.append({"key": f"{first}|{system}", "first_run": first, "retry_run": retry, "system": system, "n_shared": len(shared),
                     "em_first": ef, "em_retry": er, "diff_points": (ef - er) * 100.0})
    return rows


def failed_run_rows(store, runs_dir: Path, entries: Sequence[Mapping], split: str = "test") -> tuple[list[dict], list[dict]]:
    """(failed/excluded run rows, re-run sensitivity rows) for ``entries`` = [{"run_id", "replaced_by"?}]."""
    runs_dir = Path(runs_dir)
    rows, sens = [], []
    for e in entries or []:
        rid = e["run_id"]
        res = {}
        if (runs_dir / rid / "result.json").exists():
            res = json.loads((runs_dir / rid / "result.json").read_text(encoding="utf-8"))
        job = (yaml.safe_load((runs_dir / rid / "job.yaml").read_text(encoding="utf-8")) or {}) if (runs_dir / rid / "job.yaml").exists() else {}
        name = str(job.get("name", ""))
        same = sorted(d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "job.yaml").exists()
                      and (yaml.safe_load((d / "job.yaml").read_text(encoding="utf-8")) or {}).get("name") == name) if name else [rid]
        same.sort(key=lambda r: (_STAMP.search(r).group(1) if _STAMP.search(r) else "", r))
        attempt = same.index(rid) + 1 if rid in same else 1
        rep_by = e.get("replaced_by") or ""
        handling = KNOWN_EXCLUSIONS.get(rid) or (
            f"replaced by retry {same.index(rep_by) + 1 if rep_by in same else attempt + 1} of 3 allowed (run {rep_by}); the retry's records are used"
            if rep_by else "not status ok; left out of the analysis")
        rows.append({"key": rid, "job": name, "status": res.get("status", "no result.json"), "reason": short_reason(res.get("error")),
                     "handling": handling, "replaced_by": rep_by, "attempt": attempt, "max_attempts": 3})
        if rep_by and store is not None and P2_JOB.match(name) and name.split("-c")[0] in ("p2-h1-lb", "p2-h5-mt", "p2-h1-s4"):
            try:
                sens += rerun_sensitivity(store, runs_dir, rid, rep_by, split)
            except (KeyError, pl.MissingInput, ValueError) as ex:
                sens.append({"key": f"{rid}|n/a", "first_run": rid, "retry_run": rep_by, "system": f"not computable: {type(ex).__name__}", "n_shared": 0})
    return rows, sens


LC_COLUMNS = ["key", "arm", "opt_step", "train_loss", "own_origin_mass_fraction", "own_origin_mass_ratio_to_uniform",
              "own_thread_mass_fraction", "ghost_ratio", "router_entropy", "attrib_em", "attrib_rank_acc", "attrib_chance",
              "passkey_em", "train_seconds", "eval_seconds", "split", "run_id"]


def learning_curve_rows(curves: Sequence[tuple[str, Mapping]]) -> list[dict]:
    """Rows from learning_curve.json documents ({"curve": [...]}). ``curves``: [(run id, document)] in run order; a
    (arm, step) seen again in a later document replaces the earlier row."""
    by: dict[tuple, dict] = {}
    for rid, doc in curves:
        entries = doc.get("curve") if isinstance(doc, Mapping) else doc
        for e in entries or []:
            probe = e.get("probe") or {}
            arm, step = e.get("arm", "V3"), int(e.get("opt_step", 0))
            by[(arm, step)] = {
                "key": f"{arm}_step{step}", "arm": arm, "opt_step": step, "train_loss": f(e.get("train_loss")),
                "own_origin_mass_fraction": f(probe.get("own_origin_mass_fraction")),
                "own_origin_mass_ratio_to_uniform": f(probe.get("own_origin_mass_ratio_to_uniform")),
                "own_thread_mass_fraction": f(probe.get("own_thread_mass_fraction")),
                "ghost_ratio": f(probe.get("ghost_ratio")), "router_entropy": f(probe.get("router_entropy")),
                "attrib_em": f(e.get("attrib_em")), "attrib_rank_acc": f(e.get("attrib_rank_acc")),
                "attrib_chance": f(e.get("attrib_chance")), "passkey_em": f(e.get("passkey_em")),
                "train_seconds": f(e.get("train_seconds")), "eval_seconds": f(e.get("eval_seconds")),
                "split": "dev", "run_id": rid}
    return [by[k] for k in sorted(by, key=lambda k: (k[0], k[1]))]


def stopping_rule(lc: Sequence[Mapping], rise_threshold: float = 0.02, min_step: int = 400, cap: int = 2000) -> dict:
    """The pre-registered V3 stopping rule on the evaluated steps: the first step L >= min_step whose own-origin mass fraction rose by
    less than the threshold over the evaluation exactly 200 steps earlier. {"fired": bool, "capped": bool (last step at or beyond the cap without a qualifying small rise), "step", "rise",
    "previous_step"}."""
    v3 = {int(r["opt_step"]): r["own_origin_mass_fraction"] for r in lc if r["arm"] == "V3" and r["own_origin_mass_fraction"] is not None}
    for step in sorted(v3):
        if step >= min_step and step - 200 in v3:
            rise = v3[step] - v3[step - 200]
            if rise < rise_threshold:
                return {"fired": True, "capped": False, "step": step, "rise": rise, "previous_step": step - 200}
    last = max(v3) if v3 else None
    prev = last - 200 if last is not None and last - 200 in v3 else None
    return {"fired": False, "capped": bool(last is not None and last >= cap), "step": last,
            "rise": (v3[last] - v3[prev]) if prev is not None else None, "previous_step": prev}


def trackb_status_rows(lc: Sequence[Mapping], g2_threshold: float = 0.90, rise_threshold: float = 0.02) -> list[dict]:
    """Where the V3 learning curve stands against the G2 own-origin threshold and the stopping rule (dev monitoring set)."""
    v3 = sorted((r for r in lc if r["arm"] == "V3" and r["own_origin_mass_fraction"] is not None), key=lambda r: r["opt_step"])
    rows = []
    if not v3:
        return [{"key": "V3_curve", "value": None, "threshold": None, "status": "no curve", "note": "no learning_curve.json given"}]
    sr = stopping_rule(lc, rise_threshold)
    last = next((r for r in v3 if r["opt_step"] == sr["step"]), v3[-1]) if sr["fired"] else v3[-1]
    rows.append({"key": "V3_latest_own_origin_mass_fraction", "value": last["own_origin_mass_fraction"], "threshold": g2_threshold,
                 "status": "below G2 threshold" if last["own_origin_mass_fraction"] < g2_threshold else "at or above",
                 "step": last["opt_step"], "note": "dev monitoring set; G2 is judged on fresh dev seeds after the stopping rule fires"})
    if sr["rise"] is not None:
        rows.append({"key": "V3_rise_last_two_evaluations", "value": sr["rise"], "threshold": rise_threshold,
                     "status": ("stopping rule fires (rise below the threshold)" if sr["fired"] else
                                "stopping rule did not fire; stopped at the 2,000-step cap" if sr["capped"] else
                                "stopping rule continues (rise at or above the threshold)"),
                     "step": sr["step"], "note": f"between steps {sr['previous_step']} and {sr['step']}"})
    if sr["fired"]:
        rows.append({"key": "V3_stopping_rule_fired_step", "value": float(sr["step"]), "threshold": float(400), "status": "fired",
                     "step": sr["step"], "note": "first evaluation at or after the minimum step whose rise over 200 steps is below the threshold"})
    if sr["capped"]:
        rows.append({"key": "V3_stopped_at_cap", "value": float(sr["step"]), "threshold": 2000.0, "status": "stopped at the 2,000-step cap",
                     "step": sr["step"], "note": "no evaluation 200 steps apart rose by less than the threshold before the cap"})
    rows.append({"key": "V3_steps_trained", "value": float(v3[-1]["opt_step"]), "threshold": 2000.0, "status": "cap on optimizer steps",
                 "step": v3[-1]["opt_step"], "note": "stopping rule: minimum 400 steps, cap 2000"})
    return rows


PINNED_COLUMNS = ["key", "arm", "final_step", "run_id", "sha256", "note"]


def pinned_checkpoint_rows(repo: Path | str) -> list[dict]:
    """Rows from the pinned-checkpoints table of prereg/deviations/DEV-TRAIN-PIN.md (arm, final step, run id, SHA-256)."""
    p = Path(repo) / "prereg" / "deviations" / "DEV-TRAIN-PIN.md"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0] not in ("V3", "B1", "A1", "A2", "A3"):
            continue
        step = re.search(r"\d+", cells[1])
        run = re.search(r"\bB-[A-Za-z0-9+._-]+-[0-9a-f]{4}\b", cells[2])
        sha = re.findall(r"\b[0-9a-f]{64}\b", cells[3])
        if not sha:
            continue
        rows.append({"key": cells[0], "arm": cells[0], "final_step": int(step.group(0)) if step else None, "run_id": run.group(0) if run else "",
                     "sha256": sha[-1], "note": "from prereg/deviations/DEV-TRAIN-PIN.md"})
    return rows


def load_learning_curves(store, runs_dir: Path, ids: Sequence[str]) -> tuple[list[tuple[str, dict]], list[str]]:
    out, missing = [], []
    for rid in ids:
        doc = _read_json_local_or_store(runs_dir, store, rid, "learning_curve.json")
        if doc is None:
            missing.append(rid)
        else:
            out.append((rid, doc))
    return out, missing


def trackc_rows(env_rows: Sequence[Mapping], g2: Mapping | None = None) -> list[dict]:
    e5 = next((r for r in env_rows if r["key"] == "E5"), {})
    st = g2_state(g2)
    out = outside_status(g2)
    if st == "failed":
        c_status, c_note = "not run (gate G2 failed)", "Track C starts only if G2 passes; G2 failed, so no H9 run; owner request budget/requests/H9_exploratory.md"
    elif st == "passed":
        c_status, c_note = "may start (G2 passed)", "Track C (copy vs re-encode) is funded by E5"
    else:
        c_status, c_note = "not started", "Track C (copy vs re-encode) starts only if G2 passes; owner instruction: if G2 fails, no H9 run"
    return [
        {"key": "TrackC", "status": c_status, "gate_dependency": "G2", "envelope": "E5",
         "approved_usd": e5.get("approved_usd"), "spent_usd": e5.get("spent_usd"), "note": c_note},
        {"key": "H9", "status": out["H9"][0], "gate_dependency": "G2", "envelope": "E5", "approved_usd": e5.get("approved_usd"),
         "spent_usd": e5.get("spent_usd"), "note": "page-in by copy vs re-encoding; a measurement; " + out["H9"][1]},
        {"key": "H13", "status": out["H13"][0], "gate_dependency": "G2", "envelope": "", "approved_usd": None, "spent_usd": None,
         "note": out["H13"][1]},
    ]


# ---------------------------------------------------------------------------------------------- figure sources
def load_g2(bundle: Path | str) -> dict | None:
    """bundle/gates/G2.json (gate.schema) if it exists."""
    p = Path(bundle) / "gates" / "G2.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def trackb_table_path(bundle: Path | str, split: str = "test") -> Path:
    return Path(bundle) / "tables" / "trackb" / f"trackb_contrasts_{split}.csv"


def _read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["key"]: r for r in read_table(path)}


DEV_SRC_TEST_MODE = "tables/p2_dev_check/p2_contrasts_dev.csv"      # the file the release ships as the dev pilot context


def _src(split: str, dev_src: str) -> str:
    return "tables/p2_contrasts_test.csv" if split == "test" else dev_src


def h1_figure_rows(test_rows: Mapping[str, Mapping], dev_rows: Mapping[str, Mapping], dev_src: str = DEV_SRC_TEST_MODE) -> list[dict]:
    """Source rows of the H1 figure: the primary-cell contrasts, test and dev pilot, one row each. The funded test split
    has ONE H1 cell (LB-MHQA late h=2, 32K); depth and context variation exist nowhere else in Track A data."""
    out = []
    for split, rows in (("test", test_rows), ("dev", dev_rows)):
        for k in ("S11_vs_S2_lbmhqa_late_32k_primary", "S11_vs_S4_lbmhqa_late_32k_primary", "S11_vs_S8_late_bound"):
            r = rows.get(k)
            if not r:
                continue
            out.append({"key": f"{split}|late-h2-32k|{k}", "split": split, "cell": "LB-MHQA late h=2 32K", "depth_h": 2,
                        "context_tokens": 32768, "contrast": k, "value": f(r["value"]), "ci_lo": f(r["ci_lo"]), "ci_hi": f(r["ci_hi"]),
                        "n": int(f(r["n"])), "em_a": f(r.get("em_a")), "em_b": f(r.get("em_b")), "systems": r.get("systems", ""),
                        "source": f"{_src(split, dev_src)}#row={k}"})
    return out


def h5_figure_rows(test_rows: Mapping[str, Mapping], dev_rows: Mapping[str, Mapping], distortion: Mapping[str, Mapping],
                   dev_src: str = DEV_SRC_TEST_MODE) -> list[dict]:
    """Source rows of the H5 figure: agent-level A2 minus A1 at the one funded thread depth (T4) for test and dev pilot,
    plus the model-free dev distortion ratios at T = 4 / 8 / 16."""
    out = []
    k = "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross"
    for split, rows in (("test", test_rows), ("dev", dev_rows)):
        r = rows.get(k)
        if r:
            out.append({"key": f"{split}|T4|agent_em_diff", "panel": "agent", "split": split, "threads_T": 4, "metric": "A2_minus_A1_EM_points",
                        "value": f(r["value"]), "ci_lo": f(r["ci_lo"]), "ci_hi": f(r["ci_hi"]), "n": int(f(r["n"])),
                        "source": f"{_src(split, dev_src)}#row={k}"})
    for T in (4, 8, 16):
        r = distortion.get(f"T{T}-h3-32k-cross_v2_gold_clampnone")
        if r:
            out.append({"key": f"dev|T{T}|distortion_ratio", "panel": "distortion", "split": "dev", "threads_T": T,
                        "metric": "euclidean_over_hyperbolic_tree_distortion", "value": f(r["tree_distortion_euclidean_over_hyperbolic"]),
                        "ci_lo": None, "ci_hi": None, "n": int(f(r["n_items"]) or 0),
                        "source": f"tables/h5_distortion_dev.csv#row=T{T}-h3-32k-cross_v2_gold_clampnone"})
    return out


# ---------------------------------------------------------------------------------------------- driver
def design_rows(rules: dict, lam: float = 2.0) -> list[dict]:
    """Constants quoted in PAPER_INPUT (rule thresholds and registered design numbers), one row each, so every one is referenced."""
    def thr(hid, key):
        return next(l["threshold"] for l in cl.rule_leaves(rules["hypotheses"][hid]["rule"]) if l["key"] == key)
    return [
        {"key": "h1_design_power_at_7_points", "value": 0.80, "source": "registered sample sizes: H1 sized for 7 points (owner)"},
        {"key": "h1_design_power_at_5_points", "value": 0.52, "source": "registered sample sizes comment and Section 2b (power at the rule threshold)"},
        {"key": "h7_reanchored_growth_limit", "value": thr("H7", "reanchored_error_growth_ratio_depth40_vs_depth1_bf16"), "source": "rules.yaml H7"},
        {"key": "h3_r2_threshold", "value": thr("H3", "R2_predicted_vs_observed_success"), "source": "rules.yaml H3"},
        {"key": "g2_own_origin_threshold", "value": thr("H4", "V3_own_origin_mass_fraction"), "source": "rules.yaml H4 / G2"},
        {"key": "stopping_rule_rise_threshold", "value": 0.02, "source": "registration Section 4 (G2) stopping rule"},
        {"key": "track_a_lambda", "value": lam, "source": "registration closure item 16 (not tuned)"},
        {"key": "tost_margin_points", "value": 3.0, "source": "registration Section 6 (A2 vs A3)"},
        {"key": "h8_ni_margin_points", "value": thr("H8", "descent_beam4_vs_flat_recall_at_M"), "source": "rules.yaml H8"},
    ]


def started_after_tag(series: Mapping[str, Sequence[str]], tag_file: Path) -> tuple[int, list[str]]:
    """(runs checked, violations): the UTC timestamp in each run id (the launcher's yyyymmddHHMMSS) against the tag record's
    committed_at. A run id without a timestamp is a violation."""
    import datetime as dt
    tag = json.loads(Path(tag_file).read_text(encoding="utf-8"))
    t0 = cl.verifier().parse_time(tag["committed_at"])
    bad, n = [], 0
    for ids in series.values():
        for rid in ids:
            n += 1
            m = _STAMP.search(rid)
            if not m:
                bad.append(f"{rid}: no timestamp in the run id")
                continue
            t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
            if t <= t0:
                bad.append(f"{rid}: started {t.isoformat()} not after the tag commit {t0.isoformat()}")
    return n, bad


def _name(split: str, base: str) -> str:
    """tables/<base>[_dev].csv; ``check_name`` enforces the split rule."""
    return check_name(split, f"{base}{suffix(split)}.csv")


def write_all_tables(out: Path, split: str, *, contrasts: Mapping, units: Mapping, rules: dict, min_n: Mapping,
                     curves: Sequence[tuple[str, Mapping]], dev_rows: Mapping, repo: Path, approvals: dict | None = None,
                     ledger: list[dict] | None = None, failed: tuple | None = None) -> dict:
    """Write every interim table; returns {"verdicts": ..., "files": [...]}. ``failed`` = (failed-run rows, sensitivity rows)."""
    tdir = out / "tables"
    files: list[str] = []

    def w(name, rows, cols, dev_data=False):
        check_name(split, name, dev_data)
        files.append(str(write_table(tdir / name, rows, columns=cols)))

    g2 = load_g2(out)
    v = build_verdicts(contrasts, units, rules, min_n, split, g2=g2)
    w(_name(split, "interim_verdicts"), v["rows"], VERDICT_COLUMNS)
    w(_name(split, "interim_family"), v["family"], ["key", "hypothesis", "primary_key", "p_raw", "p_bonferroni", "p_holm_provisional", "family_m", "status", "holm_status", "family"])
    w(_name(split, "interim_units"), units_rows(units, repo / "prereg" / "deviations"),
      ["key", "unit", "status", "passed", "code_flag", "observed", "threshold", "aux_observed", "aux_name", "ulps", "criterion", "deviation", "run_id", "notes"])
    h7 = [{"key": k, "value": (units.get("h7") or {}).get(k), "run_id": (units.get("runs") or {}).get("u6_u8_h7", "")} for k in H7_KEYS
          if (units.get("h7") or {}).get(k) is not None]
    if h7:
        w(_name(split, "interim_h7"), h7, ["key", "value", "run_id"])
    c = costs_tables(approvals, ledger)
    w(_name(split, "interim_costs_envelope"), c["envelope"], ["key", "approved_usd", "spent_usd", "reserved_usd", "remaining_usd", "as_of"])
    w(_name(split, "interim_costs_group"), c["group"], ["key", "hypotheses", "spent_usd", "n_jobs", "envelopes", "as_of"])
    g = gates_tables(out)
    w(_name(split, "interim_gates"), g["gates"], ["key", "status", "decided_at", "n_criteria", "n_met", "note"])
    w(_name(split, "interim_gate_criteria"), g["criteria"], ["key", "gate", "name", "threshold", "observed_value", "observed_text", "met"])
    if g["history"]:
        w(_name(split, "interim_gate_history"), g["history"], ["key", "gate", "status", "decided_at", "n_criteria", "n_met",
                                                               "failed_criteria", "superseded_by", "note"])
    w(_name(split, "interim_deviations"), deviation_rows(repo, out), ["key", "kind", "date", "title", "source"])
    w(_name(split, "interim_design"), design_rows(rules), ["key", "value", "source"])
    w(_name(split, "interim_trackc"), trackc_rows(c["envelope"], g2), ["key", "status", "gate_dependency", "envelope", "approved_usd", "spent_usd", "note"])
    lc = learning_curve_rows(curves)
    if lc:
        w("v3_learning_curve_dev.csv", lc, LC_COLUMNS, dev_data=True)
        w("trackb_status_dev.csv", trackb_status_rows(lc), ["key", "value", "threshold", "status", "step", "note"], dev_data=True)
    if failed and failed[0]:
        w(_name(split, "interim_failed_runs"), failed[0], FAILED_COLUMNS)
        if failed[1]:
            w(_name(split, "interim_rerun_sensitivity"), failed[1], SENS_COLUMNS)
    pins = pinned_checkpoint_rows(repo)
    if pins:
        w("trackb_pinned_checkpoints_dev.csv", pins, PINNED_COLUMNS, dev_data=True)
    return {"verdicts": v, "files": files, "costs": c, "gates": g, "lc": lc}


def run(args: argparse.Namespace, store=None) -> int:
    split = args.split
    check_split(split)
    out = Path(args.out).resolve()
    root = Path(getattr(args, "project_root", None) or PROJECT_DIR).resolve()
    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "runs"
    series, icfg = load_runs_yaml(args.runs)
    rules, min_n = cl.load_prereg(root / "prereg")
    sizes = root / "prereg" / "sample_sizes.yaml"
    tag_file = root / "prereg" / "PREREG_TAG.json"
    store = store if store is not None else make_store(args.creds, root)
    tdir = out / "tables"
    p2name = f"p2_contrasts_{split}.csv"
    if args.reuse_p2:
        p2.require_prereg_tag(split, tag_file)                    # test results only after the tag record exists
        src_dir = tdir
        if not (src_dir / p2name).exists():
            raise InterimError(f"--reuse-p2: {src_dir / p2name} does not exist")
        wrong = sorted({r.get("split", "") for r in read_table(src_dir / p2name)} - {split})
        if wrong:
            raise InterimError(f"--reuse-p2: {src_dir / p2name} has rows with split {wrong}, expected only {split!r}")
    else:
        if store is None:
            raise InterimError("no GCS credentials found (config/project.yaml, credentials.md or --creds): cannot read the runs")
        src_dir = _tmpdir() / "tables" if args.dry_run else tdir      # a dry run computes but keeps nothing
        res = p2.run_p2(store, series, split, src_dir, runs_dir=runs_dir, min_n=sizes, rules=rules, tag_file=tag_file)
        for s_ in res["skipped"]:
            print("skipped:", s_)
    if split == "test":
        n_chk, bad = started_after_tag(series, tag_file)
        if bad:
            raise InterimError("test-split runs that did not start after the prereg-v2 commit:\n  " + "\n  ".join(bad))
        print(f"{n_chk} series runs checked: all started after the prereg-v2 commit (run-id timestamps)")
    contrasts = _read_rows(src_dir / p2name)
    if split == "test":          # Track B test tables (opa.analysis.trackb_test), when present, join the verdict table
        tb = _read_rows(trackb_table_path(out, split))
        bad_tb = sorted({r.get("split", "") for r in tb.values()} - {split})
        if bad_tb:
            raise InterimError(f"{trackb_table_path(out, split)} has rows with split {bad_tb}, expected only {split!r}")
        clash = sorted(set(tb) & set(contrasts))
        if clash:
            raise InterimError(f"Track B table repeats Track A keys {clash}")
        contrasts = {**contrasts, **tb}
    units = load_units(runs_dir, store, icfg)
    verdicts = build_verdicts(contrasts, units, rules, min_n, split, g2=load_g2(out))
    print_verdicts(verdicts["rows"])
    if verdicts["fails"]:
        print("verifier complaints:", *sorted(set(verdicts["fails"])), sep="\n  ")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    dev_rows = {}
    if split == "test" and not args.no_dev_context:
        dev_dir = tdir / "p2_dev_check"
        if store is not None and icfg.get("dev_series"):     # else the committed dev pilot tables are used as they are
            p2.run_p2(store, p2.load_mapping(icfg["dev_series"]), "dev", dev_dir, runs_dir=runs_dir, min_n=sizes, rules=rules)
        dev_rows = _read_rows(dev_dir / "p2_contrasts_dev.csv")
    elif split == "dev":
        dev_rows = contrasts
    curves, missing = load_learning_curves(store, runs_dir, icfg.get("learning_curve") or [])
    for m_ in missing:
        print(f"learning curve: no learning_curve.json for {m_}", file=sys.stderr)
    approvals = bud.load_approvals(root / "budget" / "approvals.yaml")
    ledger = bud.read_ledger(root / "budget" / "ledger.csv")
    failed = failed_run_rows(store, runs_dir, icfg.get("failed_runs") or [], split) if split == "test" else None
    r = write_all_tables(out, split, contrasts=contrasts, units=units, rules=rules, min_n=min_n, curves=curves,
                         dev_rows=dev_rows, repo=root, approvals=approvals, ledger=ledger, failed=failed)
    from . import interim_figures as ifig
    figs = ifig.make_figures(out, split, contrasts if split == "test" else {}, dev_rows, repo=root)
    if getattr(args, "skip_paper", False):                    # the final build writes its own PAPER_INPUT.md
        print("wrote:", *r["files"], *figs, sep="\n  ")
        return 0
    from . import paper_input_interim as pii
    md = pii.write_paper_input_interim(out, split, rules=rules, min_n=min_n, args_runs=series, icfg=icfg, root=root)
    probs = pii.check_refs(md, out)
    for p_ in probs:
        print("REF PROBLEM:", p_, file=sys.stderr)
    print("wrote:", *r["files"], *figs, str(md), sep="\n  ")
    return 1 if probs else 0


def _tmpdir() -> Path:
    import tempfile
    return Path(tempfile.mkdtemp(prefix="interim_dry_"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--runs", help="series YAML (see the module docstring)")
    ap.add_argument("--out", help="bundle directory (tables/, figures/, PAPER_INPUT_interim*.md go under it)")
    ap.add_argument("--discover", metavar="SERIES_YAML", help="write a series YAML from the collected test-split runs in runs/ and exit "
                    "(latest ok run per P2 job name; runs that are not ok are listed and left out)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print the verdicts; write nothing")
    ap.add_argument("--reuse-p2", action="store_true", help="read tables/p2_contrasts_<split>.csv instead of recomputing")
    ap.add_argument("--no-dev-context", action="store_true", help="test mode: do not (re)compute the dev pilot context")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--creds", default=None)
    ap.add_argument("--project-root", default=None, help="checkout to read prereg, runs, budget and credentials from "
                    "(default: the checkout this command runs in)")
    a = ap.parse_args(argv)
    if a.discover:
        d = discover_series(a.runs_dir or Path(a.project_root or PROJECT_DIR) / "runs", a.split or "test")
        Path(a.discover).parent.mkdir(parents=True, exist_ok=True)
        Path(a.discover).write_text("# generated by opa.analysis.interim --discover: latest ok run per job name\n"
                                    + yaml.safe_dump({"p2": d["p2"], "interim": d["interim"]}, sort_keys=False), encoding="utf-8")
        print(f"wrote {a.discover}: " + "; ".join(f"{k} x{len(v)}" for k, v in d["p2"].items()))
        for n in d["not_ok"]:
            print("not ok, left out:", n)
        for n, ids in d["repeats"].items():
            print(f"REPEAT: job {n} has {len(ids)} ok runs (latest used): " + ", ".join(ids))
        return 0
    if not (a.split and a.runs and a.out):
        ap.error("--split, --runs and --out are required (or use --discover)")
    try:
        return run(a)
    except (InterimError, ValueError, PermissionError, pl.MissingInput) as e:
        print(f"interim FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
