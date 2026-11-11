"""Track B TEST-split results: H4 (V3 vs B1 source attribution, own-origin mass measurements) and H6 (remedies R0-R7) -> contrast tables.

  python -m opa.analysis.trackb_test --runs runs_trackb_test.yaml --out bundle/tables/trackb [--split test] [--min-n prereg/sample_sizes.yaml]

``--runs`` is a YAML/JSON mapping  series -> [run id, ...]  (one id per container; any subset of the series is analysed):
  h4-v3  [p4-h4-test-v3-c01, -c02 run ids]    h4-b1  [p4-h4-test-b1-c01, -c02]     (jobs/make_p4_test_jobs.py, opa.trackb.train, eval_split test)
  h6-lb  [p4-h6-lb-c01, -c02]                 h6-mt  [p4-h6-mt-c01, -c02]           (opa.trackb.remedies)
  trainfree  [p4-trainfree-test]                                                    (opa.trackb.trainfree: V0_own_origin_mass_ratio_to_uniform)

Rows (names exactly as in prereg/rules.yaml):
  V3_vs_B1_source_attribution         value = (V3 - B1) attribution EM in points, paired by the item seed over the pooled containers, paired
                                      bootstrap 95% CI (10,000 resamples, seed 0) and exact McNemar (opa.stats.contrast). Secondary row
                                      V3_vs_B1_source_attribution_rank (rank-restricted accuracy), reported only.
  V3_own_origin_mass_fraction         measurement: mean over the V3 probe windows (the one V3 container that ran them), n = windows
  V0_own_origin_mass_ratio_to_uniform measurement: V0 in the training-free test run, n = probe windows (cross-checked against variants.V0.probe)
  R5_vs_R1_accuracy                   (R5 with r = 8) - R1 in EM points over the pooled items of both cells, paired bootstrap + exact McNemar
  best_remedy_recovery_fraction       max over R3..R6 of acc(R) / acc(R0) (deployable); CI = percentile bootstrap of the maximum over resampled
                                      items (the best remedy is re-chosen inside each resample); note names the remedy
  best_remedy_recovery_fraction_incl_oracle   the same with R2oracle and R7oracle as candidates (an oracle upper bound, reported separately)

Hard errors (nothing is written): a run whose status is not "ok" (all are listed), a split mismatch (job.yaml / manifest.json vs --split), a
final_eval.json / remedies.json / metrics.json that is missing or not complete, a duplicate item (an attribution seed or a remedy item id repeated
across or within containers), a mixed training configuration within an arm (every config field outside train.RESUME_FREE_FIELDS must agree
across the containers, and so must the resumed checkpoint sha256), items that differ between V3 and B1 for the same seed (window, target origin,
options, gold option, length), a trainfree split other than --split, and (split test) a missing prereg/PREREG_TAG.json.

Intention to treat: an evaluation record always has a 0/1 outcome here (an item that crashed would have stopped the run, status not ok), so
n_itt_imputed is 0. Attribution seeds present in only one arm are listed in the exclusions table (n_excluded_missing); remedy items that were
requested but not finished inside a container's time budget (n_requested - n_items) are reported in the series table and in n_excluded_missing of
the H6 rows, with the stop reason.

Output files (opa.analysis.tables): trackb_contrasts_<split>.csv, trackb_series_<split>.csv, trackb_exclusions_<split>.csv, in the p2_results
column format (extra columns: split, hypothesis, systems, n_pairs, n_excluded_missing, n_itt_imputed, n10, n01, acc_a, acc_b, min_n, n_short,
boot_seed, n_boot, note). A table is never written under a name containing "dev" for split test (check_table_names). Read-only on GCS."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .. import PROJECT_DIR, stats
from . import p2_results as p2
from . import pilot as pl
from .tables import write_contrast_table, write_table

BOOT_SEED, N_BOOT = 0, 10_000
SERIES_ORDER = ["h4-v3", "h4-b1", "h6-lb", "h6-mt", "trainfree"]
K_V3_B1, K_V3_B1_RANK = "V3_vs_B1_source_attribution", "V3_vs_B1_source_attribution_rank"
K_V3_OWN, K_V0_RATIO = "V3_own_origin_mass_fraction", "V0_own_origin_mass_ratio_to_uniform"
K_R5_R1 = "R5_vs_R1_accuracy"
K_BEST, K_BEST_ORACLE = "best_remedy_recovery_fraction", "best_remedy_recovery_fraction_incl_oracle"
IDENT = ("window", "target_origin", "options_origin_ids", "gold_option", "T")
EXTRA_COLUMNS = ["split", "hypothesis", "systems", "n_pairs", "n_excluded_missing", "n_itt_imputed", "n10", "n01", "acc_a", "acc_b", "min_n",
                 "n_short", "boot_seed", "n_boot", "note"]
SERIES_COLUMNS = ["key", "split", "n_runs", "run_ids", "status", "git_sha", "n_items", "n_requested", "stop_reasons", "resume_sha256"]
EXCL_COLUMNS = ["key", "contrast", "item_id", "reason", "split"]
DEPLOYABLE, ORACLE = ("R3", "R4", "R5", "R6"), ("R2oracle", "R7oracle")
TEST_SEED0, H4_N_PLANNED, PROBE_PLANNED = 100000, 2451, 16
PINNED_SHA = {"V3": "f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814"}      # V3 step-600 checkpoint; B1 is given with --expect-sha


def _get(store, rid: str, rel: str):
    return pl._get_json(store, rid, rel)


def _git_sha(runs_dir: Path, run_ids: Sequence[str]) -> str:
    """git sha(s) from the runs' local manifest.json (";"-joined when they differ), "n/a" when no manifest is present."""
    out = []
    for rid in run_ids:
        mp = Path(runs_dir) / rid / "manifest.json"
        if mp.exists():
            g = json.loads(mp.read_text(encoding="utf-8")).get("git_sha")
            if g and str(g) not in out:
                out.append(str(g))
    return ";".join(out) if out else "n/a"


def _kind(cond: str) -> str:
    return cond.split("_")[0]


# ====================================================================== H4
def _load_arm(store, arm: str, run_ids: Sequence[str], split: str, runs_dir: Path, expect_sha: str | None = None) -> dict:
    """{"em": {seed: 0/1}, "rank": {seed: 0/1}, "meta": {seed: identity}, "probes": [(run id, block)], "sha": ..., "opt_steps": ..., "n_requested": ...}"""
    from ..trackb import train as T
    em, rank, meta, probes, cfgs, shas, steps = {}, {}, {}, [], [], set(), set()
    n_requested = 0
    for rid in run_ids:
        p2._load_job(runs_dir, rid, split)
        fe = _get(store, rid, "final_eval.json")
        if fe.get("status") != "complete":
            raise ValueError(f"{rid}: final_eval.json status {fe.get('status')!r}, expected 'complete'")
        cj = _get(store, rid, "config.json")
        cfg_full = cj.get("config") or {}
        blk = fe["per_item"]["final"]
        for where, got in (("config.json arm", cj.get("arm")), ("config.json config.arm", cfg_full.get("arm")), ("final_eval.json final arm", blk.get("arm"))):
            if got != arm:
                raise ValueError(f"{rid}: {where} is {got!r}, expected {arm!r} (arms swapped or the wrong run)")
        if cfg_full.get("eval_split") != split:
            raise ValueError(f"{rid}: config eval_split is {cfg_full.get('eval_split')!r}, expected {split!r}")
        if cj.get("opt_steps") != cfg_full.get("max_opt_steps") or cj.get("train_loss_mean") is not None:
            raise ValueError(f"{rid}: not an evaluation-only run (opt_steps {cj.get('opt_steps')!r}, max_opt_steps {cfg_full.get('max_opt_steps')!r}, "
                             f"train_loss_mean {cj.get('train_loss_mean')!r})")
        steps.add(int(cj["opt_steps"]))
        e, r, m = blk["per_item"]["attrib_em"], blk["per_item"]["attrib_rank"], blk["meta"]["attrib"]
        if not (len(e) == len(r) == len(m)):
            raise ValueError(f"{rid}: attribution outcomes and identities disagree in length")
        for ee, rr, mm in zip(e, r, m):
            sd = int(mm["seed"])
            if split == "test" and sd < TEST_SEED0:
                raise ValueError(f"{rid}: attribution seed {sd} is below {TEST_SEED0} (not a test seed)")
            if sd in em:
                raise ValueError(f"{arm}: duplicate attribution item (seed {sd}), repeated in {rid}")
            em[sd], rank[sd], meta[sd] = int(ee), int(rr), {k: mm.get(k) for k in IDENT}
        n_requested += int((cfg_full.get("monitor") or {}).get("attrib", 0))
        if (fe.get("final") or {}).get("probe"):
            probes.append((rid, fe["final"]["probe"]))
        cfg = {k: v for k, v in cfg_full.items() if k not in T.RESUME_FREE_FIELDS}
        cfgs.append((rid, json.dumps(cfg, sort_keys=True)))
        sha = cj.get("resumed_from_sha256")
        if split == "test" and not sha:
            raise ValueError(f"{rid}: resumed_from_sha256 is missing (a test evaluation must resume a pinned checkpoint)")
        pin = cfg_full.get("expected_resume_sha256")
        if pin and sha != pin:
            raise ValueError(f"{rid}: resumed sha256 {sha} differs from the run's expected_resume_sha256 {pin}")
        if expect_sha and sha != expect_sha:
            raise ValueError(f"{rid}: resumed sha256 {sha} differs from the expected {arm} checkpoint {expect_sha}")
        if sha:
            shas.add(sha)
    if len({c for _, c in cfgs}) > 1:
        raise ValueError(f"{arm}: mixed training configurations across containers: {[r for r, _ in cfgs]}")
    if len(shas) > 1:
        raise ValueError(f"{arm}: containers resumed different checkpoints (sha256 {sorted(shas)})")
    if len(steps) > 1:
        raise ValueError(f"{arm}: containers evaluated different step counts {sorted(steps)}")
    return {"em": em, "rank": rank, "meta": meta, "probes": probes, "sha": next(iter(shas), None), "opt_steps": next(iter(steps)),
            "n_requested": n_requested, "run_ids": list(run_ids)}


def _row(key: str, hyp: str, base: dict, split: str, run_ids, systems: str, min_n: Mapping[str, int], *, excluded: int = 0, note: str = "") -> dict:
    n = base["n"]
    m = min_n.get(key)
    base.update(split=split, hypothesis=hyp, systems=systems, n_pairs=n, n_excluded_missing=excluded, n_itt_imputed=0, min_n=m,
                n_short=(bool(n < m) if m is not None else None), boot_seed=BOOT_SEED, n_boot=N_BOOT, run_ids=list(run_ids), note=note)
    return base


def _measurement(key: str, hyp: str, value: float, n: int, split: str, run_ids, note: str) -> dict:
    return {"key": key, "value": float(value), "ci_lo": None, "ci_hi": None, "n": int(n), "p_raw": None, "p_holm": None,
            "test": "measurement (mean over probe windows; no test)", "run_ids": list(run_ids), "split": split, "hypothesis": hyp, "systems": key,
            "n_pairs": int(n), "n_excluded_missing": 0, "n_itt_imputed": 0, "n10": None, "n01": None, "acc_a": None, "acc_b": None, "min_n": None,
            "n_short": None, "boot_seed": None, "n_boot": None, "note": note}


def h4_rows(v3: dict, b1: dict, split: str, min_n: Mapping[str, int], planned: Sequence[int] | None = None) -> tuple[list, list]:
    if v3["opt_steps"] != b1["opt_steps"]:
        raise ValueError(f"V3 and B1 were evaluated at different step counts ({v3['opt_steps']} vs {b1['opt_steps']}): not an equal-steps comparison")
    if v3["sha"] and v3["sha"] == b1["sha"]:
        raise ValueError("V3 and B1 resumed the same checkpoint sha256: one arm was evaluated twice")
    if planned is None and split == "test":
        planned = range(TEST_SEED0, TEST_SEED0 + H4_N_PLANNED)
    common = sorted(set(v3["em"]) & set(b1["em"]))
    if not common:
        raise pl.MissingInput(f"{K_V3_B1}: no attribution seed in both arms")
    for s in common:
        if v3["meta"][s] != b1["meta"][s]:
            raise ValueError(f"attribution seed {s} differs between V3 and B1 ({v3['meta'][s]} vs {b1['meta'][s]}): not the same item")
    only = [(s, "V3") for s in sorted(set(v3["em"]) - set(b1["em"]))] + [(s, "B1") for s in sorted(set(b1["em"]) - set(v3["em"]))]
    excl = [{"item_id": str(s), "reason": f"no attribution record for {arm}'s counterpart (seed {s} evaluated by {arm} only)"} for s, arm in only]
    absent = [s for s in (planned or []) if s not in v3["em"] and s not in b1["em"]]
    excl += [{"item_id": str(s), "reason": f"planned seed {s} was evaluated by neither arm"} for s in absent]
    rids = v3["run_ids"] + b1["run_ids"]
    rows = []
    for key, field_, note in ((K_V3_B1, "em", f"argmax-over-vocabulary exact match; n={len(common)} seeds {common[0]}..{common[-1]}"),
                              (K_V3_B1_RANK, "rank", "secondary, reported: accuracy restricted to the option tokens")):
        a = np.array([v3[field_][s] for s in common], float)
        b = np.array([b1[field_][s] for s in common], float)
        r = stats.contrast(a, b, key, test="mcnemar", n_boot=N_BOOT, seed=BOOT_SEED, split=split)
        r.update(acc_a=float(a.mean()), acc_b=float(b.mean()))
        rows.append(_row(key, "H4", r, split, rids, "V3 vs B1", min_n, excluded=len(only) + len(absent), note=note))
    if len(v3["probes"]) != 1:
        raise pl.MissingInput(f"{K_V3_OWN}: exactly one V3 container must have run the probe windows, found {len(v3['probes'])}")
    rid, pr = v3["probes"][0]
    rows.append(_measurement(K_V3_OWN, "H4", pr["own_origin_mass_fraction"], pr.get("n_windows", 0), split, [rid],
                             f"V3 at its step-600 checkpoint, test probe windows (C={pr.get('C')}); excl-prefix {pr.get('own_origin_mass_fraction_excl_prefix')}"))
    return rows, excl


def v0_row(store, rid: str, split: str, planned: int = PROBE_PLANNED) -> dict:
    m = _get(store, rid, "metrics.json")
    if m.get("split") != split:
        raise ValueError(f"{rid}: trainfree metrics.json split is {m.get('split')!r}, expected {split!r}")
    v0 = ((m.get("variants") or {}).get("V0") or {}).get("probe") or {}
    val = (m.get("predictions_to_read") or {}).get(K_V0_RATIO)
    raw = v0.get("own_origin_mass_ratio_to_uniform")
    if val is None or raw is None or abs(val - raw) > 1e-9:
        raise ValueError(f"{rid}: {K_V0_RATIO} missing or inconsistent between predictions_to_read and variants.V0.probe")
    pj = _get(store, rid, "probe_V0.json")
    n_ok = int(pj.get("n_ok", pj.get("n_windows", 0)))
    if n_ok < 1:
        raise ValueError(f"{rid}: probe_V0.json has no usable window")
    row = _measurement(K_V0_RATIO, "H4", val, n_ok, split, [rid],
                       f"training-free V0 (no training), test probe windows (C={v0.get('C')}); 1.0 = uniform attention over origins; "
                       f"{n_ok} of {planned} planned windows usable")
    row["n_excluded_missing"] = max(planned - n_ok, 0)
    return row


# ====================================================================== H6
def load_h6(store, name: str, run_ids: Sequence[str], split: str, runs_dir: Path) -> dict:
    per_item, ids, requested, stops, excl = [], [], 0, [], []
    for rid in run_ids:
        p2._load_job(runs_dir, rid, split)
        rj = _get(store, rid, "remedies.json")
        ra = rj.get("args") or {}
        if ra.get("split") != split:
            raise ValueError(f"{rid}: remedies.json args.split is {ra.get('split')!r}, expected {split!r}")
        if split == "test" and int(ra.get("seed0", -1)) < TEST_SEED0:
            raise ValueError(f"{rid}: remedies args.seed0 {ra.get('seed0')!r} is below {TEST_SEED0} (not a test run)")
        pi, iid = rj["per_item"], rj["item_ids"]
        if len(pi) != len(iid):
            raise ValueError(f"{rid}: remedies.json per_item and item_ids differ in length")
        dup = [i for i in iid if i in set(ids)] or [i for i in set(iid) if iid.count(i) > 1]
        if dup:
            raise ValueError(f"{name}: duplicate remedy item ids {sorted(dup)[:3]} in {rid}")
        per_item += pi
        ids += iid
        n_req = int(rj["summary"].get("n_requested", len(iid)))
        requested += n_req
        stops.append(str(rj["summary"].get("stop_reason")))
        if len(iid) < n_req:
            off = int(ra.get("offset", 0))
            excl += [{"item_id": f"{name}#{off + k}", "reason": f"requested item (global index {off + k}) not finished in {rid} ({rj['summary'].get('stop_reason')})"}
                     for k in range(len(iid), n_req)]
    return {"name": name, "run_ids": list(run_ids), "per_item": per_item, "ids": ids, "n_requested": requested, "stops": stops, "excl": excl}


def _boot_best(arr: dict, cands: list, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile CI of max over cands of acc(c) / acc(R0), the best candidate re-chosen inside every resample."""
    n = len(arr["R0"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    r0 = arr["R0"][idx].mean(1)
    best = np.max([arr[c][idx].mean(1) for c in cands], axis=0)
    ok = r0 > 0
    ratio = best[ok] / r0[ok]
    return float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))


def h6_rows(series: list, split: str, min_n: Mapping[str, int]) -> tuple[list, list, list]:
    from ..trackb import remedies as RM
    per_item = [r for s in series for r in s["per_item"]]
    ids = [i for s in series for i in s["ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate remedy item ids across cells: {sorted({i for i in ids if ids.count(i) > 1})[:3]}")
    conds = list(per_item[0])
    if any(list(r) != conds for r in per_item):
        raise ValueError("remedy items disagree on the condition list")
    arr = {c: np.array([r[c]["em"] for r in per_item], float) for c in conds}
    n = len(per_item)
    rids = [r for s in series for r in s["run_ids"]]
    n_missing = sum(s["n_requested"] - len(s["ids"]) for s in series)
    excl = [e for s in series for e in s["excl"]]
    rows, skipped = [], []
    if "R5_8" not in arr or "R1" not in arr or "R0" not in arr:
        skipped.append(f"{K_R5_R1}/{K_BEST}: conditions R0, R1 and R5_8 are all needed, found {conds}")
    if "R5_8" in arr and "R1" in arr:
        r = stats.contrast(arr["R5_8"], arr["R1"], K_R5_R1, test="mcnemar", n_boot=N_BOOT, seed=BOOT_SEED, split=split)
        r.update(acc_a=float(arr["R5_8"].mean()), acc_b=float(arr["R1"].mean()))
        rows.append(_row(K_R5_R1, "H6", r, split, rids, "R5 (r=8) vs R1", min_n, excluded=n_missing,
                         note=f"pooled over {', '.join(s['name'] for s in series)}; n={n}"))
    a0 = float(arr["R0"].mean()) if "R0" in arr else 0.0
    low = a0 < RM.R0_MIN_ACC
    if a0 <= 0:
        skipped.append(f"{K_BEST}: acc(R0) is {a0}, the recovery fraction is undefined")
    for key, kinds in ((K_BEST, DEPLOYABLE), (K_BEST_ORACLE, DEPLOYABLE + ORACLE)):
        cands = [c for c in conds if c not in ("R0", "R1") and _kind(c) in kinds]
        if not cands or a0 <= 0:
            continue
        rec = {c: float(arr[c].mean()) / a0 for c in cands}
        best = max(rec, key=lambda c: (rec[c], c))
        lo, hi = _boot_best(arr, cands)
        oracle = key == K_BEST_ORACLE
        rows.append({"key": key, "value": rec[best], "ci_lo": lo, "ci_hi": hi, "n": n, "p_raw": None, "p_holm": None,
                     "test": "measurement: acc(R)/acc(R0), percentile bootstrap of the maximum over items (10,000 resamples)",
                     "run_ids": rids, "split": split, "hypothesis": "H6", "systems": f"{best} vs R0" + (" (oracle candidates included)" if oracle else ""),
                     "n_pairs": n, "n_excluded_missing": n_missing, "n_itt_imputed": 0, "n10": None, "n01": None,
                     "acc_a": float(arr[best].mean()), "acc_b": a0, "min_n": None, "n_short": None, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
                     "note": f"best = {best}; recoveries " + ", ".join(f"{c} {rec[c]:.3f}" for c in sorted(rec))
                             + ("; ORACLE upper bound, not a deployable result" if oracle else "")
                             + (f"; r0_too_low: acc(R0) {a0:.3f} < {RM.R0_MIN_ACC}" if low else "")})
    return rows, excl, skipped


# ====================================================================== driver
def run_trackb(store, series_runs: Mapping[str, Sequence[str]], split: str, out_dir: Path | str, *, runs_dir: Path | str | None = None,
               min_n: Any = None, tag_file: Path | str | None = None, expect_sha: Mapping[str, str] | None = None,
               planned_seeds: Sequence[int] | None = None) -> dict:
    """expect_sha: {"V3": sha, "B1": sha} the checkpoints the arms must have resumed; default for split test: the pinned V3 step-600 sha."""
    runs_dir = Path(runs_dir) if runs_dir is not None else PROJECT_DIR / "runs"
    names = [f"trackb_contrasts_{split}.csv", f"trackb_series_{split}.csv", f"trackb_exclusions_{split}.csv"]
    p2.check_table_names(out_dir, names, split)
    p2.require_prereg_tag(split, tag_file)
    unknown = sorted(set(series_runs) - set(SERIES_ORDER))
    if unknown:
        raise ValueError(f"unknown series {unknown}; known: {SERIES_ORDER}")
    series_runs = {k: list(v) for k, v in series_runs.items() if v}
    mn = p2.load_min_n(min_n)
    expect = dict(PINNED_SHA if split == "test" else {})
    expect.update(expect_sha or {})
    p2.check_statuses(store, runs_dir, series_runs)
    rows, skipped, excl, ser = [], [], [], []

    def add_excl(key: str, items: Sequence[dict]):
        excl.extend({"key": f"{key}|{e['item_id']}", "contrast": key, "item_id": e["item_id"], "reason": e["reason"], "split": split} for e in items)

    if "h4-v3" in series_runs and "h4-b1" in series_runs:
        v3 = _load_arm(store, "V3", series_runs["h4-v3"], split, runs_dir, expect.get("V3"))
        b1 = _load_arm(store, "B1", series_runs["h4-b1"], split, runs_dir, expect.get("B1"))
        r, ex = h4_rows(v3, b1, split, mn, planned_seeds)
        rows += r
        add_excl(K_V3_B1, ex)
        for nm, a in (("h4-v3", v3), ("h4-b1", b1)):
            ser.append({"key": nm, "split": split, "n_runs": len(a["run_ids"]), "run_ids": a["run_ids"], "status": "ok", "git_sha": _git_sha(runs_dir, a["run_ids"]), "n_items": len(a["em"]),
                        "n_requested": a["n_requested"], "stop_reasons": "", "resume_sha256": a["sha"]})
    else:
        skipped.append(f"{K_V3_B1}: series h4-v3 and h4-b1 not both given")
    if series_runs.get("trainfree"):
        if len(series_runs["trainfree"]) != 1:
            raise ValueError("trainfree: exactly one run id expected")
        p2._load_job(runs_dir, series_runs["trainfree"][0], split)
        vr = v0_row(store, series_runs["trainfree"][0], split)
        rows.append(vr)
        ser.append({"key": "trainfree", "split": split, "n_runs": 1, "run_ids": series_runs["trainfree"], "status": "ok",
                    "git_sha": _git_sha(runs_dir, series_runs["trainfree"]), "n_items": vr["n"], "n_requested": vr["n"] + vr["n_excluded_missing"],
                    "stop_reasons": "", "resume_sha256": "n/a"})
    else:
        skipped.append(f"{K_V0_RATIO}: series trainfree not given")
    h6 = [load_h6(store, k, series_runs[k], split, runs_dir) for k in ("h6-lb", "h6-mt") if series_runs.get(k)]
    if h6:
        r, ex, sk = h6_rows(h6, split, mn)
        rows += r
        skipped += sk
        add_excl(K_R5_R1, ex)
        for s in h6:
            ser.append({"key": s["name"], "split": split, "n_runs": len(s["run_ids"]), "run_ids": s["run_ids"], "status": "ok", "git_sha": _git_sha(runs_dir, s["run_ids"]), "n_items": len(s["ids"]),
                        "n_requested": s["n_requested"], "stop_reasons": ";".join(s["stops"]), "resume_sha256": ""})
    else:
        skipped.append(f"{K_R5_R1}: no h6 series given")
    out = Path(out_dir)
    write_contrast_table(out / names[0], rows, extra_columns=EXTRA_COLUMNS)
    write_table(out / names[1], ser, columns=SERIES_COLUMNS)
    write_table(out / names[2], excl, columns=EXCL_COLUMNS)
    short = [f"{r['key']}: n={r['n']} < min_n={r['min_n']}" for r in rows if r.get("n_short")]
    return {"rows": rows, "skipped": skipped, "short": short, "written": [str(out / n) for n in names], "exclusions": excl}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--runs", required=True, help="YAML/JSON mapping series -> [run ids]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--creds", default=None)
    ap.add_argument("--expect-sha", action="append", default=[], metavar="ARM=SHA", help="checkpoint sha256 an arm must have resumed (V3 defaults to the pinned step-600 sha; give B1=...)")
    a = ap.parse_args(argv)
    creds = a.creds or pl.credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")        # never printed
    try:
        res = run_trackb(pl.GcsStore(), p2.load_mapping(a.runs), a.split, a.out, runs_dir=a.runs_dir, min_n=a.min_n,
                         expect_sha=dict(x.split("=", 1) for x in a.expect_sha))
    except (ValueError, PermissionError, pl.MissingInput) as e:
        print(f"trackb_test FAILED: {e}", file=sys.stderr)
        return 1
    for r in res["rows"]:
        print(f"{r['key']}: value={r['value']:.4g} n={r['n']}")
    for s in res["skipped"]:
        print("skipped:", s)
    for s in res["short"]:
        print("SHORT:", s)
    print("wrote:", *res["written"])
    return 0 if res["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
