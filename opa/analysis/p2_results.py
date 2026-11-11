"""P2 results pipeline: many containers per series -> the pre-registered contrast tables, in one command.

  python -m opa.analysis.p2_results --split dev  --runs runs_dev.yaml  --out bundle/tables/p2_dev_check
  python -m opa.analysis.p2_results --split test --runs runs_p2.yaml   --out bundle/tables/p2 [--min-n prereg/sample_sizes.yaml]

``--runs`` is a YAML/JSON mapping  series -> [run id, ...]  (one id per container). Series names are those of
jobs/make_p2_jobs.py:
  embed-lb sk-lb embed-mt sk-mt   prep jobs: only their status is checked
  h1-lb      S11, S2, S8 on every container (multi-system runs)       h1-s4   S4 on the first n_S4 items
  h5-mt      S12, S14 on every container, S15 on the first n_tost     h10-pin / h10-nopin   thrash runs
H2 (drift, h2-d2 / h2-d4) is deferred / not funded and is not read here.

What it does
- Merges per-item records over the containers of a series. HARD ERRORS (nothing is written): a run whose status is
  not "ok" (every such run is listed), a split mismatch (job.yaml / manifest.json vs --split), a duplicate item id
  (in a series' item lists or in one system's records), a mixed config_version / prompt_version within a (series,
  system), records missing for a system the job asked for, a per-item EM that disagrees with the run's metrics.json,
  an unexpected repeat, and (split test) a missing prereg/PREREG_TAG.json (test data is read only after the tag).
- Computes the contrast rows (names exactly as in prereg/rules.yaml):
    S11_vs_S2_lbmhqa_late_32k_primary, S11_vs_S4_lbmhqa_late_32k_primary   (H1)
    S11_vs_S8_late_bound                                                    (H3)
    (R2_predicted_vs_observed_success is deliberately NEVER emitted: not testable at 32K, no result is reported)
    A2_vs_A1_agent_mtlbmhqa_32k_primary_cross (S12 vs S14), A2_vs_A3_agent_equivalence (S12 vs S15, TOST)   (H5)
    A2_vs_A1_pages_read_before_gold_hit (S12 - S14, reported, no verdict, Wilcoxon + bootstrap)               (H5)
    descent_beam4_vs_flat_recall_at_M                                       (H8, non-inferiority, CI only)
    pinned_vs_unpinned_repeat_pagein_reduction                              (H10)
  Paired outcomes: paired bootstrap (10,000 resamples, seed recorded in the table), exact McNemar for binary EM
  (opa.stats.contrast), TOST +-3 with Tango's score test for A2 vs A3 (opa.stats.tost_paired_p), H10 through
  opa.tracks.track_a.thrash.h10_contrast (Wilcoxon).
- Intention to treat: an arm record whose status is not "ok" scores EM 0 and stays in the pair (n_itt_imputed counts
  pairs with at least one such record; n_error_a / n_error_b count per arm). H10: the worst observed value of the
  (r, M) cell (thrash.h10_contrast). A pair is dropped only when an arm that was SCHEDULED for the item has no
  record for it (n_excluded_missing, reason per item in the exclusions table). Items scheduled for only one arm by
  design (S4 beyond n_S4, S15 beyond n_tost) are not exclusions; they are counted in n_unpaired_by_design.
  Ties in every ordering (bins, pairs) are broken by item id.
- ``min_n`` (a mapping, or a YAML/JSON path like prereg/sample_sizes.yaml) gives the pre-registered minimum n per key;
  ``n_short`` is true when the achieved n is below it. p_holm is left empty: Holm runs across all primaries at P6
  (opa.claims.build_claims). The H8 row is a CI-only key (p_raw blank, opa.claims.non_inferiority_keys).
- Output files (opa.analysis.tables, deterministic): p2_contrasts_<split>.csv, p2_series_<split>.csv,
  p2_exclusions_<split>.csv. A table is never written under a name containing "test" for split dev (or "dev" for
  test): the check covers the output directory's own name and the file names.
Read-only on GCS; spends nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .. import PROJECT_DIR, stats
from . import pilot as pl
from . import routing_hits as rh
from . import routing_recall as rr
from .tables import NI_TEST_LABEL, write_contrast_table, write_table

BOOT_SEED = 0
N_BOOT = 10_000
TOST_MARGIN = 3.0
H8_M = pl.H8_M
SERIES_ORDER = ["embed-lb", "sk-lb", "h1-lb", "h1-s4", "embed-mt", "sk-mt", "h5-mt", "h10-pin", "h10-nopin"]
PREP_SERIES = ("embed-lb", "sk-lb", "embed-mt", "sk-mt")
MULTI_SERIES = ("h1-lb", "h1-s4", "h5-mt")
THRASH_SERIES = ("h10-pin", "h10-nopin")
DEFERRED_SERIES = ("h2-d2", "h2-d4")
K_TOST = pl.K_A2_A3
K_PAGES = "A2_vs_A1_pages_read_before_gold_hit"
K_H10 = pl.K_H10
K_H8 = rr.key_for(rr.PRIMARY_BEAM)
CONTRASTS = [  # (key, series a, system a, series b, system b, hypothesis)
    (pl.K_S11_S2, "h1-lb", "S11", "h1-lb", "S2", "H1"),
    (pl.K_S11_S4, "h1-lb", "S11", "h1-s4", "S4", "H1"),
    (pl.K_S11_S8, "h1-lb", "S11", "h1-lb", "S8", "H3"),
    (pl.K_A2_A1, "h5-mt", "S12", "h5-mt", "S14", "H5"),
    (K_TOST, "h5-mt", "S12", "h5-mt", "S15", "H5"),
]
EXTRA_COLUMNS = ["split", "hypothesis", "systems", "n_pairs", "n_excluded_missing", "n_itt_imputed", "n_error_a",
                 "n_error_b", "n_budget_a", "n_budget_b", "n_not_fit_a", "n_not_fit_b", "n_censored_a", "n_censored_b", "n_unpaired_by_design", "n10", "n01", "em_a", "em_b", "min_n", "n_short", "boot_seed",
                 "n_boot", "tost_margin", "tost_p_lower", "tost_p_upper", "tost_equivalent", "ni_margin", "ni_met", "imputed_worst", "note"]
SERIES_COLUMNS = ["key", "split", "n_runs", "run_ids", "n_items", "systems", "n_records", "config_version",
                  "prompt_version", "git_sha", "em_check"]
EXCL_COLUMNS = ["key", "contrast", "item_id", "reason", "split"]


class RunStatusError(ValueError):
    """At least one run of a series is not status ok (or has no result)."""


# ====================================================================== guards
def check_table_names(out_dir: Path | str, names: Sequence[str], split: str) -> None:
    """Refuse a table name (the output directory's own name or a file name) that names the other split."""
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    bad = "test" if split == "dev" else "dev"
    for n in [Path(out_dir).name, *names]:
        if bad in n.lower():
            raise ValueError(f"refusing to write {n!r} for split {split!r}: the name contains {bad!r}")


def require_prereg_tag(split: str, tag_file: Path | str | None = None) -> None:
    if split == "test":
        p = Path(tag_file) if tag_file is not None else PROJECT_DIR / "prereg" / "PREREG_TAG.json"
        if not p.exists():
            raise PermissionError(f"split test: {p} does not exist; test results are produced only after the "
                                  f"prereg-v2 tag (CLAUDE.md Section 6)")


def load_mapping(src: Any) -> dict:
    if src is None:
        return {}
    if isinstance(src, Mapping):
        return dict(src)
    return yaml.safe_load(Path(src).read_text(encoding="utf-8")) or {}


def load_min_n(src: Any) -> dict[str, int]:
    """{rule key: minimum n}. ``src``: a dict, or a YAML/JSON path shaped like prereg/sample_sizes.yaml (a ``min_n``
    section; ``equivalence.A2_vs_A3_agent_min_n`` feeds the TOST key) or a flat mapping. Null values are dropped."""
    d = load_mapping(src)
    flat = dict(d["min_n"]) if isinstance(d.get("min_n"), Mapping) else {k: v for k, v in d.items() if not isinstance(v, (Mapping, list))}
    eq = d.get("equivalence") or {}
    if isinstance(eq, Mapping) and eq.get("A2_vs_A3_agent_min_n") is not None:
        flat.setdefault(K_TOST, eq["A2_vs_A3_agent_min_n"])
    return {str(k): int(v) for k, v in flat.items() if v is not None}


def _rule_leaf(rules: dict | None, key: str) -> dict:
    if not rules:
        return {}
    from ..claims import rule_leaves
    for h in rules.get("hypotheses", {}).values():
        for leaf in rule_leaves(h["rule"]):
            if leaf.get("key") == key:
                return leaf
    return {}


def load_rules() -> dict | None:
    try:
        from ..claims import load_prereg
        return load_prereg()[0]
    except Exception:
        return None


# ====================================================================== loading
@dataclass
class Series:
    name: str
    run_ids: list[str] = field(default_factory=list)
    items: dict = field(default_factory=dict)                     # item_id -> item (all containers)
    scheduled: dict = field(default_factory=dict)                 # system -> set of item ids scheduled
    records: dict = field(default_factory=dict)                   # system -> {item_id: record}
    flat: list = field(default_factory=list)                      # thrash records (all containers)
    jobs: dict = field(default_factory=dict)                      # run_id -> job spec
    versions: dict = field(default_factory=lambda: defaultdict(lambda: {"config_version": set(), "prompt_version": set()}))
    em_check: dict = field(default_factory=dict)                  # (run_id, system) -> "ok"
    thrash_sched: dict = field(default_factory=dict)              # (item_id, M) -> r (thrash series)
    git_shas: set = field(default_factory=set)


def _load_job(runs_dir: Path, run_id: str, split: str) -> dict:
    p = runs_dir / run_id / "job.yaml"
    if not p.exists():
        raise pl.MissingInput(f"{p} not found")
    job = yaml.safe_load(p.read_text(encoding="utf-8"))
    if job.get("split") != split:
        raise ValueError(f"{run_id}: job split is {job.get('split')!r}, expected {split!r}")
    mp = runs_dir / run_id / "manifest.json"
    if mp.exists():
        ms = json.loads(mp.read_text(encoding="utf-8")).get("split")
        if ms is not None and ms != split:
            raise ValueError(f"{run_id}: manifest split is {ms!r}, expected {split!r}")
    return job


def _run_status(store, runs_dir: Path, run_id: str) -> str:
    p = runs_dir / run_id / "result.json"
    try:
        if p.exists():
            return str(json.loads(p.read_text(encoding="utf-8")).get("status"))
        return str(json.loads(store.get(pl.run_uri(store, run_id) + "/result.json")).get("status"))
    except (KeyError, ValueError):
        return "no result.json"


def check_statuses(store, runs_dir: Path, series_runs: Mapping[str, Sequence[str]]) -> None:
    bad = []
    for name, ids in series_runs.items():
        for rid in ids:
            st = _run_status(store, runs_dir, rid)
            if st != "ok":
                bad.append(f"{name}: {rid} status={st}")
    if bad:
        raise RunStatusError("runs that are not status ok (fix, re-run or record them as failed; nothing is dropped "
                             "silently):\n  " + "\n  ".join(bad))


def _note_versions(s: Series, system: str, recs: Sequence[dict], sub: dict | None) -> None:
    v = s.versions[system]
    for src in list(recs) + ([sub] if sub else []):
        for k in ("config_version", "prompt_version"):
            if src.get(k) is not None:
                v[k].add(str(src[k]))


def _check_versions(s: Series) -> None:
    for system, v in s.versions.items():
        for k, vals in v.items():
            if len(vals) > 1:
                raise ValueError(f"series {s.name} system {system}: mixed {k} {sorted(vals)}")


def load_series(store, name: str, run_ids: Sequence[str], split: str, runs_dir: Path | str) -> Series:
    """Load and merge the containers of one series (module docstring lists the hard errors)."""
    runs_dir = Path(runs_dir)
    s = Series(name, list(run_ids))
    pat = re.compile(r"/([^/]+)/r(\d+)/records\.jsonl\.gz$")
    for rid in run_ids:
        job = _load_job(runs_dir, rid, split)
        s.jobs[rid] = job
        mp = runs_dir / rid / "manifest.json"
        if mp.exists():
            g = json.loads(mp.read_text(encoding="utf-8")).get("git_sha")
            if g:
                s.git_shas.add(str(g))
        if name in PREP_SERIES:
            continue
        items = pl.load_run_items(store, job)
        dup = sorted(set(items) & set(s.items))
        if dup:
            raise ValueError(f"series {name}: item ids in more than one container: {dup[:3]} (run {rid})")
        s.items.update(items)
        if name in THRASH_SERIES:
            recs = pl.read_jsonl(store.get(pl.run_uri(store, rid) + "/records.jsonl.gz"))
            sub = pl._metrics(store, rid)
            _check_stray(items, recs, rid, "thrash")
            s.em_check[(rid, "thrash")] = _thrash_check(recs, sub, rid)
            for iid, it in items.items():
                s.thrash_sched[(iid, int(job["args"]["M"]))] = (it.get("params") or {}).get("r")
            seen = {(r["item_id"], r.get("M")) for r in s.flat}
            for r in recs:
                if (r["item_id"], r.get("M")) in seen:
                    raise ValueError(f"series {name}: duplicate record for item {r['item_id']} (M={r.get('M')}) in run {rid}")
            s.flat.extend(recs)
            _note_versions(s, "thrash", recs, sub)
            continue
        found: dict[str, dict[int, list]] = defaultdict(dict)
        subm: dict = {}
        for u in store.ls(pl.run_uri(store, rid) + "/"):
            m = pat.search(u)
            if m:
                found[m.group(1)][int(m.group(2))] = pl.read_jsonl(store.get(u))
                try:
                    subm[(m.group(1), int(m.group(2)))] = json.loads(store.get(u.replace("records.jsonl.gz", "metrics.json")))
                except (KeyError, ValueError):
                    pass
        for system in job["args"]["systems"]:
            reps = found.get(system, {})
            if not reps:
                raise ValueError(f"run {rid}: no records for {system} (the job asked for it)")
            if set(reps) != {0}:
                raise ValueError(f"run {rid} {system}: unexpected repeats {sorted(reps)}; P2 runs have repeat 0 only")
            recs = reps[0]
            _check_stray(items, recs, rid, system)
            if system == "S4" and split == "test":
                from ..tracks.track_a.systems import S4_CONFIG_VERSION
                bad = sorted({str(r.get("config_version")) for r in recs if r.get("config_version") != S4_CONFIG_VERSION})
                if bad:
                    raise ValueError(f"run {rid} S4: config_version {bad} != {S4_CONFIG_VERSION!r} (the frozen S4 configuration)")
            by = s.records.setdefault(system, {})
            for r in recs:
                if r["item_id"] in by:
                    raise ValueError(f"series {name} {system}: duplicate record for item {r['item_id']} (run {rid})")
                by[r["item_id"]] = r
            s.scheduled.setdefault(system, set()).update(items)
            _note_versions(s, system, recs, subm.get((system, 0)))
            s.em_check[(rid, system)] = _em_check(items, recs, subm.get((system, 0)), rid, system)
    _check_versions(s)
    return s


def _check_stray(items: Mapping[str, dict], recs: Sequence[dict], rid: str, system: str) -> None:
    stray = [r["item_id"] for r in recs if r["item_id"] not in items]
    if stray:
        raise ValueError(f"run {rid} {system}: {len(stray)} record(s) for items outside the run's own item list, "
                         f"first {stray[:3]}")


def _em_check(items: Mapping[str, dict], recs: Sequence[dict], sub: dict | None, rid: str, system: str) -> str:
    """Mean harness EM over ALL records == the sub-run's metrics.json em (as pilot.check_em_consistency): a mismatch
    raises ValueError, a missing metrics.json / em raises MissingInput."""
    if not sub or "em" not in sub:
        raise pl.MissingInput(f"{rid} {system} r0: no metrics.json em; cannot verify per-item EM")
    if not recs:
        raise ValueError(f"{rid} {system}: no records")
    mean = float(np.mean([pl.harness_em(items[r["item_id"]], r) for r in recs]))
    if abs(mean - float(sub["em"])) > pl.TOL:
        raise ValueError(f"{rid} {system}: mean per-item EM {mean!r} != metrics.json em {sub['em']!r}")
    return "ok"


def _thrash_check(recs: Sequence[dict], sub: dict | None, rid: str) -> str:
    """Mean repeated page-ins over the scored records (status ok / step_cap) == metrics.json repeated_page_ins_mean
    (as pilot.check_thrash_consistency)."""
    if not sub or sub.get("repeated_page_ins_mean") is None:
        raise pl.MissingInput(f"{rid}: no metrics.json repeated_page_ins_mean; cannot verify thrash outcomes")
    sc = [r["repeated_page_ins"] for r in recs if r.get("status") in pl.THRASH_SCORED]
    mean = float(np.mean(sc)) if sc else float("nan")
    if not abs(mean - float(sub["repeated_page_ins_mean"])) <= pl.TOL:
        raise ValueError(f"{rid}: mean repeated page-ins {mean!r} != metrics.json repeated_page_ins_mean "
                         f"{sub['repeated_page_ins_mean']!r}")
    return "ok"


# ====================================================================== contrasts
def _em(item: Mapping, rec: Mapping) -> tuple[float, str | None]:
    """(EM, failure class): a record whose status is not "ok" scores 0 (intention to treat); the class is "budget",
    "not_fit" (does_not_fit) or "error" (any other non-ok status), None for an ok record."""
    st = rec.get("status")
    if st == "ok":
        return pl.harness_em(item, rec), None
    return 0.0, {"budget": "budget", "does_not_fit": "not_fit"}.get(st, "error")


def paired_arrays(sa: Series, sys_a: str, sb: Series, sys_b: str) -> dict:
    """Aligned 0/1 EM arrays (sorted by item id) for scheduled-in-both items with a record in both arms."""
    if pl.ORACLE_SYSTEMS & {sys_a, sys_b}:
        raise ValueError(f"oracle systems never enter a pair ({sys_a} vs {sys_b})")
    for s, sy in ((sa, sys_a), (sb, sys_b)):
        if sy not in s.records:
            raise pl.MissingInput(f"series {s.name} has no records for {sy}")
    items = {**sa.items, **sb.items}
    sched = sa.scheduled[sys_a] & sb.scheduled[sys_b]
    by_design = len(sa.scheduled[sys_a] | sb.scheduled[sys_b]) - len(sched)
    ids, a, b, excluded = [], [], [], []
    n_imp = 0
    cnt = {(c, a): 0 for c in ('error', 'budget', 'not_fit') for a in 'ab'}
    for iid in sorted(sched):
        ra, rb = sa.records[sys_a].get(iid), sb.records[sys_b].get(iid)
        if ra is None or rb is None:
            why = "missing both" if ra is None and rb is None else f"missing {sys_a if ra is None else sys_b}"
            excluded.append({"item_id": iid, "reason": why})
            continue
        ea, ia = _em(items[iid], ra)
        eb, ib = _em(items[iid], rb)
        ids.append(iid)
        a.append(ea)
        b.append(eb)
        for arm, cls in (("a", ia), ("b", ib)):
            if cls:
                cnt[(cls, arm)] += 1
        n_imp += bool(ia or ib)
    return {"ids": ids, "a": np.array(a), "b": np.array(b), "excluded": excluded, "n_itt_imputed": n_imp,
            "n_error_a": cnt[("error", "a")], "n_error_b": cnt[("error", "b")],
            "n_budget_a": cnt[("budget", "a")], "n_budget_b": cnt[("budget", "b")],
            "n_not_fit_a": cnt[("not_fit", "a")], "n_not_fit_b": cnt[("not_fit", "b")], "n_unpaired_by_design": by_design}


def _short(n: int, min_n: Mapping[str, int], key: str) -> tuple[Any, Any]:
    m = min_n.get(key)
    return (m, bool(n < m)) if m is not None else (None, None)


def _row(key: str, hyp: str, base: dict, split: str, run_ids: Sequence[str], min_n: Mapping[str, int],
         systems: str, p: dict, note: str = "") -> dict:
    n = base["n"]
    mn, short = _short(n, min_n, key)
    base.update(split=split, hypothesis=hyp, systems=systems, n_pairs=n, n_excluded_missing=len(p["excluded"]),
                n_itt_imputed=p["n_itt_imputed"], n_error_a=p["n_error_a"], n_error_b=p["n_error_b"],
                n_budget_a=p.get("n_budget_a", 0), n_budget_b=p.get("n_budget_b", 0),
                n_not_fit_a=p.get("n_not_fit_a", 0), n_not_fit_b=p.get("n_not_fit_b", 0),
                n_unpaired_by_design=p["n_unpaired_by_design"], min_n=mn, n_short=short, boot_seed=BOOT_SEED,
                n_boot=N_BOOT, run_ids=list(run_ids), note=note)
    return base


def pair_contrast(key: str, hyp: str, sa: Series, sys_a: str, sb: Series, sys_b: str, split: str,
                  min_n: Mapping[str, int], seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    p = paired_arrays(sa, sys_a, sb, sys_b)
    if not p["ids"]:
        raise pl.MissingInput(f"{key}: no item has a record in both arms")
    row = stats.contrast(p["a"], p["b"], key, test="mcnemar", n_boot=n_boot, seed=seed, split=split)
    row.update(em_a=float(p["a"].mean()), em_b=float(p["b"].mean()))
    rids = list(dict.fromkeys(sa.run_ids + sb.run_ids))
    row = _row(key, hyp, row, split, rids, min_n, f"{sys_a} vs {sys_b}", p)
    if key == K_TOST:
        t = stats.tost_paired_p(p["a"], p["b"], TOST_MARGIN)
        row.update(p_raw=t["p"], tost_margin=TOST_MARGIN, tost_p_lower=t["p_lower"], tost_p_upper=t["p_upper"],
                   tost_equivalent=t["equivalent"],
                   test=f"paired bootstrap 95% CI (10,000 resamples); equivalence: TOST +-{TOST_MARGIN:g} points, Tango score "
                        f"test, p_raw = max of the two one-sided p")
    return row


def pages_before_gold(item: dict, rec: dict) -> int | None:
    """0-based index in rec["paged_in"] of the first origin holding a gold fact of the item; None (censored) if
    there is none or the record is not status ok."""
    if rec.get("status") != "ok":
        return None
    gold = rr.gold_origin_ids(item)
    for k, o in enumerate(rh._ids(rec.get("paged_in"))):
        if o in gold:
            return k
    return None


def pages_row(sa: Series, split: str, min_n: Mapping[str, int], sys_a: str = "S12", sys_b: str = "S14") -> tuple[dict, list]:
    """A2_vs_A1_pages_read_before_gold_hit (reported H5 secondary, no verdict, never primary): mean over paired items
    of pages_before_gold(A2) - pages_before_gold(A1). A censored item (no gold origin in paged_in, or a failed record)
    gets max(len(paged_in) of its record, largest observed pages_before_gold over both arms of the paired items).
    Bootstrap 95% CI + Wilcoxon signed-rank (zero differences dropped), scale 1."""
    for sy in (sys_a, sys_b):
        if sy not in sa.records:
            raise pl.MissingInput(f"series {sa.name} has no records for {sy}")
    sched = sa.scheduled[sys_a] & sa.scheduled[sys_b]
    ids, raw, excluded = [], [], []
    for iid in sorted(sched):
        ra, rb = sa.records[sys_a].get(iid), sa.records[sys_b].get(iid)
        if ra is None or rb is None:
            excluded.append({"item_id": iid, "reason": "missing both" if ra is None and rb is None else f"missing {sys_a if ra is None else sys_b}"})
            continue
        it = sa.items[iid]
        ids.append(iid)
        raw.append([(pages_before_gold(it, r), len(rh._ids(r.get("paged_in")))) for r in (ra, rb)])
    if not ids:
        raise pl.MissingInput(f"{K_PAGES}: no item has a record in both arms")
    seen = [v for pair in raw for v, _ in pair if v is not None]
    worst = max(seen) if seen else 0
    vals = [[v if v is not None else max(n, worst) for v, n in pair] for pair in raw]
    a, b = np.array([v[0] for v in vals], float), np.array([v[1] for v in vals], float)
    cen_a, cen_b = sum(p[0][0] is None for p in raw), sum(p[1][0] is None for p in raw)
    row = stats.contrast(a, b, K_PAGES, test="wilcoxon", scale=1.0, n_boot=N_BOOT, seed=BOOT_SEED, split=split)
    row.update(em_a=float(a.mean()), em_b=float(b.mean()), n_censored_a=cen_a, n_censored_b=cen_b)
    p = {"excluded": excluded, "n_itt_imputed": sum(any(x[0] is None for x in pr) for pr in raw), "n_error_a": 0,
         "n_error_b": 0, "n_unpaired_by_design": len(sa.scheduled[sys_a] ^ sa.scheduled[sys_b])}
    row = _row(K_PAGES, "H5", row, split, sa.run_ids, min_n, f"{sys_a} vs {sys_b}", p,
               note="em_a / em_b are mean pages read before the first gold page (A2, A1); lower is better; censored items "
                    "take max(len(paged_in), largest observed); reported, no verdict")
    row["test"] = "paired bootstrap 95% CI (10,000 resamples) + Wilcoxon signed-rank; reported, no verdict, not Holm-adjusted"
    return row, excluded


def ni_met(ci_lo: float, margin: float) -> bool:
    """Non-inferiority (require_ci_lower_above): the CI lower bound must be STRICTLY above the margin."""
    return bool(ci_lo > margin)


def h8_row(store, sa: Series, split: str, min_n: Mapping[str, int], M: int = H8_M, rules: dict | None = None) -> tuple[dict, list]:
    """Descent beam 4 vs flat recall at M slots over the items of the h5-mt containers (each container with its own
    precomputed vectors and skeleton directory); per-item recalls are merged and the contrast is computed once."""
    per: dict[str, dict] = {}
    excluded: list[dict] = []
    rids = list(sa.run_ids)
    for rid, job in sa.jobs.items():
        args = job["args"]
        emb_uri = str(args.get("embedder", ""))
        sk_uri = args.get("skeleton_llm_uri")
        if not emb_uri.startswith("precomputed:") or not sk_uri:
            raise pl.MissingInput(f"{rid}: job has no precomputed embedder / skeleton_llm_uri")
        emb_uri = emb_uri[len("precomputed:"):].replace("{bucket}", store.bucket)
        sk_uri = sk_uri.replace("{bucket}", store.bucket).rstrip("/")
        rids += re.findall(r"/runs/([^/]+)/", emb_uri + "/") + re.findall(r"/runs/([^/]+)/", sk_uri + "/")
        try:
            npz = store.get(emb_uri)
        except KeyError as e:
            raise pl.MissingInput(f"embeddings {emb_uri} not found") from e
        items = pl.load_run_items(store, job)
        skeletons = {}
        for iid in items:
            try:
                skeletons[iid] = json.loads(store.get(f"{sk_uri}/{iid}.json"))
            except KeyError:
                excluded.append({"item_id": iid, "reason": "no skeleton"})
        use = [items[i] for i in items if i in skeletons]
        if not use:
            continue
        rows = rr.routing_recall_rows(use, npz, skeletons, M=M, split=split, n_boot=1, opts=args.get("opts"))
        prim = rows[0]
        excluded += [dict(e) for e in prim["excluded"]]
        for iid, v in prim["per_item"].items():
            if iid in per:
                raise ValueError(f"H8: item {iid} routed in more than one container")
            per[iid] = v
    if not per:
        raise pl.MissingInput(f"{K_H8}: no item could be routed")
    ids = sorted(per)
    d = np.array([per[i]["descent"] for i in ids])
    f = np.array([per[i]["flat"] for i in ids])
    row = stats.contrast(d, f, K_H8, test="mcnemar", n_boot=N_BOOT, seed=BOOT_SEED, split=split)
    mn, short = _short(len(ids), min_n, K_H8)
    margin = _rule_leaf(rules, K_H8).get("require_ci_lower_above", -3.0)
    row.update(split=split, hypothesis="H8", systems="descent beam 4 vs flat scan", n_pairs=len(ids),
               n_excluded_missing=len(excluded), n_itt_imputed=0, n_error_a=0, n_error_b=0, n_unpaired_by_design=0,
               em_a=float(d.mean()), em_b=float(f.mean()), min_n=mn, n_short=short, boot_seed=BOOT_SEED, n_boot=N_BOOT,
               ni_margin=margin, ni_met=ni_met(row["ci_lo"], margin), p_raw=None, test=NI_TEST_LABEL, run_ids=list(dict.fromkeys(rids)),
               note=f"M={M}; em_a / em_b are recall (descent, flat), not EM; not an intention-to-treat case (no model run)")
    return row, excluded


def h10_row(pin: Series, nopin: Series, split: str, min_n: Mapping[str, int]) -> tuple[dict, list]:
    from ..tracks.track_a.thrash import h10_contrast
    for s_ in (pin, nopin):
        v = s_.versions["thrash"]["prompt_version"]
        if v != {pl.THRASH_PROMPT_VERSION}:
            raise ValueError(f"series {s_.name}: prompt_version {sorted(v)} (missing or wrong); only "
                             f"{pl.THRASH_PROMPT_VERSION} thrash runs enter P2")
    if set(pin.thrash_sched) != set(nopin.thrash_sched):
        d = sorted(set(pin.thrash_sched) ^ set(nopin.thrash_sched))
        raise ValueError(f"H10: pinned and unpinned runs were scheduled on different items / M ({len(d)} differ, first {d[:3]})")
    row = h10_contrast(nopin.flat, pin.flat, split=split, n_boot=N_BOOT, seed=BOOT_SEED,
                       run_ids=list(dict.fromkeys(nopin.run_ids + pin.run_ids)))
    excluded = [{"item_id": f"{e['item_id']}@M{e['M']}", "reason": e["reason"]} for e in row["excluded"]]
    have = {(r["item_id"], r["M"]) for r in pin.flat} | {(r["item_id"], r["M"]) for r in nopin.flat}
    for (iid, M), r in sorted(pin.thrash_sched.items()):
        if (iid, M) not in have and (r is None or int(r) > M):
            excluded.append({"item_id": f"{iid}@M{M}", "reason": "missing both"})
    # pairs (not records) that carry an imputed value
    bad = ("error", "budget", "does_not_fit")
    a_by = {(r["item_id"], r["M"]): r for r in nopin.flat}
    b_by = {(r["item_id"], r["M"]): r for r in pin.flat}
    out_keys = {(e["item_id"], e["M"]) for e in row["excluded"]}
    n_imp = sum(1 for k in set(a_by) & set(b_by) if k not in out_keys and int(a_by[k]["r"]) > k[1]
                and (a_by[k].get("status") in bad or b_by[k].get("status") in bad))
    mn, short = _short(row["n"], min_n, K_H10)
    row.update(hypothesis="H10", systems="unpinned minus pinned", n_pairs=row["n"], n_excluded_missing=len(excluded),
               n_itt_imputed=n_imp, n_error_a=row["n_error_unpinned"], n_error_b=row["n_error_pinned"],
               n_budget_a=row["n_budget_unpinned"], n_budget_b=row["n_budget_pinned"],
               n_not_fit_a=row["n_not_fit_unpinned"], n_not_fit_b=row["n_not_fit_pinned"], n_unpaired_by_design=0,
               em_a=row["unpinned_mean"], em_b=row["pinned_mean"], min_n=mn, n_short=short, boot_seed=BOOT_SEED,
               n_boot=N_BOOT, note="em_a / em_b are mean repeated page-ins (unpinned, pinned); failures imputed with the "
                                   "worst value of the (r, M) cell")
    row["imputed_worst"] = ";".join(f"{k}={v}" for k, v in row["imputed_worst"].items())
    return row, excluded


# ====================================================================== driver
def _series_row(s: Series, split: str) -> dict:
    systems = sorted(s.records) or (["thrash"] if s.flat else [])
    counts = ";".join(f"{x}={len(s.records[x])}" for x in sorted(s.records)) or (f"records={len(s.flat)}" if s.flat else "")
    vers = lambda k: ";".join(f"{x}={'|'.join(sorted(v[k]))}" for x, v in sorted(s.versions.items()) if v[k])
    checks = sorted(set(s.em_check.values()))
    return {"key": s.name, "split": split, "n_runs": len(s.run_ids), "run_ids": list(s.run_ids),
            "n_items": len(s.items), "systems": ";".join(systems), "n_records": counts, "config_version": vers("config_version"),
            "prompt_version": vers("prompt_version"), "git_sha": ";".join(sorted(s.git_shas)), "em_check": ";".join(checks)}


def run_p2(store, series_runs: Mapping[str, Sequence[str]], split: str, out_dir: Path | str, *,
           runs_dir: Path | str | None = None, min_n: Any = None, rules: dict | None = None, M: int = H8_M,
           tag_file: Path | str | None = None) -> dict:
    """Load, check, compute and write. Returns {"rows", "skipped", "short", "written", "exclusions"}."""
    runs_dir = Path(runs_dir) if runs_dir is not None else PROJECT_DIR / "runs"
    names = [f"p2_contrasts_{split}.csv", f"p2_series_{split}.csv", f"p2_exclusions_{split}.csv"]
    check_table_names(out_dir, names, split)
    require_prereg_tag(split, tag_file)
    unknown = sorted(set(series_runs) - set(SERIES_ORDER) - set(DEFERRED_SERIES))
    if unknown:
        raise ValueError(f"unknown series {unknown}; known: {SERIES_ORDER}")
    series_runs = {k: list(v) for k, v in series_runs.items() if k in SERIES_ORDER and v}
    mn = load_min_n(min_n)
    rules = rules if rules is not None else load_rules()
    check_statuses(store, runs_dir, series_runs)
    ser = {k: load_series(store, k, v, split, runs_dir) for k, v in series_runs.items()}
    rows, skipped, excl = [], [], []

    def add(row: dict, excluded: Sequence[dict]):
        rows.append(row)
        for e in excluded:
            excl.append({"key": f"{row['key']}|{e['item_id']}", "contrast": row["key"], "item_id": e["item_id"],
                         "reason": e["reason"], "split": split})

    for key, na, sya, nb, syb, hyp in CONTRASTS:
        if na not in ser or nb not in ser:
            skipped.append(f"{key}: series {na if na not in ser else nb} not given")
            continue
        try:
            r = pair_contrast(key, hyp, ser[na], sya, ser[nb], syb, split, mn)
            add(r, paired_arrays(ser[na], sya, ser[nb], syb)["excluded"])
        except pl.MissingInput as e:
            skipped.append(f"{key}: {e}")
    if "h5-mt" in ser:
        try:
            r, ex = pages_row(ser["h5-mt"], split, mn)
            add(r, ex)
        except pl.MissingInput as e:
            skipped.append(f"{K_PAGES}: {e}")
        try:
            r, ex = h8_row(store, ser["h5-mt"], split, mn, M, rules)
            add(r, ex)
        except pl.MissingInput as e:
            skipped.append(f"{K_H8}: {e}")
    else:
        skipped.append(f"{K_H8}: series h5-mt not given")
    if "h10-pin" in ser and "h10-nopin" in ser:
        r, ex = h10_row(ser["h10-pin"], ser["h10-nopin"], split, mn)
        add(r, ex)
    else:
        skipped.append(f"{K_H10}: series h10-pin / h10-nopin not both given")
    order = {k: i for i, k in enumerate([c[0] for c in CONTRASTS] + [K_PAGES, K_H8, K_H10])}
    rows.sort(key=lambda r: order[r["key"]])
    excl.sort(key=lambda e: (order[e["contrast"]], e["item_id"]))
    ci_only = set(_ci_only_keys(rules)) | {K_H8}
    out = Path(out_dir)
    write_contrast_table(out / names[0], rows, extra_columns=EXTRA_COLUMNS, ci_only_keys=ci_only)
    write_table(out / names[1], [_series_row(ser[k], split) for k in SERIES_ORDER if k in ser],
                columns=SERIES_COLUMNS)
    write_table(out / names[2], excl, columns=EXCL_COLUMNS)
    short = [f"{r['key']}: n={r['n']} < min_n={r['min_n']}" for r in rows if r.get("n_short")]
    return {"rows": rows, "skipped": skipped, "short": short, "written": [str(out / n) for n in names], "exclusions": excl}


def _ci_only_keys(rules: dict | None) -> list[str]:
    if not rules:
        return []
    from ..claims import non_inferiority_keys
    return non_inferiority_keys(rules)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--runs", required=True, help="YAML/JSON mapping series -> [run ids]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", default=None, help="YAML/JSON with the pre-registered minimum n (prereg/sample_sizes.yaml)")
    ap.add_argument("--M", type=int, default=H8_M, help="H8 candidate slots")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--creds", default=None)
    a = ap.parse_args(argv)
    creds = a.creds or pl.credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")        # never printed
    try:
        res = run_p2(pl.GcsStore(), load_mapping(a.runs), a.split, a.out, runs_dir=a.runs_dir, min_n=a.min_n, M=a.M)
    except (ValueError, PermissionError, pl.MissingInput) as e:
        print(f"p2_results FAILED: {e}", file=sys.stderr)
        return 1
    for r in res["rows"]:
        print(f"{r['key']}: value={r['value']:.4g} n={r['n']} excluded={r.get('n_excluded_missing', 0)} itt={r.get('n_itt_imputed', 0)}")
    for s in res["skipped"]:
        print("skipped:", s)
    for s in res["short"]:
        print("SHORT:", s)
    print("wrote:", *res["written"])
    return 0 if res["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
