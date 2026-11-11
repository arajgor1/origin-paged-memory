"""P1.4 pilot analysis on the DEV split (free, local): paired discordance, SDs of paired differences, the H3 formula
fit, run-to-run noise and the proposed sample sizes. Nothing here spends money; it reads finished runs from GCS
(records.jsonl.gz, embeddings, skeletons) and the local runs/<id>/job.yaml, and writes tables under
bundle/tables/pilot_dev/. It never writes prereg/sample_sizes.yaml: the proposed table is for the operator.

  python -m opa.analysis.pilot --out bundle/tables/pilot_dev

Sections (each is skipped with a clear message when its inputs are missing; the rest still run):
  discordance   n10, n01, n, p_d and the paired contrast for S11 vs S2 / S4 / S8 (late h2 main run), S12 vs S14 (A2 vs
                A1) and S12 vs S15 (A2 vs A3) on the MT T4 run, and H8 descent beam 4 vs flat recall (routing_recall)
  sd_paired     SD of the paired differences for H2 (drift_contrast) and H10 (thrash.h10_contrast)
  h3_fit        identity-line binned R2, fitted-line R2, logistic regression and phi for S11 (late + forward + mixed h2)
  noise         per-item flip rate and EM spread across the three late-h2 repeats for S1, S2, S8
  sample_sizes  opa.stats.n_mcnemar / n_tost and the paired-normal formula of prereg-v2.DRAFT.md Section 7

Every row has split "dev" and its run ids; oracle systems (S9, S18) are flagged and never enter a pair here.
GCP credentials: with GCP_SA_JSON unset, the CLI reads the service-account file given by --creds into the
environment of this process only; the content is never printed or written.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .. import PROJECT_DIR, scoring, stats
from . import drift_contrast as dcm
from . import routing_recall as rr
from .tables import write_contrast_table, write_table

SPLIT = "dev"
ORACLE_SYSTEMS = {"S9", "S18"}


def credentials_path(project_dir: Path | str | None = None) -> str | None:
    """Service-account key file path. Looked up in config/project.yaml (gcp.credentials_file) and, where the project
    records it, credentials.md ("Key file (primary)": `<path>`). Only the path is read, never the key content."""
    root = Path(project_dir) if project_dir is not None else PROJECT_DIR
    try:
        cfg = yaml.safe_load((root / "config" / "project.yaml").read_text(encoding="utf-8")) or {}
        path = (cfg.get("gcp") or {}).get("credentials_file")
        if path:
            return str(path)
    except Exception:
        pass
    try:
        m = re.search(r"Key file \(primary\)\*{0,2}:\s*`([^`]+)`", (root / "credentials.md").read_text(encoding="utf-8"))
        return m.group(1) if m else None
    except Exception:
        return None

# Pilot runs (brief of 2026-11-07). Short names are used in the tables.
RUNS = {
    "lb_main": "A-S1+S2+S4+S8+S11+S18-lbmhqa-late-h2-32k-a0-dev-s0-20261107154535-a0c95f6-4168",
    "lb_rep2": "A-S1+S2+S8-lbmhqa-late-h2-32k-a0-dev-s0-20261107154537-a0c95f6-a364",
    "lb_rep3": "A-S1+S2+S8-lbmhqa-late-h2-32k-a0-dev-s0-20261107154540-a0c95f6-8f07",
    "lb_fwd_mixed": "A-S11-lbmhqa-forward-h2-32k-a0+mixed-h2-32k-a0-dev-s0-20261107163640-1aaa09a-3a1a",
    "mt_arms": "A-S12+S14+S15+S16+S18-mtlbmhqa-T4-h3-32k-cross-dev-s0-20261107163638-1aaa09a-db29",
    "drift_d2": "A2-D2-drift-s50-dev-s0-20261107140753-99677c4-9925",
    "drift_d4": "A2-D4-drift-s50-dev-s0-20261107140755-99677c4-9cab",
    "thrash_pin": "A-thrash-pin-thrash-r3+r4+r6-dev-s0-20261107151336-a9bc456-d940",
    "thrash_nopin": "A-thrash-nopin-thrash-r3+r4+r6-dev-s0-20261107151338-a9bc456-c13b",
}

K_S11_S2 = "S11_vs_S2_lbmhqa_late_32k_primary"
K_S11_S4 = "S11_vs_S4_lbmhqa_late_32k_primary"
K_S11_S8 = "S11_vs_S8_late_bound"
K_A2_A1 = "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross"
K_A2_A3 = "A2_vs_A3_agent_equivalence"
K_H8 = rr.key_for(rr.PRIMARY_BEAM)
K_H2 = dcm.KEY
K_H10 = "pinned_vs_unpinned_repeat_pagein_reduction"
K_R2 = "R2_predicted_vs_observed_success"
H8_M = 4                                            # M slots (draft defaults)
PAIRS = [(K_S11_S2, "S11", "S2"), (K_S11_S4, "S11", "S4"), (K_S11_S8, "S11", "S8")]
MT_PAIRS = [(K_A2_A1, "S12", "S14"), (K_A2_A3, "S12", "S15")]


class MissingInput(Exception):
    """A pilot input is not (yet) available; the section is skipped."""


# ====================================================================== storage
class GcsStore:
    def __init__(self, bucket: str | None = None):
        from .. import gcs
        self.bucket = bucket or gcs.bucket_name()
        self._gcs = gcs

    def ls(self, prefix: str) -> list[str]:
        return self._gcs.ls(prefix)

    def get(self, uri: str) -> bytes:
        """Object bytes; a missing object raises KeyError (the one error every "not there yet" path catches)."""
        try:
            return self._gcs.get_bytes(uri)
        except Exception as e:
            if type(e).__name__ == "NotFound":            # google.api_core.exceptions.NotFound, without importing it here
                raise KeyError(uri) from e
            raise


class MemStore:
    """In-memory store for tests: {gs-uri: bytes}."""

    def __init__(self, files: Mapping[str, bytes], bucket: str = "b"):
        self.files, self.bucket = dict(files), bucket

    def ls(self, prefix: str) -> list[str]:
        return sorted(u for u in self.files if u.startswith(prefix))

    def get(self, uri: str) -> bytes:
        if uri not in self.files:
            raise KeyError(uri)
        return self.files[uri]


def gz_jsonl(rows: Iterable[dict]) -> bytes:
    return gzip.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode())


def read_jsonl(data: bytes, gz: bool = True) -> list[dict]:
    text = (gzip.decompress(data) if gz else data).decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_uri(store, run_id: str) -> str:
    return f"gs://{store.bucket}/opa/runs/{run_id}"


@dataclass
class Run:
    run_id: str
    job: dict
    items: dict = field(default_factory=dict)                 # item_id -> item
    records: dict = field(default_factory=dict)               # system -> {repeat: [records]}   (multi-system run)
    flat: list = field(default_factory=list)                  # records of a single-condition run (A2, thrash)
    metrics: dict = field(default_factory=dict)               # the run's metrics.json (prompt_version, tuning_iteration)
    sub_metrics: dict = field(default_factory=dict)           # (system, repeat) -> the sub-run's metrics.json (multi-system)


def load_job(run_id: str, runs_dir: Path | str | None = None) -> dict:
    p = Path(runs_dir or PROJECT_DIR / "runs") / run_id / "job.yaml"
    if not p.exists():
        raise MissingInput(f"{p} not found")
    job = yaml.safe_load(p.read_text(encoding="utf-8"))
    if job.get("split") != SPLIT:
        raise ValueError(f"{run_id}: split is {job.get('split')!r}; the pilot analysis reads dev runs only")
    return job


def load_run_items(store, job: dict) -> dict:
    uris = job["args"]["items_uri"]
    uris = uris if isinstance(uris, (list, tuple)) else [uris]
    out = {}
    for u in uris:
        off, n = int(job["args"].get("offset", 0)), job["args"].get("n_items")
        if isinstance(u, dict):
            off, n, u = int(u.get("offset", off)), u.get("n_items", n), u["uri"]
        u = u.replace("{bucket}", store.bucket)
        try:
            its = read_jsonl(store.get(u), gz=u.endswith(".gz"))[off:]
        except KeyError as e:
            raise MissingInput(f"items {u} not found") from e
        for it in its[: int(n)] if n is not None else its:
            out[it["item_id"]] = it
    return out


def _metrics(store, run_id: str) -> dict:
    try:
        return json.loads(store.get(run_uri(store, run_id) + "/metrics.json"))
    except (KeyError, ValueError):
        return {}


def _no_records_message(store, run_id: str, nested: bool) -> str:
    """"not finished" only if the run has no result.json; otherwise the records are somewhere else: layout mismatch."""
    listing = store.ls(run_uri(store, run_id) + "/")
    has_result = any(u.endswith("/result.json") for u in listing)
    if not has_result:
        return f"{run_id}: no records yet (run not finished)"
    found = [u.split(run_id)[1] for u in listing if u.endswith("records.jsonl.gz")]
    want = "<SYSTEM>/r<k>/records.jsonl.gz" if nested else "records.jsonl.gz"
    return (f"{run_id}: layout mismatch: result.json exists but no {want}"
            + (f"; found {found[:4]}" if found else "; no records.jsonl.gz anywhere under the run"))


def load_multi_run(store, run_id: str, runs_dir=None, with_items: bool = True) -> Run:
    """A multi-system run: <run>/<SYSTEM>/r<k>/records.jsonl.gz."""
    job = load_job(run_id, runs_dir)
    pat = re.compile(r"/([^/]+)/r(\d+)/records\.jsonl\.gz$")
    recs: dict = defaultdict(dict)
    sub: dict = {}
    for u in store.ls(run_uri(store, run_id) + "/"):
        m = pat.search(u)
        if m:
            recs[m.group(1)][int(m.group(2))] = read_jsonl(store.get(u))
            try:
                sub[(m.group(1), int(m.group(2)))] = json.loads(store.get(u.replace("records.jsonl.gz", "metrics.json")))
            except (KeyError, ValueError):
                pass
    if not recs:
        raise MissingInput(_no_records_message(store, run_id, nested=True))
    return Run(run_id, job, load_run_items(store, job) if with_items else {}, dict(recs), metrics=_metrics(store, run_id),
               sub_metrics=sub)


def load_single_run(store, run_id: str, runs_dir=None, with_items: bool = True) -> Run:
    """A single-condition run: <run>/records.jsonl.gz (Track A2 drift, thrash)."""
    job = load_job(run_id, runs_dir)
    u = run_uri(store, run_id) + "/records.jsonl.gz"
    try:
        flat = read_jsonl(store.get(u))
    except KeyError as e:
        raise MissingInput(_no_records_message(store, run_id, nested=False)) from e
    return Run(run_id, job, load_run_items(store, job) if with_items else {}, {}, flat, _metrics(store, run_id))


# ====================================================================== scoring helpers
def em_by_item(items: Mapping[str, dict], records: Sequence[dict], label: str = "") -> tuple[dict, int, set]:
    """({item_id: EM 0/1}, n_error, unknown_ids) for the records of one system. A record whose status is not "ok" (a
    missing status counts as an error) scores 0 (intention to treat) and is counted. Records of items outside
    ``items`` are returned in unknown_ids, not scored. A second record for the same item raises ValueError."""
    out, n_err, unknown, seen = {}, 0, set(), set()
    for r in records:
        iid = r["item_id"]
        if iid in seen:
            raise ValueError(f"duplicate record for item {iid}" + (f" in {label}" if label else ""))
        seen.add(iid)
        it = items.get(iid)
        if it is None:
            unknown.add(iid)
            continue
        ok = r.get("status") == "ok"
        n_err += not ok
        out[iid] = harness_em(it, r) if ok else 0.0
    return out, n_err, unknown


TOL = 1e-9


def harness_em(item: dict, record: dict) -> float:
    """Per-item EM exactly as the Track A harness computes it (scoring.score_item, which scoring.aggregate and
    run.summarize average into metrics.json)."""
    return float(scoring.score_item(item, record)["em"])


def check_em_consistency(run: "Run", system: str, repeat: int = 0, tol: float = TOL) -> float:
    """Hard check: the mean per-item EM of a sub-run's records (harness scorer, every record, no ITT zeroing) equals
    that sub-run's metrics.json em. Raises ValueError on a mismatch (the pilot would be reading a different
    quantity than the harness reported) and MissingInput if the sub-run has no metrics.json to check against."""
    recs = sys_records(run, system, repeat)
    m = run.sub_metrics.get((system, repeat))
    if not m or "em" not in m:
        raise MissingInput(f"{run.run_id}: no metrics.json for {system} r{repeat}; cannot verify per-item EM")
    missing = [r["item_id"] for r in recs if r["item_id"] not in run.items]
    if missing:
        raise ValueError(f"{run.run_id} {system} r{repeat}: records of items not in the item list: {missing[:3]}")
    mean = float(np.mean([harness_em(run.items[r["item_id"]], r) for r in recs]))
    if abs(mean - float(m["em"])) > tol:
        raise ValueError(f"{run.run_id} {system} r{repeat}: mean per-item EM {mean!r} != metrics.json em {m['em']!r}")
    return mean


def sys_records(run: Run, system: str, repeat: int = 0) -> list[dict]:
    try:
        return run.records[system][repeat]
    except KeyError as e:
        raise MissingInput(f"{run.run_id}: no records for {system} r{repeat}") from e


def pair_row(key: str, run: Run, sys_a: str, sys_b: str, verify_em: bool = True) -> dict:
    """Contrast row (a minus b, EM points) with discordance counts and p_d for two systems of one run. Items that
    cannot be paired are listed in ``excluded`` ({"item_id", "reason"}: "missing <system>", "missing both",
    "not in item list") and counted in ``n_excluded``."""
    if ORACLE_SYSTEMS & {sys_a, sys_b}:
        raise ValueError(f"{key}: oracle systems never enter a pilot pair")
    if verify_em:
        check_em_consistency(run, sys_a)
        check_em_consistency(run, sys_b)
    ea, err_a, unk_a = em_by_item(run.items, sys_records(run, sys_a), f"{run.run_id} {sys_a}")
    eb, err_b, unk_b = em_by_item(run.items, sys_records(run, sys_b), f"{run.run_id} {sys_b}")
    ids = sorted(set(ea) & set(eb))
    if not ids:
        raise MissingInput(f"{key}: no common items")
    excluded = []
    for i in sorted((set(run.items) | unk_a | unk_b) - set(ids)):
        if i in unk_a or i in unk_b:
            why = "not in item list"
        elif i not in ea and i not in eb:
            why = "missing both"
        else:
            why = f"missing {sys_a}" if i not in ea else f"missing {sys_b}"
        excluded.append({"item_id": i, "reason": why})
    a, b = np.array([ea[i] for i in ids]), np.array([eb[i] for i in ids])
    row = stats.contrast(a, b, key, test="mcnemar", scale=100.0, run_ids=[run.run_id], split=SPLIT)
    row.update(p_d=row["discordant_rate"], systems=f"{sys_a} vs {sys_b}", oracle=False,
               n_error_a=err_a, n_error_b=err_b, excluded=excluded, n_excluded=len(excluded),
               em_a=float(a.mean()), em_b=float(b.mean()), note="")
    return row


def h8_row(store, mt: Run, runs_dir=None, M: int = H8_M) -> dict:
    """H8 descent beam 4 vs flat recall on the MT items, S12 session cfg, MT skeletons, precomputed vectors."""
    args = mt.job["args"]
    emb_uri = str(args.get("embedder", ""))
    sk_uri = args.get("skeleton_llm_uri")
    if not emb_uri.startswith("precomputed:") or not sk_uri:
        raise MissingInput(f"{mt.run_id}: job has no precomputed embedder / skeleton_llm_uri")
    emb_uri = emb_uri[len("precomputed:"):].replace("{bucket}", store.bucket)
    sk_uri = sk_uri.replace("{bucket}", store.bucket).rstrip("/")
    try:
        npz = store.get(emb_uri)
    except KeyError as e:
        raise MissingInput(f"embeddings {emb_uri} not found") from e
    skeletons = {}
    for iid in mt.items:
        try:
            skeletons[iid] = json.loads(store.get(f"{sk_uri}/{iid}.json"))
        except KeyError as e:
            raise MissingInput(f"skeleton {sk_uri}/{iid}.json not found") from e
    rids = [mt.run_id] + re.findall(r"/runs/([^/]+)/", emb_uri + "/") + re.findall(r"/runs/([^/]+)/", sk_uri + "/")
    rows = rr.routing_recall_rows(list(mt.items.values()), npz, skeletons, M=M, split=SPLIT, run_ids=rids,
                                  opts=args.get("opts"))
    row = rows[0]
    row.update(p_d=row["discordant_rate"], systems="descent beam 4 vs flat scan", oracle=False, note="")
    return row


# ====================================================================== sections
def discordance_rows(main: Run | None, mt: Run | None, store=None, runs_dir=None) -> tuple[list[dict], list[str]]:
    rows, skipped = [], []
    if main is None:
        skipped.append("discordance S11 vs S2/S4/S8: late-h2 main run has no records yet")
    else:
        for key, a, b in PAIRS:
            try:
                rows.append(pair_row(key, main, a, b))
            except MissingInput as e:
                skipped.append(f"discordance {key}: {e}")
    if mt is None:
        skipped.append("discordance A2 vs A1/A3 and H8: MT T4 arms run has no records yet")
    else:
        for key, a, b in MT_PAIRS:
            try:
                rows.append(pair_row(key, mt, a, b))
            except MissingInput as e:
                skipped.append(f"discordance {key}: {e}")
        if store is not None:
            try:
                rows.append(h8_row(store, mt, runs_dir))
            except MissingInput as e:
                skipped.append(f"discordance {K_H8}: {e}")
    return rows, skipped


THRASH_PROMPT_VERSION = "thrash-v2"
THRASH_SCORED = ("ok", "step_cap")                  # statuses opa.tracks.track_a.thrash averages into metrics.json


def check_drift_consistency(run: Run, turn: int = 50, tol: float = TOL) -> float:
    """Mean per-session retention at ``turn`` (what the D4-vs-D2 contrast reads) == the run's metrics.json
    accuracy_t50. ValueError on mismatch, MissingInput without metrics."""
    m = run.metrics.get("accuracy_t50")
    if m is None:
        raise MissingInput(f"{run.run_id}: no metrics.json accuracy_t50; cannot verify drift retention")
    mean = float(np.mean([dcm.session_retention(run.items[r["item_id"]], r, turn) for r in run.flat]))
    if abs(mean - float(m)) > tol:
        raise ValueError(f"{run.run_id}: mean per-session retention {mean!r} != metrics.json accuracy_t50 {m!r}")
    return mean


def check_thrash_consistency(run: Run, tol: float = TOL) -> float:
    """Mean repeated page-ins over the scored records (status ok / step_cap) == metrics.json repeated_page_ins_mean."""
    m = run.metrics.get("repeated_page_ins_mean")
    if m is None:
        raise MissingInput(f"{run.run_id}: no metrics.json repeated_page_ins_mean; cannot verify thrash outcomes")
    sc = [r["repeated_page_ins"] for r in run.flat if r.get("status") in THRASH_SCORED]
    mean = float(np.mean(sc)) if sc else float("nan")
    if not abs(mean - float(m)) <= tol:
        raise ValueError(f"{run.run_id}: mean repeated page-ins {mean!r} != metrics.json repeated_page_ins_mean {m!r}")
    return mean


def sd_rows(d2: Run, d4: Run, pin: Run, nopin: Run) -> list[dict]:
    """H2 (per-session retention at turn 50, points) and H10 (repeated page-ins per item) with sd_diff, the
    intention-to-treat count columns, the prompt version and the tuning iteration of the runs behind them."""
    from ..tracks.track_a.thrash import h10_contrast
    for r in (pin, nopin):
        v = r.metrics.get("prompt_version")
        if v != THRASH_PROMPT_VERSION:
            raise ValueError(f"{r.run_id}: prompt_version {v!r} (missing or wrong); only {THRASH_PROMPT_VERSION} thrash "
                             f"runs enter the pilot")
    for r in (d2, d4):
        check_drift_consistency(r)
    for r in (pin, nopin):
        check_thrash_consistency(r)
    sessions = list(d2.items.values())
    h2 = dcm.drift_retention_contrast(sessions, d2.flat, d4.flat, split=SPLIT, run_ids=[d2.run_id, d4.run_id])
    h10 = h10_contrast(nopin.flat, pin.flat, split=SPLIT, run_ids=[nopin.run_id, pin.run_id])
    out = []
    for r, runs, unit in ((h2, (d2, d4), "points (retention x 100)"), (h10, (nopin, pin), "repeated page-ins per item")):
        r = dict(r)
        r["oracle"] = False
        r["unit"] = unit
        r["prompt_version"] = ";".join(sorted({str(x.metrics.get("prompt_version", "")) for x in runs}))
        r["tuning_iteration"] = ";".join(str(x.metrics.get("tuning_iteration", "")) for x in runs)
        if "imputed_worst" in r:
            r["imputed_worst"] = ";".join(f"{k}={v}" for k, v in r["imputed_worst"].items())
        out.append(r)
    return out


def _h3_inputs(items: Mapping[str, dict], records: Sequence[dict]) -> list[dict]:
    rows = []
    for r in records:
        it = items.get(r["item_id"])
        if it is None:
            continue
        cf = scoring.coverage_factors(it, r)
        ok = (r.get("status") or "ok") == "ok"
        em = harness_em(it, r) if ok else 0.0
        rows.append({"item_id": r["item_id"], "predicted": cf["predicted"], "observed": em, "h_late": cf["h_late"],
                     "gamma_s": cf["gamma_s"], "gamma_g": cf["gamma_g"], "gamma_eff": cf["gamma_eff"],
                     "skeleton_miss": cf["skeleton_miss"], "graph_miss": cf["graph_miss"]})
    return rows


def h3_fit(per_item: Sequence[dict], run_ids: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """(summary rows, bin rows). Identity-line binned R2 (primary definition), fitted-line R2, logistic regression of
    EM on h_late / gamma_s / gamma_g, and phi between skeleton and graph misses over late-bound hops."""
    pred = np.array([r["predicted"] for r in per_item])
    obs = np.array([r["observed"] for r in per_item])
    b = stats.r2_binned(pred, obs, bins=10, ids=[r["item_id"] for r in per_item])
    base = {"split": SPLIT, "run_ids": list(run_ids), "oracle": False}
    rows = [{"key": K_R2, "value": b["r2"], "n": b["n"], "detail": "identity-line R2, 10 equal-count bins, ties broken by item id", **base},
            {"key": "R2_fitted_line", "value": b["r2_linear"], "n": b["n"], "detail": "squared Pearson r of bin means", **base},
            {"key": "mean_predicted", "value": float(pred.mean()), "n": b["n"], "detail": "", **base},
            {"key": "mean_observed", "value": float(obs.mean()), "n": b["n"], "detail": "", **base}]
    lateful = [r for r in per_item if r["h_late"] > 0 and not (math.isnan(r["gamma_s"]) or math.isnan(r["gamma_g"]))]
    if len(lateful) > 5 and 0 < sum(r["observed"] for r in lateful) < len(lateful):
        X = [[r["h_late"], r["gamma_s"], r["gamma_g"]] for r in lateful]
        lg = stats.logistic_regression(X, [int(r["observed"]) for r in lateful], names=["h_late", "gamma_s", "gamma_g"])
        for nm, c, se, p in zip(lg["names"], lg["coef"], lg["se"], lg["p"]):
            rows.append({"key": f"logistic_{nm}", "value": c, "n": lg["n"],
                         "detail": f"se={se:.4g} p={p:.4g} converged={lg['converged']}", **base})
    s_miss = [m for r in lateful for m in r["skeleton_miss"]]
    g_miss = [m for r in lateful for m in r["graph_miss"]]
    if s_miss:
        rows.append({"key": "phi_skeleton_miss_graph_miss", "value": stats.phi_coefficient(s_miss, g_miss),
                     "n": len(s_miss), "detail": "per late-bound hop", **base})
    bins = [{"key": f"bin{k:02d}", **{kk: v for kk, v in bn.items()}, "split": SPLIT, "run_ids": list(run_ids)}
            for k, bn in enumerate(b["bins"], 1)]
    return rows, bins


def noise_rows(runs: Sequence[Run], systems: Sequence[str] = ("S1", "S2", "S8")) -> list[dict]:
    """Per system over the repeats (one run each): per-item flip rate (EM not identical in all repeats) and the
    spread of the per-repeat mean EM."""
    rows = []
    for s in systems:
        per = []
        for run in runs:
            check_em_consistency(run, s)
            e, _, _ = em_by_item(run.items, sys_records(run, s), f"{run.run_id} {s}")
            per.append(e)
        ids = sorted(set.intersection(*[set(e) for e in per]))
        if not ids:
            raise MissingInput(f"noise {s}: no common items across repeats")
        M = np.array([[e[i] for i in ids] for e in per])
        means = M.mean(axis=1)
        rows.append({"key": f"noise_{s}", "system": s, "oracle": s in ORACLE_SYSTEMS, "n_items": len(ids),
                     "n_repeats": len(per), "flip_rate": float((M.min(axis=0) != M.max(axis=0)).mean()),
                     "em_mean": float(means.mean()), "em_sd_across_repeats": float(means.std(ddof=1)) if len(per) > 1 else math.nan,
                     "em_min": float(means.min()), "em_max": float(means.max()),
                     "em_by_repeat": ";".join(f"{m:.3f}" for m in means), "split": SPLIT,
                     "run_ids": [r.run_id for r in runs]})
    return rows


# ====================================================================== Track B / remedies pilots (information and p_d)
K_V3_B1 = "V3_vs_B1_source_attribution"
K_R5_R1 = "R5_vs_R1_accuracy"
ATTRIB_IDENTITY_FIELDS = ("window", "target_origin", "options_origin_ids", "gold_option", "T")
TRACKB_RUNS = {"V3": "B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261107120809-37cd97c-06c8",
               "B1": "B-B1-mtlbmhqa-archive-archive64-T8-dev-s0-20261107123712-c83bf32-a14d"}
REMEDIES_RUN = "B-R0-R7-mtlbmhqa-archive-T4-h2-8k-nocross-dev-s0-20261107124615-ac57b55-5c1e"


def _get_json(store, run_id: str, rel: str):
    try:
        return json.loads(store.get(run_uri(store, run_id) + "/" + rel))
    except KeyError as e:
        raise MissingInput(f"{run_id}/{rel} not found") from e


def trackb_attribution_row(store, runs: Mapping[str, str] = TRACKB_RUNS) -> dict:
    """V3_vs_B1_source_attribution from the mini-LoRA pilot: final_eval.json (tag "final") per-item attribution EM,
    paired by the stored item identity (meta.attrib[i].seed), via stats.contrast (McNemar)."""
    per = {}
    for arm, rid in runs.items():
        d = _get_json(store, rid, "final_eval.json")
        if d.get("final", {}).get("tag") != "final":
            raise MissingInput(f"{rid}: final_eval.json has no 'final' evaluation (status {d.get('status')!r})")
        blk = d["per_item"]["final"]
        em, ident = blk["per_item"]["attrib_em"], [str(m["seed"]) for m in blk["meta"]["attrib"]]
        if len(em) != len(ident) or len(set(ident)) != len(ident):
            raise ValueError(f"{rid}: attribution items and identities disagree or repeat")
        meta = {str(m["seed"]): {k: m.get(k) for k in ATTRIB_IDENTITY_FIELDS} for m in blk["meta"]["attrib"]}
        per[arm] = (dict(zip(ident, em)), d["final"].get("opt_step"), meta)
    a, b = per["V3"][0], per["B1"][0]
    for seed in sorted(set(a) & set(b)):
        if per["V3"][2][seed] != per["B1"][2][seed]:
            raise ValueError(f"attribution item {seed} differs between the V3 and B1 runs "
                             f"({per['V3'][2][seed]} vs {per['B1'][2][seed]}): not the same item")
    ids = sorted(set(a) & set(b))
    if not ids:
        raise MissingInput(f"{K_V3_B1}: no common attribution items")
    row = stats.contrast(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]), K_V3_B1, test="mcnemar",
                         scale=100.0, run_ids=list(runs.values()), split=SPLIT)
    row.update(p_d=row["discordant_rate"], systems="V3 vs B1 (mini-LoRA)", oracle=False,
               n_excluded=len(set(a) ^ set(b)), note=f"mini-LoRA pilot, {per['V3'][1]} steps, n={len(ids)}")
    return row


def remedies_info_row(store, run_id: str = REMEDIES_RUN) -> dict:
    """R5_8 vs R1 EM discordance from the remedies pilot (items/<k>.json): information only, the R5_vs_R1 min_n
    stays fixed at the protocol's 400."""
    ids_uri = run_uri(store, run_id) + "/items/"
    files = [u for u in store.ls(ids_uri) if u.endswith(".json")]
    if not files:
        raise MissingInput(f"{run_id}: no items/<k>.json")
    r5, r1 = [], []
    for u in sorted(files):
        res = json.loads(store.get(u))["result"]
        r5.append(res["R5_8"]["em"])
        r1.append(res["R1"]["em"])
    row = stats.contrast(np.array(r5), np.array(r1), K_R5_R1 + "_remedies_pilot_info", test="mcnemar", scale=100.0,
                         run_ids=[run_id], split=SPLIT)
    row.update(p_d=row["discordant_rate"], systems="R5_8 vs R1 (8K, T4, h2, no cross)", oracle=False, n_excluded=0,
               note="information only; min_n for R5_vs_R1_accuracy stays the fixed 400")
    return row


# ====================================================================== sample sizes
Z = NormalDist().inv_cdf
ARE = 0.955


def n_paired_normal(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8, are: float = ARE) -> int:
    """Paired normal formula for a Wilcoxon test: ceil(((z_{1-a/2} + z_power) sd / effect)^2 / ARE)."""
    return int(math.ceil(((Z(1 - alpha / 2) + Z(power)) * sd / effect) ** 2 / are))


# (key, method, delta/margin/effect, floor, p_d key in the pilot table / sd key, note)
SS_SPECS = [
    (K_S11_S2, "mcnemar", 0.05, 0, K_S11_S2, "superiority, kill threshold 5 points"),
    (K_S11_S4, "mcnemar", 0.03, 0, K_S11_S4, "superiority, 3 points"),
    (K_S11_S8, "mcnemar", 0.02, 0, K_S11_S8, "superiority, 2 points"),
    (K_A2_A1 + "|delta=3", "mcnemar", 0.03, 0, K_A2_A1, "proposal delta 3.0 (TBD-P1.6)"),
    (K_A2_A1 + "|delta=5", "mcnemar", 0.05, 0, K_A2_A1, "proposal delta 5.0 (TBD-P1.6)"),
    (K_A2_A3, "tost", 0.03, 0, K_A2_A3, "equivalence margin 3 points; p_d of S12 vs S15"),
    (K_H8, "mcnemar", 0.03, 0, K_H8, "non-inferiority margin 3 points, true difference 0 (approximation)"),
    (K_H2, "paired_normal", 10.0, 200, K_H2, "effect 10 points; floor 200 sessions; Wilcoxon ARE 0.955"),
    (K_H10, "paired_normal", 0.5, 100, K_H10, "effect 0.5 page-ins (proposal TBD-P1.6); floor 100 items"),
    (K_V3_B1, "mcnemar", 0.05, 0, K_V3_B1, "Track B; proposal delta 5.0; p_d from the mini-LoRA pilot if present else 0.30"),
    ("A2_vs_A1_model_thread_confusion_task", "mcnemar", 0.05, 0, None, "Track B; proposal delta 5.0; p_d default 0.30"),
    ("V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k", "mcnemar", 0.05, 0, None, "Track B; proposal 5.0; p_d default 0.30"),
    ("V3_paged_vs_B1_full_accuracy_lbmhqa_32k", "mcnemar", 0.02, 0, None, "NI margin 2 points; p_d default 0.30"),
    ("copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k", "mcnemar", 0.02, 0, None, "NI margin 2 points; p_d default 0.30"),
    (K_R5_R1, "fixed", 400, 0, None, "protocol 5.2 data size 400 (remedies-pilot discordance is information only)"),
]
DEFAULT_PD = 0.30


def sample_size_rows(disc: Sequence[dict], sds: Sequence[dict]) -> list[dict]:
    """Proposed min_n per key (never written to prereg/): p_d and SDs from the pilot rows, 0.30 where no pilot
    outcome exists, pending (blank n) where the pilot input is not available yet."""
    pd_by = {r["key"]: r for r in disc}
    sd_by = {r["key"]: r for r in sds}
    out = []
    for key, method, delta, floor, src, note in SS_SPECS:
        row = {"key": key, "base_key": key.split("|")[0], "method": method, "delta_or_effect": delta, "floor": floor,
               "split": SPLIT, "note": note, "p_d": "", "sd_diff": "", "p_d_source": "", "pilot_n": "", "n_required": "",
               "n_proposed": "", "run_ids": []}
        if method == "fixed":
            row.update(p_d_source="fixed", n_required=int(delta), n_proposed=int(delta))
        elif method == "paired_normal":
            r = sd_by.get(src)
            if r is None or not r.get("sd_diff") or math.isnan(r["sd_diff"]) or r["sd_diff"] <= 0:
                row["p_d_source"] = "pending: no pilot SD"
            else:
                n = n_paired_normal(r["sd_diff"], delta)
                row.update(sd_diff=r["sd_diff"], p_d_source="pilot", pilot_n=r["n"], n_required=n,
                           n_proposed=max(n, floor), run_ids=r["run_ids"])
        else:
            r = pd_by.get(src) if src else None
            if src and r is None and src != K_V3_B1:      # V3 vs B1 without a mini-LoRA row keeps the 0.30 default
                row["p_d_source"] = "pending: no pilot p_d"
            else:
                pdv, srcname = (r["p_d"], "pilot") if r is not None else (DEFAULT_PD, "default 0.30 (no pilot outcome)")
                if r is not None and r.get("note"):
                    row["note"] += "; " + r["note"]
                row.update(p_d=pdv, p_d_source=srcname, pilot_n=r["n"] if r is not None else "",
                           run_ids=r["run_ids"] if r is not None else [])
                if method == "mcnemar" and pdv <= delta ** 2:
                    row["note"] += "; p_d <= delta^2: formula undefined, use the floor/BOOST judgement"
                elif pdv <= 0:
                    row["note"] += "; p_d = 0: formula undefined"
                else:
                    n = stats.n_mcnemar(pdv, delta) if method == "mcnemar" else stats.n_tost(pdv, delta)
                    row.update(n_required=n, n_proposed=max(n, floor))
        out.append(row)
    return out


# ====================================================================== driver
DISC_EXTRA = ["n10", "n01", "split", "p_d", "systems", "oracle", "n_error_a", "n_error_b", "n_excluded", "em_a", "em_b",
              "note"]
SD_EXTRA = ["n10", "n01", "split", "sd_diff", "unit", "oracle", "n_excluded", "n_error_d2", "n_error_d4",
            "n_not_fit_d2", "n_not_fit_d4", "n_error_unpinned", "n_error_pinned", "n_budget_unpinned", "n_budget_pinned",
            "n_not_fit_unpinned", "n_not_fit_pinned", "imputed_worst", "prompt_version", "tuning_iteration"]


def run_all(store, out_dir: Path | str, runs_dir=None, log: Callable[[str], None] = print) -> dict:
    """Run every section against ``store`` and write the tables; returns {"written": [...], "skipped": [...]}."""
    out = Path(out_dir)
    written, skipped = [], []

    def load(name, kind="multi", with_items=True):
        try:
            return (load_multi_run if kind == "multi" else load_single_run)(store, RUNS[name], runs_dir, with_items)
        except MissingInput as e:
            skipped.append(f"{name}: {e}")
            return None

    main, mt = load("lb_main"), load("mt_arms")
    disc, sk = discordance_rows(main, mt, store, runs_dir)
    skipped += sk
    if disc:
        written.append(str(write_contrast_table(out / "discordance.csv", disc, extra_columns=DISC_EXTRA,
                                                 ci_only_keys=[K_H8])))

    d2, d4 = load("drift_d2", "single"), load("drift_d4", "single")
    pin, nopin = load("thrash_pin", "single"), load("thrash_nopin", "single")
    sds = []
    if d2 and d4 and pin and nopin:
        try:
            sds = sd_rows(d2, d4, pin, nopin)
            written.append(str(write_contrast_table(out / "sd_paired.csv", sds, extra_columns=SD_EXTRA)))
        except MissingInput as e:
            skipped.append(f"sd_paired: {e}")
    else:
        skipped.append("sd_paired: drift or thrash records missing")

    fwd = load("lb_fwd_mixed")
    if main and fwd:
        try:
            items = {**main.items, **fwd.items}
            check_em_consistency(main, "S11")
            check_em_consistency(fwd, "S11")
            per = _h3_inputs(items, sys_records(main, "S11")) + _h3_inputs(items, sys_records(fwd, "S11"))
            rows, bins = h3_fit(per, [main.run_id, fwd.run_id])
            written.append(str(write_table(out / "h3_fit.csv", rows, columns=["key", "value", "n", "detail", "split", "run_ids", "oracle"])))
            written.append(str(write_table(out / "h3_bins.csv", bins)))
        except MissingInput as e:
            skipped.append(f"h3_fit: {e}")
    else:
        skipped.append("h3_fit: needs the late-h2 main run and the forward/mixed run")

    reps = [main, load("lb_rep2"), load("lb_rep3")]
    if all(reps):
        try:
            written.append(str(write_table(out / "noise.csv", noise_rows(reps))))
        except MissingInput as e:
            skipped.append(f"noise: {e}")
    else:
        skipped.append("noise: needs all three late-h2 repeat runs")

    extra = []
    for fn in (trackb_attribution_row, remedies_info_row):
        try:
            extra.append(fn(store))
        except MissingInput as e:
            skipped.append(f"{fn.__name__}: {e}")
    if extra:
        disc = disc + extra
        written = [w for w in written if not w.endswith("discordance.csv")]
        written.append(str(write_contrast_table(out / "discordance.csv", disc, extra_columns=DISC_EXTRA,
                                                ci_only_keys=[K_H8])))
    ss = sample_size_rows(disc, sds)
    written.append(str(write_table(out / "sample_sizes_proposed.csv", ss, columns=[
        "key", "base_key", "method", "delta_or_effect", "floor", "p_d", "sd_diff", "p_d_source", "pilot_n", "n_required",
        "n_proposed", "split", "run_ids", "note"])))
    for w in written:
        log(f"wrote {w}")
    for s in skipped:
        log(f"SKIP {s}")
    return {"written": written, "skipped": skipped}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="bundle/tables/pilot_dev")
    ap.add_argument("--creds", default=None, help="service-account key file used when GCP_SA_JSON is unset "
                                                  "(default: the path recorded in config/project.yaml / credentials.md)")
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args(argv)
    creds = a.creds or credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")        # never printed
    res = run_all(GcsStore(), Path(a.out), a.runs_dir)
    return 0 if res["written"] else 1


if __name__ == "__main__":
    sys.exit(main())
