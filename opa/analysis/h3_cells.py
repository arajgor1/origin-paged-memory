"""Blind H3 cell selection (PLAN_V3 revision 3, Section 6b): choose H3's test cells by the predictor alone.

  python -m opa.analysis.h3_cells --run <run_id> [--run <run_id> ...] --out <dir>

Each ``--run`` is a dev run of S11 (and S18) on one candidate cell. For every cell this computes ONLY predictor
statistics: per item the predicted success gamma_eff ** h_late (opa.scoring.coverage_factors, the same function the
H3 fit in opa.analysis.pilot uses) from the S11 records' skeleton and residency (edges_from_resident) fields and the
item's gold facts; then n, mean, SD (ddof 1), the share of items with predicted <= 0.8 and the number with h_late = 0.

Blindness, by construction:
  * S11 records are projected to {item_id, skeleton, edges_from_resident, budget_stop} the moment they are read, so
    the prediction, failure codes and every other outcome-bearing field never leave the loader. The status field is
    reduced to the boolean budget_stop (status == "budget"), and only its COUNT per cell, ``n_budget_stop``, is
    reported: a budget stop truncates residency and can lower the predicted gamma for a process reason, so it bears
    on predictor validity. The status value itself and per-item flags never reach the output;
  * the S11 sub-run's metrics.json is never fetched;
  * nothing in this module calls scoring.exact_match / score_item / f1_score (a test spies on them);
  * the output table has no outcome column.
The one outcome it may read is the ORACLE system's: S18's metrics.json em, reported as ``oracle_em``. A cell
``qualifies`` when SD >= 0.15 AND share(<= 0.8) >= 0.30 AND oracle_em >= 0.60 (Section 6b item 3).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import scoring
from . import pilot as pl
from .tables import write_table

SD_MIN, SHARE_MIN, ORACLE_MIN, PRED_CUT = 0.15, 0.30, 0.60, 0.8
COVERAGE_FIELDS = ("item_id", "skeleton", "edges_from_resident")        # the only S11 record fields ever kept
OUTPUT_COLUMNS = ["key", "n", "pred_mean", "pred_sd", "share_le_0_8", "n_h_late_0", "n_budget_stop", "oracle_em",
                  "qualifies", "split", "run_ids"]


def project_record(rec: dict) -> dict:
    """Keep the coverage fields and the budget-stop flag only; everything else (prediction, status, tokens, ...) is
    dropped here."""
    return {**{k: rec.get(k) for k in COVERAGE_FIELDS}, "budget_stop": rec.get("status") == "budget"}


def load_s11_coverage_records(store, run_id: str) -> list[dict]:
    """Projected S11 r0 records of a run. Raises MissingInput if the sub-run has no records yet."""
    u = pl.run_uri(store, run_id) + "/S11/r0/records.jsonl.gz"
    try:
        raw = pl.read_jsonl(store.get(u))
    except KeyError as e:
        raise pl.MissingInput(f"{run_id}: no S11 records yet") from e
    return [project_record(r) for r in raw]


def oracle_em(store, run_id: str) -> float | None:
    """S18's (oracle) sub-run em from its metrics.json: the only outcome statistic this tool reads."""
    import json
    try:
        return float(json.loads(store.get(pl.run_uri(store, run_id) + "/S18/r0/metrics.json"))["em"])
    except (KeyError, ValueError):
        return None


def predictor_stats(items: dict, records: Sequence[dict]) -> dict:
    """Predictor statistics of one cell from projected records and the items' gold facts."""
    pred, h0, nb = [], 0, 0
    for r in records:
        it = items.get(r["item_id"])
        if it is None:
            continue
        nb += bool(r.get("budget_stop"))
        cf = scoring.coverage_factors(it, r)
        pred.append(cf["predicted"])
        h0 += cf["h_late"] == 0
    if not pred:
        raise pl.MissingInput("no S11 record matches an item")
    p = np.array(pred, float)
    return {"n": len(p), "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)) if len(p) > 1 else float("nan"),
            "share_le_0_8": float((p <= PRED_CUT + 1e-12).mean()), "n_h_late_0": int(h0),
            "n_budget_stop": int(nb)}


def qualifies(stats_row: dict, oracle: float | None) -> bool:
    sd = stats_row["pred_sd"]
    return bool(oracle is not None and sd == sd and sd >= SD_MIN and stats_row["share_le_0_8"] >= SHARE_MIN
                and oracle >= ORACLE_MIN)


def cell_row(store, run_id: str, runs_dir=None) -> dict:
    job = pl.load_job(run_id, runs_dir)
    items = pl.load_run_items(store, job)
    st = predictor_stats(items, load_s11_coverage_records(store, run_id))
    o = oracle_em(store, run_id)
    return {"key": str(job.get("cell") or run_id), **st, "oracle_em": "" if o is None else o,
            "qualifies": qualifies(st, o), "split": pl.SPLIT, "run_ids": [run_id]}


def run_cells(store, run_ids: Sequence[str], out_dir: Path | str, runs_dir=None, log=print) -> dict:
    rows, skipped = [], []
    for rid in run_ids:
        try:
            row = cell_row(store, rid, runs_dir)
        except pl.MissingInput as e:
            skipped.append(f"{rid}: {e}")
            continue
        if any(r["key"] == row["key"] for r in rows):
            row["key"] = f"{row['key']}@{rid}"
        rows.append(row)
    path = None
    if rows:
        path = write_table(Path(out_dir) / "h3_cells.csv", rows, columns=OUTPUT_COLUMNS)
        for r in rows:
            log(f"{r['key']}: n={r['n']} mean={r['pred_mean']:.3f} sd={r['pred_sd']:.3f} "
                f"share<=0.8={r['share_le_0_8']:.2f} h_late0={r['n_h_late_0']} budget_stops={r['n_budget_stop']} oracle_em={r['oracle_em']} "
                f"qualifies={r['qualifies']}")
        log(f"wrote {path}")
    for s in skipped:
        log(f"SKIP {s}")
    return {"rows": rows, "skipped": skipped, "path": str(path) if path else None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="dev run id of a candidate cell (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--creds", default=None)
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args(argv)
    creds = a.creds or pl.credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")        # never printed
    res = run_cells(pl.GcsStore(), a.run, a.out, a.runs_dir)
    return 0 if res["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
