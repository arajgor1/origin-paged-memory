"""Blind routing-hit check of a candidate cell (owner request; same blindness discipline as opa.analysis.h3_cells).

  python -m opa.analysis.routing_hits --run <run_id> [--run ...] --system S16 --out <dir>

Purpose: select a candidate H5 cell (for example MT T4 h3 cross with half the resident-page budget, M = 2) on a
NEUTRAL system's routing hit rate alone, before any outcome of that system is looked at.

Definition. Take the gold chain of an item: facts f_0 ... f_{h-1} ordered by hop; fact f_j reveals the entity that
fact f_{j+1} is about. For every hop j+1 >= 1 the router must find the origin of f_{j+1} from the entity revealed by
f_j. A routing HIT at that hop is:

    lookup(reveals(f_j)) contains origin(f_{j+1})

where lookup is the skeleton lookup of opa.tracks.track_a.index.Session.lookup applied to the record's stored skeleton:
the keys of the entity are normalize_key(entity) plus its alias_keys (opa.tracks.track_a.text, the helpers
Session.lookup itself uses), and the origin matches if one of them is in that origin's skeleton. This is exactly the
measurement W3 made on the pilot ("lookup(next revealed entity) hits the next gold fact origin 100 of 100"; this tool
reproduces 100/100 on the S16 sub-run of A-S12+S14+S15+S16+S18-mtlbmhqa-T4-h3-32k-cross). ``hit_rate`` = hits / hops
over all items; ``hop<k>_rate`` is the rate at hop k (the target fact's hop number); ``n`` items, ``n_hops``.

Selection field (design ruling 2026-11-07): ``routed_rate`` = share of the same hops whose target origin is in the
initial window (the first M entries of ``paged_in``; M is the job's opts.M, default 4) or is the origin of one of
the record's ``faults`` (an origin the agent routed to by lookup/descent and paged in on demand). It responds to the
resident-page budget M, which the lookup hit rate does not. ``qualifies = routed_rate < 0.90`` (owner threshold: a
cell where routing is already near-perfect cannot show a geometry effect); this is THE selection field.
``initial_window_rate`` = the share in the first M page-ins only.

Reported beside them, not used for selection: ``hit_rate`` (the lookup hit above) and ``paged_rate`` (target origin
paged in at any point; ``paged_in`` has no timestamps, and descent-first runs scan the whole archive after the
initial window, so it is insensitive to M). ``n_budget_stop`` counts records whose status is "budget" (a count
only). ``reveals`` may be a string or a list of entities (any of them counts); origin ids may be ints or strings.

Blindness, by construction: records are projected to {item_id, skeleton, paged_in, budget_stop} the moment they are
read; the prediction, other status values, failure codes, EM/F1 never leave the loader; the system's metrics.json is
never fetched; nothing here calls an outcome scorer (tests spy on them); the output has no outcome column.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from ..tracks.track_a.text import alias_keys, normalize_key
from . import pilot as pl
from .tables import write_table

HIT_CUT = 0.90
MAX_HOPS = 4
KEPT_FIELDS = ("item_id", "skeleton", "paged_in")
OUTPUT_COLUMNS = (["key", "system", "M", "n", "n_hops", "routed_rate", "initial_window_rate", "hit_rate", "paged_rate"]
                  + [f"hop{k}_rate" for k in range(1, MAX_HOPS + 1)]
                  + ["n_budget_stop", "qualifies", "split", "run_ids"])
DEFAULT_M = 4


def project_record(rec: dict) -> dict:
    """Keep the routing fields (skeleton, paged_in, the origin ids of the faults) and the budget-stop flag;
    everything else is dropped here."""
    return {**{k: rec.get(k) for k in KEPT_FIELDS},
            "fault_origins": [f.get("origin_id") for f in (rec.get("faults") or []) if isinstance(f, dict)],
            "budget_stop": rec.get("status") == "budget"}


def load_records(store, run_id: str, system: str) -> list[dict]:
    u = f"{pl.run_uri(store, run_id)}/{system}/r0/records.jsonl.gz"
    try:
        raw = pl.read_jsonl(store.get(u))
    except KeyError as e:
        raise pl.MissingInput(f"{run_id}: no {system} records yet") from e
    return [project_record(r) for r in raw]


def lookup_hit(skeleton: dict, entity: str, origin_id) -> bool:
    """Session.lookup(entity) restricted to one origin, on a stored skeleton ({origin: [keys]})."""
    keys = {normalize_key(entity)} | alias_keys(normalize_key(entity))
    stored = {normalize_key(x) for x in (skeleton.get(str(origin_id)) or skeleton.get(origin_id) or [])}
    return bool(keys & stored)


def _ids(xs) -> list[int]:
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


def item_hops(item: dict, record: dict, M: int = DEFAULT_M) -> list[dict]:
    """[{hop, lookup_hit, paged, in_window, routed}] for hops 1..h-1 of the item's gold chain."""
    facts = sorted(item.get("gold", {}).get("facts", []), key=lambda f: f["hop"])
    order = _ids(record.get("paged_in"))
    paged, window = set(order), set(order[:M])
    faulted = set(_ids(record.get("fault_origins")))
    sk = record.get("skeleton") or {}
    out = []
    for j in range(len(facts) - 1):
        nxt = int(facts[j + 1]["origin_id"])
        ents = facts[j].get("reveals")
        ents = [ents] if isinstance(ents, str) else [e for e in (ents or []) if e]
        out.append({"hop": j + 1, "lookup_hit": any(lookup_hit(sk, e, nxt) for e in ents), "paged": nxt in paged,
                    "in_window": nxt in window, "routed": nxt in window or nxt in faulted})
    return out


def hit_stats(items: dict, records: Sequence[dict], M: int = DEFAULT_M) -> dict:
    hops, n_items, nb = [], 0, 0
    for r in records:
        it = items.get(r["item_id"])
        if it is None:
            continue
        n_items += 1
        nb += bool(r.get("budget_stop"))
        hops += item_hops(it, r, M)
    if not hops:
        raise pl.MissingInput("no multi-hop item matched a record")
    rate = lambda key: sum(h[key] for h in hops) / len(hops)
    row = {"M": M, "n": n_items, "n_hops": len(hops), "routed_rate": rate("routed"),
           "initial_window_rate": rate("in_window"), "hit_rate": rate("lookup_hit"),
           "paged_rate": rate("paged"), "n_budget_stop": int(nb)}
    for k in range(1, MAX_HOPS + 1):
        hk = [h for h in hops if h["hop"] == k]
        row[f"hop{k}_rate"] = sum(h["lookup_hit"] for h in hk) / len(hk) if hk else ""
    row["qualifies"] = bool(row["routed_rate"] < HIT_CUT)
    return row


def cell_row(store, run_id: str, system: str, runs_dir=None) -> dict:
    job = pl.load_job(run_id, runs_dir)
    items = pl.load_run_items(store, job)
    M = int(((job.get("args") or {}).get("opts") or {}).get("M", DEFAULT_M))
    st = hit_stats(items, load_records(store, run_id, system), M)
    return {"key": str(job.get("cell") or run_id), "system": system, **st, "split": pl.SPLIT, "run_ids": [run_id]}


def run_cells(store, run_ids: Sequence[str], system: str, out_dir: Path | str, runs_dir=None, log=print) -> dict:
    rows, skipped = [], []
    for rid in run_ids:
        try:
            row = cell_row(store, rid, system, runs_dir)
        except pl.MissingInput as e:
            skipped.append(f"{rid}: {e}")
            continue
        if any(r["key"] == row["key"] for r in rows):
            row["key"] = f"{row['key']}@{rid}"
        rows.append(row)
    path = write_table(Path(out_dir) / "routing_hits.csv", rows, columns=OUTPUT_COLUMNS) if rows else None
    for r in rows:
        log(f"{r['key']} {r['system']}: n={r['n']} hops={r['n_hops']} routed={r['routed_rate']:.3f} "
            f"window={r['initial_window_rate']:.3f} hit={r['hit_rate']:.3f} paged={r['paged_rate']:.3f} M={r['M']} budget_stops={r['n_budget_stop']} qualifies={r['qualifies']}")
    if path:
        log(f"wrote {path}")
    for s in skipped:
        log(f"SKIP {s}")
    return {"rows": rows, "skipped": skipped, "path": str(path) if path else None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--system", default="S16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--creds", default=None)
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args(argv)
    if a.system in pl.ORACLE_SYSTEMS:
        ap.error("the routing-hit check uses a neutral system, not an oracle")
    creds = a.creds or pl.credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")        # never printed
    res = run_cells(pl.GcsStore(), a.run, a.system, a.out, a.runs_dir)
    return 0 if res["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
