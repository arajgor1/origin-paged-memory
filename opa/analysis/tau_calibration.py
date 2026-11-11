"""tau_thread recalibration on harrier embeddings (plan v3, Section 8, P0), CPU only, free.

PRE-REGISTERED SELECTION RULE (decided by the operator before seeing any harrier data):
  Sweep tau in {0.50, 0.55, ..., 0.95}; for each tau run the harness's own thread assignment (through Session or the
  exact function index.py uses, not a re-implementation) on every item; metric = adjusted Rand index (ARI) between
  inferred and gold thread labels per item, averaged over items, pooled over cells with equal weight per cell; choose
  the tau with the highest mean ARI; ties (within 0.005) -> the larger tau (over-splitting preferred over merging, per
  the 2026-11-06 decision). Also report per tau: mean inferred thread count vs gold, purity, inverse purity, ARI per
  cell.

Implementation notes. Threads are assigned by ``opa.tracks.track_a.index.Session`` itself (skeleton "none", no graph,
use_gold_threads False, ``tau_thread`` = the swept value, embeddings from the precomputed harrier npz through
``PrecomputedEmbedder.for_item``, which checks the text hashes). Gold labels are ``origins[i]["thread_id"]`` and are
used ONLY to score, never to assign. "Within 0.005" means mean ARI >= best - 0.005; of those taus the largest wins.
Items whose ``split`` is not "dev" are refused (ValueError). The script does not write config; the chosen tau is
printed and recorded in the sidecar, and the operator applies it.

Purity = sum over inferred clusters of the largest gold-class overlap / n; inverse purity swaps the roles. Both are
averaged over items like ARI. ARI of two trivial labelings (zero denominator) is 1.

Cells: ``--cells-dir`` takes cell directories (items.jsonl.gz) and/or roots holding such directories (searched
recursively). Only dataset "mtlbmhqa" cells are used; ``-conf`` (decoy) variants are skipped unless
``--include-conf``. ``--npz-root`` is an embeddings .npz file, a directory of them, or a gs:// URI of one .npz (comma
separated for several); each item must be found in exactly one file.

Items: the first ``--n-items`` (default 10) items of each cell from ``--offset`` (default 0), in file order, exactly as
``opa.tracks.track_a.run.select_items``; an item without an embedding is a hard error. Every loaded npz must carry
identical meta (model, revision, dim, max_length, dtype, B, version) and the model must equal ``--expect-model``
(default microsoft/harrier-oss-v1-0.6b); an item must be present in exactly one file. A cell directory must have a
manifest.json with split "dev" and a "dev" path segment.

CLI: python -m opa.analysis.tau_calibration \\
        --npz-root gs://<bucket>/opa/runs/<p0-embed-dev run>/embeddings.npz,gs://<bucket>/opa/runs/<p0-embed-dev-aux run>/embeddings.npz \\
        --cells-dir <local dev v2 root> --out bundle/tables/tau_thread_calibration_dev.csv
(p0-embed-dev holds T8-h3-32k-cross, p0-embed-dev-aux the other five cells; a directory of .npz files also works.)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from opa.analysis.tables import write_table
from opa.tracks.track_a.config import TAU_THREAD
from opa.tracks.track_a.embed import PrecomputedEmbedder
from opa.tracks.track_a.index import Session

TAUS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))      # 0.50 .. 0.95
TIE_TOL = 0.005
EXPECT_MODEL = "microsoft/harrier-oss-v1-0.6b"
HASH_MODEL = "hash-bow-512"
META_KEYS = ("model", "revision", "dim", "max_length", "dtype", "B", "version")
SCRIPT_REL = "code/opa/analysis/tau_calibration.py"
COLUMNS = ["key", "level", "tau", "cell", "n_items", "mean_ari", "n_threads_inferred", "n_threads_gold", "purity",
           "inverse_purity", "n_cells", "n_items_short", "selected", "embedder", "split"]


# ---------------------------------------------------------------- metrics
def _contingency(a, b):
    a, b = list(a), list(b)
    ua, ub = sorted(set(a), key=str), sorted(set(b), key=str)
    ia, ib = {x: i for i, x in enumerate(ua)}, {x: i for i, x in enumerate(ub)}
    c = np.zeros((len(ua), len(ib)), dtype=np.int64)
    for x, y in zip(a, b):
        c[ia[x], ib[y]] += 1
    return c


def adjusted_rand_index(a, b) -> float:
    """Hubert-Arabie ARI of two labelings of the same items; 1 for identical partitions (up to renaming)."""
    if len(a) != len(b):
        raise ValueError("labelings differ in length")
    n = len(a)
    if n < 2:
        return 1.0
    c = _contingency(a, b)
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_ij = float(comb2(c).sum())
    sum_a, sum_b = float(comb2(c.sum(1)).sum()), float(comb2(c.sum(0)).sum())
    expected = sum_a * sum_b / comb2(n)
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:                                   # both labelings trivial (one cluster / all singletons)
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def purity(inferred, gold) -> float:
    """Fraction of items in the majority gold class of their inferred cluster."""
    c = _contingency(inferred, gold)
    return float(c.max(1).sum() / max(len(inferred), 1))


def inverse_purity(inferred, gold) -> float:
    return purity(gold, inferred)


def choose_tau(mean_ari: dict, tol: float = TIE_TOL) -> float:
    """Highest mean ARI; among taus within ``tol`` of it, the largest tau."""
    best = max(mean_ari.values())
    return max(t for t, v in mean_ari.items() if v >= best - tol)


# ---------------------------------------------------------------- data
def check_dev(item) -> None:
    if item.get("split") != "dev":
        raise ValueError(f"refusing item {item.get('item_id')!r}: split is {item.get('split')!r}, not 'dev'")


def bucket_text():
    try:
        from opa.gcs import bucket_name
        b = bucket_name()
        return b if b and "YOUR-" not in b else "<bucket-unset>"
    except Exception:
        return "<bucket-unset>"


def read_cell(cell_dir, n_items=10, offset=0):
    """(cell, items, input record). Refuses unless manifest.json says split "dev" and the path has a "dev" segment
    (checked before items.jsonl.gz is read); items are sliced like select_items: [offset:][:n_items]."""
    cell_dir = Path(cell_dir)
    if "dev" not in cell_dir.resolve().parts:
        raise ValueError(f"refusing {cell_dir}: no 'dev' path segment")
    man = cell_dir / "manifest.json"
    if not man.is_file():
        raise ValueError(f"refusing {cell_dir}: no manifest.json")
    mj = json.loads(man.read_text(encoding="utf-8"))
    if mj.get("split") != "dev":
        raise ValueError(f"refusing {cell_dir}: manifest split is {mj.get('split')!r}, not 'dev'")
    cell = mj.get("cell", cell_dir.name)
    raw = (cell_dir / "items.jsonl.gz").read_bytes()
    items = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines() if l.strip()]
    for it in items:
        check_dev(it)
        it.setdefault("params", {})
    items = items[int(offset):]
    items = items[: int(n_items)] if n_items is not None else items
    tail = "/".join(cell_dir.resolve().parts[-4:])
    return cell, items, {"uri": f"gs://{bucket_text()}/opa/data/{tail}/items.jsonl.gz",
                         "sha256": hashlib.sha256(raw).hexdigest()}


def find_cells(roots):
    out = []
    for r in roots:
        r = Path(r)
        if (r / "items.jsonl.gz").is_file():
            out.append(r)
        else:
            out += sorted(f.parent for f in r.rglob("items.jsonl.gz"))
    return out


def load_embedders(npz_root):
    """[(PrecomputedEmbedder, input record)] from a file, a directory of .npz, or gs:// URIs (comma separated)."""
    srcs = []
    for part in str(npz_root).split(","):
        part = part.strip()
        if part.startswith("gs://"):
            from opa import gcs
            raw = gcs.get_bytes(part)
            srcs.append((raw, {"uri": part, "sha256": hashlib.sha256(raw).hexdigest()}))
        else:
            p = Path(part)
            for f in ([p] if p.is_file() else sorted(p.rglob("*.npz"))):
                raw = f.read_bytes()
                srcs.append((raw, {"path": str(f), "sha256": hashlib.sha256(raw).hexdigest()}))
    if not srcs:
        raise FileNotFoundError(f"no .npz embeddings under {npz_root}")
    return [(PrecomputedEmbedder(raw), rec) for raw, rec in srcs]


def bound_embedder(embedders, item):
    found, last = [], None
    for emb, _ in embedders:
        try:
            found.append(emb.for_item(item))
        except KeyError as e:
            last = e
    if not found:
        raise KeyError(f"no embeddings file holds item {item['item_id']}: {last}")
    if len(found) > 1:
        raise ValueError(f"item {item['item_id']} is present in {len(found)} embeddings files; expected exactly one")
    return found[0]


def check_meta(embedders, expect_model=EXPECT_MODEL):
    """All files must share one meta (META_KEYS) and the expected model; returns that full meta."""
    metas = [{k: e.meta.get(k) for k in META_KEYS} for e, _ in embedders]
    for m, (_, rec) in zip(metas, embedders):
        if m != metas[0]:
            raise ValueError(f"embeddings files disagree on meta: {metas[0]} vs {m} ({rec})")
    if expect_model and metas[0]["model"] != expect_model:
        raise ValueError(f"embeddings model {metas[0]['model']!r} != expected {expect_model!r}")
    if expect_model and expect_model != HASH_MODEL:         # real (Hugging Face) model: no key may be missing
        missing = [k for k in META_KEYS if metas[0][k] is None]
        if missing:
            raise ValueError(f"embeddings meta lacks {missing} (required unless --expect-model {HASH_MODEL})")
    return dict(embedders[0][0].meta)


# ---------------------------------------------------------------- calibration
def item_scores(item, embedder, tau):
    """Run the harness's thread assignment (Session) at ``tau`` and score it against the gold thread labels."""
    sess = Session(item, {"skeleton": "none", "graph": False, "use_gold_threads": False, "tau_thread": tau}, embedder)
    inferred = [sess.thread_of[o] for o in range(sess.N)]
    gold = [int(o.get("thread_id", 0)) for o in item["origins"]]
    return {"ari": adjusted_rand_index(inferred, gold), "n_inf": len(set(inferred)), "n_gold": len(set(gold)),
            "purity": purity(inferred, gold), "inv_purity": inverse_purity(inferred, gold)}


def calibrate(cells, embedders, taus=TAUS, n_requested=None):
    """cells: [(cell_name, items)]. Returns (rows, chosen_tau). Rows: per tau x cell and one pooled row per tau."""
    for _, items in cells:
        for it in items:
            check_dev(it)
    bound = {(c, it["item_id"]): bound_embedder(embedders, it) for c, items in cells for it in items}
    emb_name = embedders[0][0].meta.get("model", "")
    short = {c: n_requested is not None and len(items) < n_requested for c, items in cells}
    rows, pooled_ari = [], {}
    for tau in taus:
        cell_rows = []
        for cell, items in cells:
            ms = [item_scores(it, bound[(cell, it["item_id"])], tau) for it in items]
            mean = lambda k: float(np.mean([m[k] for m in ms]))
            cell_rows.append({"key": f"tau{tau:.2f}_{cell}", "level": "cell", "tau": tau, "cell": cell,
                              "n_items": len(ms), "mean_ari": mean("ari"), "n_threads_inferred": mean("n_inf"),
                              "n_threads_gold": mean("n_gold"), "purity": mean("purity"),
                              "inverse_purity": mean("inv_purity"), "n_cells": None,
                              "n_items_short": short[cell], "selected": None,
                              "embedder": emb_name, "split": "dev"})
        rows += cell_rows
        pm = lambda k: float(np.mean([r[k] for r in cell_rows]))          # equal weight per cell
        pooled_ari[tau] = pm("mean_ari")
        rows.append({"key": f"tau{tau:.2f}_pooled", "level": "pooled", "tau": tau, "cell": "pooled",
                     "n_items": sum(r["n_items"] for r in cell_rows), "mean_ari": pooled_ari[tau],
                     "n_threads_inferred": pm("n_threads_inferred"), "n_threads_gold": pm("n_threads_gold"),
                     "purity": pm("purity"), "inverse_purity": pm("inverse_purity"), "n_cells": len(cell_rows),
                     "n_items_short": any(short.values()), "selected": None, "embedder": emb_name, "split": "dev"})
    chosen = choose_tau(pooled_ari)
    for r in rows:
        r["selected"] = (r["tau"] == chosen)
    return rows, chosen


def write_outputs(rows, chosen, out, inputs, extra=None):
    out = Path(out)
    write_table(out, rows, COLUMNS)
    side = {"script": SCRIPT_REL, "inputs": list(inputs),
            "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "table_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "chosen_tau_thread": chosen, "previous_tau_thread": TAU_THREAD, "taus": list(TAUS), "tie_tolerance": TIE_TOL}
    side.update(extra or {})
    side_path = out.with_suffix(".json")
    side_path.write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return side_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--cells-dir", action="append", required=True)
    ap.add_argument("--out", default="bundle/tables/tau_thread_calibration_dev.csv")
    ap.add_argument("--n-items", type=int, default=10, help="items per cell (first n in file order, like select_items)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--expect-model", default=EXPECT_MODEL)
    ap.add_argument("--include-conf", action="store_true", help="also use the -conf (decoy) cells")
    a = ap.parse_args(argv)
    if a.n_items <= 0:
        ap.error("--n-items must be >= 1")
    embedders = load_embedders(a.npz_root)
    meta = check_meta(embedders, a.expect_model)
    inputs = [rec for _, rec in embedders]
    cells = []
    for d in find_cells(a.cells_dir):
        cell, items, rec = read_cell(d, a.n_items, a.offset)
        if items and items[0].get("dataset", "mtlbmhqa") != "mtlbmhqa":
            continue
        if cell.endswith("-conf") and not a.include_conf:
            continue
        cells.append((cell, items))
        inputs.append(rec)
    if not cells:
        raise SystemExit("no MT-LB-MHQA dev cells found")
    rows, chosen = calibrate(cells, embedders, n_requested=a.n_items)
    for c, it in cells:
        if len(it) < a.n_items:
            print(f"WARNING: cell {c} has {len(it)} items, fewer than the {a.n_items} requested")
    write_outputs(rows, chosen, a.out, inputs,
                  {"n_items": a.n_items, "offset": a.offset, "embedding_meta": meta,
                   "n_items_short": [c for c, it in cells if len(it) < a.n_items]})
    print(f"{'tau':>5s} {'ARI':>7s} {'thr_inf':>8s} {'thr_gold':>8s} {'purity':>7s} {'inv_pur':>7s}")
    for r in rows:
        if r["level"] == "pooled":
            print(f"{r['tau']:5.2f} {r['mean_ari']:7.3f} {r['n_threads_inferred']:8.1f} {r['n_threads_gold']:8.1f} "
                  f"{r['purity']:7.3f} {r['inverse_purity']:7.3f}{'  <- chosen' if r['selected'] else ''}")
    print(f"chosen tau_thread = {chosen}")
    return chosen


if __name__ == "__main__":
    main()
