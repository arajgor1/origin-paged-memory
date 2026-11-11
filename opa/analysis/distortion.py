"""H5 distortion check (PLAN_V3.md Section 8 check 3), CPU only, free.

Question: on the dev MT-LB-MHQA address trees THE HARNESS BUILDS, how well do the A1 (Euclidean) and A2 (hyperbolic,
LCA-frame) layouts preserve the tree distance d_T (number of edges, = A3 / delta)?

The layout is taken from ``opa.tracks.track_a.index.Session`` itself (HashEmbedder, skeleton "none", no graph), so the
measurement cannot drift from the harness: each thread is a CHAIN (parent = previous origin of the same thread), every
step of thread t uses direction dirs[t] of spread_directions(64, 32, seed=0), node = origin + 1, TreeLayout(m, delta,
d_clamp) with m, delta and tau_thread left at the harness defaults (Session reads them from its cfg; only d_clamp is
set here, for the sensitivity). Two thread assignments, reported as separate rows (``threads`` column):
  gold      Session(use_gold_threads=True): the item's thread_id per origin (primary)
  inferred  Session(use_gold_threads=False, tau_thread=TAU_THREAD): threads inferred from the embedding similarity of
            the hashed bag-of-words embedder (``embedder`` column = hash-bow-512, NOT the real harrier embedder, so
            these rows are indicative only)
Two A2 clamp settings (``d_clamp`` column): ``none`` = the Track A default (clamp only on numerical collapse, replaced
by geometry.COLLAPSE_CAP and counted; PRIMARY) and ``36`` = the old fixed clamp (SECONDARY sensitivity).

Metrics. A system ranks candidate c by relevance - lambda * d(q_origin, c), q = the last origin, so the PRIMARY metric
is q-anchored: distances from q to every other origin. All-pairs over origins (subsampled to <= MAX_PAIRS pairs with a
fixed seed on big items) is a secondary column. For layout distances d and tree distances dT, r = d/dT:
  worst    = max(r) / min(r)                    worst-case multiplicative distortion (scale free, 1 is perfect)
  mean_rel = mean |s*r - 1|, s = sum(r)/sum(r^2)  (s minimises the mean squared RELATIVE error)
  affine   = RMS residual of the least-squares line d ~ a*dT + b over the q-anchored pairs, divided by mean(d): what
             a LINEAR re-ranker (relevance - lambda * d) effectively sees; 0 means d is an exact affine function of dT
  clamp    = pairs clipped by the fixed clamp (d >= d_clamp, d_clamp=36 rows) or answered by the last-resort cap
             (n_capped, d_clamp none rows; should be 0)
  beyond   = pairs whose path below the LCA is longer than geometry.SAFE_RADIUS on either side (not representable in
             float64; with d_clamp none they are computed exactly in arbitrary precision and counted in
             ``*_exact_fallback``, with d_clamp 36 the float result is clipped / unreliable)
  dist_sec   = wall seconds for the A1+A2+A3 distances of one item (q-anchored plus all-pairs; A2 dominates); kept out of the CSV (not deterministic), written to the sidecar as timing_dist_sec, mean over items
All metrics are computed per item and then averaged over the cell's items. The ratios are RATIOS OF THESE CELL MEANS
(mean A1 metric over mean A2 metric), not means of per-item ratios. ``tree_distortion_euclidean_over_hyperbolic`` is
the q-anchored worst ratio (A1 over A2, > 1 means hyperbolic is better; below 1 is possible and reported as is);
``ratio_allpairs`` and ``ratio_mean_rel`` are secondary.

"Largest ratio": computed over primary rows only (gold threads, d_clamp none) of REAL dev cells (source equals the
current generator version); synthetic and legacy-version rows are printed per source but never selected.

Sources: cell directories (items.jsonl.gz + manifest.json) and/or synthetic dev items made with
opa.data.mtlbmhqa.make_item (SyntheticFiller, so inferred-thread rows on synthetic text are not representative).

CLI: python -m opa.analysis.distortion --out bundle/tables/h5_distortion_dev.csv [--cells-dir DIR ...] [--synthetic]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from opa.analysis.tables import write_table
from opa.data import GENERATOR_VERSION
from opa.data.filler import local_data_dir
import time

from opa.geometry import SAFE_RADIUS
from opa.tracks.track_a.config import D_CLAMP, TAU_THREAD
from opa.tracks.track_a.index import HashEmbedder, Session

PRIMARY_RATIO = "tree_distortion_euclidean_over_hyperbolic"
MAX_PAIRS = 3000
DATA_ROOT = local_data_dir() / "mtlbmhqa" / "dev"
CLAMPS = (D_CLAMP, 36.0)            # primary (harness default) first, then the sensitivity
SYNTH_LENGTHS = {"8K": 8192, "16K": 16384, "32K": 32768}
SYNTH_THREADS = (2, 4, 8, 16)
SYNTH_HOPS = (2, 3)
SYNTH_SEEDS = (0, 1, 2)
SYNTH_FILLER = {"class": "SyntheticFiller", "n_books": 20, "words_per_book": 30000, "seed": 0}
SCRIPT_REL = "code/opa/analysis/distortion.py"
EMBEDDER = HashEmbedder.name


def clamp_label(d_clamp):
    return "none" if d_clamp is None else f"{d_clamp:g}"


# ---------------------------------------------------------------- metrics
def distortion(d, dT):
    """d, dT: arrays over distinct pairs (dT > 0). Returns dict(mean_rel, worst, scale)."""
    d, dT = np.asarray(d, float), np.asarray(dT, float)
    if len(d) == 0 or np.any(d <= 0):
        return {"mean_rel": float("inf"), "worst": float("inf"), "scale": float("nan")}
    r = d / dT
    s = float(np.sum(r) / np.sum(r * r))
    return {"mean_rel": float(np.mean(np.abs(s * r - 1.0))), "worst": float(np.max(r) / np.min(r)), "scale": s}


def affine_residual(d, dT):
    """RMS residual of the least-squares line d ~ a*dT + b, divided by mean(d)."""
    d, dT = np.asarray(d, float), np.asarray(dT, float)
    if len(d) < 3 or np.ptp(dT) == 0:
        return float("nan")
    A = np.stack([dT, np.ones_like(dT)], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(np.sqrt(np.mean((A @ coef - d) ** 2)) / np.mean(d))


def make_session(item, use_gold_threads, tau=TAU_THREAD, d_clamp=D_CLAMP, cfg=None):
    c = {"skeleton": "none", "graph": False, "use_gold_threads": use_gold_threads, "tau_thread": tau,
         "d_clamp": d_clamp}
    c.update(cfg or {})
    return Session(item, c, HashEmbedder())


def pair_list(n, q=None, max_pairs=MAX_PAIRS, seed=0):
    """q-anchored pairs (q, c) c != q, or all pairs of range(n) (subsampled to max_pairs, deterministic)."""
    if q is not None:
        return [(q, c) for c in range(n) if c != q]
    allp = [(i, j) for i in range(n) for j in range(i)]
    if len(allp) > max_pairs:
        idx = np.random.default_rng(seed).choice(len(allp), size=max_pairs, replace=False)
        allp = [allp[int(i)] for i in sorted(idx)]
    return allp


def session_distances(sess, pairs):
    """A1, A2, tree distances (edges) for origin pairs, straight from Session.distance (the harness path)."""
    a1 = np.array([sess.distance(i, j, "A1") for i, j in pairs])
    a2 = np.array([sess.distance(i, j, "A2") for i, j in pairs])
    dT = np.array([sess.distance(i, j, "A3") for i, j in pairs]) / sess.layout.delta
    return a1, a2, dT


def item_metrics(item, use_gold_threads, tau=TAU_THREAD, d_clamp=D_CLAMP, cfg=None):
    sess = make_session(item, use_gold_threads, tau, d_clamp, cfg)
    n, lay = sess.N, sess.layout
    out = {"n_origins": n, "n_threads_found": sess.n_threads, "delta": lay.delta, "m": lay.m}
    book, dist_sec = lay.book, 0.0
    for tag, pairs in (("q", pair_list(n, q=n - 1)), ("all", pair_list(n))):
        f0, c0 = book.n_exact_fallback, book.n_capped
        t0 = time.perf_counter()
        a1, a2, dT = session_distances(sess, pairs)
        dist_sec += time.perf_counter() - t0
        r1, r2 = distortion(a1, dT), distortion(a2, dT)
        nodes = [(i + 1, j + 1) for i, j in pairs]
        beyond = sum(1 for u, v in nodes if max(book.path_radius(u, book.lca(u, v)),
                                                book.path_radius(v, book.lca(u, v))) > SAFE_RADIUS)
        sat = (book.n_capped - c0) if lay.d_clamp is None else int(np.sum(a2 >= lay.d_clamp))
        out.update({f"{tag}_A1_worst": r1["worst"], f"{tag}_A2_worst": r2["worst"],
                    f"{tag}_A1_mean_rel": r1["mean_rel"], f"{tag}_A2_mean_rel": r2["mean_rel"],
                    f"{tag}_pairs": len(pairs), f"{tag}_clamp_count": sat, f"{tag}_clamp_frac": sat / len(pairs),
                    f"{tag}_beyond_safe_count": beyond, f"{tag}_beyond_safe_frac": beyond / len(pairs),
                    f"{tag}_exact_fallback": book.n_exact_fallback - f0})
        if tag == "q":
            out["q_A1_affine"], out["q_A2_affine"] = affine_residual(a1, dT), affine_residual(a2, dT)
    out["dist_sec"] = dist_sec
    return out


def cell_row(key, cell, source, items, use_gold_threads, tau=TAU_THREAD, d_clamp=D_CLAMP, cfg=None):
    ms = [item_metrics(it, use_gold_threads, tau, d_clamp, cfg) for it in items]
    mean = lambda k: float(np.mean([m[k] for m in ms]))
    row = {"key": key, "cell": cell, "source": source, "threads": "gold" if use_gold_threads else "inferred",
           "embedder": "" if use_gold_threads else EMBEDDER,
           "tau_thread": "" if use_gold_threads else tau, "d_clamp": clamp_label(d_clamp), "n_items": len(ms),
           "n_origins": ms[0]["n_origins"], "n_threads_found": mean("n_threads_found"), "delta": ms[0]["delta"],
           "m": ms[0]["m"]}
    p = items[0].get("params", {})
    row.update({"T": p.get("n_threads"), "h": p.get("hops"), "length_tokens": p.get("length_tokens"),
                "split": "dev"})
    for tag in ("q", "all"):
        for k in ("A1_worst", "A2_worst", "A1_mean_rel", "A2_mean_rel", "pairs", "clamp_count", "clamp_frac",
                  "beyond_safe_count", "beyond_safe_frac", "exact_fallback"):
            row[f"{tag}_{k}"] = mean(f"{tag}_{k}")
    row["q_A1_affine"], row["q_A2_affine"], row["dist_sec"] = mean("q_A1_affine"), mean("q_A2_affine"), mean("dist_sec")
    row[PRIMARY_RATIO] = row["q_A1_worst"] / row["q_A2_worst"]
    row["ratio_mean_rel"] = row["q_A1_mean_rel"] / row["q_A2_mean_rel"]
    row["ratio_allpairs"] = row["all_A1_worst"] / row["all_A2_worst"]
    return row


COLUMNS = ["key", "cell", "source", "T", "h", "length_tokens", "threads", "embedder", "tau_thread", "d_clamp",
           "n_items", "n_origins", "n_threads_found", "delta", "m", "q_pairs", "q_A1_worst", "q_A2_worst",
           "q_A1_mean_rel", "q_A2_mean_rel", "q_A1_affine", "q_A2_affine", "q_clamp_count", "q_clamp_frac",
           "q_beyond_safe_count", "q_beyond_safe_frac", "q_exact_fallback", "all_pairs", "all_A1_worst", "all_A2_worst", "all_A1_mean_rel", "all_A2_mean_rel", "all_clamp_count",
           "all_clamp_frac", "all_beyond_safe_count", "all_beyond_safe_frac", "all_exact_fallback", PRIMARY_RATIO, "ratio_mean_rel", "ratio_allpairs", "split"]


# ---------------------------------------------------------------- sources
def bucket_text():
    try:
        from opa.gcs import bucket_name
        b = bucket_name()
        return b if b and "YOUR-" not in b else "<bucket-unset>"
    except Exception:
        return "<bucket-unset>"


def read_cell_dir(cell_dir):
    """(cell_name, items, input record) of one cell directory. The record's uri mirrors the GCS layout
    gs://<bucket>/opa/data/<dataset>/<split>/<version>/<cell>/items.jsonl.gz (the last four path components)."""
    cell_dir = Path(cell_dir)
    f = cell_dir / "items.jsonl.gz"
    raw = f.read_bytes()
    items = [json.loads(line) for line in gzip.decompress(raw).decode("utf-8").splitlines() if line.strip()]
    man = cell_dir / "manifest.json"
    cell = json.loads(man.read_text(encoding="utf-8"))["cell"] if man.is_file() else cell_dir.name
    for it in items:
        it.setdefault("params", {})
    tail = "/".join(cell_dir.resolve().parts[-4:])
    return cell, items, {"uri": f"gs://{bucket_text()}/opa/data/{tail}/items.jsonl.gz",
                         "sha256": hashlib.sha256(raw).hexdigest()}


def synthetic_cells(lengths=None, threads=SYNTH_THREADS, hops=SYNTH_HOPS, seeds=SYNTH_SEEDS):
    """Valid dev cells (generator needs N/4 >= T) as (cell_name, items) from opa.data.mtlbmhqa.make_item."""
    from opa.data import mtlbmhqa
    from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
    filler = SyntheticFiller(n_books=SYNTH_FILLER["n_books"], words_per_book=SYNTH_FILLER["words_per_book"],
                             seed=SYNTH_FILLER["seed"])
    tok = WhitespaceTokenizer()
    for _lab, length in (SYNTH_LENGTHS if lengths is None else lengths).items():
        n_runs = length // 512 // 4
        for T in threads:
            if T > n_runs:
                continue
            for h in hops:
                items = [mtlbmhqa.make_item(s, T, h, length, True, 0.0, filler, tok) for s in seeds]
                yield mtlbmhqa.cell_name(T, h, length, True), items


def synthetic_provenance(lengths=None, threads=SYNTH_THREADS, hops=SYNTH_HOPS, seeds=SYNTH_SEEDS):
    return {"generator_version": GENERATOR_VERSION, "generator": "opa.data.mtlbmhqa.make_item", "split": "dev",
            "seeds": list(seeds), "filler": dict(SYNTH_FILLER), "tokenizer": "WhitespaceTokenizer",
            "grid": {"lengths": dict(SYNTH_LENGTHS if lengths is None else lengths), "threads": list(threads),
                     "hops": list(hops), "cross_thread": True, "alpha": 0.0}}


def run(cell_dirs=(), synthetic=False, clamps=CLAMPS, **kw):
    rows, inputs = [], []
    for d in cell_dirs:
        cell, items, rec = read_cell_dir(d)
        inputs.append(rec)
        src = Path(d).parent.name
        for dc in clamps:
            for gold in (True, False):
                rows.append(cell_row(f"{cell}_{src}_{'gold' if gold else 'inferred'}_clamp{clamp_label(dc)}",
                                     cell, src, items, gold, d_clamp=dc))
    if synthetic:
        for cell, items in synthetic_cells(**kw):
            for dc in clamps:
                for gold in (True, False):
                    rows.append(cell_row(f"{cell}_synth_{'gold' if gold else 'inferred'}_clamp{clamp_label(dc)}",
                                         cell, "synth", items, gold, d_clamp=dc))
    return rows, inputs


def is_primary(r):
    return r["threads"] == "gold" and r["d_clamp"] == clamp_label(CLAMPS[0])


def largest_ratio_row(rows):
    """Largest primary ratio over gold-thread, default-clamp rows of REAL dev cells (source == GENERATOR_VERSION);
    None when there are none."""
    real = [r for r in rows if is_primary(r) and r["source"] == GENERATOR_VERSION]
    return max(real, key=lambda r: r[PRIMARY_RATIO]) if real else None


def write_outputs(rows, out, inputs=(), synthetic=None):
    """CSV in opa.analysis.tables format plus sidecar <out>.json (script is bundle-relative, per BUNDLE_SPEC)."""
    out = Path(out)
    write_table(out, rows, COLUMNS)
    side = {"script": SCRIPT_REL, "inputs": list(inputs),
            "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "table_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    side["timing_dist_sec"] = {r["key"]: round(r["dist_sec"], 4) for r in rows if "dist_sec" in r}   # not in the CSV
    if synthetic:
        side["synthetic"] = synthetic
    side_path = out.with_suffix(".json")
    side_path.write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return side_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="bundle/tables/h5_distortion_dev.csv")
    ap.add_argument("--cells-dir", action="append", default=None,
                    help="cell directory with items.jsonl.gz (repeatable); default: every cell of "
                         "<local_data>/mtlbmhqa/dev/<GENERATOR_VERSION> if present")
    ap.add_argument("--synthetic", action="store_true", help="add synthetic dev cells (default if no real cells)")
    a = ap.parse_args(argv)
    dirs, synth = a.cells_dir, a.synthetic
    if dirs is None:
        vdir = DATA_ROOT / GENERATOR_VERSION
        dirs = sorted(p for p in vdir.glob("*") if (p / "items.jsonl.gz").is_file()) if vdir.is_dir() else []
        synth = synth or not dirs
    rows, inputs = run(dirs, synth)
    write_outputs(rows, a.out, inputs, synthetic_provenance() if synth else None)
    print(f"{'key':44s} {'T':>2s} {'h':>1s} {'len':>6s} {'thr':>8s} {'clamp':>5s} {'ratio_q':>8s} {'ratio_all':>9s} "
          f"{'sat_q':>6s}")
    for r in rows:
        print(f"{r['key']:44s} {r['T'] or 0:2d} {r['h'] or 0:1d} {r['length_tokens'] or 0:6d} {r['threads']:>8s} "
              f"{r['d_clamp']:>5s} {r[PRIMARY_RATIO]:8.3f} {r['ratio_allpairs']:9.3f} {r['q_clamp_frac']:8.3f}")
    best = largest_ratio_row(rows)
    if best is not None:
        print(f"largest primary ratio (gold, d_clamp none, real {GENERATOR_VERSION} cells): "
              f"{best['key']} = {best[PRIMARY_RATIO]:.3f}")
        if all(r[PRIMARY_RATIO] < 1.2 for r in rows if is_primary(r) and r["source"] == GENERATOR_VERSION):
            print("ratio < 1.2 in EVERY real cell: H5 stays, write the expected null in the prereg (PLAN_V3 Sec 8).")
    else:
        print(f"no real {GENERATOR_VERSION} cells in this run: largest primary ratio per source (gold, d_clamp none):")
        for s in sorted({r["source"] for r in rows}):
            sub = [r for r in rows if is_primary(r) and r["source"] == s]
            if sub:
                b = max(sub, key=lambda r: r[PRIMARY_RATIO])
                print(f"  {s}: {b['key']} = {b[PRIMARY_RATIO]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
