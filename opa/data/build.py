"""Write one dataset cell locally.

  python -m opa.data.build --dataset lbmhqa --split dev --cell late-h3-32k-a0 --n 100
  python -m opa.data.build --dataset mtlbmhqa --split dev --cell T8-h3-32k-cross --n 100
  python -m opa.data.build --dataset mtlbmhqa --split dev --cell T8-h3-32k-cross-conf --n 100   # thread confusion (H12)
  python -m opa.data.build --dataset drift --split dev --cell s50 --n 200
  python -m opa.data.build --dataset thrash --split dev --cell r4 --n 60
  python -m opa.data.build --preset v3-dev --split dev            # every PLAN_V3 dev cell (see v3_dev_cells)

Output: <local_data>/<dataset>/<split>/<version>/<cell>/items.jsonl.gz and manifest.json. Every item is validated and
its token length must be within 2% of the cell's target. Context lengths, B (origin_tokens_B) and the tokenizer default
to config/project.yaml (defaults.context_lengths, defaults.origin_tokens_B, paths.tokenizer); --block-tokens,
--tokenizer and --any-length override them. Seeds are seed_start .. seed_start + n - 1 (dev default 0, test default
100000; test needs the prereg-v2 tag).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import platform
import subprocess
import sys
import time
from pathlib import Path

from opa import PROJECT_DIR
from opa.data import (GENERATOR_VERSION, LENGTH_TOLERANCE, TEST_SEED_MIN, check_length, check_split, data_config,
                      default_block_tokens)
from opa.data import drift, lbmhqa, mtlbmhqa, thrash
from opa.data.filler import PG19Filler, SyntheticFiller, WhitespaceTokenizer, load_tokenizer, local_data_dir
from opa.data.io import write_items, write_manifest
from opa.data.schema import validate_item

DATASETS = ("lbmhqa", "mtlbmhqa", "drift", "thrash")


def parse_cell(dataset: str, cell: str) -> dict:
    if dataset == "lbmhqa":
        p = lbmhqa.parse_cell(cell)
        assert lbmhqa.cell_name(p["placement"], p["hops"], p["length_tokens"], p["alpha"]) == cell, "non-canonical cell"
        return p
    if dataset == "mtlbmhqa":
        p = mtlbmhqa.parse_cell(cell)
        assert mtlbmhqa.cell_name(p["n_threads"], p["hops"], p["length_tokens"], p["cross_thread"], p["alpha"],
                                  p["decoys"]) == cell
        return p
    if dataset == "thrash":
        return thrash.parse_cell(cell)
    if dataset == "drift":
        if cell != "s50":
            raise ValueError("drift has one cell, s50 (protocol 4.1: 50-turn sessions)")
        return {"n_turns": 50}
    raise ValueError(f"unknown dataset {dataset!r}")


def make_one(dataset: str, split: str, params: dict, seed: int, filler, tok, vocab=None, block_tokens=None) -> dict:
    if dataset == "lbmhqa":
        return lbmhqa.make_item(seed, params["hops"], params["length_tokens"], params["placement"], params["alpha"],
                                filler, tok, split=split, block_tokens=block_tokens)
    if dataset == "mtlbmhqa":
        return mtlbmhqa.make_item(seed, params["n_threads"], params["hops"], params["length_tokens"],
                                  params["cross_thread"], params["alpha"], filler, tok, split=split,
                                  block_tokens=block_tokens, decoys=params["decoys"])
    if dataset == "thrash":
        return thrash.make_item(seed, params["r"], filler, tok, split=split, block_tokens=block_tokens)
    if dataset == "drift":
        return drift.make_session(seed, split=split, n_turns=params["n_turns"], vocab=vocab)
    raise ValueError(dataset)


def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()
    return {"git_sha": g("rev-parse", "HEAD"), "generator_code_dirty": bool(g("status", "--porcelain", "--", "opa/data"))}


def item_tokens(it: dict) -> int:
    return sum(o["n_tokens"] for o in it["origins"])


def check_token_lengths(dataset: str, items: list[dict], tol: float = LENGTH_TOLERANCE) -> float:
    """QA items: total origin tokens must be within `tol` of params.length_tokens. Returns the worst relative error."""
    if dataset not in ("lbmhqa", "mtlbmhqa"):
        return 0.0
    worst = 0.0
    for it in items:
        err = abs(item_tokens(it) - it["params"]["length_tokens"]) / it["params"]["length_tokens"]
        worst = max(worst, err)
        if err > tol:
            raise ValueError(f"{it['item_id']}: {item_tokens(it)} tokens, target {it['params']['length_tokens']} "
                             f"({err:.2%} off, limit {tol:.0%})")
    return worst


def summarize(dataset: str, items: list[dict], tok=None) -> dict:
    out: dict = {"n_items": len(items)}
    if dataset in ("lbmhqa", "mtlbmhqa", "thrash"):
        out["n_origins"] = sorted({len(it["origins"]) for it in items})
        tot = [sum(o["n_tokens"] for o in it["origins"]) for it in items]
        per = [o["n_tokens"] for it in items for o in it["origins"]]
        out["tokens_per_item"] = {"min": min(tot), "mean": round(sum(tot) / len(tot), 1), "max": max(tot)}
        out["tokens_per_origin"] = {"min": min(per), "mean": round(sum(per) / len(per), 1), "max": max(per)}
    if dataset in ("lbmhqa", "mtlbmhqa"):
        out["length_tokens"] = items[0]["params"]["length_tokens"]
        out["max_abs_length_error_pct"] = round(100 * max(abs(item_tokens(it) - it["params"]["length_tokens"])
                                                          / it["params"]["length_tokens"] for it in items), 3)
        out["h_late"] = dict(sorted(collections.Counter(it["gold"]["h_late"] for it in items).items()))
        out["attribute"] = dict(sorted(collections.Counter(it["gold"]["attribute"] for it in items).items()))
        out["n_distractor_sentences"] = sum(len(it["gold"]["distractors"]) for it in items)
        out["n_alias_mentions"] = sum(len(v) for it in items for v in it["gold"]["aliases"].values())
    if dataset == "mtlbmhqa":
        out["cross_thread_hop"] = dict(collections.Counter(str(it["gold"]["cross_thread_hop"]) for it in items))
        out["confusers_per_item"] = sorted({len(it["gold"]["confusers"]) for it in items})
        out["question_thread"] = dict(sorted(collections.Counter(it["question_thread"] for it in items).items()))
        if items[0]["params"].get("decoys"):
            out["decoy_kinds"] = dict(sorted(collections.Counter(c["kind"] for it in items
                                                                 for c in it["gold"]["confusers"]).items()))
    if dataset == "thrash":
        out["needed_origins_per_item"] = sorted({len(it["gold"]["needed_origins"]) for it in items})
    if dataset == "drift":
        if tok is not None:
            tot = [sum(tok.count_many([t["text"] for t in it["turns"]])) for it in items]
            out["tokens_per_item"] = {"min": min(tot), "mean": round(sum(tot) / len(tot), 1), "max": max(tot)}
        out["n_turns"] = sorted({len(it["turns"]) for it in items})
        out["updates_per_session"] = sorted({sum(len(c["updates"]) for c in it["constraints"]) for it in items})
        out["near_misses_per_session"] = sorted({sum(len(c["near_misses"]) for c in it["constraints"]) for it in items})
        out["constraint_keys"] = dict(sorted(collections.Counter(c["key"] for it in items
                                                                 for c in it["constraints"]).items()))
    return out


def build(dataset: str, split: str, cell: str, n: int, seed_start: int | None = None, out_root: str | Path | None = None,
          version: str = GENERATOR_VERSION, synthetic: bool = False, force: bool = False, quiet: bool = False,
          block_tokens: int | None = None, tokenizer: str | None = None, any_length: bool = False) -> dict:
    params = parse_cell(dataset, cell)
    if "length_tokens" in params and not any_length:
        check_length(params["length_tokens"])           # config defaults.context_lengths
    block_tokens = default_block_tokens() if block_tokens is None else block_tokens
    seed_start = (0 if split == "dev" else TEST_SEED_MIN) if seed_start is None else seed_start
    seeds = list(range(seed_start, seed_start + n))
    for s in seeds:
        check_split(split, s)                  # refuses test before the prereg-v2 tag
    out_dir = Path(out_root or local_data_dir()) / dataset / split / version / cell
    if (out_dir / "items.jsonl.gz").exists() and not force:
        raise FileExistsError(f"{out_dir / 'items.jsonl.gz'} exists (use --force to rebuild)")

    if synthetic:
        filler, tok = SyntheticFiller(), WhitespaceTokenizer()
    else:
        filler = PG19Filler("validation" if split == "dev" else "test")
        tok = load_tokenizer(tokenizer)               # default: paths.tokenizer from config/project.yaml
    vocab = filler.vocab() if dataset == "drift" else None

    t0 = time.time()
    items = []
    for s in seeds:
        it = make_one(dataset, split, params, s, filler if dataset != "drift" else None, tok, vocab, block_tokens)
        validate_item(it)
        items.append(it)
        if not quiet and (len(items) % 10 == 0 or len(items) == n):
            print(f"  {dataset}/{cell}: {len(items)}/{n} items, {time.time() - t0:.1f}s", flush=True)
    gen_s = time.time() - t0
    check_token_lengths(dataset, items)
    info = write_items(items, out_dir / "items.jsonl.gz")

    fsrc = filler.source()
    if dataset != "drift":
        used = sorted({o["book_id"] for it in items for o in it["origins"]})
        titles = {}
        for it in items:
            for o in it["origins"]:
                titles.setdefault(o["book_id"], o["doc_title"])
        fsrc["books_used"] = [{"book_id": b, "title": titles[b]} for b in used]
    else:
        fsrc = {"vocabulary_check_only": fsrc}
    manifest = {
        "dataset": dataset, "split": split, "version": version, "generator_version": GENERATOR_VERSION, "cell": cell,
        "params": params, "seeds": seeds, "seed_range": [seeds[0], seeds[-1]],
        "counts": summarize(dataset, items, tok),
        "items_file": "items.jsonl.gz", "items_sha256": info["sha256"], "items_bytes": info["bytes"],
        "filler": fsrc, "tokenizer": tok.name, "block_tokens": block_tokens,
        **git_info(),
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation_seconds": round(gen_s, 1),
        "python": platform.python_version(),
        "validated": True,
    }
    write_manifest(out_dir, manifest)
    if not quiet:
        print(f"wrote {info['path']} ({info['bytes']} bytes, sha256 {info['sha256'][:12]}) in {gen_s:.1f}s")
    return {"dir": str(out_dir), "manifest": manifest, "items": items}


def v3_dev_cells(lengths: list[int] | None = None) -> list[tuple[str, str, int]]:
    """(dataset, cell, n_items) for every PLAN_V3 dev cell. `lengths` defaults to config defaults.context_lengths; the
    largest is the primary length. LB-MHQA: forward/late/mixed at h = 3 and h = 2 at the primary length, late h = 3 at
    the other lengths. MT-LB-MHQA and the thread-confusion variant (-conf): T in {4, 8, 16}, no-cross and cross, h = 3, at
    the primary length. Thrash r2..r6 (32 origins = 16K tokens, fits the primary length), drift s50."""
    lengths = sorted(data_config()["context_lengths"] if lengths is None else lengths)
    top, lows = lengths[-1], lengths[:-1]
    cells: list[tuple[str, str, int]] = []
    for h in (3, 2):
        for pl in lbmhqa.PLACEMENTS:
            cells.append(("lbmhqa", lbmhqa.cell_name(pl, h, top, 0.0), 100))
    for low in lows:
        cells.append(("lbmhqa", lbmhqa.cell_name("late", 3, low, 0.0), 100))
    for decoys in (False, True):
        for t in (4, 8, 16):
            for cross in (False, True):
                cells.append(("mtlbmhqa", mtlbmhqa.cell_name(t, 3, top, cross, 0.0, decoys), 100))
    cells += [("thrash", thrash.cell_name(r), 60) for r in (2, 3, 4, 5, 6)]
    cells.append(("drift", "s50", 200))
    return cells


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opa.data.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--cell")
    ap.add_argument("--n", type=int)
    ap.add_argument("--preset", choices=("v3-dev",), help="build every PLAN_V3 dev cell (needs --split dev)")
    ap.add_argument("--block-tokens", type=int, default=None, help="default: defaults.origin_tokens_B in config")
    ap.add_argument("--tokenizer", default=None, help="default: paths.tokenizer in config")
    ap.add_argument("--any-length", action="store_true", help="allow a cell length outside defaults.context_lengths")
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--out-root", default=None, help="default: paths.local_data from config/project.yaml")
    ap.add_argument("--version", default=GENERATOR_VERSION)
    ap.add_argument("--synthetic", action="store_true", help="SyntheticFiller + whitespace tokenizer (testing only)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(block_tokens=a.block_tokens, tokenizer=a.tokenizer, any_length=a.any_length)
    if a.preset:
        if a.split != "dev":
            ap.error("--preset v3-dev builds dev cells only")
        rows = []
        for ds, cell, n in v3_dev_cells():
            t0 = time.time()
            r = build(ds, a.split, cell, a.n or n, a.seed_start, a.out_root, a.version, a.synthetic, a.force,
                      quiet=True, **kw)
            c = r["manifest"]["counts"]
            rows.append((ds, cell, c["n_items"], (c.get("tokens_per_item") or {}).get("mean"), time.time() - t0))
            print(f"{ds:9s} {cell:26s} n={c['n_items']:4d} mean_tokens={rows[-1][3]} build_s={rows[-1][4]:.1f}", flush=True)
        return 0
    if not (a.dataset and a.cell and a.n):
        ap.error("--dataset, --cell and --n are required without --preset")
    build(a.dataset, a.split, a.cell, a.n, a.seed_start, a.out_root, a.version, a.synthetic, a.force, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
