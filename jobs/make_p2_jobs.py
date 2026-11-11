"""P2 test-split job TEMPLATES (Track A: H1, H3's S11 - S8 leaf, H5 agent arms, H2, H10). Run: python jobs/make_p2_jobs.py [--package owner|96|126|full|none] [--sizes FILE]   (default: the owner's package, H2 deferred)

TEMPLATES ONLY. No test data is generated or read here, and nothing is launched: the launcher refuses test jobs before the prereg-v2 tag. Templates go to
jobs/p2/<name>.yaml and name test paths gs://{bucket}/opa/data/<ds>/test/<version>/<cell>/items.jsonl.gz, which exist only after P2.1 (the version is the
generator version, `v2`, like the dev cells; the brief said v1). Run ids of the prep jobs fill the placeholders P2_<KIND>_<CELL>_C<nn>_RUN_ID
(--ids JSON merged into jobs/launch/run_ids.json; filled specs go to jobs/launch/). All jobs: split test, seed 100000 + 100 x series + container index (distinct across series; test seeds start at
100000; temperature 0), L4, max_api_usd 0, hypotheses of the line, `notes` naming the package line.

SAMPLE SIZES come from one input (--sizes JSON/YAML with H1, S4, H5, tost, H2, H10; prereg/sample_sizes.yaml once it exists and is flattened to these keys);
the default is the "fits $96" package: H1 1191, S4 391, H5 249, TOST 191, H2 200, H10 860. --package drops what the package CSV
(bundle/tables/plan_package_<p>.csv) does not include, takes n from its alternative label (H1 607 at delta 7, ...), and funds BOOST items from BOOST: the
first containers of an item whose cumulative estimate fits the item's BOOST dollars carry envelope BOOST, the rest E2 (raise-funded money is also E2).

CONTAINERS. Every job is split into containers whose estimate is <= $2.50 (item offset ranges that partition [0, n) exactly, no overlap), priced with the plan_costs
unit costs (S2/S8/S11/S4 at 16 workers, S4 34.06 s/item; the MT arms at workers-8 seconds, skeletons at 24 workers net $0.0246 + $0.066 start-up, embed $0.0018 per item,
drift and thrash per session / pair). The prep jobs (embed, skeletons) and the runs share one partition, so a run container reads exactly one vector file and one
skeleton directory. max_usd = 2 x estimate.
  (1) P2.1 test-data generation is LOCAL CPU work after the tag (see datagen_commands): the generator + create-only upload.
  (2) embed + sllm-v2 skeleton jobs (workers 24) for the LB late-h2 32K and MT T4 h3 cross test cells.
  (3) H1 runs: [S11, S2, S8] at workers 16 on every container; S4 at workers 16 on the first n_S4 items.
  (4) H5 arms [S12, S14] on n_H5 items and S15 on the first n_tost (the partition has an edge at n_tost), workers 8.
  (5) H2 drift D2 and D4 on n_H2 sessions.  (6) H10 thrash pin / nopin (prompt version thrash-v2) on r3, r4, r6 (n_H10 items split over the three cells), M = 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_p0_jobs as p0  # noqa: E402
import make_p1_jobs as p1  # noqa: E402
from opa import launch  # noqa: E402
from opa.analysis import plan_costs as pc  # noqa: E402
from opa.analysis import plan_packages as pk  # noqa: E402
from opa.data import GENERATOR_VERSION, TEST_SEED_MIN  # noqa: E402

OUT_DIR = HERE / "p2"
TEST_VERSION = GENERATOR_VERSION
BUCKET = p0.BUCKET
DEFAULT_SIZES = {"H1": 1191, "S4": 391, "H5": 249, "tost": 191, "H2": 200, "H10": 860}
CELLS = {"lb": ("lbmhqa", "late-h2-32k-a0"), "mt": ("mtlbmhqa", "T4-h3-32k-cross"), "drift": ("drift", "s50"),
         "r3": ("thrash", "r3"), "r4": ("thrash", "r4"), "r6": ("thrash", "r6")}
ITEM_PREFIX = {"H1": "H1 ", "H5": "H5 agent arms", "H2": "H2 drift", "H10": "H10 thrash"}
MODEL = {"kind": "open_local", "name": "qwen", "max_model_len": 40960}
MAX_CONTAINER_USD = 2.5                      # container estimate cap (extra start-ups priced); the owner package uses $2.50
START = pc.PARAMS["start_usd"]


def cell_uri(key: str) -> str:
    ds, cell = CELLS[key]
    return f"gs://{BUCKET}/opa/data/{ds}/test/{TEST_VERSION}/{cell}/items.jsonl.gz"


def ph(kind: str, cell: str, idx: int) -> str:
    return f"P2_{kind.upper()}_{cell.upper()}_C{idx:02d}_RUN_ID"


# ------------------------------------------------------------------------------------------------ sizes and package
def load_sizes(path: str | None) -> dict:
    sizes = dict(DEFAULT_SIZES)
    if path:
        text = Path(path).read_text(encoding="utf-8")
        sizes.update({k: int(v) for k, v in (yaml.safe_load(text) or {}).items() if k in DEFAULT_SIZES})
    return sizes


def load_package(name: str, csv_path: Path | None = None) -> dict:
    """{key: {"n": parsed n or None, "usd", "base", "boost", "raise", "item", "alt"}} for the P2 items the package includes."""
    f = csv_path or (HERE.parent / "bundle" / "tables" / f"plan_package_{name}.csv")
    out: dict = {}
    for r in csv.DictReader(f.open(encoding="utf-8", newline="")):
        if "line_usd" in r:                                   # the owner package CSV: every row is an included line
            r = {"status": "included", "item": r["item"], "alt": r["alt"], "usd": r["line_usd"], "base_usd": r["base_usd"], "boost_usd": r["boost_usd"], "raise_usd": 0}
        if r["status"] not in ("included", "reduced"):
            continue
        for key, pre in ITEM_PREFIX.items():
            if r["item"].startswith(pre):
                d = out.setdefault(key, {"item": r["item"], "alt": r["alt"], "usd": 0.0, "base": 0.0, "boost": 0.0, "raise": 0.0, "status": r["status"]})
                d["usd"] += float(r["usd"] or 0)
                d["base"] += float(r["base_usd"] or 0)
                d["boost"] += float(r["boost_usd"] or 0)
                d["raise"] += float(r["raise_usd"] or 0)
    for d in out.values():
        m = re.search(r"n=([\d,]+)", d["alt"])
        d["n"] = int(m.group(1).replace(",", "")) if m else None
    return out


# ------------------------------------------------------------------------------------------------ partitions
def balanced(n: int, max_items: int) -> list[tuple[int, int]]:
    """Partition [0, n) into the fewest balanced (offset, count) ranges of at most max_items items."""
    if n <= 0:
        return []
    k = math.ceil(n / max_items)
    base, extra = divmod(n, k)
    out, off = [], 0
    for i in range(k):
        c = base + (1 if i < extra else 0)
        out.append((off, c))
        off += c
    return out


def split_at(ranges: list[tuple[int, int]], edges: list[int]) -> list[tuple[int, int]]:
    """Cut ranges at extra edges (e.g. n_tost) without changing their union."""
    out = []
    for off, c in ranges:
        cuts = [off] + sorted(e for e in edges if off < e < off + c) + [off + c]
        out += [(a, b - a) for a, b in zip(cuts, cuts[1:])]
    return out


def max_items(per_item_usd: float, fixed: float = START) -> int:
    return max(1, int((MAX_CONTAINER_USD - fixed) / per_item_usd))


# ------------------------------------------------------------------------------------------------ unit costs
def unit_costs():
    x = pk.context(24)
    rate = pc.RATE
    lb_h1 = sum(x.lb[s] for s in ("S11", "S2", "S8")) * rate
    return {"x": x, "embed": x.em, "skeleton": x.sk, "h1": lb_h1, "s4": x.lb["S4"] * rate, "h5": (x.mt["S12"] + x.mt["S14"]) * rate, "s15": x.mt["S15"] * rate,
            "d2": pc.PARAMS["drift_usd_per_session"]["D2"], "d4": pc.PARAMS["drift_usd_per_session"]["D4"], "thrash": pc.PARAMS["thrash_usd_per_pair"] / 2.0}


# ------------------------------------------------------------------------------------------------ spec builders
SERIES = ["p2-embed-lb", "p2-sk-lb", "p2-h1-lb", "p2-h1-s4", "p2-embed-mt", "p2-sk-mt", "p2-h5-mt", "p2-h2-d2", "p2-h2-d4", "p2-h10-pin", "p2-h10-nopin"]


def series_seed(name: str) -> int:
    """Distinct seeds across series: 100000 + 100 x series index + container index - 1 (test seeds start at 100000; temperature 0 anyway)."""
    kind, idx = name.rsplit("-c", 1)
    return TEST_SEED_MIN + 100 * SERIES.index(kind) + int(idx) - 1


def _spec(name, track, system, dataset, cell, entrypoint, args, models, compute, hyp, seed, usd, work, items, notes):
    seed = series_seed(name)
    spec = p0.base(name, track, system, dataset, cell, entrypoint, args, models, compute, hyp=hyp, seed=seed)
    spec["split"] = "test"
    spec, meta = p1.finish_usd(spec, usd, work, items=items, notes=notes, suffix=f"; max_usd = 2 x the estimate = ${2 * usd:.3f}")
    meta["est_raw"] = usd
    return spec, meta


def entry(key: str, off: int, n: int) -> dict:
    return {"uri": cell_uri(key), "n_items": n, "offset": off}


def effective_sizes(sizes: dict, package: dict | None) -> dict:
    """The sizes actually planned: the package's alternative label sets H1 / H5 (e.g. n=607 at delta 7) unless --sizes overrides them."""
    n = dict(sizes)
    if package:
        for k in ("H1", "H5"):
            if k in package and package[k]["n"]:
                n[k] = package[k]["n"] if sizes.get(k) == DEFAULT_SIZES[k] else sizes[k]
    return n


def plan(sizes: dict, package: dict | None):
    """Entries [(spec, meta)] in launch order. package None = everything in `sizes`; else only the items the package includes (n from its label)."""
    u = unit_costs()
    keys = ["H1", "H5", "H2", "H10"] if package is None else [k for k in ("H1", "H5", "H2", "H10") if k in package]
    n = effective_sizes(sizes, package)
    entries: list[tuple[dict, dict]] = []
    line = lambda k: (f"package line: {package[k]['item']} [{package[k]['alt']}]" if package else f"all items, no package (sizes input), item {k}")

    def add(key, spec_meta, order_note=""):
        s, m = spec_meta
        m["item_key"] = key
        s["notes"].append(line(key))
        entries.append((s, m))

    if "H1" in keys:
        parts = balanced(n["H1"], min(max_items(u["embed"] * 1, 0.0), max_items(u["skeleton"]), max_items(u["h1"])))
        for i, (off, c) in enumerate(parts, 1):
            tag = f"c{i:02d}"
            add("H1", _spec(f"p2-embed-lb-{tag}", "A", "embed", "lbmhqa", CELLS["lb"][1], "opa.tracks.track_a:embed_items",
                            {"items_uri": [entry("lb", off, c)], "embedder": "harrier", "B": 512, "batch": 16}, {"embedder": p0.EMBEDDER},
                            p0.gpu_compute("L4", 12), ("H1",), TEST_SEED_MIN + i - 1, c * u["embed"], f"{c} items x ${u['embed']}/item (measured)", c,
                            [f"items {off}-{off + c - 1} of the LB late-h2 test cell"]))
        for i, (off, c) in enumerate(parts, 1):
            tag = f"c{i:02d}"
            add("H1", _spec(f"p2-sk-lb-{tag}", "A", "sllm", "lbmhqa", CELLS["lb"][1], "opa.tracks.track_a:build_skeletons",
                            {"items_uri": [entry("lb", off, c)], "workers": 24, "model": dict(MODEL)}, {"agent_model": p0.QWEN}, p0.vllm_compute(),
                            ("H1",), TEST_SEED_MIN + i - 1, c * u["skeleton"] + START,
                            f"{c} items x ${u['skeleton']:.4f}/item (sllm-v2, 24 workers, net) + ${START} start-up", c,
                            [f"items {off}-{off + c - 1}; sllm-v2 rebuilt builder"]))
        for i, (off, c) in enumerate(parts, 1):
            tag = f"c{i:02d}"
            vec = f"precomputed:gs://{BUCKET}/opa/runs/{ph('embed', 'lb', i)}/embeddings.npz"
            skd = f"gs://{BUCKET}/opa/runs/{ph('sk', 'lb', i)}/skeleton_llm"
            add("H1", _spec(f"p2-h1-lb-{tag}", "A", "S11+S2+S8", "lbmhqa", CELLS["lb"][1], "opa.tracks.track_a:run_system",
                            {"items_uri": [entry("lb", off, c)], "systems": ["S11", "S2", "S8"], "embedder": vec, "workers": 16, "model": dict(MODEL),
                             "opts": {"lambda": 2.0}, "skeleton_llm_uri": skd}, {"agent_model": p0.QWEN}, p0.vllm_compute(16), ("H1", "H3"),
                            TEST_SEED_MIN + i - 1, c * u["h1"] + START,
                            f"{c} items x (S11 {u['x'].lb['S11']} + S2 {u['x'].lb['S2']} + S8 {u['x'].lb['S8']} s at 16 workers) x ${pc.RATE:.6f}/s + ${START} start-up", 3 * c,
                            [f"items {off}-{off + c - 1}; S11 - S2 (H1) and S11 - S8 (H3 leaf) on the same items"]))
        s4_ranges = []
        for off, c in split_at(parts, [n["S4"]]):
            if off >= n["S4"]:
                continue
            s4_ranges += [(off + o2, c2) for o2, c2 in balanced(c, max_items(u["s4"]))]
        for j, (off, c) in enumerate(s4_ranges, 1):
            i = next(k for k, (po, pc_) in enumerate(parts, 1) if po <= off < po + pc_)
            vec = f"precomputed:gs://{BUCKET}/opa/runs/{ph('embed', 'lb', i)}/embeddings.npz"
            add("H1", _spec(f"p2-h1-s4-c{j:02d}", "A", "S4", "lbmhqa", CELLS["lb"][1], "opa.tracks.track_a:run_system",
                            {"items_uri": [entry("lb", off, c)], "systems": ["S4"], "embedder": vec, "workers": 16, "model": dict(MODEL), "opts": {"lambda": 2.0}},
                            {"agent_model": p0.QWEN}, p0.vllm_compute(16), ("H1",), TEST_SEED_MIN + 100 + j - 1, c * u["s4"] + START,
                            f"{c} items x S4 {u['x'].lb['S4']:.1f} s (workers-16) x ${pc.RATE:.6f}/s + ${START} start-up", c,
                            [f"items {off}-{off + c - 1} of the first n_S4={n['S4']}; S4 at workers 16"]))
    if "H5" in keys:
        n5, nt = n["H5"], min(n["tost"], n["H5"])
        parts = balanced(n5, min(max_items(u["skeleton"]), max_items(u["h5"] + u["s15"])))
        for i, (off, c) in enumerate(parts, 1):
            tag = f"c{i:02d}"
            add("H5", _spec(f"p2-embed-mt-{tag}", "A", "embed", "mtlbmhqa", CELLS["mt"][1], "opa.tracks.track_a:embed_items",
                            {"items_uri": [entry("mt", off, c)], "embedder": "harrier", "B": 512, "batch": 16}, {"embedder": p0.EMBEDDER},
                            p0.gpu_compute("L4", 12), ("H5",), TEST_SEED_MIN + i - 1, c * u["embed"], f"{c} items x ${u['embed']}/item (measured)", c,
                            [f"items {off}-{off + c - 1} of the MT T4 h3 cross test cell"]))
        for i, (off, c) in enumerate(parts, 1):
            tag = f"c{i:02d}"
            add("H5", _spec(f"p2-sk-mt-{tag}", "A", "sllm", "mtlbmhqa", CELLS["mt"][1], "opa.tracks.track_a:build_skeletons",
                            {"items_uri": [entry("mt", off, c)], "workers": 24, "model": dict(MODEL)}, {"agent_model": p0.QWEN}, p0.vllm_compute(),
                            ("H5",), TEST_SEED_MIN + i - 1, c * u["skeleton"] + START,
                            f"{c} items x ${u['skeleton']:.4f}/item (sllm-v2, 24 workers, net) + ${START} start-up", c, [f"items {off}-{off + c - 1}; sllm-v2 rebuilt builder"]))
        for j, (off, c) in enumerate(split_at(parts, [nt]), 1):
            i = next(k for k, (po, pc_) in enumerate(parts, 1) if po <= off < po + pc_)
            with_s15 = off + c <= nt
            systems = ["S12", "S14"] + (["S15"] if with_s15 else [])
            per = u["h5"] + (u["s15"] if with_s15 else 0.0)
            vec = f"precomputed:gs://{BUCKET}/opa/runs/{ph('embed', 'mt', i)}/embeddings.npz"
            skd = f"gs://{BUCKET}/opa/runs/{ph('sk', 'mt', i)}/skeleton_llm"
            add("H5", _spec(f"p2-h5-mt-c{j:02d}", "A", "+".join(systems), "mtlbmhqa", CELLS["mt"][1], "opa.tracks.track_a:run_system",
                            {"items_uri": [entry("mt", off, c)], "systems": systems, "embedder": vec, "workers": 8, "model": dict(MODEL),
                             "opts": {"lambda": 2.0}, "skeleton_llm_uri": skd}, {"agent_model": p0.QWEN}, p0.vllm_compute(16), ("H5",),
                            TEST_SEED_MIN + 200 + j - 1, c * per + START,
                            f"{c} items x ({'+'.join(systems)} at workers-8 seconds) x ${pc.RATE:.6f}/s + ${START} start-up", c * len(systems),
                            [f"items {off}-{off + c - 1}; S15 only on the first n_tost={nt} items" + ("" if with_s15 else " (this range is past n_tost)")]))
    if "H2" in keys:
        for cond, key in (("D2", "d2"), ("D4", "d4")):
            for j, (off, c) in enumerate(balanced(n["H2"], max_items(u[key])), 1):
                add("H2", _spec(f"p2-h2-{cond.lower()}-c{j:02d}", "A2", cond, "drift", CELLS["drift"][1], "opa.tracks.track_a2:run_condition",
                                {"items_uri": [entry("drift", off, c)], "condition": cond, "model": {"kind": "open_local", "name": "qwen"}, "workers": min(30, c), "opts": {}},
                                {"subject": p0.QWEN}, p0.vllm_compute(16), ("H2",), TEST_SEED_MIN + j - 1, c * u[key] + START,
                                f"{c} sessions x ${u[key]:.4f}/session ({cond}, measured) + ${START} start-up", c, [f"sessions {off}-{off + c - 1} of the drift s50 test cell"]))
    if "H10" in keys:
        per_cell = balanced(n["H10"], max(1, math.ceil(n["H10"] / 3)))                 # r3, r4, r6: the 860 items split over three cells
        cells = [("r3", per_cell[0][1] if len(per_cell) > 0 else 0), ("r4", per_cell[1][1] if len(per_cell) > 1 else 0), ("r6", per_cell[2][1] if len(per_cell) > 2 else 0)]
        for arm, pin in (("pin", True), ("nopin", False)):
            for j, grp in enumerate(_thrash_groups(cells, u["thrash"]), 1):
                items = sum(c for _, _, c in grp)
                add("H10", _spec(f"p2-h10-{arm}-c{j:02d}", "A", f"thrash-{arm}", "thrash", "+".join(k for k, _, _ in grp), "opa.tracks.track_a:run_thrash",
                                 {"items_uri": [entry(k, off, c) for k, off, c in grp], "M": 2, "pinning": pin, "workers": 8, "model": dict(MODEL)},
                                 {"agent_model": p0.QWEN}, p0.vllm_compute(16), ("H10",), TEST_SEED_MIN + j - 1, items * u["thrash"] + START,
                                 f"{items} items x ${u['thrash']:.5f} per item-arm (pair cost / 2, measured) + ${START} start-up", items,
                                 ["thrash prompt version thrash-v2 (opa.tracks.track_a.thrash.PROMPT_VERSION); M = 2; cells " + ", ".join(f"{k}[{o}:{o + c}]" for k, o, c in grp)]))
    assign_envelopes(entries, package)
    return entries


def _thrash_groups(cells, per_item):
    """Group (cell, offset, count) ranges into containers of estimate <= $3, splitting a cell when needed."""
    cap = max_items(per_item)
    groups, cur, used = [], [], 0
    for k, c in cells:
        off = 0
        while c > 0:
            take = min(c, cap - used)
            cur.append((k, off, take))
            off, c, used = off + take, c - take, used + take
            if used >= cap:
                groups.append(cur)
                cur, used = [], 0
    if cur:
        groups.append(cur)
    return groups


def assign_envelopes(entries, package):
    """E2 for everything; with a package, the first containers of an item (launch order) whose cumulative estimate fits its BOOST dollars carry BOOST."""
    cum: dict[str, float] = {}
    for s, m in entries:
        key = m["item_key"]
        boost = package[key]["boost"] if package else 0.0
        cum[key] = cum.get(key, 0.0) + m["est"]
        env = "BOOST" if boost > 0 and cum[key] <= boost + 1e-9 else "E2"
        s["envelope"] = m["envelope"] = env
        if env == "BOOST":
            s["notes"].append("BOOST-funded container (PLAN_V3 s9 use (b): more test items for the H1, H5 and H12 primary cells)")


# ------------------------------------------------------------------------------------------------ local data generation (after the tag)
def datagen_commands(sizes: dict, keys=("H1", "H5", "H2", "H10")) -> list[str]:
    """P2.1: LOCAL CPU commands, to be run only after the prereg-v2 tag (the generator and the uploader refuse earlier). Test seeds start at 100000."""
    cmds = []
    spec = {"H1": [("lbmhqa", CELLS["lb"][1], sizes["H1"])], "H5": [("mtlbmhqa", CELLS["mt"][1], sizes["H5"])], "H2": [("drift", "s50", sizes["H2"])],
            "H10": [("thrash", c, n) for c, n in zip(("r3", "r4", "r6"), (x[1] for x in balanced(sizes["H10"], max(1, math.ceil(sizes["H10"] / 3)))))]}
    for k in keys:
        for ds, cell, n in spec[k]:
            cmds.append(f"python -m opa.data.build --dataset {ds} --split test --cell {cell} --n {n} --seed-start {TEST_SEED_MIN}")
    for ds in dict.fromkeys(ds for k in keys for ds, _, _ in spec[k]):
        cmds.append(f"python -m opa.data.upload --dataset {ds} --split test --version {TEST_VERSION}")
    return cmds


# ------------------------------------------------------------------------------------------------ output
def dump(spec: dict, subs: dict[str, str]) -> bytes:
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000)
    for k in sorted(subs, key=len, reverse=True):
        if subs[k]:
            text = text.replace(k, subs[k])
    return text.encode("utf-8")


def placeholders(entries) -> set[str]:
    return {m for s, _ in entries for m in re.findall(r"\bP2_[A-Z0-9_]*_RUN_ID\b", yaml.safe_dump(s))}


def filled(spec: dict) -> dict:
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000)
    return yaml.safe_load(re.sub(r"\bP2_[A-Z0-9_]*_RUN_ID\b", "dummy-run", text))


def write_launch(entries, ids: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "run_ids.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    known.update({k: v for k, v in ids.items() if v})
    f.write_bytes((json.dumps(known, indent=1, sort_keys=True) + "\n").encode())
    written, skipped = [], {}
    for spec, _ in entries:
        data = dump(spec, known)
        left = p0.unresolved(data.decode())
        if left:
            skipped[spec["name"]] = left
            continue
        (out_dir / f"{spec['name']}.yaml").write_bytes(data)
        written.append(spec["name"])
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="owner", choices=("owner", "96", "126", "full", "none"))
    ap.add_argument("--sizes", help="JSON/YAML with H1, S4, H5, tost, H2, H10")
    ap.add_argument("--ids", help="JSON of run ids to merge into jobs/launch/run_ids.json and fill the launch specs")
    a = ap.parse_args(argv)
    sizes = load_sizes(a.sizes)
    package = None if a.package == "none" else load_package(a.package)
    entries = plan(sizes, package)
    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("p2-*.yaml"):
        old.unlink()
    for spec, _ in entries:
        errs = launch.validate(filled(spec))
        assert not errs, (spec["name"], errs)
        (OUT_DIR / f"{spec['name']}.yaml").write_bytes(dump(spec, {}))
    if a.ids:
        written, skipped = write_launch(entries, json.loads(Path(a.ids).read_text()), HERE / "launch")
        print(f"jobs/launch: {len(written)} written; refused (unresolved placeholders): {len(skipped)}")
    print(f"P2 templates for package {a.package}: {len(entries)} jobs in {OUT_DIR}")
    print(f"{'job':<22}{'env':>6}{'items':>7}{'est $':>8}{'max $':>8}  item")
    for s, m in entries:
        print(f"{m['name']:<22}{m['envelope']:>6}{m['items']:>7}{m['est']:>8.2f}{m['max']:>8.2f}  {m['item_key']}")
    for k in sorted({m['item_key'] for _, m in entries}):
        ms = [m for _, m in entries if m['item_key'] == k]
        print(f"  {k}: {len(ms)} containers, estimate ${sum(m['est'] for m in ms):.2f}, max ${sum(m['max'] for m in ms):.2f}, largest container est ${max(m['est'] for m in ms):.2f}"
              + (f", package line ${package[k]['usd']:.2f} (base ${package[k]['base']:.2f} + BOOST ${package[k]['boost']:.2f})" if package else ""))
    print(f"TOTAL estimate ${sum(m['est'] for _, m in entries):.2f}, max ${sum(m['max'] for _, m in entries):.2f}")
    print("P2.1 local data generation (run only after the prereg-v2 tag):")
    for c in datagen_commands(effective_sizes(sizes, package), tuple(k for k in ("H1", "H5", "H2", "H10") if package is None or k in package)):
        print("  " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
