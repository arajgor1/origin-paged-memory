"""Write the dev-only job specs for the two cheapest candidate H3 cells (PLAN_V3 revision 3, Section 6b). Run: python jobs/make_h3cells_jobs.py

Each cell uses 30 FRESH dev items, offset 50 (items 50-79; the P1 pilot used items 0-49 only):
  (a) LB-MHQA late h2 32K a0 with the resident-page budget halved: job-level opts M = 2 instead of the default 4 (Fold.M = opts["M"],
      opa/tracks/track_a/systems.py). One container with systems [S11, S18]: the Section 6b oracle check (S18 >= 60%) must hold under the
      same condition as the cell, so S18 also runs at M = 2 (ruling). Systems are ordered [S18, S11] so the oracle check finishes first.
      M = 2 also reduces S11's read window from 3 origins to 1 (one slot kept free for faults, per design: read_window = max(1, M - 1)):
      the cell changes how S11 reads, not only residency, so S11 and S18 make about 2-2.4x the calls and are priced higher.
  (b) LB-MHQA late h3 32K a0 with the default M: one container with systems [S11, S18].
Preparation: one harrier embed job for items 50-79 of both cells (one npz), and one sllm-v2 skeleton job per cell (workers 24;
a job reads one skeleton directory). Run ids fill the placeholders H3CELLS_VEC_RUN_ID, H3CELLS_SK2_RUN_ID, H3CELLS_SK3_RUN_ID
(--vec-run-id, --sk2-run-id, --sk3-run-id; kept in jobs/launch/run_ids.json; filled specs go to jobs/launch/). The opts are the pilot's
(lambda 2.0) plus M where (a) says. All jobs: dev split, L4 vLLM image (embed: L4 gpu image), seed 0, max_api_usd 0, hypotheses ["H3"].

FROZEN PRICES (measured unit costs): S11 $0.0064/item, S18 $0.0039/item (cell (a) at M = 2: S11 $0.014, S18 $0.008), embed $0.0018/item, vLLM start-up $0.066 per container,
skeleton $0.026/item + $0.066 start-up. max_usd = 2 x estimate. Envelopes: E1 while (E1 estimates so far + max_usd) <= E1_HEADROOM (1.92,
frozen from `python -m opa.budget status`), then E2 (E2_HEADROOM 8.71). The generator does not read the ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_p0_jobs as p0  # noqa: E402
import make_p1_jobs as p1  # noqa: E402
from opa import launch  # noqa: E402

OFFSET, N = 50, 30
PH_VEC, PH_SK2, PH_SK3 = "H3CELLS_VEC_RUN_ID", "H3CELLS_SK2_RUN_ID", "H3CELLS_SK3_RUN_ID"
ALL_PH = (PH_VEC, PH_SK2, PH_SK3)
BUCKET = p0.BUCKET
VEC_URI = f"precomputed:gs://{BUCKET}/opa/runs/{PH_VEC}/embeddings.npz"
SK2_URI = f"gs://{BUCKET}/opa/runs/{PH_SK2}/skeleton_llm"
SK3_URI = f"gs://{BUCKET}/opa/runs/{PH_SK3}/skeleton_llm"

PRICE_ITEM = {"S11": 0.0064, "S18": 0.0039}
PRICE_ITEM_M2 = {"S11": 0.014, "S18": 0.008}      # cell (a): read window 1 instead of 3, ~2-2.4x the calls (checker ruling)
EMBED_PER_ITEM, SKELETON_PER_ITEM, STARTUP_USD = 0.0018, 0.026, 0.066
E1_HEADROOM, E2_HEADROOM = 1.92, 8.71
A_OPTS = {"lambda": 2.0, "M": 2}              # cell (a): halved resident-page budget for both systems (the oracle is checked under the same condition)
DEFAULT_OPTS = {"lambda": 2.0}
HYP = ("H3",)
MODEL = {"kind": "open_local", "name": "qwen", "max_model_len": 40960}


def uri_entry(key: str) -> dict:
    return {"uri": p1.items_uri(key), "n_items": N, "offset": OFFSET}


def done(spec, usd, work, items, notes):
    spec["hypotheses"] = list(HYP)
    return p1.finish_usd(spec, usd, work, items=items, notes=notes, suffix=f"; max_usd = 2 x the estimate = ${2 * usd:.3f}")


def embed_job():
    keys = ["lb_late_h2", "lb_late_h3"]
    args = {"items_uri": [uri_entry(k) for k in keys], "embedder": "harrier", "B": 512, "batch": 16}
    spec = p0.base("h3cells-embed", "A", "embed", "lbmhqa", "late-h2+late-h3", "opa.tracks.track_a:embed_items", args,
                   {"embedder": p0.EMBEDDER}, p0.gpu_compute("L4", 12), hyp=HYP)
    items = N * len(keys)
    usd = items * EMBED_PER_ITEM
    work = f"{items} items (late-h2-32k-a0 and late-h3-32k-a0, items {OFFSET}-{OFFSET + N - 1}) x ${EMBED_PER_ITEM}/item (measured, incl. start-up)"
    return done(spec, usd, work, items, ["one embeddings.npz for both cells (keys item_id/origin_id); items 50-79 are fresh (the pilot used 0-49)"])


def skeleton_job(name: str, key: str, note: str):
    args = {"items_uri": [uri_entry(key)], "workers": 24, "model": dict(MODEL)}
    spec = p0.base(name, "A", "sllm", "lbmhqa", p1.CELLS[key][1], "opa.tracks.track_a:build_skeletons", args, {"agent_model": p0.QWEN},
                   p0.vllm_compute(), hyp=HYP)
    usd = N * SKELETON_PER_ITEM + STARTUP_USD
    work = (f"{N} items ({p1.CELLS[key][1]}, items {OFFSET}-{OFFSET + N - 1}) x ${SKELETON_PER_ITEM}/item (sllm-v2 rebuilt builder, workers 24) + "
            f"${STARTUP_USD} start-up")
    return done(spec, usd, work, N, [note, "REBUILD with the fixed builder (prompt_version sllm-v2)",
                                     "estimate = 30 x $0.026 + $0.066; max_usd = 2 x the estimate"])


def run_job(name: str, systems: list[str], key: str, sk_uri: str, opts: dict, notes: list[str], price: dict | None = None):
    price = price or PRICE_ITEM
    args = {"items_uri": [uri_entry(key)], "systems": list(systems), "embedder": VEC_URI, "workers": 8, "model": dict(MODEL),
            "opts": dict(opts), "skeleton_llm_uri": sk_uri}
    spec = p0.base(name, "A", "+".join(systems), "lbmhqa", p1.CELLS[key][1], "opa.tracks.track_a:run_system", args, {"agent_model": p0.QWEN},
                   p0.vllm_compute(16), hyp=HYP, seed=0)
    items = N * len(systems)
    usd = STARTUP_USD + N * sum(price[s] for s in systems)
    work = f"${STARTUP_USD} start-up + {N} items x (" + " + ".join(f"{s} ${price[s]}" for s in systems) + ") per item"
    return done(spec, usd, work, items, notes + ["fresh items 50-79; dev only; the predictor is computed from the skeleton and residency logs before any EM is opened (PLAN_V3 6b)"])


def jobs():
    out = [embed_job(),
           skeleton_job("h3cells-sk-late-h2", "lb_late_h2", "the run id of this job replaces the late-h2 skeleton placeholder in h3cells-late-h2-*"),
           skeleton_job("h3cells-sk-late-h3", "lb_late_h3", "the run id of this job replaces the late-h3 skeleton placeholder in h3cells-late-h3-*"),
           run_job("h3cells-late-h2-m2-s11-s18", ["S18", "S11"], "lb_late_h2", SK2_URI, A_OPTS,
                   ["cell (a): resident-page budget halved, opts M = 2 (default 4), for S11 and the S18 oracle alike; systems ordered [S18, S11] "
                    "so the oracle check finishes first",
                    "M = 2 also reduces S11's read window from 3 origins to 1 (one slot kept free for faults, per design) — the cell changes how "
                    "S11 reads, not only residency",
                    "priced S11 $0.014/item and S18 $0.008/item (about 2-2.4x the calls of the default M)"], price=PRICE_ITEM_M2),
           run_job("h3cells-late-h3-s11-s18", ["S11", "S18"], "lb_late_h3", SK3_URI, DEFAULT_OPTS, ["cell (b): default M for both systems"])]
    return assign(out)


ORDER = ["h3cells-embed", "h3cells-sk-late-h2", "h3cells-sk-late-h3", "h3cells-late-h2-m2-s11-s18", "h3cells-late-h3-s11-s18"]
BUDGET = {"E1": E1_HEADROOM, "E2": E2_HEADROOM}


def assign(js):
    by = {m["name"]: (s, m) for s, m in js}
    spent, env, out = 0.0, "E1", []
    for n in ORDER:
        s, m = by[n]
        if env == "E1" and spent + m["max"] > E1_HEADROOM + 1e-9:
            env, spent = "E2", 0.0
        s["envelope"], m["envelope"] = env, env
        s["notes"].append(f"headroom frozen at generation: E1 {E1_HEADROOM:.2f}, E2 {E2_HEADROOM:.2f}; envelope assigned greedily in launch order")
        spent += m["est"]
        out.append((s, m))
    return out


def reservation_check(entries):
    rows, spent = [], {"E1": 0.0, "E2": 0.0}
    for _, m in entries:
        e = m["envelope"]
        need = spent[e] + m["max"]
        rows.append({"name": m["name"], "envelope": e, "prior_est": round(spent[e], 2), "max": m["max"], "need": round(need, 2),
                     "budget": BUDGET[e], "ok": need <= BUDGET[e] + 1e-9})
        spent[e] += m["est"]
    return rows


def dump(spec: dict, subs: dict[str, str]) -> bytes:
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000)
    for k in sorted(subs, key=len, reverse=True):
        if subs[k]:
            text = text.replace(k, subs[k])
    return text.encode("utf-8")


DUMMY = {k: f"dummy-{k.lower()}" for k in ALL_PH}


def filled(spec: dict) -> dict:
    return yaml.safe_load(dump(spec, DUMMY))


def write_launch(entries, ids: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "run_ids.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    known.update({k: v for k, v in ids.items() if v})
    f.write_bytes((json.dumps(known, indent=1, sort_keys=True) + "\n").encode())
    subs = {k: known.get(k) for k in ALL_PH}
    written, skipped = [], {}
    for spec, _ in entries:
        data = dump(spec, subs)
        left = p0.unresolved(data.decode())
        if left:
            skipped[spec["name"]] = left
            continue
        (out_dir / f"{spec['name']}.yaml").write_bytes(data)
        written.append(spec["name"])
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-run-id"), ap.add_argument("--sk2-run-id"), ap.add_argument("--sk3-run-id")
    a = ap.parse_args(argv)
    entries = jobs()
    for spec, _ in entries:
        errs = launch.validate(filled(spec))
        assert not errs, (spec["name"], errs)
        (HERE / f"{spec['name']}.yaml").write_bytes(dump(spec, {}))
    ids = {PH_VEC: a.vec_run_id, PH_SK2: a.sk2_run_id, PH_SK3: a.sk3_run_id}
    if any(ids.values()):
        written, skipped = write_launch(entries, ids, HERE / "launch")
        print(f"jobs/launch: {len(written)} written; refused (unresolved placeholders): {skipped or 'none'}")
    print(f"{'job':<26}{'env':>4}{'items':>6}{'est $':>8}{'max $':>8}")
    for _, m in entries:
        print(f"{m['name']:<26}{m['envelope']:>4}{m['items']:>6}{m['est']:>8.2f}{m['max']:>8.2f}")
    for e in ("E1", "E2"):
        ms = [m for _, m in entries if m["envelope"] == e]
        if ms:
            print(f"TOTAL {e}: estimate ${sum(m['est'] for m in ms):.2f}  max ${sum(m['max'] for m in ms):.2f}   (headroom ${BUDGET[e]:.2f})")
    print(f"TOTAL all: estimate ${sum(m['est'] for _, m in entries):.2f}")
    for r in reservation_check(entries):
        print(f"  {r['name']:<26} {r['envelope']} prior est {r['prior_est']:>5.2f} + max {r['max']:>5.2f} = {r['need']:>5.2f} / {r['budget']:.2f}  {'ok' if r['ok'] else 'OVER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
