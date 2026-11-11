"""Dev-only job specs for the H5 primary MT cell candidate T8 h3 cross (PLAN_V3 Section 8 item 3: the cell with the largest distortion
ratio among cells where the oracle S18 reaches >= 60% on dev; T4 h3 cross failed it, next by distortion is T8, ratio 1.291).
Run: python jobs/make_h5cell_jobs.py [--vec-run-id ID --sk-run-id ID]

  h5cell-embed       fresh harrier embed of MT T8 h3 cross items 5-54 (CHOSEN over reusing the P0 npz: that one holds T8 items 0-9 only and a
                     job reads exactly one npz, so a fresh 50-item job is the simplest)
  h5cell-sk-t8       sllm-v2 skeletons for those 50 items (workers 24; 50 x $0.026 + $0.066, max 2x)
  h5cell-t8-oracle   systems [S18] on the 50 items, opts {lambda: 2.0} (the T4 arms job's opts), workers 8, seed 0, reading the two above
  h5cell-t8-arms     CONDITIONAL (generated, not to be launched unless S18 >= 0.60 on T8): systems [S12, S14, S15, S16], same opts
Prices: S18 measured on MT T4 cross, 1,142 s / 50 items = $0.0079/item; S12/S14/S15/S16 measured on the same job (A-S12+S14+S15+S16+S18-...-db29):
1,090.6 / 1,373.9 / 1,082.7 / 1,331.5 s per 50 items, priced at the container rate (T8 is extrapolated from T4); embed $0.0018/item;
start-up $0.066. max_usd = 2 x estimate. Envelope E2 (frozen headroom 6.85 from `python -m opa.budget status`; the generator does not read
the ledger). Placeholders H5CELL_VEC_RUN_ID and H5CELL_SK_RUN_ID are filled from run ids (jobs/launch/run_ids.json, filled specs in jobs/launch/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_p0_jobs as p0  # noqa: E402
import make_p1_jobs as p1  # noqa: E402
from opa import launch  # noqa: E402

N = 50
OFFSET = 5                              # items 5-54: no overlap with the P0 micro-pilot T8 items 0-4 (cell has 100 items)
PH_VEC, PH_SK = "H5CELL_VEC_RUN_ID", "H5CELL_SK_RUN_ID"
ALL_PH = (PH_VEC, PH_SK)
BUCKET = p0.BUCKET
VEC_URI = f"precomputed:gs://{BUCKET}/opa/runs/{PH_VEC}/embeddings.npz"
SK_URI = f"gs://{BUCKET}/opa/runs/{PH_SK}/skeleton_llm"
E2_HEADROOM = 6.85
EMBED_PER_ITEM, SKELETON_PER_ITEM, STARTUP_USD = 0.0018, 0.026, 0.066
WALL_PER_50 = {"S18": 1142.2, "S12": 1090.6, "S14": 1373.9, "S15": 1082.7, "S16": 1331.5}      # measured on MT T4 cross, 50 items
OPTS = {"lambda": 2.0}
MODEL = {"kind": "open_local", "name": "qwen", "max_model_len": 40960}
HYP = ("H5",)
ORDER = ["h5cell-embed", "h5cell-sk-t8", "h5cell-t8-oracle", "h5cell-t8-arms"]
CONDITIONAL = {"h5cell-t8-arms"}


def uri_entry() -> dict:
    return {"uri": p1.items_uri("mt_T8_cross"), "n_items": N, "offset": OFFSET}


def done(spec, usd, work, items, notes, conditional=False):
    spec["hypotheses"] = list(HYP)
    spec, meta = p1.finish_usd(spec, usd, work, items=items, notes=notes, suffix=f"; max_usd = 2 x the estimate = ${2 * usd:.3f}")
    meta["conditional"] = conditional
    spec["envelope"] = meta["envelope"] = "E2"
    spec["notes"].append(f"headroom frozen at generation: E2 {E2_HEADROOM:.2f}; envelope E2 (E1 is exhausted)")
    return spec, meta


def embed_job():
    args = {"items_uri": [uri_entry()], "embedder": "harrier", "B": 512, "batch": 16}
    spec = p0.base("h5cell-embed", "A", "embed", "mtlbmhqa", "T8-h3-32k-cross", "opa.tracks.track_a:embed_items", args,
                   {"embedder": p0.EMBEDDER}, p0.gpu_compute("L4", 12), hyp=HYP)
    return done(spec, N * EMBED_PER_ITEM, f"{N} items (T8-h3-32k-cross, {OFFSET}-{OFFSET + N - 1}) x ${EMBED_PER_ITEM}/item (measured, incl. start-up)", N,
                ["fresh 50-item embed (the P0 npz holds T8 items 0-9 only and a job reads one npz)"])


def skeleton_job():
    args = {"items_uri": [uri_entry()], "workers": 24, "model": dict(MODEL)}
    spec = p0.base("h5cell-sk-t8", "A", "sllm", "mtlbmhqa", "T8-h3-32k-cross", "opa.tracks.track_a:build_skeletons", args, {"agent_model": p0.QWEN},
                   p0.vllm_compute(), hyp=HYP)
    usd = N * SKELETON_PER_ITEM + STARTUP_USD
    return done(spec, usd, f"{N} items (T8-h3-32k-cross) x ${SKELETON_PER_ITEM}/item (sllm-v2, workers 24) + ${STARTUP_USD} start-up", N,
                ["the run id of this job replaces the skeleton placeholder in h5cell-t8-oracle and h5cell-t8-arms",
                 "REBUILD with the fixed builder (prompt_version sllm-v2); estimate = 50 x $0.026 + $0.066, max_usd = 2 x the estimate"])


def run_job(name: str, systems: list[str], notes: list[str], conditional=False):
    args = {"items_uri": [uri_entry()], "systems": list(systems), "embedder": VEC_URI, "workers": 8, "model": dict(MODEL),
            "opts": dict(OPTS), "skeleton_llm_uri": SK_URI}
    spec = p0.base(name, "A", "+".join(systems), "mtlbmhqa", "T8-h3-32k-cross", "opa.tracks.track_a:run_system", args, {"agent_model": p0.QWEN},
                   p0.vllm_compute(16), hyp=HYP, seed=0)
    rate = launch.rate_per_second(spec["compute"], p0.prices())
    wall = sum(WALL_PER_50[s] for s in systems)
    usd = STARTUP_USD + wall * rate
    work = (f"${STARTUP_USD} start-up + (" + " + ".join(f"{s} {WALL_PER_50[s]} s" for s in systems) +
            f" per 50 items, measured on MT T4 cross) = {wall:.0f} s x ${rate:.6f}/s")
    return done(spec, usd, work, N * len(systems), notes, conditional)


def jobs():
    return [embed_job(), skeleton_job(),
            run_job("h5cell-t8-oracle", ["S18"], ["oracle-solvability check of T8 h3 cross: S18 >= 0.60 qualifies the cell (PLAN_V3 s8 item 3)"]),
            run_job("h5cell-t8-arms", ["S12", "S14", "S15", "S16"],
                    ["CONDITIONAL: launch only if S18 >= 0.60 on T8 (h5cell-t8-oracle); H5 arms A2/A1/A3/A0 with the T4 arms job's opts (lambda 2.0)"],
                    conditional=True)]


def reservation_check(entries):
    rows, spent = [], 0.0
    for _, m in entries:                     # worst case: the conditional job is launched too
        need = spent + m["max"]
        rows.append({"name": m["name"], "prior_est": round(spent, 2), "max": m["max"], "need": round(need, 2), "ok": need <= E2_HEADROOM + 1e-9,
                     "conditional": m["conditional"]})
        spent += m["est"]
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
    ap.add_argument("--vec-run-id"), ap.add_argument("--sk-run-id")
    a = ap.parse_args(argv)
    entries = jobs()
    for spec, _ in entries:
        errs = launch.validate(filled(spec))
        assert not errs, (spec["name"], errs)
        (HERE / f"{spec['name']}.yaml").write_bytes(dump(spec, {}))
    if a.vec_run_id or a.sk_run_id:
        written, skipped = write_launch(entries, {PH_VEC: a.vec_run_id, PH_SK: a.sk_run_id}, HERE / "launch")
        print(f"jobs/launch: {len(written)} written; refused (unresolved placeholders): {skipped or 'none'}")
    print(f"{'job':<20}{'env':>4}{'items':>6}{'est $':>8}{'max $':>8}")
    for _, m in entries:
        print(f"{m['name']:<20}{m['envelope']:>4}{m['items']:>6}{m['est']:>8.2f}{m['max']:>8.2f}" + ("  (conditional)" if m["conditional"] else ""))
    print(f"TOTAL E2 incl. conditional: estimate ${sum(m['est'] for _, m in entries):.2f}  max ${sum(m['max'] for _, m in entries):.2f}  (headroom ${E2_HEADROOM:.2f})")
    for r in reservation_check(entries):
        print(f"  {r['name']:<20} prior est {r['prior_est']:>5.2f} + max {r['max']:>5.2f} = {r['need']:>5.2f} / {E2_HEADROOM:.2f}  {'ok' if r['ok'] else 'OVER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
