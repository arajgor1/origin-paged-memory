"""Throughput benchmark job specs (owner request 2026-11-07): measure Track A seconds per item on one L4 at 16-24 concurrent requests
(the pilot ran at workers 8). Dev split only; outcomes of these items were already seen; NOT a result. Run: python jobs/make_bench_jobs.py

  bench-mt-w24   systems [S12, S14] on MT T4 h3 cross items 0-23, workers 24; vectors, skeletons and opts as the launched p1-mt-t4-arms
  bench-lb-w16   systems [S2, S8, S11, S4] on LB late h2 32K items 0-47 (the already-seen pilot items; items 50-79 are the blind H3
                 candidate items and are never used here), workers 16; vectors, skeletons and opts as the launched p1-lb-late-h2
  bench-lb-w24   identical with workers 24
Specs are launch-ready: they name the existing run ids directly (no placeholders). opa.analysis.plan_costs reads the measured seconds per item
of these runs (job names bench-lb-w<c>, bench-mt-w<c>) with --concurrency 16 / 24. Skeleton cost at 24 workers needs no new job: it is computed
from the runs A-sllm-...-97eb ($1.322, 50 items), ...-88f9 ($0.814, 30 items) and ...-2477 ($0.879, 30 items), net of start-up.

PRICING is conservative: the measured workers-8 seconds per item (opa.analysis.plan_costs PARAMS) + $0.066 start-up, max_usd = 2 x the estimate
(the higher concurrency should be faster, never slower). Envelopes: E0 (remaining 2.39, owner-approved, unused setup money) while the reservation
rule holds (E0 estimates so far + max_usd <= 2.39), else E2 (remaining 3.10), else E3 (remaining 10.00); the cheapest job is launched first so that it fits E0. A job that
fits neither goes to E3 (logged reallocation ruling), which has $10.00 unused.
"""
from __future__ import annotations

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

LB_VEC = "A-embed-lbmhqa+mtlbmhqa-3cells-dev-s0-20261107041041-2b257da-a07c"
LB_SK = "A-sllm-lbmhqa-late-h2-32k-a0-dev-s0-20261107140747-99677c4-97eb"
MT_VEC = "A-embed-lbmhqa+mtlbmhqa-5cells-dev-s0-20261107114211-40530d3-59b1"
MT_SK = "A-sllm-mtlbmhqa-T4-h3-32k-cross-dev-s0-20261107140751-99677c4-251f"
BUDGET = {"E0": 2.39, "E2": 3.10, "E3": 10.00}   # frozen from `python -m opa.budget status` (the generator does not read the ledger)
NOTE = "throughput benchmark (owner request 2026-11-07); outcomes already seen; not a result"
ORDER = ["bench-mt-w24", "bench-lb-w16", "bench-lb-w24"]
MODEL = {"kind": "open_local", "name": "qwen", "max_model_len": 40960}


def make(name, kind, systems, n, workers):
    lb = kind == "lb"
    key = "lb_late_h2" if lb else "mt_T4_cross"
    args = {"items_uri": [{"uri": p1.items_uri(key), "n_items": n, "offset": 0}], "systems": list(systems),
            "embedder": f"precomputed:gs://{p0.BUCKET}/opa/runs/{LB_VEC if lb else MT_VEC}/embeddings.npz", "workers": workers,
            "model": dict(MODEL), "opts": {"lambda": 2.0},
            "skeleton_llm_uri": f"gs://{p0.BUCKET}/opa/runs/{LB_SK if lb else MT_SK}/skeleton_llm"}
    spec = p0.base(name, "A", "+".join(systems), "lbmhqa" if lb else "mtlbmhqa", p1.CELLS[key][1], "opa.tracks.track_a:run_system", args,
                   {"agent_model": p0.QWEN}, p0.vllm_compute(16), hyp=("throughput-benchmark",), seed=0)
    s_item = pc.PARAMS["s_per_item_lb" if lb else "s_per_item_mt"]
    sec = sum(s_item[s] * n for s in systems)
    usd = sec * pc.RATE + pc.PARAMS["start_usd"]
    work = (f"{n} items x (" + " + ".join(f"{s} {s_item[s]:.2f} s" for s in systems) + f") measured at workers 8 = {sec:.0f} s x ${pc.RATE:.6f}/s + "
            f"${pc.PARAMS['start_usd']} start-up (conservative: workers {workers} should be faster)")
    spec, meta = p1.finish_usd(spec, usd, work, items=n * len(systems), seconds=sec,
                               notes=[NOTE, f"workers {workers}; same opts, vectors and skeletons as the launched "
                                      + ("p1-lb-late-h2" if lb else "p1-mt-t4-arms") + "; items 0-" + str(n - 1)
                                      + ("; items 50-79 are the blind H3 candidate items and are not used" if lb else "")],
                               suffix=f"; max_usd = 2 x the estimate = ${2 * usd:.3f}")
    return spec, meta


def jobs():
    built = {"bench-mt-w24": make("bench-mt-w24", "mt", ["S12", "S14"], 24, 24),
             "bench-lb-w16": make("bench-lb-w16", "lb", ["S2", "S8", "S11", "S4"], 48, 16),
             "bench-lb-w24": make("bench-lb-w24", "lb", ["S2", "S8", "S11", "S4"], 48, 24)}
    out, spent, env = [], {"E0": 0.0, "E2": 0.0, "E3": 0.0}, "E0"
    for n in ORDER:
        s, m = built[n]
        while env != "E3" and spent[env] + m["max"] > BUDGET[env] + 1e-9:
            env = {"E0": "E2", "E2": "E3"}[env]
        m["envelope"] = s["envelope"] = env
        if env == "E3":
            s["notes"].append("envelope E3 (remaining $10.00, unused): logged reallocation ruling in state/decisions.md (2026-11-07) after E0 and E2 could not hold this benchmark's reservation; "
                              f"headroom frozen at generation: E3 {BUDGET['E3']:.2f}")
        else:
            s["notes"].append(f"headroom frozen at generation: E0 {BUDGET['E0']:.2f}, E2 {BUDGET['E2']:.2f}; envelope assigned greedily in launch order")
        spent[env] += m["est"]
        out.append((s, m))
    return out


def reservation_check(entries):
    rows, spent = [], {"E0": 0.0, "E2": 0.0, "E3": 0.0}
    for _, m in entries:
        e = m["envelope"]
        need = spent[e] + m["max"]
        rows.append({"name": m["name"], "envelope": e, "prior_est": round(spent[e], 2), "max": m["max"], "need": round(need, 2),
                     "budget": BUDGET[e], "ok": need <= BUDGET[e] + 1e-9})
        spent[e] += m["est"]
    return rows


def dump(spec: dict) -> bytes:
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000).encode("utf-8")


def main(argv=None) -> int:
    entries = jobs()
    for spec, _ in entries:
        errs = launch.validate(spec)
        assert not errs, (spec["name"], errs)
        (HERE / f"{spec['name']}.yaml").write_bytes(dump(spec))
    print(f"{'job':<16}{'env':>4}{'items':>6}{'est $':>8}{'max $':>8}")
    for _, m in entries:
        print(f"{m['name']:<16}{m['envelope']:>4}{m['items']:>6}{m['est']:>8.2f}{m['max']:>8.2f}")
    for r in reservation_check(entries):
        print(f"  {r['name']:<16} {r['envelope']} prior est {r['prior_est']:>5.2f} + max {r['max']:>5.2f} = {r['need']:>5.2f} / {r['budget']:.2f}  {'ok' if r['ok'] else 'OVER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
