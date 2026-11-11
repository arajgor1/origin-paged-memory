"""Write the P1 Track A pilot job specs. Run: python jobs/make_p1_jobs.py [--primary-h {2,3}]

TWO STAGES.
  Stage A (launch now, always written): p1-embed-pilot (vectors for every item any stage-B variant could need: LB forward and mixed at
  h2 AND h3, MT T4 h3 cross; LB late h2/h3 vectors already exist from p0-embed-dev; MT dev cells exist at h = 3 ONLY, so MT never has an h2
  variant), p1-skeleton-lateh3 (rebuilt sllm-v2 skeletons), p1-g0-s18-h3-recheck (S18 on 50 LB late h3 items).
  Stage B (only with --primary-h; decided by the recheck: S18 >= 80% EM at h3 -> 3, else 2), all at the primary h:
    p1-lb-late-h{h}: S1, S2, S4, S8, S11 (+ S18 when h = 2, the oracle on rebuilt skeletons at the primary h) and two repeat containers
    p1-lb-late-h{h}-rep1/-rep2 (S1, S2, S8; separate containers, seed 0), p1-lb-forward-mixed-h{h} (S11), p1-mt-t4-arms (S12 = A2,
    S14 = A1, S15 = A3, S16 = A0 and S18 on MT T4 h3 cross, the leading H5 cell; A4 = S12F runs in P2), drift D2/D4 (30 sessions, H2), thrash
    pin/nopin (r3, r4, r6, 30 items each, H10), and the skeleton jobs those need (h = 3 reuses the stage-A late-h3 skeletons).
  Track B pilot jobs (minilora V3, minilora B1, remedies; est $1.34, max $2.40 together) come out of E1 too: they are reserved in the
  tables right after stage A (their specs are written elsewhere).
Templates jobs/p1-*.yaml are never substituted. Run ids (--vectors-run-id, --skeleton-lateh3-run-id, --skeleton-lateh2-run-id,
--skeleton-lb-run-id, --skeleton-mt-run-id) go to jobs/launch/run_ids.json (shared with make_p0_jobs.py) and the filled specs to
jobs/launch/<name>.yaml; {bucket} is left for run time; a spec with an unfilled run-id placeholder is refused. Every skeleton reader points
at a placeholder for a rebuilt (prompt_version sllm-v2) skeleton directory; no old skeleton run is reused.

HEADROOM is frozen (the generator does not read the live ledger). Stage A (embed-pilot, skeleton-lateh3, g0 recheck) and the Track B pilot
jobs have ALREADY RUN or are reserved: their spend is in the ledger (Track B: V3 $0.472, B1 $0.179, remedies $0.217 + re-run $0.325), so
the tables cover stage B only. E1_HEADROOM = 2.33 (E1 left after the current reservations, recheck reserved); E2_APPROVED = 22.00.
Stage B takes E1 while (E1 estimates so far + max_usd) <= E1_HEADROOM, then E2. E1_AT_STAGE_A = 5.65 is the headroom stage A was checked
against (kept only for the record test).

ESTIMATES come from MEASURED numbers in state/unit_costs.md (micro-pilot re-run): seconds per item per system = sub-run wall_seconds / 10
(8 workers, container dedicated to one sub-run), vLLM start-up 189 s per container at $0.000346/s. Embeddings: measured 5.7 s per item on L4.
Skeletons (rebuilt builder, untested on Modal): items x $0.028 + $0.066 start-up, max_usd = 2 x that. NOT measured (flagged in notes): drift
and thrash (token counts converted with throughputs derived from measured runs: prefill 2,570 tok/s, decode 160 tok/s at 8 streams, 400
assumed at 30 streams). The micro-pilot cells were 5 LB late h3 + 5 MT T8 cross items, so every other cell is extrapolated.
"""
from __future__ import annotations

import argparse
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
from opa import launch  # noqa: E402

CELLS = {
    **p0.CELLS,
    "lb_forward_h2": ("lbmhqa", "forward-h2-32k-a0"),
    "lb_mixed_h2": ("lbmhqa", "mixed-h2-32k-a0"),
    "drift_s50": ("drift", "s50"),
    "thrash_r3": ("thrash", "r3"), "thrash_r4": ("thrash", "r4"), "thrash_r6": ("thrash", "r6"),
}
N = 50                                   # pilot items per cell
N_THRASH = 30
BUCKET = p0.BUCKET
PH_VEC, PH_SK_LB, PH_SK_MT = "PILOT_VECTORS_RUN_ID", "PILOT_SKELETON_LB_RUN_ID", "PILOT_SKELETON_MT_RUN_ID"
PH_SK_H3, PH_SK_LATE_H2 = "PILOT_SKELETON_LATEH3_RUN_ID", "PILOT_SKELETON_LATEH2_RUN_ID"
VEC_URI = f"precomputed:gs://{BUCKET}/opa/runs/{PH_VEC}/embeddings.npz"
SK_LB_URI = f"gs://{BUCKET}/opa/runs/{PH_SK_LB}/skeleton_llm"
SK_MT_URI = f"gs://{BUCKET}/opa/runs/{PH_SK_MT}/skeleton_llm"
SK_H3_URI = f"gs://{BUCKET}/opa/runs/{PH_SK_H3}/skeleton_llm"
SK_LATE_H2_URI = f"gs://{BUCKET}/opa/runs/{PH_SK_LATE_H2}/skeleton_llm"
ALL_PH = (PH_VEC, PH_SK_LB, PH_SK_MT, PH_SK_H3, PH_SK_LATE_H2, p0.PH_EMBED)

# measured (state/unit_costs.md, micro-pilot re-run): wall seconds per item at 8 workers
S_PER_ITEM = {"S1": 12.3, "S2": 1.0, "S4": 29.5, "S8": 34.7, "S11": 29.1, "S12": 15.7, "S14": 15.5, "S15": 15.3,
              "S16": 33.3, "S18": 15.8, "S12F": 96.0}
START = 189                              # s per vLLM container (measured 184-189)
SKELETON_PER_ITEM = 0.034                # MEASURED: A-sllm-...-20261107113922-f2e104d-7134 (lateh3 rebuild), $1.681 / 50 items = $0.0336 incl. start-up
SKELETON_SOURCE_RUN = "A-sllm-lbmhqa-late-h3-32k-a0-dev-s0-20261107113922-f2e104d-7134"
SKELETON_WORKERS = 24
SKELETON_STARTUP_USD = 0.066             # ~190 s x $0.000346/s, one vLLM start-up per container
EMBED_S_PER_ITEM = 5.7                   # measured incl. start-up and self-check, L4
P_AGG, D_AGG = 2570.0, 160.0             # derived aggregate prefill / decode tok/s (see docstring)


E1_AT_STAGE_A = 5.65                    # headroom stage A was checked against (record)
E1_HEADROOM = 2.33                      # frozen; the ledger now shows E1 $1.65 remaining after the recheck reservation; stage B goes to E2 either way
                                        # (the generator does not read the ledger)
E2_APPROVED = 22.00
N_DRIFT = 30
DRIFT_WORKERS = 30                       # all sessions concurrently
D_AGG_30 = 400.0                         # ASSUMED aggregate decode tok/s at 30 streams (measured: 215 at 12, ~158 at 8)
SKELETON_NOTE = ("estimate = items x $0.034 + $0.066 start-up, measured on the lateh3 rebuild (" + SKELETON_SOURCE_RUN + ": $1.681 for 50 items, 4,866 s at "
                 "12 workers); max_usd = 2 x the estimate")
HYP_PREP = ("P1-prep",)


def items_uri(key: str) -> str:
    ds, cell = CELLS[key]
    return f"gs://{BUCKET}/opa/data/{ds}/dev/v2/{cell}/items.jsonl.gz"


def uri_entry(key: str, n: int, offset: int = 0) -> dict:
    d = {"uri": items_uri(key), "n_items": n}
    if offset:
        d["offset"] = offset
    return d


def to_e1(pair):
    spec, meta = pair
    spec["envelope"] = "E1"
    return spec, meta


def rate_of(compute: dict) -> float:
    return launch.rate_per_second(compute, p0.prices())


def finish_usd(spec: dict, usd: float, work: str, *, items: int, notes: list[str], seconds: float | None = None,
               max_basis: float | None = None, suffix: str = ""):
    """Estimate given directly in dollars. max_usd = 2 x max_basis (default: the estimate)."""
    rate, rtxt = p0.rate_text(spec["compute"])
    sec = seconds if seconds is not None else usd / rate
    est = math.ceil(usd * 100) / 100
    mx = math.ceil(2 * (usd if max_basis is None else max_basis) * 100) / 100
    spec["estimate_formula"] = f"{work} = ${usd:.3f} (container rate {rtxt}){suffix}"
    spec["estimate_usd"], spec["max_usd"], spec["max_api_usd"], spec["notes"] = est, mx, 0, notes
    return spec, {"name": spec["name"], "items": items, "seconds": sec, "est": est, "max": mx, "raw": usd, "conditional": False}


# ----------------------------------------------------------------------------------------------- preparation


def vectors_job():
    keys = [("lb_forward_h2", N), ("lb_mixed_h2", N), ("lb_forward_h3", N), ("lb_mixed_h3", N), ("mt_T4_cross", N)]
    args = {"items_uri": [uri_entry(k, n) for k, n in keys], "embedder": "harrier", "B": 512, "batch": 16}
    spec = p0.base("p1-embed-pilot", "A", "embed", "lbmhqa+mtlbmhqa", f"{len(keys)}cells", "opa.tracks.track_a:embed_items", args,
                   {"embedder": p0.EMBEDDER}, p0.gpu_compute("L4", 12), hyp=HYP_PREP)
    items = sum(n for _, n in keys)
    sec = items * EMBED_S_PER_ITEM
    work = (f"{items} items ({', '.join(f'{CELLS[k][1]} x {n}' for k, n in keys)}) x {EMBED_S_PER_ITEM} s/item (measured, incl. start-up) "
            f"= {sec:.0f} s")
    notes = ["every item any stage-B variant could need, except those p0-embed-dev already holds (LB late h2 and h3, 50 items each, "
             "which the late-cell jobs keep reading); a job reads one npz, so forward/mixed at both h and MT T4 cross (h3 only: no MT h2 cells "
             "exist) are all here; the 10 forward/mixed h3 and T4 items of p0-embed-dev-aux are re-embedded in full (50)"]
    return to_e1(p0.finish(spec, sec, work, 0, items=items, notes=notes))


def skeleton_job(name: str, keys: list[tuple[str, int]], ph_note: str, frozen_launch: bool = False):
    """frozen_launch=True reproduces EXACTLY what was launched for stage A (workers 12, $0.028/item, the original notes and formula)."""
    args = {"items_uri": [uri_entry(k, n) for k, n in keys], "workers": 12 if frozen_launch else SKELETON_WORKERS,
            "model": {"kind": "open_local", "name": "qwen", "max_model_len": 40960}}
    spec = p0.base(name, "A", "sllm", "+".join(dict.fromkeys(CELLS[k][0] for k, _ in keys)), "+".join(CELLS[k][1] for k, _ in keys),
                   "opa.tracks.track_a:build_skeletons", args, {"agent_model": p0.QWEN}, p0.vllm_compute(), hyp=HYP_PREP)
    items = sum(n for _, n in keys)
    if frozen_launch:
        usd = items * 0.028 + SKELETON_STARTUP_USD
        work = (f"{items} items ({', '.join(f'{CELLS[k][1]} x {n}' for k, n in keys)}) x $0.028/item (rebuilt builder, sllm-v2) + "
                f"${SKELETON_STARTUP_USD} start-up (190 s x $0.000346/s); max_usd = 2 x that = ${2 * usd:.3f}")
        notes = [ph_note, "REBUILD with the fixed builder (prompt_version sllm-v2); no old skeleton run is reused",
                 "ESTIMATE (untested on Modal): $0.028/item + $0.066 start-up; max_usd = 2 x that; replace with the measured cost after the first rebuild",
                 "run_system reads ONE skeleton directory (skeleton_llm_uri is a single string), so each directory holds every item its readers need"]
        return to_e1(finish_usd(spec, usd, work, items=items, notes=notes))
    usd = items * SKELETON_PER_ITEM + SKELETON_STARTUP_USD
    work = (f"{items} items ({', '.join(f'{CELLS[k][1]} x {n}' for k, n in keys)}) x ${SKELETON_PER_ITEM}/item (measured on the lateh3 rebuild "
            f"{SKELETON_SOURCE_RUN}) + ${SKELETON_STARTUP_USD} start-up (190 s x $0.000346/s)")
    notes = [ph_note, "REBUILD with the fixed builder (prompt_version sllm-v2); no old skeleton run is reused", SKELETON_NOTE,
             f"workers {SKELETON_WORKERS}: a decode-bound vLLM on an L4 gains aggregate throughput from more concurrent streams; the estimate "
             "stays at the measured 12-worker rate as a conservative bound",
             "run_system reads ONE skeleton directory (skeleton_llm_uri is a single string), so each directory holds every item its readers need"]
    return to_e1(finish_usd(spec, usd, work, items=items, notes=notes, suffix=f"; max_usd = 2 x the estimate = ${2 * usd:.3f}"))


# ----------------------------------------------------------------------------------------------- pilots


def systems_job(name, systems, keys, *, embedder, skeleton_uri, hyp, extra_notes=(), single=False):
    needs_sk = any(p0.PROFILE[s][5] for s in systems)
    args = {"items_uri": [uri_entry(k, n) for k, n in keys], "systems": list(systems), "embedder": embedder, "workers": 8,
            "model": {"kind": "open_local", "name": "qwen", "max_model_len": 40960}}
    if single:                                   # one system, one run at the run root (as the P0 G0 jobs)
        args = {"items_uri": items_uri(keys[0][0]), "n_items": keys[0][1], "system": systems[0], "embedder": embedder, "workers": 8,
                "model": args["model"]}
    if any(s in p0.launch_fold_systems() for s in systems):
        args["opts"] = dict(p0.FOLD_OPTS)
    if needs_sk:
        args["skeleton_llm_uri"] = skeleton_uri
    spec = p0.base(name, "A", "+".join(systems), "+".join(dict.fromkeys(CELLS[k][0] for k, _ in keys)),
                   "+".join(CELLS[k][1] for k, _ in keys), "opa.tracks.track_a:run_system", args, {"agent_model": p0.QWEN},
                   p0.vllm_compute(16), hyp=hyp, seed=0)
    items = sum(n for _, n in keys)
    parts = [(s, items * S_PER_ITEM[s]) for s in systems]
    sec = START + sum(t for _, t in parts)
    work = (f"{START} s start-up + {items} items x (" + " + ".join(f"{s} {S_PER_ITEM[s]} s" for s in systems) +
            ") per item (measured wall s/item, micro-pilot re-run)")
    notes = list(extra_notes) + ["estimate = measured micro-pilot seconds per item (5 LB late h3 + 5 MT T8 cross items); other cells are extrapolated"]
    spec, meta = to_e1(p0.finish(spec, sec, work, START, items=items * len(systems), notes=notes))
    return spec, meta


def lb_key(kind: str, h: int) -> str:
    return f"lb_{kind}_h{h}"


def stage_a():
    """Stage A (launch now): vectors for every item any stage-B variant needs, the late-h3 skeletons, the G0 oracle re-check."""
    return [vectors_job(),
            skeleton_job("p1-skeleton-lateh3", [("lb_late_h3", N)],
                         "the run id of this job replaces the late-h3 skeleton placeholder in p1-g0-s18-h3-recheck (and in p1-lb-late-h3 if h = 3)",
                         frozen_launch=True),
            systems_job("p1-g0-s18-h3-recheck", ["S18"], [("lb_late_h3", N)], embedder=p0.EMBED_URI, skeleton_uri=SK_H3_URI, hyp=("G0",),
                        single=True, extra_notes=["G0 oracle re-check at h = 3 with the rebuilt skeletons (sllm-v2; separate job, seed 0); vectors from "
                                                  "p0-embed-dev (LB late h3, 50 items). S18 >= 80% EM -> primary h = 3, else h = 2"])]


def stage_b(h: int):
    """Stage B for primary h (2 or 3): LB cells at h, MT T4 cross (MT cells exist at h = 3 only), drift, thrash, and the skeleton
    jobs those need. h = 3 reuses the stage-A late-h3 skeletons (the recheck is the oracle); h = 2 adds S18 to the main late job."""
    assert h in (2, 3)
    late, fwd, mix = lb_key("late", h), lb_key("forward", h), lb_key("mixed", h)
    out = []
    if h == 2:
        out.append(skeleton_job("p1-skeleton-lateh2", [(late, N)], "the run id of this job replaces the late-h2 skeleton placeholder in the p1-lb-late-h2 specs"))
    out.append(skeleton_job("p1-skeleton-lb", [(fwd, N), (mix, N)], f"the run id of this job replaces the LB skeleton placeholder in p1-lb-forward-mixed-h{h}"))
    out.append(skeleton_job("p1-skeleton-mt", [("mt_T4_cross", N)], "the run id of this job replaces the MT skeleton placeholder in p1-mt-t4-arms"))
    late_sk = SK_LATE_H2_URI if h == 2 else SK_H3_URI
    systems = ["S1", "S2", "S4", "S8", "S11"] + (["S18"] if h == 2 else [])
    out.append(systems_job(f"p1-lb-late-h{h}", systems, [(late, N)], embedder=p0.EMBED_URI, skeleton_uri=late_sk, hyp=("H1", "H3"),
                           extra_notes=[f"vectors from p0-embed-dev (LB late h{h}, 50 items); skeletons from the "
                                        + ("rebuilt late-h2 skeleton run; S18 added so the oracle is measured on rebuilt skeletons at the primary h"
                                           if h == 2 else "stage-A late-h3 skeleton run (the G0 recheck is the oracle)")]))
    for r in (1, 2):
        out.append(systems_job(f"p1-lb-late-h{h}-rep{r}", ["S1", "S2", "S8"], [(late, N)], embedder=p0.EMBED_URI, skeleton_uri=late_sk,
                               hyp=("H1",), extra_notes=[f"run-to-run variance: separate container, seed 0, same inputs as p1-lb-late-h{h}"]))
    out.append(systems_job(f"p1-lb-forward-mixed-h{h}", ["S11"], [(fwd, N), (mix, N)], embedder=VEC_URI, skeleton_uri=SK_LB_URI, hyp=("H3",)))
    out.append(systems_job("p1-mt-t4-arms", ["S12", "S14", "S15", "S16", "S18"], [("mt_T4_cross", N)], embedder=VEC_URI,
                           skeleton_uri=SK_MT_URI, hyp=("H5",),
                           extra_notes=["MT T4 h3 cross is the leading H5 cell (distortion ratio 1.361 vs 1.291 for T8); S18 here is the oracle-solvability check",
                                        "arms: S12 = A2 hyperbolic, S14 = A1 Euclidean, S15 = A3 tree distance, S16 = A0 (lambda 0); the A4 arm (S12F) runs in P2"]))
    out += drift_jobs() + thrash_jobs()
    return out


def drift_jobs():
    from opa.tracks.track_a2.run import estimate_session, job_spec
    out = []
    for cond in ("D2", "D4"):
        spec = job_spec(f"p1-drift-{cond.lower()}", cond, items_uri("drift_s50"), n_items=N_DRIFT, envelope="E1", model="qwen", split="dev",
                        cell="s50", seed=0, workers=DRIFT_WORKERS, gpu="L4", cpu=2.0, memory_gib=16)
        spec.pop("throughput_placeholders", None)
        spec["hypotheses"] = ["H2"]
        spec["compute"] = p0.vllm_compute(16)
        e = estimate_session(cond)["typical"]
        per = e["tokens_in"] / P_AGG + e["tokens_out"] / D_AGG_30             # per session of aggregate GPU time
        sec = START + N_DRIFT * per
        work = (f"{START} s start-up + {N_DRIFT} sessions x {per:.1f} s ({cond}: {e['calls']} calls, {e['tokens_in']} in tok / {P_AGG:.0f} + "
                f"{e['tokens_out']} out tok / {D_AGG_30:.0f} tok/s at {DRIFT_WORKERS} concurrent sessions, estimate_session typical)")
        notes = ["NOT MEASURED: no drift session has run; tokens from estimate_session (typical), rates derived from measured Track A runs",
                 "all 30 sessions run concurrently (workers 30); decode throughput at 30 streams (400 tok/s) is an ASSUMPTION extrapolated from 215 tok/s at 12 streams",
                 "worst case per estimate_session is about 2.4x the typical output tokens for D2: ceiling 2.5x for D2"]
        for k in ("estimate_formula", "estimate_usd", "max_usd"):
            spec.pop(k, None)
        out.append(p0.finish(spec, sec, work, START, items=N_DRIFT, notes=notes, max_factor=2.5 if cond == "D2" else 2.0))
    return out


def thrash_jobs():
    out = []
    steps = [0.8 * (4 * r + 4) for r in (3, 4, 6)]                  # r3, r4, r6 step caps x 0.8 typical use
    in_tok_step, out_tok_step = 2500, 80                           # prompt (2 slots x 512 + state) / action
    per = (sum(steps) / 3) * (in_tok_step / P_AGG + out_tok_step / D_AGG)
    for pin in (True, False):
        arm = "pin" if pin else "nopin"
        keys = [("thrash_r3", N_THRASH), ("thrash_r4", N_THRASH), ("thrash_r6", N_THRASH)]
        args = {"items_uri": [uri_entry(k, n) for k, n in keys], "M": 2, "pinning": pin, "workers": 8,
                "model": {"kind": "open_local", "name": "qwen", "max_model_len": 40960}}
        spec = p0.base(f"p1-thrash-{arm}", "A", f"thrash-{arm}", "thrash", "r3+r4+r6", "opa.tracks.track_a:run_thrash", args,
                       {"agent_model": p0.QWEN}, p0.vllm_compute(16), hyp=("H10",), seed=0)
        items = sum(n for _, n in keys)
        sec = START + items * per
        work = (f"{START} s start-up + {items} items x {per:.1f} s (avg {sum(steps)/3:.1f} steps x ({in_tok_step} in tok / {P_AGG:.0f} + "
                f"{out_tok_step} out tok / {D_AGG:.0f} tok/s))")
        notes = ["NOT MEASURED: no thrash run has happened; steps = 0.8 x the step cap 4r+4, prompt ~2,500 tokens per step (assumed)",
                 "M = 2 (cells r3, r4, r6 are above the slot count)"]
        out.append(p0.finish(spec, sec, work, START, items=items, notes=notes))
    return out


def stage_b_order(h: int) -> list[str]:
    return (["p1-skeleton-lateh2"] if h == 2 else []) + ["p1-skeleton-lb", "p1-skeleton-mt", f"p1-lb-late-h{h}", f"p1-lb-late-h{h}-rep1",
                                                           f"p1-lb-late-h{h}-rep2", f"p1-lb-forward-mixed-h{h}", "p1-mt-t4-arms",
                                                           "p1-drift-d2", "p1-drift-d4", "p1-thrash-pin", "p1-thrash-nopin"]


STAGE_A_ORDER = ["p1-embed-pilot", "p1-skeleton-lateh3", "p1-g0-s18-h3-recheck"]      # already run / reserved: spend is in the ledger


def order(h: int | None) -> list[str]:
    """Reservation order: stage B for h only (stage A and Track B are in the ledger). h = None: the stage A record."""
    return stage_b_order(h) if h else list(STAGE_A_ORDER)


def plan(h: int | None = None):
    """Entries (spec, meta) in launch order with envelopes assigned. h = None: the stage A jobs (all E1, record only). Otherwise stage B for h:
    E1 while (E1 estimates so far + max_usd) <= E1_HEADROOM, then E2 for this job and every later one."""
    jobs = stage_a() if h is None else stage_b(h)
    by = {m["name"]: (s, m) for s, m in jobs}
    head = E1_AT_STAGE_A if h is None else E1_HEADROOM
    spent, env, out = 0.0, "E1", []
    for n in order(h):
        s, m = by[n]
        if h is None:
            assert spent + m["max"] <= head + 1e-9, (n, spent, m["max"])
        elif env == "E1" and spent + m["max"] > head + 1e-9:
            env, spent = "E2", 0.0
        m["envelope"] = s["envelope"] = env
        s.setdefault("notes", []).append(f"headroom frozen at generation: E1 {head:.2f}, E2 {E2_APPROVED:.2f}; "
                                         + ("" if h is None else "stage A and Track B spend are in the ledger; ")
                                         + "envelope assigned greedily in launch order")
        spent += m["est"]
        out.append((s, m))
    return out


BUDGET = {"E1": E1_HEADROOM, "E2": E2_APPROVED}
BUDGET_STAGE_A = {"E1": E1_AT_STAGE_A, "E2": E2_APPROVED}


def reservation_check(entries, budget_by_env: dict | None = None) -> list[dict]:
    """Per envelope, in launch order: estimates of that envelope's earlier jobs + this job's max_usd must fit the envelope."""
    budget_by_env = budget_by_env or BUDGET
    rows, spent = [], {"E1": 0.0, "E2": 0.0}
    for _, m in entries:
        e = m["envelope"]
        need = spent[e] + m["max"]
        rows.append({"name": m["name"], "envelope": e, "prior_est": round(spent[e], 2), "max": m["max"], "need": round(need, 2),
                     "budget": budget_by_env[e], "ok": need <= budget_by_env[e] + 1e-9})
        spent[e] += m["est"]
    return rows


# ----------------------------------------------------------------------------------------------- output


def unresolved(text: str) -> list[str]:
    return p0.unresolved(text)


def dump(spec: dict, subs: dict[str, str]) -> bytes:
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000)
    for k in sorted(subs, key=len, reverse=True):
        if subs[k]:
            text = text.replace(k, subs[k])
    return text.encode("utf-8")


DUMMY = {k: f"dummy-{k.lower()}" for k in ALL_PH}


def filled(spec: dict) -> dict:
    return yaml.safe_load(dump(spec, DUMMY))


def write_launch(jobs, ids: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "run_ids.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    known.update({k: v for k, v in ids.items() if v})
    f.write_bytes((json.dumps(known, indent=1, sort_keys=True) + "\n").encode())
    subs = {k: known.get(k) for k in ALL_PH}
    written, skipped = [], {}
    for spec, _ in jobs:
        if spec is None:
            continue
        data = dump(spec, subs)
        left = unresolved(data.decode())
        if left:
            skipped[spec["name"]] = left
            continue
        (out_dir / f"{spec['name']}.yaml").write_bytes(data)
        written.append(spec["name"])
    return written, skipped


def print_table(title: str, entries, budget_by_env: dict | None = None) -> None:
    budget_by_env = budget_by_env or BUDGET
    print(f"\n== {title}")
    print(f"{'job':<26}{'env':>4}{'items':>6}{'est s':>8}{'est $':>8}{'max $':>8}")
    for _, m in entries:
        print(f"{m['name']:<26}{m['envelope']:>4}{m['items']:>6}{m['seconds']:>8.0f}{m['est']:>8.2f}{m['max']:>8.2f}")
    for e in ("E1", "E2"):
        ms = [m for _, m in entries if m["envelope"] == e]
        if ms:
            print(f"TOTAL {e}: estimate ${sum(m['est'] for m in ms):.2f}  max ${sum(m['max'] for m in ms):.2f}   (budget ${budget_by_env[e]:.2f})")
    print("reservation check (estimates of earlier jobs in the envelope + this job's max_usd):")
    for r in reservation_check(entries, budget_by_env):
        print(f"  {r['name']:<26} {r['envelope']} prior est {r['prior_est']:>6.2f} + max {r['max']:>5.2f} = {r['need']:>6.2f} / {r['budget']:.2f}  {'ok' if r['ok'] else 'OVER'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-h", type=int, choices=(2, 3), help="generate stage B for this primary h (decided by p1-g0-s18-h3-recheck)")
    ap.add_argument("--vectors-run-id"), ap.add_argument("--skeleton-lb-run-id"), ap.add_argument("--skeleton-mt-run-id")
    ap.add_argument("--skeleton-lateh3-run-id"), ap.add_argument("--skeleton-lateh2-run-id")
    a = ap.parse_args(argv)
    variants = {None: plan(None), 2: plan(2), 3: plan(3)}
    for entries in variants.values():                                     # every variant must validate
        for spec, _ in entries:
            if spec is not None:
                errs = launch.validate(filled(spec))
                assert not errs, (spec["name"], errs)
    write = [e for e in variants[a.primary_h] if e[0] is not None]
    for spec, _ in write:
        (HERE / f"{spec['name']}.yaml").write_bytes(dump(spec, {}))
    ids = {PH_VEC: a.vectors_run_id, PH_SK_LB: a.skeleton_lb_run_id, PH_SK_MT: a.skeleton_mt_run_id, PH_SK_H3: a.skeleton_lateh3_run_id,
           PH_SK_LATE_H2: a.skeleton_lateh2_run_id}
    if any(ids.values()):
        written, skipped = write_launch(write, ids, HERE / "launch")
        print(f"jobs/launch: {len(written)} specs written; refused (unresolved placeholders): {skipped or 'none'}")
    print(f"templates written: {[s['name'] for s, _ in write]}")
    print_table("STAGE A record (already run / reserved)", variants[None], BUDGET_STAGE_A)
    print_table("STAGE B, primary h = 2 (E1 headroom 2.33 after current reservations)", variants[2])
    print_table("STAGE B, primary h = 3 (E1 headroom 2.33 after current reservations)", variants[3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
