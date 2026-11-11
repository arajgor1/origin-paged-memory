"""Price the REMAINING work of PLAN_V3 from MEASURED unit costs only (free, local; never calls Modal).

  python -m opa.analysis.plan_costs --out bundle/tables/plan_costs.csv

Every input is in PARAMS below (value + source). Money = container seconds x RATE ($/s, L4 + 2 cores + 16 GiB priced at the limits,
launch.rate_per_second) + START_USD per vLLM container. Lines are grouped by phase and envelope (CLAUDE.md phase table; the P1 overflow
is charged to E2 by ruling). Basis column: "measured" (a measured per-item or per-step number), "assumption" (a number with no
measurement; named in the note), "TBD" (no estimate yet; $0 placeholder). Scenario column: "base" lines are counted; lines with
counted = "alt" are alternatives not in the base total (H5 delta 5, the training lengths other than the base L, the conditional
top-up cell); the CLI prints the base totals and one total per training length L.

Unit costs and where they come from (state/unit_costs.md and runs/*/result.json sub-run wall_seconds, see MEASURED_RUNS):
  Track A, LB late h2 32K, 50 items, 8 workers: S1 619 s, S2 46 s, S8 854 s, S11 921 s, S18 568 s (run ...-a0c95f6-4168), S4 2,607 s
  (fixed S4, run ...-1d7070d-f13c; the workers-8 value). At 16 workers (run ...-55d852f-141a, 48 items): S4 1,634.80 s = 34.06 s/item,
  priced via lb_w16_measured for concurrency 16 and 24. MT T4 h3 cross, 50 items: S12 1,091, S14 1,374, S15 1,083, S16 1,332, S18 1,142 s (run ...-1aaa09a-db29);
  these stand in for the MT T8 cell. Skeleton $0.0336/item (lateh3 rebuild, 50 items $1.681 incl. start-up), embed $0.0018/item
  (A-embed 60 items $0.114), thrash $0.0015 per pinned+unpinned pair (two runs of 90 items, $0.139 + $0.128, minus two start-ups),
  drift D2 $0.0224 and D4 $0.0029 per session (30 sessions, $0.737 and $0.153, minus a start-up), remedies 19.5 s/item (48 items, 936 s).
  Track B (mini-LoRA pilot): V3 21 s per optimizer step, 314 s load + preflight + step-0 evals, 171 s final eval of 32 items
  (= 5.34 s per eval item for the full monitoring suite); B1 5 s per step, total 494 s for 39 steps (eval 3.1 s/item derived, assumes a
  100 s load). The eval price per item is the largest uncertainty of P4/P5: the pilot suite (attribution, passkey, perplexity, mass) is
  not the same work as an H11/H12 accuracy eval. EVAL_SCALE multiplies every eval line.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import json

from .. import PROJECT_DIR, budget, launch

RATE = launch.rate_per_second({"kind": "gpu", "gpu": "L4", "image": "vllm", "cpu": 2.0, "memory_gib": 16}, budget.load_prices())

# ------------------------------------------------------------------------------------------------ PARAMETERS
PARAMS = {
    # money
    "rate_usd_per_s": RATE,                                  # 0.000345 (launch.rate_per_second, limits priced)
    "start_usd": 0.066,                                      # vLLM start-up per container (189 s x rate), state/unit_costs.md
    "max_container_s": 8500,                                  # (3 - 0.066) / RATE: a container's estimate stays <= $3; longer runs split (one more start-up each)
    # Track A seconds per item (measured wall_seconds / items)
    "s_per_item_lb": {"S1": 619 / 50, "S2": 46 / 50, "S8": 854 / 50, "S11": 921 / 50, "S18": 568 / 50, "S4": 2607 / 50},
    "s_per_item_mt": {"S12": 1091 / 50, "S14": 1374 / 50, "S15": 1083 / 50, "S16": 1332 / 50, "S18": 1142 / 50},
    "skeleton_usd_per_item": 0.0336,                         # measured, lateh3 rebuild at 12 workers incl. start-up share (concurrency 8)
    # skeletons measured at 24 workers, NET of one $0.066 start-up each, item-weighted over all five 24-worker runs (SKELETON_RUNS_W24 below)
    "skeleton_net_usd_per_item_w24": (1.322 + 0.814 + 0.879 + 1.2506 + 2.4708 - 5 * 0.066) / (50 + 30 + 30 + 50 + 100),
    # measured at 16 workers (LB systems only), sub-runs of A-S2+S8+S11+S4-lbmhqa-late-h2-32k-a0-dev-s0-20261107202540-55d852f-141a (state/decisions.md "THROUGHPUT RESULTS")
    "lb_w16_measured": {"S2": 0.9, "S8": 13.8, "S11": 14.4,
                        "S4": 1634.80 / 48},   # S4: sub-run "S4/r0" of that run, wall_seconds 1634.80 over n 48 items at workers 16 (0 errors, s4_harness_failure_rate 0.0, s4-v2) = 34.06 s/item
    "lb_w16_source": "A-S2+S8+S11+S4-lbmhqa-late-h2-32k-a0-dev-s0-20261107202540-55d852f-141a",
    # MT arms measured at 24 workers were SLOWER than at 8 (S12 31.6 vs 21.8 s/item, S14 30.9 vs 27.5): they stay at the workers-8 numbers
    "mt_w24_measured_slower": {"S12": (31.6, 21.8), "S14": (30.9, 27.5)},
    "concurrency": 8,                                        # 8 (pilot default) | 16 | 24: selects measured seconds/item from the bench runs
    "embed_usd_per_item": 0.0018,
    "thrash_usd_per_pair": (0.139 + 0.128 - 2 * 0.066) / 90,
    "drift_usd_per_session": {"D2": (0.737 - 0.066) / 30, "D4": (0.153 - 0.066) / 30},
    "remedies_s_per_item": 19.5,
    # Track B seconds (mini-LoRA pilot)
    "v3_load_s": 314, "v3_step_s": 21, "b1_load_s": 100, "b1_step_s": 5,
    "eval_s_item": {"V3": 171 / 32, "A1": 171 / 32, "A3": 171 / 32, "B1": (494 - 39 * 5 - 100) / 64},
    "eval_load_s": 200,                                      # ASSUMPTION: model load per eval container (the pilot's 314 s included step-0 evals)
    "eval_scale": 1.0,
    "n_monitor": 64,                                         # monitoring-set items per learning-curve / stopping-rule eval
    "n_g2": 200,                                             # fresh dev set for G2
    # sample sizes (bundle/tables/pilot_dev/sample_sizes_proposed.csv; see check_sample_sizes)
    "n": {"H1": 1191, "S4": 391, "H3": 400, "H5_delta3": 696, "H5_delta5": 249, "tost": 191, "H10": 860, "H2": 200,
          "attribution": 2451, "thread_confusion": 940, "H11": 5885, "H13": 5885, "R5": 400},
    # assumptions with no measurement (named in the line notes)
    "n_trainfree": 400, "trainfree_variants": 4, "u6_u8_usd": 0.05, "h7_s": 600,
    "trackc_micro_s": 3600, "gemma_factor": 1.6, "gemma_start_usd": 0.14, "n_gemma_h1": 300, "n_gemma_h5": 200,
    "n_embrob": 400, "embrob_usd_per_item": 0.170 / 50, "n_bsweep": 200, "n_repro": 200,
    "h3_cell": "b", "h3_cell_factor": {"a": 2.2, "b": 1.0},   # cell (a) M = 2 makes ~2-2.4x the calls; (b) default M
    "L_values": [400, 800, 1200, 2000], "L_base": 800,
    "natural_benchmarks_usd": 0.0,                           # TBD (protocol 3.4)
}

MEASURED_RUNS = {   # run-id fragment -> {sub-run: (wall_seconds, items)} used for the Track A numbers above (checked in the tests)
    "lbmhqa-late-h2-32k-a0-dev-s0-20261107154535-a0c95f6-4168": {"S1/r0": (619, 50), "S2/r0": (46, 50), "S8/r0": (854, 50),
                                                                 "S11/r0": (921, 50), "S18/r0": (568, 50)},
    "lbmhqa-late-h2-32k-a0-dev-s0-20261107185331-1d7070d-f13c": {"S4/r0": (2607, 50)},
    "mtlbmhqa-T4-h3-32k-cross-dev-s0-20261107163638-1aaa09a-db29": {"S12/r0": (1091, 50), "S14/r0": (1374, 50), "S15/r0": (1083, 50),
                                                                    "S16/r0": (1332, 50), "S18/r0": (1142, 50)},
}
SAMPLE_SIZE_KEYS = {   # PARAMS["n"] key -> (csv key, n_proposed)
    "H1": ("S11_vs_S2_lbmhqa_late_32k_primary", 1191), "H5_delta3": ("A2_vs_A1_agent_mtlbmhqa_32k_primary_cross|delta=3", 696),
    "H5_delta5": ("A2_vs_A1_agent_mtlbmhqa_32k_primary_cross|delta=5", 249), "tost": ("A2_vs_A3_agent_equivalence", 191),
    "H10": ("pinned_vs_unpinned_repeat_pagein_reduction", 860), "H2": ("D4_vs_D2_retention_t50", 200),
    "attribution": ("V3_vs_B1_source_attribution", 2451), "thread_confusion": ("V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k", 940),
    "H11": ("V3_paged_vs_B1_full_accuracy_lbmhqa_32k", 5885), "H13": ("copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k", 5885),
    "R5": ("R5_vs_R1_accuracy", 400),
}
SKELETON_RUNS_W24 = {   # run-id fragment -> (compute_usd, items): the five skeleton runs at workers 24 (all measured; none excluded)
    "late-h2-32k-a0-dev-s0-20261107140747-99677c4-97eb": (1.322, 50), "late-h2-32k-a0-dev-s0-20261107184143-516dfae-88f9": (0.814, 30),
    "late-h3-32k-a0-dev-s0-20261107184144-516dfae-2477": (0.879, 30), "mtlbmhqa-T4-h3-32k-cross-dev-s0-20261107140751-99677c4-251f": (1.2506, 50),
    "forward-h2-32k-a0+mixed-h2-32k-a0-dev-s0-20261107140749-99677c4-4dac": (2.4708, 100)}
BENCH_JOBS = {"lb": "bench-lb-w{c}", "mt": "bench-mt-w{c}"}     # job names of jobs/make_bench_jobs.py (read from runs/*/job.yaml)
FIELDS = ["phase", "envelope", "line", "formula", "units", "unit", "usd", "basis", "scenario", "counted", "note"]


# ------------------------------------------------------------------------------------------------ line builders
def _row(phase, env, line, formula, units, unit, usd, basis, scenario="base", counted="yes", note="", containers=0):
    return {"containers": containers, "phase": phase, "envelope": env, "line": line, "formula": formula, "units": units, "unit": unit, "usd": round(usd, 4),
            "basis": basis, "scenario": scenario, "counted": counted, "note": note}


def vllm(phase, env, line, parts, *, basis="measured", scenario="base", counted="yes", note="", factor=1.0):
    """parts: [(label, seconds_per_item, n_items)] in one container group; start-ups = ceil(seconds / max_container_s)."""
    sec = factor * sum(s * n for _, s, n in parts)
    k = max(1, math.ceil(sec / PARAMS["max_container_s"]))
    usd = sec * RATE + k * PARAMS["start_usd"]
    formula = " + ".join(f"{lbl} {n} x {s:.2f} s" for lbl, s, n in parts)
    formula = (f"({formula})" + (f" x {factor}" if factor != 1 else "") + f" = {sec:.0f} s x ${RATE:.6f}/s + {k} x ${PARAMS['start_usd']}")
    return _row(phase, env, line, formula, sum(n for _, _, n in parts), "items", usd, basis, scenario, counted, note, containers=k)


def per_item(phase, env, line, label, n, unit_usd, *, basis="measured", scenario="base", counted="yes", note=""):
    return _row(phase, env, line, f"{n} x ${unit_usd:.4f} ({label})", n, "items", n * unit_usd, basis, scenario, counted, note)


def seconds_line(phase, env, line, formula_parts, seconds, *, basis, units=1, unit="job", scenario="base", counted="yes", note="", extra_usd=0.0):
    return _row(phase, env, line, f"{formula_parts} = {seconds:.0f} s x ${RATE:.6f}/s" + (f" + ${extra_usd:.3f}" if extra_usd else ""),
                units, unit, seconds * RATE + extra_usd, basis, scenario, counted, note)


def train(phase, env, line, load_s, step_s, steps, n_monitor, eval_s, *, scenario="base", counted="yes", note="", n_evals=None):
    n_evals = n_evals or max(1, steps // 200)
    sec = load_s + steps * step_s + PARAMS["eval_scale"] * n_evals * n_monitor * eval_s
    r = seconds_line(phase, env, line, f"{load_s} s load + {steps} steps x {step_s} s + {n_evals} evals x {n_monitor} items x {eval_s:.2f} s",
                     sec, basis="measured", units=steps, unit="opt steps", scenario=scenario, counted=counted, note=note)
    r["eval_usd"] = PARAMS["eval_scale"] * n_evals * n_monitor * eval_s * RATE
    return r


def evaluation(phase, env, line, model, n, *, basis="assumption", note=""):
    s = PARAMS["eval_s_item"][model]
    sec = PARAMS["eval_load_s"] + PARAMS["eval_scale"] * n * s
    r = seconds_line(phase, env, line, f"{PARAMS['eval_load_s']} s load + {n} items x {s:.2f} s ({model} eval)", sec, basis=basis,
                     units=n, unit="eval items", note=note or "eval price per item from the Track B pilot's final eval (171 s / 32 items for V3)")
    r["eval_usd"] = PARAMS["eval_scale"] * n * s * RATE
    return r


def load_bench(concurrency: int, runs_dir: Path | None = None) -> dict:
    """Measured seconds per item at `concurrency` workers from the throughput-benchmark runs (job names bench-lb-w<c>, bench-mt-w<c>), when
    present: {"lb": {system: s/item}, "mt": {...}}. Each sub-run's wall_seconds / n; missing runs give empty dicts."""
    out = {"lb": {}, "mt": {}}
    seen: dict[str, str] = {}
    d = runs_dir or PROJECT_DIR / "runs"
    if not d.exists():
        return out
    for run in sorted(d.iterdir()):
        try:
            name = __import__("yaml").safe_load((run / "job.yaml").read_text(encoding="utf-8")).get("name", "")
            res = json.loads((run / "result.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        for kind, pat in BENCH_JOBS.items():
            if name == pat.format(c=concurrency) and res.get("status") == "ok":
                if name in seen:
                    raise ValueError(f"duplicate bench runs for {name}: {seen[name]} and {run.name}; keep one (move the other aside) so the choice is explicit")
                seen[name] = run.name
                for sub, v in (res.get("metrics", {}).get("runs") or {}).items():
                    if v.get("n"):
                        out[kind][sub.split("/")[0]] = v["wall_seconds"] / v["n"]
    return out


# ------------------------------------------------------------------------------------------------ the plan
def build_lines(p: dict | None = None, concurrency: int | None = None, bench: dict | None = None) -> list[dict]:
    """concurrency 8 = the pilot's workers-8 numbers. 16 / 24: seconds per item of the systems the bench runs measured at that worker count
    (when the runs exist; the rest stay at the workers-8 value) and, for skeletons, the 24-worker measurement net of start-up (+ a start-up
    per skeleton line); at 16 only systems measured at 16 change but skeletons use the 24-worker price only at 24."""
    p = p or PARAMS
    conc = int(concurrency or p.get("concurrency", 8))
    lb, mt = dict(p["s_per_item_lb"]), dict(p["s_per_item_mt"])
    if conc != 8:
        b = bench if bench is not None else load_bench(conc)
        lb.update(b["lb"]), mt.update(b["mt"])
    n = p["n"]
    use_net = conc == 24
    sk, em = (p["skeleton_net_usd_per_item_w24"] if use_net else p["skeleton_usd_per_item"]), p["embed_usd_per_item"]

    def sk_line(ph, e, line, nn, **kw):
        if not use_net:
            return per_item(ph, e, line, "skeleton", nn, sk, **kw)
        r = per_item(ph, e, line, "skeleton, 24 workers, net of start-up", nn, sk, **kw)
        r["usd"] = round(r["usd"] + p["start_usd"], 4)
        r["formula"] += f" + ${p['start_usd']} start-up"
        return r
    out: list[dict] = []
    A = out.append

    # P1 remaining (charged to E2)
    ph, e = "P1 remaining", "E2"
    A(per_item(ph, e, "H5 T8 oracle: embed 50 items", "embed", 50, em))
    A(sk_line(ph, e, "H5 T8 oracle: skeletons 50 items", 50) if use_net else
      _row(ph, e, "H5 T8 oracle: skeletons 50 items", f"50 x ${sk} + ${p['start_usd']}", 50, "items", 50 * sk + p["start_usd"], "measured"))
    A(vllm(ph, e, "H5 T8 oracle: S18 on 50 items", [("S18", mt["S18"], 50)], note="MT T4 numbers stand in for T8"))
    A(vllm(ph, e, "H5 T8 conditional arms S12/S14/S15/S16 (only if S18 >= 0.60 on T8)",
           [(s, mt[s], 50) for s in ("S12", "S14", "S15", "S16")], counted="yes",
           note="conditional; counted in the base total (worst case)"))
    topup = (60 * em + 30 * sk + p["start_usd"]) + vllm(ph, e, "", [("S11", lb["S11"] * 2.2, 30), ("S18", lb["S18"] * 2.2, 30)])["usd"]
    A(_row(ph, e, "H3 qualifying-cell top-up (one more candidate cell, 30 items; only if needed)",
           "embed 60 x $0.0018 + skeleton 30 x $0.0336 + $0.066 + (S11 + S18 at M=2 x2.2) 30 items", 30, "items", topup, "assumption",
           counted="alt", note="not in the base total"))

    # P2 test split (Track A), E2
    ph, e = "P2 test split", "E2"
    A(_row(ph, e, "test-data generation (CPU, local)", "0", 0, "items", 0.0, "measured", note="generators run on the local machine"))
    A(per_item(ph, e, "H1 cell late h2: embed", "embed", n["H1"], em))
    A(sk_line(ph, e, "H1 cell late h2: skeletons", n["H1"], note="needed by S8 and S11"))
    A(vllm(ph, e, "H1 cell late h2: S11, S2, S8 at n_H1", [(s, lb[s], n["H1"]) for s in ("S11", "S2", "S8")]))
    A(vllm(ph, e, "H1 cell late h2: S4 at n_S4", [("S4", lb["S4"], n["S4"])], note="S4_vs_S11 n 6,452 is not priced (default option n_S4 = 391)"))
    f = p["h3_cell_factor"][p["h3_cell"]]
    A(per_item(ph, e, f"H3 R2 cell {p['h3_cell']}: embed", "embed", n["H3"], em, basis="assumption", note="n_H3 400 pooled over the chosen cell(s); cell TBD"))
    A(sk_line(ph, e, f"H3 R2 cell {p['h3_cell']}: skeletons", n["H3"], basis="assumption"))
    A(vllm(ph, e, f"H3 R2 cell {p['h3_cell']}: S11 + S18 at n_H3", [("S11", lb["S11"], n["H3"]), ("S18", lb["S18"], n["H3"])], factor=f, basis="assumption",
           note=f"LB late h2 per-item seconds x {f} (cell (a) M=2 ~2.2x)"))
    A(_row(ph, e, "H3 S11 - S8 (late-bound leaf)", "from the H1 cell run: 0", 0, "items", 0.0, "measured", note="S8 runs in the H1 cell"))
    for tag, nn, cnt in (("delta=3 (n 696)", n["H5_delta3"], "yes"), ("delta=5 (n 249)", n["H5_delta5"], "alt")):
        sc = f"H5 {tag}"
        A(per_item(ph, e, f"H5 agent arms [{tag}]: embed", "embed", nn, em, scenario=sc, counted=cnt))
        A(sk_line(ph, e, f"H5 agent arms [{tag}]: skeletons", nn, scenario=sc, counted=cnt))
        A(vllm(ph, e, f"H5 agent arms [{tag}]: S12, S14 at n_H5", [("S12", mt["S12"], nn), ("S14", mt["S14"], nn)], scenario=sc, counted=cnt,
               note="MT T4 per-item seconds stand in for the chosen MT cell"))
    A(vllm(ph, e, "H5 TOST arm S15 at n_tost", [("S15", mt["S15"], n["tost"])], note="same items as the H5 arms"))
    A(_row(ph, e, "H8 descent vs flat recall (CPU)", "0", 0, "items", 0.0, "measured"))
    A(_row(ph, e, "H10 thrash pin + nopin at n_H10", f"{n['H10']} pairs x ${p['thrash_usd_per_pair']:.4f} + 2 x ${p['start_usd']}", n["H10"], "pairs",
           n["H10"] * p["thrash_usd_per_pair"] + 2 * p["start_usd"], "measured"))
    for c in ("D2", "D4"):
        A(_row(ph, e, f"H2 drift {c} at n_H2", f"{n['H2']} x ${p['drift_usd_per_session'][c]:.4f} + ${p['start_usd']}", n["H2"], "sessions",
               n["H2"] * p["drift_usd_per_session"][c] + p["start_usd"], "measured"))
    A(_row(ph, e, "natural benchmarks (protocol 3.4)", "TBD", 0, "items", p["natural_benchmarks_usd"], "TBD", note="placeholder; no estimate yet"))

    # P3 (E3)
    ph, e = "P3", "E3"
    A(train(ph, e, "learning-curve check: V3 400 steps + evals at 50/100/200/400", p["v3_load_s"], p["v3_step_s"], 400, p["n_monitor"], p["eval_s_item"]["V3"],
            n_evals=4, note="PLAN_V3 6a step 1: monitoring evals at steps 50, 100, 200, 400"))
    A(_row(ph, e, "U6-U8 unit tests", "assumption", 3, "tests", p["u6_u8_usd"], "assumption", note="U1-U5 cost $0.011 on L4"))
    A(_row(ph, e, f"remedies at full n (R5_vs_R1 fixed {n['R5']} items)",
           f"{n['R5']} x {p['remedies_s_per_item']} s x ${RATE:.6f}/s + ${p['start_usd']}", n["R5"], "items",
           n["R5"] * p["remedies_s_per_item"] * RATE + p["start_usd"], "measured"))
    A(evaluation(ph, e, "training-free variants V0, V1, B2, B4", "V3", p["n_trainfree"] * p["trainfree_variants"],
                 note=f"ASSUMPTION: {p['n_trainfree']} items per variant x {p['trainfree_variants']} variants at the V3 eval price"))
    A(seconds_line(ph, e, "H7 precision in the kernel", f"{p['h7_s']} s microbenchmark", p["h7_s"], basis="assumption", note="no measurement"))

    # P4 (E4)
    ph, e = "P4", "E4"
    for L in p["L_values"]:
        sc, cnt = f"L={L}", "yes" if L == p["L_base"] else "alt"
        A(train(ph, e, f"V3 to L={L} steps", p["v3_load_s"], p["v3_step_s"], L, p["n_monitor"], p["eval_s_item"]["V3"], scenario=sc, counted=cnt))
        A(train(ph, e, f"B1 to L={L} steps", p["b1_load_s"], p["b1_step_s"], L, p["n_monitor"], p["eval_s_item"]["B1"], scenario=sc, counted=cnt))
        for arm in ("A1", "A3"):
            A(train(ph, e, f"{arm} to L={L} steps (if G2 passes)", p["v3_load_s"], p["v3_step_s"], L, p["n_monitor"], p["eval_s_item"][arm],
                    scenario=sc, counted=cnt, note="same dense-bias cost per step as V3 (assumed)"))
    A(evaluation(ph, e, "G2 fresh dev eval: V3", "V3", p["n_g2"]))
    A(evaluation(ph, e, "G2 fresh dev eval: B1", "B1", p["n_g2"]))
    for model, items, what in (("V3", n["attribution"] + n["thread_confusion"] + n["H11"], "attribution + thread confusion + H11"),
                               ("B1", n["attribution"] + n["thread_confusion"] + n["H11"], "attribution + thread confusion + H11"),
                               ("A1", n["thread_confusion"], "thread confusion"), ("A3", n["thread_confusion"], "thread confusion")):
        A(evaluation(ph, e, f"evals {model}: {what}", model, items))

    # P5 (E5)
    ph, e = "P5", "E5"
    A(seconds_line(ph, e, "Track C L4 microbenchmarks H9", f"{p['trackc_micro_s']} s", p["trackc_micro_s"], basis="assumption", note="no measurement"))
    A(seconds_line(ph, e, "Track C H13 reload timing", f"{p['trackc_micro_s']} s", p["trackc_micro_s"], basis="assumption", note="no measurement"))
    A(evaluation(ph, e, "H13 accuracy: recompute, copy+remedy, naive reuse", "V3", 3 * n["H13"],
                 note="ASSUMPTION: 3 arms x n_H13 at the V3 eval price; the CacheBlend-style arm is BOOST-only and not priced"))
    g = p["gemma_factor"]
    A(vllm(ph, e, "Gemma replication: H1 primaries (S11, S2, S8) at reduced n", [(s, lb[s], p["n_gemma_h1"]) for s in ("S11", "S2", "S8")], factor=g,
           basis="assumption", note=f"Qwen seconds x {g} (Gemma ~8B); Qwen skeletons/vectors reused; plus Gemma start-up ${p['gemma_start_usd']}"))
    A(vllm(ph, e, "Gemma replication: H5 primaries (S12, S14) at reduced n", [(s, mt[s], p["n_gemma_h5"]) for s in ("S12", "S14")], factor=g, basis="assumption"))
    A(_row(ph, e, "Gemma start-up (405 s measured)", f"${p['gemma_start_usd']}", 1, "container", p["gemma_start_usd"], "measured"))
    A(_row(ph, e, "embedder robustness: Qwen3-Embedding-4B embeds + S2", f"{p['n_embrob']} x ${p['embrob_usd_per_item']:.4f} + S2 {p['n_embrob']} x {lb['S2']:.2f} s",
           p["n_embrob"], "items", p["n_embrob"] * (p["embrob_usd_per_item"] + lb["S2"] * RATE), "measured", note="A-embedcheck $0.170 for 50 items"))
    A(_row(ph, e, "B sweep 1024, 2048 at 32K: embeds, skeletons, S11", f"2 x {p['n_bsweep']} x (embed + skeleton + S11)", 2 * p["n_bsweep"], "items",
           2 * p["n_bsweep"] * (em + sk + lb["S11"] * RATE) + 2 * p["start_usd"] + (2 * p["start_usd"] if use_net else 0.0), "assumption",
           counted="alt", note="DROPPED from the base package (owner, 2026-11-07); 2 S11 start-ups, plus 2 skeleton start-ups when skeletons are priced net of start-up"))
    A(vllm(ph, e, "reproducibility re-run: H1 (S11, S2) and H5 (S12, S14) primaries", [("S11", lb["S11"], p["n_repro"]), ("S2", lb["S2"], p["n_repro"]),
                                                                                       ("S12", mt["S12"], p["n_repro"]), ("S14", mt["S14"], p["n_repro"])],
           basis="assumption", note=f"{p['n_repro']} items each; reuses P2 vectors and skeletons"))

    # P6 (E6)
    A(_row("P6", "E6", "bundle assembly, verification, reconcile", "CPU / free", 0, "-", 0.0, "measured", note="compute ~ $0"))
    return out


# ------------------------------------------------------------------------------------------------ totals and checks
def totals(rows: list[dict]) -> dict:
    """Base totals per envelope (counted == yes, scenario base or the base L / delta rows), and one total per training length."""
    base = {}
    for r in rows:
        if r["counted"] == "yes":
            base[r["envelope"]] = base.get(r["envelope"], 0.0) + r["usd"]
    per_L = {}
    for L in PARAMS["L_values"]:
        d = dict(base)
        if L != PARAMS["L_base"]:
            d["E4"] = d.get("E4", 0.0) - sum(r["usd"] for r in rows if r["scenario"] == f"L={PARAMS['L_base']}") + sum(r["usd"] for r in rows if r["scenario"] == f"L={L}")
        per_L[L] = d
    return {"base": base, "per_L": per_L}


def current_state() -> dict:
    st = budget.envelope_states()
    return {"approved": {k: v.approved for k, v in st.items()}, "committed": sum(v.spent + v.reserved for v in st.values()),
            "remaining": {k: v.remaining for k, v in st.items()}}


def check_sample_sizes(csv_path: Path | None = None) -> list[str]:
    """Differences between PARAMS['n'] and bundle/tables/pilot_dev/sample_sizes_proposed.csv (empty when consistent or the file is absent)."""
    f = csv_path or PROJECT_DIR / "bundle" / "tables" / "pilot_dev" / "sample_sizes_proposed.csv"
    if not f.exists():
        return []
    rows = {r["key"]: r for r in csv.DictReader(f.open(encoding="utf-8", newline=""))}
    bad = []
    for k, (key, want) in SAMPLE_SIZE_KEYS.items():
        got = rows.get(key, {}).get("n_proposed")
        if got is None or int(float(got)) != PARAMS["n"][k] or want != PARAMS["n"][k]:
            bad.append(f"{k}: params {PARAMS['n'][k]} csv {got}")
    return bad


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bundle/tables/plan_costs.csv")
    ap.add_argument("--concurrency", type=int, choices=(8, 16, 24), default=8,
                    help="workers: 8 = pilot numbers; 16/24 read the bench runs (bench-lb-w<c>, bench-mt-w<c>) when present")
    ap.add_argument("--package", choices=("96", "126", "full", "owner"), help="fill a costed-request package greedily by priority tier (see opa.analysis.plan_packages)")
    a = ap.parse_args(argv)
    if a.package:
        from . import plan_packages
        out = a.out if a.out != "bundle/tables/plan_costs.csv" else f"bundle/tables/plan_package_{a.package}.csv"
        try:
            plan_packages.run(a.package, PROJECT_DIR / out if not Path(out).is_absolute() else Path(out))
        except ValueError as e:                      # e.g. duplicate bench runs: say so plainly instead of a traceback
            print(f"ERROR: {e}")
            return 2
        return 0
    rows = build_lines(concurrency=a.concurrency)
    write_csv(rows, PROJECT_DIR / a.out if not Path(a.out).is_absolute() else Path(a.out))
    print(f"{'phase':<14}{'env':>4} {'usd':>9}  {'basis':<12}{'cnt':<4} line")
    for r in rows:
        print(f"{r['phase']:<14}{r['envelope']:>4} {r['usd']:>9.2f}  {r['basis']:<12}{r['counted']:<4} {r['line']}" + (f"  [{r['scenario']}]" if r['scenario'] != 'base' else ""))
    t, cur = totals(rows), current_state()
    print("\nBASE totals per envelope vs approved (counted lines):")
    for e in sorted(t["base"]):
        ap_ = cur["approved"].get(e, 0.0)
        print(f"  {e}: ${t['base'][e]:8.2f}   approved ${ap_:.2f}   remaining now ${cur['remaining'].get(e, 0.0):.2f}")
    tot = sum(t["base"].values())
    print(f"  ALL: ${tot:.2f}   committed so far ${cur['committed']:.2f}   cumulative after the plan ${cur['committed'] + tot:.2f}   cap $130.00")
    print("Per training length L (E4 and total):")
    for L, d in t["per_L"].items():
        print(f"  L={L:<5} E4 ${d.get('E4', 0):8.2f}   plan total ${sum(d.values()):8.2f}   cumulative ${cur['committed'] + sum(d.values()):8.2f}")
    diff = check_sample_sizes()
    print("sample sizes vs bundle/tables/pilot_dev/sample_sizes_proposed.csv:", diff or "consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
