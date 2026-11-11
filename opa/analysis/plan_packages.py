"""Costed-request packages (owner instructions 2026-11-07, PLAN_V3 revisions 3-4): fill a package greedily, tier by tier, from MEASURED unit costs.

  python -m opa.analysis.plan_costs --package {96,126,full} [--out bundle/tables/plan_package_<name>.csv]

MONEY. Base money left = $100 - (ORCH + E0..E6 committed, spent + reserved, from opa.budget); BOOST = $30 - committed BOOST. By ruling BOOST pays
its eligible lines FIRST and base money pays everything else in priority order (the literal base-first order is printed as the alternative;
BOOST needs the owner's approval, which the request asks for). BOOST may fund only PLAN_V3 Section 9 uses (the rest of a package comes from a RAISE pool, money beyond the cap, 0 for \"96\"): (b) more test items for the H1, H5 and H12 primary cells, and (d)
longer training (the stopping-rule training beyond L=400); a line's remainder goes to BOOST only if it is such a line. Packages: "96" = what the
base money plus BOOST can hold today (cap - committed); "126" = $126.00 of plan spend (cap minus a $4 safety), "full" = everything; when the
base money is not enough the output says "needs owner to raise the cap by $X" (the amount of non-BOOST money beyond the base money left).
RESERVATION SLACK. Every job is split into containers whose estimate is <= $3 (one $0.066 start-up each, priced; a split eval container is
not charged a second model load). The launcher reserves max_usd = 2 x estimate, so each envelope keeps a reserve >= its largest container's
estimate (= max_usd - estimate); the fill stops before using that reserve, and the slack line shows it (held back from base money).
PRICES (final ruling). Skeletons at 24 workers (five measured runs, item-weighted, net of start-up, $0.0246/item). S2, S8 and S11 at 16 workers (measured
in A-S2+S8+S11+S4-...-141a: 0.9, 13.8, 14.4 s/item); S4 too (1,634.80 s / 48 items = 34.06 s/item, S4/r0 of the same run); S1, S18 and the MT arms stay at the workers-8 numbers (MT at 24 workers was slower: S12 31.6 vs 21.8,
S14 30.9 vs 27.5). Failed bench runs are ignored (load_bench reads status ok only). Track B eval prices (5.34 s per V3
eval item from the pilot's final eval) are the largest uncertainty: sensitivity rows give the package total at EVAL_SCALE 2x and 3x.

STRICT PRIORITY (owner ruling): the default fill walks the items in order and stops at the first one that does not fit; the money left is shown as
unallocated and is never spent on lower-priority items. The greedy-skip result is printed as an alternative (a lower-priority item may be included
when a higher-priority one does not fit, EXCEPT that an
evaluation of a trained model is excluded when the training it needs (V3 + B1 to the stopping rule, G2) is excluded (REQUIRES).

Priority tiers (within a tier, in this order):
 1  H1 (S11 vs S2 at n=1,191 for delta 5, S8
    on the same items for H3's S11 - S8, S4 at n_S4=391; falls back to delta 7 = 607; the delta 7 row is always printed); the H5 agent arms on MT T4
    default M at delta 5 (n=249, S12 and S14) with S15 for the TOST (191), with option rows delta 3 = 696, "if the M=2 cell qualifies" (same n, x2.2) and the OPTIONAL blind M=2 cell check (30 items, S16 at M=2 x2.2; not in the default
    packages: routed_rate is already 0.83 < 0.90 on the T4 default-M cell, so H5 stays on T4 default M).
 2  learning curve (400 V3 steps, E3), the cheap Track B items (U6-U8, remedies at full n for R5 vs R1, training-free variants, H7), V3 + B1 to the
    stopping rule (L=800 base; L=400/1,200/2,000 as option rows), the G2 fresh-dev eval, then the H5 MODEL-level arm (A1/A3 training at L + the
    thread-confusion eval, n=940 per arm; conditional on G2 passing; its steps beyond L=400 are BOOST use (d)), H12 thread confusion (940 per arm),
    the H4 test-split attribution eval (2,451 per arm).
 3  H2 drift (200 sessions), H10 thrash (860 pairs). (H3's R2 leaf is accepted as inconclusive, owner ruling: no R2 cell is priced; H3 keeps only
    S11 - S8, shared with the H1 cell.)
 4  H11 at the 3-point margin (n=2,614 per arm; rules file now -3.0), H13 at its 2-point margin priced at ~1,000 questions (3 arms), Track C H9.
 5  Gemma replication, embedder robustness, reproducibility re-run of two primaries. The page-size (B) sweep is DROPPED from the base package
    (excluded option row).
 6  (lowest) 64K H3 predictor-only check on 30 LB late-h2 64K items: embed, skeletons and S11 at 64K, all priced x2 tokens/time vs 32K (ASSUMPTION);
    it would also need a 64K vLLM config and 64K data generation (CPU).
POWER. Design power inverts the sample-size formulas of opa.stats (Connor McNemar, TOST, paired-normal with the Wilcoxon ARE 0.955) at the planned
n. For rules with a non-zero min_delta threshold the table adds P(pass) = P(point estimate >= threshold AND, where the rule requires it, the 95% CI
excludes zero) at the PILOT effect (an upper bound: the Holm correction is not applied); S11 - S4 has no CI/p condition, so it reports
P(observed >= 3 points) = Phi((delta - 3) / SE), SE = sqrt((p_d - delta^2) / n), at the pilot S4-v2 effect and at delta = 3.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

from .. import PROJECT_DIR, budget
from . import plan_costs as pc

CAP = 130.0
BASE_CAP, BOOST_CAP = 100.0, 30.0
BASE_IDS = ("ORCH", "E0", "E1", "E2", "E3", "E4", "E5", "E6")
PACKAGES = ("96", "126", "full", "owner")
_PHI = NormalDist()
INF = float("inf")
CONTAINER_MAX_USD = 3.0
WILCOXON_ARE = 0.955
# an evaluation of a trained model needs the training: dependants of an excluded item are excluded with it
REQUIRES = {"g2_eval": ("g2_train",), "a1a3": ("g2_train", "g2_eval"), "h12": ("g2_train",), "h4": ("g2_train",), "h11": ("g2_train",), "h13": ("g2_train",)}
GREEDY_RULE = "Strict priority: the fill stops at the first item that does not fit; leftover money is unallocated. Greedy-skip (a lower-priority item may be included when a higher one does not fit) is printed as an alternative."

# primary contrasts: key -> (label, method, delta, p_d, sd, n_required)   (sample_sizes_proposed.csv; H11 3-point and H13 below)
PRIMARIES = {
    "S11_vs_S2": ("H1 S11 - S2 (delta 5)", "mcnemar", 0.05, 0.38, None, 1191),
    "S11_vs_S4": ("H1 S11 - S4 (min delta 3)", "min_delta", 0.03, 0.50, None, None),
    "S11_vs_S8": ("H3 S11 - S8 late-bound (delta 2)", "mcnemar", 0.02, 0.02, None, 391),
    "A2_vs_A1_agent": ("H5 agent A2 - A1 (delta 5)", "mcnemar", 0.05, 0.08, None, 249),
    "A2_vs_A3_tost": ("H5 agent A2 ~ A3 TOST (margin 3)", "tost", 0.03, 0.02, None, 191),
    "H10": ("H10 pinned vs unpinned page-ins", "paired_normal", 0.5, None, 5.114, 860),
    "H2": ("H2 D4 vs D2 retention at t50", "paired_normal", 10.0, None, 9.072, 200),
    "H4_attr": ("H4 V3 vs B1 source attribution", "mcnemar", 0.05, 0.78125, None, 2451),
    "H12": ("H12 V3 vs B1 thread confusion", "mcnemar", 0.05, 0.30, None, 940),
    "H5_model": ("H5 model A2 vs A1 thread confusion", "mcnemar", 0.05, 0.30, None, 940),
    "H11": ("H11 V3 paged vs B1 full (NI, 3 points)", "mcnemar", 0.03, 0.30, None, 2614),
    "H13": ("H13 copy-reuse vs recompute (NI, 2 points)", "mcnemar", 0.02, 0.312, None, 6120),
}
# rules with a non-zero min_delta threshold (prereg/rules.yaml): threshold in the contrast's own units, whether a CI/p condition is required,
# the pilot effect and the discordance behind it, and the scenarios shown. Effects are fractions (0.10 = 10 points) except H2 (points).
RULES = {
    "S11_vs_S2": dict(thr=0.05, sig=True, scen=[(0.10, 0.38, "pilot effect 10 pts (CI -6..26), p_d 0.38")]),
    "S11_vs_S4": dict(thr=0.03, sig=False, scen=[(0.46, 0.50, "pilot S4 v2: n10 24, n01 1 -> delta 0.46, p_d 0.50"), (0.03, 0.50, "delta = 3 (threshold)")],
                      n_fixed=391),
    "S11_vs_S8": dict(thr=0.02, sig=True, scen=[(0.02, 0.02, "pilot effect 2 pts")], flag="p_d rests on 1 discordant pair of 50: UNRELIABLE"),
    "H2": dict(thr=10.0, sig=True, scen=[(2.667, None, "pilot effect 2.7 pts (CI -0.3..6.0)")]),
}
PRIMARY_RULE_NOTE = "McNemar / non-inferiority power: Connor approximation of opa.stats.n_mcnemar inverted at the planned n (p_d from the pilot, else 0.30; H13 uses 0.312 from the remedies pilot)"


def power(method: str, n: int, delta: float, p_d: float | None, sd: float | None, alpha: float = 0.05) -> float:
    if n <= 0:
        return 0.0
    if method in ("mcnemar", "min_delta"):
        if p_d <= delta ** 2:
            return float("nan")
        z = (delta * math.sqrt(n) - _PHI.inv_cdf(1 - alpha / 2) * math.sqrt(p_d)) / math.sqrt(p_d - delta ** 2)
        return _PHI.cdf(z)
    if method == "tost":
        z2 = delta * math.sqrt(n / p_d) - _PHI.inv_cdf(1 - alpha)
        return max(0.0, 2 * _PHI.cdf(z2) - 1)
    if method == "paired_normal":                         # Wilcoxon signed-rank has ARE 0.955 against the paired t: effective n x 0.955
        return _PHI.cdf(delta * math.sqrt(n * WILCOXON_ARE) / sd - _PHI.inv_cdf(1 - alpha / 2))
    raise ValueError(method)


def p_pass(key: str, n: int, effect: float, p_d: float | None, alpha: float = 0.05) -> float:
    """P(rule passes) at a true effect: point estimate >= threshold and, where required, estimate > z_{1-alpha/2} x SE (Holm not applied)."""
    r = RULES[key]
    if n <= 0:
        return 0.0
    if key == "H2":
        se = PRIMARIES["H2"][4] / math.sqrt(n * WILCOXON_ARE)
    else:
        if p_d <= effect ** 2:
            return float("nan")
        se = math.sqrt((p_d - effect ** 2) / n)
    cut = max(r["thr"], _PHI.inv_cdf(1 - alpha / 2) * se) if r["sig"] else r["thr"]
    return _PHI.cdf((effect - cut) / se)


def money_state() -> tuple[float, float, float]:
    """(base money left, BOOST left, committed total) from the ledger: base = $100 - ORCH/E0..E6 committed, BOOST = $30 - BOOST committed."""
    st = budget.envelope_states()
    c_base = sum(v.spent + v.reserved for k, v in st.items() if k in BASE_IDS)
    c_boost = sum(v.spent + v.reserved for k, v in st.items() if k == "BOOST")
    return BASE_CAP - c_base, BOOST_CAP - c_boost, sum(v.spent + v.reserved for v in st.values())


def package_pools(name: str) -> tuple[float, float, float]:
    """(base money, BOOST, raise): the raise pool is money beyond the cap the owner would have to add; 0 for "96"."""
    base_left, boost_left, _ = money_state()
    if name == "96":
        return base_left, boost_left, 0.0
    if name == "126":
        return base_left, boost_left, max(0.0, 126.0 - (base_left + boost_left))
    return base_left, boost_left, INF


# ------------------------------------------------------------------------------------------------ catalog
def context(concurrency: int = 24, bench: dict | None = None, p: dict | None = None) -> SimpleNamespace:
    p = p or pc.PARAMS
    lb, mt = dict(p["s_per_item_lb"]), dict(p["s_per_item_mt"])
    b = {"lb": {}, "mt": {}}
    if concurrency != 8:                                      # final pricing ruling: S2/S8/S11 at 16 workers (measured), the rest at workers-8
        lb.update(p.get("lb_w16_measured", {}))
        b = bench if bench is not None else {"lb": dict(p.get("lb_w16_measured", {})), "mt": {}}
        lb.update(b["lb"]), mt.update(b["mt"])
    use_net = concurrency == 24
    return SimpleNamespace(p=p, n=p["n"], lb=lb, mt=mt, em=p["embed_usd_per_item"], use_net=use_net, bench=b, bench_used=bool(b["lb"] or b["mt"]),
                           sk=p["skeleton_net_usd_per_item_w24"] if use_net else p["skeleton_usd_per_item"], conc=concurrency)


def sk_row(x, ph, env, line, n, **kw):
    r = pc.per_item(ph, env, line, "skeleton, 24 workers, net" if x.use_net else "skeleton", n, x.sk, **kw)
    if x.use_net:
        r["usd"] = round(r["usd"] + x.p["start_usd"], 4)
        r["formula"] += f" + ${x.p['start_usd']} start-up"
        r["containers"] = 1
    return r


def finish_rows(rows, p):
    """Split every row into containers whose estimate is <= $3 (one start-up each, priced) and record containers and the largest container estimate."""
    start = p["start_usd"]
    for r in rows:
        if r["usd"] <= 0:
            r["containers"], r["container_est"] = 0, 0.0
            continue
        have = r.get("containers", 0)
        work = max(r["usd"] - have * start, 0.0)
        k = max(1, math.ceil(work / (CONTAINER_MAX_USD - start)), have)
        r["usd"] = round(work + k * start, 4)
        r["containers"], r["container_est"] = k, r["usd"] / k
    return rows


def cost(rows) -> float:
    return sum(r["usd"] for r in rows)


def alt(label, rows, p, pn=None, boost=None, elig=None):
    """boost: None | "b" (more test items for H1/H5/H12) | "d" (stopping-rule training beyond L=400); elig = BOOST-eligible dollars of the alt."""
    finish_rows(rows, p)
    c = cost(rows)
    env_est: dict[str, float] = {}
    for r in rows:
        env_est[r["envelope"]] = max(env_est.get(r["envelope"], 0.0), r["container_est"])
    why = {"b": "PLAN_V3 s9 use (b): more test items for the H1, H5 and H12 primary cells",
           "d": "PLAN_V3 s9 use (d): longer training (stopping-rule steps beyond L=400)"}.get(boost, "")
    return {"label": label, "rows": rows, "usd": c, "primary_n": pn or {}, "env_container_est": env_est, "boost": boost, "boost_why": why,
            "boost_elig": 0.0 if boost is None else (c if elig is None else elig), "eval_usd": sum(r.get("eval_usd", 0.0) for r in rows)}


def catalog(x) -> list[dict]:
    p, n, lb, mt, em = x.p, x.n, x.lb, x.mt, x.em
    V3, B1 = "V3", "B1"
    I = []

    def item(tier, key, name, env, alts, options=(), primaries=()):
        I.append({"tier": tier, "key": key, "name": name, "env": env, "alts": alts, "options": list(options), "primaries": list(primaries),
                  "requires": list(REQUIRES.get(key, ()))})

    # ---- tier 1
    ph, e = "P2 test split", "E2"
    nb = 30
    chk_rows = lambda: [pc.per_item(ph, e, "H5 cell check: embed 30 items", "embed", nb, em),
                        sk_row(x, ph, e, "H5 cell check: skeletons 30 items", nb),
                        pc.vllm(ph, e, "H5 cell check: S16 (A0, lambda 0) at M=2, 30 items", [("S16", mt["S16"], nb)], factor=2.2, basis="assumption",
                                note="M=2 shrinks the read window 3 -> 1: ~2.2x the calls, as cell (a)")]
    def h1(nn):
        ph, e = "P2 test split", "E2"
        return [pc.per_item(ph, e, f"H1 cell late h2 ({nn}): embed", "embed", nn, em), sk_row(x, ph, e, f"H1 cell late h2 ({nn}): skeletons", nn),
                pc.vllm(ph, e, f"H1 cell late h2: S11, S2, S8 at n={nn}", [(s, lb[s], nn) for s in ("S11", "S2", "S8")]),
                pc.vllm(ph, e, f"H1 cell late h2: S4 at n_S4={n['S4']}", [("S4", lb["S4"], n["S4"])])]
    d5 = alt("delta 5, n=1,191", h1(n["H1"]), p, {"S11_vs_S2": n["H1"], "S11_vs_S4": n["S4"], "S11_vs_S8": n["H1"]}, boost="b")
    d7 = alt("delta 7, n=607", h1(607), p, {"S11_vs_S2": 607, "S11_vs_S4": n["S4"], "S11_vs_S8": 607}, boost="b")
    item(1, "H1", "H1 (+ H3 S11 - S8 leaf)", "E2", [d5, d7], options=[{**d7, "label": "option: delta 7, n=607 (always shown)"}],
         primaries=["S11_vs_S2", "S11_vs_S4", "S11_vs_S8"])

    def h5(nn, fac=1.0):
        return [pc.per_item(ph, e, f"H5 agent arms n={nn}: embed", "embed", nn, em), sk_row(x, ph, e, f"H5 agent arms n={nn}: skeletons", nn),
                pc.vllm(ph, e, f"H5 agent arms n={nn}: S12, S14", [("S12", mt["S12"], nn), ("S14", mt["S14"], nn)], factor=fac),
                pc.vllm(ph, e, f"H5 TOST arm S15 at n_tost={n['tost']}", [("S15", mt["S15"], n["tost"])], factor=fac)]
    item(1, "H5", "H5 agent arms (delta 5) + TOST on MT T4 default M", "E2",
         [alt("delta 5, n=249, TOST 191", h5(n["H5_delta5"]), p, {"A2_vs_A1_agent": n["H5_delta5"], "A2_vs_A3_tost": n["tost"]}, boost="b")],
         options=[alt("OPTIONAL: blind M=2 cell check, MT T4 h3 cross, 30 fresh dev items 50-79, S16 at M=2 (not in the default packages: routed_rate is already 0.83 < 0.90 on the T4 default-M cell, so H5 stays on T4 default M)",
                      chk_rows(), p),
                  alt("option: delta 3, n=696", h5(n["H5_delta3"]), p, {"A2_vs_A1_agent": n["H5_delta3"], "A2_vs_A3_tost": n["tost"]}, boost="b"),
                  alt("variant: if the M=2 cell qualifies (hit rate < 0.90), same n, systems x2.2", h5(n["H5_delta5"], 2.2), p,
                      {"A2_vs_A1_agent": n["H5_delta5"], "A2_vs_A3_tost": n["tost"]}, boost="b")],
         primaries=["A2_vs_A1_agent", "A2_vs_A3_tost"])

    # ---- tier 2
    ph = "P3"
    item(2, "learning_curve", "learning-curve check (V3, 400 steps)", "E3",
         [alt("400 steps", [pc.train(ph, "E3", "learning-curve check: V3 400 steps + evals at 50/100/200/400", p["v3_load_s"], p["v3_step_s"], 400, p["n_monitor"],
                                     p["eval_s_item"][V3], n_evals=4)], p)])
    item(2, "u6u8", "U6-U8 unit tests", "E3", [alt("3 tests", [pc._row(ph, "E3", "U6-U8 unit tests", "assumption", 3, "tests", p["u6_u8_usd"], "assumption")], p)])
    item(2, "remedies", "remedies at full n (R5 vs R1, 400 items)", "E3",
         [alt("n=400", [pc._row(ph, "E3", "remedies at full n", f"{n['R5']} x {p['remedies_s_per_item']} s x ${pc.RATE:.6f}/s + ${p['start_usd']}", n["R5"], "items",
                                n["R5"] * p["remedies_s_per_item"] * pc.RATE + p["start_usd"], "measured", containers=1)], p)])
    item(2, "trainfree", "training-free variants V0, V1, B2, B4", "E3",
         [alt("4 variants", [pc.evaluation(ph, "E3", "training-free variants", V3, p["n_trainfree"] * p["trainfree_variants"])], p)])
    item(2, "h7", "H7 precision in the kernel", "E3", [alt("microbenchmark", [pc.seconds_line(ph, "E3", "H7 precision in the kernel", f"{p['h7_s']} s", p["h7_s"], basis="assumption")], p)])

    def g2_train(L):
        return [pc.train("P4", "E4", f"V3 to L={L}", p["v3_load_s"], p["v3_step_s"], L, p["n_monitor"], p["eval_s_item"][V3]),
                pc.train("P4", "E4", f"B1 to L={L}", p["b1_load_s"], p["b1_step_s"], L, p["n_monitor"], p["eval_s_item"][B1])]
    c400 = cost(finish_rows(g2_train(400), p))
    base_alt = alt(f"L={p['L_base']}", g2_train(p["L_base"]), p, boost="d")
    base_alt["boost_elig"] = max(0.0, base_alt["usd"] - c400)                # only the steps beyond L=400 are BOOST use (d)
    item(2, "g2_train", "V3 + B1 to the stopping rule", "E4", [base_alt],
         options=[alt(f"option: L={L}", g2_train(L), p) for L in p["L_values"] if L != p["L_base"]])
    item(2, "g2_eval", "G2 fresh-dev eval", "E4", [alt(f"n={p['n_g2']}", [pc.evaluation("P4", "E4", "G2 eval V3", V3, p["n_g2"]), pc.evaluation("P4", "E4", "G2 eval B1", B1, p["n_g2"])], p)])
    def a1a3_rows(L):
        return [pc.train("P4", "E4", f"{a} to L={L}", p["v3_load_s"], p["v3_step_s"], L, p["n_monitor"], p["eval_s_item"][a]) for a in ("A1", "A3")]
    a1a3 = alt(f"L={p['L_base']}", a1a3_rows(p["L_base"]) + [pc.evaluation("P4", "E4", f"{a} thread-confusion eval", a, n["thread_confusion"]) for a in ("A1", "A3")],
               p, {"H5_model": n["thread_confusion"]}, boost="d")
    a1a3["boost_elig"] = max(0.0, cost(finish_rows(a1a3_rows(p["L_base"]), p)) - cost(finish_rows(a1a3_rows(400), p)))        # steps beyond L=400 only
    item(2, "a1a3", "H5 model level: A1/A3 training at L + thread-confusion evals (n=940 per arm; conditional on G2 passing)", "E4", [a1a3], primaries=["H5_model"])
    item(2, "h12", "H12 thread confusion (V3, B1)", "E4",
         [alt("n=940 per arm", [pc.evaluation("P4", "E4", "H12 eval V3", V3, n["thread_confusion"]), pc.evaluation("P4", "E4", "H12 eval B1", B1, n["thread_confusion"])],
              p, {"H12": n["thread_confusion"]}, boost="b")], primaries=["H12"])
    item(2, "h4", "H4 test-split attribution (V3, B1; judged by G2)", "E4",
         [alt("n=2,451 per arm", [pc.evaluation("P4", "E4", "H4 attribution eval V3", V3, n["attribution"]), pc.evaluation("P4", "E4", "H4 attribution eval B1", B1, n["attribution"])],
              p, {"H4_attr": n["attribution"]})], primaries=["H4_attr"])

    # ---- tier 3
    ph, e = "P2 test split", "E2"
    item(3, "h2", "H2 drift D2 + D4 (200 sessions)", "E2",
         [alt("n=200", [pc._row(ph, e, f"H2 drift {c}", f"{n['H2']} x ${p['drift_usd_per_session'][c]:.4f} + ${p['start_usd']}", n["H2"], "sessions",
                                n["H2"] * p["drift_usd_per_session"][c] + p["start_usd"], "measured", containers=1) for c in ("D2", "D4")], p, {"H2": n["H2"]})],
         primaries=["H2"])
    item(3, "h10", "H10 thrash (860 pairs)", "E2",
         [alt("n=860", [pc._row(ph, e, "H10 thrash pin + nopin", f"{n['H10']} pairs x ${p['thrash_usd_per_pair']:.4f} + 2 x ${p['start_usd']}", n["H10"], "pairs",
                                n["H10"] * p["thrash_usd_per_pair"] + 2 * p["start_usd"], "measured", containers=2)], p, {"H10": n["H10"]})], primaries=["H10"])

    # ---- tier 4
    n11 = 2614
    item(4, "h11", "H11 at the 3-point margin (n=2,614 per arm)", "E4",
         [alt("n=2,614 per arm", [pc.evaluation("P4", "E4", "H11 eval V3 (paged)", V3, n11), pc.evaluation("P4", "E4", "H11 eval B1 (full)", B1, n11)], p, {"H11": n11})],
         options=[alt("option: 2-point margin, n=5,885 per arm", [pc.evaluation("P4", "E4", "H11 eval V3 (2 pt)", V3, n["H11"]), pc.evaluation("P4", "E4", "H11 eval B1 (2 pt)", B1, n["H11"])],
                      p, {"H11": n["H11"]})], primaries=["H11"])
    n13 = 1000
    item(4, "h13", "H13 at ~1,000 questions (2-point margin)", "E5",
         [alt("n=1,000, 3 arms", [pc.evaluation("P5", "E5", "H13 accuracy: recompute, copy+remedy, naive reuse", V3, 3 * n13, note="3 arms x 1,000 at the V3 eval price"),
                                  pc.seconds_line("P5", "E5", "Track C H13 reload timing", f"{p['trackc_micro_s']} s", p["trackc_micro_s"], basis="assumption")], p, {"H13": n13})],
         primaries=["H13"])
    item(4, "h9", "Track C H9 (copy vs prefill) microbenchmark", "E5",
         [alt("microbenchmark", [pc.seconds_line("P5", "E5", "Track C L4 microbenchmark H9", f"{p['trackc_micro_s']} s", p["trackc_micro_s"], basis="assumption")], p)])

    # ---- tier 5
    ph, e = "P5", "E5"
    g = p["gemma_factor"]
    item(5, "gemma", "Gemma replication (H1/H5 primaries, reduced n)", e,
         [alt("reduced n", [pc.vllm(ph, e, "Gemma H1 primaries", [(s, lb[s], p["n_gemma_h1"]) for s in ("S11", "S2", "S8")], factor=g, basis="assumption"),
                            pc.vllm(ph, e, "Gemma H5 primaries", [(s, mt[s], p["n_gemma_h5"]) for s in ("S12", "S14")], factor=g, basis="assumption"),
                            pc._row(ph, e, "Gemma start-up", f"${p['gemma_start_usd']}", 1, "container", p["gemma_start_usd"], "measured", containers=1)], p)])
    item(5, "embrob", "embedder robustness (Qwen3-Embedding-4B + S2)", e,
         [alt(f"n={p['n_embrob']}", [pc._row(ph, e, "embedder robustness", f"{p['n_embrob']} x (${p['embrob_usd_per_item']:.4f} + S2)", p["n_embrob"], "items",
                                             p["n_embrob"] * (p["embrob_usd_per_item"] + lb["S2"] * pc.RATE), "measured")], p)])
    item(5, "repro", "reproducibility re-run of two primaries (H1, H5)", e,
         [alt(f"n={p['n_repro']}", [pc.vllm(ph, e, "reproducibility re-run", [("S11", lb["S11"], p["n_repro"]), ("S2", lb["S2"], p["n_repro"]),
                                                                              ("S12", mt["S12"], p["n_repro"]), ("S14", mt["S14"], p["n_repro"])], basis="assumption")], p)])
    n64 = 30
    item(6, "h3_64k", "64K H3 predictor-only check, 30 items (needs a 64K vLLM config and 64K data generation on CPU)", "E2",
         [alt("30 LB late-h2 64K items", [pc.per_item("P2 test split", "E2", "64K H3 check: embed 30 items (x2 tokens)", "embed", n64, 2 * em, basis="assumption"),
                                          sk_row(x, "P2 test split", "E2", "64K H3 check: skeletons 30 items (x2 tokens = 60 item-equivalents)", 2 * n64, basis="assumption"),
                                          pc.vllm("P2 test split", "E2", "64K H3 check: S11 at 64K, 30 items", [("S11", lb["S11"], n64)], factor=2.0, basis="assumption",
                                                  note="ASSUMPTION: x2 tokens/time vs 32K")], p)])
    item(5, "bsweep", "page-size (B) sweep (DROPPED from the base package)", e, [],
         options=[alt("option: B in {1024, 2048} at 32K", [pc._row(ph, e, "B sweep 1024, 2048 at 32K", f"2 x {p['n_bsweep']} x (embed + skeleton + S11)", 2 * p["n_bsweep"], "items",
                                                                   2 * p["n_bsweep"] * (em + x.sk + lb["S11"] * pc.RATE) + 4 * p["start_usd"], "assumption", containers=4)], p)])
    return I


# ------------------------------------------------------------------------------------------------ the fill
def fill(base_pool: float, boost_pool: float, raise_pool: float = 0.0, concurrency: int = 24, bench: dict | None = None, p: dict | None = None,
         boost_first: bool = True, strict: bool = True) -> dict:
    """Greedy fill. Money order (default, ruling): BOOST pays the eligible lines first (use (b) H1/H5/H12 test items, use (d) stopping-rule training
    beyond L=400); base money pays everything else in priority order; the rest comes from the raise pool (money beyond the cap). The reserve
    (largest container estimate per envelope) must fit inside base money, else it also draws on the raise pool. boost_first=False is the literal
    base-first order, printed as the alternative. strict=True (default, owner ruling: the order is a PRIORITY order) walks the items in order and STOPS
    at the first one that fits in none of its stated sizes; money left over is reported as unallocated and never spent on lower-priority items.
    strict=False is the greedy-skip alternative (a lower-priority item may be included when a higher one does not fit)."""
    x = context(concurrency, bench, p)
    base_used = boost_used = raise_used = 0.0
    env_est: dict[str, float] = {}
    inc, exc, opts = [], [], []
    stopped = None
    cat = catalog(x)
    for it in cat:
        opts += [{**it, "option": o} for o in it["options"]]
    for it in cat:
        if not it["alts"]:
            continue
        if stopped:
            exc.append({**it, "why": f"not reached (stopped at {stopped['name']})"})
            continue
        chosen = None
        missing = [r for r in it["requires"] if r not in {i["key"] for i in inc}]
        for a in ([] if missing else it["alts"]):
            merged = dict(env_est)
            for e, v in a["env_container_est"].items():
                merged[e] = max(merged.get(e, 0.0), v)
            reserve = sum(merged.values())
            c = a["usd"]
            boost_avail = min(a["boost_elig"], boost_pool - boost_used)
            if boost_first:
                r1 = max(0.0, min(c, boost_avail))
                b = max(0.0, min(c - r1, base_pool - base_used - reserve))
            else:
                b = max(0.0, min(c, base_pool - base_used - reserve))
                r1 = max(0.0, min(c - b, boost_avail))
            r2 = c - b - r1
            shortfall = max(0.0, base_used + b + reserve - base_pool)          # reserve not covered by base money
            if raise_used + r2 + shortfall <= raise_pool + 1e-9:
                chosen = (a, b, r1, r2, merged)
                break
        if chosen:
            a, b, r1, r2, env_est = chosen
            base_used, boost_used, raise_used = base_used + b, boost_used + r1, raise_used + r2
            inc.append({**it, "chosen": a, "reduced": a is not it["alts"][0], "base_usd": b, "boost_usd": r1, "raise_usd": r2})
        else:
            exc.append({**it, "why": ("needs " + ", ".join(missing)) if missing else "does not fit"})
            if strict:
                stopped = {"key": it["key"], "name": it["name"], "need": it["alts"][-1]["usd"], "tier": it["tier"]}
    by_env: dict[str, float] = {}
    for it in inc:
        for row in it["chosen"]["rows"]:
            by_env[row["envelope"]] = by_env.get(row["envelope"], 0.0) + row["usd"]
    planned = {}
    for it in inc:
        planned.update(it["chosen"]["primary_n"])
    base_left, boost_left, committed = money_state()
    reserve = sum(env_est.values())
    total = base_used + boost_used + raise_used
    cap_raise = raise_used + max(0.0, base_used + reserve - base_pool)
    return {"base_pool": base_pool, "boost_pool": boost_pool, "raise_pool": raise_pool, "concurrency": concurrency, "included": inc, "excluded": exc,
            "options": opts, "total": total, "base_used": base_used, "boost_used": boost_used, "raise_used": raise_used, "by_env": by_env,
            "committed": committed, "cumulative": committed + total, "base_left": base_left, "boost_left": boost_left, "reserve_by_env": env_est,
            "stopped": stopped, "unallocated_base": (max(0.0, base_pool - base_used - reserve) if base_pool != INF else 0.0),
            "unallocated_boost": max(0.0, boost_pool - boost_used), "strict": strict,
            "reserve": reserve, "eval_usd": sum(it["chosen"]["eval_usd"] for it in inc), "cap_raise": cap_raise, "bench_used": x.bench_used,
            "powers": powers_table(inc, exc, planned)}


def powers_table(inc, exc, planned) -> list[dict]:
    out = []
    for key, (label, method, delta, p_d, sd, n_req) in PRIMARIES.items():
        if not any(key in it["primaries"] for it in inc + exc):
            continue
        nn = planned.get(key, 0)
        row = {"key": key, "label": label, "method": method, "n_planned": nn, "n_required": n_req, "excluded": nn == 0, "reduced": bool(n_req and 0 < nn < n_req),
               "power": None if method == "min_delta" else power(method, nn, delta, p_d, sd), "p_pass": [], "flag": ""}
        if key in RULES:
            r = RULES[key]
            row["flag"] = r.get("flag", "")
            for eff, pd_, why in r["scen"]:
                pd_use = PRIMARIES[key][3] if pd_ is None else pd_
                row["p_pass"].append((why, p_pass(key, nn, eff, pd_use)))
        out.append(row)
    return out


def rows_for_csv(name: str, res: dict) -> list[dict]:
    out = []
    for it in res["included"]:
        cu = it["chosen"]["usd"] or 1.0
        share_b, share_o, share_r = it["base_usd"] / cu, it["boost_usd"] / cu, it["raise_usd"] / cu
        for r in it["chosen"]["rows"]:
            out.append({"package": name, "status": "reduced" if it["reduced"] else "included", "tier": it["tier"], "item": it["name"], "alt": it["chosen"]["label"],
                        "envelope": r["envelope"], "line": r["line"], "usd": r["usd"], "base_usd": round(r["usd"] * share_b, 4),
                        "boost_usd": round(r["usd"] * share_o, 4), "raise_usd": round(r["usd"] * share_r, 4), "containers": r.get("containers", 0), "basis": r["basis"], "formula": r["formula"]})
    for it in res["excluded"]:
        a = it["alts"][-1]
        out.append({"package": name, "status": "excluded", "tier": it["tier"], "item": it["name"], "alt": a["label"], "envelope": it["env"],
                    "line": "(excluded)", "usd": round(a["usd"], 4), "base_usd": "", "boost_usd": "", "raise_usd": "", "containers": "", "basis": "",
                    "formula": "would cost this much at its smallest stated n"})
    for o in res["options"]:
        out.append({"package": name, "status": "option", "tier": o["tier"], "item": o["name"], "alt": o["option"]["label"], "envelope": o["env"],
                    "line": "(option row, not included)", "usd": round(o["option"]["usd"], 4), "base_usd": "", "boost_usd": "", "raise_usd": "", "containers": "", "basis": "", "formula": ""})
    return out


CSV_FIELDS = ["package", "status", "tier", "item", "alt", "envelope", "line", "usd", "base_usd", "boost_usd", "raise_usd", "containers", "basis", "formula"]


def print_package(name: str, res: dict) -> None:
    print(f"\n=== package {name} (concurrency {res['concurrency']}; committed now ${res['committed']:.2f}; base money left ${res['base_left']:.2f}, BOOST left ${res['boost_left']:.2f})")
    pp = pc.PARAMS
    print("PRICES: S2/S8/S11/S4 at 16 workers, measured (" + ", ".join(f"{k} {v} s/item" for k, v in pp["lb_w16_measured"].items()) + f"; source run {pp['lb_w16_source']}); "
          "S1, S18 and the MT arms S12/S14/S15/S16 at workers-8 seconds/item (MT at 24 workers was slower: "
          + ", ".join(f"{k} {a} vs {b} s/item" for k, (a, b) in pp["mt_w24_measured_slower"].items()) + "); skeletons at 24 workers, net $"
          f"{pp['skeleton_net_usd_per_item_w24']:.4f}/item (five measured runs)")
    print(GREEDY_RULE)
    print("INCLUDED (base / BOOST):")
    for it in res["included"]:
        bo = f" + BOOST ${it['boost_usd']:.2f} [{it['chosen']['boost_why']}]" if it["boost_usd"] > 1e-9 else ""
        ra = f" + RAISE ${it['raise_usd']:.2f}" if it["raise_usd"] > 1e-9 else ""
        print(f"  T{it['tier']} {it['env']} ${it['chosen']['usd']:8.2f} = base ${it['base_usd']:.2f}{bo}{ra}  {it['name']} [{it['chosen']['label']}]" + ("  (REDUCED)" if it["reduced"] else ""))
    print("EXCLUDED:")
    for it in res["excluded"]:
        print(f"  T{it['tier']} {it['env']} ${it['alts'][-1]['usd']:8.2f}  {it['name']}  [{it.get('why', 'does not fit')}]")
    print("OPTION ROWS (not included):")
    for o in res["options"]:
        print(f"  T{o['tier']} {o['env']} ${o['option']['usd']:8.2f}  {o['name']} -- {o['option']['label']}")
    if res["stopped"]:
        s_ = res["stopped"]
        print(f"STOPPED (strict priority order) at T{s_['tier']} {s_['name']}: needs ${s_['need']:.2f} at its smallest stated size. UNALLOCATED: base ${res['unallocated_base']:.2f} "
              f"(after the reserve slack), BOOST ${res['unallocated_boost']:.2f} (eligible lines only); the money is NOT spent on lower-priority items (next item needs ${s_['need']:.2f}).")
    print("TOTALS: " + "  ".join(f"{e} ${v:.2f}" for e, v in sorted(res["by_env"].items())) + f"  | plan ${res['total']:.2f}  | cumulative committed ${res['cumulative']:.2f} of ${CAP:.0f}")
    print(f"MONEY: base used ${res['base_used']:.2f} of ${res['base_left']:.2f} left | BOOST used ${res['boost_used']:.2f} of ${res['boost_left']:.2f} | "
          f"raise (beyond the cap) ${res['raise_used']:.2f} | "
          + ("; ".join(f"{it['name']}: {it['chosen']['boost_why']}" for it in res["included"] if it["boost_usd"] > 1e-9) or "no BOOST used"))
    print("RESERVATION SLACK (largest container estimate per envelope, held back from base money): " +
          "  ".join(f"{e} ${v:.2f}" for e, v in sorted(res["reserve_by_env"].items())) + f"  = ${res['reserve']:.2f}; containers are split at <= ${CONTAINER_MAX_USD:.0f}")
    if res["cap_raise"] > 0.005:
        print(f"CAP: needs owner to raise the cap by ${res['cap_raise']:.2f} (money beyond base + eligible BOOST, incl. the reserve slack)")
    else:
        print("CAP: fits under the current cap (no raise needed)")
    e = res["eval_usd"]
    print(f"SENSITIVITY (Track B eval cost ${e:.2f} at 1x): at 2x plan ${res['total'] + e:.2f} (+${e:.2f}); at 3x plan ${res['total'] + 2 * e:.2f} (+${2 * e:.2f})")
    greedy = fill(*package_pools(name), res["concurrency"], strict=False)
    inc_keys = {i["key"] for i in res["included"]}
    add = [it["name"] for it in greedy["included"] if it["key"] not in inc_keys]
    print(f"GREEDY ALTERNATIVE (skip what does not fit and continue): plan ${greedy['total']:.2f} (base ${greedy['base_used']:.2f}, BOOST ${greedy['boost_used']:.2f}, "
          f"raise ${greedy['raise_used']:.2f}); would add: {', '.join(add) or 'nothing'}")
    if any(it["key"] == "g2_train" for it in res["excluded"]):
        l400 = fill(*package_pools(name), res["concurrency"], p=dict(pc.PARAMS, L_base=400))
        gain = [it["name"] for it in l400["included"] if it["key"] not in inc_keys]
        fits = any(it["key"] == "g2_train" for it in l400["included"])
        print(f"L=400 MINIMUM ALTERNATIVE (the stopping rule's floor; V3 + B1 at L=400 cost ${next(o for o in res['options'] if o['key'] == 'g2_train' and o['option']['label'].endswith('L=400'))['option']['usd']:.2f}): "
              f"{'fits' if fits else 'still does not fit'}; strict plan ${l400['total']:.2f}; would add: {', '.join(gain) or 'nothing'}. "
              "Valid ONLY if the own-origin fraction rises by less than 0.02 between steps 200 and 400 (the curve plateaus by 400); G2 is still judged after the "
              "stopping rule, so L=400 allows G2/A1/A3/H12/H4/H11/H13 only on that condition and does NOT allow longer training (no BOOST use (d)).")
    alt_res = fill(*package_pools(name), res["concurrency"], boost_first=False)
    gained = [("+" if it["key"] not in {i["key"] for i in res["included"]} else "") + it["name"] for it in alt_res["included"] if it["key"] not in {i["key"] for i in res["included"]}] + ["-" + it["name"] for it in res["included"] if it["key"] not in {i["key"] for i in alt_res["included"]}]
    print(f"ALTERNATIVE ORDER (literal base-first: base pays first, BOOST only the remainder of eligible lines): plan ${alt_res['total']:.2f}, base ${alt_res['base_used']:.2f}, "
          f"BOOST ${alt_res['boost_used']:.2f}, raise ${alt_res['raise_used']:.2f}; differs from the default by: {', '.join(gained) or 'nothing (same items)'}")
    print("POWER (design power at the planned n; P(pass) at the pilot effect, Holm not applied):")
    for r in res["powers"]:
        tag = "EXCLUDED" if r["excluded"] else ("REDUCED" if r["reduced"] else "")
        pw = "n/a (min_delta only)" if r["power"] is None else f"design power {r['power']:.2f}"
        pp = "; ".join(f"P(pass) {v:.2f} at {w}" for w, v in r["p_pass"])
        print(f"  {r['label']:<42} n {r['n_planned']:>5}" + (f" / required {r['n_required']:>5}" if r["n_required"] else "") + f"  {pw} {tag}" + (f"; {pp}" if pp else "") + (f"  [{r['flag']}]" if r["flag"] else ""))


# ------------------------------------------------------------------------------------------------ the owner's package
OWNER_ITEMS = ["H1", "H5", "learning_curve", "u6u8", "remedies", "trainfree", "h7", "g2_train", "g2_eval", "a1a3", "h12", "h4", "h10", "h9"]
OWNER_OUT = ["H2 (not funded: deferred, disclosed)", "H13", "Gemma replication", "embedder robustness", "reproducibility re-run", "B sweep", "64K H3 check"]
LEARNING_CURVE_ESTIMATE = 3.70            # p3-learning-curve-v3 is already running (ledger: reserved 7.40 = 2 x the 3.70 estimate): its estimate counts as committed
OWNER_CAPS = (130.0, 140.0)


ORCH_APPROVED, E6_BUNDLE = 1.00, 0.50       # ORCH stays approved at $1 (the verifier needs ORCH > $0 for its ledger check); E6 gets $0.50 for the bundle step


def owner_package(concurrency: int = 24, bench: dict | None = None, p: dict | None = None, container_cap: float = 2.5) -> dict:
    """See _owner_package; containers are split at an estimate of at most `container_cap` dollars (default $2.50, extra start-ups priced)."""
    global CONTAINER_MAX_USD
    old_cap, old_s = CONTAINER_MAX_USD, pc.PARAMS["max_container_s"]
    CONTAINER_MAX_USD = container_cap
    pc.PARAMS["max_container_s"] = int((container_cap - pc.PARAMS["start_usd"]) / pc.RATE)
    try:
        return _owner_package(concurrency, bench, p)
    finally:
        CONTAINER_MAX_USD, pc.PARAMS["max_container_s"] = old_cap, old_s


def _owner_package(concurrency: int = 24, bench: dict | None = None, p: dict | None = None) -> dict:
    """The owner's choice: "fits 126" with H1 sized for delta 7 (n=607), H5 delta 5 on T4 default M, tier 2 as in 126 (the learning curve is already running),
    H10 and H9 added, H2 deferred, H13 / Gemma / embedder robustness / reproducibility / B sweep / 64K out, H11 (3-point margin) as a separately priced ADD-ON.
    Mapped onto the PLAN_V3 envelopes: base (ORCH + E0..E6) <= $100; BOOST <= $30, only for use (b) (H1/H5/H12 test items) and use (d) (V3/B1/A1/A3 training
    beyond L=400), BOOST paying its eligible lines first in priority order; per-envelope reserve slack = the largest container estimate (containers <= the cap, $2.50)."""
    x = context(concurrency, bench, p)
    cat = {it["key"]: it for it in catalog(x)}
    st = budget.envelope_states()
    committed_env = {k: v.spent + v.reserved for k, v in st.items()}
    open_lc = sum(float(r["amount_usd"]) * (1 if r["kind"] == "reserve" else -1 if r["kind"] == "release" else 0)
                  for r in budget.read_ledger() if r["job"] == "p3-learning-curve-v3")          # its reservation still held (0 once it has finished)
    if open_lc > 1e-9:
        committed_env["E3"] = committed_env.get("E3", 0.0) - open_lc + LEARNING_CURVE_ESTIMATE
    approved = {k: v.approved for k, v in st.items()}
    boost_left = BOOST_CAP - committed_env.get("BOOST", 0.0)
    elig_slack = max(row["container_est"] for k in ("H1", "H5", "h12", "g2_train", "a1a3") for row in (cat[k]["alts"][1] if k == "H1" else cat[k]["alts"][0])["rows"])
    boost_pool = boost_left - elig_slack                         # the BOOST envelope keeps its own reserve slack inside the $30
    lines, boost_used = [], 0.0
    for key in OWNER_ITEMS:
        it = cat[key]
        a = it["alts"][1] if key == "H1" else it["alts"][0]                       # H1 at delta 7 (n=607)
        if key == "learning_curve":
            lines.append({"key": key, "name": it["name"] + " (ALREADY RUNNING: its $3.70 estimate is committed, not new money)", "alt": a, "env": it["env"], "tier": it["tier"],
                          "usd": 0.0, "base": 0.0, "boost": 0.0, "committed_note": LEARNING_CURVE_ESTIMATE})
            continue
        r = max(0.0, min(a["usd"], a["boost_elig"], boost_pool - boost_used))
        boost_used += r
        lines.append({"key": key, "name": it["name"], "alt": a, "env": it["env"], "tier": it["tier"], "usd": a["usd"], "base": a["usd"] - r, "boost": r})
    orch_new = max(0.0, ORCH_APPROVED - committed_env.get("ORCH", 0.0))
    for nm, env, usd in (("ORCH kept approved at $1.00 (the verifier needs ORCH > $0 for its ledger lines)", "ORCH", orch_new),
                         ("P6 bundle step, verification, upload (E6)", "E6", E6_BUNDLE)):
        lines.append({"key": env.lower() + "_fixed", "name": nm, "alt": {"label": "fixed", "rows": [], "boost_why": "", "primary_n": {}}, "env": env, "tier": 6,
                      "usd": usd, "base": usd, "boost": 0.0})
    h11 = cat["h11"]["alts"][0]
    by_env: dict[str, float] = {}
    slack_env: dict[str, float] = {}
    for ln in lines:
        if ln["key"].endswith("_fixed"):
            by_env[ln["env"]] = by_env.get(ln["env"], 0.0) + ln["usd"]
            continue
        if ln["usd"] <= 0:
            continue
        for row in ln["alt"]["rows"]:
            share = ln["base"] / ln["usd"]
            by_env[row["envelope"]] = by_env.get(row["envelope"], 0.0) + row["usd"] * share
            slack_env[row["envelope"]] = max(slack_env.get(row["envelope"], 0.0), row["container_est"])
    boost_slack = max([row["container_est"] for ln in lines if ln["boost"] > 0 for row in ln["alt"]["rows"]] or [0.0])
    h11_env = {"E4": h11["usd"]}
    needed = {}
    for e in ("E0", "E1"):
        needed[e] = committed_env.get(e, 0.0)                         # E0 and E1 shrink to their committed amounts
    needed["ORCH"] = committed_env.get("ORCH", 0.0) + by_env.get("ORCH", 0.0)
    for e in ("E2", "E3", "E4", "E5", "E6"):
        needed[e] = committed_env.get(e, 0.0) + by_env.get(e, 0.0) + (slack_env.get(e, 0.0) if by_env.get(e, 0.0) > 0 else 0.0)
    base_total = sum(needed.values())
    boost_needed = boost_used + boost_slack
    plan_total = sum(l["usd"] for l in lines)
    slack_total = sum(slack_env.get(e, 0.0) for e in ("E2", "E3", "E4", "E5", "E6") if by_env.get(e, 0.0) > 0) + (boost_slack if boost_used > 0 else 0.0)
    committed_total = sum(committed_env.values())
    h11_slack = max(slack_env.get("E4", 0.0), max(r["container_est"] for r in h11["rows"]))
    grand = committed_total + plan_total + slack_total
    grand_h11 = grand + h11["usd"] + (h11_slack - slack_env.get("E4", 0.0))
    # power
    planned = {}
    for ln in lines:
        planned.update(ln["alt"]["primary_n"])
    powers = powers_table([{"primaries": cat[k]["primaries"]} for k in OWNER_ITEMS if k in planned_keys(lines)], [{"primaries": cat[k]["primaries"]} for k in OWNER_ITEMS if k not in planned_keys(lines)], planned)
    h1_extra = {
        "n": 607, "design_power_delta7": power("mcnemar", 607, 0.07, 0.38, None), "power_delta5_at_607": power("mcnemar", 607, 0.05, 0.38, None),
        "p_pass_pilot_effect_10pts": p_pass("S11_vs_S2", 607, 0.10, 0.38), "n_required_delta7": 607,
    }
    return {"lines": lines, "h11": h11, "by_env": by_env, "slack_env": slack_env, "boost_slack": boost_slack, "boost_used": boost_used, "committed_env": committed_env,
            "approved": approved, "needed": needed, "boost_needed": boost_needed, "base_total": base_total, "plan_total": plan_total, "slack_total": slack_total,
            "committed_total": committed_total, "grand": grand, "grand_h11": grand_h11, "h11_extra_needed": h11["usd"], "powers": powers, "h1": h1_extra,
            "boost_left": boost_left, "out": OWNER_OUT, "container_cap": CONTAINER_MAX_USD}


def planned_keys(lines):
    return {ln["key"] for ln in lines}


def print_owner(res: dict) -> None:
    print("\n=== package OWNER (fits 126 with H1 at delta 7, H10 + H9 added, H2 deferred; BOOST pays its eligible lines first)")
    print("PRICES: S2/S8/S11/S4 at 16 workers (measured, run …-141a); S1, S18 and the MT arms at workers-8; skeletons at 24 workers net; containers split at <= $2.50")
    print(f"{'item':<76}{'env':>6}{'$':>9}{'base':>9}{'BOOST':>9}")
    for ln in res["lines"]:
        env = ln["env"]
        lbl = f"T{ln['tier']} {ln['name']} [{ln['alt']['label']}]"
        print(f"{lbl[:75]:<76}{env:>6}{ln['usd']:>9.2f}{ln['base']:>9.2f}{ln['boost']:>9.2f}" + (f"   (BOOST portion on envelope BOOST: {ln['alt']['boost_why']})" if ln["boost"] > 0 else ""))
    print("OUT: " + "; ".join(res["out"]) + "; H11 is an ADD-ON row below")
    h = res["h11"]
    print(f"ADD-ON (not in the totals): H11 at the 3-point margin, n=2,614 per arm: E4 ${h['usd']:.2f} (the 3-point margin is already in the rules file, revision 4)")
    print(f"BY ENVELOPE (new base money): " + "  ".join(f"{e} ${v:.2f}" for e, v in sorted(res["by_env"].items())) + f"   | BOOST ${res['boost_used']:.2f} of ${res['boost_left']:.2f}")
    print("RESERVE SLACK (largest container estimate): " + "  ".join(f"{e} ${v:.2f}" for e, v in sorted(res["slack_env"].items()) if res["by_env"].get(e, 0) > 0) + f"  BOOST ${res['boost_slack']:.2f}"
          f"  = ${res['slack_total']:.2f}   (the running learning curve also holds its 2 x $3.70 reservation in E3 until it finishes)")
    print(f"{'envelope':<10}{'approved now':>14}{'needed':>10}{'committed':>11}   (needed = committed + new base + slack; ORCH/E0/E1 shrink to committed)")
    for e in ("ORCH", "E0", "E1", "E2", "E3", "E4", "E5", "E6"):
        print(f"{e:<10}{res['approved'].get(e, 0.0):>14.2f}{res['needed'][e]:>10.2f}{res['committed_env'].get(e, 0.0):>11.2f}")
    print(f"{'BOOST':<10}{res['approved'].get('BOOST', 0.0):>14.2f}{res['boost_needed']:>10.2f}{0.0:>11.2f}")
    over = max(0.0, res["base_total"] - 100.0)
    print(f"BASE total needed ${res['base_total']:.2f} (limit $100.00) {'OK' if over <= 1e-9 else f'OVER by ${over:.2f}: the base envelopes need the owner to raise the cap'} | "
          f"BOOST needed ${res['boost_needed']:.2f} (limit $30.00, incl. its reserve slack)")
    print(f"GRAND TOTAL = committed ${res['committed_total']:.2f} + plan ${res['plan_total']:.2f} + slack ${res['slack_total']:.2f} = ${res['grand']:.2f}"
          f"  -> {'fits $130' if res['grand'] <= 130 else ('fits the offered $140' if res['grand'] <= 140 else 'exceeds $140')}"
          f" (needs a cap raise of ${max(0.0, res['grand'] - 130):.2f} over $130)")
    gap = res["grand"] - 140.0
    if gap > 1e-9:
        biggest = max(res["slack_env"].items(), key=lambda kv: kv[1])
        print(f"OVER $140 by ${gap:.2f}: closing it needs one change, e.g. no reserve slack in {biggest[0]} (-${biggest[1]:.2f}) or a lower container cap, or dropping an item of at least ${gap:.2f}")
    print(f"WITH the H11 add-on: ${res['grand_h11']:.2f}  -> {'fits $130' if res['grand_h11'] <= 130 else ('fits the offered $140' if res['grand_h11'] <= 140 else 'exceeds $140')}"
          f" (cap raise of ${max(0.0, res['grand_h11'] - 130):.2f} over $130)")
    h1 = res["h1"]
    print(f"H1 AT n=607 (delta 7): design power {h1['design_power_delta7']:.2f} for a 7-point effect; power for the 5-point kill-threshold effect {h1['power_delta5_at_607']:.2f}; "
          f"P(pass) {h1['p_pass_pilot_effect_10pts']:.2f} at the pilot effect (10 points, p_d 0.38; point estimate >= 5 and CI above 0, Holm not applied)")
    print("POWER of the funded primaries (design power at the planned n; P(pass) at the pilot effect):")
    for r in res["powers"]:
        tag = "EXCLUDED" if r["excluded"] else ("REDUCED" if r["reduced"] else "")
        pw = "n/a (min_delta only)" if r["power"] is None else f"design power {r['power']:.2f}"
        pp = "; ".join(f"P(pass) {v:.2f} at {w}" for w, v in r["p_pass"])
        print(f"  {r['label']:<42} n {r['n_planned']:>5}" + (f" / required {r['n_required']:>5}" if r["n_required"] else "") + f"  {pw} {tag}" + (f"; {pp}" if pp else ""))


def run_owner(out: Path | None = None) -> dict:
    res = owner_package()
    print_owner(res)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["tier", "item", "alt", "envelope", "line", "line_usd", "base_usd", "boost_usd", "funding_envelope_of_boost_portion"])
            for ln in res["lines"]:
                cu = ln["usd"] or 1.0
                if ln["key"].endswith("_fixed"):
                    w.writerow([ln["tier"], ln["name"], "fixed", ln["env"], ln["name"], round(ln["usd"], 4), round(ln["base"], 4), 0, ""])
                for r in ln["alt"]["rows"]:
                    if ln["usd"] <= 0:
                        continue
                    w.writerow([ln["tier"], ln["name"], ln["alt"]["label"], r["envelope"], r["line"], round(r["usd"], 4), round(r["usd"] * ln["base"] / cu, 4),
                                round(r["usd"] * ln["boost"] / cu, 4), "BOOST" if ln["boost"] > 0 else ""])
            for r in res["h11"]["rows"]:
                w.writerow([4, "ADD-ON H11 (3-point margin, n=2,614 per arm)", res["h11"]["label"], r["envelope"], r["line"], round(r["usd"], 4), "", "", ""])
    return res


def run(name: str, out: Path | None = None, concurrency: int = 24) -> dict:
    if name == "owner":
        return run_owner(out)
    res = fill(*package_pools(name), concurrency)
    print_package(name, res)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows_for_csv(name, res))
        pw = out.with_name(out.stem + "_power.csv")
        with pw.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "label", "method", "n_planned", "n_required", "power", "p_pass", "reduced", "excluded", "flag"], lineterminator="\n")
            w.writeheader()
            for r in res["powers"]:
                w.writerow({**r, "p_pass": "; ".join(f"{why}: {v:.3f}" for why, v in r["p_pass"])})
    return res
