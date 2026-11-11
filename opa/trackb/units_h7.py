"""Track B unit tests U6-U8 (protocol 5.1) and the H7 measurements, on Qwen3-4B-Instruct-2507 (PLAN_V3 P3.1 and P3.4).

Entry points
  opa.trackb.units_h7:main(args, ctx)   Modal job: H7 sweeps (CPU torch, no model needed), then loads args["model"] (default
                                        Qwen/Qwen3-4B-Instruct-2507) in float32, runs U7 and U8 on its own logits, writes
                                        tests/U6.json, tests/U7.json, tests/U8.json and h7.json through ctx.put_json, logs one event
                                        per test and returns {"metrics": {...}} with the four H7 keys of prereg/rules.yaml.
  python -m opa.trackb.units_h7 --tiny  local check on the tiny random Qwen3 model (no download).

U6 / H7 (precision; protocol 5.1 U6, hypothesis H7; failure threshold 0.1 relative error, as in the reference code
opa/reference/checks_v2.py section 3)
  Global coordinates. A chain of hyperbolic steps of length delta = 1.0 outward from the root; the node at depth n has global
  radius tanh(n * delta / 2). The coordinate is stored in the working dtype (bfloat16 / float16 / float32, round to nearest even
  by torch) and the distance of the n -> n+1 step is computed from the STORED radii in float64. The step "fails" at depth n when a
  stored radius rounds to >= 1 (distance undefined) or the relative error |d - delta| / delta exceeds 0.1; the first failing depth is
  global_fail_depth_<dtype>. Prediction: bf16 4, fp16 7, fp32 16; U6 passes when each is within one step of that, H7 when they are
  in [3, 5], [6, 8], [15, 17]. The same sweep with real m-dimensional coordinates (random ray, every coordinate rounded) is reported
  as global_fail_depth_<dtype>_vector.
  Re-anchored offsets. The kernel's own path: a chain of nodes with offsets expmap0(delta/2 * u_n) (random unit u_n in R^m), offsets
  STORED in the working dtype, composed in float32 in the lowest-common-ancestor frame by opa.geometry.AddressBookTorch (the
  routine tables_from_book uses). The error at depth n is the mean over chains of |d(n+2, n) - d_float64(n+2, n)| / d_float64, i.e.
  the distance between a node and its grandchild at depth n, against the same chain composed in float64 from unrounded offsets.
  re-anchored error growth ratio = error(depth 40) / error(depth 1) (bf16 for the H7 key; limit 2.0). A bootstrap over chains gives
  the 95% band of every error curve and of the ratio.

U7 (factorization on real logits). Attention in OAA mode with live parameters (A ~ N(0, 0.5^2), lambda = 2, g = 1) in float32 on the
model's own forward pass; for recorded layers, heads and query positions the attention weights equal pi_c * alpha(v | c) (router
mass of the origin times the local softmax inside it) and a @ v reproduces the SDPA output: max abs error <= 1e-5.

U8 (paging identity). Two REAL passes of the OPA kernel per case. The full pass keeps every origin resident. The paged pass repeats
the forward with the evicted origins' keys hidden from the tail's query rows by the kernel itself, in the one layer being checked
(attention.oaa_inputs(evict_layers, evict_rows, evict_from_query); the layers before it are identical, so q, k, v and the OAA bias of
that layer are those of the full pass - the precondition of the identity, which is also measured). Each pass is recorded; then
TV(a_full, a_paged) = 0.5 * sum |a_full - a_paged|, each from its own pass, must equal the evicted router mass 1 - sum_{resident} pi
(pi from the full pass) within 1e-5, the paged pass must put exactly zero weight on the evicted keys, and its SDPA output must equal
a_paged @ v within 1e-5 (as in U7). Corrupting the paged pass (its kernel mask or its output) makes U8 fail (tests).

Notes for the H7 report
  * The re-anchored growth ratio uses d(n+2, n): the lowest common ancestor of nodes n+2 and n is node n, so the distance is the
    composition of two stored offsets in node n's frame. It measures same-branch composition (the error cannot depend on depth) and
    cannot show growth across branches; say so wherever H7 is reported. It is also stored in h7.json under "notes".
  * torch's float16 rounding differs from numpy's in about 6 of 60,000 values (ties and subnormal handling); this has no effect on the
    sweep (the failing depths are identical, tests compare torch and numpy rounding on the values the sweep uses).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from opa.geometry import AddressBook, AddressBookTorch, dist_t
from opa.trackb import attention as att
from opa.trackb import units

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
PREDICTED_DEPTH = {"bfloat16": 4, "float16": 7, "float32": 16}
H7_RANGE = {"bfloat16": (3, 5), "float16": (6, 8), "float32": (15, 17)}
FAIL_TOL = 0.1                      # relative error that counts as failure (reference code)
GROWTH_LIMIT = 2.0                  # H7: re-anchored error growth ratio depth 40 vs depth 1 (bf16)
TOL_FACT = 1e-5


# ============================================================================ H7 / U6 sweeps


def _round(x: float, dtype) -> float:
    return float(torch.tensor(float(x), dtype=torch.float64).to(dtype).to(torch.float64))


def global_step_error(n: int, dtype, delta: float = 1.0) -> float:
    """Relative error of the n -> n+1 step distance computed from radii stored in `dtype`; nan if a stored radius is >= 1."""
    r0, r1 = _round(math.tanh(n * delta / 2), dtype), _round(math.tanh((n + 1) * delta / 2), dtype)
    if r1 >= 1.0 or r0 >= 1.0:
        return float("nan")
    d = math.acosh(1 + 2 * (r1 - r0) ** 2 / ((1 - r0 ** 2) * (1 - r1 ** 2)))
    return abs(d - delta) / delta


def global_fail_depth(dtype, delta: float = 1.0, max_depth: int = 40, tol: float = FAIL_TOL) -> int:
    for n in range(max_depth):
        e = global_step_error(n, dtype, delta)
        if math.isnan(e) or e > tol:
            return n
    return -1


def global_vector_step_error(n: int, dtype, u: torch.Tensor, delta: float = 1.0) -> float:
    """Same with real m-dimensional points on a ray: every coordinate rounded to `dtype`, distance in float64 (dist_t)."""
    p0 = (math.tanh(n * delta / 2) * u).to(dtype).to(torch.float64)
    p1 = (math.tanh((n + 1) * delta / 2) * u).to(dtype).to(torch.float64)
    if float((p0 * p0).sum()) >= 1.0 or float((p1 * p1).sum()) >= 1.0:
        return float("nan")
    d = float(dist_t(p0, p1))
    return float("nan") if not math.isfinite(d) else abs(d - delta) / delta


def global_vector_fail_depth(dtype, m: int = 32, delta: float = 1.0, max_depth: int = 40, tol: float = FAIL_TOL, seed: int = 0) -> int:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(m, generator=g, dtype=torch.float64)
    u = u / u.norm()
    for n in range(max_depth):
        e = global_vector_step_error(n, dtype, u, delta)
        if math.isnan(e) or e > tol:
            return n
    return -1


def _chain_book(m: int, depth: int, delta: float, rng: np.random.Generator) -> AddressBook:
    book = AddressBook(m, dtype=np.float64)
    for c in range(1, depth + 3):
        u = rng.standard_normal(m)
        book.add(c, c - 1, (delta / 2.0) * u / np.linalg.norm(u))
    return book


def reanchored_errors(dtype, chains: int = 128, depth: int = 40, m: int = 32, delta: float = 1.0, seed: int = 0,
                      work_dtype=torch.float32) -> np.ndarray:
    """errors[chain, n-1] for depth n = 1..depth: relative error of d(n+2, n) with offsets stored in `dtype`, composed in
    `work_dtype` in the LCA frame, against the float64 chain built from the unrounded offsets."""
    rng = np.random.default_rng(seed)
    out = np.zeros((chains, depth))
    for ci in range(chains):
        book = _chain_book(m, depth, delta, rng)
        tq, tx = AddressBookTorch(book, dtype=dtype), AddressBookTorch(book, dtype=torch.float64)
        for n in range(1, depth + 1):
            d_true = float(tx.pair_distance(n + 2, n, work_dtype=torch.float64))
            d_q = float(tq.pair_distance(n + 2, n, work_dtype=work_dtype))
            out[ci, n - 1] = abs(d_q - d_true) / d_true
    return out


def _band(errs: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    """Per-depth mean error with a 95% bootstrap band over chains, and the ratio error(last depth) / error(depth 1)."""
    rng = np.random.default_rng(seed)
    C = errs.shape[0]
    idx = rng.integers(0, C, size=(n_boot, C))
    boots = errs[idx].mean(axis=1)                                       # [n_boot, depth]
    mean = errs.mean(0)
    ratio = float(mean[-1] / mean[0])
    rb = boots[:, -1] / boots[:, 0]
    return {"depth": list(range(1, errs.shape[1] + 1)), "mean": [float(x) for x in mean],
            "lo": [float(x) for x in np.percentile(boots, 2.5, axis=0)], "hi": [float(x) for x in np.percentile(boots, 97.5, axis=0)],
            "ratio_last_vs_first": ratio, "ratio_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
            "chains": int(C)}


def _clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def h7_measurements(chains: int = 128, depth: int = 40, m: int = 32, delta: float = 1.0, seed: int = 0, depths_table: int = 20) -> dict:
    """All H7 numbers. Keys named as in prereg/rules.yaml: global_fail_depth_{bf16,fp16,fp32} and
    reanchored_error_growth_ratio_depth40_vs_depth1_bf16 (when depth = 40; the key suffix follows `depth`)."""
    short = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    out: dict = {"delta": delta, "m": m, "fail_tolerance": FAIL_TOL, "predicted_depths": dict(PREDICTED_DEPTH), "chains": chains,
                 "global": {}, "reanchored": {}}
    for name, dt in DTYPES.items():
        out[f"global_fail_depth_{short[name]}"] = global_fail_depth(dt, delta)
        out[f"global_fail_depth_{short[name]}_vector"] = global_vector_fail_depth(dt, m, delta, seed=seed)
        out["global"][name] = {"step_relative_error": [global_step_error(n, dt, delta) for n in range(depths_table)]}
        band = _band(reanchored_errors(dt, chains, depth, m, delta, seed))
        out["reanchored"][name] = band
        out[f"reanchored_error_growth_ratio_depth{depth}_vs_depth1_{short[name]}"] = band["ratio_last_vs_first"]
    out["notes"] = {
        "reanchored_pair": "d(n+2, n): the LCA of nodes n+2 and n is node n, so the distance is the composition of two stored offsets in "
                           "node n's frame. It measures same-branch composition and cannot show cross-branch growth.",
        "fp16_rounding": "torch float16 rounding differs from numpy's in about 6 of 60,000 values; no effect on the sweep.",
        "failure_threshold": "relative step-distance error > 0.1 (reference code), or a stored radius >= 1"}
    return _clean(out)


def u6_precision(h7: dict, run_id: str = "local") -> dict:
    ratio_key = [k for k in h7 if k.startswith("reanchored_error_growth_ratio_depth") and k.endswith("_bf16")][0]
    gaps = {n: h7[f"global_fail_depth_{s}"] - PREDICTED_DEPTH[n] for n, s in (("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32"))}
    depth_ok = all(abs(g) <= 1 for g in gaps.values())
    ratio = h7[ratio_key]
    flat = ratio <= GROWTH_LIMIT
    passed = depth_ok and flat
    metrics = {"fail_depths": {n: h7[f"global_fail_depth_{s}"] for n, s in (("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32"))},
               "fail_depths_vector_m": {n: h7[f"global_fail_depth_{s}_vector"] for n, s in (("bfloat16", "bf16"), ("float16", "fp16"),
                                                                                           ("float32", "fp32"))},
               "predicted_depths": PREDICTED_DEPTH, "gap_to_prediction": gaps, "depth_within_one_step": depth_ok,
               "reanchored_growth_ratio_bf16": ratio, "reanchored_growth_ratio_ci95_bf16": h7["reanchored"]["bfloat16"]["ratio_ci95"],
               "reanchored_growth_ratio": {n: h7["reanchored"][n]["ratio_last_vs_first"] for n in DTYPES}, "growth_limit": GROWTH_LIMIT,
               "reanchored_flat": flat, "fail_tolerance": FAIL_TOL, "chains": h7["chains"],
               "reanchored_error_depth1_bf16": h7["reanchored"]["bfloat16"]["mean"][0],
               "reanchored_error_last_bf16": h7["reanchored"]["bfloat16"]["mean"][-1]}
    d = metrics["fail_depths"]
    details = (f"Global coordinates (radius stored in the dtype, step distance from the stored radii, failure = relative error > {FAIL_TOL} or "
               f"radius >= 1): first failing depth bf16 {d['bfloat16']} / fp16 {d['float16']} / fp32 {d['float32']} (predicted 4 / 7 / 16; "
               f"within one step: {depth_ok}); with real {h7['m']}-dim coordinates {list(metrics['fail_depths_vector_m'].values())}. "
               f"Re-anchored (offsets stored in the dtype, composed in float32 in the LCA frame, {h7['chains']} chains): error at depth 40 over "
               f"depth 1 = {ratio:.3f} for bf16 (95% band {metrics['reanchored_growth_ratio_ci95_bf16'][0]:.2f}-"
               f"{metrics['reanchored_growth_ratio_ci95_bf16'][1]:.2f}, limit {GROWTH_LIMIT}), flat: {flat}.")
    return units._record("U6", "Precision table: global coordinates fail at the predicted depths, re-anchored error stays flat to depth 40",
                         passed, details, run_id, _clean(metrics),
                         "first failing depth within one step of 4 / 7 / 16 (bf16 / fp16 / fp32); re-anchored bf16 error growth ratio depth 40 vs 1 <= 2.0")


# ============================================================================ U7, U8 on the model


def _oaa_setup(model, sizes, seed: int):
    """Inputs of the OAA window shared by U7 and U8: tables, ids, layout inputs, token origins, geometry facts."""
    device = next(model.parameters()).device
    if getattr(model, "_opa_state", None) is None:
        att.install_oaa(model, m=sizes["m"], seed=seed)
    B, tail = sizes["B"], sizes["tail"]
    lay, window, cur = units.u4_layout(sizes["m"], seed=seed)
    tables = att.tables_from_layout(lay, sorted(set(window + [cur])), kind="A2")
    segs = [att.Segment(c, B) for c in window] + [att.Segment(cur, tail)]
    T = sum(s_.length for s_ in segs)
    ids = units._ids(model, T, seed + 100, device)
    inputs = att.layout_inputs(segs, "M5", device=device)
    origins = att.token_origins(segs)
    win_rows = [int(r) for r in tables.rows(torch.tensor(window))]
    return {"tables": tables, "ids": ids, "inputs": inputs, "origins": origins, "T": T, "tail": tail, "window_rows": win_rows,
            "n_origins": len(window) + 1}


def _forward_records(model, setup, seed: int, record_layers, evict_layers=(), evict_rows=None):
    """One forward pass with live OAA parameters (identical for every call with the same seed); returns the recorded layers."""
    units._set_oaa(model, A_std=0.5, lam=2.0, g=1.0, seed=seed)
    try:
        with att.oaa_inputs(model, setup["tables"], setup["origins"], record_layers=set(record_layers), evict_layers=set(evict_layers),
                            evict_rows=evict_rows, evict_from_query=setup["T"] - setup["tail"]) as st:
            units._logits(model, setup["ids"], **setup["inputs"])
            return dict(st.records)
    finally:
        units._set_oaa(model, 0.0, 0.0, 0.0)


def _oaa_records(model, sizes, seed: int):
    """Full pass (every origin resident) with recorded layers 0, middle, last. Returns (records, T, tail, n_layers, n_heads, n_origins)."""
    setup = _oaa_setup(model, sizes, seed)
    n_layers = len(att.attention_modules(model))
    layers = sorted({0, n_layers // 2, n_layers - 1})
    records = _forward_records(model, setup, seed, layers)
    return records, setup["T"], setup["tail"], n_layers, model.config.num_attention_heads, setup["n_origins"]


def _cases(records, T, tail, H):
    heads = sorted({0, H // 2, H - 1})
    qpos = sorted({T - 1, T - tail, T // 2})
    return [(l, h, q) for l in sorted(records) for h in heads for q in qpos]


@torch.no_grad()
def u7_factorization(model, sizes, run_id="local", seed=7) -> dict:
    records, T, tail, n_layers, H, n_orig = _oaa_records(model, sizes, seed)
    worst_f, worst_o, worst_b, n = 0.0, 0.0, 0.0, 0
    for l, h, q in _cases(records, T, tail, H):
        res = att.factorization_check(records[l], head=h, query_pos=q)
        assert res["n_keys"] == q + 1
        worst_f, worst_o = max(worst_f, res["factorization_max_abs_err"]), max(worst_o, res["sdpa_output_max_abs_err"])
        worst_b = max(worst_b, res["bias_constant_within_origin_max_dev"])
        assert abs(sum(res["router_mass"]) - 1.0) < 1e-9
        n += 1
    passed = worst_f <= TOL_FACT and worst_o <= TOL_FACT and worst_b <= TOL_FACT
    metrics = {"factorization_max_abs_err": worst_f, "sdpa_output_max_abs_err": worst_o, "bias_constant_within_origin_max_dev": worst_b,
               "cases": n, "layers": sorted(records), "heads_per_layer": len(sorted({0, H // 2, H - 1})), "tolerance": TOL_FACT,
               "live_settings": {"A_std": 0.5, "lambda": 2.0, "g": 1.0}, "T": T, "origins_in_window": n_orig}
    details = (f"{n} (layer, head, query) cases on the model's own forward pass with live OAA (A ~ N(0, 0.5^2), lambda 2, g 1), layers "
               f"{sorted(records)}: max |a - pi * alpha| = {worst_f:.2e}, max |a @ v - SDPA output| = {worst_o:.2e}, bias constant within an "
               f"origin to {worst_b:.2e} (tolerance {TOL_FACT:g}, fp32).")
    return units._record("U7", "Factorization on real logits: attention = router mass x local softmax (Theorem 6)", passed, details,
                         run_id, metrics, "max abs error <= 1e-5 in fp32")


def _paged_record(model, setup, seed: int, layer: int, evicted_rows):
    """The paged pass: the kernel hides the evicted origins' keys from the tail's query rows in `layer`; that layer is recorded."""
    return _forward_records(model, setup, seed, [layer], evict_layers=[layer], evict_rows=evicted_rows)[layer]


@torch.no_grad()
def u8_paging(model, sizes, run_id="local", seed=8, fractions=(0.25, 0.5, 0.75)) -> dict:
    setup = _oaa_setup(model, sizes, seed)
    n_layers, H = len(att.attention_modules(model)), model.config.num_attention_heads
    layers = sorted({0, n_layers // 2, n_layers - 1})
    full = _forward_records(model, setup, seed, layers)
    T, tail = setup["T"], setup["tail"]
    heads = sorted({0, H // 2, H - 1})
    qpos = sorted({T - 1, T - tail, T - max(tail // 2, 1)})                    # query rows of the tail: they see every window origin
    rng = np.random.default_rng(seed)
    wrows = setup["window_rows"]
    worst, worst_out, worst_in, worst_ev, n, lo, hi, passes = 0.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0, 0
    for l in layers:
        for f in fractions:
            k = min(max(1, int(round(f * len(wrows)))), len(wrows) - 1)
            evicted = sorted(int(r) for r in rng.choice(wrows, size=k, replace=False))
            paged = _paged_record(model, setup, seed, l, evicted)
            passes += 1
            for h in heads:
                for q in qpos:
                    res = att.paging_check(full[l], paged, h, q, evicted)
                    worst, worst_out = max(worst, res["abs_err"]), max(worst_out, res["paged_sdpa_output_max_abs_err"])
                    worst_in, worst_ev = max(worst_in, res["inputs_max_abs_diff_between_passes"]), max(worst_ev, res["paged_weight_on_evicted_keys"])
                    lo, hi = min(lo, res["evicted_router_mass"]), max(hi, res["evicted_router_mass"])
                    n += 1
    passed = (worst <= TOL_FACT and worst_out <= TOL_FACT and worst_in <= 1e-6 and worst_ev <= 1e-12 and hi > 1e-3)
    metrics = {"tv_vs_evicted_router_mass_max_abs_err": worst, "paged_sdpa_output_max_abs_err": worst_out,
               "inputs_max_abs_diff_between_passes": worst_in, "paged_weight_on_evicted_keys_max": worst_ev, "cases": n,
               "paged_passes": passes, "layers": layers, "fractions": list(fractions), "tolerance": TOL_FACT,
               "evicted_router_mass_range": [lo, hi], "live_settings": {"A_std": 0.5, "lambda": 2.0, "g": 1.0}, "T": T,
               "query_rows": "tail"}
    details = (f"{passes} paged kernel passes (the kernel hides the evicted origins' keys from the tail's query rows in one layer; layers {layers}), "
               f"{n} (layer, head, query, resident subset) cases: TV between the full pass's and the paged pass's attention vs the evicted router "
               f"mass: max abs difference {worst:.2e}; paged SDPA output vs a_paged @ v {worst_out:.2e}; q,k,v of the two passes differ by "
               f"{worst_in:.1e}; weight on evicted keys {worst_ev:.1e} (tolerance {TOL_FACT:g}, fp32); evicted masses covered {lo:.3g} to {hi:.3g}.")
    return units._record("U8", "Paging identity: TV(full, paged) = evicted router mass, paged attention from a real kernel pass", passed,
                         details, run_id, metrics, "max abs error <= 1e-5 in fp32 (TV vs evicted mass, paged output vs a_paged @ v)")


# ============================================================================ runners


def run_all(model, sizes, h7: dict, run_id="local") -> list:
    torch.use_deterministic_algorithms(False)
    return [u6_precision(h7, run_id), u7_factorization(model, sizes, run_id), u8_paging(model, sizes, run_id)]


def main(args: dict, ctx) -> dict:
    """Modal job entry point (jobs/p3_units_u6_u8_h7.yaml). args: model, chains (128), sizes (overrides), depth (40)."""
    t0 = time.time()
    name = args.get("model", units.DEFAULT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ctx.log({"action": "units_h7_start", "model": name, "device": str(device)})
    t = time.time()
    h7 = h7_measurements(chains=int(args.get("chains", 128)), depth=int(args.get("depth", 40)), m=int(args.get("m", 32)))
    ctx.put_json("h7.json", h7)
    ctx.log({"action": "h7_sweeps", "seconds": round(time.time() - t, 1)})
    model = units.load_model(name, device)
    env = units.environment(model, name)
    sizes = dict(units.SIZES["real"], **args.get("sizes", {}))
    metrics = {"env": env, "sizes": sizes}
    key_prefix = f"depth{int(args.get('depth', 40))}"
    for k in ("global_fail_depth_bf16", "global_fail_depth_fp16", "global_fail_depth_fp32",
              f"reanchored_error_growth_ratio_{key_prefix}_vs_depth1_bf16"):
        metrics[k] = h7[k]
    for fn in (lambda: u6_precision(h7, ctx.run_id), lambda: u7_factorization(model, sizes, ctx.run_id),
               lambda: u8_paging(model, sizes, ctx.run_id)):
        t = time.time()
        rec = fn()
        rec["env"] = env
        ctx.put_json(f"tests/{rec['id']}.json", rec)
        ctx.log({"action": "unit_test", "test": rec["id"], "passed": rec["passed"], "seconds": round(time.time() - t, 3), "model": name,
                 "details": rec["details"]})
        metrics[f"{rec['id']}_passed"] = rec["passed"]
    metrics["all_passed"] = all(metrics[f"U{i}_passed"] for i in (6, 7, 8))
    metrics["seconds"] = round(time.time() - t0, 1)
    ctx.log({"action": "units_h7_end", "all_passed": metrics["all_passed"], "seconds": metrics["seconds"]})
    return {"metrics": metrics}


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track B unit tests U6-U8 and the H7 measurements")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--chains", type=int, default=32)
    a = ap.parse_args(argv)
    model, sizes = units.tiny_model(), units.SIZES["tiny"]
    h7 = h7_measurements(chains=a.chains)
    recs = run_all(model, sizes, h7)
    print(json.dumps({"h7_keys": {k: v for k, v in h7.items() if k.startswith(("global_fail", "reanchored_error"))}, "results": recs}, indent=2, default=str))
    print(chr(10).join(f"{r['id']}: {'PASS' if r['passed'] else 'FAIL'}  {r['details']}" for r in recs))
    return 0 if all(r["passed"] for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
