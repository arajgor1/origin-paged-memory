"""Write the v3 P0 job specs (envelope E0). Run: python jobs/make_p0_jobs.py
The templates jobs/p0-*.yaml are always written unsubstituted. Specs with their run ids filled go to jobs/launch/<name>.yaml
(--embed-run-id ID --skeleton-run-id ID --skeleton-h2-run-id ID; every id ever given is kept in
jobs/launch/run_ids.json, so a later call never undoes an earlier one). A launch spec that still contains an uppercase
run-id placeholder is refused (not written). The bucket placeholder in the gs:// URIs is NOT substituted here:
the job substitutes it at run time (select_items, make_embedder, run_system).

LAUNCH ORDER (the E0 reservation rule is checked for exactly this order, see reservation_check and test_p0_jobs.py):
  1 p0-smoke-vllm  2 p0-smoke-vllm-gemma  3 p0-embed-dev  4 p0-embed-dev-aux  5 p0_trackb_units  6 p0-skeleton-sllm
  7 p0-embedcheck, then local CPU work (tau_thread recalibration, H5 distortion re-run)
  7b p0-detprobe-bi-r1, bi-r2, w1-r1, w1-r2 (determinism probes, added after G0 attempt 1)
  8 p0-micropilot-g1, g2, g3  9 p0-g0-s18, p0-g0-s1-rep1, rep2, rep3
  10 conditionals (G0 fallback): p0-skeleton-sllm-h2, p0-g0-s18-h2
Rule: the launcher reserves max_usd, so for every job k: sum(estimates of the jobs before k) + max_usd(k) <= 9.00 (E0 is
requested at $9).


Everything here is a PRE-MEASUREMENT estimate. No run has happened yet; the P0 micro-pilot itself produces the measured
unit costs (state/unit_costs.md) that E1 and later are computed from. The arithmetic is transparent and lives in this
file; every spec carries its own `estimate_formula`.

Jobs written (all envelope E0, max_api_usd 0, dev split only, model ids from config/project.yaml):
  p0-embed-dev, p0-embed-dev-aux      embed_items with harrier (fp32) over the dev v2 cells; one embeddings.npz per job
  p0-embedcheck                       PLAN_V3 s8 check 1: recall@k of gold-fact origins, harrier (fp32 precomputed vectors) vs
                                      Qwen3-Embedding-4B (bf16 on the GPU). The two sides therefore differ in precision as well as model.
                                      If it decides switch_to_challenger, the operator STOPS and files an E0 top-up request before the
                                      micro-pilot (re-embedding, tau_thread recalibration and the CPU fallback cost change together).
                                      There is deliberately no re-embedding job in this E0 job set.
  p0-smoke-vllm, p0-smoke-vllm-gemma  tiny L4 vLLM smokes (Qwen3-4B; Gemma 4 E4B with image/audio inputs disabled)
  p0-skeleton-sllm(-h2)               S-llm entity skeletons (shared by S8-S12/S14-S18, which need skeleton_llm_uri)
  p0-micropilot-g1..g3                10 dev items per Track A system (5 LB late h3 32K + 5 MT T8 h3 32K cross), 19 systems
                                      (S1-S18 and S12F = arm A4) grouped into 3 vLLM jobs. The micro-pilot covers ONLY these two
                                      cells; E1's estimates for every other cell (other placements, h2, T4/T16, -conf, 8K/16K)
                                      are extrapolated from them, not measured.
  p0-g0-s18, p0-g0-s18-h2, p0-g0-s1-rep1..3   G0 checks
  p0-detprobe-bi-r1/-r2, -w1-r1/-r2   determinism probes after G0 attempt 1 failed its repeat criterion (S1 EM 0.44/0.42/0.40),
                                      partly from bf16 numerics under 8-way concurrent batching. Each: S1 on the first 10 dev
                                      LB late h3 32K items, own container, seed 0. bi = VLLM_BATCH_INVARIANT=1 with the
                                      FLASH_ATTN backend (flag --attention-backend, verified to exist in the vLLM v0.32.0
                                      arg_utils; W3 had reported --attention-config.backend, which is not an argument there),
                                      8 workers; w1 = 1 worker and prefix caching off. Estimates use measured numbers: vLLM
                                      start 180 s, S1 12.4 s/item at 8 workers (622 s / 50 items), ~14 s/item at 1 worker (assumed);
                                      batch-invariant slowdown unknown: 2x in the estimate, max_usd = 2x the estimate, which
                                      still covers a 4x slowdown (180 s + 10 x 12.4 s x 4 = 676 s < 2 x 428 s).
  jobs/p0_trackb_units.yaml is kept as is (U1-U5, not written here). tau_thread recalibration and the H5 distortion
  re-run are local CPU work: no job.

Cost model (assumptions, not measurements):
  Rate. launch.rate_per_second(compute, prices): GPU + CPU and memory priced at their LIMITS (2x the request), from the
  human-owned budget/prices.yaml. For compute {L4, cpu 2.0, memory 16 GiB} that is 0.000222 + 4.0 x 0.0000131 +
  32 x 0.00000222 = 0.000346 $/s ($1.25/h).
  vLLM prefill, Qwen3-4B bf16 on one L4: 8 GFLOP/token for the weights; L4 bf16 dense peak 121 TFLOPS; ~35-40% sustained
  by vLLM gives ~4,000 tok/s on prompts up to ~10K tokens, and ~1,800 tok/s on one 34K-token prompt (attention FLOPs
  roughly double the per-token cost; FlashAttention). ASSUMPTION, to be replaced by the micro-pilot.
  vLLM decode, 8 concurrent request streams: weights 8 GB / ~255 GB/s sustained + KV reads ~ 50 ms per step, so ~160
  tok/s aggregate, ~120 tok/s with prefill interleaved. ASSUMPTION. (12 streams for the skeleton job: ~200 tok/s; the KV
  cache holds ~76K tokens: 24 GB x 0.90 - 8 GB weights - ~2 GB activations = ~11 GB / 144 KB per token.)
  OVERHEAD 1.25 on model time (HTTP, scheduling, JSON retries, tail effects: items finish unevenly).
  Start-up: vLLM load + CUDA graphs 240 s (weights come from the opa-hf-cache volume; the first job pays the 8 GB
  download, which the smoke job takes), +45 s when the job needs the live CPU fallback embedder (agent sub-queries).
  Per-item prefill/decode tokens per system are read off systems.py:
    S1  one call over the whole archive (32,768 tokens + origin tags = ~34.5K)       S2  top-8 origins (~4.8K)
    S3  6 IRCoT steps over 4..24 growing origins (~42K total)                         S4  16 gists + lookup + answer (~50K in)
    S5  4 chain workers of 16 origins (~38K)                                          S13 8 cluster summaries + answer (~44K)
    S6-S11 fold-left: 22 windows of 3 origins (~2.4K per call incl. pins) + re-reads and faults; the per-question token
        budget (62.5K, config.TOKEN_BUDGET) stops the item, so 62K is the cap and also the late-placement typical
    S12/S14-S17/S12F descent-first: M=4 candidates first, then the fold continues; ~2/3 of a full fold (42K)
    S18 oracle: descent + oracle skeleton and graph, early stop on the answer (~30K)
  Skeleton job per item: 8 calls x ~4.7K in = 37.6K, ~3.8K out of entity lists.
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

from opa import budget, launch  # noqa: E402

# --------------------------------------------------------------------------------------------- dataset v2 cells
# One place for every cell name (dataset, cell). They are the names `python -m opa.data.build --preset v3-dev` writes
# under <local_data>/<dataset>/dev/v2/<cell>/ and that sit in gs://{bucket}/opa/data/<dataset>/dev/v2/<cell>/.
CELLS = {
    "lb_late_h3": ("lbmhqa", "late-h3-32k-a0"),          # G0 / primary candidate
    "lb_late_h2": ("lbmhqa", "late-h2-32k-a0"),          # G0 fallback
    "lb_forward_h3": ("lbmhqa", "forward-h3-32k-a0"),
    "lb_mixed_h3": ("lbmhqa", "mixed-h3-32k-a0"),
    "mt_T8_cross": ("mtlbmhqa", "T8-h3-32k-cross"),      # micro-pilot MT benchmark
    "mt_T4_cross": ("mtlbmhqa", "T4-h3-32k-cross"),
    "mt_T4_nocross": ("mtlbmhqa", "T4-h3-32k-nocross"),
    "mt_T8_nocross": ("mtlbmhqa", "T8-h3-32k-nocross"),
    "mt_T16_cross": ("mtlbmhqa", "T16-h3-32k-cross"),
    "mt_T16_nocross": ("mtlbmhqa", "T16-h3-32k-nocross"),
    "conf_T4_cross": ("mtlbmhqa", "T4-h3-32k-cross-conf"),      # thread-confusion cells (H12)
    "conf_T4_nocross": ("mtlbmhqa", "T4-h3-32k-nocross-conf"),
    "conf_T8_cross": ("mtlbmhqa", "T8-h3-32k-cross-conf"),
    "conf_T8_nocross": ("mtlbmhqa", "T8-h3-32k-nocross-conf"),
    "conf_T16_cross": ("mtlbmhqa", "T16-h3-32k-cross-conf"),
    "conf_T16_nocross": ("mtlbmhqa", "T16-h3-32k-nocross-conf"),
}
EMBED_CORE = [("lb_late_h3", 50), ("lb_late_h2", 50), ("mt_T8_cross", 10)]   # G0, embedder check, micro-pilot (first 5 MT), tau_thread
EMBED_AUX = ["lb_forward_h3", "lb_mixed_h3", "mt_T4_cross", "mt_T4_nocross", "mt_T8_nocross", "mt_T16_cross",
             "mt_T16_nocross"] + [k for k in CELLS if k.startswith("conf_")]   # 10 items each (tau_thread, tree/H12 look)
N_CORE, N_AUX, N_G0 = 50, 10, 50
MICRO_KEYS = [("lb_late_h3", 5), ("mt_T8_cross", 5)]        # micro-pilot: 10 items per system in total (PLAN_V3 P0)
SKELETON_KEYS = [("lb_late_h3", 50), ("mt_T8_cross", 5)]    # 50 LB for G0 S18 (first 5 also feed the micro-pilot) + 5 MT

BUCKET = "{bucket}"       # replaced at run time by the job (select_items, make_embedder, run_system); never at generation time
PH_EMBED, PH_SK, PH_SK_H2 = "EMBED_RUN_ID", "SKELETON_RUN_ID", "SKELETON_H2_RUN_ID"


def items_uri(key: str) -> str:
    ds, cell = CELLS[key]
    return f"gs://{BUCKET}/opa/data/{ds}/dev/v2/{cell}/items.jsonl.gz"


EMBED_URI = f"precomputed:gs://{BUCKET}/opa/runs/{PH_EMBED}/embeddings.npz"
SKELETON_URI = f"gs://{BUCKET}/opa/runs/{PH_SK}/skeleton_llm"
SKELETON_H2_URI = f"gs://{BUCKET}/opa/runs/{PH_SK_H2}/skeleton_llm"
QWEN, GEMMA = "Qwen/Qwen3-4B-Instruct-2507", "google/gemma-4-E4B-it"
EMBEDDER, EMBEDDER_BACKUP = "microsoft/harrier-oss-v1-0.6b", "Qwen/Qwen3-Embedding-4B"

# --------------------------------------------------------------------------------------------- cost model
P_SHORT, P_LONG = 4000.0, 1800.0         # vLLM prefill tok/s on an L4 (assumption, see docstring)
D_AGG = 120.0                            # aggregate decode tok/s with 8 streams (assumption)
OVERHEAD = 1.25
START_VLLM, START_FALLBACK = 240, 45     # s
EMBED_TOK_S = {"L4": 8000.0, "T4": 3000.0}   # harrier-0.6b fp32 forward tokens/s (assumption: 30% of fp32 peak)
EMBED_TOK_PER_ITEM = 34000 * 1.10        # 34K tokens per 32K item + padding to a multiple of the longest in the batch
EMBED_START = 120                        # s: 1.2 GB download + load + 3-string reference self-check
QE_TOK_S = 4800.0                        # Qwen3-Embedding-4B bf16 forward tokens/s on an L4 (7.2 GFLOP/token at 35 TFLOPS)
QE_SELFCHECK = 90                        # s: fp32 CPU reference for the bf16 self-check (loads ~16 GB)
QE_START = 360                           # s: 8 GB download + load

# system -> (prefill tokens/item, output tokens/item, long prompt?, live fallback embedder?, cpu s/item, needs S-llm skeleton?)
PROFILE = {
    "S1": (34500, 40, True, False, 0.5, False),
    "S2": (4800, 40, False, False, 0.5, False),
    "S3": (42000, 350, False, True, 0.5, False),
    "S4": (50000, 1800, False, False, 0.5, False),
    "S5": (38000, 1200, False, False, 0.5, False),
    "S13": (44000, 1500, False, True, 1.0, False),
    **{s: (62000, 2800, False, True, 1.5, s in ("S8", "S9", "S10", "S11")) for s in ("S6", "S7", "S8", "S9", "S11")},
    "S10": (62000, 3500, False, True, 1.5, True),                      # abstractive notes: longer outputs
    **{s: (42000, 1900, False, True, 1.5, True) for s in ("S12", "S14", "S15", "S16", "S17")},
    "S12F": (42000, 1900, False, True, 4.5, True),                     # + fit_addresses (arm A4)
    "S18": (30000, 1400, False, True, 1.5, True),
}
SYSTEM_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16",
                "S17", "S18", "S12F"]
FOLD_OPTS = {"lambda": 2.0}


def vllm_compute(memory_gib: int = 16) -> dict:
    return {"kind": "gpu", "gpu": "L4", "image": "vllm", "cpu": 2.0, "memory_gib": memory_gib}


def gpu_compute(gpu: str = "L4", memory_gib: int = 16) -> dict:
    return {"kind": "gpu", "gpu": gpu, "cpu": 2.0, "memory_gib": memory_gib}


_PRICES = None


def prices() -> dict:
    global _PRICES
    if _PRICES is None:
        _PRICES = budget.load_prices()
    return _PRICES


def rate_text(compute: dict) -> tuple[float, str]:
    """($/s, text) from launch.rate_per_second and launch.resources (limits), never re-implemented."""
    p = prices()["modal"]
    r = launch.resources(compute)
    rate = launch.rate_per_second(compute, prices())
    gpu = compute.get("gpu") or "L4"
    txt = (f"{gpu} {p['gpu_per_second'][gpu]:.6f} + {r['cpu_limit']:g} cores(limit) x {p['cpu_core_per_second']:.7f} + "
           f"{r['memory_gib_limit']:g} GiB(limit) x {p['memory_gib_per_second']:.8f} = {rate:.6f} $/s")
    return rate, txt


def finish(spec: dict, seconds: float, work_txt: str, startup_s: float, *, items: int, conditional: bool = False,
           notes: list[str] | None = None, max_factor: float = 2.0) -> tuple[dict, dict]:
    """Add formula, estimate and ceiling. max_usd = max(max_factor x estimate, estimate + start-up cost), rounded up to cents."""
    rate, rtxt = rate_text(spec["compute"])
    raw = seconds * rate
    est = math.ceil(raw * 100) / 100
    mx = math.ceil(max(max_factor * raw, raw + startup_s * rate) * 100) / 100
    spec["estimate_formula"] = f"{work_txt} = {seconds:.0f} s x ({rtxt}) = ${raw:.3f}"
    spec["estimate_usd"], spec["max_usd"], spec["max_api_usd"] = est, mx, 0
    if notes:
        spec["notes"] = notes
    meta = {"name": spec["name"], "items": items, "seconds": seconds, "est": est, "max": mx, "conditional": conditional,
            "raw": raw}
    return spec, meta


HYP_UNIT = ("P0-unit-costs",)


def base(name, track, system, dataset, cell, entrypoint, args, models, compute, hyp=HYP_UNIT, seed=0) -> dict:
    return {"name": name, "envelope": "E0", "track": track, "hypotheses": list(hyp), "system": system, "dataset": dataset,
            "cell": cell, "split": "dev", "seed": seed, "entrypoint": entrypoint, "args": args, "models": models,
            "compute": compute}


# --------------------------------------------------------------------------------------------- the jobs


def uri_entry(key: str, n: int) -> dict:
    return {"uri": items_uri(key), "n_items": n}


def embed_job(name: str, keys: list[tuple[str, int]], gpu: str = "L4") -> tuple[dict, dict]:
    args = {"items_uri": [uri_entry(k, n) for k, n in keys], "embedder": "harrier", "B": 512, "batch": 16}
    spec = base(name, "A", "embed", "lbmhqa+mtlbmhqa", f"{len(keys)}cells", "opa.tracks.track_a:embed_items", args,
                {"embedder": EMBEDDER}, gpu_compute(gpu, 12))
    items = sum(n for _, n in keys)
    sec = EMBED_START + items * EMBED_TOK_PER_ITEM / EMBED_TOK_S[gpu]
    work = (f"{EMBED_START} s start-up + {items} items ({', '.join(f'{CELLS[k][1]} x {n}' for k, n in keys)}) x "
            f"{EMBED_TOK_PER_ITEM:.0f} tokens / {EMBED_TOK_S[gpu]:.0f} tok/s (harrier-0.6b fp32 on {gpu}, assumed)")
    notes = ["one embeddings.npz holds every cell of this job (keys item_id/origin_id)",
             "L4 vs T4 per item: see the printed comparison; L4 is cheaper per item (T4 is ~2.7x slower for 0.74x the price)"]
    return finish(spec, sec, work, EMBED_START, items=items, notes=notes)


def embedcheck_job() -> tuple[dict, dict]:
    ks = [1, 2, 4, 8, 16]
    args = {"items_uri": items_uri("lb_late_h3"), "n_items": N_CORE, "offset": 0, "M": 4, "ks": ks, "B": 512,
            "embedders": [{"label": "harrier-0.6b", "name": f"precomputed:gs://{BUCKET}/opa/runs/{PH_EMBED}/embeddings.npz"},
                          {"label": "qwen3-embedding-4b", "name": EMBEDDER_BACKUP, "dtype": "bfloat16", "batch": 8}],
            "baseline": "harrier-0.6b", "challenger": "qwen3-embedding-4b"}
    spec = base("p0-embedcheck", "A", "embedcheck", "lbmhqa", CELLS["lb_late_h3"][1], "opa.tracks.track_a:embedder_check",
                args, {"embedder": EMBEDDER, "embedder_backup": EMBEDDER_BACKUP}, gpu_compute("L4", 24),
                hyp=("P0-pilot-check",))
    toks = N_CORE * EMBED_TOK_PER_ITEM
    start = QE_START + QE_SELFCHECK
    sec = start + toks / QE_TOK_S
    work = (f"{start} s start-up (8 GB download + bf16 GPU load {QE_START} s, + {QE_SELFCHECK} s CPU fp32 reference for the self-check) + "
            f"{N_CORE} items x {EMBED_TOK_PER_ITEM:.0f} tokens / {QE_TOK_S:.0f} tok/s (Qwen3-Embedding-4B bf16; harrier vectors are "
            f"read from p0-embed-dev, not recomputed)")
    return finish(spec, sec, work, start, items=N_CORE,
                  notes=["Qwen3-Embedding-4B runs bf16 on the GPU; its self-check (threshold 0.99 for bf16, set by embedder_check) "
                         "loads the fp32 reference on the CPU (~16 GB), hence memory_gib 24 (limit 48)",
                         f"the run id of p0-embed-dev replaces the embedding placeholder in the harrier path; the first {N_CORE} items there are the same items (offset 0)"])


def smoke_job(name: str, model: str, hf: str, extra_args: list[str] | None, startup: int, mem: int) -> tuple[dict, dict]:
    m = {"kind": "open_local", "name": model, "max_model_len": 40960}
    if extra_args:
        m["extra_args"] = extra_args
    args = {"items_uri": items_uri("lb_late_h3"), "n_items": 2, "system": "S1", "embedder": "hash", "workers": 2, "model": m}
    spec = base(name, "A", "S1", "lbmhqa", CELLS["lb_late_h3"][1], "opa.tracks.track_a:run_system", args, {"agent_model": hf},
                vllm_compute(mem))
    per = 34500 / P_LONG + 40 / D_AGG
    sec = startup + 2 * per * OVERHEAD
    work = (f"{startup} s start-up (first download of the weights + vLLM load) + 2 items x {per:.1f} s (S1: 34.5K prefill tok / "
            f"{P_LONG:.0f} tok/s + 40 out tok / {D_AGG:.0f} tok/s) x {OVERHEAD}")
    notes = ["smoke: proves the vLLM image, the memory at max_model_len 40960 and one S1 answer; embedder 'hash' is used on purpose "
             "(S1 does not use embeddings), so this job does not depend on p0-embed-dev",
             "check the run's metrics: n=2, errors=0, does_not_fit=0, and that the prediction field is non-empty"]
    if extra_args:
        notes.append("Gemma: image/audio inputs disabled (JSON form of --limit-mm-per-prompt, vLLM >= 0.10; if the served vLLM "
                     "wants the legacy form use ['--limit-mm-per-prompt', 'image=0,audio=0']). Risk: ~16 GB bf16 weights leave "
                     "only ~3-4 GB of KV on an L4; if vLLM refuses 40960, retry once with max_model_len 32768 and log it")
    return finish(spec, sec, work, startup, items=2, notes=notes)


def skeleton_job(name: str, keys: list[tuple[str, int]], conditional: bool = False) -> tuple[dict, dict]:
    workers, d_agg = 12, 200.0
    args = {"items_uri": [uri_entry(k, n) for k, n in keys], "workers": workers,
            "model": {"kind": "open_local", "name": "qwen", "max_model_len": 40960}}
    spec = base(name, "A", "sllm", "+".join(dict.fromkeys(CELLS[k][0] for k, _ in keys)), "+".join(CELLS[k][1] for k, _ in keys),
                "opa.tracks.track_a:build_skeletons", args, {"agent_model": QWEN}, vllm_compute())
    items = sum(n for _, n in keys)
    per = 37600 / P_SHORT + 3800 / d_agg
    sec = START_VLLM + items * per * OVERHEAD
    work = (f"{START_VLLM} s start-up + {items} items ({', '.join(f'{CELLS[k][1]} x {n}' for k, n in keys)}) x {per:.1f} s (8 calls: "
            f"37.6K prefill tok / {P_SHORT:.0f} tok/s + 3.8K entity-list out tok / {d_agg:.0f} tok/s at {workers} streams) x {OVERHEAD}")
    notes = ["the skeleton directory is <this run's gs:// prefix>/skeleton_llm; the run id of this job replaces the "
             f"{'h2 ' if conditional else ''}skeleton placeholder in the later specs of the systems that need an S-llm skeleton "
             "(S8-S12, S14-S18, S12F)"]
    return finish(spec, sec, work, START_VLLM, items=items, conditional=conditional, notes=notes)


DET_START, DET_S1_8W, DET_S1_1W, DET_ITEMS = 180, 12.4, 14.0, 10


def detprobe_job(name: str, mode: str) -> tuple[dict, dict]:
    bi = mode == "bi"
    model = {"kind": "open_local", "name": "qwen", "max_model_len": 40960}
    if bi:
        model["env"] = {"VLLM_BATCH_INVARIANT": "1"}
        model["extra_args"] = ["--attention-backend", "FLASH_ATTN"]
    else:
        model["prefix_caching"] = False
    args = {"items_uri": items_uri("lb_late_h3"), "n_items": DET_ITEMS, "offset": 0, "system": "S1", "embedder": EMBED_URI,
            "workers": 8 if bi else 1, "model": model}
    spec = base(name, "A", "S1", "lbmhqa", CELLS["lb_late_h3"][1], "opa.tracks.track_a:run_system", args, {"agent_model": QWEN},
                vllm_compute(16), hyp=("P0-pilot-check",), seed=0)
    per = DET_S1_8W * 2 if bi else DET_S1_1W
    sec = DET_START + DET_ITEMS * per
    how = (f"{DET_S1_8W} s measured at 8 workers x 2 assumed batch-invariant slowdown" if bi
           else f"{DET_S1_1W} s assumed at 1 worker, prefix caching off")
    work = f"{DET_START} s vLLM start (measured) + {DET_ITEMS} items x {per:.1f} s (S1: {how})"
    notes = ["determinism probe: separate container, seed 0, same 10 items as the G0 attempt-1 S1 repeats (first 10 of the 50)",
             "max_usd = 2x the estimate, which still covers a 4x slowdown" if bi
             else "no batching and no prefix caching: tests whether concurrency causes the run-to-run spread",
             "fill the embedding run-id placeholder from p0-embed-dev (S1 itself does not use the vectors)"]
    return finish(spec, sec, work, DET_START, items=DET_ITEMS, notes=notes)


def system_time(system: str, items: int) -> tuple[float, str]:
    prefill, out, long_, fallback, cpu_s, needs_sk = PROFILE[system]
    p = P_LONG if long_ else P_SHORT
    per = (prefill / p + out / D_AGG) * OVERHEAD + cpu_s
    return items * per, f"{system} {items} x {per:.1f} s [({prefill} prefill tok / {p:.0f} + {out} out tok / {D_AGG:.0f}) x {OVERHEAD} + {cpu_s} s CPU]"


def systems_job(name: str, systems: list[str], keys: list[tuple[str, int]], *, seed: int = 0, hyp=HYP_UNIT,
                skeleton_uri: str | None = None, conditional: bool = False, extra_notes: list[str] | None = None,
                single: bool = False, max_factor: float = 2.0) -> tuple[dict, dict]:
    """One vLLM container for `systems` over the same items. single=True writes the `system` key (one system, one run
    at the root); otherwise the `systems` key (outputs under <system>/r0/)."""
    fallback = any(PROFILE[s][3] for s in systems)
    needs_sk = any(PROFILE[s][5] for s in systems)
    args = {"items_uri": [uri_entry(k, n) for k, n in keys] if not single else items_uri(keys[0][0])}
    if single:
        args["n_items"] = keys[0][1]
        args["system"] = systems[0]
    else:
        args["systems"] = list(systems)
    args.update(embedder=EMBED_URI, workers=8, model={"kind": "open_local", "name": "qwen", "max_model_len": 40960})
    if any(s in launch_fold_systems() for s in systems):
        args["opts"] = dict(FOLD_OPTS)
    if needs_sk:
        args["skeleton_llm_uri"] = skeleton_uri or SKELETON_URI
    dsets = "+".join(dict.fromkeys(CELLS[k][0] for k, _ in keys))
    cells = "+".join(CELLS[k][1] for k, _ in keys)
    spec = base(name, "A", systems[0] if single else "+".join(systems), dsets, cells, "opa.tracks.track_a:run_system", args,
                {"agent_model": QWEN}, vllm_compute(16), hyp=hyp, seed=seed)
    items = sum(n for _, n in keys)
    startup = START_VLLM + (START_FALLBACK if fallback else 0)
    parts = [system_time(s, items) for s in systems]
    sec = startup + sum(t for t, _ in parts)
    work = (f"{startup} s start-up (vLLM {START_VLLM}{' + live fallback embedder ' + str(START_FALLBACK) if fallback else ''}) + "
            + " + ".join(txt for _, txt in parts))
    notes = list(extra_notes or [])
    notes.append("fill the embedding run-id placeholder" + (" and the skeleton run-id placeholder" if needs_sk else "") +
                 " from the jobs they name (python jobs/make_p0_jobs.py --embed-run-id ... writes jobs/launch/)")
    return finish(spec, sec, work, startup, items=items * len(systems), conditional=conditional, notes=notes, max_factor=max_factor)


def launch_fold_systems() -> set[str]:
    return {"S6", "S7", "S8", "S9", "S10", "S11", "S12", "S14", "S15", "S16", "S12F", "S17", "S18"}


def micropilot_groups(n_groups: int = 3) -> list[list[str]]:
    """Longest-processing-time packing of the 19 systems into n_groups jobs of similar run time; deterministic."""
    items = sum(n for _, n in MICRO_KEYS)
    order = sorted(SYSTEM_ORDER, key=lambda s: (-system_time(s, items)[0], SYSTEM_ORDER.index(s)))
    groups, loads = [[] for _ in range(n_groups)], [0.0] * n_groups
    for s in order:
        i = loads.index(min(loads))
        groups[i].append(s)
        loads[i] += system_time(s, items)[0]
    return [sorted(g, key=SYSTEM_ORDER.index) for g in groups]


def all_jobs() -> list[tuple[dict, dict]]:
    out = [embed_job("p0-embed-dev", EMBED_CORE),
           embed_job("p0-embed-dev-aux", [(k, N_AUX) for k in EMBED_AUX]),
           embedcheck_job(),
           smoke_job("p0-smoke-vllm", "qwen", QWEN, None, 360, 16),
           smoke_job("p0-smoke-vllm-gemma", "gemma", GEMMA, ["--limit-mm-per-prompt", '{"image": 0, "audio": 0}'], 480, 24),
           skeleton_job("p0-skeleton-sllm", SKELETON_KEYS),
           skeleton_job("p0-skeleton-sllm-h2", [("lb_late_h2", N_G0)], conditional=True)]
    for i, g in enumerate(micropilot_groups(), 1):
        out.append(systems_job(f"p0-micropilot-g{i}", g, MICRO_KEYS,
                               extra_notes=[f"10 dev items per system: 5 LB late h3 32K + 5 MT T8 h3 32K cross; systems: {', '.join(g)}; "
                                            "outputs under <system>/r0/"]))
    for mode in ("bi", "w1"):
        for r in (1, 2):
            out.append(detprobe_job(f"p0-detprobe-{mode}-r{r}", mode))
    out.append(systems_job("p0-g0-s18", ["S18"], [("lb_late_h3", N_G0)], single=True, hyp=("G0",)))
    out.append(systems_job("p0-g0-s18-h2", ["S18"], [("lb_late_h2", N_G0)], single=True, skeleton_uri=SKELETON_H2_URI,
                           conditional=True, hyp=("G0",), max_factor=1.5,
                           extra_notes=["ceiling 1.5x (not 2x) so that the worst case, every conditional job triggered, passes the $9 reservation rule", "G0 fallback: launch only if S18 scores 60-80% EM at h = 3 (PLAN_V3 s6); needs p0-skeleton-sllm-h2 first"]))
    for r in (1, 2, 3):
        out.append(systems_job(f"p0-g0-s1-rep{r}", ["S1"], [("lb_late_h3", N_G0)], single=True, seed=0, hyp=("G0",),
                               extra_notes=["harness determinism check: separate containers, same inputs"]))
    return out


LAUNCH_ORDER = ["p0-smoke-vllm", "p0-smoke-vllm-gemma", "p0-embed-dev", "p0-embed-dev-aux", "p0_trackb_units", "p0-skeleton-sllm",
                "p0-embedcheck", "p0-detprobe-bi-r1", "p0-detprobe-bi-r2", "p0-detprobe-w1-r1",
                "p0-detprobe-w1-r2", "p0-micropilot-g1", "p0-micropilot-g2", "p0-micropilot-g3", "p0-g0-s18", "p0-g0-s1-rep1",
                "p0-g0-s1-rep2", "p0-g0-s1-rep3", "p0-skeleton-sllm-h2", "p0-g0-s18-h2"]
E0_REQUEST = 9.00
COMMITTED_NOW, FROM_NOW = 3.47, "p0-detprobe-bi-r1"     # E0 spent 1.95 + reserved 1.52 at 2026-11-07 (informational)


def reservation_check(jobs=None, committed: float = 0.0, from_name: str | None = None) -> list[dict]:
    """For every job k in LAUNCH_ORDER: estimates of the jobs before k (assumed spent) + max_usd(k) must be <= E0_REQUEST,
    because the launcher reserves max_usd. The kept p0_trackb_units.yaml supplies its own estimate and ceiling."""
    jobs = jobs or all_jobs()
    by = {m["name"]: (m["est"], m["max"]) for _, m in jobs}
    u = yaml.safe_load((HERE / "p0_trackb_units.yaml").read_text())
    by["p0_trackb_units"] = (float(u["estimate_usd"]), float(u["max_usd"]))
    assert sorted(by) == sorted(LAUNCH_ORDER), set(by) ^ set(LAUNCH_ORDER)
    assert from_name is None or from_name in by
    rows, spent = [], float(committed)
    order = LAUNCH_ORDER[LAUNCH_ORDER.index(from_name):] if from_name else LAUNCH_ORDER
    for n in order:
        est, mx = by[n]
        rows.append({"name": n, "prior_est": round(spent, 2), "max": mx, "need": round(spent + mx, 2), "ok": spent + mx <= E0_REQUEST + 1e-9})
        spent += est
    return rows


# --------------------------------------------------------------------------------------------- output


DUMMY_IDS = {PH_EMBED: "dummy-embed-run", PH_SK: "dummy-skeleton-run", PH_SK_H2: "dummy-skeleton-h2-run"}


def filled(spec: dict) -> dict:
    """A copy with dummy run ids, for validation only (launch.validate refuses leftover run-id placeholders)."""
    return yaml.safe_load(dump(spec, DUMMY_IDS))


def dump(spec: dict, subs: dict[str, str]) -> bytes:
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=1000)
    for k, v in subs.items():
        if v:
            text = text.replace(k, v)
    return text.encode("utf-8")        # LF, no BOM


RX_PLACEHOLDER = re.compile(r"\b[A-Z][A-Z0-9_]*_RUN_ID\b")


def unresolved(text: str) -> list[str]:
    return sorted(set(RX_PLACEHOLDER.findall(text)))


def write_launch(jobs, ids: dict, out_dir: Path) -> tuple[list[str], dict]:
    """Write jobs/launch/<name>.yaml with every known id substituted; a spec that still contains an uppercase *_RUN_ID
    placeholder is refused (skipped, reported). ids: the three run-id placeholders. {bucket} is left for run time.
    Previously filled ids are read from out_dir/run_ids.json and merged (new non-empty values win)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "run_ids.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    known.update({k: v for k, v in ids.items() if v})
    f.write_bytes((json.dumps(known, indent=1, sort_keys=True) + "\n").encode())
    subs = {k: known.get(k) for k in (PH_EMBED, PH_SK_H2, PH_SK)}
    written, skipped = [], {}
    for spec, _ in jobs:
        data = dump(spec, subs)
        left = unresolved(data.decode())
        if left:
            skipped[spec["name"]] = left
            continue
        (out_dir / f"{spec['name']}.yaml").write_bytes(data)
        written.append(spec["name"])
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-run-id"), ap.add_argument("--skeleton-run-id")
    ap.add_argument("--skeleton-h2-run-id")
    a = ap.parse_args(argv)
    jobs = all_jobs()
    for spec, _ in jobs:
        errs = launch.validate(filled(spec))
        assert not errs, (spec["name"], errs)
        (HERE / f"{spec['name']}.yaml").write_bytes(dump(spec, {}))          # templates: never substituted
    ids = {PH_EMBED: a.embed_run_id, PH_SK: a.skeleton_run_id, PH_SK_H2: a.skeleton_h2_run_id}
    if any(ids.values()) or (HERE / "launch" / "run_ids.json").exists():
        written, skipped = write_launch(jobs, ids, HERE / "launch")
        print(f"jobs/launch: {len(written)} specs written; refused (unresolved placeholders): {skipped or 'none'}")
    by = {m["name"]: m for _, m in jobs}
    core = [m for _, m in jobs if not m["conditional"]]
    print(f"{len(jobs)} job templates written ({len(core)} unconditional)")
    print(f"{'job':<26}{'items':>6}{'est s':>8}{'est $':>8}{'max $':>8}")
    for _, m in jobs:
        flag = "  (conditional)" if m["conditional"] else ""
        print(f"{m['name']:<26}{m['items']:>6}{m['seconds']:>8.0f}{m['est']:>8.2f}{m['max']:>8.2f}{flag}")
    units = yaml.safe_load((HERE / "p0_trackb_units.yaml").read_text())
    ue, um = float(units["estimate_usd"]), float(units["max_usd"])
    print(f"{'p0_trackb_units (kept)':<26}{'':>6}{'':>8}{ue:>8.2f}{um:>8.2f}")
    est_core, max_core = sum(m["est"] for m in core) + ue, sum(m["max"] for m in core) + um
    est_all, max_all = sum(m["est"] for _, m in jobs) + ue, sum(m["max"] for _, m in jobs) + um
    print(f"TOTAL unconditional: estimate ${est_core:.2f}  max ${max_core:.2f}")
    print(f"TOTAL with all conditional jobs: estimate ${est_all:.2f}  max ${max_all:.2f}")
    print(f"reservation check (E0 requested at ${E0_REQUEST:.2f}; prior estimates + this job's max_usd <= {E0_REQUEST:.2f}):")
    for r in reservation_check(jobs):
        print(f"  {r['name']:<26} prior est {r['prior_est']:>6.2f} + max {r['max']:>5.2f} = {r['need']:>6.2f}  {'ok' if r['ok'] else 'OVER'}")
    print(f"from now (E0 spent+reserved ${COMMITTED_NOW:.2f} on 2026-11-07, `python -m opa.budget status`; jobs from {FROM_NOW} on):")
    for r in reservation_check(jobs, COMMITTED_NOW, FROM_NOW):
        print(f"  {r['name']:<26} committed+prior est {r['prior_est']:>6.2f} + max {r['max']:>5.2f} = {r['need']:>6.2f}  {'ok' if r['ok'] else 'OVER'}")
    micro = [m for _, m in jobs if m["name"].startswith("p0-micropilot-")]
    print(f"micro-pilot: {len(micro)} jobs, {sum(m['items'] for m in micro)} system-items, ${sum(m['est'] for m in micro):.2f}, "
          f"groups: {micropilot_groups()}")
    for g in ("L4", "T4"):
        rate_g = launch.rate_per_second(gpu_compute(g, 12), prices())
        print(f"embedding 1 item on {g}: {EMBED_TOK_PER_ITEM / EMBED_TOK_S[g]:.2f} s = ${EMBED_TOK_PER_ITEM / EMBED_TOK_S[g] * rate_g:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
