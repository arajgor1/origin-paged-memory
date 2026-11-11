"""Generator for the PLAN_V3 Section 6a Step 2-3 job specs (P4: stopping-rule training of V3, equal-length B1, G2 evaluation) and the operator-side
stopping-rule helper.

    python jobs/make_p4_train_jobs.py v3 [--chunks 8]            V3 continuation chunks jobs/p4/p4-v3-c01.yaml ... (600, 800, ..., 2000 steps)
    python jobs/make_p4_train_jobs.py b1 --steps N [--chunk 1000]  B1 from scratch to exactly N steps (N = V3's final step), in chunks
    python jobs/make_p4_train_jobs.py g2 --v3-step S --b1-step N   G2 evaluation of the final V3 and B1 on the fresh dev seeds 3000+
    python jobs/make_p4_train_jobs.py --decide curve1.json curve2.json ...   apply stopping rule 4 to learning_curve.json files

Nothing here launches or spends anything. The specs are launched one at a time by the operator with `python -m opa.launch run`.

The chain. Chunk k of V3 resumes from the checkpoint the previous chunk saved (k = 1: the step-400 checkpoint of the learning-curve run
STEP1_RUN_ID), trains to optimizer step 400 + 200 k with the IDENTICAL training configuration (TrainConfig's resume check refuses any
difference), evaluates once on the monitoring set (dev seeds 2000-2063, the same evaluation as the curve) at that step, and saves the
checkpoint and the adapter. The resumed state is not re-evaluated (skip_resume_eval), which is what keeps a chunk at one evaluation. The
previous chunk's run id is a placeholder P4_V3_C<k-1>_RUN_ID that the operator fills (the launcher refuses unfilled *_RUN_ID placeholders).
After each chunk: `--decide` on the learning-curve json of the earlier chunks (rule 4).

Prices. Rate 0.00034544 $/s (L4 + CPU and memory at their limits). V3 chunk: 300 s load + 200 steps x 22.75 s (measured) + 240 s evaluation
(measured) + 60 s preflight (guessed) = 5,150 s = $1.78; max_usd = 2 x estimate. B1 (about 5 s per step, measured in the B1 pilot run B-B1-*-20261107123712-c83bf32-a14d, state/unit_costs.md): chunks of 1,000 steps.
Each finished run records the SHA-256 of every checkpoint it wrote (config.json "checkpoint_sha256", run metrics checkpoint_sha256_step<N>
and final_checkpoint_sha256, event checkpoint_saved) so DEV-TRAIN-PIN can be filled.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jobs" / "p4"

STEP1_RUN_ID = "B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261107223205-86f471f-64ea"
STEP1_STEP = 400
CHUNK = 200                      # evaluation interval of the stopping rule
MAX_STEPS = 2000
MIN_STEPS = 400
RISE_THRESHOLD = 0.02            # absolute rise of the own-origin mass fraction between two evaluations 200 steps apart

RATE = 0.00034544                # L4 0.000222 + 4 cores(limit) x 0.0000131 + 32 GiB(limit) x 0.00000222
LOAD_S, V3_STEP_S, EVAL_S, PREFLIGHT_S = 300.0, 22.75, 240.0, 60.0
CKPT_S = 15.0                    # one checkpoint upload (about 0.15 GB for B1), guessed
B1_STEP_S, B1_CHUNK = 5.0, 1000          # B1_STEP_S: measured, state/unit_costs.md (B1 mini-LoRA run B-B1-*-20261107123712-c83bf32-a14d: ~5 s per optimizer step)
G2_SEED0, G2_ATTRIB, G2_PASSKEY, G2_PROBE = 3000, 128, 32, 6
PROBE_ITEM_EQUIV = 10             # assumed cost of one probe window in item-equivalents
PER_CONTAINER_CAP_USD = 2.50
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MONITOR = {"attrib": 64, "passkey": 32, "probe": 3}


def placeholder(arm: str, k: int) -> str:
    return f"P4_{arm}_C{k:02d}_RUN_ID"


def money(seconds: float) -> float:
    return round(seconds * RATE, 2)


def _yaml_args(args: dict) -> str:
    out = []
    for k, v in args.items():
        out.append(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}")
    return "\n".join(out)


def _spec(name: str, system: str, header: str, args: dict, seconds: float, formula: str, hypotheses="[H4, G2]") -> tuple:
    est = money(seconds)
    if est > PER_CONTAINER_CAP_USD:
        raise ValueError(f"{name}: estimate ${est} exceeds the ${PER_CONTAINER_CAP_USD} per-container cap")
    mx = round(2 * est, 2)
    text = f"""{header}
name: {name}
envelope: E4
track: B
hypotheses: {hypotheses}
system: {system}
dataset: mtlbmhqa-archive
cell: archive64-T8
split: dev
seed: 0
entrypoint: opa.trackb.train:main
args:
{_yaml_args(args)}
models:
  track_b_model: {MODEL}
compute:
  kind: gpu
  gpu: L4
  cpu: 2.0
  memory_gib: 16                             # request; the launcher bills CPU and memory at the limits (2x the request)
# Rate from the human-owned rate card (verified 2026-11-06), USD per second, CPU and memory at their limits:
#   L4 0.000222 + 4.0 cores(limit) x 0.0000131 + 32 GiB(limit) x 0.00000222 = 0.00034544 /s (= $1.24/h)
estimate_formula: "{formula} = {seconds:.0f} s x 0.00034544 $/s = ${est:.2f}"
estimate_usd: {est}
max_usd: {mx}                                # 2x the estimate; launcher timeout = floor({mx} / 0.00034544) = {int(mx / RATE):,} s
max_api_usd: 0
"""
    return text, est, mx


def base_args(arm: str) -> dict:
    """The training configuration shared by every run of an arm (identical to the learning-curve run / the B1 pilot except length)."""
    return {"arm": arm, "model": MODEL, "filler": "pg19", "hf_dir": "/cache/hf/pg19", "seed": 0, "headers": True, "eval_every": 0,
            "monitor": dict(MONITOR)}


def v3_chunk(k: int) -> tuple:
    end = STEP1_STEP + CHUNK * k
    prev_end = end - CHUNK
    prev = STEP1_RUN_ID if k == 1 else placeholder("V3", k - 1)
    args = {**base_args("V3"), "max_opt_steps": end, "eval_at_steps": [end], "save_checkpoint_at": [end],
            "resume_from": f"gs://{{bucket}}/opa/models/{prev}/step{prev_end}/checkpoint.pt", "skip_resume_eval": True,
            "time_budget_s": 9000}
    seconds = LOAD_S + PREFLIGHT_S + CHUNK * V3_STEP_S + EVAL_S
    header = (f"# P4 Step 2 (PLAN_V3 6a): V3 chunk {k}, optimizer steps {prev_end} -> {end}, resumed from the checkpoint of "
              f"{'the learning-curve run' if k == 1 else placeholder('V3', k - 1)} (step{prev_end}). One evaluation (monitoring set, dev seeds 2000-2063)\n"
              f"# at step {end}; saves step{end}/checkpoint.pt and adapter.pt. "
              + (f"Fill {placeholder('V3', k - 1)} with the run id of chunk {k - 1} before launching; then apply rule 4 with --decide on the finished chunks."
                 if k > 1 else "No placeholder: chunk 1 resumes from the learning-curve run. Afterwards apply rule 4 with --decide."))
    return _spec(f"p4-v3-c{k:02d}", "V3", header, args, seconds,
                 f"300 s load + 60 s preflight (guessed) + {CHUNK} steps x {V3_STEP_S} s (measured) + 240 s evaluation (measured)")


def b1_chunk(j: int, n_steps: int, chunk: int = B1_CHUNK) -> tuple:
    check_steps(n_steps)
    n_chunks = math.ceil(n_steps / chunk)
    if not 1 <= j <= n_chunks:
        raise ValueError("chunk index out of range")
    end = min(n_steps, j * chunk)
    prev_end = (j - 1) * chunk
    args = {**base_args("B1"), "max_opt_steps": end, "eval_at_steps": [end], "save_checkpoint_at": [end], "time_budget_s": 9000}
    if j > 1:
        args["resume_from"] = f"gs://{{bucket}}/opa/models/{placeholder('B1', j - 1)}/step{prev_end}/checkpoint.pt"
        args["skip_resume_eval"] = True
    seconds = LOAD_S + PREFLIGHT_S + (end - prev_end) * B1_STEP_S + EVAL_S * (1 if j > 1 else 2)
    header = (f"# P4 Step 2 (PLAN_V3 6a rule 6): B1 chunk {j} of {n_chunks}, optimizer steps {prev_end} -> {end} of exactly N = {n_steps} (V3's final step). "
              f"B1 is trained from scratch (chunk 1) with the pilot's data and configuration (text headers, LoRA r16, same stream) except length;\n"
              f"# evaluated once at the end of the chunk on the monitoring set. "
              f"{'Fill ' + placeholder('B1', j - 1) + ' with the previous chunk run id.' if j > 1 else 'No resume.'}")
    return _spec(f"p4-b1-c{j:02d}", "B1", header, args, seconds,
                 f"300 s load + 60 s preflight (guessed) + {end - prev_end} steps x {B1_STEP_S} s (measured, B1 pilot run c83bf32-a14d, state/unit_costs.md) + {'240' if j > 1 else '2 x 240'} s evaluation")


def b1par_saves(j: int, chunk: int = B1_CHUNK, n_steps: int = MAX_STEPS) -> list:
    """Checkpoint steps of B1-parallel chunk j: every multiple of 200 from 400 up to the chunk end that lies after the previous chunk's end,
    plus the chunk end itself (V3's stopping rule can stop at any of 400, 600, ..., 2000, and B1 must be evaluated at exactly that step)."""
    end, prev_end = min(n_steps, j * chunk), (j - 1) * chunk
    return sorted({s for s in range(MIN_STEPS, end + 1, CHUNK) if s > prev_end} | {end})


def b1par_chunk_of(step: int, chunk: int = B1_CHUNK, n_steps: int = MAX_STEPS) -> int:
    """The chunk whose run saved the step-`step` checkpoint (the first chunk whose range contains it)."""
    check_steps(step)
    for j in range(1, math.ceil(n_steps / chunk) + 1):
        if step in b1par_saves(j, chunk, n_steps) and step > (j - 1) * chunk:
            return j
    raise ValueError(f"no B1-parallel chunk saves step {step}")


def b1par_chunk(j: int, chunk: int = B1_CHUNK, n_steps: int = MAX_STEPS) -> tuple:
    """B1 from scratch to 2,000 steps IN PARALLEL with V3 (owner instruction 2026-11-08): chunk j trains to min(2000, j x chunk), saves a checkpoint
    at every multiple of 200 from 400 on (and at its end), evaluates nothing (no_eval), and chunk j+1 resumes from its end checkpoint. When V3's
    stopping rule fires at L, `g2 --v3-step L --b1-parallel` evaluates B1 from the step-L checkpoint."""
    n_chunks = math.ceil(n_steps / chunk)
    if not 1 <= j <= n_chunks:
        raise ValueError("chunk index out of range")
    end, prev_end = min(n_steps, j * chunk), (j - 1) * chunk
    saves = b1par_saves(j, chunk, n_steps)
    args = {k: v for k, v in base_args("B1").items() if k != "monitor"}
    args.update({"max_opt_steps": end, "save_checkpoint_at": saves, "no_eval": True, "time_budget_s": 10000})
    if j > 1:
        args["resume_from"] = f"gs://{{bucket}}/opa/models/{placeholder('B1PAR', j - 1)}/step{prev_end}/checkpoint.pt"
    seconds = LOAD_S + PREFLIGHT_S + (end - prev_end) * B1_STEP_S + CKPT_S * len(saves)
    header = (f"# P4 Step 2 (owner instruction 2026-11-08): B1 IN PARALLEL with V3, chunk {j} of {n_chunks}, optimizer steps {prev_end} -> {end} of 2000. From scratch (chunk 1) "
              f"with the pilot's data and configuration; NO evaluations (no_eval); checkpoints at steps {saves}.\n"
              f"# {'Fill ' + placeholder('B1PAR', j - 1) + ' with the previous chunk run id.' if j > 1 else 'No resume.'} "
              f"When V3 stops at L: `g2 --v3-step L --b1-parallel` evaluates B1 at its step-L checkpoint.")
    return _spec(f"p4-b1-par-c{j:02d}", "B1", header, args, seconds,
                 f"300 s load + 60 s preflight (guessed) + {end - prev_end} steps x {B1_STEP_S} s (measured, B1 pilot run c83bf32-a14d) + {len(saves)} checkpoint uploads x {CKPT_S:.0f} s (guessed)")


V3_FINAL_SHA = {600: "f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814"}      # sha256 of the V3 checkpoint by step (the L = 600 stop)
B1_FINAL_SHA = {600: "eeccb50e8f89f95dd82f008a376a6b6b245137c633a3017fd01b10b136ed8bff"}      # sha256 of the B1 step-600 checkpoint (prereg/deviations/DEV-TRAIN-PIN.md)


def g2_eval(arm: str, step: int, b1_parallel: bool = False, sha: str | None = None) -> tuple:
    """Evaluation only (zero training steps) of a final checkpoint on the fresh G2 dev set: seeds 3000 and up, never used before.
    With b1_parallel the B1 checkpoint is the step-`step` one saved by whichever B1-parallel chunk reached it."""
    if arm == "B1" and b1_parallel:
        run = placeholder("B1PAR", b1par_chunk_of(step))
    else:
        run = "P4_V3_FINAL_RUN_ID" if arm == "V3" else "P4_B1_FINAL_RUN_ID"
    args = {**base_args(arm), "max_opt_steps": step, "eval_at_steps": [step], "save_checkpoint_at": [],
            "resume_from": f"gs://{{bucket}}/opa/models/{run}/step{step}/checkpoint.pt", "skip_resume_eval": True, "save_adapter": False,
            "monitor_seed0": G2_SEED0, "monitor": {"attrib": G2_ATTRIB, "passkey": G2_PASSKEY, "probe": G2_PROBE}, "time_budget_s": 1200}
    sha = sha or (V3_FINAL_SHA if arm == "V3" else B1_FINAL_SHA).get(step)
    if sha:
        args["expected_resume_sha256"] = sha
    # the measured 240 s evaluation = 64 attribution + 32 passkey items + 3 probe windows. A probe window is weighted as PROBE_ITEM_EQUIV = 10
    # item-equivalents (an assumption: it was not timed separately), so 240 s = (64 + 32 + 3 x 10) = 126 units; G2 has 128 + 32 + 6 x 10 = 220 units
    unit = EVAL_S / (64 + 32 + 3 * PROBE_ITEM_EQUIV)
    seconds = LOAD_S + PREFLIGHT_S + unit * (G2_ATTRIB + G2_PASSKEY + G2_PROBE * PROBE_ITEM_EQUIV)
    header = (f"# P4 Step 3 (PLAN_V3 6a): G2 evaluation of the final {arm} checkpoint (step {step}) on FRESH dev seeds {G2_SEED0}+ "
              f"({G2_ATTRIB} attribution, {G2_PASSKEY} passkey items, {G2_PROBE} probe windows). Evaluation only: the entrypoint resumes, trains 0 steps,\n"
              f"# and writes final_eval.json (per-item correctness and identities) and learning_curve.json. Fill {run} with the run id of the final {arm} chunk.")
    return _spec(f"p4-g2-eval-{arm.lower()}", arm, header, args, seconds,
                 "300 s load + 60 s preflight (guessed) + the measured 240 s evaluation (64 + 32 items + 3 probe windows = 126 units with a probe window "
                 f"weighted as {PROBE_ITEM_EQUIV} items, an assumption) scaled to {G2_ATTRIB} + {G2_PASSKEY} items + {G2_PROBE} windows = "
                 f"{G2_ATTRIB + G2_PASSKEY + G2_PROBE * PROBE_ITEM_EQUIV} units")


# ============================================================================ the stopping rule (PLAN_V3 6a Step 2, rule 4)


def curve_points(paths) -> dict:
    """{opt_step: own-origin mass fraction} from learning_curve.json files (the {"curve": [...]} form, or a bare list of entries);
    entries with tag step<N> and a probe block; a later file overrides an earlier one for the same step."""
    pts: dict = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        entries = data["curve"] if isinstance(data, dict) else data
        for e in entries:
            tag = str(e.get("tag", ""))
            if tag.startswith("step") and e.get("probe"):
                pts[int(e["opt_step"])] = float(e["probe"]["own_origin_mass_fraction"])
    return pts


def decide(points: dict, threshold: float = RISE_THRESHOLD, max_steps: int = MAX_STEPS, min_steps: int = MIN_STEPS, interval: int = CHUNK) -> dict:
    """Rule 4, applied to the whole history. Walk s = 400, 600, ... up to the latest evaluated step and return the FIRST STOP:
    STOP at s >= max_steps, or when the absolute rise of the own-origin mass fraction between steps s - 200 and s is below 0.02 (a rise
    of exactly 0.02 continues; the difference is rounded to 10 decimals so that float noise cannot flip the boundary). Never stop below
    step 400. If the first STOP is before the latest step the run has trained past its stopping point and this raises. A missing
    evaluation in the chain raises. The checkpoint to use on STOP is the one at the stopping step - the final one, never the best."""
    if not points:
        raise ValueError("no evaluation points")
    latest = max(points)
    best = max(points, key=points.get)
    out = {"latest_step": latest, "latest_fraction": points[latest], "best_step_observed": best,
           "note": "the final checkpoint is used, never the best (best_step_observed is informational only)"}
    if latest < min_steps:
        return {**out, "decision": "CONTINUE", "reason": f"never stop before step {min_steps}", "next_chunk_end": latest + interval}
    stop = None
    s = min_steps
    while s <= latest:
        if s not in points:
            raise ValueError(f"the evaluation at step {s} is missing from the chain {min_steps}, {min_steps + interval}, ..., {latest}")
        if s >= max_steps:
            stop = (s, f"reached the {max_steps}-step limit", None, None)
            break
        prev = s - interval
        if prev in points:
            rise = points[s] - points[prev]
            if round(rise, 10) < threshold:
                stop = (s, f"rise {rise:.4f} < {threshold} between steps {prev} and {s}", prev, rise)
                break
        elif s > min_steps:
            raise ValueError(f"the evaluation at step {prev} is missing (needed for the rise to step {s})")
        s += interval
    if stop is not None:
        step, reason, prev, rise = stop
        if step < latest:
            raise ValueError(f"trained past the stopping point at step {step} (latest evaluated step {latest}): {reason}")
        res = {**out, "decision": "STOP", "reason": reason, "use_checkpoint_step": step}
        if prev is not None:
            res.update({"previous_step": prev, "rise": rise})
        return res
    prev = latest - interval
    rise = points[latest] - points[prev] if prev in points else None
    return {**out, "previous_step": prev if rise is not None else None, "rise": rise, "decision": "CONTINUE",
            "reason": (f"rise {rise:.4f} >= {threshold} between steps {prev} and {latest}" if rise is not None
                       else "no evaluation 200 steps earlier to compare with"),
            "next_chunk_end": latest + interval, "next_chunk_index": (latest + interval - STEP1_STEP) // interval}


def check_steps(n: int) -> int:
    """N (V3's final step) must be a multiple of 200 in [400, 2000]: the stopping rule only stops at an evaluation."""
    if n % CHUNK or not MIN_STEPS <= n <= MAX_STEPS:
        raise ValueError(f"N = {n} is not a multiple of {CHUNK} in [{MIN_STEPS}, {MAX_STEPS}]")
    return n


def steps_from_decision(path) -> int:
    """V3's final step from the JSON printed by --decide (it must be a STOP)."""
    d = json.loads(Path(path).read_text())
    if d.get("decision") != "STOP":
        raise ValueError(f"{path}: the decision is {d.get('decision')!r}, not STOP")
    return check_steps(int(d["use_checkpoint_step"]))


# ============================================================================ CLI


def write(texts: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (text, est, mx) in texts:
        (OUT / f"{name}.yaml").write_text(text, encoding="utf8", newline="\n")
        print(f"wrote jobs/p4/{name}.yaml  estimate ${est:.2f}  max ${mx:.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="?", choices=["v3", "b1", "b1-parallel", "g2"])
    ap.add_argument("--chunks", type=int, default=(MAX_STEPS - STEP1_STEP) // CHUNK)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--steps-from", metavar="DECIDE_JSON", help="N = use_checkpoint_step of a --decide STOP json")
    ap.add_argument("--chunk", type=int, default=B1_CHUNK)
    ap.add_argument("--v3-step", type=int)
    ap.add_argument("--b1-step", type=int)
    ap.add_argument("--b1-sha", default=None, help="g2: sha256 of the B1 step-L checkpoint, set as expected_resume_sha256")
    ap.add_argument("--b1-parallel", action="store_true", help="g2: evaluate B1 from its parallel run's step-L checkpoint")
    ap.add_argument("--decide", nargs="+", metavar="CURVE_JSON")
    a = ap.parse_args(argv)
    if a.decide:
        res = decide(curve_points(a.decide))
        print(json.dumps(res, indent=2))
        return 0
    if a.what == "v3":
        write([(f"p4-v3-c{k:02d}", v3_chunk(k)) for k in range(1, a.chunks + 1)])
    elif a.what == "b1":
        n = steps_from_decision(a.steps_from) if a.steps_from else a.steps
        if not n:
            ap.error("b1 needs --steps N or --steps-from DECIDE_JSON (V3's final step)")
        check_steps(n)
        write([(f"p4-b1-c{j:02d}", b1_chunk(j, n, a.chunk)) for j in range(1, math.ceil(n / a.chunk) + 1)])
    elif a.what == "b1-parallel":
        write([(f"p4-b1-par-c{j:02d}", b1par_chunk(j, a.chunk)) for j in range(1, math.ceil(MAX_STEPS / a.chunk) + 1)])
    elif a.what == "g2":
        if a.b1_parallel:
            L = a.v3_step
            if not L:
                ap.error("g2 --b1-parallel needs --v3-step L")
            check_steps(L)
            write([("p4-g2-eval-v3", g2_eval("V3", L)), ("p4-g2-eval-b1", g2_eval("B1", L, b1_parallel=True, sha=a.b1_sha))])
        else:
            if not (a.v3_step and a.b1_step):
                ap.error("g2 needs --v3-step and --b1-step")
            if a.v3_step != a.b1_step:
                ap.error("B1 must be trained for exactly V3's step count")
            check_steps(a.v3_step)
            write([("p4-g2-eval-v3", g2_eval("V3", a.v3_step)), ("p4-g2-eval-b1", g2_eval("B1", a.b1_step, sha=a.b1_sha))])
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
