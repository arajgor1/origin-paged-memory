"""Generator for the TEST-SPLIT Track B job specs (gate G1 passed; G2 failed, so H5 model level, H11, H12 and H13 are not run):

    H4  V3 vs B1 source attribution on 2,451 test items (+ V3's own-origin mass on test probe windows)  jobs/p4/p4-h4-test-*.yaml  (E4)
    H6  remedies R0-R7 on 200 LB-MHQA + 200 MT-LB-MHQA test items at 16K                              jobs/p4/p4-h6-*.yaml       (E3)
    H4  V0 own-origin-mass ratio and the training-free variants V0/V1/B2/B4, 80 per task per variant   jobs/p4/p4-trainfree-test.yaml (E3)

    python jobs/make_p4_test_jobs.py                 write all specs
    python jobs/make_p4_test_jobs.py --data-commands print the commands that generate and upload the two H6 test cells (run by the operator)

Nothing here launches or spends anything. All specs are split test: the entrypoints refuse a test split unless the prereg-v2 tag exists
(args.prereg_tag carries {tag, commit} because the container has no prereg/ folder) and need test seeds from 100000.

H4 evaluation (opa.trackb.train:main, eval_split "test"). V3 and B1 resume their step-600 checkpoints (the stopping rule fired at L = 600), train
0 steps and evaluate the same attribution items (Track B test archive: 64 origins in 8 threads, PG-19 TEST books, windows of C = 6 origins, text
headers, seeds from 100000; the construction of the G2 / monitoring attribution task). 2,451 items = 1,226 (seeds 100000-101225) + 1,225 (101226-102450),
two containers per arm; identical items for both arms (the item is a function of the seed). V3's first container also runs 16 probe windows of C = 8
(V3_own_origin_mass_fraction). The training preflight is skipped in test mode (it would read test books through a training example).
Per-item records (correctness, rank correctness, seed, window, target origin, option ids) are in final_eval.json per container for the McNemar and
bootstrap pairing over the 2,451 seeds.

Time per item. The G2 evaluation run B-V3-...-063b486-d41d (dev, 128 attribution + 32 passkey items + 6 probe windows) took run() 472.2 s of 490.9 s wall
including the preflight: 472.2 / 160 = 2.95 s per evaluation item as an upper bound (the earlier 5.34 s per item was a first guess). Overhead 270 s =
20 s container (as in that run) + 250 s allowance for tokenising the PG-19 test books on first use (about 100 books, not cached on the volume).

H6 (opa.trackb.remedies:main, split test, items_uri). The pilot (19.5 s per item, 13 conditions with greedy generation, 8K) is the cost basis. A window
is always C = 6 origins of about 512 tokens plus their two parents' text whatever the item length, so 16K items cost the same per item as 8K ones; a
10% margin is added. 100 items per container, 2 containers per cell. The items come from files because the generator needs the git tag to emit test data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jobs" / "p4"
_sp = importlib.util.spec_from_file_location("make_p4_train_jobs", ROOT / "jobs" / "make_p4_train_jobs.py")
P4 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P4)

RATE = P4.RATE
PREREG_TAG = {"tag": "prereg-v2", "commit": "c8a0a5e910abf2da315bd00425f5c90d9e82600c"}        # prereg/PREREG_TAG.json
TEST_SEED0 = 100000
V3_RUN = "B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108031640-6c44911-9560"            # V3 final checkpoint, step 600 (stopping rule: L = 600)
V3_SHA = "f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814"                                   # sha256 of the V3 step-600 checkpoint
B1_RUN = "B-B1-mtlbmhqa-archive-archive64-T8-dev-s0-20261108044732-a400cda-cae6"            # B1 parallel run; its step600 checkpoint must exist
L = 600
H4_N = 2451                                                                                   # n_mcnemar(0.78125, 0.05), sample_sizes.yaml
H4_CHUNKS = (1226, 1225)
S_PER_EVAL_ITEM = 2.95                                                                        # upper bound, measured on the G2 evaluation run
H4_OVERHEAD_S = 270.0
PROBE_WINDOWS, PROBE_WINDOW_S = 16, 10.0                                                      # guessed
H6_CELLS = (("lb", "lbmhqa", "late-h2-16k-a0"), ("mt", "mtlbmhqa", "T4-h2-16k-nocross"))
H6_ITEMS, H6_PER_CHUNK, H6_S_PER_ITEM, H6_MARGIN, H6_OVERHEAD_S = 200, 100, 19.5, 1.1, 300.0
TRAINFREE_SMOKE_S, TRAINFREE_SCALE = 913.4, 5                                                 # dev smoke 16/task: 913 s wall; test 80/task = x5
CAP_USD = 2.50


def money(seconds: float) -> float:
    return round(seconds * RATE, 2)


def _fmt(v) -> str:
    return v if isinstance(v, str) else json.dumps(v)


def render(name, env, track, hyp, system, dataset, cell, seed, entry, args, seconds, formula, header) -> tuple:
    est = money(seconds)
    if est > CAP_USD:
        raise ValueError(f"{name}: estimate ${est} exceeds the ${CAP_USD} per-container cap")
    mx = round(2 * est, 2)
    arg_lines = "\n".join(f"  {k}: {_fmt(v)}" for k, v in args.items())
    text = f"""{header}
name: {name}
envelope: {env}
track: {track}
hypotheses: {hyp}
system: {system}
dataset: {dataset}
cell: {cell}
split: test
seed: {seed}
entrypoint: {entry}
args:
{arg_lines}
models:
  track_b_model: {P4.MODEL}
compute:
  kind: gpu
  gpu: L4
  cpu: 2.0
  memory_gib: 16                             # request; the launcher bills CPU and memory at the limits (2x the request)
# Rate from the human-owned rate card (verified 2026-11-06), USD per second, CPU and memory at their limits:
#   L4 0.000222 + 4.0 cores(limit) x 0.0000131 + 32 GiB(limit) x 0.00000222 = 0.00034544 /s (= $1.24/h)
estimate_formula: "{formula} = {seconds:.0f} s x 0.00034544 $/s = ${est:.2f}"
estimate_usd: {est}
max_usd: {mx}                                # first-time job type: 2x the estimate; launcher timeout = floor({mx} / 0.00034544) = {int(mx / RATE):,} s
max_api_usd: 0
"""
    return text, est, mx


# ---------------------------------------------------------------- H4 evaluation
def h4_eval(arm: str, chunk: int, b1_sha: str | None = None) -> tuple:
    n = H4_CHUNKS[chunk - 1]
    seed0 = TEST_SEED0 + sum(H4_CHUNKS[:chunk - 1])
    probe = PROBE_WINDOWS if (arm == "V3" and chunk == 1) else 0
    run = V3_RUN if arm == "V3" else B1_RUN
    args = {**P4.base_args(arm), "max_opt_steps": L, "eval_at_steps": [L], "save_checkpoint_at": [], "eval_split": "test", "prereg_tag": PREREG_TAG,
            "resume_from": f"gs://{{bucket}}/opa/models/{run}/step{L}/checkpoint.pt", "skip_resume_eval": True, "save_adapter": False,
            "monitor_seed0": seed0, "monitor": {"attrib": n, "passkey": 0, "probe": probe}, "time_budget_s": 6000}
    sha = V3_SHA if arm == "V3" else (b1_sha or P4.B1_FINAL_SHA.get(L))
    if sha:
        args["expected_resume_sha256"] = sha
    seconds = H4_OVERHEAD_S + n * S_PER_EVAL_ITEM + probe * PROBE_WINDOW_S
    header = (f"# P4 H4 TEST evaluation (PLAN_V3 s6 after G2 failed; H4 still runs on the test split): {arm} at its step-{L} checkpoint ({run}), zero training steps, "
              f"attribution items {seed0}..{seed0 + n - 1} of the test archive"
              + (f" and {probe} probe windows (V3_own_origin_mass_fraction)" if probe else "") + ".\n"
              f"# Chunk {chunk} of {len(H4_CHUNKS)}; the same seeds are evaluated for both arms (pairing for McNemar + bootstrap over {H4_N} items). "
              f"Needs the prereg-v2 tag (args.prereg_tag) and {'the step600 checkpoint of the B1 run' if arm == 'B1' else 'the V3 final checkpoint'}.")
    return render(f"p4-h4-test-{arm.lower()}-c{chunk:02d}", "E4", "B", "[H4]", arm, "mtlbmhqa-archive", "archive64-T8-test", TEST_SEED0 + sum(H4_CHUNKS[:chunk - 1]),
                  "opa.trackb.train:main", args, seconds,
                  f"{H4_OVERHEAD_S:.0f} s overhead + {n} items x {S_PER_EVAL_ITEM} s (measured upper bound, G2 eval run)" + (f" + {probe} probe windows x {PROBE_WINDOW_S:.0f} s (guessed)" if probe else ""),
                  header)


# ---------------------------------------------------------------- H6 remedies
def h6_items_uri(dataset: str, cell: str) -> str:
    return f"gs://{{bucket}}/opa/data/{dataset}/test/v2/{cell}/items.jsonl.gz"


def h6_chunk(kind: str, dataset: str, cell: str, chunk: int) -> tuple:
    offset = (chunk - 1) * H6_PER_CHUNK
    args = {"model": P4.MODEL, "hf_dir": "/cache/hf/pg19", "split": "test", "seed0": TEST_SEED0, "prereg_tag": PREREG_TAG,
            "items_uri": h6_items_uri(dataset, cell), "offset": offset, "n_items": H6_PER_CHUNK, "C": 6, "time_budget_s": 3200, "item_margin_s": 90}
    seconds = H6_OVERHEAD_S + H6_PER_CHUNK * H6_S_PER_ITEM * H6_MARGIN
    header = (f"# P4 H6 TEST (closure item 18): remedies R0-R7 on {cell} test items {offset}..{offset + H6_PER_CHUNK - 1} of {H6_ITEMS} ({dataset}), windows of C = 6 origins, k = 2 parents, greedy EM.\n"
              f"# The cell must exist first: python -m opa.data.build --dataset {dataset} --split test --cell {cell} --n {H6_ITEMS}; python -m opa.data.upload --dataset {dataset} --split test\n"
              f"# (see `python jobs/make_p4_test_jobs.py --data-commands`). best_remedy_recovery_fraction = max over R3-R6 of acc(R)/acc(R0) pooled over both cells and chunks.")
    return render(f"p4-h6-{kind}-c{chunk:02d}", "E3", "B", "[H6]", "R0-R7", dataset, cell, TEST_SEED0 + offset, "opa.trackb.remedies:main", args, seconds,
                  f"{H6_OVERHEAD_S:.0f} s overhead + {H6_PER_CHUNK} items x {H6_S_PER_ITEM} s (remedies pilot, 8K; a window is 6 origins whatever the item length) x {H6_MARGIN} margin",
                  header)


# ---------------------------------------------------------------- training-free test run
def trainfree_test() -> tuple:
    args = {"model": P4.MODEL, "filler": "pg19", "hf_dir": "/cache/hf/pg19", "split": "test", "seed0": TEST_SEED0, "prereg_tag": PREREG_TAG, "n": 80, "C": 6,
            "probe_C": 8, "n_probe_windows": 16, "headers": True, "variants": ["V0", "V1", "B2", "B4"]}
    seconds = TRAINFREE_SMOKE_S * TRAINFREE_SCALE
    header = ("# P4 training-free variants on the TEST split (closure item 14): V0, V1, B2, B4 x tasks 1-5, 80 items per task per variant (400 per variant), PG-19 test books, "
              "seeds from 100000,\n# 16 probe windows of C = 8 per variant: supplies V0_own_origin_mass_ratio_to_uniform (H4) and the reported contrasts against V0. One container.")
    return render("p4-trainfree-test", "E3", "B", "[H4]", "trainfree", "mtlbmhqa-archive", "archive64-T8-test", TEST_SEED0, "opa.trackb.trainfree:main", args, seconds,
                  f"the dev smoke ({TRAINFREE_SMOKE_S:.0f} s wall at 16 items per task, 12 probe windows) x {TRAINFREE_SCALE} (80 per task; the 64 windows scale less than x5, so this is conservative)",
                  header)


def data_commands() -> list:
    cmds = []
    for _, ds, cell in H6_CELLS:
        cmds.append(f"python -m opa.data.build --dataset {ds} --split test --cell {cell} --n {H6_ITEMS}")
    for ds in sorted({d for _, d, _ in H6_CELLS}):
        cmds.append(f"python -m opa.data.upload --dataset {ds} --split test")
    return cmds


def all_specs(b1_sha: str | None = None) -> dict:
    specs = {}
    for arm in ("V3", "B1"):
        for c in (1, 2):
            specs[f"p4-h4-test-{arm.lower()}-c{c:02d}"] = h4_eval(arm, c, b1_sha)
    for kind, ds, cell in H6_CELLS:
        for c in (1, 2):
            specs[f"p4-h6-{kind}-c{c:02d}"] = h6_chunk(kind, ds, cell, c)
    specs["p4-trainfree-test"] = trainfree_test()
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-commands", action="store_true")
    ap.add_argument("--b1-sha", default=None, help="override the pinned B1 step-600 sha256 (default P4.B1_FINAL_SHA)")
    a = ap.parse_args(argv)
    if a.data_commands:
        print("# only the MT-LB-MHQA 16K build + upload was still needed (both cells are uploaded now)")
        print("\n".join(data_commands()))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (text, est, mx) in all_specs(a.b1_sha).items():
        (OUT / f"{name}.yaml").write_text(text, encoding="utf8", newline="\n")
        print(f"wrote jobs/p4/{name}.yaml  estimate ${est:.2f}  max ${mx:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
