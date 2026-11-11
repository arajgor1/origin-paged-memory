"""Training-free Track B variants V0, V1, B2, B4 (protocol 5.3) on the synthetic archive: tasks 1-5 and the per-layer / per-head measures.

    entrypoint opa.trackb.trainfree:main(args, ctx)       job spec jobs/p3_trainfree_v0_v1_b2_b4.yaml

The variants (all on the base Qwen3-4B-Instruct-2507 in bf16, nothing trained; positions are M5 fresh frames for all four)
  V0  nothing else (OAA off).
  V1  fixed-lambda hyperbolic penalty (OAA with the structural term only, lambda = 2 on every head = the protocol's initial value) on the
      training-free tree layout of paper Section 7.5 (opa.geometry.TreeLayout: 64 origins, 8 threads of 8, each origin one hyperbolic
      step of length 1 outward from its parent along its thread's ray; distance = Poincare distance in the lowest-common-ancestor
      frame, float32). A token's penalty is -lambda * d(origin of the query, origin of the key); an unanchored query has none.
  B2  APE-style: one shared prefix ("The following is a document.") before the window, every origin's fresh frame starting at its end.
  B4  Graph-KV-style structured mask: an origin attends only to itself and to its graph parent (the previous origin of its thread) if that
      is co-loaded; the unanchored question / continuation attends to everything.

Tasks (protocol 5.3), n items each per variant, window of C = 6 origins of ~512 tokens drawn at random from the 64 and loaded in random order
  1 passkey   one origin holds "The access code of X is <5 digits>"; EM of the digits (teacher-forced = greedy).
  2 order     two co-loaded origins of DIFFERENT threads state different values of "The setting of X"; "Which value was recorded first? A: ..
              B: .." where first means the lower archive id (arrival order). Rank accuracy over the tokens " A" / " B"; chance 0.5; needs order, which M5 removes.
  3 attrib    which page (label word) contains a quoted sentence; EM and rank accuracy over the window's labels (chance 1/C).
  4 confuse   thread confusion: two threads each hold "The pressure limit of X is <value> kPa" for the SAME entity name at the SAME token
              offset of their (headered) origin sequence; the question asks for the value "in the current thread" and the question tokens are
              anchored at the working origin, an origin ADJACENT to the target fact in the same thread (so its tree distance to the target
              fact is smaller than to the decoy; checked against the tables for every item). Choice A/B, rank accuracy, chance 0.5. Decisions: identical entity names rather than one-character twins
              (the protocol says near-duplicate; identical names make the item purely structural); the working origin is a co-loaded origin of
              the target thread next to the fact (the archive has no spare node for a next origin). The thread cue exists only for V1
              (disclosed): the other variants see no thread information in the question.
  5 lm        perplexity of the 48 tokens continuing the last origin's book, after the window.
Per-layer / per-head measures (protocol 5.3) on probe windows of C = 8: own-origin mass, own-thread mass, ghost ratio, router entropy H(pi)
(opa.trackb.metrics.AttentionProbe, arrays [layer][head] in probe.json), and a linear probe (ridge regression on one-hot labels, one per
layer and head) that predicts the global origin id of a token from the output of that head, trained on some probe windows and tested on others
(chance 1/64; the result also gives the majority-label baseline and the share of test labels unseen in training). B4 positions are the M5
shared frames, not Graph-KV's after-parent positions (disclosed). The uniform baselines of the probe use the keys each variant can actually see
(its mask), and router entropy is over the origin masses renormalised by their sum (the prefix mass of B2 is reported separately). Predictions to read, not tuned: V0 own-origin mass near 1/C and chance on tasks 2-4; V1 better than V0 on task 4.

Splits and seeds. split "dev" with seeds from seed0 (default 900, the pilot evaluation range 900-999); split "test" is refused unless the
pre-registration tag exists (prereg/PREREG_TAG.json, or args["prereg_tag"] = {"tag": "prereg-v2", "commit": ...} passed by the launcher) and
needs seed0 >= 100000 and the PG-19 test books. Intention-to-treat: an item whose forward pass raises is scored wrong (or perplexity
missing, counted in n_failed), never dropped. Outputs: items/<variant>_<task>.json (per-item raw outputs), inputs/<task>.json (the shared
inputs), probe.json, metrics.json, plus the flat run metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import torch

from opa import PROJECT_DIR
from opa.trackb import arms as A
from opa.trackb import attention as att
from opa.trackb import data as D
from opa.trackb import metrics as M

VARIANTS = ("V0", "V1", "B2", "B4")
TASKS = ("passkey", "order", "attrib", "confuse", "lm")
SHARED_PREFIX_TEXT = "The following is a document."
ROLE_CODE = {"train": 0, "eval": 1, "test": 2}
CHANCE = {"order": 0.5, "confuse": 0.5}           # attrib: 1 / C (per run), passkey: ~0


class EngineeringStop(RuntimeError):
    """Every item of a (variant, task) failed: the run stops (recorded as failed, nothing scored)."""


class SplitRefused(RuntimeError):
    """Raised for split=test before the pre-registration tag exists."""


def check_split(split: str, seed0: int, args: dict | None = None, n: int = 1) -> None:
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if split == "dev":
        if not 0 <= int(seed0) or int(seed0) + int(n) - 1 >= 100000:
            raise ValueError(f"dev seeds are below 100000 (seed0 {seed0} + n {n} - 1 reaches the test range)")
        return
    tag = (args or {}).get("prereg_tag") or {}
    has_file = (PROJECT_DIR / "prereg" / "PREREG_TAG.json").exists()
    if not (has_file or (tag.get("tag") == "prereg-v2" and tag.get("commit"))):
        raise SplitRefused("split=test needs the prereg-v2 tag (prereg/PREREG_TAG.json or args['prereg_tag'])")
    if int(seed0) < 100000:
        raise ValueError("test-split seeds start at 100000")


# ============================================================================ task builders (tasks 2 and 4; 1, 3, 5 come from data.Sampler)


def _label_tokens(enc) -> tuple:
    a, b = enc.encode(" A"), enc.encode(" B")
    if len(a) != 1 or len(b) != 1 or a == b:
        raise RuntimeError("the tokenizer must encode ' A' and ' B' as distinct single tokens")
    return a[0], b[0]


def _digits(rng, n):
    return "".join(str(int(x)) for x in rng.integers(0, 10, n))


def _splice(t, sent, pos):
    return t[:pos] + sent + t[pos:]


def build_order(sm: D.Sampler, seed: int, C: int, role: str) -> D.Example:
    rng = np.random.default_rng([int(seed), 5, ROLE_CODE[role]])
    enc = sm.enc
    for _ in range(100):                                                            # a window with two origins of different threads
        order, tokens, _ = sm.window(rng, C, role)
        if len({sm.arch.thread(c) for c in order}) >= 2:
            break
    cands = [(x, y) for x in order for y in order if x < y and sm.arch.thread(x) != sm.arch.thread(y)]
    a, b = cands[int(rng.integers(len(cands)))]                                      # a = lower archive id = arrived first; different threads
    from opa.data.names import NameGen
    ent = NameGen(rng).entity(p_suffix=0.0).full
    va = _digits(rng, 5)
    vb = _digits(rng, 5)
    while vb == va:
        vb = _digits(rng, 5)
    vals = {a: va, b: vb}
    toks = {}
    for c in order:
        t = tokens[c]
        if c in vals:
            sent = enc.encode(f" The setting of {ent} is {vals[c]}.")
            t = _splice(t, sent, int(rng.integers(8, max(9, len(t) - 8))))
        toks[c] = sm._origin_tokens(c, t)
    first_is_A = bool(rng.random() < 0.5)
    left, right = (va, vb) if first_is_A else (vb, va)
    q = enc.encode(f"\n\nThe setting of {ent} was recorded in two places. Which value was recorded first? A: {left}. B: {right}.\nAnswer:")
    tok_a, tok_b = _label_tokens(enc)
    gold = tok_a if first_is_A else tok_b
    return sm._assemble("order", order, toks, q + [gold], len(q), 1, options=[tok_a, tok_b], gold_option=0 if first_is_A else 1,
                        meta={"first_origin": a, "second_origin": b, "first_value": va, "second_value": vb, "first_is_A": first_is_A})


def build_confuse(sm: D.Sampler, seed: int, C: int, role: str) -> D.Example:
    rng = np.random.default_rng([int(seed), 6, ROLE_CODE[role]])
    arch, enc = sm.arch, sm.enc
    tA, tB = (int(t) for t in rng.choice(arch.n_threads, size=2, replace=False))
    thr_ids = lambda t: [c for c in arch.ids if arch.thread(c) == t]
    fa = int(rng.choice(thr_ids(tA)))                                              # fact origin of the target thread
    ca = int(rng.choice([x for x in (fa - 1, fa + 1) if x in thr_ids(tA)]))        # working origin: ADJACENT to it in the same thread
    fb = int(rng.choice(thr_ids(tB)))
    dist = arch.tables.dist
    ix = lambda c: arch.tables.ids.index(c)
    if float(dist.abs().max()) > 0:                                                # structural property V1 relies on: working origin nearer to the target fact
        for _ in range(50):
            if float(dist[ix(ca), ix(fa)]) < float(dist[ix(ca), ix(fb)]):
                break
            tB = int(rng.choice([t for t in range(arch.n_threads) if t != tA]))
            fb = int(rng.choice(thr_ids(tB)))
        else:
            raise RuntimeError("could not place the decoy farther from the working origin than the target fact")
    rest = [c for c in arch.ids if c not in (fa, ca, fb)]
    extra = [int(x) for x in rng.choice(rest, size=max(C - 3, 0), replace=False)]
    order = [int(x) for x in rng.permutation([fa, ca, fb] + extra)]
    by_thread: dict = {}
    for c in sorted(order):
        by_thread.setdefault(arch.thread(c), []).append(c)
    pool = sm.pools[role]
    threads = sorted(by_thread)
    pick = rng.choice(pool, size=len(threads), replace=len(pool) < len(threads))
    tokens = {}
    for t, book in zip(threads, pick):
        blocks = sm.filler.take_blocks(rng, len(by_thread[t]), sm.ftok, sm.B, books=[int(book)])
        for c, blk in zip(by_thread[t], blocks):
            tokens[c] = enc.encode(blk["text"])
    from opa.data.names import NameGen
    ent = NameGen(rng).entity(p_suffix=0.0).full
    vA = int(rng.integers(100, 1000))
    vB = int(rng.integers(100, 1000))
    while vB == vA:
        vB = int(rng.integers(100, 1000))
    toks = {c: sm._origin_tokens(c, tokens[c]) for c in order}                      # headers first: the offset is in the final origin sequence
    hdr = max(len(sm._origin_tokens(c, [])) for c in (fa, fb))
    pos = int(rng.integers(hdr + 8, max(hdr + 9, min(len(toks[fa]), len(toks[fb])) - 40)))   # the SAME token offset in both threads' sequences
    for c in (fa, fb):
        toks[c] = _splice(toks[c], enc.encode(f" The pressure limit of {ent} is {vA if c == fa else vB} kPa."), pos)
    target_first = bool(rng.random() < 0.5)
    left, right = (vA, vB) if target_first else (vB, vA)
    q = enc.encode(f"\n\nIn the current thread, what is the pressure limit of {ent}? A: {left} kPa. B: {right} kPa.\nAnswer:")
    tok_a, tok_b = _label_tokens(enc)
    gold = tok_a if target_first else tok_b
    ex = sm._assemble("confuse", order, toks, q + [gold], len(q), 1, options=[tok_a, tok_b], gold_option=0 if target_first else 1,
                      meta={"target_thread": tA, "other_thread": tB, "fact_origin": fa, "decoy_origin": fb, "working_origin": ca,
                            "target_value": vA, "decoy_value": vB, "offset": pos})
    n_win = ex.T - len(q) - 1
    ex.origin_tok[n_win:] = ca                       # the question tokens sit at the working origin (struct term of V1 only; the segment stays unanchored)
    return ex


def build_example(sm: D.Sampler, task: str, seed: int, C: int, role: str) -> D.Example:
    if task == "order":
        return build_order(sm, seed, C, role)
    if task == "confuse":
        return build_confuse(sm, seed, C, role)
    return sm.make(task, seed, C, role=role)


# ============================================================================ scoring


def score_example(ex: D.Example, logits: torch.Tensor, loss) -> dict:
    """Task scorer; logits [ans_len, V]. Returns correct (0/1) for the choice tasks, the loss for lm, and raw outputs."""
    out: dict = {"kind": ex.kind}
    if ex.kind == "lm":
        out.update({"loss": float(loss), "n_tokens": int(ex.ans_len)})
        return out
    greedy = logits.argmax(-1).cpu()
    out["greedy_tokens"] = [int(x) for x in greedy]
    out["top5_first"] = [int(x) for x in logits[0].topk(5).indices.cpu()]
    if ex.kind == "passkey":
        out["correct"] = int(bool((greedy == ex.targets).all()))
        return out
    opt = torch.tensor(ex.options, device=logits.device)
    ol = logits[0, opt].float().cpu()
    out["option_logits"] = [float(x) for x in ol]
    out["pred_option"] = int(ol.argmax())
    out["gold_option"] = int(ex.gold_option)
    out["correct"] = int(int(ol.argmax()) == int(ex.gold_option))
    out["correct_greedy"] = int(int(greedy[0]) == int(ex.targets[0]))
    return out


# ============================================================================ evaluation


def arm_example(ex: D.Example, arm: A.Arm, prefix_ids: list) -> D.Example:
    return A.with_shared_prefix(ex, prefix_ids) if arm.shared_prefix else ex


@torch.no_grad()
def eval_task(model, arm: A.Arm, task: str, sm: D.Sampler, tables, prefix_ids, n: int, seed0: int, C: int, role: str, ctx=None) -> tuple:
    """ITT evaluation of one (variant, task): returns (per-item records, inputs). A failure is recorded and counted, never dropped."""
    model.eval()
    recs, inputs = [], []
    for i in range(n):
        seed = seed0 + i
        rec = {"seed": seed, "task": task, "variant": arm.name, "failed": False}
        try:
            ex = build_example(sm, task, seed, C, role)
            tail_len = ex.segments[-1].length if ex.segments[-1].origin == att.NULL_ORIGIN else 0
            inputs.append({"seed": seed, "task": task, "window": ex.window, "T": ex.T, "tail_ids": [int(t) for t in ex.ids[ex.T - tail_len:]],
                           "meta": json.loads(json.dumps(ex.meta, default=int)), "gold_ids": [int(t) for t in ex.targets],
                           "options": ex.options, "gold_option": ex.gold_option})
            exv = arm_example(ex, arm, prefix_ids)
            rec["ids_sha256"] = hashlib.sha256(ex.ids.numpy().tobytes()).hexdigest()              # the shared input
            rec["variant_ids_sha256"] = hashlib.sha256(exv.ids.numpy().tobytes()).hexdigest()     # what this variant actually read
            loss, logits = A.forward_example(model, exv, arm, tables)
            rec.update(score_example(exv, logits, loss))
            rec["T"] = exv.T
        except Exception as e:                      # intention to treat: scored wrong, kept
            rec.update({"failed": True, "error": f"{type(e).__name__}: {e}", "correct": 0 if task != "lm" else None})
            if ctx is not None:
                ctx.log({"action": "item_failed", "variant": arm.name, "task": task, "seed": seed, "error": rec["error"]})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        recs.append(rec)
    return recs, inputs


def summarize_task(task: str, recs: list, C: int | None = None) -> dict:
    from opa import stats
    n = len(recs)
    out = {"n": n, "n_failed": int(sum(r["failed"] for r in recs))}
    if task == "lm":
        ok = [r for r in recs if not r["failed"]]
        tot = sum(r["loss"] * r["n_tokens"] for r in ok)
        cnt = sum(r["n_tokens"] for r in ok)
        out["ppl"] = math.exp(tot / cnt) if cnt else None
        return out
    c = np.array([r.get("correct", 0) for r in recs], float)
    out["accuracy"] = float(c.mean()) if n else None
    if n:
        out["ci95_points"] = [float(x) for x in stats.paired_bootstrap(c, np.zeros(n))[1]]
    out["chance"] = (1.0 / C if C else None) if task == "attrib" else CHANCE.get(task)
    return out


def probe_variant(model, arm: A.Arm, sm: D.Sampler, tables, prefix_ids, n_windows: int, seed0: int, C: int, role: str, stride: int = 16,
                  ctx=None) -> dict:
    """Own-origin / own-thread / ghost / entropy per layer and head, and the linear probe of the origin id from head outputs."""
    model.eval()
    n_layers = len(att.attention_modules(model))
    layers = sorted({0, n_layers // 2, n_layers - 1})
    summ, lh, outs, labels, ok = [], [], [], [], 0
    for i in range(n_windows):
        try:
            ex = arm_example(sm.make("probe", seed0 + i, C, role=role), arm, prefix_ids)
            probe = M.AttentionProbe(ex, sm.arch)
            idx, lab = M.probe_token_indices(ex, stride)
            coll = M.HeadOutputCollector(layers, idx)
            att.PROBE, att.PROBE_OUT = probe, coll
            try:
                with torch.no_grad():
                    A.forward_example(model, ex, arm, tables, probe_kernel=True)
            finally:
                att.PROBE = att.PROBE_OUT = None
            summ.append(probe.summary())
            lh.append(probe.layer_head())
            outs.append(coll.buf)
            labels.append(lab)
            ok += 1
        except Exception as e:
            if ctx is not None:
                ctx.log({"action": "probe_failed", "variant": arm.name, "window": i, "error": f"{type(e).__name__}: {e}"})
    res = {"n_windows": n_windows, "n_ok": ok, "C": C}
    if not summ:
        return res
    res["summary"] = M.mean_probe(summ)
    res["layer_head"] = {k: (np.mean([x[k] for x in lh], axis=0).tolist() if k != "layers" else lh[0]["layers"])
                         for k in lh[0]}
    test_w = [i for i in range(len(outs)) if i % 4 == 3] or [len(outs) - 1]
    train_w = [i for i in range(len(outs)) if i not in test_w] or [0]
    cat = lambda ws, l: torch.cat([outs[w][l] for w in ws], dim=1)
    train = {l: cat(train_w, l) for l in layers}
    test = {l: cat(test_w, l) for l in layers}
    ytr = [y for w in train_w for y in labels[w]]
    yte = [y for w in test_w for y in labels[w]]
    res["linear_probe"] = M.linear_probe_accuracy(train, [y - 1 for y in ytr], test, [y - 1 for y in yte], n_classes=len(sm.arch.ids))
    res["linear_probe"]["mean_accuracy_by_layer"] = {str(l): float(np.mean(v)) for l, v in res["linear_probe"]["accuracy"].items()}
    return res


def contrasts(per_variant: dict, tasks=TASKS, base: str = "V0") -> dict:
    from opa import stats
    out: dict = {}
    for t in tasks:
        if t == "lm":
            continue
        a0 = np.array([r.get("correct", 0) for r in per_variant[base][t]], float)
        for v, d in per_variant.items():
            if v == base:
                continue
            a = np.array([r.get("correct", 0) for r in d[t]], float)
            if len(a) == len(a0) and len(a):
                ct = stats.contrast(a, a0, key=f"{v}_vs_{base}_{t}")
                out[f"{v}_vs_{base}_{t}"] = {k: ct[k] for k in ("value", "ci_lo", "ci_hi", "n", "n10", "n01", "p_raw")}
    return out


def evaluate_all(model, sm, tables, enc, cfg: dict, ctx) -> dict:
    prefix_ids = enc.encode(SHARED_PREFIX_TEXT)
    role = cfg["role"]
    per_variant: dict = {}
    summary: dict = {}
    inputs_by_task: dict = {}
    for v in cfg["variants"]:
        arm = A.ARMS[v]
        per_variant[v], summary[v] = {}, {"tasks": {}}
        for t in cfg["tasks"]:
            t0 = time.time()
            recs, inputs = eval_task(model, arm, t, sm, tables, prefix_ids, cfg["n"], cfg["seed0"], cfg["C"], role, ctx)
            per_variant[v][t] = recs
            if recs and all(r["failed"] for r in recs):            # engineering stop: nothing to score, recorded as failed
                ctx.put_json(f"items/{v}_{t}.json", recs)
                ctx.put_json("metrics.json", {"status": "engineering_stop", "variant": v, "task": t, "n": len(recs),
                                              "first_error": recs[0].get("error"), "completed": summary})
                raise EngineeringStop(f"all {len(recs)} items failed for {v} x {t}: {recs[0].get('error')}")
            inputs_by_task.setdefault(t, inputs)
            summary[v]["tasks"][t] = summarize_task(t, recs, cfg["C"])
            ctx.put_json(f"items/{v}_{t}.json", recs)
            ctx.log({"action": "task_done", "variant": v, "task": t, "seconds": round(time.time() - t0, 1), **summary[v]["tasks"][t]})
        t0 = time.time()
        pr = probe_variant(model, arm, sm, tables, prefix_ids, cfg["n_probe_windows"], cfg["seed0"], cfg["probe_C"], role, ctx=ctx)
        summary[v]["probe"] = pr.get("summary")
        summary[v]["linear_probe"] = pr.get("linear_probe", {}).get("mean_accuracy_by_layer")
        ctx.put_json(f"probe_{v}.json", pr)
        ctx.log({"action": "probe_done", "variant": v, "seconds": round(time.time() - t0, 1), "n_ok": pr["n_ok"]})
    for t, inp in inputs_by_task.items():
        ctx.put_json(f"inputs/{t}.json", inp)
    summary["contrasts_vs_V0"] = contrasts(per_variant, cfg["tasks"]) if "V0" in per_variant else {}
    return summary


def build_setup(model, enc, ftok, filler, cfg: dict):
    """Archive with the training-free layout's distances, the sampler (headers as in the pilot) and V1's fixed lambda."""
    from opa.trackb import units
    arch = D.Archive(enc, m=cfg["m"], with_dist=True, kind="A2")
    sm = D.Sampler(arch, filler, ftok, enc, B=cfg["B"], headers=cfg["headers"])
    if cfg["split"] == "test":                                    # all test books, no train/eval pool split
        sm.pools = {"test": list(range(filler.n_books()))}
    att.install_oaa(model, m=cfg["m"], lam0=A.V1_LAMBDA, seed=cfg["seed"])
    units._set_oaa(model, A_std=0.0, lam=A.V1_LAMBDA, g=0.0)
    return arch, sm


DEFAULTS = dict(model="Qwen/Qwen3-4B-Instruct-2507", tokenizer=None, filler="pg19", hf_dir="/cache/hf/pg19", split="dev", seed0=D.DEV_EVAL_SEED0,
                n=16, C=6, probe_C=8, n_probe_windows=16, B=D.B_REF, m=32, headers=True, seed=0, variants=list(VARIANTS), tasks=list(TASKS),
                dtype="bfloat16", prereg_tag=None)


def run(cfg: dict, model, ftok, enc, filler, ctx) -> dict:
    t0 = time.time()
    cfg = {**DEFAULTS, **cfg}
    check_split(cfg["split"], cfg["seed0"], cfg, max(cfg["n"], cfg["n_probe_windows"]))
    cfg["role"] = "eval" if cfg["split"] == "dev" else "test"
    arch, sm = build_setup(model, enc, ftok, filler, cfg)
    ctx.log({"action": "trainfree_start", "split": cfg["split"], "seed0": cfg["seed0"], "n": cfg["n"], "variants": cfg["variants"],
             "tasks": cfg["tasks"], "lambda_V1": A.V1_LAMBDA})
    summary = evaluate_all(model, sm, arch.tables, enc, cfg, ctx)
    contr = summary.pop("contrasts_vs_V0")
    v0p = (summary.get("V0") or {}).get("probe") or {}
    metrics = {"split": cfg["split"], "seed0": cfg["seed0"], "n_per_task": cfg["n"], "C": cfg["C"], "probe_C": cfg["probe_C"],
               "one_over_C_probe": 1.0 / cfg["probe_C"], "lambda_V1": A.V1_LAMBDA, "headers": cfg["headers"], "variants": summary, "contrasts_vs_V0": contr,
               "predictions_to_read": {
                   "V0_own_origin_mass_fraction": v0p.get("own_origin_mass_fraction"),
                   "V0_own_origin_mass_ratio_to_uniform": v0p.get("own_origin_mass_ratio_to_uniform"),
                   "V1_vs_V0_task4": contr.get("V1_vs_V0_confuse"),
                   "note": "read, not tuned: V0 near 1/C own-origin mass and chance on tasks 2-4; V1 better than V0 on task 4"},
               "seconds": round(time.time() - t0, 1)}
    ctx.put_json("metrics.json", metrics)
    flat = {"split": cfg["split"], "n_per_task": cfg["n"], "seconds": metrics["seconds"]}
    for v in cfg["variants"]:
        for t, s in summary[v]["tasks"].items():
            flat[f"{v}_{t}_accuracy" if t != "lm" else f"{v}_lm_ppl"] = s.get("accuracy", s.get("ppl"))
            flat[f"{v}_{t}_n_failed"] = s["n_failed"]
        if summary[v].get("probe"):
            flat[f"{v}_own_origin_mass_fraction"] = summary[v]["probe"]["own_origin_mass_fraction"]
            flat[f"{v}_own_origin_mass_ratio_to_uniform"] = summary[v]["probe"]["own_origin_mass_ratio_to_uniform"]     # the H4 key
    return {"metrics": flat}


def main(args: dict, ctx) -> dict:
    from opa.trackb import units
    cfg = {**DEFAULTS, **args}
    check_split(cfg["split"], cfg["seed0"], cfg, max(cfg["n"], cfg["n_probe_windows"]))      # before anything is loaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftok, enc = D.make_tokenizers(cfg["tokenizer"] or cfg["model"])
    if cfg["split"] == "dev":
        filler = D.make_filler(cfg["filler"], "dev", hf_dir=cfg["hf_dir"])
    else:
        from opa.data import filler as fl
        filler = fl.PG19Filler("test", local_dir=cfg["hf_dir"])
    dtype = getattr(torch, cfg["dtype"])
    model = units.load_model(cfg["model"], device, dtype=dtype)
    return run(cfg, model, ftok, enc, filler, ctx)
