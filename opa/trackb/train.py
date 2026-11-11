"""Mini-LoRA trainer for the H4 / H6 pilot check (PLAN_V3 Section 8 check 4): arms B1 and V3 on dev data, one L4, bf16.

    entrypoint opa.trackb.train:main(args, ctx)       args -> TrainConfig (see below), job specs jobs/p1_trackb_minilora_*.yaml

What a run does
  1. Loads args["model"] (Qwen/Qwen3-4B-Instruct-2507) in bf16 on the GPU, freezes it, wraps q/k/v/o with LoRA (rank 16) and, for V3,
     installs the OAA parameters (opa.trackb.attention.install_oaa; A = 1e-3 * N(0, 1) and g = 0 at the start, so step 0 equals V0 up to the small A; a_init_std = 0 gives exactly V0).
  2. Evaluates the untrained models: V0 (M5, OAA off) and the arm at step 0 (full evaluation size).
  3. Trains with AdamW (LoRA lr 1e-4, new OAA parameters lr 1e-3, 10 warm-up steps then constant, grad clip 1.0, gradient
     accumulation 4 sequences per optimizer step) on a stream of examples (50% LM continuation, 30% source attribution, 20% passkey;
     4-6 co-loaded origins of 512 tokens, at most max_train_seq_len = 4096 tokens), loss only on the target tokens. B1 and V3 get
     the same stream (same seeds), LoRA rank, steps and learning rates. Stops at `max_opt_steps` optimizer steps (B1 gets V3's
     measured count; a non-integer or missing-from-spec placeholder is refused) or, as the stop criterion of the first (V3) run /
     a safety stop of the second, when the wall-clock budget `time_budget_s` (jobs: 1400 s for everything, preflight and
     evaluations included; PLAN check 4: at most 30 GPU-minutes with start-up) minus the reserve for the final full evaluation
     is used up. A memory preflight (longest of the first 50 examples forward+backward + a C=8 probe) runs first.
  4. Every `eval_every` optimizer steps a light evaluation is logged (the learning curve); at the end the full evaluation runs.
  5. Writes through ctx (after the adapter is saved, so "complete" is only written for a finished run; any failure writes
     status "partial" files instead): learning_curve.json ({"status", "curve"}), final_eval.json (per-item correctness and identities
     for every checkpoint, for the paired H4 test), config.json and
     metrics (flat keys named like the prereg keys); the adapter (LoRA + OAA parameters) to gs://<bucket>/opa/models/<run_id>/.

The memory plan, the learning-curve fields and the design choices the protocol leaves open are in the report of work unit W4b and in
the comments of this module. Everything is dev-split only.
"""
from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from opa.trackb import arms as A
from opa.trackb import attention as att
from opa.trackb import data as D
from opa.trackb import metrics as M

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class TrainConfig:
    arm: str = "V3"
    model: str = DEFAULT_MODEL
    tokenizer: str | None = None            # default: same as model ("hash" for the tiny test model)
    filler: str = "pg19"                    # "pg19" (validation split) | "synthetic" (tests)
    hf_dir: str | None = "/cache/hf/pg19"
    seed: int = 0
    lora_r: int = 16
    lora_alpha: float = 32.0
    lr_lora: float = 1e-4
    lr_new: float = 1e-3
    warmup_steps: int = 10
    grad_clip: float = 1.0
    grad_accum: int = 4
    B: int = D.B_REF
    m: int = 32
    headers: bool = True               # PLAN_V3 ruling: B1 and V3 see the same inputs, both with "[label]" origin headers
    a_init_std: float = 1e-3               # protocol 5.4: "A small" (0 gives exactly V0 at step 0; a test covers both)
    struct: bool = False
    train_origins: tuple = (4, 6)
    max_train_seq_len: int = 4096
    mix: dict = field(default_factory=lambda: {"lm": 0.5, "attrib": 0.3, "passkey": 0.2})
    time_budget_s: float = 1400.0
    max_opt_steps: int | None = None        # None: stop by time only (V3, the reference run); an integer is required for B1
    eval_every: int = 40
    log_every: int = 5
    grad_checkpointing: bool = True
    save_adapter: bool = True
    eval_full: dict = field(default_factory=lambda: {"attrib": 32, "passkey": 16, "lm": 16, "std": 16, "probe": 3})
    eval_light: dict = field(default_factory=lambda: {"attrib": 16, "passkey": 8, "lm": 8, "std": 8, "probe": 1})
    eval_C: int = 6
    probe_C: int = 8
    final_eval_reserve_factor: float = 1.3
    # ---- learning-curve mode (PLAN_V3 6a Step 1) and checkpoints
    eval_at_steps: list = field(default_factory=list)       # e.g. [50, 100, 200, 400]: evaluate on the fixed monitoring set at these steps
    save_checkpoint_at: list = field(default_factory=list)  # e.g. [400]: LoRA + OAA + optimizer state to models/<run_id>/step<N>/checkpoint.pt
    resume_from: str | None = None                          # checkpoint path or gs:// uri to continue from (Step 2)
    monitor: dict = field(default_factory=lambda: {"attrib": 64, "passkey": 32, "probe": 3})   # n_monitor = 64 attribution items
    monitor_seed0: int = D.MONITOR_SEED0
    # a resumed run need not re-evaluate the state it starts from (the previous chunk already did): saves ~240 s per 200-step chunk
    skip_resume_eval: bool = False
    # no_eval: every evaluation (the step-0 / resume-start one and the final one) runs with zero items, so a run only trains and saves
    # checkpoints (B1 in parallel with V3: its evaluations are not needed for the stopping rule). Free for the resume check.
    no_eval: bool = False
    # eval_split "test": EVALUATION ONLY on the PG-19 test books and test seeds (>= 100000). Refused unless the prereg-v2 tag exists
    # (prereg/PREREG_TAG.json or prereg_tag = {tag, commit} in the spec), unless the run resumes a checkpoint at max_opt_steps (zero training steps),
    # and the training preflight, which would read test books through the training stream, is skipped. Free for the resume check.
    eval_split: str = "dev"
    prereg_tag: dict | None = None
    # expected_resume_sha256: refuse to run unless the loaded checkpoint's SHA-256 equals this (pins the exact V3 / B1 checkpoint). Free for the resume check.
    expected_resume_sha256: str | None = None
    assumed_eval_seconds: float = 240.0          # stands in for the measured full-evaluation time (final-evaluation reserve) when skipped
    preflight_examples: int = 50            # longest of the first N stream examples is run forward+backward before training
    mem_limit_gib: float = 21.0             # abort when the preflight peak exceeds this (L4 has 24 GiB)

    @classmethod
    def from_args(cls, args: dict) -> "TrainConfig":
        known = {k for k in cls.__dataclass_fields__}
        bad = set(args) - known
        if bad:
            raise ValueError(f"unknown train args: {sorted(bad)}")
        cfg = cls(**args)
        if "max_opt_steps" in args and (isinstance(cfg.max_opt_steps, bool) or not isinstance(cfg.max_opt_steps, int)
                                        or cfg.max_opt_steps < 1):
            raise ValueError(f"max_opt_steps must be a positive integer, got {cfg.max_opt_steps!r} "
                             "(an unfilled placeholder such as V3_OPT_STEPS must be replaced by V3's measured opt_steps)")
        cfg.eval_at_steps = sorted(int(x) for x in cfg.eval_at_steps)
        cfg.save_checkpoint_at = sorted(int(x) for x in cfg.save_checkpoint_at)
        if cfg.eval_at_steps and cfg.max_opt_steps is None:
            raise ValueError("eval_at_steps needs max_opt_steps")
        if cfg.max_opt_steps is not None and any(x > cfg.max_opt_steps for x in cfg.eval_at_steps + cfg.save_checkpoint_at):
            raise ValueError("eval_at_steps / save_checkpoint_at beyond max_opt_steps")
        cfg.train_origins = tuple(cfg.train_origins)
        if cfg.arm not in ("B1", "V3"):
            raise ValueError("arm must be B1 or V3")
        if cfg.eval_split not in ("dev", "test"):
            raise ValueError(f"eval_split must be 'dev' or 'test', got {cfg.eval_split!r}")
        n_eval = max(int(cfg.monitor.get("attrib", 0)), int(cfg.monitor.get("passkey", 0)), int(cfg.monitor.get("probe", 0)), 1)
        from opa.trackb import trainfree
        if cfg.eval_at_steps or cfg.eval_split == "test":      # the seeds of the evaluation set must belong to the split (test: tag required, >= 100000)
            trainfree.check_split(cfg.eval_split, cfg.monitor_seed0, {"prereg_tag": cfg.prereg_tag}, n_eval)
        if cfg.eval_split == "test" and not (cfg.resume_from and cfg.eval_at_steps and cfg.max_opt_steps is not None
                                             and cfg.eval_at_steps == [cfg.max_opt_steps] and not cfg.save_checkpoint_at):
            raise ValueError("eval_split='test' is evaluation only: resume_from a checkpoint at max_opt_steps, eval_at_steps == [max_opt_steps], "
                             "no save_checkpoint_at")
        if cfg.arm == "B1" and cfg.max_opt_steps is None:
            raise ValueError("B1 requires an integer max_opt_steps (V3's measured opt_steps); a missing key would run it time-only")
        return cfg

    def sizes(self, full: bool) -> M.EvalSizes:
        role = "test" if self.eval_split == "test" else "eval"
        if self.no_eval:
            return M.EvalSizes(attrib=0, passkey=0, lm=0, std=0, probe=0, C=self.eval_C, probe_C=self.probe_C, seed0=self.monitor_seed0)
        if self.eval_at_steps:                       # learning-curve mode: always the fixed monitoring set
            return M.EvalSizes(**{"lm": 0, "std": 0, **self.monitor}, C=self.eval_C, probe_C=self.probe_C, seed0=self.monitor_seed0,
                               role=role)
        return M.EvalSizes(**(self.eval_full if full else self.eval_light), C=self.eval_C, probe_C=self.probe_C)


# ============================================================================ setup


def prepare_model(model, cfg: TrainConfig):
    """LoRA on every attention layer; V3 additionally the OAA parameters. Returns (lora_params, new_params)."""
    lora = A.apply_lora(model, cfg.lora_r, cfg.lora_alpha, seed=cfg.seed)
    new = []
    if cfg.arm == "V3":
        att.install_oaa(model, m=cfg.m, lam0=0.0, a_init_std=cfg.a_init_std, seed=cfg.seed)
        for _, p in att.oaa_parameters(model):
            new += [p.W_r.weight, p.A, p.U.weight, p.g]
            p.lam.requires_grad_(cfg.struct)
            if cfg.struct:
                new.append(p.lam)
    else:
        model.set_attn_implementation("sdpa")
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return lora, new


def make_stream(cfg: TrainConfig, sampler: D.Sampler):
    """Example i of the training stream: kind from the mix, C origins (shrunk until the sequence fits max_train_seq_len)."""
    kinds = list(cfg.mix)
    p = np.array([cfg.mix[k] for k in kinds], dtype=float)
    p /= p.sum()

    def example(i: int) -> D.Example:
        rng = np.random.default_rng([cfg.seed, 77, i])
        kind = kinds[int(rng.choice(len(kinds), p=p))]
        C = int(rng.integers(cfg.train_origins[0], cfg.train_origins[1] + 1))
        while True:
            ex = sampler.make(kind, i, C, role="train")
            if ex.T <= cfg.max_train_seq_len or C <= 2:
                return ex
            C -= 1
    return example


def _lr_factor(opt_step: int, warmup: int) -> float:
    return min(1.0, (opt_step + 1) / max(1, warmup))


# ============================================================================ the run


def _peak_gib(device) -> float:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0


def preflight(cfg, model, arm, arch, sampler, example, params, device, ctx) -> dict:
    """(Test-split evaluation: skipped - it would run a training step on test books and there is no training.)
    One forward+backward on the longest of the first N stream examples and a C=probe_C probe forward; logs the peak memory
    and aborts above cfg.mem_limit_gib (only meaningful on a GPU)."""
    if cfg.eval_split == "test":
        info = {"skipped": "test-split evaluation (zero training steps; no test book enters a training example)", "longest_T": 0,
                "kind": None, "probe_T": 0, "peak_gib_train": 0.0, "peak_gib": 0.0, "limit_gib": cfg.mem_limit_gib}
        ctx.log({"action": "preflight", **info})
        return info
    cands = [example(i) for i in range(cfg.preflight_examples)]
    ex = max(cands, key=lambda e: e.T)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    A.forward_example(model, ex, arm, arch.tables, backward=True)
    for p in params:
        p.grad = None
    peak_train = _peak_gib(device)
    model.eval()
    pw = sampler.make("probe", D.DEV_EVAL_SEED0, cfg.probe_C, role="eval")
    M.probe_window(model, pw, arm, arch, arch.tables)
    peak = _peak_gib(device)
    info = {"longest_T": ex.T, "kind": ex.kind, "probe_T": pw.T, "peak_gib_train": round(peak_train, 2),
            "peak_gib": round(peak, 2), "limit_gib": cfg.mem_limit_gib}
    ctx.log({"action": "preflight", **info})
    if peak > cfg.mem_limit_gib:
        raise RuntimeError(f"memory preflight failed: peak {peak:.1f} GiB > {cfg.mem_limit_gib} GiB at T={ex.T} "
                           f"(train) / {pw.T} (probe). Reduce max_train_seq_len or train_origins.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    return info


def projection(curve: list, threshold: float = 0.02, target: float = 0.90) -> dict:
    """Stopping-rule projection from the monitoring evaluations (tags step<N>): own-origin mass fraction per eval, its gain between
    evaluations, whether the gain from step 200 to 400 is below `threshold` ("plateau by 400" - the PLAN_V3 6a Step 2 rule would then
    stop right after step 400), and a linear extrapolation of the steps needed to reach `target` from the last two points."""
    pts = [r for r in curve if str(r.get("tag", "")).startswith("step") and r.get("probe")]
    pts = sorted(pts, key=lambda r: r["opt_step"])
    pts = [{"step": r["opt_step"], "own_origin_mass_fraction": r["probe"]["own_origin_mass_fraction"],
            "own_origin_mass_ratio_to_uniform": r["probe"]["own_origin_mass_ratio_to_uniform"],
            "attrib_em": r.get("attrib_em"), "passkey_em": r.get("passkey_em"), "train_loss": r.get("train_loss")} for r in pts]
    out: dict = {"points": pts, "threshold": threshold, "target_fraction": target}
    by = {p["step"]: p for p in pts}
    if 200 in by and 400 in by:
        gain = by[400]["own_origin_mass_fraction"] - by[200]["own_origin_mass_fraction"]
        out["fraction_gain_200_to_400"] = gain
        out["plateau_by_400"] = bool(gain < threshold)
        out["attrib_gain_200_to_400"] = None if by[400]["attrib_em"] is None else by[400]["attrib_em"] - by[200]["attrib_em"]
    if len(pts) >= 2:
        a, b = pts[-2], pts[-1]
        slope = (b["own_origin_mass_fraction"] - a["own_origin_mass_fraction"]) / max(b["step"] - a["step"], 1)
        out["last_interval"] = [a["step"], b["step"]]
        out["fraction_gain_per_200_steps_last_interval"] = slope * 200
        if slope > 0:
            out["linear_steps_to_target"] = int(math.ceil(round(b["step"] + max(target - b["own_origin_mass_fraction"], 0.0) / slope, 6)))
        else:
            out["linear_steps_to_target"] = None
    return out


def checkpoint_bytes(model, opt, cfg, opt_step: int, micro: int, nan_steps: int, stop_reason: str = "planned") -> bytes:
    """LoRA + OAA + optimizer state. A checkpoint written because the run ended before the planned step carries
    stop_reason (time_budget / crash: ...) and equal_steps_met False; a planned one has stop_reason "planned"."""
    planned = max(cfg.save_checkpoint_at) if cfg.save_checkpoint_at else None
    buf = io.BytesIO()
    torch.save({"lora": A.lora_state(model), "oaa": A.oaa_state(model), "opt": opt.state_dict(), "opt_step": opt_step,
                "micro": micro, "nan_steps": nan_steps, "config": asdict(cfg), "stop_reason": stop_reason,
                "planned_steps": planned, "equal_steps_met": None if planned is None else bool(opt_step >= planned)}, buf)
    return buf.getvalue()


# configuration fields that may differ between a run and the checkpoint it resumes (step counts, evaluation, checkpointing,
# time budget, logging); everything else is the training configuration and must be identical (PLAN_V3 6a Step 2 rule 1)
RESUME_FREE_FIELDS = {"max_opt_steps", "eval_at_steps", "eval_every", "eval_full", "eval_light", "eval_C", "probe_C", "monitor",
                      "monitor_seed0", "save_checkpoint_at", "resume_from", "time_budget_s", "log_every", "final_eval_reserve_factor",
                      "preflight_examples", "mem_limit_gib", "save_adapter", "skip_resume_eval", "assumed_eval_seconds", "no_eval", "eval_split", "prereg_tag", "expected_resume_sha256"}


def _norm(v):
    return list(v) if isinstance(v, (tuple, list)) else v


def resume_config_diff(cfg: TrainConfig, saved: dict) -> dict:
    """{field: (checkpoint value, current value)} for the training-configuration fields that differ."""
    now = asdict(cfg)
    return {k: (_norm(saved.get(k)), _norm(now[k])) for k in now
            if k not in RESUME_FREE_FIELDS and _norm(saved.get(k)) != _norm(now[k])}


def resolve_bucket(uri: str, ctx=None) -> str:
    """Specs write gs://{bucket}/... and nothing in the launcher substitutes it for Track B: take the bucket from the job spec
    (ctx.spec["bucket"]) and fall back to the project configuration. Returns the uri unchanged when it has no {bucket}."""
    if not uri or "{bucket}" not in str(uri):
        return uri
    from opa import gcs
    bucket = (getattr(ctx, "spec", None) or {}).get("bucket") or gcs.bucket_name()
    return str(uri).replace("{bucket}", gcs.normalize_bucket(bucket))


def load_checkpoint(src: str) -> dict:
    """The checkpoint dict, with "_sha256" (SHA-256 of the bytes that were read) added."""
    if str(src).startswith("gs://"):
        from opa import gcs
        data = gcs.get_bytes(src)
    else:
        with open(src, "rb") as f:
            data = f.read()
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    ck["_sha256"] = hashlib.sha256(data).hexdigest()
    return ck


def run(cfg: TrainConfig, model, ftok, enc, filler, ctx, clock=time.time) -> dict:
    t_start = clock()
    device = next(model.parameters()).device
    arch = D.Archive(enc, m=cfg.m, seed=cfg.seed)
    sampler = D.Sampler(arch, filler, ftok, enc, B=cfg.B, headers=cfg.headers)
    if cfg.eval_split == "test":                    # all PG-19 test books, only for evaluation windows (role "test")
        sampler.pools = {"test": list(range(filler.n_books()))}
    arm = A.ARMS[cfg.arm]
    torch.manual_seed(cfg.seed)
    lora, new = prepare_model(model, cfg)
    n_lora, n_new = A.count_params(lora), A.count_params(new)
    ctx.log({"action": "train_setup", "arm": cfg.arm, "lora_params": n_lora, "new_params": n_new,
             "trainable_total": n_lora + n_new, "model": cfg.model, "device": str(device), "headers": cfg.headers})
    opt = torch.optim.AdamW([{"params": lora, "lr": cfg.lr_lora, "base_lr": cfg.lr_lora},
                             *([{"params": new, "lr": cfg.lr_new, "base_lr": cfg.lr_new}] if new else [])],
                            weight_decay=0.0, betas=(0.9, 0.999))
    stream = make_stream(cfg, sampler)
    memo: dict = {}

    def example(i: int) -> D.Example:               # the preflight samples are reused by the first training steps
        if i not in memo:
            memo[i] = stream(i)
        return memo[i]

    curve: list = []
    all_items: dict = {}
    state = {"opt_step": 0, "micro": 0, "nan": 0}
    written: set = set()

    def put(name, obj):
        ctx.put_json(name, obj)
        written.add(name)

    def evaluate(tag: str, a: A.Arm, full: bool, opt_step: int, micro: int, train_loss) -> dict:
        t = clock()
        res = M.eval_tasks(model, a, arch, sampler, cfg.sizes(full), tables=arch.tables)
        all_items[tag] = {"arm": a.name, "per_item": res["per_item"], "meta": res["per_item_meta"]}
        rec = {"tag": tag, "arm": a.name, "opt_step": opt_step, "micro_steps": micro,
               "train_seconds": round(clock() - t_start, 1), "eval_seconds": round(clock() - t, 1),
               "train_loss": train_loss, "full": full,
               **{k: v for k, v in res.items() if k not in ("per_item", "per_item_meta")}, **M.learn_state(model)}
        ctx.log({"action": "eval", **{k: v for k, v in rec.items() if k != "probe"},
                 "own_origin_mass_fraction": rec.get("probe", {}).get("own_origin_mass_fraction")})
        return rec

    saved: set = set()

    ck_hashes: dict = {}                                   # SHA-256 of every checkpoint this run wrote (DEV-TRAIN-PIN)
    resume_sha_box = [None]                                # SHA-256 of the checkpoint this run resumed from (for the partial config)

    def save_ckpt(step: int, micro_: int, nan_: int, reason: str) -> None:
        data = checkpoint_bytes(model, opt, cfg, step, micro_, nan_, reason)
        ck_hashes[f"step{step}"] = hashlib.sha256(data).hexdigest()
        _put_model_bytes(ctx, f"step{step}/checkpoint.pt", data)
        saved.add(step)
        ctx.log({"action": "checkpoint_saved", "opt_step": step, "path": f"step{step}/checkpoint.pt", "stop_reason": reason,
                 "sha256": ck_hashes[f"step{step}"], "bytes": len(data)})

    base = v0 = None
    try:
        info = preflight(cfg, model, arm, arch, sampler, example, lora + new, device, ctx)
        # ---- step 0: V0 (M5, nothing trained) and the arm at initialisation (V3 at step 0 is V0 up to the small A init)
        monitor = bool(cfg.eval_at_steps)
        resume_uri = resolve_bucket(cfg.resume_from, ctx) if cfg.resume_from else None
        resume = load_checkpoint(resume_uri) if resume_uri else None
        resume_sha = None if resume is None else resume["_sha256"]
        resume_sha_box[0] = resume_sha
        if cfg.expected_resume_sha256 and resume_sha != cfg.expected_resume_sha256:
            raise ValueError(f"resume refused: checkpoint {resume_uri} has sha256 {resume_sha}, the spec expects {cfg.expected_resume_sha256}")
        if resume is not None:
            diff = resume_config_diff(cfg, resume.get("config", {}))
            ctx.log({"action": "resume_config_check", "differs": {k: list(v) for k, v in diff.items()}, "ok": not diff})
            if diff:
                raise ValueError("resume refused: the training configuration differs from the checkpoint's: "
                                 + "; ".join(f"{k}: checkpoint {a!r} vs now {b!r}" for k, (a, b) in diff.items()))
            A.load_lora_state(model, resume["lora"])
            if cfg.arm == "V3":
                A.load_oaa_state(model, resume["oaa"])
            opt.load_state_dict(resume["opt"])
            ctx.log({"action": "resumed", "from": resume_uri, "opt_step": resume["opt_step"], "micro": resume["micro"],
                     "resumed_from_sha256": resume_sha})
        start_step = 0 if resume is None else int(resume["opt_step"])
        start_micro = 0 if resume is None else int(resume["micro"])
        start_nan = 0 if resume is None else int(resume.get("nan_steps", 0))
        if cfg.eval_split == "test" and start_step != cfg.max_opt_steps:
            raise ValueError(f"eval_split=test is evaluation only: the checkpoint is at step {start_step}, max_opt_steps is {cfg.max_opt_steps} "
                             "(a test run must not train)")
        state.update(opt_step=start_step, micro=start_micro, nan=start_nan)
        base_tag = "step0" if resume is None else "resume_start"
        v0 = (evaluate("v0_untrained", A.ARMS["V0"], True, 0, 0, None)
              if (cfg.arm == "V3" and resume is None and not monitor) else None)
        if resume is not None and cfg.skip_resume_eval:
            base, t_full_eval = None, float(cfg.assumed_eval_seconds)
            ctx.log({"action": "resume_eval_skipped", "opt_step": start_step})
        else:
            base = evaluate(base_tag, arm, True, start_step, start_micro, None)
            t_full_eval = base["eval_seconds"]
            curve.append(dict(base))
        reserve = cfg.final_eval_reserve_factor * t_full_eval + 20.0

        # ---- training: exactly cfg.max_opt_steps optimizer steps unless the wall-clock budget stops it first
        model.train()
        opt_step, micro, nan_steps, loss_acc, loss_n, ema = start_step, start_micro, start_nan, 0.0, 0, None
        last_loss_acc, last_loss_n = 0.0, 0
        stop_reason = "max_opt_steps"
        all_params = lora + new
        step_limit = cfg.max_opt_steps if cfg.max_opt_steps is not None else 10 ** 9
        while opt_step < step_limit:
            if clock() - t_start > cfg.time_budget_s - reserve:
                stop_reason = "time_budget"
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * _lr_factor(opt_step, cfg.warmup_steps)
            step_loss = 0.0
            for _ in range(cfg.grad_accum):
                ex = example(micro)
                micro += 1
                loss, _ = A.forward_example(model, ex, arm, arch.tables, backward=True, scale=1.0 / cfg.grad_accum)
                step_loss += float(loss) / cfg.grad_accum
            gn = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            if not math.isfinite(float(gn)) or not math.isfinite(step_loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                ctx.log({"action": "nan_step", "opt_step": opt_step, "grad_norm": float(gn)})
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)
                loss_acc += step_loss
                loss_n += 1
                last_loss_acc += step_loss
                last_loss_n += 1
                ema = step_loss if ema is None else 0.9 * ema + 0.1 * step_loss
            opt_step += 1
            for k in list(memo):                         # keep memory flat: drop consumed examples
                if k < micro:
                    del memo[k]
            state.update(opt_step=opt_step, micro=micro, nan=nan_steps)
            if opt_step % cfg.log_every == 0:
                ctx.log({"action": "train_step", "opt_step": opt_step, "micro": micro, "loss": step_loss, "loss_ema": ema,
                         "grad_norm": float(gn), "nan_steps": nan_steps, "seconds": round(clock() - t_start, 1),
                         "peak_mem_gib": round(_peak_gib(device), 2), **M.learn_state(model)})
            if opt_step in cfg.save_checkpoint_at:               # BEFORE the evaluation: a crash in the eval must not lose it
                save_ckpt(opt_step, micro, nan_steps, "planned")
            if monitor and opt_step in cfg.eval_at_steps:
                tl = last_loss_acc / max(last_loss_n, 1) if last_loss_n else None
                last_loss_acc, last_loss_n = 0.0, 0
                model.eval()
                r = evaluate(f"step{opt_step}", arm, True, opt_step, micro, tl)
                curve.append(r | {"nan_steps": nan_steps})
                model.train()
            if (not monitor and cfg.eval_every and opt_step % cfg.eval_every == 0 and opt_step < step_limit
                    and clock() - t_start < cfg.time_budget_s - reserve - 0.5 * t_full_eval):
                tl = last_loss_acc / max(last_loss_n, 1)
                last_loss_acc, last_loss_n = 0.0, 0
                model.eval()
                r = evaluate(f"step{opt_step}", arm, False, opt_step, micro, tl)
                curve.append(r | {"nan_steps": nan_steps})
                model.train()
        if cfg.save_checkpoint_at and stop_reason != "max_opt_steps" and opt_step > 0 and opt_step not in saved:
            save_ckpt(opt_step, micro, nan_steps, stop_reason)   # the run ended before the planned step: keep what exists, flagged
        planned_missed = bool(cfg.save_checkpoint_at) and opt_step < max(cfg.save_checkpoint_at)
        run_status = "incomplete_steps" if planned_missed else "complete"
        model.eval()
        if monitor and curve and curve[-1]["tag"] == f"step{opt_step}":      # the last monitoring evaluation is the final one
            final = curve[-1]
            all_items["final"] = all_items[f"step{opt_step}"]
        else:
            final = evaluate("final", arm, True, opt_step, micro, loss_acc / max(loss_n, 1) if loss_n else None)
            curve.append(final | {"nan_steps": nan_steps})

        # ---- outputs: adapter first, then the files; "complete" is written only after the adapter is saved
        out = {
            "status": run_status, "config": asdict(cfg), "headers_on": cfg.headers, "arm": cfg.arm, "opt_steps": opt_step,
            "max_opt_steps_requested": cfg.max_opt_steps,
            "equal_steps_met": None if cfg.max_opt_steps is None else opt_step == cfg.max_opt_steps,   # None: reference run
            "micro_steps": micro, "nan_steps": nan_steps, "stop_reason": stop_reason, "lora_params": n_lora,
            "new_params": n_new, "total_seconds": round(clock() - t_start, 1), "t_full_eval_seconds": t_full_eval,
            "train_loss_mean": loss_acc / max(loss_n, 1) if loss_n else None, "preflight": info,
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None,
        }
        out["checkpoint_sha256"] = dict(ck_hashes)
        out["resume_from_resolved"], out["resumed_from_sha256"] = resume_uri, resume_sha
        if cfg.save_adapter:
            buf = io.BytesIO()
            torch.save({"lora": A.lora_state(model), "oaa": A.oaa_state(model), "config": asdict(cfg), "opt_steps": opt_step}, buf)
            out["adapter_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
            _put_model_bytes(ctx, "adapter.pt", buf.getvalue())
        put("final_eval.json", {"status": run_status, "final": final, "step0": base, "v0": v0,
                                "per_item_final": all_items["final"]["per_item"],
                                "per_item_step0": None if base is None else all_items[base_tag]["per_item"],
                                "per_item_v0": None if v0 is None else all_items["v0_untrained"]["per_item"],
                                "per_item": all_items})
        lc = {"status": run_status, "curve": curve}
        if monitor:
            lc["projection"] = projection(curve)
            lc["monitor"] = {"n_attrib": cfg.monitor.get("attrib"), "n_passkey": cfg.monitor.get("passkey"),
                             "n_probe_windows": cfg.monitor.get("probe"), "seed0": cfg.monitor_seed0, "eval_at_steps": cfg.eval_at_steps}
        put("learning_curve.json", lc)
        put("config.json", out)
        result = flat_metrics(cfg, out, base or {}, final, v0)
    except BaseException as e:                           # partial outputs, then re-raise
        try:
            err = f"{type(e).__name__}: {e}"
            if cfg.save_checkpoint_at and state["opt_step"] > 0 and state["opt_step"] not in saved:
                try:                                     # crash after the last completed step: keep that state, flagged
                    save_ckpt(state["opt_step"], state["micro"], state["nan"], f"crash: {err}")
                except Exception:
                    pass
            if "learning_curve.json" not in written:
                put("learning_curve.json", {"status": "partial", "error": err, "curve": curve})
            if "final_eval.json" not in written:
                put("final_eval.json", {"status": "partial", "error": err, "per_item": all_items})
            if "config.json" not in written:
                put("config.json", {"status": "partial", "error": err, "config": asdict(cfg), "headers_on": cfg.headers,
                                    "arm": cfg.arm, "opt_steps": state["opt_step"], "micro_steps": state["micro"],
                                    "nan_steps": state["nan"], "checkpoint_sha256": dict(ck_hashes),
                                    "resumed_from_sha256": resume_sha_box[0]})
        except Exception:
            pass
        raise
    return {"metrics": result}


def _put_model_bytes(ctx, name: str, data: bytes) -> None:
    """Adapters go to gs://<bucket>/opa/models/<run_id>/ (CLAUDE.md layout); the context only knows runs/<run_id>/, so derive it."""
    prefix = getattr(ctx, "prefix", None)
    if prefix and "/runs/" in prefix:
        from opa import gcs
        info = gcs.put_bytes(prefix.replace("/runs/", "/models/") + "/" + name, data)
        getattr(ctx, "outputs", []).append(info)
    else:
        ctx.put_bytes(f"models/{ctx.run_id}/{name}", data)


def flat_metrics(cfg: TrainConfig, out: dict, base: dict, final: dict, v0: dict | None) -> dict:
    """Flat keys: measurements named like the prereg keys (pilot, dev), plus training facts."""
    a = cfg.arm
    m = {"arm": a, "status": out["status"], "opt_steps": out["opt_steps"],
         "max_opt_steps_requested": out["max_opt_steps_requested"], "equal_steps_met": out["equal_steps_met"],
         "preflight_peak_gib": out["preflight"]["peak_gib"], "micro_steps": out["micro_steps"], "nan_steps": out["nan_steps"],
         "stop_reason": out["stop_reason"], "total_seconds": out["total_seconds"], "peak_mem_gib": out["peak_mem_gib"],
         "trainable_params": out["lora_params"] + out["new_params"], "train_loss_mean": out["train_loss_mean"],
         f"{a}_source_attribution": final.get("attrib_em"), f"{a}_source_attribution_rank": final.get("attrib_rank_acc"),
         f"{a}_passkey_em": final.get("passkey_em"), f"{a}_lm_ppl": final.get("lm_ppl"), f"{a}_std_ppl": final.get("std_ppl"),
         f"{a}_step0_std_ppl": base.get("std_ppl"), f"{a}_step0_source_attribution": base.get("attrib_em"),
         f"{a}_own_origin_mass_fraction": (final.get("probe") or {}).get("own_origin_mass_fraction"),
         f"{a}_own_origin_mass_ratio_to_uniform": (final.get("probe") or {}).get("own_origin_mass_ratio_to_uniform"),
         f"{a}_ghost_ratio": (final.get("probe") or {}).get("ghost_ratio"), f"{a}_router_entropy": (final.get("probe") or {}).get("router_entropy"),
         "attrib_chance": final.get("attrib_chance")}
    if v0 is not None:
        m["V0_own_origin_mass_ratio_to_uniform"] = v0["probe"]["own_origin_mass_ratio_to_uniform"]
        m["V0_own_origin_mass_fraction"] = v0["probe"]["own_origin_mass_fraction"]
        m["V0_ghost_ratio"] = v0["probe"]["ghost_ratio"]
        m["V0_source_attribution"] = v0.get("attrib_em")
    else:
        if base.get("probe"):
            m["B1_step0_own_origin_mass_ratio_to_uniform"] = base["probe"]["own_origin_mass_ratio_to_uniform"]
    for k, h in (out.get("checkpoint_sha256") or {}).items():
        m[f"checkpoint_sha256_{k}"] = h
    if out.get("checkpoint_sha256"):
        m["final_checkpoint_sha256"] = list(out["checkpoint_sha256"].values())[-1]
    if out.get("adapter_sha256"):
        m["adapter_sha256"] = out["adapter_sha256"]
    if out.get("resumed_from_sha256"):
        m["resumed_from_sha256"] = out["resumed_from_sha256"]
    return m


def main(args: dict, ctx) -> dict:
    from opa.trackb import units
    cfg = TrainConfig.from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx.log({"action": "train_start", "arm": cfg.arm, "model": cfg.model, "device": str(device)})
    ftok, enc = D.make_tokenizers(cfg.tokenizer or cfg.model)
    if cfg.eval_split == "test":
        from opa.data import filler as fl
        filler = fl.PG19Filler("test", local_dir=cfg.hf_dir)
    else:
        filler = D.make_filler(cfg.filler, "dev", hf_dir=cfg.hf_dir)
    model = units.load_model(cfg.model, device, dtype=torch.bfloat16)
    return run(cfg, model, ftok, enc, filler, ctx)
