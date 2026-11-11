"""Track B mini-LoRA pilot code: data, LoRA, attention probe, trainer (B1 and V3) on a tiny random Qwen3 (CPU, no download)."""
import copy
import json

import numpy as np
import pytest
import torch

from opa.trackb import arms as A
from opa.trackb import attention as att
from opa.trackb import data as D
from opa.trackb import metrics as M
from opa.trackb import remedies as R
from opa.trackb import train as T
from opa.trackb import units

VOCAB = 512
B = 24


@pytest.fixture(scope="module")
def tk():
    return D.make_tokenizers("hash", VOCAB)


@pytest.fixture(scope="module")
def filler():
    return D.make_filler("synthetic", n_books=24, words_per_book=6000)


@pytest.fixture(scope="module")
def archive(tk):
    return D.Archive(tk[1], m=8)


@pytest.fixture(scope="module")
def sampler(tk, filler, archive):
    return D.Sampler(archive, filler, tk[0], tk[1], B=B)


def _cfg(arm, **kw):
    base = dict(arm=arm, model="tiny", tokenizer="hash", filler="synthetic", hf_dir=None, B=B, m=8, seed=3, grad_accum=2,
                max_opt_steps=3, warmup_steps=2, eval_every=2, log_every=1, train_origins=(3, 4), eval_C=3, probe_C=4,
                eval_full={"attrib": 4, "passkey": 3, "lm": 3, "std": 2, "probe": 1},
                eval_light={"attrib": 2, "passkey": 2, "lm": 2, "std": 1, "probe": 1}, time_budget_s=1e9)
    base.update(kw)
    if base.get("max_opt_steps", 0) is None:
        del base["max_opt_steps"]                       # reference run: no step count
    return T.TrainConfig.from_args(base)


class Ctx:
    run_id = "B-minilora-test"

    def __init__(self):
        self.events, self.put, self.bytes = [], {}, {}

    def log(self, e):
        self.events.append(e)

    def put_json(self, rel, obj):
        assert rel not in self.put, f"second write to {rel}"
        self.put[rel] = json.loads(json.dumps(obj))
        return {"uri": rel}

    def put_bytes(self, rel, data, content_type=""):
        assert rel not in self.bytes
        self.bytes[rel] = data
        return {"uri": rel}


def _run(cfg, seed=0, clock=None):
    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    model = units.tiny_model(seed=seed, vocab=VOCAB)
    ctx = Ctx()
    kw = {} if clock is None else {"clock": clock}
    out = T.run(cfg, model, ftok, enc, filler, ctx, **kw)
    return out, ctx, model


# ---------------------------------------------------------------- data


def test_label_table_is_64_distinct_single_tokens(tk):
    lt = D.label_table(tk[1])
    assert len(lt) == 64 and len({t for _, t in lt}) == 64
    assert all(len(tk[1].encode(" " + w)) == 1 for w, _ in lt)


def test_archive_tree_and_addresses(archive):
    assert archive.thread(1) == 0 and archive.thread(9) == 1 and archive.thread(64) == 7
    assert archive.parent(1) == 0 and archive.parent(2) == 1 and archive.parent(9) == 0
    assert archive.tables.vec.shape == (64, 8) and float(archive.tables.vec.norm(dim=1).min()) > 0
    assert float(archive.tables.dist.abs().max()) == 0.0                        # V3 has no penalty: no distances computed


def test_examples_targets_and_determinism(sampler, tk):
    enc = tk[1]
    for kind in ("attrib", "passkey", "lm", "probe"):
        a, b = sampler.make(kind, 5, 4), sampler.make(kind, 5, 4)
        assert torch.equal(a.ids, b.ids) and a.window == b.window and a.ans_start == b.ans_start
        assert len(a.window) == 4 and a.ids.numel() == a.origin_tok.numel() == sum(s.length for s in a.segments)
        assert not torch.equal(a.ids, sampler.make(kind, 6, 4).ids)
    ex = sampler.make("attrib", 1, 4)
    tgt = ex.meta["target_origin"]
    assert ex.ans_len == 1 and int(ex.targets[0]) == sampler.arch.label_token(tgt)
    assert ex.options[ex.gold_option] == int(ex.targets[0]) and len(ex.options) == 4
    assert ex.segments[-1].origin == att.NULL_ORIGIN and ex.origin_tok[-1] == -1
    ex = sampler.make("passkey", 2, 3)
    assert ex.targets.tolist() == enc.encode(" " + ex.meta["code"])
    ex = sampler.make("lm", 3, 3)
    assert ex.ans_len == ex.T - ex.ans_start and 0 < ex.ans_len <= D.CONT_TOKENS
    pr = sampler.make("probe", 4, 4)
    assert pr.ans_len == 0 and pr.segments[-1].origin != att.NULL_ORIGIN


def test_train_and_eval_texts_do_not_overlap(sampler):
    assert not set(sampler.pools["train"]) & set(sampler.pools["eval"])
    t = sampler.make("lm", 11, 3, role="train")
    e = sampler.make("lm", 11, 3, role="eval")
    assert not torch.equal(t.ids, e.ids)


def test_positions(sampler):
    ex = sampler.make("attrib", 7, 3)
    m5, ct = D.positions(ex, "M5"), D.positions(ex, "contig")
    assert ct.tolist() == list(range(ex.T))
    s0 = ex.segments[0].length
    assert m5[:s0].tolist() == list(range(s0)) and int(m5[s0]) == 0                  # origins restart at 0
    n_win = ex.T - ex.segments[-1].length
    assert int(m5[n_win]) == int(m5[:n_win].max()) + 1                                # tail starts right after the largest key position


def test_m5_tail_starts_after_every_window_position_for_real_lengths(sampler):
    """Origins are 480-561 tokens long including a header (real Qwen3 / PG-19 distribution): the tail must still start
    strictly after the largest window position (a fixed start of 512 would collide with keys of the longer origins)."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        order = [int(c) for c in rng.choice(sampler.arch.ids, size=6, replace=False)]
        tokens = {c: [int(x) for x in rng.integers(4, VOCAB, int(rng.integers(480, 562)))] for c in order}
        tail = [int(x) for x in rng.integers(4, VOCAB, 40)]
        ex = sampler._assemble("attrib", order, tokens, tail, 39, 1)
        m5 = D.positions(ex, "M5")
        n_win = ex.T - len(tail)
        assert int(m5[n_win:].min()) > int(m5[:n_win].max()), (trial, int(m5[n_win:].min()), int(m5[:n_win].max()))
        assert m5[n_win:].tolist() == list(range(max(len(t) for t in tokens.values()), max(len(t) for t in tokens.values()) + 40))


def test_headers_add_label_tokens(tk, filler, archive):
    s0 = D.Sampler(archive, filler, tk[0], tk[1], B=B, headers=False)
    s1 = D.Sampler(archive, filler, tk[0], tk[1], B=B, headers=True)
    a, b = s0.make("probe", 3, 3), s1.make("probe", 3, 3)
    assert b.T == a.T + sum(len(tk[1].encode(f"[{archive.label(c)}]\n")) for c in a.window)


def test_dev_only_filler():
    with pytest.raises(ValueError):
        D.make_filler("synthetic", split="test")


# ---------------------------------------------------------------- LoRA


def test_lora_zero_init_is_identity_and_only_lora_trains():
    m = units.tiny_model(seed=1, vocab=VOCAB)
    ids = torch.randint(4, VOCAB, (1, 20))
    ref = units._logits(m, ids)
    params = A.apply_lora(m, r=4, alpha=8.0, seed=0)
    assert units._maxdiff(units._logits(m, ids), ref) == 0.0
    assert len(params) == 2 * 4 * 2 and all(p.requires_grad for p in params)
    assert sum(p.requires_grad for p in m.parameters()) == len(params)
    with torch.no_grad():
        params[1].normal_()                                   # B != 0 changes the logits
    assert units._maxdiff(units._logits(m, ids), ref) > 1e-4


# ---------------------------------------------------------------- attention probe


class _FakeMod:
    layer_idx = 0


def _probe_inputs(ex, archive, H=2, d=4):
    p = M.AttentionProbe(ex, archive, chunk=7)
    q = torch.zeros(1, H, ex.T, d)
    k = torch.randn(1, H, ex.T, d)
    causal = torch.full((1, H, ex.T, ex.T), torch.finfo(torch.float32).min).triu(1) * 0
    mask = torch.zeros(1, H, ex.T, ex.T)
    mask.masked_fill_(torch.ones(ex.T, ex.T).triu(1).bool()[None, None], torch.finfo(torch.float32).min)
    return p, q, k, mask


def test_probe_uniform_attention_has_ratio_one(sampler, archive):
    ex = sampler.make("probe", 2, 4)
    p, q, k, mask = _probe_inputs(ex, archive)
    p(_FakeMod(), q, k, mask, 1.0)                              # q = 0: uniform causal attention
    s = p.summary()
    assert abs(s["own_origin_mass_ratio_to_uniform"] - 1.0) < 1e-5
    assert abs(s["ghost_ratio"] - 1.0) < 1e-4 and 0 < s["own_origin_mass_fraction"] < 1
    assert s["router_entropy"] <= s["router_entropy_max"] + 1e-6


def test_probe_own_origin_only_mask_gives_fraction_one(sampler, archive):
    ex = sampler.make("probe", 2, 4)
    p, q, k, mask = _probe_inputs(ex, archive)
    w = torch.tensor([i for i, s in enumerate(ex.segments) for _ in range(s.length)])
    other = (w[:, None] != w[None, :])[None, None].expand_as(mask)
    mask = mask.masked_fill(other, torch.finfo(torch.float32).min)
    p(_FakeMod(), q, k, mask, 1.0)
    s = p.summary()
    assert abs(s["own_origin_mass_fraction"] - 1.0) < 1e-6 and s["ghost_ratio"] < 1e-6 and s["router_entropy"] < 1e-5
    # the uniform baseline now uses the keys the mask lets a query see (only its own origin here), so attending to them is exactly uniform
    assert abs(s["own_origin_mass_ratio_to_uniform"] - 1.0) < 1e-6


def test_probe_through_the_model_hook(sampler, archive):
    m = units.tiny_model(seed=2, vocab=VOCAB)
    ex = sampler.make("probe", 3, 4)
    s = M.probe_window(m, ex, A.ARMS["V0"], archive, archive.tables)
    assert att.PROBE is None and s["layers"] == 2 and s["heads"] == 8 and 0 < s["own_origin_mass_fraction"] < 1
    s2 = M.probe_window(m, ex, A.ARMS["B1"], archive, archive.tables)         # B1: contiguous positions, opa kernel for the probe
    assert s2["layers"] == 2


# ---------------------------------------------------------------- trainer


def test_b1_and_v3_run_with_identical_config(tmp_path):
    results = {}
    for arm in ("B1", "V3"):
        out, ctx, model = _run(_cfg(arm))
        results[arm] = (out, ctx)
        m = out["metrics"]
        assert m["arm"] == arm and m["opt_steps"] == 3 and m["micro_steps"] == 6 and m["nan_steps"] == 0
        assert f"{arm}_source_attribution" in m and f"{arm}_own_origin_mass_fraction" in m
        assert 0.0 <= m[f"{arm}_own_origin_mass_fraction"] <= 1.0 and m[f"{arm}_std_ppl"] > 0
        assert sorted(ctx.put) == ["config.json", "final_eval.json", "learning_curve.json"]
        assert ctx.put["config.json"]["headers_on"] is True and ctx.put["config.json"]["config"]["headers"] is True
        assert ctx.put["learning_curve.json"]["status"] == "complete"
        curve = ctx.put["learning_curve.json"]["curve"]
        assert [c["tag"] for c in curve] == ["step0", "step2", "final"]               # eval_every = 2 -> light eval at step 2
        assert curve[-1]["opt_step"] == 3 and curve[1]["full"] is False and curve[-1]["probe"]["C"] == 4
        assert any(e["action"] == "train_step" for e in ctx.events)
        adapter = torch.load(__import__("io").BytesIO(ctx.bytes[f"models/{ctx.run_id}/adapter.pt"]))
        assert len(adapter["lora"]) == 2 * 4 * 2 and (len(adapter["oaa"]) > 0) == (arm == "V3")
    # same LoRA budget: V3 adds the OAA parameters on top of identical LoRA parameters
    assert results["B1"][0]["metrics"]["trainable_params"] < results["V3"][0]["metrics"]["trainable_params"]
    cfg_b, cfg_v = (results[a][1].put["config.json"]["config"] for a in ("B1", "V3"))
    assert {k: v for k, v in cfg_b.items() if k != "arm"} == {k: v for k, v in cfg_v.items() if k != "arm"}
    v3 = results["V3"][0]["metrics"]
    assert "V0_own_origin_mass_ratio_to_uniform" in v3 and "V0_source_attribution" in v3
    # step 0 of V3 equals V0 up to the small A initialisation (protocol 5.4: A small); the exact case is a separate test
    fe = results["V3"][1].put["final_eval.json"]
    assert abs(fe["step0"]["probe"]["own_origin_mass_fraction"] - fe["v0"]["probe"]["own_origin_mass_fraction"]) < 5e-2
    assert fe["per_item_final"]["attrib_rank"] and len(fe["per_item_final"]["attrib_em"]) == 4
    assert fe["status"] == "complete" and results["V3"][1].put["config.json"]["status"] == "complete"
    # per-item identities for every checkpoint and for V0
    assert set(fe["per_item"]) == {"v0_untrained", "step0", "step2", "final"}
    meta = fe["per_item"]["final"]["meta"]
    assert len(meta["attrib"]) == 4 and {"seed", "window", "target_origin", "options_origin_ids"} <= set(meta["attrib"][0])
    assert meta["attrib"][0]["seed"] == D.DEV_EVAL_SEED0 and len(meta["passkey"]) == 3 and "seed" in meta["lm"][0]
    same = [fe["per_item"][t]["meta"]["attrib"] for t in ("v0_untrained", "step0", "final")]
    assert same[0] == same[1] == same[2]                                              # the same items at every checkpoint


def test_v3_step0_equals_v0_exactly_with_zero_a_init():
    out, ctx, _ = _run(_cfg("V3", a_init_std=0.0, max_opt_steps=1, eval_every=0))
    fe = ctx.put["final_eval.json"]
    assert fe["step0"]["attrib_em"] == fe["v0"]["attrib_em"]
    assert abs(fe["step0"]["probe"]["own_origin_mass_fraction"] - fe["v0"]["probe"]["own_origin_mass_fraction"]) < 1e-12
    assert fe["step0"]["lm_ppl"] == fe["v0"]["lm_ppl"]


def test_default_a_init_is_small_not_zero():
    assert T.TrainConfig().a_init_std == 1e-3 and T.TrainConfig().eval_every == 40


# ---- equal steps (blocking 1)


def test_both_arms_stop_at_exactly_max_opt_steps():
    for arm in ("V3", "B1"):
        out, ctx, _ = _run(_cfg(arm, max_opt_steps=2, eval_every=0, time_budget_s=1e9))
        m = out["metrics"]
        assert m["opt_steps"] == 2 and m["max_opt_steps_requested"] == 2 and m["equal_steps_met"] is True
        assert m["stop_reason"] == "max_opt_steps"


def test_reference_run_without_max_opt_steps_reports_null_equal_steps():
    t = {"now": 0.0}

    def clock():
        t["now"] += 400.0
        return t["now"]

    out, ctx, _ = _run(_cfg("V3", max_opt_steps=None, time_budget_s=1500.0), clock=clock)
    m = out["metrics"]
    assert m["equal_steps_met"] is None and m["max_opt_steps_requested"] is None and m["stop_reason"] == "time_budget"
    assert ctx.put["config.json"]["equal_steps_met"] is None


def test_complete_is_written_only_after_the_adapter_is_saved(monkeypatch):
    class FailingCtx(Ctx):
        def put_bytes(self, rel, data, content_type=""):
            raise OSError("adapter upload failed")

    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx = FailingCtx()
    with pytest.raises(OSError, match="adapter upload failed"):
        T.run(_cfg("B1", max_opt_steps=1, eval_every=0), units.tiny_model(seed=0, vocab=VOCAB), ftok, enc, filler, ctx)
    assert all(ctx.put[k]["status"] == "partial" for k in ("config.json", "final_eval.json", "learning_curve.json"))


def test_b1_requires_an_integer_max_opt_steps_but_v3_does_not():
    with pytest.raises(ValueError, match="B1 requires"):
        T.TrainConfig.from_args({"arm": "B1"})
    assert T.TrainConfig.from_args({"arm": "B1", "max_opt_steps": 9}).max_opt_steps == 9
    assert T.TrainConfig.from_args({"arm": "V3"}).max_opt_steps is None            # V3: time-limited reference run, unchanged


def test_unfilled_placeholder_for_max_opt_steps_is_refused():
    for bad in ("V3_OPT_STEPS", 2.5, None, 0, True):
        with pytest.raises(ValueError, match="max_opt_steps"):
            T.TrainConfig.from_args({"arm": "B1", "max_opt_steps": bad})
    assert T.TrainConfig.from_args({"arm": "B1", "max_opt_steps": 17}).max_opt_steps == 17


def test_safety_time_stop_is_flagged_when_steps_are_not_reached():
    t = {"now": 0.0}

    def clock():
        t["now"] += 400.0
        return t["now"]

    out, _, _ = _run(_cfg("B1", time_budget_s=1500.0, max_opt_steps=50), clock=clock)
    assert out["metrics"]["equal_steps_met"] is False and out["metrics"]["stop_reason"] == "time_budget"


# ---- memory preflight and partial outputs (blocking 4)


def test_preflight_abort_writes_partial_outputs_and_reraises(monkeypatch):
    monkeypatch.setattr(T, "_peak_gib", lambda device: 30.0)
    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx = Ctx()
    with pytest.raises(RuntimeError, match="memory preflight failed"):
        T.run(_cfg("V3"), units.tiny_model(seed=0, vocab=VOCAB), ftok, enc, filler, ctx)
    assert sorted(ctx.put) == ["config.json", "final_eval.json", "learning_curve.json"]
    assert ctx.put["config.json"]["status"] == "partial" and "preflight" in ctx.put["config.json"]["error"]
    assert ctx.put["final_eval.json"]["status"] == "partial"
    assert any(e["action"] == "preflight" for e in ctx.events)


def test_failure_during_training_keeps_the_curve(monkeypatch):
    calls = {"n": 0}
    orig = A.forward_example

    def boom(*a, **k):
        if k.get("backward"):
            calls["n"] += 1
            if calls["n"] == 4:                                  # preflight is call 1; fail in training step 2
                raise RuntimeError("cuda oom (simulated)")
        return orig(*a, **k)

    monkeypatch.setattr(A, "forward_example", boom)
    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx = Ctx()
    with pytest.raises(RuntimeError, match="simulated"):
        T.run(_cfg("B1", grad_accum=2, max_opt_steps=5, eval_every=0), units.tiny_model(seed=0, vocab=VOCAB), ftok, enc, filler, ctx)
    cfg = ctx.put["config.json"]
    assert cfg["status"] == "partial" and cfg["opt_steps"] == 1 and "simulated" in cfg["error"]
    assert ctx.put["learning_curve.json"]["status"] == "partial"
    assert [c["tag"] for c in ctx.put["learning_curve.json"]["curve"]] == ["step0"]


def test_preflight_logs_peak_and_longest_example():
    out, ctx, _ = _run(_cfg("V3", max_opt_steps=1, eval_every=0))
    pf = [e for e in ctx.events if e["action"] == "preflight"][0]
    assert pf["longest_T"] > 0 and pf["probe_T"] > 0 and pf["limit_gib"] == 21.0
    assert ctx.put["config.json"]["preflight"]["longest_T"] == pf["longest_T"]


def test_v3_trains_the_oaa_parameters_and_lora():
    out, ctx, model = _run(_cfg("V3", max_opt_steps=3))
    ps = list(att.oaa_parameters(model))
    assert any(float(p.A.abs().max()) > 0 for _, p in ps)                  # A moved off 0
    assert all(float(p.lam.abs().max()) == 0 for _, p in ps)               # no penalty in V3
    assert any(float(m.B.abs().max()) > 0 for mod in att.attention_modules(model) for m in (mod.q_proj, mod.v_proj))
    assert out["metrics"]["trainable_params"] > 0


def test_training_is_deterministic_for_a_fixed_seed():
    a, _, _ = _run(_cfg("V3"), seed=5)
    b, _, _ = _run(_cfg("V3"), seed=5)
    for k in ("train_loss_mean", "V3_lm_ppl", "V3_std_ppl", "V3_own_origin_mass_fraction", "V3_source_attribution"):
        assert a["metrics"][k] == b["metrics"][k], k
    c, _, _ = _run(_cfg("V3", seed=4), seed=5)
    assert c["metrics"]["train_loss_mean"] != a["metrics"]["train_loss_mean"]


def test_same_example_stream_for_both_arms(tk, filler, archive):
    """Headers are part of the shared config: B1 and V3 get byte-identical token inputs for the same seed (only the
    attention arm differs), and the origins carry the "[label]" header tokens."""
    cb, cv = _cfg("B1"), _cfg("V3")
    assert cb.headers and cv.headers                                  # default: headers on for both arms
    streams = []
    for c in (cb, cv):
        smp = D.Sampler(archive, filler, tk[0], tk[1], B=c.B, headers=c.headers)
        streams.append(T.make_stream(c, smp))
    for i in range(8):
        a, b = streams[0](i), streams[1](i)
        assert a.kind == b.kind and a.ids.numpy().tobytes() == b.ids.numpy().tobytes()
        assert a.origin_tok.numpy().tobytes() == b.origin_tok.numpy().tobytes() and a.ans_start == b.ans_start
        assert a.T <= 4096
        first = a.segments[0]
        hdr = tk[1].encode(f"[{archive.label(a.window[0])}]\n")
        assert a.ids[:len(hdr)].tolist() == hdr and first.length > len(hdr)
    # the probe window of V3 is built by the same sampler, so it carries the same headers
    pw = D.Sampler(archive, filler, tk[0], tk[1], B=cv.B, headers=cv.headers).make("probe", 900, 4, role="eval")
    assert pw.ids[:len(hdr)].tolist() == tk[1].encode(f"[{archive.label(pw.window[0])}]\n")


def test_time_budget_stops_training_but_still_evaluates_and_saves():
    t = {"now": 0.0}

    def clock():
        t["now"] += 400.0
        return t["now"]

    out, ctx, _ = _run(_cfg("B1", time_budget_s=1500.0, max_opt_steps=50), clock=clock)
    m = out["metrics"]
    assert m["stop_reason"] == "time_budget" and m["opt_steps"] < 50
    assert "learning_curve.json" in ctx.put and f"models/{ctx.run_id}/adapter.pt" in ctx.bytes
    assert ctx.put["learning_curve.json"]["curve"][-1]["tag"] == "final"


def test_gradient_checkpointing_matches_plain_gradients(sampler, archive):
    """Backward inside the OAA context under gradient checkpointing gives the same gradients as without it."""
    grads = {}
    for gc in (False, True):
        m = units.tiny_model(seed=9, vocab=VOCAB)
        cfg = _cfg("V3", grad_checkpointing=gc, a_init_std=0.05)
        lora, new = T.prepare_model(m, cfg)
        m.train()
        ex = sampler.make("attrib", 3, 3)
        A.forward_example(m, ex, A.ARMS["V3"], archive.tables, backward=True)
        grads[gc] = [p.grad.clone() for p in lora + new]
    for a, b in zip(grads[False], grads[True]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
    assert any(float(g.abs().max()) > 0 for g in grads[True][-8:])             # OAA parameters receive gradient


def test_real_shape_one_optimizer_step():
    """The real Qwen3-4B head geometry (2 layers, random weights): one V3 optimizer step end to end."""
    ftok, enc = D.make_tokenizers("hash", 4096)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    model = units.real_shape_model(seed=0, layers=2, vocab=4096)
    cfg = _cfg("V3", m=32, max_opt_steps=1, grad_accum=1, eval_every=0, train_origins=(3, 3),
               eval_full={"attrib": 1, "passkey": 1, "lm": 1, "std": 1, "probe": 1})
    ctx = Ctx()
    out = T.run(cfg, model, ftok, enc, filler, ctx)
    assert out["metrics"]["opt_steps"] == 1 and out["metrics"]["nan_steps"] == 0
    assert out["metrics"]["trainable_params"] > 1e6


def test_unknown_args_rejected():
    with pytest.raises(ValueError):
        T.TrainConfig.from_args({"arm": "V3", "lerning_rate": 1})
    with pytest.raises(ValueError):
        T.TrainConfig.from_args({"arm": "A2"})


# ---------------------------------------------------------------- remedies (protocol 5.2)

PFX = 5                                    # length of the shared chat prefix in the tests


def _ritem(n_docs=3, L=20, seed=0, with_parents=True):
    rng = np.random.default_rng(seed)
    tok = lambda n: [int(x) for x in rng.integers(4, VOCAB, n)]
    docs = [{"origin_id": i, "tokens": tok(L), "parents": [tok(L), tok(L)] if with_parents else []} for i in range(n_docs)]
    return R.RItem("it", docs, tok(6), tok(3), tok(PFX), ["83 C", "83"], "83 C")


def _r0(eng, it):
    """R0 cache pieces: full keys/values of prefix + documents, the document-only slice, and the prefix length."""
    p = len(it.prefix_ids)
    ids = it.prefix_ids + [t for d in it.docs for t in d["tokens"]]
    k0, v0, _ = eng.prefill(ids, list(range(len(ids))))
    return k0, v0, ([x[:, :, p:] for x in k0], [x[:, :, p:] for x in v0]), len(ids), p


@pytest.fixture(scope="module")
def eng():
    return R.Engine(units.tiny_model(seed=11, vocab=VOCAB))


def test_single_document_r1_and_full_recompute_equal_r0(eng):
    """With one document the shared-prefix + fresh-frame layout IS the contiguous layout: R1 and R2oracle at 100% reproduce R0."""
    it = _ritem(n_docs=1)
    k0, v0, r0_kv, n_ids, p = _r0(eng, it)
    ref = eng.answer(k0, v0, None, it.q_ids, it.a_ids, n_ids)
    _, enc = D.make_tokenizers("hash", VOCAB)
    for cond in ("R1", "R2oracle_100"):
        c = R.build_cache(eng, it, cond, enc, r0_kv)
        assert c.q_start == n_ids
        lg = eng.answer(c.keys, c.values, c.vis, it.q_ids, it.a_ids, c.q_start)
        assert float((lg - ref).abs().max()) < 1e-4, cond


def test_rerotate_matches_keys_computed_at_the_new_position(eng):
    ids = _ritem(1).docs[0]["tokens"]
    k_a, _, _ = eng.prefill(ids, list(range(len(ids))))
    k_b, _, _ = eng.prefill(ids, [p + 37 for p in range(len(ids))])
    moved = eng.rerotate(k_a[0], [37] * len(ids))                     # layer 0 keys depend on position only through RoPE
    assert float((moved - k_b[0]).abs().max()) < 1e-4


def test_conditions_run_and_differ_from_r0_on_several_documents(eng):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=3)
    res = R.run_item(eng, it, enc)
    assert list(res) == R.CONDITIONS and res["R0"]["kl"] == 0.0
    assert res["R1"]["kl"] > 1e-6 and all(res[c]["kl"] >= 0 for c in res)
    assert all(isinstance(res[c]["text"], str) and res[c]["em"] in (0, 1) and "gold_logprob" in res[c] for c in res)
    assert len(res["R1"]["v_cos"]) == 2 and res["R1"]["v_cos"][0] > 0.999            # layer 0 values are context-free
    assert res["R1"]["v_cos"][1] < res["R1"]["v_cos"][0]


def test_r2_replaces_exactly_the_requested_fraction_with_r0_values(eng):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=3)
    k0, v0, r0, _, p = _r0(eng, it)
    ids_n = 3 * 20
    base = R.build_cache(eng, it, "R1", enc, r0)
    c = R.build_cache(eng, it, "R2oracle_15", enc, r0)
    memo = {}
    c_memo = R.build_cache(eng, it, "R2oracle_15", enc, r0, memo)                    # via the shared R1 base: same result
    assert all(torch.equal(a, b) for a, b in zip(c.values, c_memo.values)) and "R1" in memo
    assert torch.equal(R.build_cache(eng, it, "R1", enc, r0).values[1], base.values[1])        # base untouched
    changed = (c.values[1] != base.values[1]).any(-1).any(1).any(0)
    n_changed = int(changed.sum())
    assert int(np.ceil(0.15 * ids_n)) - 3 <= n_changed <= int(np.ceil(0.15 * ids_n))
    i = int(changed.nonzero()[0, 0])
    assert torch.equal(c.values[1][0, :, i], v0[1][0, :, p + c.abs0[i]].to(c.values[1].dtype))
    # re-rotated key: at its cached position it equals the R0 key moved there
    moved = eng.rerotate(k0[1][:, :, p + c.abs0[i]:p + c.abs0[i] + 1].float(), [c.pos[i] - (p + c.abs0[i])])
    assert float((c.keys[1][:, :, i:i + 1] - moved).abs().max()) < 1e-5


def test_r3_r4_r5_r6_cache_layouts(eng):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=3, L=20)
    n, p = 3 * 20, PFX
    r3 = R.build_cache(eng, it, "R3", enc)
    ns = p + len(enc.encode(R.SHARED_PREFIX))
    assert r3.keys[0].shape[2] == ns + n and r3.doc[:ns] == [-1] * ns and r3.pos[ns] == ns
    r1b = R.build_cache(eng, it, "R1", enc)
    assert r1b.keys[0].shape[2] == p + n and r1b.doc[:p] == [-1] * p and r1b.pos[p] == p and r1b.p0 == p
    r4 = R.build_cache(eng, it, "R4", enc)
    assert r4.vis is None and r4.keys[0].shape[2] == p + n                            # recompute, not masking
    first = [i for i, t in enumerate(r4.tok) if t == 0]
    changed = (r4.values[1] != r1b.values[1]).any(-1).any(1).any(0)
    assert [i for i in range(p + n) if bool(changed[i])] == first[1:]                 # only first tokens of documents 1.. changed
    assert torch.allclose(r4.keys[0], r1b.keys[0], atol=1e-5)                          # layer 0 key is context-free
    r5 = R.build_cache(eng, it, "R5_8", enc)
    assert r5.keys[0].shape[2] == p + n + 3 * 8 and r5.tok[p:p + 8] == [-1] * 8 and r5.pos[p + 8] == p + 8
    assert r5.q_start == p + 8 + 20
    r6 = R.build_cache(eng, it, "R6_2", enc)
    assert r6.keys[0].shape[2] == p + n and r6.pos[p] == p + 40 and r6.q_start == p + 60
    r1 = R.build_cache(eng, _ritem(n_docs=3, L=20, with_parents=False), "R5_8", enc)
    assert r1.keys[0].shape[2] == p + n                                               # no parent: R5 degenerates to R1


def test_r4_two_docs_recomputed_first_token_equals_r0_at_its_contiguous_position(eng):
    """With two documents, document 0's R1 KV equals R0's, so the recomputed first token of document 1 (viewed at its contiguous
    position) must reproduce the R0 key and value in every layer. At its cached position against keys at the same positions it
    would not."""
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=2, L=20)
    k0, v0, r0, _, p = _r0(eng, it)
    r1 = R.build_cache(eng, it, "R1", enc)
    r4 = R.build_cache(eng, it, "R4", enc)
    s = p + 20
    assert r4.pos[s] == p and r4.tok[s] == 0 and r4.doc[s] == 1
    for l in range(eng.L):
        moved = eng.rerotate(r4.keys[l][:, :, s:s + 1].float(), [20])                 # cached position p -> contiguous p + 20
        assert float((moved - k0[l][:, :, s:s + 1]).abs().max()) < 1e-4, ("key", l)
        assert float((r4.values[l][:, :, s:s + 1] - v0[l][:, :, s:s + 1]).abs().max()) < 1e-4, ("value", l)
    assert float((r1.values[1][:, :, s:s + 1] - v0[1][:, :, s:s + 1]).abs().max()) > 1e-3        # R1 itself is wrong there
    keep = [i for i in range(p + 40) if i != s]
    assert torch.equal(r4.values[1][:, :, keep], r1.values[1][:, :, keep])


def test_r4_attends_to_earlier_documents_at_contiguous_positions(eng, monkeypatch):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=3, L=20)
    p = PFX
    seen = []
    orig = eng.append_token

    def spy(keys, values, token, pos):
        seen.append((keys[0].clone(), pos))
        return orig(keys, values, token, pos)

    monkeypatch.setattr(eng, "append_token", spy)
    R.build_cache(eng, it, "R4", enc)
    assert [q for _, q in seen] == [p + 20, p + 40]                                   # P = prefix length + offset of the document
    prefix, _ = seen[1]
    assert prefix.shape[2] == p + 40
    ref, _, _ = eng.prefill(it.prefix_ids, list(range(p)))
    assert float((prefix[:, :, :p] - ref[0]).abs().max()) < 1e-4
    for j, d in enumerate(it.docs[:2]):                                               # layer-0 keys of docs 0 and 1 at contiguous positions
        ref, _, _ = eng.prefill(d["tokens"], list(range(p + 20 * j, p + 20 * j + 20)))
        assert float((prefix[:, :, p + 20 * j:p + 20 * j + 20] - ref[0]).abs().max()) < 1e-4, j


def test_summary_recovery_oracle_split_and_stats():
    per = []
    # 10 items: R0 right on 8, R1 right on 4, R5_8 right on 6, R3 on 5, R2oracle_5 on 8, R7oracle on 8
    for i in range(10):
        per.append({c: {"em": 0, "kl": 0.1, "ttft_s": 1.0} for c in R.CONDITIONS})
        for c, k in (("R0", 8), ("R1", 4), ("R5_8", 6), ("R3", 5), ("R2oracle_5", 8), ("R7oracle", 8)):
            per[-1][c]["em"] = int(i < k)
    s = R.summarize(per)
    assert s["R0_correct"] == 8 and s["acc"]["R0"] == 0.8 and s["acc"]["R1"] == 0.4 and s["r0_too_low"] is False
    assert s["best_remedy"] == "R5_8" and abs(s["best_remedy_recovery_fraction"] - 0.75) < 1e-12          # oracles excluded
    assert s["best_remedy_incl_oracle"] in ("R2oracle_5", "R7oracle") and abs(s["best_remedy_recovery_fraction_incl_oracle"] - 1.0) < 1e-12
    assert abs(s["gap_closed"]["R5_8"] - 0.5) < 1e-12
    assert abs(s["R5_vs_R1_accuracy"] - 20.0) < 1e-9 and s["R5_vs_R1"]["n10"] == 2 and s["R5_vs_R1"]["n01"] == 0
    assert s["R5_vs_R1"]["p_raw"] == pytest.approx(0.5) and s["R5_vs_R1"]["ci_lo"] <= 20.0 <= s["R5_vs_R1"]["ci_hi"]
    lo, hi = s["acc_ci95_points"]["R0"]
    assert lo <= 80.0 <= hi and s["ttft_s_mean"]["R1"] == 1.0
    assert not hasattr(R, "mcnemar_exact_p")
    low = [{c: {"em": int(i < 1 and c == "R0"), "kl": 0.0} for c in R.CONDITIONS} for i in range(10)]
    assert R.summarize(low)["r0_too_low"] is True                                      # 10% < 20%


def test_cos_and_ttft_fields_for_r1_r5_r6(eng):
    _, enc = D.make_tokenizers("hash", VOCAB)
    res = R.run_item(eng, _ritem(n_docs=3), enc, ["R0", "R1", "R5_8", "R6_2", "R4"])
    for c in ("R1", "R5_8", "R6_2"):
        assert len(res[c]["k_cos"]) == 2 and len(res[c]["v_cos"]) == 2
        assert res[c]["k_cos"][0] > 0.999 and res[c]["v_cos"][0] > 0.999            # layer 0 is context-free
    assert all(res[c]["ttft_s"] > 0 for c in res)


# ---- generation and scoring (greedy from the composed cache, Track A scorer)


def test_generation_path_runs_is_bounded_and_matches_teacher_forcing(eng):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=2)
    k0, v0, r0, n_ids, p = _r0(eng, it)
    gen, ttft = eng.generate(k0, v0, it.q_ids, n_ids, enc)
    assert 0 < len(gen) <= R.MAX_NEW and ttft > 0 and all(isinstance(t, int) for t in gen)
    lg = eng.answer(k0, v0, None, it.q_ids, gen, n_ids)                                 # greedy tokens are the argmax chain
    assert lg.argmax(-1).tolist() == gen
    c = R.build_cache(eng, it, "R5_8", enc, r0)
    gen2, _ = eng.generate(c.keys, c.values, it.q_ids, c.q_start, enc)
    lg2 = eng.answer(c.keys, c.values, None, it.q_ids, gen2, c.q_start)
    assert lg2.argmax(-1).tolist() == gen2


def test_generation_stops_at_stop_token_and_at_newline(eng, monkeypatch):
    _, enc = D.make_tokenizers("hash", VOCAB)
    it = _ritem(n_docs=1)
    k0, v0, _, n_ids, p = _r0(eng, it)
    # stop token: make the first greedy token a stop id
    first, _ = eng.generate(k0, v0, it.q_ids, n_ids, enc, max_new=1)
    monkeypatch.setattr(R, "stop_ids", lambda e: {first[0]})
    gen, _ = eng.generate(k0, v0, it.q_ids, n_ids, enc)
    assert gen == []
    # newline: decode() reports a newline after the second token
    monkeypatch.setattr(R, "stop_ids", lambda e: set())
    monkeypatch.setattr(R, "decode", lambda e, ids: "ab\n" if len(ids) >= 2 else "a")
    gen, _ = eng.generate(k0, v0, it.q_ids, n_ids, enc)
    assert len(gen) == 2


def test_scoring_uses_the_track_a_scorer_with_aliases():
    from opa import scoring
    it = R.RItem("x", [], [], [], [], ["83 °C", "83 C", "83"], "83 °C")
    assert R.score_text(it, "83 °C") == 1 and R.score_text(it, "83") == 1 and R.score_text(it, "  83 C.\nextra line") == 1
    assert R.score_text(it, "about 90 °C") == 0 and R.score_text(it, "") == 0
    assert R.score_text(it, "83 °C") == int(scoring.exact_match("83 °C", it.answers))


def test_items_from_generator_dev_items_and_window():
    from opa.data import mtlbmhqa
    from opa.data.filler import SyntheticFiller, WhitespaceTokenizer
    f, t = SyntheticFiller(n_books=40, words_per_book=30000), WhitespaceTokenizer()
    its = [mtlbmhqa.make_item(s, 4, 2, 8192, False, 0.0, f, t, split="dev", run_blocks=2, block_tokens=512) for s in (0, 1)]
    _, enc = D.make_tokenizers("hash", VOCAB)
    items = R.items_from_mtlbmhqa(its, enc, C=5, seed=0)
    sep = enc.encode(R.DOC_SEP)
    for it, raw in zip(items, its):
        assert len(it.docs) == 5 and set(raw["gold"]["fact_origins"]) <= set(it.meta["window"]) and it.meta["facts_in_window"]
        assert it.meta["window"] == sorted(it.meta["window"])
        assert it.prefix_ids == enc.encode(R.CHAT_PREFIX) and it.answers[0] == raw["answer"] and set(raw["answer_aliases"]) <= set(it.answers)
        assert it.gold == raw["answer"] and (not sep or it.docs[0]["tokens"][-len(sep):] == sep)
        thread_of = raw["gold"]["thread_of"]
        for d in it.docs:
            exp = [q for q in range(d["origin_id"] - 1, -1, -1) if thread_of[q] == thread_of[d["origin_id"]]][:2]
            assert len(d["parents"]) == len(exp)


def test_remedies_main_entry_with_tiny_model(monkeypatch):
    monkeypatch.setattr(units, "load_model", lambda name, device, dtype=torch.float32: units.tiny_model(seed=2, vocab=VOCAB))
    monkeypatch.setattr(R, "build_items", lambda *a, **k: [_ritem(3, 20, s) for s in range(3)])
    monkeypatch.setattr(D, "make_filler", lambda *a, **k: None)
    ctx = Ctx()
    out = R.main({"model": "tiny", "tokenizer": "hash", "filler": "synthetic", "n_items": 3}, ctx)
    m = out["metrics"]
    assert m["n_items"] == 3 and "best_remedy_recovery_fraction" in m and "R5_vs_R1_accuracy" in m and m["R0_correct"] in (0, 1, 2, 3)
    assert m["r0_too_low"] is True                                                    # random tiny model: R0 = 0%, flagged, job finishes
    rec = ctx.put["remedies.json"]
    assert len(rec["per_item"]) == 3 and set(rec["per_item"][0]) == set(R.CONDITIONS)
    assert sorted(k for k in ctx.put if k.startswith("items/")) == ["items/000.json", "items/001.json", "items/002.json"]
    one = ctx.put["items/000.json"]
    assert one["gold"] == "83 C" and one["answers"] == ["83 C", "83"] and isinstance(one["result"]["R1"]["text"], str)
    assert rec["summary"]["r0_too_low"] is True


def test_remedies_time_guard_stops_cleanly_and_writes_what_it_has(monkeypatch):
    monkeypatch.setattr(units, "load_model", lambda name, device, dtype=torch.float32: units.tiny_model(seed=2, vocab=VOCAB))
    monkeypatch.setattr(R, "build_items", lambda *a, **k: [_ritem(3, 20, s) for s in range(5)])
    monkeypatch.setattr(D, "make_filler", lambda *a, **k: None)
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr(R.time, "time", lambda: float(next(ticks)))
    ctx = Ctx()
    out = R.main({"model": "tiny", "tokenizer": "hash", "filler": "synthetic", "n_items": 5, "time_budget_s": 600.0,
                  "item_margin_s": 50.0, "conditions": ["R0", "R1"]}, ctx)
    assert out["metrics"]["stop_reason"] == "time_budget" and 0 < out["metrics"]["n_items"] < 5
    assert len([k for k in ctx.put if k.startswith("items/")]) == out["metrics"]["n_items"] and "remedies.json" in ctx.put


# ---------------------------------------------------------------- learning-curve mode, checkpoints, resume (PLAN_V3 6a Step 1)


def _lc_cfg(**kw):
    base = dict(max_opt_steps=4, eval_at_steps=[1, 2, 4], save_checkpoint_at=[2], eval_every=0, log_every=1,
                monitor={"attrib": 3, "passkey": 2, "probe": 1})
    base.update(kw)
    return _cfg("V3", **base)


def test_eval_at_steps_runs_monitoring_evals_and_saves_the_checkpoint():
    out, ctx, _ = _run(_lc_cfg())
    lc = ctx.put["learning_curve.json"]
    assert [c["tag"] for c in lc["curve"]] == ["step0", "step1", "step2", "step4"]            # no extra final eval: step 4 is the last
    assert out["metrics"]["opt_steps"] == 4 and out["metrics"]["stop_reason"] == "max_opt_steps"
    meta = ctx.put["final_eval.json"]["per_item"]["step4"]["meta"]
    assert meta["attrib"][0]["seed"] == D.MONITOR_SEED0 and len(meta["attrib"]) == 3 and len(meta["passkey"]) == 2
    assert D.MONITOR_SEED0 == 2000 and D.G2_SEED0 == 3000 and D.DEV_EVAL_SEED0 == 900            # reserved, disjoint ranges
    for c in lc["curve"]:
        assert "attrib_em" in c and "passkey_em" in c and "probe" in c and "lm_ppl" not in c        # the monitored quantities only
    assert lc["curve"][2]["train_loss"] is not None and lc["monitor"]["seed0"] == 2000
    assert f"models/{ctx.run_id}/step2/checkpoint.pt" in ctx.bytes and any(e["action"] == "checkpoint_saved" for e in ctx.events)
    assert ctx.put["final_eval.json"]["per_item_final"] == ctx.put["final_eval.json"]["per_item"]["step4"]["per_item"]


def test_monitoring_set_is_disjoint_from_the_pilot_eval_set(sampler):
    a = sampler.make("attrib", D.MONITOR_SEED0, 3, role="eval")
    b = sampler.make("attrib", D.DEV_EVAL_SEED0, 3, role="eval")
    c = sampler.make("attrib", D.G2_SEED0, 3, role="eval")
    assert not torch.equal(a.ids, b.ids) and not torch.equal(a.ids, c.ids) and not torch.equal(b.ids, c.ids)
    assert D.DEV_EVAL_SEED0 + 100 <= D.MONITOR_SEED0 and D.MONITOR_SEED0 + 64 <= D.G2_SEED0


def test_eval_at_steps_validation():
    with pytest.raises(ValueError, match="needs max_opt_steps"):
        T.TrainConfig.from_args({"arm": "V3", "eval_at_steps": [10]})
    with pytest.raises(ValueError, match="beyond max_opt_steps"):
        T.TrainConfig.from_args({"arm": "V3", "max_opt_steps": 5, "eval_at_steps": [10]})
    c = T.TrainConfig.from_args({"arm": "V3", "max_opt_steps": 400, "eval_at_steps": [400, 50, 200, 100], "save_checkpoint_at": [400]})
    assert c.eval_at_steps == [50, 100, 200, 400] and c.save_checkpoint_at == [400]


def test_time_budget_does_not_end_a_learning_curve_run_early_when_large():
    out, _, _ = _run(_lc_cfg(time_budget_s=1e9))
    assert out["metrics"]["opt_steps"] == 4 and out["metrics"]["equal_steps_met"] is True


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    def losses(ctx):
        return {e["opt_step"]: e["loss"] for e in ctx.events if e["action"] == "train_step"}

    full, ctx_full, _ = _run(_cfg("V3", max_opt_steps=4, eval_every=0, log_every=1))
    part, ctx_part, _ = _run(_cfg("V3", max_opt_steps=2, eval_every=0, log_every=1, save_checkpoint_at=[2]))
    ck = tmp_path / "checkpoint.pt"
    ck.write_bytes(ctx_part.bytes[f"models/{ctx_part.run_id}/step2/checkpoint.pt"])
    res, ctx_res, _ = _run(_cfg("V3", max_opt_steps=4, eval_every=0, log_every=1, resume_from=str(ck)))
    lf, lp, lr_ = losses(ctx_full), losses(ctx_part), losses(ctx_res)
    assert sorted(lr_) == [3, 4] and sorted(lp) == [1, 2]
    for s in (1, 2):
        assert lp[s] == pytest.approx(lf[s], abs=1e-6)
    for s in (3, 4):
        assert lr_[s] == pytest.approx(lf[s], abs=1e-4), (s, lr_[s], lf[s])             # same loss at the next steps after resuming
    assert res["metrics"]["opt_steps"] == 4 and any(e["action"] == "resumed" and e["opt_step"] == 2 for e in ctx_res.events)
    assert [c["tag"] for c in ctx_res.put["learning_curve.json"]["curve"]][0] == "resume_start"
    assert any(e["action"] == "resumed" for e in ctx_res.events)


def test_checkpoint_contains_lora_oaa_and_optimizer_state():
    _, ctx, _ = _run(_lc_cfg(max_opt_steps=2, eval_at_steps=[2], save_checkpoint_at=[2]))
    ck = torch.load(__import__("io").BytesIO(ctx.bytes[f"models/{ctx.run_id}/step2/checkpoint.pt"]), weights_only=False)
    assert ck["opt_step"] == 2 and ck["micro"] == 2 * 2 and len(ck["lora"]) == 2 * 4 * 2 and len(ck["oaa"]) > 0
    assert len(ck["opt"]["state"]) > 0 and "exp_avg" in next(iter(ck["opt"]["state"].values()))


def test_projection_plateau_rule_and_linear_extrapolation():
    def rec(step, frac):
        return {"tag": f"step{step}", "opt_step": step, "probe": {"own_origin_mass_fraction": frac, "own_origin_mass_ratio_to_uniform": 1.0},
                "attrib_em": 0.3, "passkey_em": 0.5, "train_loss": 2.0}
    flat = T.projection([rec(50, 0.20), rec(100, 0.30), rec(200, 0.40), rec(400, 0.41)])
    assert abs(flat["fraction_gain_200_to_400"] - 0.01) < 1e-12 and flat["plateau_by_400"] is True
    rising = T.projection([rec(50, 0.20), rec(100, 0.30), rec(200, 0.40), rec(400, 0.60)])
    assert rising["plateau_by_400"] is False and abs(rising["fraction_gain_200_to_400"] - 0.2) < 1e-12
    assert abs(rising["fraction_gain_per_200_steps_last_interval"] - 0.2) < 1e-12
    assert rising["linear_steps_to_target"] == 400 + 300                                          # slope 0.001/step, 0.30 to go
    assert T.projection([rec(200, 0.5), rec(400, 0.4)])["linear_steps_to_target"] is None
    assert T.projection([rec(200, 0.5), rec(400, 0.52)])["plateau_by_400"] is False                # exactly 0.02 is not below
    _, ctx, _ = _run(_lc_cfg())
    pr = ctx.put["learning_curve.json"]["projection"]
    assert [p["step"] for p in pr["points"]] == [0, 1, 2, 4] and "fraction_gain_per_200_steps_last_interval" in pr


# ---- checkpoint robustness (checker a45de95)


def _ck(ctx, step):
    return torch.load(__import__("io").BytesIO(ctx.bytes[f"models/{ctx.run_id}/step{step}/checkpoint.pt"]), weights_only=False)


def test_checkpoint_is_saved_before_the_monitoring_eval_at_the_same_step():
    _, ctx, _ = _run(_lc_cfg(max_opt_steps=2, eval_at_steps=[2], save_checkpoint_at=[2]))
    acts = [(i, e["action"], e.get("opt_step")) for i, e in enumerate(ctx.events)]
    i_save = next(i for i, a, s in acts if a == "checkpoint_saved" and s == 2)
    i_eval = next(i for i, e in enumerate(ctx.events) if e["action"] == "eval" and e.get("tag") == "step2")
    assert i_save < i_eval
    assert _ck(ctx, 2)["stop_reason"] == "planned" and _ck(ctx, 2)["equal_steps_met"] is True


def test_time_budget_before_the_save_step_still_writes_a_flagged_checkpoint():
    t = {"now": 0.0}

    def clock():
        t["now"] += 130.0
        return t["now"]

    out, ctx, _ = _run(_cfg("V3", max_opt_steps=8, save_checkpoint_at=[8], eval_every=0, time_budget_s=1500.0, log_every=1), clock=clock)
    m = out["metrics"]
    assert m["stop_reason"] == "time_budget" and 1 <= m["opt_steps"] < 8 and m["equal_steps_met"] is False
    assert m["status"] == "incomplete_steps" and ctx.put["config.json"]["status"] == "incomplete_steps"
    assert ctx.put["learning_curve.json"]["status"] == "incomplete_steps" and ctx.put["final_eval.json"]["status"] == "incomplete_steps"
    ck = _ck(ctx, m["opt_steps"])
    assert ck["stop_reason"] == "time_budget" and ck["equal_steps_met"] is False and ck["planned_steps"] == 8
    assert ck["opt_step"] == m["opt_steps"] and f"models/{ctx.run_id}/step8/checkpoint.pt" not in ctx.bytes


def test_crash_after_a_completed_step_keeps_a_flagged_checkpoint(monkeypatch):
    calls = {"n": 0}
    orig = A.forward_example

    def boom(*a, **k):
        if k.get("backward"):
            calls["n"] += 1
            if calls["n"] == 6:                                  # preflight = call 1, steps 1-2 = calls 2-5, crash in step 3
                raise RuntimeError("cuda oom (simulated)")
        return orig(*a, **k)

    monkeypatch.setattr(A, "forward_example", boom)
    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx = Ctx()
    with pytest.raises(RuntimeError, match="simulated"):
        T.run(_cfg("V3", grad_accum=2, max_opt_steps=6, save_checkpoint_at=[6], eval_every=0), units.tiny_model(seed=0, vocab=VOCAB),
              ftok, enc, filler, ctx)
    ck = _ck(ctx, 2)
    assert ck["opt_step"] == 2 and ck["stop_reason"].startswith("crash: RuntimeError") and ck["equal_steps_met"] is False
    assert ctx.put["config.json"]["status"] == "partial"


def test_resume_refuses_a_changed_training_configuration(tmp_path):
    _, ctx, _ = _run(_cfg("V3", max_opt_steps=2, eval_every=0, save_checkpoint_at=[2]))
    ck = tmp_path / "c.pt"
    ck.write_bytes(ctx.bytes[f"models/{ctx.run_id}/step2/checkpoint.pt"])
    ftok, enc = D.make_tokenizers("hash", VOCAB)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx2 = Ctx()
    with pytest.raises(ValueError, match="resume refused.*lr_lora"):
        T.run(_cfg("V3", max_opt_steps=4, eval_every=0, resume_from=str(ck), lr_lora=5e-4), units.tiny_model(seed=0, vocab=VOCAB),
              ftok, enc, filler, ctx2)
    chk = [e for e in ctx2.events if e["action"] == "resume_config_check"][0]
    assert chk["ok"] is False and "lr_lora" in chk["differs"] and ctx2.put["config.json"]["status"] == "partial"
    # step counts, evaluation and time budget may differ
    c1 = T.TrainConfig.from_args({"arm": "V3", "max_opt_steps": 400, "time_budget_s": 1.0, "eval_at_steps": [50]})
    c2 = T.TrainConfig.from_args({"arm": "V3", "max_opt_steps": 800, "time_budget_s": 99.0, "eval_at_steps": [200], "resume_from": "x"})
    from dataclasses import asdict
    assert T.resume_config_diff(c2, asdict(c1)) == {}
    assert "headers" in T.resume_config_diff(T.TrainConfig.from_args({"arm": "V3", "headers": False}), asdict(c1))


def test_resume_restores_nan_steps(tmp_path):
    _, ctx, _ = _run(_cfg("V3", max_opt_steps=2, eval_every=0, save_checkpoint_at=[2]))
    ck = torch.load(__import__("io").BytesIO(ctx.bytes[f"models/{ctx.run_id}/step2/checkpoint.pt"]), weights_only=False)
    ck["nan_steps"] = 3
    path = tmp_path / "c.pt"
    torch.save(ck, path)
    out, ctx2, _ = _run(_cfg("V3", max_opt_steps=4, eval_every=0, resume_from=str(path)))
    assert out["metrics"]["nan_steps"] == 3 and ctx2.put["config.json"]["nan_steps"] == 3
