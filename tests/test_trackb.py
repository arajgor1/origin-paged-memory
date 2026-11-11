"""Track B: position maps, Origin-Anchored Attention and unit tests U1-U5 on a tiny random Qwen3 (GQA 4:1, q/k norm,
rope theta 5e6), on a 2-layer random model with the real Qwen3-4B head geometry, and on the Qwen2 fallback (CPU, no download)."""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch

from opa.trackb import attention as att
from opa.trackb import units

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "bundle_spec" / "test.schema.json").read_text())
SIZES = units.SIZES["tiny"]


@pytest.fixture(scope="module")
def model():
    return units.tiny_model(seed=0)


@pytest.fixture(scope="module")
def oaa_model():
    m = units.tiny_model(seed=0)
    att.install_oaa(m, m=SIZES["m"], seed=0)
    return m


def _check(rec, uid):
    jsonschema.validate(rec, SCHEMA)
    json.dumps(rec)                                 # must be JSON-serialisable for ctx.put_json
    assert rec["id"] == uid
    assert rec["passed"], rec["details"]


@pytest.fixture()
def real_model():
    return units.real_shape_model(seed=0, layers=2, vocab=4096)


@pytest.fixture(scope="module")
def qwen2_model():
    return units.tiny_model(seed=0, arch="qwen2")


# ---------------------------------------------------------------- Qwen3 architecture facts


def test_defaults_and_architecture(model, real_model):
    assert units.DEFAULT_MODEL == "Qwen/Qwen3-4B-Instruct-2507" and units.FALLBACK_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert type(model).__name__ == "Qwen3ForCausalLM"
    cfg = model.config
    assert cfg.num_attention_heads // cfg.num_key_value_heads == 4 and cfg.rope_parameters["rope_theta"] == 5e6
    mod = att.attention_modules(model)[0]
    assert mod.q_norm is not None and mod.k_norm is not None and mod.k_proj.bias is None
    rc = real_model.config
    assert (rc.num_attention_heads, rc.num_key_value_heads, rc.head_dim, rc.hidden_size) == (32, 8, 128, 2560)
    assert real_model.config._attn_implementation == att.OPA_ATTN


def test_patch_keeps_norm_then_rope_order(model):
    """The OAA function is called with the stock module's post-q/k-norm, post-RoPE tensors, so its output equals the stock
    eager and sdpa attention up to kernel rounding (randomised norm gains would expose a wrong order)."""
    T = 40
    ids = units._ids(model, T, 21, "cpu")
    ours = units._logits(model, ids)
    with att.attn_implementation(model, "eager"):
        eager = units._logits(model, ids)
    assert units._maxdiff(ours, eager) < 1e-4
    # the gains are really non-trivial: perturbing k_norm changes the logits
    mod = att.attention_modules(model)[0]
    old = mod.k_norm.weight.detach().clone()
    try:
        with torch.no_grad():
            mod.k_norm.weight.mul_(1.5)
        assert units._maxdiff(units._logits(model, ids), ours) > 1e-3
    finally:
        with torch.no_grad():
            mod.k_norm.weight.copy_(old)


# ---------------------------------------------------------------- U1-U5


def test_u1_reindex(model):
    rec = units.u1_reindex(model, SIZES)
    _check(rec, "U1")
    assert rec["metrics"]["fp32_max_abs_err"] <= 1e-5 and rec["metrics"]["reported_literal_1e5_abs_pass"]
    assert rec["metrics"]["definition"].startswith("U1 two-part definition, deviation DEV-U1")
    assert rec["metrics"]["part_a_float64_pass"] and rec["metrics"]["part_b_fp32_ulps_pass"]
    assert rec["metrics"]["float64_max_abs_err"] <= 1e-12
    assert rec["metrics"]["layout_check_max_rel_err"] == 0.0


def test_u1_verdict_two_part_definition():
    ulp = float(np.spacing(np.float32(298.4)))
    assert abs(ulp - 3.05e-5) < 1e-7
    assert units.u1_verdict(2 * ulp, 1.1e-13, 298.4)["passed"]            # the observed case: 2 ulp, > 1e-5 absolute
    assert units.u1_verdict(4 * ulp, 1.1e-13, 298.4)["passed"]
    assert not units.u1_verdict(5 * ulp, 1.1e-13, 298.4)["passed"]
    assert not units.u1_verdict(2 * ulp, 1e-9, 298.4)["passed"]            # float64 error 1e-9 fails part (a)
    assert not units.u1_verdict(ulp, 1e-13, 298.4, layout_ok=False)["passed"]


def test_u2_virtual_slot(model):
    rec = units.u2_virtual_slot(model, SIZES)
    _check(rec, "U2")
    assert rec["metrics"]["max_abs_logit_diff"] == 0.0


def test_u3_fresh_frame(model):
    rec = units.u3_fresh_frame(model, SIZES)
    _check(rec, "U3")
    assert all(v["max_abs_logit_diff"] == 0.0 for v in rec["metrics"].values())


def test_u4_oaa_identity(oaa_model):
    rec = units.u4_oaa_identity(oaa_model, SIZES)
    _check(rec, "U4")
    assert rec["metrics"]["max_abs_logit_diff"] == 0.0
    assert rec["metrics"]["live_check_max_abs_logit_diff"] > 1e-3


def test_u5_geometry():
    _check(units.u5_geometry(), "U5")


def test_u1_hook_sees_k_norm_output(model):
    """U1's layout sub-check compares the cached keys with RoPE of the captured pre-RoPE keys: it passes only if the
    capture point is the k_norm output (a k_proj capture would differ by the norm)."""
    rec = units.u1_reindex(model, SIZES)
    assert rec["metrics"]["layout_check_max_rel_err"] == 0.0
    mods = att.attention_modules(model)
    captured = {}
    h = mods[0].k_proj.register_forward_hook(lambda m, a, o: captured.setdefault("kproj", o.detach()))
    h2 = mods[0].k_norm.register_forward_hook(lambda m, a, o: captured.setdefault("knorm", o.detach()))
    try:
        units._logits(model, units._ids(model, 8, 0, "cpu"))
    finally:
        h.remove(); h2.remove()
    hd = mods[0].head_dim
    assert not torch.allclose(captured["kproj"].view(1, 8, -1, hd), captured["knorm"].view(1, 8, -1, hd))


def test_real_shape_u1_to_u5(real_model):
    sizes = units.SIZES["real"]
    recs = units.run_all(real_model, sizes)
    for r in recs:
        _check(r, r["id"])
    assert recs[0]["metrics"]["fp32_max_abs_err"] <= 1e-5
    assert recs[1]["metrics"]["max_abs_logit_diff"] == 0.0 and recs[1]["bf16"]["metrics"]["max_abs_logit_diff"] == 0.0
    assert all(v["max_abs_logit_diff"] == 0.0 for v in recs[2]["metrics"].values())
    assert all(v["max_abs_logit_diff"] == 0.0 for v in recs[2]["bf16"]["metrics"].values())
    assert recs[3]["metrics"]["max_abs_logit_diff"] == 0.0 and recs[3]["bf16"]["metrics"]["max_abs_logit_diff"] == 0.0
    assert recs[3]["metrics"]["live_check_max_abs_logit_diff"] > 1e-3
    assert all(p.A.dtype == torch.float32 for _, p in att.oaa_parameters(real_model))


def test_run_all_leaves_rotary_buffers_and_oaa_params_untouched():
    m = units.tiny_model(seed=4)
    inv = {n: b.detach().clone() for n, b in m.named_buffers()}
    assert inv and all(b.dtype == torch.float32 for b in inv.values())
    recs = units.run_all(m, SIZES, bf16=True)
    for n, b in m.named_buffers():
        assert b.dtype == torch.float32 and torch.equal(b, inv[n]), n
    assert next(m.parameters()).dtype == torch.bfloat16
    for _, p in att.oaa_parameters(m):
        assert all(q.dtype == torch.float32 for q in p.parameters())
    # OAA params: rerun the fp32 U4 init on a fresh model with the same seed and compare bitwise
    ref = units.tiny_model(seed=4)
    units.u4_oaa_identity(ref, SIZES)
    for (_, p), (_, q) in zip(att.oaa_parameters(m), att.oaa_parameters(ref)):
        for (n1, a), (n2, b) in zip(p.named_parameters(), q.named_parameters()):
            assert torch.equal(a, b), n1
    assert all(r["passed"] for r in recs)


def test_state_changed_during_bf16_is_recorded(monkeypatch):
    m = units.tiny_model(seed=6)
    orig = units.u4_oaa_identity
    calls = {"n": 0}

    def tamper(*a, **k):
        calls["n"] += 1
        out = orig(*a, **k)
        if calls["n"] == 2:                      # the bf16 call: leave a change behind
            next(iter(att.oaa_parameters(m)))[1].lam.data += 1.0
        return out

    monkeypatch.setattr(units, "u4_oaa_identity", tamper)
    recs = units.run_all(m, SIZES, bf16=True)
    assert any("oaa:" in n for n in recs[3]["bf16"]["state_restored"]) and recs[3]["passed"] is False
    assert not next(iter(att.oaa_parameters(m)))[1].lam.any()                    # restored


class _CreateOnlyCtx:
    """Fake ctx whose put_json raises on a second write to the same path (the bucket is create-only)."""
    run_id = "B-units-test"

    def __init__(self):
        self.put, self.events = {}, []

    def put_json(self, rel, obj):
        if rel in self.put:
            raise FileExistsError(rel)
        self.put[rel] = json.loads(json.dumps(obj))
        return {"uri": rel}

    def log(self, e):
        self.events.append(e)


def test_main_writes_each_object_once_normal_path(monkeypatch):
    monkeypatch.setattr(units, "load_model", lambda name, device: units.tiny_model(seed=7))
    ctx = _CreateOnlyCtx()
    out = units.main({"model": "tiny", "sizes": SIZES}, ctx)
    assert sorted(ctx.put) == [f"tests/U{i}.json" for i in range(1, 6)] and out["metrics"]["all_passed"]
    assert all(ctx.put[f"tests/U{i}.json"]["bf16"]["passed"] for i in (2, 3, 4))


def test_main_writes_each_object_once_with_tamper(monkeypatch):
    m = units.tiny_model(seed=8)
    monkeypatch.setattr(units, "load_model", lambda name, device: m)
    orig = units.u4_oaa_identity
    calls = {"n": 0}

    def tamper(*a, **k):
        calls["n"] += 1
        out = orig(*a, **k)
        if calls["n"] == 2:
            next(iter(att.oaa_parameters(m)))[1].lam.data += 1.0
        return out

    monkeypatch.setattr(units, "u4_oaa_identity", tamper)
    ctx = _CreateOnlyCtx()
    out = units.main({"model": "tiny", "sizes": SIZES}, ctx)
    u4 = ctx.put["tests/U4.json"]
    assert u4["passed"] is False and u4["bf16"]["passed"] is False
    assert any("oaa:" in n for n in u4["bf16"]["state_restored"])
    assert ctx.put["tests/U2.json"]["passed"] is False            # state_restored fails all three bf16-pass records
    assert out["metrics"]["all_passed"] is False
    assert any(e["action"] == "bf16_state_restored" for e in ctx.events)


def test_bf16_failure_keeps_fp32_records(monkeypatch):
    m = units.tiny_model(seed=5)
    got = []
    orig = units.u3_fresh_frame
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:                      # the bf16 call
            raise RuntimeError("bf16 kernel missing")
        return orig(*a, **k)

    monkeypatch.setattr(units, "u3_fresh_frame", boom)
    recs = units.run_all(m, SIZES, bf16=True, on_record=lambda r, final: got.append((r["id"], final, r["passed"])))
    assert [g for g in got if g[0] == "U3"][0] == ("U3", False, True)          # fp32 U3 emitted before the bf16 pass
    assert recs[2]["bf16"]["passed"] is False and "bf16 kernel missing" in recs[2]["bf16"]["error"]
    assert recs[2]["passed"] is False and got[-1][:2] == ("U4", True)


def test_u2_negative_control_changes_logits(model):
    rec = units.u2_virtual_slot(model, SIZES)
    assert rec["metrics"]["negative_control_M1_max_abs_logit_diff"] > 1e-3


def test_bf16_pass_exact_zero_on_tiny_qwen3():
    m = units.tiny_model(seed=3)
    recs = units.run_all(m, SIZES, bf16=True)
    for r in recs[1:4]:
        assert r["bf16"]["passed"] and r["passed"]
    assert recs[1]["bf16"]["metrics"]["max_abs_logit_diff"] == 0.0
    assert recs[3]["bf16"]["metrics"]["live_check_max_abs_logit_diff"] > 1e-3


def test_qwen2_fallback_path(qwen2_model):
    """The Qwen2.5-1.5B fallback keeps working: tiny Qwen2 (k_proj bias, GQA 2:1) passes U1-U5."""
    recs = units.run_all(qwen2_model, SIZES, bf16=False)
    for r in recs:
        _check(r, r["id"])
    assert not hasattr(att.attention_modules(qwen2_model)[0], "k_norm")


def test_main_fallback_switch(monkeypatch):
    seen = {}
    monkeypatch.setattr(units, "load_model", lambda name, device: seen.setdefault("name", name) and units.tiny_model())

    class Ctx:
        run_id = "r"

        def put_json(self, rel, obj):
            return {}

        def log(self, e):
            pass

    units.main({"fallback": True, "sizes": SIZES}, Ctx())
    assert seen["name"] == units.FALLBACK_MODEL


def test_main_entrypoint_writes_records(monkeypatch):
    """main(args, ctx) with a fake ctx and the tiny model in place of the download."""
    monkeypatch.setattr(units, "load_model", lambda name, device: units.tiny_model())

    class Ctx:
        run_id = "B-units-none-u1-u5-dev-s0-test"
        put, events = {}, []

        def put_json(self, rel, obj):
            self.put[rel] = json.loads(json.dumps(obj))
            return {"uri": rel}

        def log(self, e):
            self.events.append(e)

    ctx = Ctx()
    out = units.main({"model": "tiny", "sizes": SIZES}, ctx)
    assert sorted(ctx.put) == [f"tests/U{i}.json" for i in range(1, 6)]          # no stage object anywhere
    stages = [e for e in ctx.events if e["action"] == "unit_test_fp32_stage"]
    assert sorted(e["test"] for e in stages) == ["U2", "U3", "U4"] and all(e["stage"] == "fp32_only" for e in stages)
    for rel, rec in ctx.put.items():
        jsonschema.validate(rec, SCHEMA)
        assert rec["run_id"] == ctx.run_id
    assert out["metrics"]["all_passed"] is True
    assert all(ctx.put[f"tests/U{i}.json"]["bf16"]["passed"] for i in (2, 3, 4))
    assert {e["action"] for e in ctx.events} == {"unit_tests_start", "unit_test", "unit_test_fp32_stage", "unit_tests_end"}
    assert sum(e["action"] == "unit_test" for e in ctx.events) == 5
    json.dumps(out)


# ---------------------------------------------------------------- position maps and masks


def test_position_maps():
    B, r = 8, 2
    segs = [att.Segment(5, B, slot=1), att.Segment(2, r + B, slot=0, prefix_len=r, prefix_origin=1),
            att.Segment(att.NULL_ORIGIN, 3)]
    m1 = att.position_ids(segs, "M1", B)
    assert m1[:B].tolist() == list(range(40, 48))
    assert m1[B:B + r + B].tolist() == list(range(16 - r, 16 + B))       # prefix right before origin 2's frame
    assert m1[-3:].tolist() == [24, 25, 26]                               # unanchored tail continues
    m2 = att.position_ids(segs, "M2", B)
    assert m2[:B].tolist() == list(range(8, 16)) and m2[B:B + r + B].tolist() == list(range(-r, B))
    m5 = att.position_ids(segs, "M5")
    assert m5[:B].tolist() == list(range(B))
    assert m5[B:B + r + B].tolist() == list(range(r + B))                 # red dot 0..r-1, origin r..r+B-1
    assert m5[-3:].tolist() == [0, 1, 2]
    org = att.token_origins(segs)
    assert org[B:B + r].tolist() == [1] * r and org[B + r:B + r + B].tolist() == [2] * B and org[-1] == -1


def test_explicit_mask_bypasses_packed_sequence_detection(model):
    """Restarting position ids without an explicit mask make transformers block attention across segments; the
    explicit causal mask keeps co-loaded origins visible, and mode='segment' reproduces the isolated encoding."""
    segs = [att.Segment(1, 12), att.Segment(2, 12), att.Segment(3, 6)]
    T = 30
    ids = units._ids(model, T, 5, "cpu")
    pos = att.position_ids(segs, "M5")[None]
    causal = units._logits(model, ids, **att.layout_inputs(segs, "M5"))
    seg_only = units._logits(model, ids, **att.layout_inputs(segs, "M5", mode="segment"))
    implicit = units._logits(model, ids, position_ids=pos)
    assert units._maxdiff(causal, seg_only) > 1e-4                       # the mask matters
    assert units._maxdiff(implicit, seg_only) < 1e-5                     # transformers' packed-sequence behaviour
    # isolated encoding of the last segment equals running it alone
    alone = units._logits(model, ids[:, 24:])
    assert units._maxdiff(seg_only[:, 24:], alone) < 1e-5


# ---------------------------------------------------------------- the patch is live and factorizes


def _window(model):
    lay, window, cur = units.u4_layout(SIZES["m"])
    tables = att.tables_from_layout(lay, sorted(set(window + [cur])), kind="A2")
    segs = [att.Segment(c, SIZES["B"]) for c in window] + [att.Segment(cur, SIZES["tail"])]
    T = sum(s.length for s in segs)
    return tables, segs, units._ids(model, T, 11, "cpu")


def test_nonzero_lambda_and_A_change_logits(oaa_model):
    tables, segs, ids = _window(oaa_model)
    inputs, org = att.layout_inputs(segs, "M5"), att.token_origins(segs)
    units._set_oaa(oaa_model, 0.0, 0.0, 0.0)
    base = units._logits(oaa_model, ids, **inputs)
    for A_std, lam, g in ((0.0, 2.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 1.0)):
        units._set_oaa(oaa_model, A_std, lam, g, seed=1)
        with att.oaa_inputs(oaa_model, tables, org):
            out = units._logits(oaa_model, ids, **inputs)
        assert units._maxdiff(out, base) > 1e-3, (A_std, lam, g)
    # with OAA switched off again the model is back to M5 exactly
    units._set_oaa(oaa_model, 0.0, 0.0, 0.0)
    assert units._maxdiff(units._logits(oaa_model, ids, **inputs), base) == 0.0


def test_lambda_concentrates_mass_on_near_origins(oaa_model):
    """Struct-only OAA (V1): a larger lambda moves router mass to the query's own origin and, among the others, to the
    structurally near one (origin 2, same thread, d = 2 tree steps) away from the other thread (origin 5)."""
    tables, segs, ids = _window(oaa_model)
    inputs, org = att.layout_inputs(segs, "M5"), att.token_origins(segs)
    T = ids.shape[1]
    own, ratio = [], []
    row = {c: tables.rows(torch.tensor([c]))[0].item() for c in (2, 5, 7)}
    for lam in (0.0, 4.0):
        units._set_oaa(oaa_model, 0.0, lam, 0.0)
        with att.oaa_inputs(oaa_model, tables, org, router=False, value=False, record_layers={0}) as st:
            units._logits(oaa_model, ids, **inputs)
            res = att.factorization_check(st.records[0], head=0, query_pos=T - 1)
        pi = dict(zip(res["origins_rows"], res["router_mass"]))
        own.append(pi[row[7]])
        ratio.append(pi[row[2]] / pi[row[5]])
    units._set_oaa(oaa_model, 0.0, 0.0, 0.0)
    assert own[1] > own[0] and own[1] > 0.9
    assert ratio[1] > ratio[0]


def test_factorization_router_times_local(oaa_model):
    """Paper Theorem 6 on the model's own attention: a = pi * alpha for one head (U7-style, <= 1e-5 in fp32)."""
    tables, segs, ids = _window(oaa_model)
    inputs, org = att.layout_inputs(segs, "M5"), att.token_origins(segs)
    T = ids.shape[1]
    units._set_oaa(oaa_model, A_std=0.5, lam=2.0, g=1.0, seed=2)
    try:
        with att.oaa_inputs(oaa_model, tables, org, record_layers={1}) as st:
            units._logits(oaa_model, ids, **inputs)
            for qpos in (T - 1, T - SIZES["tail"]):
                res = att.factorization_check(st.records[1], head=1, query_pos=qpos)
                assert res["n_keys"] == qpos + 1
                assert len(res["origins_rows"]) == len(segs)
                assert res["bias_constant_within_origin_max_dev"] < 1e-5
                assert res["factorization_max_abs_err"] <= 1e-5
                assert res["sdpa_output_max_abs_err"] <= 1e-5
                assert abs(sum(res["router_mass"]) - 1.0) < 1e-9
    finally:
        units._set_oaa(oaa_model, 0.0, 0.0, 0.0)


def test_kv_cache_decode_matches_full_forward(oaa_model):
    """Prefill + one cached decoding step with OAA active equals the full forward (cache path keeps origins)."""
    tables, segs, ids = _window(oaa_model)
    inputs, org = att.layout_inputs(segs, "M5"), att.token_origins(segs)
    T = ids.shape[1]
    units._set_oaa(oaa_model, A_std=0.5, lam=2.0, g=1.0, seed=3)
    try:
        with torch.no_grad(), att.oaa_inputs(oaa_model, tables, org):
            full = oaa_model(input_ids=ids, **inputs).logits
        pos = inputs["position_ids"]
        pre_mask = inputs["attention_mask"]["full_attention"][:, :, :T - 1, :T - 1]
        with torch.no_grad():
            with att.oaa_inputs(oaa_model, tables, org[:T - 1]):
                out = oaa_model(input_ids=ids[:, :T - 1], position_ids=pos[:, :T - 1], use_cache=True,
                                attention_mask={"full_attention": pre_mask})
            step_mask = torch.ones(1, 1, 1, T, dtype=torch.bool)
            with att.oaa_inputs(oaa_model, tables, org, query_origins=org[T - 1:]):
                step = oaa_model(input_ids=ids[:, T - 1:], position_ids=pos[:, T - 1:], past_key_values=out.past_key_values,
                                 use_cache=True, attention_mask={"full_attention": step_mask}).logits
        assert units._maxdiff(step[:, -1], full[:, -1]) < 1e-5
    finally:
        units._set_oaa(oaa_model, 0.0, 0.0, 0.0)


def test_tables_distances_follow_geometry():
    lay, window, cur = units.u4_layout(8)
    ids = sorted(set(window + [cur]))
    t2 = att.tables_from_layout(lay, ids, kind="A2")
    t3 = att.tables_from_layout(lay, ids, kind="A3")
    i, j = ids.index(2), ids.index(cur)
    assert abs(float(t2.dist[i, j]) - lay.distance(2, cur, "A2")) < 1e-5     # fp32 LCA frame vs float64 layout
    assert float(t3.dist[i, j]) == 2.0                                          # 7 -> 3 -> 2: two tree steps
    assert torch.allclose(t2.dist, t2.dist.T, atol=1e-5) and float(t2.dist.diagonal().abs().max()) == 0.0
    assert t2.rows(torch.tensor([cur, 999, -1])).tolist() == [ids.index(cur), len(ids), len(ids)]
