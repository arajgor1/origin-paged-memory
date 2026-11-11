"""Track B unit tests U1-U5 (protocol 5.1), target model Qwen/Qwen3-4B-Instruct-2507.

Entry points
  opa.trackb.units:main(args, ctx)     Modal GPU job: loads args["model"] (default Qwen/Qwen3-4B-Instruct-2507; the logged
                                       fallback Qwen/Qwen2.5-1.5B-Instruct is selected with args {"fallback": true} or
                                       args {"model": FALLBACK_MODEL}), runs U1-U5 in float32 and then U2-U4 again in
                                       bfloat16, writes tests/U<i>.json through ctx.put_json, logs one event per test,
                                       returns {"metrics": {...}}.
  python -m opa.trackb.units --tiny    local check on a tiny, seeded, randomly initialised Qwen3Config (no download);
                                       --arch qwen2 gives the old Qwen2 tiny model, --real-shape the 2-layer
                                       random-weight model with the real Qwen3-4B head geometry.

Qwen3-4B architecture facts the code relies on (checked against transformers 5.22.2 modeling_qwen3.py)
  36 layers, 32 query heads, 8 KV heads (GQA 4:1), head_dim 128 (hidden 2560, so q_proj is 2560 -> 4096), full
  attention with standard RoPE in every layer (rope_theta 5e6, no scaling), no q/k/v/o bias, tied embeddings.
  Qwen3 applies a per-head RMSNorm to q and k BEFORE RoPE:  k = rope(k_norm(k_proj(h))). The OAA attention function
  receives query/key after both (it is called with the post-norm, post-RoPE, post-cache tensors), so the patch itself
  needs no change and cannot reorder norm and RoPE; what changes is U1, which must capture the pre-RoPE keys at the
  output of k_norm, not of k_proj.
  Qwen2.5 had a bias on k_proj (large, which made the 1e-5 absolute U1 tolerance a question); Qwen3 has none, and
  the keys entering RoPE are RMS-normalised times a learned gain, so |k| is bounded by sqrt(128) * max|k_norm.weight|
  per head. The size of that gain on the real checkpoint is not known before the GPU run; U1 reports max |k|, the
  error in ulps and the relative error either way.

Memory plan for U1-U5 on one L4 (24 GB), 4.02 B parameters (tied embedding 151,936 x 2560 = 0.39 B counted once)
  load      from_pretrained(dtype=bfloat16) on the host (8.0 GB host RAM), .to("cuda") (8.0 GB on the GPU), then
            model.float() converts parameter by parameter (the bf16 copy of each tensor is freed as its fp32 copy is
            made): resident 16.1 GB, transient peak about 16.1 GB + the largest tensor (0.39 B x 4 B = 1.6 GB).
  fp32 pass U1-U4: 16.1 GB weights + OAA parameters (36 layers x ~0.7 M params = 25 M params x 4 B = 0.1 GB) +
            activations. The longest sequence is U2/U4: 4 x 128 + 16 = 528 tokens: logits 528 x 151,936 x 4 B = 0.32 GB (a few copies,
            < 1.5 GB), dense float masks [1, 32, 528, 528] x 4 B = 36 MB per layer (freed per layer), hidden states
            and MLP (528 x 9728 x 4 B) well under 0.1 GB. Total about 18 GB, leaving ~6 GB headroom.
  bf16 pass U2-U4: `to_dtype` casts the model's PARAMETERS to bfloat16 in place (the fp32 copy of each tensor is
            freed as it is cast: 8.0 GB resident). Buffers (rotary inv_freq, original_inv_freq) and the OAA parameters are
            not cast: they stay fp32 and are checked bitwise unchanged after the pass. After run_all(bf16=True) the model
            STAYS in bf16 (casting back would only give bf16-rounded weights). Peak is far below the fp32 pass. This is
            the dtype the experiments use, so exact-zero logit identity is shown there too (same kernel, same dtype on
            both sides).
  Output records: only tests/U<i>.json is final. The fp32-stage copies of U2-U4 (made before the bf16 pass, stage
            "fp32_only") are never written under tests/ or any path a collector scans; they appear only as
            `unit_test_fp32_stage` events in the run's event log.
  U5 is CPU-side numpy/torch on a tiny tree. The whole job runs on an L4; a T4 (16 GB) could not hold the fp32 pass.

How "logits identical" is tested (U2, U3, U4)
  Both forward passes of a comparison go through the same attention function ("opa_oaa": SDPA with a dense float
  mask of shape [b, H, Tq, Tk]) with the same dtype, shapes and device, so the criterion is the literal one:
  max |logit difference| == 0.0. The reference ("unpatched model") is the stock model call -- no position ids, no
  mask, transformers builds both itself -- routed through that kernel. As information only, each record also gives
  the difference against transformers' own stock "sdpa" attention (a different kernel call: is_causal=True and no
  mask), which is not bitwise comparable; that number is not gated.

U1 (re-indexing, Lemma 1)
  GATE (deviation DEV-U1): (a) float64 max abs error <= 1e-10 and (b) fp32 max abs error <= 4 ulps of the largest
  |pre-RoPE key|, plus the layout check. The literal 1e-5 absolute tolerance is reported but not gated: on the real
  Qwen3-4B the largest key is 298.4, where one fp32 ulp is 3.05e-5 > 1e-5.
  The gated check uses the model's real pre-RoPE keys from every layer (captured in a forward pass at the input of RoPE: the k_norm output on Qwen3, the
  k_proj output on Qwen2),
  the model's inv_freq, attention_scaling and its own apply_rotary_pos_emb (rotate_half layout), with rotation
  angles p * inv_freq evaluated in float64 and all key arithmetic in float32: keys rotated to m and then by Delta
  must match keys rotated directly to m + Delta to <= 1e-5 (max abs, fp32). Angles are formed in float64 because
  transformers' rotary embedding forms p * inv_freq in float32, whose rounding (about ulp(p)/2 radians, e.g. 1e-3
  rad at p = 3e4) is an error of the angle, not of Lemma 1; that path is reported separately, ungated. A layout
  sub-check requires the keys the model actually cached to equal apply_rotary_pos_emb(pre-RoPE key, the model's own
  cos/sin) -- this proves the hook sees the real keys and the layout is the model's.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import torch

from opa.trackb import attention as att

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # logged fallback (decisions.md), not the default
U1_TOL = 1e-5                 # the literal protocol 5.1 tolerance: reported, no longer the gate (DEV-U1)
U1_F64_TOL = 1e-10            # U1 (a): float64 max abs error (algorithmic exactness)
U1_FP32_ULPS = 4.0            # U1 (b): fp32 max abs error in ulps of the largest |pre-RoPE key| (machine precision)
U1_DEFINITION = "U1 two-part definition, deviation DEV-U1, see state/decisions.md"

SIZES = {   # per-test sizes; "real" is used for the Modal run on the 4B model
    "tiny": {"B": 16, "n_origins": 4, "tail": 8, "u1_T": 32, "u1_m0": 2560, "u3_len": 48, "red_dot": 4, "m": 8},
    "real": {"B": 128, "n_origins": 4, "tail": 16, "u1_T": 64, "u1_m0": 2560, "u3_len": 256, "red_dot": 8, "m": 32},
}
U1_DELTAS = (-2560, 1, 511, 4096, 30000, 131072)    # -2560: archival position 5*512 + l back to the fresh frame l


# ============================================================================ helpers


def tiny_model(seed: int = 0, device="cpu", arch: str = "qwen3", vocab: int = 128):
    """A tiny randomly initialised model, float32, eval mode, already switched to the "opa_oaa" attention.

    arch "qwen3": Qwen3ForCausalLM scaled down with the real GQA ratio (8 query heads : 2 KV groups = 4:1, as 32:8),
    per-head q_norm / k_norm, no attention bias, tied embeddings and the large rope_theta (5e6) of Qwen3-4B.
    arch "qwen2": the old Qwen2 tiny model (GQA 2:1, k_proj bias), kept for the Qwen2.5-1.5B fallback path.
    The q_norm / k_norm gains are randomised (the default init of ones would hide an ordering error between the norm
    and RoPE)."""
    torch.manual_seed(seed)
    if arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=16, max_position_embeddings=40960,
                          rope_theta=5e6, tie_word_embeddings=True, initializer_range=0.1, attention_bias=False)
        model = Qwen3ForCausalLM(cfg)
        _randomise_qk_norm(model, seed)
    elif arch == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768,
                          rope_theta=1e6, tie_word_embeddings=True, initializer_range=0.1)
        model = Qwen2ForCausalLM(cfg)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model = model.to(device=device, dtype=torch.float32).eval()
    att.use_opa_attention(model)
    return model


def _randomise_qk_norm(model, seed: int) -> None:
    g = torch.Generator().manual_seed(seed + 1000)
    with torch.no_grad():
        for mod in att.attention_modules(model):
            for nm in ("q_norm", "k_norm"):
                n = getattr(mod, nm, None)
                if n is not None:
                    n.weight.copy_(0.5 + torch.rand(n.weight.shape, generator=g) * 2.0)


def real_shape_model(seed: int = 0, layers: int = 2, vocab: int = 4096, device="cpu"):
    """Random weights with the real Qwen3-4B attention geometry (hidden 2560, 32 query heads, 8 KV heads, head_dim 128,
    intermediate 9728, rope_theta 5e6, q/k norm, no bias), only `layers` layers and a small vocabulary so that it fits
    a CPU test. float32, eval mode, "opa_oaa" attention."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=2560, intermediate_size=9728, num_hidden_layers=layers,
                      num_attention_heads=32, num_key_value_heads=8, head_dim=128, max_position_embeddings=262144,
                      rope_theta=5e6, tie_word_embeddings=True, attention_bias=False, initializer_range=0.02)
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg)
    _randomise_qk_norm(model, seed)
    model = model.to(device=device, dtype=torch.float32).eval()
    att.use_opa_attention(model)
    return model


def load_model(name: str, device, dtype=torch.float32):
    """Checkpoint -> `device` -> `dtype`. Loaded as bf16 (the checkpoint dtype; 8 GB host RAM for the 4B model), moved to
    the device, then upcast there parameter by parameter, so host RAM never holds the fp32 copy."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model = model.to(dtype).eval()
    att.use_opa_attention(model)
    return model


def to_dtype(model, dtype):
    """Cast the model's parameters in place, except the OAA parameters (kept float32, untouched). Buffers (rotary
    inv_freq / original_inv_freq) are not cast: they stay float32. Note a float32 -> bf16 cast is lossy; casting back
    does not recover the fp32 weights."""
    skip = {id(q) for _, p in att.oaa_parameters(model) for q in p.parameters()}
    for prm in model.parameters():
        if id(prm) not in skip and prm.is_floating_point():
            prm.data = prm.data.to(dtype)
    return model


def _ids(model, T: int, seed: int, device):
    g = torch.Generator().manual_seed(seed)
    hi = min(int(model.config.vocab_size), 150000)        # stay below Qwen's special-token range
    return torch.randint(0, hi, (1, T), generator=g).to(device)


@torch.no_grad()
def _logits(model, ids, **kw):
    return model(input_ids=ids, use_cache=False, **kw).logits.float()


def _maxdiff(a, b) -> float:
    return float((a - b).abs().max())


def _record(uid, name, passed, details, run_id, metrics, criterion) -> dict:
    return {"id": uid, "name": name, "passed": bool(passed), "details": details, "run_id": run_id,
            "criterion": criterion, "metrics": metrics}


def _stock_sdpa_diff(model, ids, ref) -> float:
    with att.attn_implementation(model, "sdpa"):
        return _maxdiff(_logits(model, ids), ref)


def _apply_rope_fn(model):
    mod = sys.modules[type(att.attention_modules(model)[0]).__module__]
    return mod.apply_rotary_pos_emb


# ============================================================================ U1


def u1_verdict(worst32: float, worst64: float, kmax: float, layout_ok: bool = True) -> dict:
    """U1 gate (DEV-U1): (a) float64 error <= 1e-10, (b) fp32 error <= 4 ulps of the largest |pre-RoPE key|."""
    ulp = float(np.spacing(np.float32(kmax)))
    a = worst64 <= U1_F64_TOL
    b = worst32 <= U1_FP32_ULPS * ulp
    return {"a": bool(a), "b": bool(b), "passed": bool(a and b and layout_ok), "ulp": ulp}


@torch.no_grad()
def u1_reindex(model, sizes, run_id="local", seed=0, deltas=U1_DELTAS):
    device = next(model.parameters()).device
    T, m0 = sizes["u1_T"], sizes["u1_m0"]
    ids = _ids(model, T, seed, device)
    mods = att.attention_modules(model)
    pre = {}
    # pre-RoPE keys = the input of apply_rotary_pos_emb: k_norm(k_proj(h)) on Qwen3, k_proj(h) on Qwen2
    hooks = [(getattr(mod, "k_norm", None) or mod.k_proj).register_forward_hook(
        lambda mm, a, out, i=i: pre.__setitem__(i, out.detach())) for i, mod in enumerate(mods)]
    try:
        inputs = att.layout_inputs([att.Segment(att.NULL_ORIGIN, T, start=m0)], "M1", B=1, device=device)
        out = model(input_ids=ids, use_cache=True, **inputs)
    finally:
        for h in hooks:
            h.remove()
    cache = out.past_key_values
    rope = _apply_rope_fn(model)
    remb = (model.model if hasattr(model, "model") else model).rotary_emb
    inv_freq, scale = remb.inv_freq.detach(), float(remb.attention_scaling)

    def cs64(pos, dtype):
        ang = pos.to(torch.float64)[..., None] * inv_freq.to(torch.float64)
        emb = torch.cat([ang, ang], -1)
        return (emb.cos() * scale).to(dtype), (emb.sin() * scale).to(dtype)

    def cs_hf(pos):
        return remb(torch.zeros(1, dtype=torch.float32, device=device), pos)

    def rot(k, cs):
        return rope(k, k, cs[0], cs[1])[1]

    pos_m = (m0 + torch.arange(T, device=device))[None]
    worst32, worst64, worst_hf, worst_layout_rel, kmax = 0.0, 0.0, 0.0, 0.0, 0.0
    per_delta = {}
    for li, mod in enumerate(mods):
        hd = mod.head_dim
        k_pre = pre[li].view(1, T, -1, hd).transpose(1, 2).to(torch.float32)        # [1, G, T, d]
        kmax = max(kmax, float(k_pre.abs().max()))
        cached = cache.layers[li].keys if hasattr(cache, "layers") else cache.key_cache[li]
        layout_ref = rot(k_pre, cs_hf(pos_m))
        worst_layout_rel = max(worst_layout_rel, _maxdiff(cached.float(), layout_ref) / max(float(layout_ref.abs().max()), 1e-30))
        k_m32 = rot(k_pre, cs64(pos_m, torch.float32))
        k_m64 = rot(k_pre.double(), cs64(pos_m, torch.float64))
        for dlt in deltas:
            pos_d = torch.full_like(pos_m, dlt)
            e32 = _maxdiff(rot(k_m32, cs64(pos_d, torch.float32)), rot(k_pre, cs64(pos_m + dlt, torch.float32)))
            e64 = _maxdiff(rot(k_m64, cs64(pos_d, torch.float64)), rot(k_pre.double(), cs64(pos_m + dlt, torch.float64)))
            ehf = _maxdiff(rot(cached.float(), cs_hf(pos_d)), rot(k_pre, cs_hf(pos_m + dlt)))
            worst32, worst64, worst_hf = max(worst32, e32), max(worst64, e64), max(worst_hf, ehf)
            per_delta[str(dlt)] = max(per_delta.get(str(dlt), 0.0), e32)
    layout_ok = worst_layout_rel <= 1e-6
    ulp = float(np.spacing(np.float32(kmax)))
    v = u1_verdict(worst32, worst64, kmax, layout_ok)
    passed = v["passed"]
    metrics = {
        "definition": U1_DEFINITION, "part_a_float64_pass": v["a"], "part_b_fp32_ulps_pass": v["b"],
        "float64_tolerance": U1_F64_TOL, "fp32_tolerance_ulps": U1_FP32_ULPS,
        "reported_literal_1e5_abs_pass": worst32 <= U1_TOL, "reported_literal_tolerance": U1_TOL,
        "fp32_max_abs_err": worst32, "fp32_max_abs_err_by_delta": per_delta, "tolerance": U1_TOL,
        "float64_max_abs_err": worst64, "max_abs_pre_rope_key": kmax,
        "fp32_rel_err": worst32 / max(kmax, 1e-30),
        "fp32_ulp_at_max_key": ulp, "fp32_err_in_ulps_of_max_key": worst32 / ulp if ulp > 0 else None,
        "reported_relative_criterion_pass": worst32 / max(kmax, 1e-30) <= 1e-6,
        "layout_check_max_rel_err": worst_layout_rel, "layout_check_pass": layout_ok,
        "reported_hf_fp32_angle_path_max_abs_err": worst_hf,
        "positions": {"m": [m0, m0 + T - 1], "deltas": list(deltas), "max_position": m0 + T - 1 + max(deltas)},
        "layers": len(mods), "T": T,
    }
    details = (f"{U1_DEFINITION}. (a) float64 end to end max abs err {worst64:.2e} (<= {U1_F64_TOL:g}): "
               f"{'pass' if v['a'] else 'FAIL'}; (b) fp32 max abs err {worst32:.3e} = {worst32 / ulp if ulp > 0 else 0:.2f} ulp "
               f"of the largest |pre-RoPE key| {kmax:.4g} (ulp {ulp:.2e}; <= {U1_FP32_ULPS:g} ulp): "
               f"{'pass' if v['b'] else 'FAIL'}. Layout check (cached keys == model's own RoPE of the pre-RoPE key) max "
               f"rel err {worst_layout_rel:.1e}: {'pass' if layout_ok else 'FAIL'}. Lemma 1 on the model's real keys, "
               f"all {len(mods)} layers, m in [{m0}, {m0 + T - 1}], Delta in {list(deltas)}; angles in float64, rotations "
               f"in fp32. Reported, not gated: literal 1e-5 absolute criterion "
               f"{'pass' if worst32 <= U1_TOL else 'FAIL'} (relative error {worst32 / max(kmax, 1e-30):.1e}); with "
               f"transformers' float32 angle computation the error is {worst_hf:.3e}, dominated by rounding of p*inv_freq "
               f"at positions up to {m0 + T - 1 + max(deltas)}.")
    return _record("U1", "Re-indexing cached keys by Delta matches keys computed at m + Delta (Lemma 1)", passed,
                   details, run_id, metrics,
                   f"(a) float64 max abs err <= {U1_F64_TOL:g} and (b) fp32 max abs err <= {U1_FP32_ULPS:g} ulp of max "
                   f"|pre-RoPE key| and layout check; literal 1e-5 absolute reported only ({U1_DEFINITION})")


# ============================================================================ U2, U3


def _archival_segments(sizes, archive_ids=(3, 9, 14, 40)):
    B = sizes["B"]
    segs = [att.Segment(c, B, slot=i) for i, c in enumerate(sorted(archive_ids))]
    segs.append(att.Segment(att.NULL_ORIGIN, sizes["tail"]))
    return segs


@torch.no_grad()
def u2_virtual_slot(model, sizes, run_id="local", seed=1):
    device = next(model.parameters()).device
    segs = _archival_segments(sizes)
    T = sum(s.length for s in segs)
    pos = att.position_ids(segs, "M2", sizes["B"])
    ids = _ids(model, T, seed, device)
    ref = _logits(model, ids)
    got = _logits(model, ids, **att.layout_inputs(segs, "M2", sizes["B"], device=device))
    d = _maxdiff(got, ref)
    pos_ok = bool(torch.equal(pos, torch.arange(T)))
    stock = _stock_sdpa_diff(model, ids, ref)
    # negative control: the same window with archival M1 positions (origin * B + l, non-uniform gaps) must change the
    # logits, i.e. the exact-zero check above is able to fail
    neg = _maxdiff(_logits(model, ids, **att.layout_inputs(segs, "M1", sizes["B"], device=device)), ref)
    passed = pos_ok and d == 0.0 and neg > 0.0
    metrics = {"max_abs_logit_diff": d, "negative_control_M1_max_abs_logit_diff": neg, "positions_equal_arange": pos_ok,
               "T": T, "B": sizes["B"],
               "archive_ids": [s.origin for s in segs[:-1]], "slots": [s.slot for s in segs[:-1]],
               "reported_vs_stock_sdpa_kernel_max_abs_diff": stock}
    details = (f"{len(segs) - 1} origins (archive ids {metrics['archive_ids']}) in archival order in contiguous slots "
               f"0..{len(segs) - 2} of B={sizes['B']} plus a {sizes['tail']}-token tail; M2 position ids equal 0..T-1: "
               f"{pos_ok}; max |logit diff| vs the unpatched model = {d!r} (same kernel; criterion exact 0). "
               f"Negative control (archival M1 positions, non-uniform gaps) changes the logits by {neg:.3e} (> 0). "
               f"Ungated: vs transformers' stock sdpa kernel {stock:.2e}.")
    return _record("U2", "M2 virtual slots in archival order reproduce the unpatched model", passed, details, run_id,
                   metrics, "logits identical (max abs diff == 0, same attention kernel)")


@torch.no_grad()
def u3_fresh_frame(model, sizes, run_id="local", seed=2):
    device = next(model.parameters()).device
    L, r = sizes["u3_len"], sizes["red_dot"]
    out = {}
    for label, seg in (("single_origin", att.Segment(7, L)),
                       ("single_origin_red_dot_prefix", att.Segment(7, r + L, prefix_len=r, prefix_origin=6))):
        T = seg.length
        ids = _ids(model, T, seed, device)
        ref = _logits(model, ids)
        pos = att.position_ids([seg], "M5")
        got = _logits(model, ids, **att.layout_inputs([seg], "M5", device=device))
        out[label] = {"max_abs_logit_diff": _maxdiff(got, ref), "positions_equal_arange": bool(torch.equal(pos, torch.arange(T))),
                      "T": T, "reported_vs_stock_sdpa_kernel_max_abs_diff": _stock_sdpa_diff(model, ids, ref)}
    passed = all(v["max_abs_logit_diff"] == 0.0 and v["positions_equal_arange"] for v in out.values())
    d0, d1 = out["single_origin"]["max_abs_logit_diff"], out["single_origin_red_dot_prefix"]["max_abs_logit_diff"]
    details = (f"M5 fresh frame, one origin of {L} tokens: max |logit diff| vs the unpatched model = {d0!r}; with a "
               f"{r}-token red-dot prefix (positions 0..{r - 1}, origin {r}..{r + L - 1}): {d1!r} (same kernel; criterion "
               f"exact 0). Ungated: vs stock sdpa {out['single_origin']['reported_vs_stock_sdpa_kernel_max_abs_diff']:.2e}.")
    return _record("U3", "M5 with a single origin reproduces the unpatched model", passed, details, run_id, out,
                   "logits identical (max abs diff == 0, same attention kernel)")


# ============================================================================ U4


def u4_layout(m: int, seed: int = 0):
    """Two threads of three origins under the root, plus the current origin 7 (child of 3). Co-loaded window:
    origins 1, 5, 2, 6 (not in archival order) and the current origin's tail."""
    from opa.geometry import TreeLayout, spread_directions
    lay = TreeLayout(m=m, delta=1.0)
    d = spread_directions(2, m, seed=seed)
    lay.add_node(1, 0, d[0]); lay.add_node(2, 1, d[0]); lay.add_node(3, 2, d[0])
    lay.add_node(4, 0, d[1]); lay.add_node(5, 4, d[1]); lay.add_node(6, 5, d[1])
    lay.add_node(7, 3, d[0])
    return lay, [1, 5, 2, 6], 7


def _set_oaa(model, A_std=0.0, lam=0.0, g=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in att.oaa_parameters(model):
            p.A.copy_(torch.randn(p.A.shape, generator=gen) * A_std if A_std > 0 else torch.zeros_like(p.A))
            p.lam.fill_(lam)
            p.g.fill_(g)


@torch.no_grad()
def u4_oaa_identity(model, sizes, run_id="local", seed=3):
    device = next(model.parameters()).device
    st = getattr(model, "_opa_state", None) or att.install_oaa(model, m=sizes["m"], seed=seed)
    B, tail = sizes["B"], sizes["tail"]
    lay, window, cur = u4_layout(sizes["m"], seed=seed)
    tables = att.tables_from_layout(lay, sorted(set(window + [cur])), kind="A2")
    segs = [att.Segment(c, B) for c in window] + [att.Segment(cur, tail)]
    T = sum(s.length for s in segs)
    ids = _ids(model, T, seed, device)
    inputs = att.layout_inputs(segs, "M5", device=device)
    origins = att.token_origins(segs)
    ref = _logits(model, ids, **inputs)                               # M5, OAA inactive
    _set_oaa(model, 0.0, 0.0, 0.0)
    with att.oaa_inputs(model, tables, origins):
        got = _logits(model, ids, **inputs)
    d = _maxdiff(got, ref)
    _set_oaa(model, A_std=0.5, lam=2.0, g=1.0, seed=seed)             # live check: the patch must change the logits
    with att.oaa_inputs(model, tables, origins):
        live = _logits(model, ids, **inputs)
    _set_oaa(model, 0.0, 0.0, 0.0)
    d_live = _maxdiff(live, ref)
    wr_norm = float(sum(p.W_r.weight.norm() for _, p in att.oaa_parameters(model)))
    u_norm = float(sum(p.U.weight.norm() for _, p in att.oaa_parameters(model)))
    off = tables.dist[~torch.eye(len(tables.ids), dtype=torch.bool)]
    passed = d == 0.0 and d_live > 0.0
    metrics = {"max_abs_logit_diff": d, "live_check_max_abs_logit_diff": d_live,
               "live_check_settings": {"A_std": 0.5, "lambda": 2.0, "g": 1.0},
               "W_r_frobenius_sum": wr_norm, "U_frobenius_sum": u_norm,
               "distance_range": [float(off.min()), float(off.max())], "address_norm_max": float(tables.vec.norm(dim=1).max()),
               "window_origins": window, "current_origin": cur, "m": sizes["m"], "T": T, "B": B}
    details = (f"M5 window of {len(window)} origins x {B} tokens + {tail}-token tail of origin {cur}; OAA active with "
               f"A = 0, lambda = 0, g = 0 but random W_r and U (sum of norms {wr_norm:.3g}, {u_norm:.3g}), nonzero "
               f"addresses and hyperbolic LCA-frame distances in [{metrics['distance_range'][0]:.3f}, "
               f"{metrics['distance_range'][1]:.3f}]: max |logit diff| vs M5 = {d!r} (same kernel; criterion exact 0). "
               f"Live check (A ~ N(0, 0.5^2), lambda = 2, g = 1) changes logits by {d_live:.3e}.")
    return _record("U4", "OAA with A = 0, lambda = 0, g = 0 reproduces M5", passed, details, run_id, metrics,
                   "logits identical (max abs diff == 0, same attention kernel); live check diff > 0")


# ============================================================================ U5


def u5_geometry(run_id="local"):
    from opa.geometry import u5_report
    rep = u5_report()
    details = (f"opa.geometry.u5_report(): primitives fp32 max err {rep['a_primitives_fp32_max_abs_err']:.2e} (<= 1e-6), "
               f"LCA distances float64 max err {rep['b_lca_distance_float64_max_abs_err']:.2e} (<= 1e-10), fp32 LCA "
               f"distances: port err {rep['c_port_fp32_err_vs_truth']:.2e} vs reference's own "
               f"{rep['c_reference_fp32_err_vs_truth']:.2e} (port <= 2x reference + 1e-7).")
    return _record("U5", "Torch port of the Poincare functions and LCA-frame distances matches protocol_snippets_v2.py",
                   rep["passed"], details, run_id, rep, "see opa.geometry.u5_report docstring (a, b, c)")


# ============================================================================ runners


def environment(model, model_name: str) -> dict:
    import transformers
    dev = next(model.parameters()).device
    return {"model": model_name, "dtype": str(next(model.parameters()).dtype), "device": str(dev),
            "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "",
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "cuda": torch.version.cuda or "",
            "attn_implementation": att.OPA_ATTN,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if dev.type == "cuda" else False}


def run_all(model, sizes, run_id="local", bf16: bool = True, on_record=None) -> list[dict]:
    """U1-U5 in the model's dtype (float32). `on_record(rec, final)` is called for every fp32 record as soon as it
    exists (before any bf16 work, so a bf16 failure cannot lose them); final is False for U2-U4 when a bf16 pass
    follows. Then, if `bf16`, the model's parameters are cast to bfloat16 IN PLACE (the model stays bf16 afterwards;
    rotary buffers and OAA parameters stay float32 and are checked unchanged) and U2-U4 are run again. The bf16 result
    goes to rec["bf16"] ({dtype, passed, metrics, details}, or {dtype, passed: False, error} if it raised) and is
    ANDed into rec["passed"]. After all three bf16 tests and the buffer / OAA state check, `on_record(rec, True)` is
    called exactly once for each of U2-U4."""
    torch.use_deterministic_algorithms(False)
    recs, timing = [], {}
    for fn in (lambda: u1_reindex(model, sizes, run_id), lambda: u2_virtual_slot(model, sizes, run_id),
               lambda: u3_fresh_frame(model, sizes, run_id), lambda: u4_oaa_identity(model, sizes, run_id),
               lambda: u5_geometry(run_id)):
        t = time.time()
        rec = fn()
        rec["seconds"] = round(time.time() - t, 3)
        recs.append(rec)
        if on_record:
            on_record(rec, final=not (bf16 and rec["id"] in ("U2", "U3", "U4")))
    if bf16:
        before = {n: b.detach().clone() for n, b in model.named_buffers()}
        opa_before = {(i, n): q.detach().clone() for i, p in att.oaa_parameters(model) for n, q in p.named_parameters()}
        to_dtype(model, torch.bfloat16)
        for rec, fn in ((recs[1], lambda: u2_virtual_slot(model, sizes, run_id)),
                        (recs[2], lambda: u3_fresh_frame(model, sizes, run_id)),
                        (recs[3], lambda: u4_oaa_identity(model, sizes, run_id))):
            t = time.time()
            try:
                r = fn()
                rec["bf16"] = {"dtype": "bfloat16", "passed": bool(r["passed"]), "metrics": r["metrics"],
                               "details": r["details"]}
            except Exception as e:           # a bf16 failure must not lose the fp32 records
                r = {"passed": False}
                rec["bf16"] = {"dtype": "bfloat16", "passed": False, "error": f"{type(e).__name__}: {e}"}
            rec["seconds"] = round(rec["seconds"] + time.time() - t, 3)
            rec["passed"] = bool(rec["passed"] and r["passed"])
            rec["details"] += (f" bfloat16 pass (parameters cast in place, OAA parameters and rotary buffers fp32): "
                               f"{'pass' if r['passed'] else 'FAIL'}, see rec['bf16'].")
        changed = [f"buffer:{n}" for n, b in model.named_buffers() if not torch.equal(b, before[n])]
        changed += [f"oaa:{i}.{n}" for i, p in att.oaa_parameters(model) for n, q in p.named_parameters()
                    if not torch.equal(q, opa_before[(i, n)])]
        if changed:                          # nothing should have touched them: restore, but say so
            for n, b in model.named_buffers():
                if f"buffer:{n}" in changed:
                    b.data = before[n]
            for i, p in att.oaa_parameters(model):
                for n, q in p.named_parameters():
                    if f"oaa:{i}.{n}" in changed:
                        q.data = opa_before[(i, n)]
            for rec in recs[1:4]:
                if "bf16" in rec:
                    rec["bf16"]["state_restored"] = changed
                    rec["bf16"]["passed"] = False
                    rec["passed"] = False
                    rec["details"] += f" State check FAILED: {changed} restored."
        if on_record:                        # each of U2-U4 is emitted exactly once as final (the bucket is create-only)
            for rec in recs[1:4]:
                on_record(rec, final=True)
    return recs


def main(args: dict, ctx) -> dict:
    """Modal GPU job entry point (jobs/p0_trackb_units.yaml). args: model (default DEFAULT_MODEL), fallback (bool: use
    FALLBACK_MODEL, the Qwen2.5-1.5B path), sizes (overrides), bf16 (default true)."""
    t0 = time.time()
    name = FALLBACK_MODEL if args.get("fallback") else args.get("model", DEFAULT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":                       # true fp32 matmuls (no TF32) for the fp32 comparisons
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ctx.log({"action": "unit_tests_start", "model": name, "device": str(device)})
    model = load_model(name, device)
    env = environment(model, name)
    if device.type == "cuda":
        env["gpu_mem_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    sizes = dict(SIZES["real"], **args.get("sizes", {}))
    metrics = {"env": env, "sizes": sizes}

    def emit(rec, final=True):
        rec["env"] = env
        if not final:
            # fp32-only stage of U2-U4: event log only, never a tests/ object (only tests/U<i>.json is final)
            ctx.log({"action": "unit_test_fp32_stage", "stage": "fp32_only", "test": rec["id"], "passed": rec["passed"],
                     "seconds": rec["seconds"], "model": name, "details": rec["details"],
                     "metrics_fp32": rec["metrics"]})
            return
        ctx.put_json(f"tests/{rec['id']}.json", rec)
        ctx.log({"action": "unit_test", "test": rec["id"], "passed": rec["passed"], "seconds": rec["seconds"],
                 "model": name, "details": rec["details"]})
        if rec.get("bf16", {}).get("state_restored"):
            ctx.log({"action": "bf16_state_restored", "test": rec["id"], "names": rec["bf16"]["state_restored"]})
        metrics[f"{rec['id']}_passed"] = rec["passed"]
        if rec["id"] == "U3":
            summary = {k: v["max_abs_logit_diff"] for k, v in rec["metrics"].items()}
        else:
            summary = {k: v for k, v in rec["metrics"].items() if not isinstance(v, (dict, list))}
        metrics[f"{rec['id']}_summary"] = summary

    run_all(model, sizes, ctx.run_id, bf16=bool(args.get("bf16", True)), on_record=emit)
    if device.type == "cuda":
        metrics["gpu_peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    metrics["all_passed"] = all(metrics[f"U{i}_passed"] for i in range(1, 6))
    metrics["seconds"] = round(time.time() - t0, 1)
    ctx.log({"action": "unit_tests_end", "all_passed": metrics["all_passed"], "seconds": metrics["seconds"]})
    return {"metrics": metrics}


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track B unit tests U1-U5")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on CPU (no download)")
    ap.add_argument("--arch", default="qwen3", choices=["qwen3", "qwen2"], help="architecture of the tiny model")
    ap.add_argument("--real-shape", action="store_true",
                    help="2-layer random weights with the real Qwen3-4B head geometry")
    ap.add_argument("--model", default=None, help="HF model id (downloads weights; normally run on Modal)")
    a = ap.parse_args(argv)
    if a.real_shape:
        model, sizes, name = real_shape_model(), SIZES["real"], "random-qwen3-4b-shape-2layers"
    elif a.tiny or not a.model:
        model, sizes, name = tiny_model(arch=a.arch), SIZES["tiny"], f"tiny-random-{a.arch}"
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, sizes, name = load_model(a.model, dev), SIZES["real"], a.model
    env = environment(model, name)                 # captured before run_all (which leaves the model in bf16)
    recs = run_all(model, sizes)
    print(json.dumps({"env": env, "results": recs}, indent=2, default=str))
    ok = all(r["passed"] for r in recs)
    print(chr(10).join(f"{r['id']}: {'PASS' if r['passed'] else 'FAIL'}  {r['details']}" for r in recs))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
