"""U6-U8 and the H7 measurements (opa.trackb.units_h7) on CPU: closed-form precision sweeps against the numpy reference, the re-anchored
sweep through the kernel's AddressBookTorch, and U7/U8 on the tiny random Qwen3 (no download)."""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch

from opa.trackb import attention as att
from opa.trackb import units, units_h7 as H

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "bundle_spec" / "test.schema.json").read_text())
SIZES = units.SIZES["tiny"]


@pytest.fixture(scope="module")
def h7():
    return H.h7_measurements(chains=24, seed=0)


@pytest.fixture(scope="module")
def model():
    return units.tiny_model(seed=0)


def _check(rec, uid):
    jsonschema.validate(rec, SCHEMA)
    json.dumps(rec, allow_nan=False)
    assert rec["id"] == uid and rec["passed"], rec["details"]


# ---- the numpy reference (opa/reference/checks_v2.py section 3), copied here because that module is a script


def _round_bf16(v):
    v = np.asarray(v, dtype=np.float32)
    i = v.view(np.uint32).astype(np.uint64)
    i = ((i + 0x7FFF + ((i >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return float(i.view(np.float32).astype(np.float64))


REF_ROUNDERS = {"bfloat16": _round_bf16, "float16": lambda v: float(np.float16(v)), "float32": lambda v: float(np.float32(v))}


def _ref_step(n, delta, rnd):
    r0, r1 = rnd(np.tanh(n * delta / 2)), rnd(np.tanh((n + 1) * delta / 2))
    if r1 >= 1.0 or r0 >= 1.0:
        return np.nan
    return float(np.arccosh(1 + 2 * (r1 - r0) ** 2 / ((1 - r0 ** 2) * (1 - r1 ** 2))))


@pytest.mark.parametrize("name", ["bfloat16", "float16", "float32"])
def test_global_sweep_matches_the_reference_code(name):
    dt = H.DTYPES[name]
    for n in range(20):
        ref = _ref_step(n, 1.0, REF_ROUNDERS[name])
        ref_err = np.nan if np.isnan(ref) else abs(ref - 1.0)
        got = H.global_step_error(n, dt)
        assert (np.isnan(ref_err) and np.isnan(got)) or abs(got - ref_err) < 1e-9, (name, n, got, ref_err)
    steps = [_ref_step(n, 1.0, REF_ROUNDERS[name]) for n in range(40)]
    ref_depth = next((n for n, g in enumerate(steps) if np.isnan(g) or abs(g - 1.0) > 0.1), -1)
    assert H.global_fail_depth(dt) == ref_depth == H.PREDICTED_DEPTH[name]


def test_torch_rounding_is_round_to_nearest_even_like_the_reference():
    rng = np.random.default_rng(0)
    for v in rng.uniform(0, 1, 500):
        assert H._round(v, torch.bfloat16) == _round_bf16(v)
        assert H._round(v, torch.float16) == float(np.float16(v))
        assert H._round(v, torch.float32) == float(np.float32(v))


def test_h7_keys_and_ranges(h7):
    assert h7["global_fail_depth_bf16"] == 4 and h7["global_fail_depth_fp16"] == 7 and h7["global_fail_depth_fp32"] == 16
    for short, name in (("bf16", "bfloat16"), ("fp16", "float16"), ("fp32", "float32")):
        lo, hi = H.H7_RANGE[name]
        assert lo <= h7[f"global_fail_depth_{short}"] <= hi
        assert abs(h7[f"global_fail_depth_{short}_vector"] - h7[f"global_fail_depth_{short}"]) <= 1      # real m-dim points agree within a step
    r = h7["reanchored_error_growth_ratio_depth40_vs_depth1_bf16"]
    assert 0.2 < r <= H.GROWTH_LIMIT
    band = h7["reanchored"]["bfloat16"]
    assert band["depth"][0] == 1 and band["depth"][-1] == 40 and len(band["mean"]) == 40 and band["chains"] == 24
    assert all(lo <= m <= hi for lo, m, hi in zip(band["lo"], band["mean"], band["hi"]))
    json.dumps(h7, allow_nan=False)                                                                  # nan replaced by null


def test_global_error_grows_while_reanchored_error_stays_small(h7):
    """Negative control: the dtype that fails at depth 4 in global coordinates has a re-anchored error far below the 0.1 threshold at
    every depth up to 40, so the measurement can tell the two apart."""
    g = h7["global"]["bfloat16"]["step_relative_error"]
    assert g[0] < 0.05 and (g[4] is None or g[4] > 0.1)
    assert max(h7["reanchored"]["bfloat16"]["hi"]) < 0.02
    assert h7["reanchored"]["bfloat16"]["mean"][0] > h7["reanchored"]["float32"]["mean"][0]       # rounding of stored offsets shows up


def test_h7_is_deterministic():
    a, b = H.h7_measurements(chains=8, seed=3), H.h7_measurements(chains=8, seed=3)
    assert a == b
    assert H.h7_measurements(chains=8, seed=4)["reanchored"]["bfloat16"]["mean"] != a["reanchored"]["bfloat16"]["mean"]


def test_u6_record_passes_and_fails_when_it_should(h7):
    _check(H.u6_precision(h7), "U6")
    bad = json.loads(json.dumps(h7))
    bad["global_fail_depth_bf16"] = 7                                      # three steps off the prediction
    assert H.u6_precision(bad)["passed"] is False
    bad = json.loads(json.dumps(h7))
    bad["reanchored_error_growth_ratio_depth40_vs_depth1_bf16"] = 3.0      # re-anchored error would be growing
    assert H.u6_precision(bad)["passed"] is False
    ok = json.loads(json.dumps(h7))
    ok["global_fail_depth_fp32"] = 17                                      # one step off is still within tolerance
    assert H.u6_precision(ok)["passed"] is True


# ---- U7, U8


def test_u7_factorization(model):
    rec = H.u7_factorization(model, SIZES)
    _check(rec, "U7")
    assert rec["metrics"]["factorization_max_abs_err"] <= 1e-5 and rec["metrics"]["sdpa_output_max_abs_err"] <= 1e-5
    assert rec["metrics"]["cases"] == 2 * 3 * 3 and rec["metrics"]["layers"] == [0, 1]


def test_u8_paging(model):
    rec = H.u8_paging(model, SIZES)
    _check(rec, "U8")
    m = rec["metrics"]
    assert m["tv_vs_evicted_router_mass_max_abs_err"] <= 1e-5 and m["paged_sdpa_output_max_abs_err"] <= 1e-5
    assert m["cases"] == 2 * 3 * 3 * 3 and m["paged_passes"] == 2 * 3                  # one real paged kernel pass per (layer, subset)
    assert m["inputs_max_abs_diff_between_passes"] <= 1e-6 and m["paged_weight_on_evicted_keys_max"] == 0.0
    assert m["evicted_router_mass_range"][1] > 1e-3                                    # not tested at zero only


def _setup_and_full(model, layers=(1,)):
    setup = H._oaa_setup(model, SIZES, 8)
    full = H._forward_records(model, setup, 8, layers)
    return setup, full


def test_kernel_eviction_hides_the_keys_only_in_the_chosen_layer(model):
    setup, full = _setup_and_full(model, (0, 1))
    T, tail = setup["T"], setup["tail"]
    ev = setup["window_rows"][:2]
    paged = H._paged_record(model, setup, 8, 1, ev)
    # layers before the paged one are untouched, so q/k/v of layer 1 coincide with the full pass
    for n in ("query", "key", "value"):
        assert torch.equal(paged[n], full[1][n])
    base_p, base_f = paged["base"][0, 0], full[1]["base"][0, 0]
    rows = full[1]["key_rows"][0]
    hidden = torch.isin(rows, torch.tensor(ev))
    assert bool((base_p[T - tail:, :][:, hidden] != 0).all())                           # tail rows cannot see evicted keys
    assert bool((base_p[:T - tail] == base_f[:T - tail]).all()) and bool((base_p[T - tail:][:, ~hidden] == base_f[T - tail:][:, ~hidden]).all())
    # nothing evicted: bitwise the full pass
    none = H._forward_records(model, setup, 8, [1], evict_layers=[1], evict_rows=[])
    assert torch.equal(none[1]["output"], full[1]["output"]) and torch.equal(none[1]["base"], full[1]["base"])
    # the state is clean afterwards
    assert model._opa_state.evict_rows is None and model._opa_state.evict_layers == set()


def test_paging_check_from_two_real_passes(model):
    setup, full = _setup_and_full(model)
    T, tail = setup["T"], setup["tail"]
    ev = setup["window_rows"][:2]
    paged = H._paged_record(model, setup, 8, 1, ev)
    res = att.paging_check(full[1], paged, head=0, query_pos=T - 1, evicted_rows=ev)
    assert res["abs_err"] < 1e-9 and 0 < res["evicted_router_mass"] < 1 and res["paged_weight_on_evicted_keys"] == 0.0
    assert res["inputs_max_abs_diff_between_passes"] == 0.0 and res["paged_sdpa_output_max_abs_err"] < 1e-5
    none = H._forward_records(model, setup, 8, [1], evict_layers=[1], evict_rows=[])
    zero = att.paging_check(full[1], none[1], 0, T - 1, [])
    assert zero["tv"] < 1e-12 and zero["evicted_router_mass"] == 0.0
    # claiming an eviction the kernel did not perform breaks the identity (it is not an algebraic tautology of one pass)
    wrong = att.paging_check(full[1], none[1], 0, T - 1, ev)
    assert wrong["abs_err"] > 1e-4 and abs(wrong["abs_err"] - wrong["evicted_router_mass"]) < 1e-9     # TV = 0 but mass evicted


def _u8_with(monkeypatch, model, corrupt):
    orig = H._paged_record

    def wrapped(model_, setup, seed, layer, evicted):
        return corrupt(orig(model_, setup, seed, layer, evicted), model_, setup, seed, layer, evicted)

    monkeypatch.setattr(H, "_paged_record", wrapped)
    return H.u8_paging(model, SIZES)


def test_u8_fails_when_the_paged_pass_output_is_corrupted(model, monkeypatch):
    def corrupt(rec, *a):
        rec["output"] = rec["output"] + 0.01                                              # the kernel output no longer equals a_paged @ v
        return rec

    rec = _u8_with(monkeypatch, model, corrupt)
    assert rec["passed"] is False and rec["metrics"]["paged_sdpa_output_max_abs_err"] > 1e-3


def test_u8_fails_when_the_kernel_evicts_something_else(model, monkeypatch):
    def corrupt(rec, model_, setup, seed, layer, evicted):                               # the kernel hides only one of the claimed origins
        return H._forward_records(model_, setup, seed, [layer], evict_layers=[layer], evict_rows=evicted[:1])[layer]

    rec = _u8_with(monkeypatch, model, corrupt)
    assert rec["passed"] is False and rec["metrics"]["tv_vs_evicted_router_mass_max_abs_err"] > 1e-5


def test_u8_fails_when_evicted_keys_stay_visible(model, monkeypatch):
    def corrupt(rec, model_, setup, seed, layer, evicted):                               # the recorded mask shows the evicted keys again
        T, tail = setup["T"], setup["tail"]
        hidden = torch.isin(rec["key_rows"][0], torch.tensor(evicted))
        causal = torch.ones(T, T).tril().bool()[T - tail:]
        base = rec["base"].clone()
        base[0, 0, T - tail:] = base[0, 0, T - tail:].masked_fill(causal & hidden[None, :], 0.0)
        rec["base"] = base
        return rec

    rec = _u8_with(monkeypatch, model, corrupt)
    assert rec["passed"] is False and rec["metrics"]["paged_weight_on_evicted_keys_max"] > 0


def test_h7_notes_state_the_pair_and_rounding_caveats(h7):
    assert "same-branch" in h7["notes"]["reanchored_pair"] and "cross-branch" in h7["notes"]["reanchored_pair"]
    assert "float16" in h7["notes"]["fp16_rounding"] and "0.1" in h7["notes"]["failure_threshold"]
    assert "same-branch" in H.__doc__ and "6 of 60,000" in H.__doc__


def test_main_entrypoint_writes_records_and_h7_keys(monkeypatch):
    monkeypatch.setattr(units, "load_model", lambda name, device: units.tiny_model())

    class Ctx:
        run_id = "B-units-u6u8-test"

        def __init__(self):
            self.put, self.events = {}, []

        def put_json(self, rel, obj):
            assert rel not in self.put
            self.put[rel] = json.loads(json.dumps(obj, allow_nan=False))
            return {"uri": rel}

        def log(self, e):
            self.events.append(e)

    ctx = Ctx()
    out = H.main({"model": "tiny", "sizes": SIZES, "chains": 8}, ctx)
    assert sorted(ctx.put) == ["h7.json", "tests/U6.json", "tests/U7.json", "tests/U8.json"]
    for k in ("tests/U6.json", "tests/U7.json", "tests/U8.json"):
        jsonschema.validate(ctx.put[k], SCHEMA)
        assert ctx.put[k]["run_id"] == ctx.run_id and ctx.put[k]["passed"] is True
    m = out["metrics"]
    assert m["all_passed"] is True and m["global_fail_depth_bf16"] == 4 and m["global_fail_depth_fp16"] == 7 and m["global_fail_depth_fp32"] == 16
    assert "reanchored_error_growth_ratio_depth40_vs_depth1_bf16" in m
    assert {e["action"] for e in ctx.events} == {"units_h7_start", "h7_sweeps", "unit_test", "units_h7_end"}
