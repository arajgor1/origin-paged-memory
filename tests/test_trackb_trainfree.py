"""Training-free Track B variants V0, V1, B2, B4 (opa.trackb.trainfree): masks, biases, task builders and scorers, ITT, split guard,
determinism, and the proof that the existing arms are unchanged. Tiny random Qwen3 on CPU, hash tokenizer, synthetic books."""
import json

import numpy as np
import pytest
import torch

from opa.trackb import arms as A
from opa.trackb import attention as att
from opa.trackb import data as D
from opa.trackb import metrics as M
from opa.trackb import trainfree as TF
from opa.trackb import units

VOCAB, B = 512, 24
CFG = dict(split="dev", seed0=900, n=3, C=4, probe_C=4, n_probe_windows=4, B=B, m=8, headers=True, seed=0, dtype="float32")


@pytest.fixture(scope="module")
def tk():
    return D.make_tokenizers("hash", VOCAB)


@pytest.fixture(scope="module")
def filler():
    return D.make_filler("synthetic", n_books=24, words_per_book=6000)


@pytest.fixture(scope="module")
def setup(tk, filler):
    model = units.tiny_model(seed=0, vocab=VOCAB)
    arch, sm = TF.build_setup(model, tk[1], tk[0], filler, {**TF.DEFAULTS, **CFG})
    return model, arch, sm


class Ctx:
    run_id = "B-trainfree-test"

    def __init__(self):
        self.put, self.events = {}, []

    def put_json(self, rel, obj):
        assert rel not in self.put, rel
        self.put[rel] = json.loads(json.dumps(obj, allow_nan=False))
        return {"uri": rel}

    def log(self, e):
        self.events.append(e)


# ---------------------------------------------------------------- arm definitions and masks


def test_arm_definitions():
    assert set(TF.VARIANTS) <= set(A.ARMS)
    v0, v1, b2, b4 = (A.ARMS[v] for v in TF.VARIANTS)
    assert not v0.oaa and v0.positions == "M5" and v0.mask == "causal" and not v0.shared_prefix
    assert v1.oaa and v1.struct and not v1.router and not v1.value and v1.positions == "M5"
    assert b2.shared_prefix and not b2.oaa and b4.mask == "graph" and not b4.oaa
    assert A.V1_LAMBDA == 2.0


def test_graph_mask_blocks_non_parents_sees_a_later_loaded_parent_and_the_tail_sees_everything():
    segs = [att.Segment(2, 3), att.Segment(1, 2), att.Segment(9, 2), att.Segment(10, 2), att.Segment(att.NULL_ORIGIN, 2)]
    m = A.graph_mask(segs)[0, 0]
    T = 11
    assert m.shape == (T, T) and m.dtype == torch.bool
    own2, own1, own9, own10, tail = slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 11)
    assert bool(m[own2, own1].all())                                 # origin 2's parent (1) is visible although it is loaded later
    assert not bool(m[own1, own2].any())                             # origin 1's parent is the root: nothing else visible
    assert not bool(m[own9, own2].any()) and not bool(m[own9, own1].any())            # 9 is the first of thread 1: no parent
    assert bool(m[own10, own9].all())                                # 10's parent is 9
    assert not bool(m[own10, own2].any())                           # non-parents are blocked
    assert bool(torch.equal(m[own2, own2], torch.ones(3, 3).tril().bool()))            # causal inside an origin
    assert bool(torch.equal(m[tail, :9], torch.ones(2, 9).bool())) and bool(m[10, 9]) and not bool(m[9, 10])   # tail sees all, causally
    assert A.archive_parent(1) == 0 and A.archive_parent(9) == 0 and A.archive_parent(10) == 9 and A.archive_parent(64) == 63


def test_b4_first_segment_logits_equal_v0_and_the_tail_differs(setup):
    model, arch, sm = setup
    ex = sm.make("lm", 5, 4, role="eval")
    first = ex.segments[0].length
    lv0, lb4 = [], []
    for name, store in (("V0", lv0), ("B4", lb4)):
        arm = A.ARMS[name]
        kw = A.example_kwargs(ex, arm, "cpu")
        with torch.no_grad():
            store.append(model(input_ids=ex.ids[None], use_cache=False, **kw).logits[0])
    assert torch.allclose(lv0[0][:first], lb4[0][:first], atol=1e-6)               # the first origin sees only itself in both masks
    assert float((lv0[0][-1] - lb4[0][-1]).abs().max()) > 1e-4                      # the tail reads a different context


@torch.no_grad()
def test_b2_shared_prefix_positions_and_no_origin_signal(setup, tk):
    model, arch, sm = setup
    ex = sm.make("attrib", 3, 4, role="eval")
    prefix = tk[1].encode(TF.SHARED_PREFIX_TEXT)
    p = len(prefix)
    e2 = A.with_shared_prefix(ex, prefix)
    assert e2.T == ex.T + p and e2.ans_start == ex.ans_start + p and e2.ids[:p].tolist() == prefix and e2.ids[p:].tolist() == ex.ids.tolist()
    pos = D.positions(e2, "M5")
    L = max(s.length for s in ex.segments if s.origin != att.NULL_ORIGIN)
    s0 = e2.segments[1].length
    assert pos[:p].tolist() == list(range(p)) and pos[p:p + s0].tolist() == list(range(p, p + s0))     # fresh frames start after the prefix
    assert int(pos[p + s0]) == p                                                                         # next origin restarts at p
    assert int(pos[e2.T - e2.segments[-1].length]) == p + L                                              # tail after the largest position
    assert (e2.origin_tok[:p] == att.NULL_ORIGIN).all() and torch.equal(e2.origin_tok[p:], ex.origin_tok)   # no origin signal in the prefix
    # with an empty prefix B2 is exactly V0
    e0 = A.with_shared_prefix(ex, [])
    l_b2 = A.forward_example(model, e0, A.ARMS["B2"], arch.tables)[1]
    l_v0 = A.forward_example(model, ex, A.ARMS["V0"], arch.tables)[1]
    assert torch.equal(l_b2, l_v0)


def test_v1_penalty_grows_with_tree_distance_and_is_exactly_the_bias(setup):
    model, arch, sm = setup
    ex = sm.make("probe", 7, 4, role="eval")
    tables = arch.tables
    st_lam = [float(p.lam.detach().mean()) for _, p in att.oaa_parameters(model)]
    assert st_lam == [2.0, 2.0]                                                       # fixed lambda = 2 on every head (build_setup)
    with torch.no_grad(), att.oaa_inputs(model, tables, ex.origin_tok, router=False, struct=True, value=False, record_layers={0}) as st:
        kw = A.example_kwargs(ex, A.ARMS["V1"], "cpu")
        model(input_ids=ex.ids[None], use_cache=False, **kw)
        bias = st.records[0]["bias"][0]                                              # [H, T, T] float32
        rows = st.records[0]["key_rows"][0]
    n = len(tables.ids)
    dist = torch.zeros(n + 1, n + 1)
    dist[:n, :n] = tables.dist
    q = torch.arange(ex.T)
    expect = -2.0 * dist[rows[q][:, None], rows[None, :]]
    assert torch.allclose(bias[0], expect, atol=1e-6) and torch.allclose(bias[3], expect, atol=1e-6)
    # bigger tree distance, bigger penalty: same thread < different thread, for a fixed query origin
    c0 = ex.window[0]
    same = [c for c in ex.window if c != c0 and arch.thread(c) == arch.thread(c0)]
    other = [c for c in ex.window if arch.thread(c) != arch.thread(c0)]
    d = lambda a, b: float(tables.dist[tables.ids.index(a), tables.ids.index(b)])
    assert all(d(c0, s) < d(c0, o) for s in same for o in other)
    chain = [1, 2, 3, 4]
    assert d(1, 2) < d(1, 3) < d(1, 4)                                               # along one thread the distance grows with the path
    assert d(1, 2) < d(1, 9)                                                         # and a neighbour is closer than another thread's first origin


@torch.no_grad()      # with grad enabled, the lambda parameter makes the mask require grad and SDPA picks a different kernel
def test_v1_with_lambda_zero_equals_v0_exactly(setup):
    model, arch, sm = setup
    ex = sm.make("lm", 4, 4, role="eval")
    units._set_oaa(model, A_std=0.0, lam=0.0, g=0.0)
    try:
        l1 = A.forward_example(model, ex, A.ARMS["V1"], arch.tables)[1]
        l0 = A.forward_example(model, ex, A.ARMS["V0"], arch.tables)[1]
    finally:
        units._set_oaa(model, A_std=0.0, lam=A.V1_LAMBDA, g=0.0)
    assert torch.equal(l1, l0)
    l1b = A.forward_example(model, ex, A.ARMS["V1"], arch.tables)[1]
    assert float((l1b - l0).abs().max()) > 1e-5                                       # lambda 2 does change the logits


@torch.no_grad()
def test_v0_equals_oaa_off_and_existing_arms_are_unchanged(setup):
    """Bitwise: forward_example for V0, V3 and B1 equals the direct pre-change construction (layout_inputs / oaa_inputs / stock kernel)."""
    model, arch, sm = setup
    ex = sm.make("attrib", 6, 4, role="eval")
    ids = ex.ids[None]
    sl = slice(ex.ans_start - 1, ex.ans_start - 1 + ex.ans_len)

    def direct_logits(h):
        return model.lm_head(h[0, sl]).float()

    kw = att.layout_inputs(ex.segments, "M5")
    with torch.no_grad():
        ref_v0 = direct_logits(model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state)
    assert torch.equal(A.forward_example(model, ex, A.ARMS["V0"], arch.tables)[1], ref_v0)
    # V3: router + value, live parameters
    units._set_oaa(model, A_std=0.3, lam=0.0, g=0.5, seed=1)
    try:
        with torch.no_grad(), att.oaa_inputs(model, arch.tables, ex.origin_tok, router=True, struct=False, value=True):
            ref_v3 = direct_logits(model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state)
        assert torch.equal(A.forward_example(model, ex, A.ARMS["V3"], arch.tables)[1], ref_v3)
        # the read-only output hook does not change V3
        calls = []
        att.PROBE_OUT = lambda module, out: calls.append(module.layer_idx)
        try:
            hooked = A.forward_example(model, ex, A.ARMS["V3"], arch.tables)[1]
        finally:
            att.PROBE_OUT = None
        assert torch.equal(hooked, ref_v3) and calls == [0, 1]
    finally:
        units._set_oaa(model, A_std=0.0, lam=A.V1_LAMBDA, g=0.0)
    # B1: stock kernel, contiguous positions
    pos = torch.arange(ex.T)[None]
    with att.attn_implementation(model, "sdpa"):
        with torch.no_grad():
            ref_b1 = direct_logits(model.model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state)
        got_b1 = A.forward_example(model, ex, A.ARMS["B1"], arch.tables)[1]
    assert torch.equal(got_b1, ref_b1)


# ---------------------------------------------------------------- tasks and scorers


def test_order_task_semantics(setup, tk):
    model, arch, sm = setup
    enc = tk[1]
    for seed in (900, 901, 902, 903):
        ex = TF.build_order(sm, seed, 4, "eval")
        a, b = ex.meta["first_origin"], ex.meta["second_origin"]
        assert a < b and a in ex.window and b in ex.window                           # first = lower archive id
        assert arch.thread(a) != arch.thread(b)                                       # the two origins belong to different threads
        tok_a, tok_b = TF._label_tokens(enc)
        assert ex.options == [tok_a, tok_b] and int(ex.targets[0]) == ex.options[ex.gold_option]
        assert (ex.gold_option == 0) == ex.meta["first_is_A"]
        sent = enc.encode(f" The setting of")                                         # the key is stated in exactly two origins
        pos = 0
        n_hits = 0
        for s in ex.segments[:-1]:
            seg = ex.ids[pos:pos + s.length].tolist()
            n_hits += any(seg[i:i + len(sent)] == sent for i in range(len(seg)))
            pos += s.length
        assert n_hits == 2
        assert ex.segments[-1].origin == att.NULL_ORIGIN and (ex.origin_tok[-ex.segments[-1].length:] == att.NULL_ORIGIN).all()


def test_confuse_task_semantics(setup, tk):
    model, arch, sm = setup
    enc = tk[1]
    for seed in (900, 901, 902):
        ex = TF.build_confuse(sm, seed, 4, "eval")
        m = ex.meta
        assert arch.thread(m["fact_origin"]) == m["target_thread"] == arch.thread(m["working_origin"])
        assert arch.thread(m["decoy_origin"]) == m["other_thread"] != m["target_thread"]
        assert m["fact_origin"] != m["working_origin"] and {m["fact_origin"], m["decoy_origin"], m["working_origin"]} <= set(ex.window)
        assert (ex.origin_tok[-(ex.T - sum(s.length for s in ex.segments[:-1])):] == m["working_origin"]).all()      # question tokens sit at the working origin
        assert ex.segments[-1].origin == att.NULL_ORIGIN                              # but the segment (positions, B4 mask) stays unanchored
        # the same entity at the same token offset in both threads' origins
        starts, pos = {}, 0
        for s in ex.segments[:-1]:
            starts[s.origin] = pos
            pos += s.length
        key = enc.encode(" The pressure limit of")
        for c in (m["fact_origin"], m["decoy_origin"]):
            seg = ex.ids[starts[c]:starts[c] + 1000].tolist()
            assert seg[m["offset"]:m["offset"] + len(key)] == key, c          # offset counted in the headered origin sequence
        assert int(ex.targets[0]) == ex.options[ex.gold_option]
        # the target value is the one in the target thread's origin: A/B text order follows target_first
        assert m["target_value"] != m["decoy_value"]


def test_working_origin_is_structurally_closer_to_the_target_fact(setup):
    model, arch, sm = setup
    ex = TF.build_confuse(sm, 905, 4, "eval")
    m = ex.meta
    d = lambda a, b: float(arch.tables.dist[arch.tables.ids.index(a), arch.tables.ids.index(b)])
    assert d(m["working_origin"], m["fact_origin"]) < d(m["working_origin"], m["decoy_origin"])


def test_scorers():
    ex = D.Example("attrib", torch.tensor([1, 2, 3]), torch.tensor([-1, -1, -1]), [], ans_start=2, ans_len=1, options=[7, 8, 9], gold_option=1)
    ex.ids[2] = 8
    lg = torch.zeros(1, 20)
    lg[0, 8] = 5.0
    r = TF.score_example(ex, lg, torch.tensor(1.0))
    assert r["correct"] == 1 and r["correct_greedy"] == 1 and r["pred_option"] == 1 and r["gold_option"] == 1
    lg = torch.zeros(1, 20)
    lg[0, 7] = 2.0
    lg[0, 15] = 9.0                                                                    # greedy token outside the options: rank still counts
    r = TF.score_example(ex, lg, torch.tensor(1.0))
    assert r["correct"] == 0 and r["correct_greedy"] == 0 and r["pred_option"] == 0
    pk = D.Example("passkey", torch.tensor([1, 5, 6]), torch.tensor([-1, -1, -1]), [], ans_start=1, ans_len=2)
    lg = torch.zeros(2, 10)
    lg[0, 5], lg[1, 6] = 1.0, 1.0
    assert TF.score_example(pk, lg, torch.tensor(0.0))["correct"] == 1
    lg[1, 6], lg[1, 2] = 0.0, 1.0
    assert TF.score_example(pk, lg, torch.tensor(0.0))["correct"] == 0
    lm = D.Example("lm", torch.tensor([1, 2, 3]), torch.tensor([-1, -1, -1]), [], ans_start=1, ans_len=2)
    assert TF.score_example(lm, torch.zeros(2, 5), torch.tensor(1.5)) == {"kind": "lm", "loss": 1.5, "n_tokens": 2}


# ---------------------------------------------------------------- metrics: linear probe, per layer/head


def test_linear_probe_recovers_a_planted_signal_and_is_at_chance_without_one():
    rng = np.random.default_rng(0)
    n, d, classes = 400, 8, 5
    y = rng.integers(0, classes, n)
    signal = torch.tensor(np.eye(classes)[y][:, :d].tolist() if d >= classes else [], dtype=torch.float32)
    X = torch.randn(1, n, d) * 0.3
    X[0, :, :classes] += signal
    noise = torch.randn(1, n, d)
    tr, te = slice(0, 300), slice(300, 400)
    good = M.linear_probe_accuracy({0: X[:, tr]}, y[tr], {0: X[:, te]}, y[te], classes)
    bad = M.linear_probe_accuracy({0: noise[:, tr]}, y[tr], {0: noise[:, te]}, y[te], classes)
    assert good["accuracy"][0][0] > 0.9 and bad["accuracy"][0][0] < 0.45 and abs(good["chance"] - 0.2) < 1e-12


# ---------------------------------------------------------------- ITT, split guard, determinism, end to end


def test_itt_failed_items_are_scored_wrong_and_kept(setup, tk, monkeypatch):
    model, arch, sm = setup
    orig = A.forward_example
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated OOM")
        return orig(*a, **k)

    monkeypatch.setattr(A, "forward_example", flaky)
    ctx = Ctx()
    recs, inputs = TF.eval_task(model, A.ARMS["V0"], "attrib", sm, arch.tables, tk[1].encode(TF.SHARED_PREFIX_TEXT), 4, 900, 4, "eval", ctx)
    assert len(recs) == 4 and [r["failed"] for r in recs] == [False, True, False, False]
    assert recs[1]["correct"] == 0 and "simulated OOM" in recs[1]["error"]
    s = TF.summarize_task("attrib", recs)
    assert s["n"] == 4 and s["n_failed"] == 1
    assert s["accuracy"] == pytest.approx(sum(r["correct"] for r in recs) / 4)             # the denominator includes the failed item
    assert any(e["action"] == "item_failed" for e in ctx.events)
    lm_recs = [{"failed": True, "correct": None}, {"failed": False, "loss": 2.0, "n_tokens": 10}]
    assert TF.summarize_task("lm", lm_recs)["n_failed"] == 1


def test_split_guard(tmp_path, monkeypatch):
    TF.check_split("dev", 900)
    with pytest.raises(ValueError):
        TF.check_split("dev", 100000)
    with pytest.raises(ValueError):
        TF.check_split("holdout", 0)
    monkeypatch.setattr(TF, "PROJECT_DIR", tmp_path)                                 # no prereg/PREREG_TAG.json
    with pytest.raises(TF.SplitRefused):
        TF.check_split("test", 100000, {})
    with pytest.raises(TF.SplitRefused):
        TF.check_split("test", 100000, {"prereg_tag": {"tag": "other", "commit": "abc"}})
    TF.check_split("test", 100000, {"prereg_tag": {"tag": "prereg-v2", "commit": "abc123"}})
    with pytest.raises(ValueError):
        TF.check_split("test", 5, {"prereg_tag": {"tag": "prereg-v2", "commit": "abc123"}})
    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "PREREG_TAG.json").write_text("{}")
    TF.check_split("test", 100000, {})
    # main refuses before loading anything
    monkeypatch.setattr(TF, "PROJECT_DIR", tmp_path / "none")
    from opa.trackb import units as U
    monkeypatch.setattr(U, "load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not be loaded")))
    with pytest.raises(TF.SplitRefused):
        TF.main({"split": "test", "seed0": 100000}, Ctx())


def _run_once(tk, filler, variants=("V0", "V1", "B2", "B4")):
    model = units.tiny_model(seed=0, vocab=VOCAB)
    ctx = Ctx()
    out = TF.run({**CFG, "variants": list(variants)}, model, tk[0], tk[1], filler, ctx)
    return out, ctx


def test_end_to_end_outputs_and_determinism(tk, filler):
    out, ctx = _run_once(tk, filler)
    m = ctx.put["metrics.json"]
    for v in TF.VARIANTS:
        for t in TF.TASKS:
            assert f"items/{v}_{t}.json" in ctx.put and len(ctx.put[f"items/{v}_{t}.json"]) == 3
            assert m["variants"][v]["tasks"][t]["n"] == 3 and m["variants"][v]["tasks"][t]["n_failed"] == 0
        pr = ctx.put[f"probe_{v}.json"]
        lh = pr["layer_head"]
        assert len(lh["own_origin_mass"]) == 2 and len(lh["own_origin_mass"][0]) == 8 and len(lh["router_entropy"][0]) == 8
        assert 0.0 <= pr["summary"]["own_origin_mass_fraction"] <= 1.0 and pr["n_ok"] == 4
        lp = pr["linear_probe"]
        assert lp["chance"] == pytest.approx(1 / 64) and all(0.0 <= a <= 1.0 for accs in lp["accuracy"].values() for a in accs)
        assert pr["summary"]["C"] == 4
    for t in TF.TASKS:
        assert f"inputs/{t}.json" in ctx.put
    # raw outputs per item
    one = ctx.put["items/V1_confuse.json"][0]
    assert {"seed", "correct", "option_logits", "greedy_tokens", "top5_first", "failed"} <= set(one) and one["seed"] == 900
    assert "V1_vs_V0_confuse" in m["contrasts_vs_V0"] and m["lambda_V1"] == 2.0 and m["predictions_to_read"]["V0_own_origin_mass_fraction"] is not None
    assert out["metrics"]["V0_passkey_accuracy"] is not None and out["metrics"]["B4_lm_ppl"] > 0
    # determinism
    out2, ctx2 = _run_once(tk, filler)
    strip = lambda d: {k: v for k, v in d.items() if k != "seconds"}
    assert strip(ctx.put["metrics.json"]) == strip(ctx2.put["metrics.json"])
    assert ctx.put["items/B4_attrib.json"] == ctx2.put["items/B4_attrib.json"]
    # V0 and B2 differ (the prefix changes the positions), V1 differs from V0 (the penalty is live), B4 differs (mask)
    ppl = {v: m["variants"][v]["tasks"]["lm"]["ppl"] for v in TF.VARIANTS}
    assert len({round(p, 6) for p in ppl.values()}) == 4


def test_main_entrypoint_with_tiny_model(tk, filler, monkeypatch):
    monkeypatch.setattr(units, "load_model", lambda name, device, dtype=torch.float32: units.tiny_model(seed=0, vocab=VOCAB))
    monkeypatch.setattr(D, "make_tokenizers", lambda spec, vocab=128: tk)
    monkeypatch.setattr(D, "make_filler", lambda *a, **k: filler)
    ctx = Ctx()
    out = TF.main({**CFG, "model": "tiny", "tokenizer": "hash", "variants": ["V0", "B4"], "tasks": ["passkey", "order"], "n_probe_windows": 2}, ctx)
    assert sorted(k for k in ctx.put if k.startswith("items/")) == ["items/B4_order.json", "items/B4_passkey.json", "items/V0_order.json", "items/V0_passkey.json"]
    assert "metrics.json" in ctx.put and out["metrics"]["split"] == "dev"


# ---- checker a9f38df fixes


class VarHeaderEnc(D.HashEncoder):
    """Header lengths differ between labels (characters of the label become separate tokens); everything else as the hash encoder."""

    def encode(self, text):
        if text.startswith("[") and "]" in text:
            label = text[1:text.index("]")]
            return [self.reserved + (ord(ch) % (self.vocab - self.reserved)) for ch in label] + [self.reserved + 1]
        return super().encode(text)


def test_task4_same_offset_holds_with_headers_of_different_lengths(filler):
    enc = VarHeaderEnc(VOCAB)
    ftok, _ = D.make_tokenizers("hash", VOCAB)
    arch = D.Archive(enc, m=8, with_dist=True, kind="A2")
    sm = D.Sampler(arch, filler, ftok, enc, B=B, headers=True)
    lens = set()
    key = enc.encode(" The pressure limit of")
    for seed in range(40):
        ex = TF.build_confuse(sm, 900 + seed, 4, "eval")
        m = ex.meta
        starts, pos = {}, 0
        for s in ex.segments[:-1]:
            starts[s.origin] = pos
            pos += s.length
        lens.add(len(sm._origin_tokens(m["fact_origin"], [])) - len(sm._origin_tokens(m["decoy_origin"], [])))
        for c in (m["fact_origin"], m["decoy_origin"]):
            seg = ex.ids[starts[c]:starts[c] + 1000].tolist()
            assert seg[m["offset"]:m["offset"] + len(key)] == key, (seed, c)       # identical position in both headered sequences
    assert len(lens) > 1                                                             # the test really exercised different header lengths


def test_task4_working_origin_is_adjacent_and_nearer_over_200_seeds(setup):
    model, arch, sm = setup
    ix = lambda c: arch.tables.ids.index(c)
    for seed in range(200):
        m = TF.build_confuse(sm, seed, 4, "eval").meta
        assert m["working_origin"] != m["fact_origin"] and abs(m["working_origin"] - m["fact_origin"]) == 1
        assert arch.thread(m["working_origin"]) == arch.thread(m["fact_origin"]) != arch.thread(m["decoy_origin"])
        assert float(arch.tables.dist[ix(m["working_origin"]), ix(m["fact_origin"])]) < float(arch.tables.dist[ix(m["working_origin"]), ix(m["decoy_origin"])])


def test_run_on_the_test_split_with_a_tag(tk, filler, monkeypatch, tmp_path):
    monkeypatch.setattr(TF, "PROJECT_DIR", tmp_path)                                 # no PREREG_TAG.json: the args tag is the evidence
    model = units.tiny_model(seed=0, vocab=VOCAB)
    ctx = Ctx()
    cfg = {**CFG, "split": "test", "seed0": 100000, "prereg_tag": {"tag": "prereg-v2", "commit": "abc123"}, "variants": ["V0", "B2"]}
    out = TF.run(cfg, model, tk[0], tk[1], filler, ctx)
    m = ctx.put["metrics.json"]
    assert m["split"] == "test" and out["metrics"]["split"] == "test"
    for v in ("V0", "B2"):
        for t in TF.TASKS:
            assert m["variants"][v]["tasks"][t]["n_failed"] == 0 and m["variants"][v]["tasks"][t]["n"] == 3
    assert ctx.put["items/V0_passkey.json"][0]["seed"] == 100000
    with pytest.raises(TF.SplitRefused):
        TF.run({**cfg, "prereg_tag": None}, units.tiny_model(seed=0, vocab=VOCAB), tk[0], tk[1], filler, Ctx())


def test_dev_seed_guard_counts_n():
    TF.check_split("dev", 900, None, 16)
    TF.check_split("dev", 99984, None, 16)
    with pytest.raises(ValueError):
        TF.check_split("dev", 99990, None, 16)                                      # 99990 + 16 - 1 reaches 100005


def test_engineering_stop_when_every_item_fails(tk, filler, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated CUDA error")

    monkeypatch.setattr(A, "forward_example", boom)
    ctx = Ctx()
    with pytest.raises(TF.EngineeringStop, match="all 3 items failed"):
        TF.run({**CFG, "variants": ["V0"]}, units.tiny_model(seed=0, vocab=VOCAB), tk[0], tk[1], filler, ctx)
    assert ctx.put["metrics.json"]["status"] == "engineering_stop" and ctx.put["metrics.json"]["task"] == "passkey"
    assert all(r["failed"] for r in ctx.put["items/V0_passkey.json"])


def test_item_records_carry_input_hashes_and_chance_is_one_over_c(tk, filler):
    out, ctx = _run_once(tk, filler, ("V0", "B2"))
    r0, r2 = ctx.put["items/V0_attrib.json"][0], ctx.put["items/B2_attrib.json"][0]
    assert len(r0["ids_sha256"]) == 64 and r0["ids_sha256"] == r2["ids_sha256"]               # the same shared input
    assert r0["variant_ids_sha256"] == r0["ids_sha256"] and r2["variant_ids_sha256"] != r2["ids_sha256"]   # B2 read a prefix as well
    m = ctx.put["metrics.json"]
    assert m["variants"]["V0"]["tasks"]["attrib"]["chance"] == pytest.approx(1 / CFG["C"])
    assert m["variants"]["V0"]["tasks"]["order"]["chance"] == 0.5
    assert out["metrics"]["V0_own_origin_mass_ratio_to_uniform"] > 0 and "B2_own_origin_mass_ratio_to_uniform" in out["metrics"]
    lp = ctx.put["probe_V0.json"]["linear_probe"]
    assert {"n_classes_seen", "majority_baseline", "test_labels_unseen_in_training", "n_classes"} <= set(lp) and 0 <= lp["majority_baseline"] <= 1


def _uniform_probe(ex, arch, allowed):
    """Run the probe on uniform attention (q = 0) over the keys `allowed` ([1, 1, T, T] bool)."""
    p = M.AttentionProbe(ex, arch, chunk=7)
    q = torch.zeros(1, 2, ex.T, 4)
    k = torch.randn(1, 2, ex.T, 4)
    mask = torch.zeros(1, 2, ex.T, ex.T).masked_fill(~allowed.expand(1, 2, ex.T, ex.T), torch.finfo(torch.float32).min)

    class Mod:
        layer_idx = 0

    p(Mod(), q, k, mask, 1.0)
    return p.summary()


def test_probe_baselines_use_the_actual_visibility_mask(setup):
    """Uniform attention over the keys a variant can see has ratio 1 for every baseline, under B4's mask as under the causal one."""
    model, arch, sm = setup
    ex = sm.make("probe", 12, 5, role="eval")
    s_b4 = _uniform_probe(ex, arch, A.graph_mask(ex.segments))
    assert abs(s_b4["own_origin_mass_ratio_to_uniform"] - 1.0) < 1e-5 and abs(s_b4["own_thread_mass_ratio_to_uniform"] - 1.0) < 1e-5
    s_c = _uniform_probe(ex, arch, att.layout_mask(ex.segments, "causal"))
    assert abs(s_c["own_origin_mass_ratio_to_uniform"] - 1.0) < 1e-5 and abs(s_c["own_thread_mass_ratio_to_uniform"] - 1.0) < 1e-5
    assert s_b4["own_origin_mass_fraction"] > s_c["own_origin_mass_fraction"]                    # B4 hides other origins: own share is larger
    assert abs(s_b4["ghost_ratio"] - 1.0) < 1e-4 or s_b4["ghost_ratio"] == 0.0


def test_b2_router_entropy_is_renormalised_and_prefix_mass_reported_separately(setup, tk):
    model, arch, sm = setup
    prefix = tk[1].encode(TF.SHARED_PREFIX_TEXT)
    ex = A.with_shared_prefix(sm.make("probe", 13, 4, role="eval"), prefix)
    s = _uniform_probe(ex, arch, att.layout_mask(ex.segments, "causal"))
    assert s["prefix_mass_fraction"] > 0.0 and s["router_entropy"] <= s["router_entropy_max"] + 1e-6
    assert s["own_origin_mass_fraction_excl_prefix"] > s["own_origin_mass_fraction"]           # excluding the prefix mass raises the own share
    assert s["own_origin_mass_ratio_to_uniform"] > 0
