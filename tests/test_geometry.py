"""P0.4 / U5: opa.geometry against the tested numpy reference (opa/reference/protocol_snippets_v2.py)."""
import numpy as np
import pytest
import torch

from opa import geometry as g
from opa.reference import protocol_snippets_v2 as ref


def _book(rng, threads=3, depth=5, m=8):
    ab_new, ab_ref = g.AddressBook(m), ref.AddressBook(m)
    nid = 1
    for _ in range(threads):
        par, u = 0, rng.standard_normal(m)
        u /= np.linalg.norm(u)
        for _ in range(depth):
            d = 0.5 * (u + 0.2 * rng.standard_normal(m))
            ab_new.add(nid, par, d)
            ab_ref.add(nid, par, d)
            par = nid
            nid += 1
    return ab_new, ab_ref, nid


def test_primitives_match_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = g.expmap0(rng.standard_normal(8) * 0.6), g.expmap0(rng.standard_normal(8) * 0.6)
        assert np.allclose(g.mobius_add(x, y), ref.mobius_add(x, y), atol=1e-15)
        assert abs(g.dist(x, y) - ref.dist(x, y)) < 1e-12
        assert abs(g.dist0(x) - ref.dist0(x)) < 1e-12


def test_isometry_and_lca_frame():
    rng = np.random.default_rng(1)
    a, x, y = (g.expmap0(rng.standard_normal(8) * 0.5) for _ in range(3))
    assert abs(g.dist(g.mobius_add(-a, x), g.mobius_add(-a, y)) - g.dist(x, y)) < 1e-12
    ab, abr, n = _book(rng)
    for q in range(1, n):
        for c in range(1, n):
            d_new = ab.pair_distance(q, c, work_dtype=np.float64)
            d_ref = abr.pair_distance(q, c, work_dtype=np.float64)
            assert abs(d_new - d_ref) < 1e-12
            dg = g.dist(ab.global_point(q), ab.global_point(c)) if q != c else 0.0
            assert abs(d_new - dg) < 1e-9


def test_u5_report_passes():
    """U5 as defined in opa.geometry.u5_report (primitives at 1e-6 fp32, float64 identity, fp32 no worse than ref)."""
    r = g.u5_report(seed=0)
    assert r["passed"], r
    assert r["a_primitives_fp32_max_abs_err"] <= 1e-6


def test_u5_torch_primitives_fp32():
    rng = np.random.default_rng(2)
    x = torch.tensor(g.expmap0(rng.standard_normal(8) * 0.4), dtype=torch.float32)
    y = torch.tensor(g.expmap0(rng.standard_normal(8) * 0.4), dtype=torch.float32)
    assert torch.allclose(g.mobius_add_t(x, y), torch.tensor(ref.mobius_add(x.numpy(), y.numpy())), atol=1e-6)
    assert abs(float(g.dist_t(x, y)) - ref.dist(x.numpy().astype(float), y.numpy().astype(float))) < 1e-5
    v = torch.tensor(rng.standard_normal(8), dtype=torch.float32)
    assert torch.allclose(g.expmap0_t(v), torch.tensor(ref.expmap0(v.numpy())), atol=1e-6)


def test_deep_chain_lca_frame_stays_exact_fp32():
    deep = g.AddressBook(2)
    for k in range(1, 41):
        deep.add(k, k - 1, np.array([0.5, 0.0]))
    assert abs(deep.pair_distance(39, 40, work_dtype=np.float32) - 1.0) < 1e-5
    tb = g.AddressBookTorch(deep)
    assert abs(float(tb.pair_distance(39, 40)) - 1.0) < 1e-5
    # far pairs through a collapsed frame are clamped, not NaN
    assert deep.pair_distance(0, 40, work_dtype=np.float32, d_clamp=18.25) == 18.25


def test_oaa_paging_descent_match_reference():
    rng = np.random.default_rng(3)
    s_loc = rng.standard_normal((8, 64)) * 2
    rb, ds = rng.standard_normal(8), np.abs(rng.standard_normal(8)) * 2
    a, pi, al = g.oaa_attention(s_loc, rb, ds, 2.0)
    ar, pir, alr = ref.oaa_attention(s_loc, rb, ds, 2.0)
    assert np.allclose(a, ar) and np.allclose(pi, pir) and np.allclose(al, alr)
    assert np.abs(a - pi[:, None] * al).max() < 1e-12
    tv, pred = g.paging_tv(a, pi, [0, 1, 2])
    assert abs(tv - pred) < 1e-12
    children, leaves = g.make_tree(8, 4)
    parent = {ch: n for n, chs in children.items() for ch in chs}
    target, anc = leaves[1234], set()
    n = target
    while n in parent:
        anc.add(n)
        n = parent[n]
    got, n_scores = g.descend(children, lambda node: 1.0 if node in anc else 0.0, beam=2)
    assert target in got and n_scores == 8 + 3 * 2 * 8


def test_tree_layout_arms_share_the_tree():
    lay = g.TreeLayout(m=8, delta=1.0)
    dirs = g.spread_directions(2, 8, seed=0)
    nid, heads = 1, []
    for t in range(2):
        par = 0
        for _ in range(5):
            lay.add_node(nid, par, dirs[t])
            par = nid
            nid += 1
        heads.append(par)
    # same thread: all three geometries give |i - j| * delta
    for kind in ("A1", "A2", "A3"):
        assert abs(lay.distance(1, 4, kind) - 3.0) < 1e-9
    # across threads (orthogonal rays, depth 5 each): tree 10, euclidean sqrt(50), hyperbolic between them
    t, e, h = (lay.distance(heads[0], heads[1], k) for k in ("A3", "A1", "A2"))
    assert abs(t - 10.0) < 1e-9 and abs(e - np.sqrt(50)) < 1e-9
    assert e < h <= t + 1e-9
    assert lay.distance(heads[0], heads[1], "A0") == 0.0
    d = g.spread_directions(40, 32, seed=1)
    cos = np.abs(d @ d.T - np.eye(40))
    assert cos[:32, :32].max() < 1e-9 + 1.0 and np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_fit_poincare_recovers_tree_structure():
    children, leaves = g.make_tree(3, 2)                       # 13 nodes
    edges = [(p, c) for p, cs in children.items() for c in cs]
    x = g.fit_poincare(13, edges, dim=4, epochs=150, seed=0)
    assert np.all(np.linalg.norm(x, axis=1) < 1)
    linked = np.mean([g.dist(x[u], x[v]) for u, v in edges])
    rng = np.random.default_rng(0)
    others = [(int(a), int(b)) for a, b in rng.integers(0, 13, size=(200, 2)) if a != b and (a, b) not in edges and (b, a) not in edges]
    unlinked = np.mean([g.dist(x[u], x[v]) for u, v in others])
    assert linked < unlinked


# ---------------------------------------------------------------- d_clamp=None: clamp only on numerical collapse
def _two_rays(angle, a=32.0, b=32.0):
    """Root with two children at hyperbolic radius a and b, on rays at `angle` (rad) in the plane."""
    lay = g.TreeLayout(m=2, delta=a, d_clamp=None)       # step delta = hyperbolic radius a
    lay.add_node(1, 0, np.array([1.0, 0.0]))
    lay2 = lay
    lay2.book.add(2, 0, (b / 2.0) * np.array([np.cos(angle), np.sin(angle)]))
    return lay2


def test_none_clamp_far_points_use_exact_distance():
    for angle in (np.pi / 2, 1.0, 2.5):
        lay = _two_rays(angle)
        a = b = 32.0
        closed = float(np.arccosh(np.cosh(a) * np.cosh(b) - np.sinh(a) * np.sinh(b) * np.cos(angle)))
        d = lay.distance(1, 2, "A2")
        assert abs(d - closed) < 0.01, (angle, d, closed)
        assert lay.n_capped == 0 and lay.n_exact_fallback == 0 and d < g.COLLAPSE_CAP
    # perpendicular rays: arccosh(cosh a cosh b)
    lay = _two_rays(np.pi / 2)
    assert abs(lay.distance(1, 2, "A2") - float(np.arccosh(np.cosh(32.0) ** 2))) < 0.01


def _two_chains(n, angle):
    """Root plus two chains of n unit steps (hyperbolic length 1 each) on rays `angle` apart: nodes 1..n and n+1..2n."""
    lay = g.TreeLayout(m=2, delta=1.0, d_clamp=None)
    for k in range(1, n + 1):
        lay.add_node(k, k - 1 if k > 1 else 0, np.array([1.0, 0.0]))
    for k in range(1, n + 1):
        lay.add_node(n + k, n + k - 1 if k > 1 else 0, np.array([np.cos(angle), np.sin(angle)]))
    return lay


def test_exact_fallback_band_34_to_45_matches_closed_form():
    for n in (34, 36, 37, 40, 45):
        for angle in (0.3, 1.0, np.pi / 2, 2.5):
            lay = _two_chains(n, angle)
            closed = float(np.arccosh(np.cosh(n) ** 2 - np.sinh(n) ** 2 * np.cos(angle)))
            before = lay.n_exact_fallback
            d = lay.distance(n, 2 * n, "A2")
            assert abs(d - closed) < 0.01, (n, angle, d, closed)
            assert lay.n_exact_fallback == before + 1 and lay.n_capped == 0
        # same-chain pairs: root to the deep node, and a shallow node to the deep node
        assert abs(lay.distance(0, n, "A2") - n) < 0.01
        assert abs(lay.distance(1, n, "A2") - (n - 1)) < 0.01
    # the plain float64 LCA frame is silently wrong here (this is why the fallback exists)
    lay = _two_chains(37, 1.0)
    plain = lay.book.pair_distance(0, 37, work_dtype=np.float64)
    assert abs(plain - 37) > 0.05
    assert lay.distance(0, 37, "A2") == lay.distance(0, 37, "A2")      # deterministic


def test_fallback_counters_and_safe_pairs_stay_float():
    lay = _two_chains(34, 1.0)
    assert lay.n_exact_fallback == 0
    lay.distance(5, 20, "A2")                                       # path 15 below the LCA: float path
    lay.distance(0, 34, "A2")                                       # path 34 > SAFE_RADIUS: exact
    assert (lay.n_exact_fallback, lay.n_capped) == (1, 0)
    assert abs(lay.distance(10, 12, "A2") - 2.0) < 1e-9 and lay.n_exact_fallback == 1
    # a fixed clamp keeps the old behaviour and never uses the fallback
    fixed = g.TreeLayout(m=2, delta=1.0, d_clamp=36.0)
    for k in range(1, 41):
        fixed.add_node(k, k - 1, np.array([1.0, 0.0]))
    assert fixed.distance(0, 40, "A2") == 36.0 and fixed.n_exact_fallback == 0
    assert g.TreeLayout().d_clamp == 36.0                            # Track B / U5 default untouched


def test_cap_is_the_counted_last_resort():
    lay = g.TreeLayout(m=2, delta=2.0, d_clamp=None)
    for k in range(1, 5):
        lay.add_node(k, k - 1, np.array([1.0, 0.0]))
    lay.book.offset[2] = np.array([1.0, 0.0])                        # an unrepresentable (norm 1) step: exact path fails
    lay.book.hlen[2] = 80.0
    assert lay.distance(0, 4, "A2") == g.COLLAPSE_CAP
    assert (lay.n_capped, lay.n_exact_fallback) == (1, 0)


def test_tables_from_layout_rejects_none_clamp():
    from opa.trackb import attention
    lay = g.TreeLayout(m=2, delta=1.0, d_clamp=None)
    lay.add_node(1, 0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="numeric d_clamp"):
        attention.tables_from_layout(lay, [0, 1])


def test_pair_counters_are_unique_pairs_not_calls():
    lay = _two_chains(36, 1.0)
    for _ in range(3):
        lay.distance(0, 36, "A2")
    lay.distance(36, 0, "A2")                                       # same unordered pair
    lay.distance(0, 35, "A2")
    assert lay.n_exact_fallback == 5 and lay.n_exact_pairs == 2 and lay.n_capped_pairs == 0
