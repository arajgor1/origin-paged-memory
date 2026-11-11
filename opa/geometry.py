"""Hyperbolic geometry for OPA (protocol 5.1 U5, paper Sections 7-8), numpy and torch.

Ported from opa/reference/protocol_snippets_v2.py; tests in tests/test_geometry.py compare against that reference.

Contents
  numpy:  mobius_add, dist, dist0, expmap0, AddressBook (red-dot frames, LCA-frame distances), oaa_attention,
          paging_tv, descend, make_tree
  torch:  mobius_add_t, dist_t, expmap0_t, AddressBookTorch.pair_distance (same semantics, any float dtype)
  layout: spread_directions, TreeLayout -- the training-free layout of paper 7.5 (one step delta outward per origin
          along its thread's ray) and the distances the H5 arms use on the SAME tree:
            A1 euclidean   flat embedding of the same tree (same rule, Euclidean steps)
            A2 hyperbolic  Poincare distance computed in the lowest-common-ancestor frame
            A3 tree        exact tree path length through the lowest common ancestor (times delta)
          A4 (fitted Poincare embeddings of tree + graph edges) is fit_poincare().
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- numpy primitives (curvature -1)


def mobius_add(x, y):
    xy, x2, y2 = np.dot(x, y), np.dot(x, x), np.dot(y, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / (1 + 2 * xy + x2 * y2)


def dist(x, y):
    diff = np.dot(x - y, x - y)
    den = (1 - np.dot(x, x)) * (1 - np.dot(y, y))
    return float(np.arccosh(1 + 2 * diff / den))


def dist0(y):
    return float(2 * np.arctanh(min(np.linalg.norm(y), 1 - 1e-15)))


def expmap0(v):
    n = np.linalg.norm(v)
    return np.tanh(n) * v / n if n > 0 else v


# ---------------------------------------------------------------- red-dot frames (numpy)


class AddressBook:
    """Origin 0 is the session root at the centre. Every other origin c stores offset[c] = exp0(delta_c),
    its position in the frame of its parent's red dot (paper Prop. 9-10)."""

    def __init__(self, m, dtype=np.float64):
        self.m, self.dtype = m, dtype
        self.parent = {0: None}
        self.offset = {0: np.zeros(m, dtype=dtype)}
        self.depth = {0: 0}
        self.hlen = {0: 0.0}          # hyperbolic length of each node's step (2 |delta|), for the representability test
        self.n_exact_fallback = 0     # pair_distance calls answered by the arbitrary-precision path (exact_fallback)
        self.n_capped = 0             # calls that fell back to collapse_cap (collapse detected, or exact path failed)
        self.exact_pairs, self.capped_pairs = set(), set()   # unique (node, node) pairs; n_* above count CALLS
        self._dec_off, self._dec_frame = {}, {}

    def add(self, c, par, delta, d_max=None):
        delta = np.asarray(delta, dtype=np.float64)
        if d_max is not None and np.linalg.norm(delta) > d_max:
            delta = delta * (d_max / np.linalg.norm(delta))
        self.parent[c] = par
        self.offset[c] = expmap0(delta).astype(self.dtype)
        self.hlen[c] = 2.0 * float(np.linalg.norm(delta))
        self.depth[c] = self.depth[par] + 1

    def path_to_root(self, c):
        out = []
        while c is not None:
            out.append(c)
            c = self.parent[c]
        return out

    def lca(self, a, b):
        anc = set(self.path_to_root(a))
        for n in self.path_to_root(b):
            if n in anc:
                return n
        raise KeyError((a, b))

    def coords_in_frame(self, node, ancestor, work_dtype=np.float32):
        path, n = [], node
        while n != ancestor:
            path.append(n)
            n = self.parent[n]
        y = np.zeros(self.m, dtype=work_dtype)
        for n in path:                                   # deepest first
            y = mobius_add(self.offset[n].astype(work_dtype), y).astype(work_dtype)
        return y

    def path_radius(self, node, ancestor):
        """Upper bound of the hyperbolic distance from `ancestor` to `node` (sum of the step lengths on the path)."""
        r, n = 0.0, node
        while n != ancestor:
            r += self.hlen[n]
            n = self.parent[n]
        return r

    def _exact_frame(self, node, ancestor, prec):
        """node's coordinates in the frame of `ancestor`, composed in decimal arithmetic from the stored float64
        offsets taken as exact numbers (the same right-nested composition as coords_in_frame)."""
        from decimal import Decimal
        key = (node, ancestor, prec)
        if key in self._dec_frame:
            return self._dec_frame[key]
        path, n = [], node
        while n != ancestor:
            path.append(n)
            n = self.parent[n]
        y = [Decimal(0)] * self.m
        for n in path:                                   # deepest first
            if n not in self._dec_off:
                xs = [Decimal(float(v)) for v in self.offset[n]]
                self._dec_off[n] = (xs, sum(v * v for v in xs))
            xs, x2 = self._dec_off[n]
            xy = sum(a * b for a, b in zip(xs, y))
            y2 = sum(b * b for b in y)
            A, B, den = 1 + 2 * xy + y2, 1 - x2, 1 + 2 * xy + x2 * y2
            y = [(A * a + B * b) / den for a, b in zip(xs, y)]
        self._dec_frame[key] = y
        return y

    def exact_pair_distance(self, q, c):
        """LCA-frame Poincare distance in arbitrary precision (decimal, precision grows with the path radius)."""
        from decimal import localcontext
        a = self.lca(q, c)
        rad = max(self.path_radius(q, a), self.path_radius(c, a))
        prec = max(50, int(0.5 * rad) + 40)
        with localcontext() as ctx:
            ctx.prec = prec
            yq, yc = self._exact_frame(q, a, prec), self._exact_frame(c, a, prec)
            den = (1 - sum(v * v for v in yq)) * (1 - sum(v * v for v in yc))
            arg = 1 + 2 * sum((u - v) ** 2 for u, v in zip(yq, yc)) / den
            return float((arg + (arg * arg - 1).sqrt()).ln())

    def pair_distance(self, q, c, work_dtype=np.float32, d_clamp=None, collapse_cap=None, exact_fallback=False):
        """Distance in the LCA frame. If composition collapses onto the boundary in work_dtype (the precision
        horizon of paper Prop. 10), the distance is reported as d_clamp (or inf if no clamp is given).
        d_clamp clips every distance at that value.
        exact_fallback: a pair whose path below the LCA is longer than SAFE_RADIUS on either side is not
        representable in float64 (the nested composition sticks just below |y| = 1 and returns finite WRONG values),
        so it is computed exactly in arbitrary precision and counted in n_exact_fallback.
        collapse_cap (when not None) is the last resort: a non-finite / collapsed float result, or a failure of the
        exact path, is replaced by that finite value and counted in n_capped."""
        if q == c:
            return 0.0
        a = self.lca(q, c)
        if exact_fallback and max(self.path_radius(q, a), self.path_radius(c, a)) > SAFE_RADIUS:
            try:
                d = self.exact_pair_distance(q, c)
                self.n_exact_fallback += 1
                self.exact_pairs.add((min(q, c), max(q, c)))
                return min(d, d_clamp) if d_clamp is not None else d
            except Exception:
                d = math.inf
        else:
            yq = self.coords_in_frame(q, a, work_dtype).astype(np.float64)
            yc = self.coords_in_frame(c, a, work_dtype).astype(np.float64)
            den = (1 - np.dot(yq, yq)) * (1 - np.dot(yc, yc))
            if not np.isfinite(den) or den <= 0:
                d = math.inf
            else:
                with np.errstate(all="ignore"):
                    d = float(np.arccosh(1 + 2 * np.dot(yq - yc, yq - yc) / den))
        if collapse_cap is not None and not math.isfinite(d):
            d = float(collapse_cap)
            self.n_capped += 1
            self.capped_pairs.add((min(q, c), max(q, c)))
        if d_clamp is not None:
            d = min(d, d_clamp)
        return d

    def global_point(self, c, work_dtype=np.float64):
        return self.coords_in_frame(c, 0, work_dtype) if c != 0 else np.zeros(self.m, work_dtype)


# ---------------------------------------------------------------- attention identities (numpy)


def oaa_attention(s_loc, router_bias, d_struct, lam):
    """One query against C co-loaded origins. Returns attention a [C,B], router masses pi [C], local softmax alpha."""
    b = router_bias - lam * d_struct
    full = s_loc + b[:, None]
    a = np.exp(full - full.max())
    a /= a.sum()
    lse = np.log(np.exp(s_loc - s_loc.max(1, keepdims=True)).sum(1)) + s_loc.max(1)
    z = b + lse
    pi = np.exp(z - z.max())
    pi /= pi.sum()
    alpha = np.exp(s_loc - lse[:, None])
    return a, pi, alpha


def paging_tv(a, pi, resident):
    mask = np.zeros(a.shape[0], bool)
    mask[list(resident)] = True
    a_s = np.where(mask[:, None], a, 0.0)
    a_s /= a_s.sum()
    return 0.5 * np.abs(a - a_s).sum(), 1.0 - pi[mask].sum()


# ---------------------------------------------------------------- beam descent


def descend(children, score, root=0, beam=2):
    """children: node -> list of child ids; score: node -> float. Returns (leaves, number of scores computed)."""
    frontier, n_scores, leaves = [root], 0, []
    while frontier:
        cands = [ch for n in frontier for ch in children.get(n, [])]
        if not cands:
            leaves = frontier
            break
        s = np.array([score(ch) for ch in cands])
        n_scores += len(cands)
        keep = np.argsort(-s, kind="stable")[:beam]
        frontier = [cands[i] for i in keep]
        if all(not children.get(n) for n in frontier):
            leaves = frontier
            break
    return leaves, n_scores


def make_tree(b, L):
    children, nxt, level = {}, 1, [0]
    for _ in range(L):
        new = []
        for n in level:
            children[n] = list(range(nxt, nxt + b))
            new += children[n]
            nxt += b
        level = new
    return children, level


# ---------------------------------------------------------------- torch port


def _torch():
    import torch
    return torch


def mobius_add_t(x, y):
    """Batched Mobius addition over the last dimension."""
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / (1 + 2 * xy + x2 * y2)


def dist_t(x, y):
    torch = _torch()
    diff = ((x - y) ** 2).sum(-1)
    den = (1 - (x * x).sum(-1)) * (1 - (y * y).sum(-1))
    return torch.arccosh(1 + 2 * diff / den)


def expmap0_t(v):
    torch = _torch()
    n = v.norm(dim=-1, keepdim=True)
    safe = torch.where(n > 0, n, torch.ones_like(n))
    return torch.where(n > 0, torch.tanh(safe) * v / safe, v)


class AddressBookTorch:
    """Torch twin of AddressBook: offsets stored in `dtype`, composed in `work_dtype` in the LCA frame."""

    def __init__(self, book: AddressBook, dtype=None):
        torch = _torch()
        self.book = book
        self.dtype = dtype or torch.float32
        self.offset = {k: torch.as_tensor(np.asarray(v, dtype=np.float64)).to(self.dtype) for k, v in book.offset.items()}

    def coords_in_frame(self, node, ancestor, work_dtype=None):
        torch = _torch()
        work_dtype = work_dtype or torch.float32
        path, n = [], node
        while n != ancestor:
            path.append(n)
            n = self.book.parent[n]
        y = torch.zeros(self.book.m, dtype=work_dtype)
        for n in path:
            y = mobius_add_t(self.offset[n].to(work_dtype), y)
        return y

    def pair_distance(self, q, c, work_dtype=None, d_clamp=None, dist_dtype=None):
        """Compose offsets in work_dtype in the LCA frame; evaluate the final arccosh in dist_dtype (float64 by
        default, exactly as the numpy reference does)."""
        torch = _torch()
        work_dtype = work_dtype or torch.float32
        dist_dtype = dist_dtype or torch.float64
        if q == c:
            return torch.zeros((), dtype=dist_dtype)
        a = self.book.lca(q, c)
        d = dist_t(self.coords_in_frame(q, a, work_dtype).to(dist_dtype), self.coords_in_frame(c, a, work_dtype).to(dist_dtype))
        if d_clamp is not None:
            d = torch.clamp(torch.nan_to_num(d, nan=d_clamp, posinf=d_clamp), max=d_clamp)
        return d


# ---------------------------------------------------------------- U5 (protocol 5.1)


def u5_report(seed=0, n_pairs=1000, threads=4, depth=6, m=8):
    """Unit test U5: the torch port matches protocol_snippets_v2.py.

    Three parts (definition recorded in state/decisions.md and the pre-registration):
      a) primitives in fp32: mobius_add, expmap0, dist on n_pairs random points with norm <= 0.9; torch fp32 vs the
         numpy reference on the same fp32 inputs; max abs error <= 1e-6 (the protocol's tolerance).
      b) function identity: LCA-frame distances on a threads x depth tree, torch vs reference both in float64,
         max abs error <= 1e-10.
      c) fp32 LCA-frame distances: the port's error against the float64 truth is no larger than 2x the reference's
         own fp32 error (+1e-7). The literal 1e-6 is unattainable here for any independent fp32 implementation
         (the reference itself is off the float64 truth by up to ~3e-5 at distance 12), so it is reported, not gated.
    """
    import torch
    from .reference import protocol_snippets_v2 as ref

    rng = np.random.default_rng(seed)
    worst_prim = 0.0
    for _ in range(n_pairs):
        x = (g := rng.standard_normal(m)) / np.linalg.norm(g) * rng.uniform(0, 0.9)
        y = (h := rng.standard_normal(m)) / np.linalg.norm(h) * rng.uniform(0, 0.9)
        x32, y32 = x.astype(np.float32), y.astype(np.float32)
        tx, ty = torch.tensor(x32), torch.tensor(y32)
        worst_prim = max(worst_prim,
                         float(np.abs(mobius_add_t(tx, ty).numpy() - ref.mobius_add(x32, y32)).max()),
                         abs(float(dist_t(tx, ty)) - float(ref.dist(x32, y32))),
                         float(np.abs(expmap0_t(tx).numpy() - ref.expmap0(x32)).max()))
    book, rbook = AddressBook(m), ref.AddressBook(m)
    nid = 1
    for _ in range(threads):
        par, u = 0, rng.standard_normal(m)
        u /= np.linalg.norm(u)
        for _ in range(depth):
            d = 0.5 * (u + 0.2 * rng.standard_normal(m))
            book.add(nid, par, d)
            rbook.add(nid, par, d)
            par = nid
            nid += 1
    t64, t32 = AddressBookTorch(book, dtype=torch.float64), AddressBookTorch(book, dtype=torch.float32)
    worst_id, worst_port32, worst_ref32, worst_gap32, worst_gap32_horizon = 0.0, 0.0, 0.0, 0.0, 0.0
    for q in range(1, nid):
        for c in range(1, nid):
            if q == c:
                continue
            truth = rbook.pair_distance(q, c, work_dtype=np.float64)
            worst_id = max(worst_id, abs(float(t64.pair_distance(q, c, work_dtype=torch.float64)) - truth))
            p32 = float(t32.pair_distance(q, c, work_dtype=torch.float32))
            r32 = rbook.pair_distance(q, c, work_dtype=np.float32)
            worst_port32, worst_ref32 = max(worst_port32, abs(p32 - truth)), max(worst_ref32, abs(r32 - truth))
            worst_gap32 = max(worst_gap32, abs(p32 - r32))
            if truth <= 4.61:                                   # attention horizon at lambda = 2, tau = 1e-4
                worst_gap32_horizon = max(worst_gap32_horizon, abs(p32 - r32))
    checks = {
        "a_primitives_fp32_max_abs_err": worst_prim, "a_pass": worst_prim <= 1e-6,
        "b_lca_distance_float64_max_abs_err": worst_id, "b_pass": worst_id <= 1e-10,
        "c_port_fp32_err_vs_truth": worst_port32, "c_reference_fp32_err_vs_truth": worst_ref32,
        "c_pass": worst_port32 <= 2 * worst_ref32 + 1e-7,
        "reported_port_vs_reference_fp32_max_abs_gap": worst_gap32,
        "reported_port_vs_reference_fp32_gap_within_d_4.61": worst_gap32_horizon,
        "tree": {"threads": threads, "depth": depth, "m": m, "seed": seed, "pairs": (nid - 1) * (nid - 2)},
    }
    checks["passed"] = bool(checks["a_pass"] and checks["b_pass"] and checks["c_pass"])
    return checks


# ---------------------------------------------------------------- training-free layout and the H5 distances


def spread_directions(n, m, seed=0):
    """n unit directions in R^m, well spread: +-e_i of a seeded random orthonormal basis first (cosine 0 or -1
    between any two), then greedy farthest-point picks from random candidates."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    base = [q[:, i] for i in range(m)] + [-q[:, i] for i in range(m)]
    out = base[:n]
    while len(out) < n:
        cand = rng.standard_normal((256, m))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        worst = np.max(np.abs(cand @ np.array(out).T), axis=1)
        out.append(cand[int(np.argmin(worst))])
    return np.array(out[:n])


# 2 x the float64 precision horizon of the Poincare ball (dist0 of a point at 1 - 2**-53 is ~37.4); last resort only
COLLAPSE_CAP = 75.0
# a float64 LCA-frame composition is trustworthy only while 1 - |y|^2 ~ 4 exp(-radius) >> eps: at radius 36-37 the
# error is already 0.3-0.7, so any pair with a path longer than this below the LCA is computed exactly.
# Measured float64 error bound at radius 33: <= 1.4e-3.
SAFE_RADIUS = 33.0


class TreeLayout:
    """The address-book tree of one session, with every node's position under three geometries.

    Node 0 is the root. add_node(c, parent, direction) places c one step `delta` (hyperbolic length) away from its
    parent along `direction` (a unit vector in the parent's frame, paper 7.5). The same tree and the same directions
    give the Euclidean embedding (A1) and the exact tree distance (A3), so the H5 arms differ only in the distance.
    """

    def __init__(self, m=32, delta=1.0, d_clamp=36.0):
        """d_clamp: clip every A2 distance at this value (default 36, Track B and U5). d_clamp=None means no fixed
        clamp (Track A default): pairs with a path > SAFE_RADIUS below their LCA are computed exactly in arbitrary
        precision (n_exact_fallback); COLLAPSE_CAP is a counted last resort (n_capped)."""
        self.m, self.delta = m, float(delta)
        self.d_clamp = None if d_clamp is None else float(d_clamp)
        self.book = AddressBook(m, dtype=np.float64)
        self.euc = {0: np.zeros(m)}

    def add_node(self, c, parent, direction):
        u = np.asarray(direction, dtype=np.float64)
        u = u / np.linalg.norm(u)
        # expmap0(v) with |v| = delta/2 lies at hyperbolic distance delta from the frame origin
        self.book.add(c, parent, (self.delta / 2.0) * u)
        self.euc[c] = self.euc[parent] + self.delta * u

    @property
    def n_exact_fallback(self):
        return self.book.n_exact_fallback

    @property
    def n_capped(self):
        return self.book.n_capped

    @property
    def n_exact_pairs(self):
        return len(self.book.exact_pairs)

    @property
    def n_capped_pairs(self):
        return len(self.book.capped_pairs)

    def parent(self, c):
        return self.book.parent[c]

    def depth(self, c):
        return self.book.depth[c]

    def distance(self, q, c, kind):
        if kind in ("none", "A0"):
            return 0.0
        if q == c:
            return 0.0
        if kind in ("hyperbolic", "A2"):
            return self.book.pair_distance(q, c, work_dtype=np.float64, d_clamp=self.d_clamp,
                                           collapse_cap=COLLAPSE_CAP if self.d_clamp is None else None,
                                           exact_fallback=self.d_clamp is None)
        if kind in ("euclidean", "A1"):
            return float(np.linalg.norm(self.euc[q] - self.euc[c]))
        if kind in ("tree", "A3"):
            a = self.book.lca(q, c)
            return self.delta * (self.depth(q) + self.depth(c) - 2 * self.depth(a))
        raise ValueError(kind)


def fit_poincare(n_nodes, edges, dim=10, epochs=300, lr=0.3, neg=10, seed=0, init=None):
    """Poincare embeddings of an undirected graph (Nickel & Kiela 2017), Riemannian SGD in numpy.

    edges: list of (u, v) node-index pairs (tree edges plus graph edges). Returns an array [n_nodes, dim] inside
    the unit ball. Used by arm A4 (fitted addresses) at the agent level.
    """
    rng = np.random.default_rng(seed)
    x = init.copy() if init is not None else rng.uniform(-1e-3, 1e-3, size=(n_nodes, dim))
    edges = np.asarray(edges, dtype=int)
    if len(edges) == 0:
        return x
    adj = {i: set() for i in range(n_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    eps = 1e-5

    def proj(p):
        n = np.linalg.norm(p)
        return p * ((1 - eps) / n) if n >= 1 - eps else p

    def grad_d(u, v):
        """Euclidean gradient of d(u, v) with respect to u."""
        uu, vv = np.dot(u, u), np.dot(v, v)
        alpha, beta = 1 - uu, 1 - vv
        gamma = 1 + 2 * np.dot(u - v, u - v) / (alpha * beta)
        denom = beta * math.sqrt(max(gamma ** 2 - 1, 1e-15))
        return 4.0 / denom * ((vv - 2 * np.dot(u, v) + 1) / alpha ** 2 * u - v / alpha)

    for ep in range(epochs):
        rate = lr if ep >= 20 else lr / 10.0                   # burn-in, as in the paper
        for i in rng.permutation(len(edges)):
            u, v = edges[i]
            negs = [w for w in rng.integers(0, n_nodes, size=neg * 2) if w != u and w not in adj[u]][:neg]
            cands = [v] + negs
            ds = np.array([dist(x[u], x[w]) for w in cands])
            ex = np.exp(-ds - np.max(-ds))
            p = ex / ex.sum()
            # loss = -log softmax(-d)[0] = d_0 + log sum_k exp(-d_k);  dL/dd_0 = 1 - p_0,  dL/dd_k = -p_k (k > 0)
            coef = -p
            coef[0] += 1.0
            gu = np.zeros(dim)
            for k, w in enumerate(cands):
                gu += coef[k] * grad_d(x[u], x[w])
                gw = coef[k] * grad_d(x[w], x[u])
                x[w] = proj(x[w] - rate * ((1 - np.dot(x[w], x[w])) ** 2 / 4) * gw)
            x[u] = proj(x[u] - rate * ((1 - np.dot(x[u], x[u])) ** 2 / 4) * gu)
    return x
