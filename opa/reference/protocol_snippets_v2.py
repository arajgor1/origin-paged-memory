"""
OPA v2 code starters (numpy reference). Port to torch by swapping np for torch.

Contents
  1. Poincare-ball primitives (curvature -1): mobius_add, dist, expmap0
  2. Red-dot frames: AddressBook stores each origin as an offset in its parent's frame.
     Offsets compose right-nested, x = u1 (+) (u2 (+) (... (+) un)), so a node's
     coordinates in any ancestor's frame need only the offsets below that ancestor.
     pair_distance(q, c) works in the frame of the lowest common ancestor.
  3. Position map M5 (fresh frame) next to the v1 maps.
  4. Origin-Anchored Attention for one query: logits, router masses, paging error.
  5. Coarse-to-fine beam descent over a b-ary shell tree.
  6. make_mt_item: multi-thread late-binding item generator (MT-LB-MHQA).
Run `python protocol_snippets_v2.py` to execute the self-tests.
"""
import numpy as np

# ---------------------------------------------------------------- 1. Poincare ball
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


# ---------------------------------------------------------------- 2. red-dot frames
class AddressBook:
    """Origin 0 is the session root at the centre. Every other origin c stores
    offset[c] = exp0(delta_c): its position in the frame of its parent's red dot."""

    def __init__(self, m, dtype=np.float64):
        self.m, self.dtype = m, dtype
        self.parent = {0: None}
        self.offset = {0: np.zeros(m, dtype=dtype)}
        self.depth = {0: 0}

    def add(self, c, par, delta, d_max=None):
        delta = np.asarray(delta, dtype=np.float64)
        if d_max is not None and np.linalg.norm(delta) > d_max:
            delta = delta * (d_max / np.linalg.norm(delta))
        self.parent[c] = par
        self.offset[c] = expmap0(delta).astype(self.dtype)
        self.depth[c] = self.depth[par] + 1

    def path_to_root(self, c):
        out = []
        while c is not None:
            out.append(c); c = self.parent[c]
        return out                                   # [c, par(c), ..., 0]

    def lca(self, a, b):
        anc = set(self.path_to_root(a))
        for n in self.path_to_root(b):
            if n in anc:
                return n

    def coords_in_frame(self, node, ancestor, work_dtype=np.float32):
        """Right-nested composition of the offsets strictly below `ancestor` on the path to `node`."""
        path = []
        n = node
        while n != ancestor:
            path.append(n); n = self.parent[n]
        y = np.zeros(self.m, dtype=work_dtype)
        for n in path:                               # deepest first: y = u_n, then u_{n-1} (+) y, ...
            y = mobius_add(self.offset[n].astype(work_dtype), y).astype(work_dtype)
        return y

    def pair_distance(self, q, c, work_dtype=np.float32, d_clamp=None):
        a = self.lca(q, c)
        yq = self.coords_in_frame(q, a, work_dtype)
        yc = self.coords_in_frame(c, a, work_dtype)
        d = dist(yq.astype(np.float64), yc.astype(np.float64)) if q != c else 0.0
        return min(d, d_clamp) if d_clamp is not None else d

    def global_point(self, c, work_dtype=np.float64):
        return self.coords_in_frame(c, 0, work_dtype) if c != 0 else np.zeros(self.m, work_dtype)


# ---------------------------------------------------------------- 3. position maps incl. M5
def inv_freq(head_dim, base=10000.0):
    return base ** (-np.arange(0, head_dim, 2) / head_dim)       # [P], plane 0 = highest freq

def plane_positions(map_name, c, l, slot, B, P, n_local=48, g=None):
    """Per-plane position ids [T, P] for maps M1-M5 (rotate_half layout uses them per plane)."""
    g = g or (lambda c_, B_: c_ * B_)
    arch = np.repeat((c * B + l)[:, None], P, axis=1).astype(float)
    virt = np.repeat((slot * B + l)[:, None], P, axis=1).astype(float)
    if map_name == "M1": return arch
    if map_name == "M2": return virt
    if map_name == "M5": return np.repeat(l[:, None], P, axis=1).astype(float)   # fresh frame
    pos = np.empty((len(c), P))
    if map_name == "M3":
        pos[:, :n_local] = l[:, None]; pos[:, n_local:] = c[:, None]; return pos
    if map_name == "M4":
        pos[:, :n_local] = virt[:, :n_local]; pos[:, n_local:] = g(c, B)[:, None]; return pos
    raise ValueError(map_name)


# ---------------------------------------------------------------- 4. Origin-Anchored Attention (one query)
def oaa_attention(s_loc, router_bias, d_struct, lam):
    """s_loc: [C, B] local RoPE logits of one query against C co-loaded origins.
    router_bias: [C] values r_u^T A o(c).  d_struct: [C] hyperbolic distances.
    Returns attention a [C, B], router masses pi [C], local softmax alpha [C, B]."""
    b = router_bias - lam * d_struct
    full = s_loc + b[:, None]
    a = np.exp(full - full.max()); a /= a.sum()
    lse = np.log(np.exp(s_loc - s_loc.max(1, keepdims=True)).sum(1)) + s_loc.max(1)
    z = b + lse
    pi = np.exp(z - z.max()); pi /= pi.sum()
    alpha = np.exp(s_loc - lse[:, None])
    return a, pi, alpha

def paging_tv(a, pi, resident):
    """Total-variation distance between full and paged attention, and the predicted 1 - sum(pi[S])."""
    mask = np.zeros(a.shape[0], bool); mask[list(resident)] = True
    a_s = np.where(mask[:, None], a, 0.0); a_s /= a_s.sum()
    return 0.5 * np.abs(a - a_s).sum(), 1.0 - pi[mask].sum()


# ---------------------------------------------------------------- 5. beam descent
def descend(children, score, root=0, beam=2):
    """children: dict node -> list of child ids (leaves have none). score: node -> float.
    Returns (ranked leaves, number of scores computed)."""
    frontier, n_scores, leaves = [root], 0, []
    while frontier:
        cands = [ch for n in frontier for ch in children.get(n, [])]
        if not cands:
            leaves = frontier; break
        s = np.array([score(ch) for ch in cands]); n_scores += len(cands)
        keep = np.argsort(-s)[:beam]
        frontier = [cands[i] for i in keep]
        if all(not children.get(n) for n in frontier):
            leaves = frontier; break
    return leaves, n_scores

def make_tree(b, L):
    children, nxt, level = {}, 1, [0]
    for _ in range(L):
        new = []
        for n in level:
            children[n] = list(range(nxt, nxt + b)); new += children[n]; nxt += b
        level = new
    return children, level                           # level = leaves


# ---------------------------------------------------------------- 6. MT-LB-MHQA generator
def make_mt_item(filler_chunks, n_threads, origins_per_thread, hops, rng, cross_thread=True, confusers=True):
    """Multi-thread late-binding item. Threads are runs of consecutive origins (topics);
    the question's thread is the most recent one. Its chain's facts are placed in reverse
    order (late-bound). If cross_thread, the last hop's fact is stated in an earlier, other
    thread. Confusers: every other thread gets a near-duplicate fact whose entity name
    differs by one character."""
    def name():
        return "Ent" + "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 5))
    ents = [name() for _ in range(hops + 1)]
    attr = f"{rng.integers(40, 120)} degC"
    facts = [f"{ents[i]} sources parts from {ents[i + 1]}." for i in range(hops - 1)]
    facts.append(f"Parts from {ents[hops - 1]} fail above {attr}.")
    n_orig = n_threads * origins_per_thread
    thread_of = np.repeat(np.arange(n_threads), origins_per_thread)
    qt = n_threads - 1                                   # the question's thread is the latest
    texts = [filler_chunks[i % len(filler_chunks)] for i in range(n_orig)]
    tq = [i for i in range(n_orig) if thread_of[i] == qt]
    fact_origin = [tq[min(len(tq) - 1, len(tq) - 1 - j)] if j < len(tq) else tq[0] for j in range(hops - 1)]
    if cross_thread and n_threads > 1:
        fact_origin.append(int(rng.choice([i for i in range(n_orig) if thread_of[i] != qt])))
    else:
        fact_origin.append(tq[0])
    for f, o in zip(facts, fact_origin):
        texts[o] = texts[o] + " " + f
    if confusers and n_threads > 1:
        twin = ents[hops - 1][:-1] + ("Z" if ents[hops - 1][-1] != "Z" else "Y")
        for t in range(n_threads):
            if t == qt:
                continue
            o = int(rng.choice([i for i in range(n_orig) if thread_of[i] == t]))
            texts[o] += f" Parts from {twin} fail above {rng.integers(40, 120)} degC."
    question = f"Do {ents[0]}'s parts survive {attr}? Give the failure threshold."
    # hop j is late-bound if its fact lies before the origin where its entity first becomes relevant
    relevant_at = {ents[0]: 0}                           # the question names ents[0] from the start
    for j in range(hops - 1):
        relevant_at.setdefault(ents[j + 1], fact_origin[j])
    h_late = sum(1 for j in range(hops) if fact_origin[j] < relevant_at[ents[j]])
    return {"texts": texts, "thread_of": thread_of.tolist(), "question_thread": qt, "question": question,
            "answer": attr, "fact_origins": fact_origin, "h_late": int(h_late),
            "cross_thread_hop": bool(cross_thread and n_threads > 1)}


# ---------------------------------------------------------------- self-tests
def _round_bf16(v):
    v = np.asarray(v, dtype=np.float32)
    i = v.view(np.uint32).astype(np.uint64)
    i = ((i + 0x7FFF + ((i >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return i.view(np.float32)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # T1 Mobius translation is an isometry
    a, x, y = (expmap0(rng.standard_normal(8) * 0.5) for _ in range(3))
    e1 = abs(dist(mobius_add(-a, x), mobius_add(-a, y)) - dist(x, y))
    assert e1 < 1e-12, e1

    # T2 LCA-frame distances equal root-frame distances (float64, shallow tree)
    ab = AddressBook(8)
    nid = 1
    for t in range(3):                              # 3 threads of depth 5, sharing the root
        par, u = 0, rng.standard_normal(8); u /= np.linalg.norm(u)
        for k in range(5):
            ab.add(nid, par, 0.5 * (u + 0.2 * rng.standard_normal(8))); par = nid; nid += 1
    worst = 0.0
    for q in range(1, nid):
        for c in range(1, nid):
            dg = dist(ab.global_point(q), ab.global_point(c)) if q != c else 0.0
            worst = max(worst, abs(ab.pair_distance(q, c, work_dtype=np.float64) - dg))
    assert worst < 1e-9, worst

    # T3 deep chain: neighbours at depth 40 stay exact in the LCA frame even in float32,
    #    while float32 global coordinates have collapsed to the boundary
    deep = AddressBook(2)
    u = np.array([0.5, 0.0])                          # hyperbolic step length 1.0
    for k in range(1, 41):
        deep.add(k, k - 1, u)
    d_local = deep.pair_distance(39, 40, work_dtype=np.float32)
    g39, g40 = deep.coords_in_frame(39, 0, np.float32), deep.coords_in_frame(40, 0, np.float32)
    assert abs(d_local - 1.0) < 1e-5, d_local
    assert np.linalg.norm(g40) >= 1.0 - 1e-7 or abs(dist(g39.astype(float), g40.astype(float)) - 1.0) > 0.1

    # T4 M5 restarts every plane: same-offset tokens in different origins get identical positions
    P, B = 64, 512
    c = np.array([0, 3, 7]); l = np.array([100, 100, 100])
    pos = plane_positions("M5", c, l, np.zeros(3, int), B, P)
    assert np.all(pos == 100)

    # T5 OAA factorization and paging error = evicted router mass
    C_ = 8
    s_loc = rng.standard_normal((C_, 64)) * 2
    a_, pi_, al_ = oaa_attention(s_loc, rng.standard_normal(C_), np.abs(rng.standard_normal(C_)) * 2, lam=2.0)
    assert np.abs(a_ - pi_[:, None] * al_).max() < 1e-12
    tv, pred = paging_tv(a_, pi_, resident=[0, 1, 2])
    assert abs(tv - pred) < 1e-12, (tv, pred)

    # T6 beam descent: noise-free finds the target with b + (L-1) w b scores
    children, leaves = make_tree(8, 4)
    parent = {ch: n for n, chs in children.items() for ch in chs}
    target = leaves[1234]
    anc = set()
    n = target
    while n in parent:
        anc.add(n); n = parent[n]
    got, n_scores = descend(children, lambda node: 1.0 if node in anc else 0.0, beam=2)
    assert target in got and n_scores == 8 + 3 * 2 * 8, (got, n_scores)

    # T7 MT generator: cross-thread last hop sits in another thread and is late-bound
    item = make_mt_item(["filler text."] * 50, n_threads=4, origins_per_thread=6, hops=3, rng=rng)
    assert item["thread_of"][item["fact_origins"][-1]] != item["question_thread"]
    assert item["h_late"] >= 1

    print("all 7 self-tests passed;",
          f"isometry err {e1:.1e}, LCA-frame vs global {worst:.1e}, depth-40 neighbour distance {d_local:.6f} (float32),",
          f"paging TV {tv:.4f} = predicted {pred:.4f}, descent scores {n_scores}, MT item h_late {item['h_late']}")
