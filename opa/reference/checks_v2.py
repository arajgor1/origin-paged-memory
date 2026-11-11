"""
v2 numerical checks for Origin-Paged Attention with a hyperbolic shell memory.
Synthetic and seeded (numpy, seed 0). Tests geometry, algebra and routing logic,
not language-model quality. Writes fig/fig3..fig6 and summary_v2.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
os.makedirs("fig", exist_ok=True)
S = {}
d, P, B = 128, 64, 512
theta = 10000.0 ** (-2 * np.arange(P) / d)

# ------------------------------------------------ Poincare-ball primitives (curvature -1)
def mobius_add(x, y):
    xy, x2, y2 = np.dot(x, y), np.dot(x, x), np.dot(y, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / (1 + 2 * xy + x2 * y2)

def dist(x, y):
    diff = np.dot(x - y, x - y)
    return np.arccosh(1 + 2 * diff / ((1 - np.dot(x, x)) * (1 - np.dot(y, y))))

def expmap0(v):
    n = np.linalg.norm(v)
    return np.tanh(n) * v / n if n > 0 else v

# Mobius translation is an isometry: d(-a (+) x, -a (+) y) == d(x, y)
a = expmap0(rng.standard_normal(4) * 0.4); x = expmap0(rng.standard_normal(4) * 0.6); y = expmap0(rng.standard_normal(4) * 0.6)
S["mobius_isometry_abs_error"] = float(abs(dist(mobius_add(-a, x), mobius_add(-a, y)) - dist(x, y)))

# ------------------------------------------------ 1. margin needed for own-chunk focus
def own_share(beta, C):
    return np.exp(beta) / (np.exp(beta) + C - 1)
S["beta_needed_90pct_C16"] = float(np.log(0.9 * 15 / 0.1))
S["own_share_beta0_C16"] = float(own_share(0.0, 16))

# fresh frames (all planes restart at 0): same-offset kernel is exactly 1
K5_same_offset = float(np.cos(0 * theta).mean())
S["fresh_frame_same_offset_kernel"] = K5_same_offset

# ------------------------------------------------ 2. same-shell arc vs radial distance
def d_same_radius(R, deg):
    t = np.radians(deg)
    return float(np.arccosh(np.cosh(R) ** 2 - np.sinh(R) ** 2 * np.cos(t)))
S["arc_distance_table"] = {f"R={R}": {f"{deg}deg": round(d_same_radius(R, deg), 2) for deg in (1, 10, 45, 90, 180)} for R in (1, 3, 6)}
S["circumference_hyperbolic_vs_flat"] = {f"R={R}": [round(2 * np.pi * np.sinh(R), 1), round(2 * np.pi * R, 1)] for R in (1, 3, 6, 10)}

# thread directions in 128-D
dirs = rng.standard_normal((4096, 128)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
G = dirs @ dirs.T; np.fill_diagonal(G, 0)
S["4096_thread_directions_mean_abs_cos"] = float(np.abs(G).mean()); S["4096_thread_directions_max_abs_cos"] = float(np.abs(G).max())

# ------------------------------------------------ 3. precision: global coordinates vs re-anchoring
def round_bf16(v):
    v = np.asarray(v, dtype=np.float32)
    i = v.view(np.uint32).astype(np.uint64)
    i = ((i + 0x7FFF + ((i >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return i.view(np.float32).astype(np.float64)

def stepdist_global(n, delta, rnd):
    r0, r1 = rnd(np.tanh(n * delta / 2)), rnd(np.tanh((n + 1) * delta / 2))
    if r1 >= 1.0 or r0 >= 1.0:
        return np.nan
    return float(np.arccosh(1 + 2 * (r1 - r0) ** 2 / ((1 - r0 ** 2) * (1 - r1 ** 2))))

rounders = {"float32": lambda v: float(np.float32(v)), "float16": lambda v: float(np.float16(v)), "bfloat16": lambda v: float(round_bf16(v))}
delta = 1.0
depths = np.arange(0, 20)
err = {}
for name, rnd in rounders.items():
    e = []
    for n in depths:
        g = stepdist_global(n, delta, rnd)
        e.append(np.nan if np.isnan(g) else abs(g - delta) / delta)
    err[name] = e
# re-anchored: the next point is always expressed relative to the previous red dot -> radius tanh(delta/2)
reanch = {name: abs(stepdist_global(0, delta, rnd) - delta) / delta for name, rnd in rounders.items()}
S["reanchored_step_rel_error"] = reanch
S["first_depth_where_global_coords_fail"] = {name: int(next((n for n, v in zip(depths, e) if np.isnan(v) or v > 0.1), -1)) for name, e in err.items()}

fig, ax = plt.subplots(figsize=(8.6, 3.4))
cols = {"float32": "#1f77b4", "float16": "#ff7f0e", "bfloat16": "#2ca02c"}
for name, e in err.items():
    ax.semilogy(depths, np.where(np.isnan(e), 10.0, np.maximum(e, 1e-8)), marker="o", ms=3, color=cols[name], label=f"global coords, {name}")
for name, v in reanch.items():
    ax.axhline(max(v, 1e-8), ls=":", lw=1.2, color=cols[name], label=f"re-anchored, {name} (same at every depth)")
ax.set_ylim(1e-8, 20); ax.set_xlabel("depth along a chain (hyperbolic steps of 1.0)")
ax.set_ylabel("relative error of one step"); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_title("Precision: why the origin must move to each red dot", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("fig/fig5_precision_reanchoring.png", dpi=160); plt.close(fig)

# ------------------------------------------------ 4. structural penalty sweep (fresh frames + hyperbolic distance)
T, Cper = 4, 4                       # 4 threads x 4 chunks
pts, thread_of = [], []
for t in range(T):
    ang = np.pi / 4 + t * np.pi / 2
    for j in range(Cper):
        hyp_r = 1.0 + 0.6 * j        # hyperbolic radius along the thread
        rr = np.tanh(hyp_r / 2)
        jitter = 0.05 * (j - 1.5)
        pts.append(rr * np.array([np.cos(ang + jitter), np.sin(ang + jitter)])); thread_of.append(t)
pts = np.array(pts); thread_of = np.array(thread_of)
C = len(pts); q_chunk = 1
D = np.array([dist(pts[q_chunk], p) if i != q_chunk else 0.0 for i, p in enumerate(pts)])
cs = np.repeat(np.arange(C), B); ls = np.tile(np.arange(B), C); lq = 100
K5 = np.cos(np.outer(lq - ls, theta)).mean(1)              # fresh frame per chunk
lams = np.linspace(0, 6, 61)
own_chunk, own_thread, ghosts_other = [], [], []
ghost = (ls == lq) & (thread_of[cs] != thread_of[q_chunk])
for lam in lams:
    logit = 12.0 * K5 - lam * D[cs]
    w = np.exp(logit - logit.max()); w /= w.sum()
    own_chunk.append(w[cs == q_chunk].sum()); own_thread.append(w[thread_of[cs] == thread_of[q_chunk]].sum()); ghosts_other.append(w[ghost].sum())
S["lambda_sweep"] = {f"{l:.1f}": {"own_chunk": round(float(oc), 3), "own_thread": round(float(ot), 3), "ghosts_other_threads": round(float(g), 4)}
                     for l, oc, ot, g in zip(lams[::10], own_chunk[::10], own_thread[::10], ghosts_other[::10])}
S["hyperbolic_distance_to_same_thread_neighbour"] = float(sorted(D[thread_of == thread_of[q_chunk]])[1])
S["hyperbolic_distance_to_other_thread_min"] = float(D[thread_of != thread_of[q_chunk]].min())

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(lams, own_thread, label="mass on the query's own thread", color="#3b5b92")
ax.plot(lams, own_chunk, label="mass on the query's own chunk", color="#1D9E75")
ax.plot(lams, ghosts_other, label="mass on same-offset ghosts in other threads", color="#c0392b")
ax.set_xlabel(r"penalty slope $\lambda$ (logits per unit of hyperbolic distance)"); ax.set_ylabel("attention mass")
ax.set_title("Fresh frames + hyperbolic penalty: 4 threads x 4 chunks, positional head", fontsize=10, loc="left")
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig/fig4_lambda_sweep.png", dpi=160); plt.close(fig)

# ------------------------------------------------ 5. coarse-to-fine descent vs flat scan under noisy summaries
def descent_success(b, depth, sigma, beam, trials=3000):
    ok = 0
    for _ in range(trials):
        target = [rng.integers(b) for _ in range(depth)]
        frontier = [()]
        for lvl in range(depth):
            cands = [p + (k,) for p in frontier for k in range(b)]
            scores = np.array([(1.0 if list(c) == target[:lvl + 1] else 0.0) + sigma * rng.standard_normal() for c in cands])
            keep = np.argsort(-scores)[:beam]
            frontier = [cands[i] for i in keep]
        ok += any(list(f) == target for f in frontier)
    return ok / trials

def flat_success(n, sigma, trials=3000):
    s = 1.0 + sigma * rng.standard_normal(trials)
    m = sigma * rng.standard_normal((trials, n - 1)).max(1)
    return float((s > m).mean())

b, depth = 8, 4                      # 4096 leaf chunks
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
res = {"flat": [flat_success(b ** depth, s) for s in sigmas]}
for beam in (1, 2, 4):
    res[f"beam{beam}"] = [descent_success(b, depth, s, beam, trials=1500) for s in sigmas]
S["routing_success"] = {k: [round(v, 3) for v in vals] for k, vals in res.items()}
S["routing_sigmas"] = sigmas
S["routing_scores_per_query"] = {"flat": b ** depth, "beam1": b * depth, "beam2": 2 * b * depth - b, "beam4": 4 * b * depth - 3 * b}

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(sigmas, res["flat"], marker="s", color="#888780", label=f"flat scan of {b**depth} chunks")
for beam, col in zip((1, 2, 4), ("#7F77DD", "#534AB7", "#26215C")):
    ax.plot(sigmas, res[f"beam{beam}"], marker="o", color=col, label=f"shell descent, beam {beam}")
ax.set_xlabel(r"score noise $\sigma$ (summary quality, lower is better)"); ax.set_ylabel("finds the right chunk")
ax.set_title("Routing: descent through shells vs one flat scan", fontsize=10, loc="left")
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig/fig6_descent_vs_flat.png", dpi=160); plt.close(fig)

# ------------------------------------------------ 6. picture: shells, threads, cross-link (for the paper)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#A32D2D", lw=1.6))
for hr in (1.2, 2.4, 3.6):
    ax.add_patch(plt.Circle((0, 0), np.tanh(hr / 2), fill=False, color="#A32D2D", lw=0.6, ls="--", alpha=.6))
base = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
leaves = {}
for i, a0 in enumerate(base):
    p1 = np.tanh(1.2 / 2) * np.array([np.cos(a0), np.sin(a0)])
    ax.plot([0, p1[0]], [0, p1[1]], color="gray", lw=.7)
    for j, da in enumerate((-0.45, 0, 0.45)):
        a1 = a0 + da; p2 = np.tanh(2.4 / 2) * np.array([np.cos(a1), np.sin(a1)])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=.6)
        for k, db in enumerate((-0.12, 0.12)):
            a2 = a1 + db; p3 = np.tanh(3.6 / 2) * np.array([np.cos(a2), np.sin(a2)])
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color="gray", lw=.5)
            ax.plot(*p3, "o", color="#D85A30", ms=3); leaves[(i, j, k)] = p3
        ax.plot(*p2, "o", color="#444441", ms=3.5)
    ax.plot(*p1, "o", color="#444441", ms=4.5)
ax.plot(0, 0, "o", color="black", ms=6)
pa, pb = leaves[(0, 2, 1)], leaves[(2, 0, 0)]
ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="-", ls="--", color="#534AB7", connectionstyle="arc3,rad=-0.35"))
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Shells = detail levels; directions = threads; dashed = graph edge", fontsize=9)
fig.tight_layout(); fig.savefig("fig/fig3_shell_memory.png", dpi=160); plt.close(fig)


# ------------------------------------------------ 7. extra checks used by the v2 paper (separate rng so earlier results are unchanged)
r2 = np.random.default_rng(1)

# 7a. factorization: softmax(local + per-chunk bias) == router(chunk) x local softmax within chunk
Cn, Bn = 6, 32
s_loc = r2.standard_normal((Cn, Bn)) * 2.0
bias = r2.standard_normal(Cn) * 3.0
full = s_loc + bias[:, None]
a_full = np.exp(full - full.max()); a_full /= a_full.sum()
lse = np.log(np.exp(s_loc).sum(1))
router = np.exp(bias + lse - (bias + lse).max()); router /= router.sum()
local = np.exp(s_loc - s_loc.max(1, keepdims=True)); local /= local.sum(1, keepdims=True)
S["factorization_router_x_local_max_abs_error"] = float(np.abs(a_full - router[:, None] * local).max())

# 7b. margin needed for 90% own-chunk mass grows like log C
S["beta_needed_90pct_by_C"] = {str(Cc): round(float(np.log(0.9 * (Cc - 1) / 0.1)), 2) for Cc in (4, 16, 64, 256, 1024, 4096)}

# 7c. precision horizon in the query's own frame: a chunk at true distance d is stored at radius tanh(d/2)
def horizon(rnd, tol=0.1):
    for dd in np.arange(0.25, 40.0, 0.25):
        r = rnd(np.tanh(dd / 2))
        if r >= 1.0:
            return float(dd)
        est = 2 * np.arctanh(r)
        if abs(est - dd) / dd > tol:
            return float(dd)
    return 40.0
S["query_frame_precision_horizon_10pct"] = {name: horizon(rnd) for name, rnd in rounders.items()}
# distance beyond which a chunk gets < 1e-4 of the mass relative to a chunk at distance 0, for several lambda
S["attention_horizon_tau1e-4"] = {str(l): round(float(np.log(1e4) / l), 2) for l in (1.0, 2.0, 5.0)}

# 7d. chain of local offsets: store each step as a bf16/fp16 offset of radius tanh(delta/2), compose in float64
def compose_error(rnd, steps=20, delta=1.0):
    ang = r2.uniform(0, 2 * np.pi)
    u = np.array([np.cos(ang), np.sin(ang)])
    x_true, x_q = np.zeros(2), np.zeros(2)
    errs = []
    for n in range(1, steps + 1):
        off_true = np.tanh(delta / 2) * u
        off_q = np.array([rnd(v) for v in off_true])
        x_true = mobius_add(x_true, off_true); x_q = mobius_add(x_q, off_q)
        errs.append(abs(dist(np.zeros(2), x_q) - dist(np.zeros(2), x_true)) / dist(np.zeros(2), x_true))
    return float(max(errs[:12]))   # depths 1..12 stay inside float64 range
S["local_offset_chain_max_rel_error_depth12"] = {name: compose_error(rnd) for name, rnd in rounders.items()}

# 7e. effective coverage when a late-bound fact can be recovered by a skeleton entry OR a graph edge
trials = 200000
gs, gg = 0.8, 0.5
hit = (r2.random(trials) < gs) | (r2.random(trials) < gg)
S["coverage_skeleton_or_graph"] = {"gamma_s": gs, "gamma_g": gg, "predicted": 1 - (1 - gs) * (1 - gg), "simulated": round(float(hit.mean()), 4),
                                   "two_late_hops_skeleton_only": gs ** 2, "two_late_hops_with_graph": round((1 - (1 - gs) * (1 - gg)) ** 2, 4)}


# 7f. thread directions at the ball dimensions we actually suggest (16, 32) as well as 128
dd_stats = {}
for mdim in (16, 32, 128):
    D_ = r2.standard_normal((4096, mdim)); D_ /= np.linalg.norm(D_, axis=1, keepdims=True)
    G_ = np.abs(D_ @ D_.T); np.fill_diagonal(G_, 0)
    dd_stats[str(mdim)] = {"mean_abs_cos": round(float(G_.mean()), 3), "max_abs_cos": round(float(G_.max()), 3),
                           "frac_pairs_within_30deg": float((G_ > np.cos(np.radians(30))).sum() / (4096 * 4095))}
S["thread_directions_by_dim_4096"] = dd_stats

json.dump(S, open("summary_v2.json", "w"), indent=2)
print(json.dumps(S, indent=2))
