"""
Numerical checks for the Origin-Paged Attention (OPA) analysis.
Everything here is synthetic (no pretrained weights). It checks the algebra and
the positional geometry; it does not measure language-model quality.

Outputs: figures in ./fig and a JSON summary in ./summary.json
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
os.makedirs("fig", exist_ok=True)
S = {}

d = 128            # head dim
P = d // 2         # number of 2-D rotation planes
B = 512            # chunk size
base = 10000.0
theta = base ** (-2 * np.arange(P) / d)          # standard RoPE frequencies (high -> low)

def rot(x, ang):
    """Rotate each 2-D plane of x (..., d) by angles ang (P,) or (..., P)."""
    xr = x.reshape(*x.shape[:-1], P, 2)
    c, s = np.cos(ang), np.sin(ang)
    out = np.stack([xr[..., 0] * c - xr[..., 1] * s, xr[..., 0] * s + xr[..., 1] * c], -1)
    return out.reshape(x.shape)

# ---------------------------------------------------------------- 1. exactness
k = rng.standard_normal(d)
m, delta = 1234, 5000
err_rerot = np.abs(rot(rot(k, m * theta), delta * theta) - rot(k, (m + delta) * theta)).max()
c_, l_ = 7, 301
err_fact = np.abs(rot(rot(k, l_ * theta), c_ * B * theta) - rot(k, (c_ * B + l_) * theta)).max()
S["rerotation_max_abs_error"] = float(err_rerot)
S["factorisation_max_abs_error"] = float(err_fact)

# ---------------------------------------------------- 2. positional kernel geometry
# Alignment kernel: K = (1/P) * sum_i cos(angle_i(query) - angle_i(key)).
# Equals the normalised logit when q and k have identical, unit-magnitude content in
# every plane: the envelope RoFormer uses for its long-term-decay argument.
n_loc = 48                 # planes given to the local offset (Gemini's k = 48)
n_org = P - n_loc          # 16 planes given to the origin id (Gemini's p = 16)
th_loc = base ** (-2 * np.arange(n_loc) / (2 * n_loc))   # Gemini eq. for theta^loc
ph_org = base ** (-2 * np.arange(n_org) / (2 * n_org))   # Gemini eq. for phi^orig

C = 16
cq, lq = 8, 100
cs = np.repeat(np.arange(C), B)
ls = np.tile(np.arange(B), C)

def kernel_standard(cq, lq, cs, ls):
    dm = (cq * B + lq) - (cs * B + ls)
    return np.cos(np.outer(dm, theta)).mean(1)

def kernel_orope(cq, lq, cs, ls):
    a = np.cos(np.outer(lq - ls, th_loc)).sum(1)
    b = np.cos(np.outer(cq - cs, ph_org)).sum(1)
    return (a + b) / P

def kernel_hybrid(cq, lq, cs, ls, slot_of):
    # local planes: virtual working-window position (slot*B + l); origin planes: archival c
    sq = slot_of[cq]; sk = slot_of[cs]
    a = np.cos(np.outer((sq * B + lq) - (sk * B + ls), th_loc)).sum(1)
    b = np.cos(np.outer(cq - cs, ph_org)).sum(1)
    return (a + b) / P

Ks = kernel_standard(cq, lq, cs, ls)
Ko = kernel_orope(cq, lq, cs, ls)
slot_of = np.arange(C)            # every chunk resident, slot = archival order
Kh = kernel_hybrid(cq, lq, cs, ls, slot_of)

ghost = (ls == lq) & (cs != cq)
self_ = (ls == lq) & (cs == cq)
S["kernel_at_self"] = float(Ko[self_][0])
S["ghost_kernel_mean_ORoPE"] = float(Ko[ghost].mean())
S["ghost_kernel_mean_standard"] = float(Ks[ghost].mean())
S["ghost_kernel_mean_hybrid"] = float(Kh[ghost].mean())
# position-only margin between chunk 0 and chunk 2 at the same offset (Gemini's "proof")
S["ORoPE_margin_dc0_vs_dc2"] = float(1 - (n_loc + np.cos(2 * ph_org).sum()) / P)

# softmax mass of a purely positional head (q = k content) on ghost slots
def ghost_mass(K, scale=12.0):
    w = np.exp(scale * (K - K.max())); w /= w.sum()
    return float(w[ghost].sum()), float(w[cs == cq].sum())
for name, K in [("standard", Ks), ("ORoPE", Ko), ("hybrid", Kh)]:
    g, own = ghost_mass(K)
    S[f"positional_head_mass_on_ghosts_{name}"] = g
    S[f"positional_head_mass_on_own_chunk_{name}"] = own

fig, ax = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
x = cs * B + ls
for a_, K, t in [(ax[0], Ks, "Standard RoPE (archival positions)"),
                 (ax[1], Ko, "O-RoPE as specified (48 local planes + 16 origin planes)"),
                 (ax[2], Kh, "Hybrid: virtual-slot positions in local planes + origin id in 16 planes")]:
    a_.plot(x, K, lw=0.6, color="#3b5b92")
    a_.scatter(x[ghost], K[ghost], s=10, color="#c0392b", zorder=3, label="same offset, other chunk")
    a_.axvline(cq * B + lq, color="k", lw=0.8, ls=":")
    a_.set_title(t, fontsize=10, loc="left"); a_.set_ylim(-0.3, 1.05); a_.set_ylabel("kernel")
    a_.grid(alpha=0.25)
ax[0].legend(loc="upper left", fontsize=8, frameon=False)
ax[2].set_xlabel(f"key position (16 chunks x B={B}); query at chunk {cq}, offset {lq} (dotted line)")
fig.tight_layout(); fig.savefig("fig/fig1_positional_kernel.png", dpi=160); plt.close(fig)

# ------------------------------------------- 3. deeper-layer KV depends on context
# Toy pre-LN transformer, random weights, causal attention with RoPE.
dm_, L_ = 64, 4
Pm = dm_ // 2
th_m = base ** (-2 * np.arange(Pm) / dm_)
def rot_m(X, pos):
    Xr = X.reshape(X.shape[0], Pm, 2); ang = np.outer(pos, th_m)
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([Xr[..., 0] * c - Xr[..., 1] * s, Xr[..., 0] * s + Xr[..., 1] * c], -1).reshape(X.shape)
W = [{k_: rng.standard_normal((dm_, dm_)) / np.sqrt(dm_) for k_ in "qkvo"} | {"f1": rng.standard_normal((dm_, 4*dm_))/np.sqrt(dm_), "f2": rng.standard_normal((4*dm_, dm_))/np.sqrt(4*dm_)} for _ in range(L_)]
def ln(X): return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
def forward_keys(X, pos):
    keys = []; H = X.copy(); T = len(X)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for w in W:
        Z = ln(H)
        Q = rot_m(Z @ w["q"], pos); K = rot_m(Z @ w["k"], pos); V = Z @ w["v"]
        keys.append(K)
        A = Q @ K.T / np.sqrt(dm_) + mask
        A = np.exp(A - A.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
        H = H + (A @ V) @ w["o"]
        H = H + np.maximum(ln(H) @ w["f1"], 0) @ w["f2"]
    return keys
Tc = 64
A_ctx = rng.standard_normal((Tc, dm_)); Cchunk = rng.standard_normal((Tc, dm_))
k_alone = forward_keys(Cchunk, np.arange(Tc))                       # chunk encoded by itself
k_inctx = forward_keys(np.vstack([A_ctx, Cchunk]), np.arange(2*Tc))   # chunk encoded after A
cos_by_layer = []
for l in range(L_):
    ka = rot_m(k_alone[l], np.full(Tc, Tc))          # re-rotate isolated keys by +Tc (exact RoPE fix)
    kc = k_inctx[l][Tc:]
    cos = (ka * kc).sum(1) / (np.linalg.norm(ka, axis=1) * np.linalg.norm(kc, axis=1))
    cos_by_layer.append(float(cos.mean()))
S["toy_key_cosine_isolated_vs_incontext_by_layer"] = cos_by_layer

# ---------------------------------- 4. paraphrase drift vs reference pinning (model)
eps, T = 0.02, 50
S["summary_retention_after_50_rewrites_eps_0.02"] = float((1 - eps) ** T)
S["summary_retention_after_50_rewrites_eps_0.005"] = float((1 - 0.005) ** T)

# ------------------- 5. fold-left + skeleton page-fault backtracking (logic simulation)
def simulate(n_chunks=24, hops=3, late=True, policy="fold+skeleton", gamma=1.0, trials=4000):
    """A chain e0 -> e1 -> ... -> e_h. The fact (e_j -> e_{j+1}) lives in a random chunk.
    late=True: fact for hop j sits in a chunk *earlier* than the chunk that reveals e_j
    is relevant, so a single forward pass that keeps only currently relevant facts misses it.
    policy: 'topk' (retrieve chunks mentioning e0 only), 'fold' (forward pass, keep relevant),
            'fold+skeleton' (forward pass + page-faults via a skeleton index with coverage gamma)."""
    ok = 0; faults = []
    for _ in range(trials):
        # chunk holding hop-j fact
        if late:
            loc = sorted(rng.choice(n_chunks, hops, replace=False))[::-1]  # later hops in earlier chunks
        else:
            loc = sorted(rng.choice(n_chunks, hops, replace=False))
        if policy == "topk":
            ok += int(hops == 1); faults.append(0); continue
        known = 0; f = 0
        for c in range(n_chunks):                 # forward fold
            while known < hops and loc[known] == c:
                known += 1
        # after forward pass, 'known' hops resolved in order; remaining need backtracking
        while known < hops and policy == "fold+skeleton":
            if rng.random() < gamma:              # skeleton indexed the entity e_known
                f += 1; known += 1
                # continue forward from that chunk for any hops that now chain onward
            else:
                break
        ok += int(known == hops); faults.append(f)
    return ok / trials, float(np.mean(faults))

sim = {}
for pol in ["topk", "fold", "fold+skeleton"]:
    sim[pol] = simulate(policy=pol)
gam = np.linspace(0.5, 1.0, 11)
curve = [simulate(policy="fold+skeleton", gamma=g, trials=3000)[0] for g in gam]
S["fold_sim_late_binding_3hop"] = {k_: {"success": v[0], "mean_page_faults": v[1]} for k_, v in sim.items()}
S["fold_sim_success_vs_gamma"] = {f"{g:.2f}": c for g, c in zip(gam, curve)}

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(gam, curve, marker="o", color="#3b5b92", label="simulated")
ax.plot(gam, gam ** 2, ls="--", color="#c0392b", label=r"$\gamma^{2}$ (2 late-bound hops)")
ax.set_xlabel(r"skeleton coverage $\gamma$"); ax.set_ylabel("answer success")
ax.set_title("3-hop chain, late binding: fold-left + skeleton backtracking", fontsize=10, loc="left")
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig/fig2_skeleton_coverage.png", dpi=160); plt.close(fig)

# ------------------------------- 6. M4 origin-plane deviation and KV page-in arithmetic
dev = (B - 1) * theta[P - n_org:]                 # g(c) = c*B quantization on the 16 lowest planes
S["M4_origin_plane_deviation_rad_min_max"] = [float(dev.min()), float(dev.max())]
layers, kv_heads, head_dim, bytes_per = 32, 8, 128, 2   # Llama-3-8B-shaped model, 16-bit KV
per_token = 2 * layers * kv_heads * head_dim * bytes_per
S["kv_bytes_per_token"] = per_token
S["kv_MiB_per_512_token_origin"] = per_token * 512 / 2**20
S["copy_ms_at_25GBps"] = per_token * 512 / 25e9 * 1e3
S["prefill_flops_8B_512_tokens"] = 2 * 8e9 * 512

json.dump(S, open("summary.json", "w"), indent=2)
print(json.dumps(S, indent=2))
