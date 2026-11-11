"""Reference implementations for the OPA experiment protocol (numpy, tested).
Port to torch by swapping np for torch; Track B unit tests compare against these."""
import random
import numpy as np

# ---------------------------------------------------------------- position maps
def inv_freq(head_dim, base=10000.0):
    """Per-plane frequencies, shape [P]; plane 0 is the highest frequency."""
    return base ** (-np.arange(0, head_dim, 2) / head_dim)

def plane_positions(map_name, c, l, slot, B, P, n_local, g=None):
    """Per-plane position ids, shape [T, P].
    c: origin index, l: offset in origin, slot: working slot (int arrays [T]).
    M1 archival, M2 virtual slot, M3 split (local planes <- l, origin planes <- c),
    M4 hybrid (local planes <- slot*B + l, origin planes <- g(c))."""
    g = g or (lambda c_, B_: c_ * B_)
    arch = np.repeat((c * B + l)[:, None], P, axis=1).astype(float)
    virt = np.repeat((slot * B + l)[:, None], P, axis=1).astype(float)
    if map_name == "M1":
        return arch
    if map_name == "M2":
        return virt
    pos = np.empty((len(c), P))
    if map_name == "M3":
        pos[:, :n_local] = l[:, None]
        pos[:, n_local:] = c[:, None]
        return pos
    if map_name == "M4":
        pos[:, :n_local] = virt[:, :n_local]
        pos[:, n_local:] = g(c, B)[:, None]
        return pos
    raise ValueError(map_name)

def rope_cos_sin(pos, freqs):
    """pos [T, P], freqs [P] -> cos, sin [T, 2P] in the Hugging Face rotate_half layout."""
    ang = pos * freqs[None, :]
    ang = np.concatenate([ang, ang], axis=-1)
    return np.cos(ang), np.sin(ang)

def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

def reindex(k_rot, delta_pos, freqs):
    """Lemma 1: shift cached (already rotated) keys by delta_pos [T, P] per plane."""
    cos, sin = rope_cos_sin(delta_pos, freqs)
    return apply_rope(k_rot, cos, sin)

# ---------------------------------------------------------------- LB-MHQA generator
def make_item(filler_chunks, n_origins, hops, placement, rng):
    """One late-binding multi-hop item. filler_chunks: list of filler strings (one per origin).
    Returns origins (list of str), question, answer, fact_origin (list), h_late."""
    names = [f"Ent{rng.randrange(10**6):06d}" for _ in range(hops + 1)]
    answer = f"{rng.randrange(40, 120)} C"
    facts = [f"{names[j]} sources parts from {names[j + 1]}." for j in range(hops - 1)]
    facts.append(f"Parts from {names[hops - 1]} fail above {answer}.")
    slots = sorted(rng.sample(range(n_origins), hops))
    if placement == "late":
        slots = slots[::-1]                       # hop j's fact before hop j-1's
    elif placement == "mixed":
        rng.shuffle(slots)
    origins = [filler_chunks[i % len(filler_chunks)] for i in range(n_origins)]
    for f, s in zip(facts, slots):
        origins[s] = origins[s] + " " + f
    # hop j is late-bound if its fact sits before the origin that made its entity relevant
    h_late = sum(1 for j in range(1, hops) if slots[j] < slots[j - 1])
    question = f"Above what temperature do parts in {names[0]}'s supply chain fail?"
    return dict(origins=origins, question=question, answer=answer,
                fact_origin=slots, h_late=h_late)

# ---------------------------------------------------------------- tests
if __name__ == "__main__":
    d, P, B, n_local = 128, 64, 512, 48
    fr = inv_freq(d)
    rng = np.random.default_rng(0)
    T = 4 * B
    c = np.repeat(np.arange(4), B); l = np.tile(np.arange(B), 4); slot = c.copy()
    k = rng.standard_normal((T, d))

    # 1. M2 with slot == c equals M1 (contiguous, archival order)
    assert np.allclose(plane_positions("M1", c, l, slot, B, P, n_local),
                       plane_positions("M2", c, l, slot, B, P, n_local))
    # 2. Re-indexing (Lemma 1) in the rotate_half layout
    p1 = plane_positions("M1", c, l, slot, B, P, n_local)
    k1 = apply_rope(k, *rope_cos_sin(p1, fr))
    delta = np.full_like(p1, 3000.0)
    err = np.abs(reindex(k1, delta, fr) - apply_rope(k, *rope_cos_sin(p1 + delta, fr))).max()
    assert err < 1e-9, err
    # 3. M4 with g(c)=c*B differs from M1 only on origin planes, by at most (B-1)*freq
    p4 = plane_positions("M4", c, l, slot, B, P, n_local)
    dev = np.abs((p4 - p1) * fr[None, :])
    assert np.allclose(dev[:, :n_local], 0) and dev.max() <= (B - 1) * fr[n_local] + 1e-12
    # 4. M3 ghost: same offset in another origin has identical local-plane angles
    p3 = plane_positions("M3", c, l, slot, B, P, n_local)
    i, j = 100, 2 * B + 100
    assert np.allclose(p3[i, :n_local], p3[j, :n_local])
    # 5. Generator: late placement makes every hop after the first late-bound
    rr = random.Random(0)
    item = make_item(["filler text"] * 8, 24, 3, "late", rr)
    assert item["h_late"] == 2 and len(item["origins"]) == 24
    item = make_item(["filler text"] * 8, 24, 3, "forward", rr)
    assert item["h_late"] == 0
    print("all protocol snippet tests passed; reindex err", err, "max M4 dev", dev.max())
