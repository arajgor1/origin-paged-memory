"""Origin-Anchored Attention (OAA) and fresh-frame position maps for Hugging Face Qwen3 / Qwen2 / Llama models.

Qwen3 (target, Qwen3-4B: 36 layers, 32 query heads, 8 KV heads, head_dim 128): q/k per-head RMSNorm is applied inside the stock
attention module before RoPE; the "opa_oaa" function receives the post-norm, post-RoPE tensors, so no change is needed here.

Protocol Section 5 ("Implementation notes", "Where the bias goes"); paper Sections 5.1, 6.1, 6.6.

What this module provides
-------------------------
* Position maps over a layout of segments (one segment = one co-loaded origin, or an unanchored span):
    M1 archival      p = origin * B + l
    M2 virtual slot  p = slot * B + l
    M5 fresh frame   p = l  (every segment restarts at 0; with an r-token red-dot prefix the prefix gets 0..r-1 and
                     the origin's own tokens r..r+B-1, i.e. the segment is simply numbered 0..len-1)
  `position_ids()` returns them; `layout_inputs()` returns the `position_ids` and an explicit attention mask to pass
  to the model. The mask is passed as a dict {"full_attention": mask} so transformers does NOT build its own: with
  restarting position ids and no mask, transformers 5 would detect a "packed sequence" and silently block attention
  between segments.
* An attention function registered with transformers' AttentionInterface under the name "opa_oaa" (plus the
  matching mask function). `use_opa_attention(model)` switches a model to it. With OAA inactive it is exactly
  scaled_dot_product_attention with a float mask (0 / finfo.min), so every forward pass of the unit tests goes
  through one and the same kernel.
* OAA parameters per layer (`install_oaa`) and a context manager (`oaa_inputs`) that supplies, for one forward pass,
  the origin of every key and query token and the origin tables (addresses o(c), pairwise distances d(c, c')).

The OAA logit (paper Definition 4)
----------------------------------
    s(u, v) = q_u . k_v / sqrt(d)  +  b_u(c_v),      b_u(c) = r_u^T A o(c) - lambda * d(o(c_u), o(c))
    v_v    <- W_V h_v + g U o(c_v)

Design choices (recorded here so the paper and the decision log can cite them)
* r_u = W_r h_u, where h_u is the attention block's input (the post-input_layernorm hidden state, the same tensor
  W_Q is applied to), as in paper Definition 4. It is captured by a forward pre-hook on the attention module.
  Granularity: one r_u in R^m per KV group (shared by the query heads of that group) and one A in R^{m x m} per KV
  group. At m = 32 on Qwen2.5-1.5B (hidden 1536, 2 KV groups, 28 layers) W_r + A + U + g + lambda are about
  3.0 M parameters, which is the budget stated in protocol 5.4. On Qwen3-4B (hidden 2560, 8 KV groups, 36 layers) the
  same formula gives 36 x (2560x256 + 8x32x32 + 32x1024 + 32 + 1) = about 36 x 0.70 M = 25 M parameters (fp32, 0.1 GB).
* A is initialised to 0 (a_init_std = 0) so an installed OAA starts as the M5 model; W_r and U get N(0, 0.02).
* lambda is one slope per query head (ALiBi-like), all heads initialised to `lam0`; the training-free variant V1
  uses equal slopes. g is one scalar per layer, initialised to 0 (protocol 5.4).
* The router and value terms use o(c) as given in the origin tables (by default the global address, composed in
  float64 and stored in float32); the structural term uses the pairwise distance table, computed in float32 in the
  lowest-common-ancestor frame by opa.geometry.AddressBookTorch (or by TreeLayout for the A1 / A3 arms).
* Bias and value term are computed in float32 and cast to the activation dtype only when added to the mask / values.
  In bf16 runs this rounds the bias to bf16 (about 3 significant digits); fp32 runs are unaffected.
* The bias is a dense [batch, heads, Tq, Tk] float mask added to the causal mask and passed to
  torch.nn.functional.scaled_dot_product_attention. That disables the FlashAttention backend (as the protocol
  allows); the memory-efficient or math backend is used.
* The value term is added inside the attention function (after the KV cache update), so the KV cache keeps pure
  content values and the origin term is applied at read time. For a cache this is the same as adding it after
  v_proj, because the term depends only on the key's origin.
* Origin id -1 marks an unanchored span (no origin): its keys get bias 0 and no value term; a query with origin -1
  gets the router term but no structural term.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OPA_ATTN = "opa_oaa"
PROBE = None                    # optional read-only callable(module, query, k_rep, mask, scaling) per attention call
PROBE_OUT = None                # optional read-only callable(module, out [b, H, Tq, d]) after the attention (linear probe of head outputs)
NULL_ORIGIN = -1

# ============================================================================ layouts and position maps


@dataclass
class Segment:
    """A contiguous run of tokens in the working window.

    origin        origin id (archive index; M1 uses origin * B). -1 = unanchored span (system prompt, question...).
    length        number of tokens, including the red-dot prefix if any.
    slot          working slot for M2 (required for anchored segments under M2).
    prefix_len    red-dot prefix tokens at the start of the segment (the parent's last r tokens).
    prefix_origin origin id that owns the prefix KV (default: the parent is not known here, so `origin`).
    start         explicit first position id (overrides the map's rule for this segment).
    """
    origin: int
    length: int
    slot: int | None = None
    prefix_len: int = 0
    prefix_origin: int | None = None
    start: int | None = None


def position_ids(segments: list[Segment], mapping: str, B: int | None = None) -> torch.Tensor:
    """Per-token position ids [T] (long) for maps M1, M2, M5.

    M1 archival:     origin * B - prefix_len + i   (i = 0..length-1; the prefix sits just before the origin)
    M2 virtual slot: slot * B - prefix_len + i
    M5 fresh frame:  i                              (prefix 0..r-1, origin r..r+B-1)
    An unanchored segment (origin -1) under M1/M2 continues from the previous segment unless `start` is given.
    """
    out, prev_end = [], 0
    for s in segments:
        if s.start is not None:
            first = s.start
        elif mapping == "M5":
            first = 0
        elif mapping in ("M1", "M2"):
            if B is None:
                raise ValueError(f"{mapping} needs B")
            if s.origin == NULL_ORIGIN:
                first = prev_end
            elif mapping == "M1":
                first = s.origin * B - s.prefix_len
            else:
                if s.slot is None:
                    raise ValueError("M2 needs a slot for every anchored segment")
                first = s.slot * B - s.prefix_len
        else:
            raise ValueError(f"unknown map {mapping!r} (M1, M2, M5)")
        out.append(torch.arange(first, first + s.length, dtype=torch.long))
        prev_end = first + s.length
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def token_origins(segments: list[Segment]) -> torch.Tensor:
    """Origin id of every token [T] (long). Red-dot prefix tokens belong to `prefix_origin` (default: the segment)."""
    out = []
    for s in segments:
        po = s.origin if s.prefix_origin is None else s.prefix_origin
        out.append(torch.tensor([po] * s.prefix_len + [s.origin] * (s.length - s.prefix_len), dtype=torch.long))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def layout_mask(segments: list[Segment], mode: str = "causal", visible=None, batch: int = 1) -> torch.Tensor:
    """Boolean attention mask [batch, 1, T, T] (True = may attend).

    causal   every token sees every earlier token in the window (co-loaded origins all visible).
    segment  every token sees only earlier tokens of its own segment (each origin encoded alone, as in R1).
    visible  optional [S, S] boolean matrix: segment i may see segment j (j < i) -- e.g. graph-parent masks (B4).
             Within a segment attention is always causal.
    """
    seg_id = torch.cat([torch.full((s.length,), i, dtype=torch.long) for i, s in enumerate(segments)])
    T = seg_id.numel()
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    if mode == "causal" and visible is None:
        m = causal
    else:
        same = seg_id[:, None] == seg_id[None, :]
        if visible is not None:
            vis = torch.as_tensor(np.asarray(visible), dtype=torch.bool)
            allow = vis[seg_id[:, None], seg_id[None, :]] | same
        elif mode == "segment":
            allow = same
        else:
            raise ValueError(mode)
        m = causal & allow
    return m[None, None].expand(batch, 1, T, T).contiguous()


def layout_inputs(segments: list[Segment], mapping: str, B: int | None = None, mode: str = "causal", visible=None,
                  batch: int = 1, device=None) -> dict:
    """Keyword arguments for model(...): explicit position_ids and an explicit mask that bypasses transformers'
    own mask construction (and its packed-sequence detection)."""
    pos = position_ids(segments, mapping, B)[None].expand(batch, -1).to(device)
    mask = layout_mask(segments, mode, visible, batch).to(device)
    return {"position_ids": pos, "attention_mask": {"full_attention": mask, "sliding_attention": mask}}


# ============================================================================ origin tables


@dataclass
class OriginTables:
    """Addresses and pairwise distances of the origins that can appear in a forward pass.

    ids   origin ids, row order of `vec` and `dist`
    vec   [n, m] float32 addresses o(c) used by the router and value terms
    dist  [n, n] float32 structural distances d(c, c') used by the penalty
    A null row (index n, all zeros) is appended internally for origin -1 and unknown ids.
    """
    ids: list[int]
    vec: torch.Tensor
    dist: torch.Tensor
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ids = [int(i) for i in self.ids]
        self.vec = torch.as_tensor(self.vec, dtype=torch.float32)
        self.dist = torch.as_tensor(self.dist, dtype=torch.float32)
        n = len(self.ids)
        if self.vec.shape[0] != n or self.dist.shape != (n, n):
            raise ValueError("OriginTables: vec must be [n, m] and dist [n, n]")
        if not torch.isfinite(self.dist).all():
            raise ValueError("OriginTables: distances must be finite (use d_clamp)")
        self._row = {c: i for i, c in enumerate(self.ids)}

    @property
    def m(self) -> int:
        return int(self.vec.shape[1])

    def padded(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.ids)
        vec = torch.cat([self.vec, torch.zeros(1, self.m)], 0)
        dist = torch.zeros(n + 1, n + 1)
        dist[:n, :n] = self.dist
        return vec.to(device), dist.to(device)

    def rows(self, origin_ids: torch.Tensor) -> torch.Tensor:
        n = len(self.ids)
        flat = [self._row.get(int(c), n) for c in origin_ids.reshape(-1).tolist()]
        return torch.tensor(flat, dtype=torch.long).reshape(origin_ids.shape)


def tables_from_book(book, ids, d_clamp: float = 36.0, vec_work_dtype=torch.float64) -> OriginTables:
    """Tables from an opa.geometry.AddressBook: distances composed AND evaluated in float32 in the LCA frame
    (AddressBookTorch.pair_distance), clamped at d_clamp; addresses o(c) composed in `vec_work_dtype` from the root
    frame and stored in float32."""
    from opa.geometry import AddressBookTorch
    tb = AddressBookTorch(book, dtype=torch.float32)
    ids = [int(i) for i in ids]
    n = len(ids)
    dist = torch.zeros(n, n, dtype=torch.float32)
    for i, q in enumerate(ids):
        for j, c in enumerate(ids):
            if i != j:
                dist[i, j] = tb.pair_distance(q, c, work_dtype=torch.float32, d_clamp=d_clamp,
                                              dist_dtype=torch.float32).to(torch.float32)
    tb64 = AddressBookTorch(book, dtype=vec_work_dtype)
    vec = torch.stack([tb64.coords_in_frame(c, 0, work_dtype=vec_work_dtype) if c != 0
                       else torch.zeros(book.m, dtype=vec_work_dtype) for c in ids]).to(torch.float32)
    return OriginTables(ids, vec, dist, meta={"distance": "hyperbolic_lca_fp32", "d_clamp": d_clamp})


def tables_from_layout(layout, ids, kind: str = "A2") -> OriginTables:
    """Tables from an opa.geometry.TreeLayout. kind: A2/hyperbolic (float32 LCA frame, clamped at layout.d_clamp),
    A1/euclidean, A3/tree, A0/none (all zero distances). Addresses are always the hyperbolic ones."""
    if layout.d_clamp is None:
        raise ValueError("tables_from_layout needs a TreeLayout with a numeric d_clamp (Track B uses the float32 LCA "
                         "frame, which is only valid with a clamp); d_clamp=None is the Track A exact-fallback mode")
    if kind in ("A2", "hyperbolic"):
        t = tables_from_book(layout.book, ids, d_clamp=layout.d_clamp)
        t.meta["arm"] = "A2"
        return t
    base = tables_from_book(layout.book, ids, d_clamp=layout.d_clamp)
    ids = [int(i) for i in ids]
    dist = torch.tensor([[layout.distance(q, c, kind) for c in ids] for q in ids], dtype=torch.float32)
    return OriginTables(ids, base.vec, dist, meta={"distance": kind, "arm": kind})


# ============================================================================ OAA parameters and state


class OAAState:
    """Per-forward inputs shared by all layers (set by `oaa_inputs`)."""

    def __init__(self):
        self.active = False
        self.vec = self.dist = None            # padded tables on device
        self.key_rows = None                   # [b, Tk] rows into the tables
        self.query_rows = None                 # [b, Tq] or None (= last Tq key rows)
        self.router = self.struct = self.value = True
        self.record_layers: set[int] = set()
        self.records: dict[int, dict] = {}
        # paging (U8): in the layers of evict_layers the query rows >= evict_from_query cannot see the keys of the origins in evict_rows
        self.evict_layers: set[int] = set()
        self.evict_rows: set[int] | None = None
        self.evict_from_query = 0


class OAAParams(nn.Module):
    """New parameters of one attention layer (kept in float32 whatever the model dtype)."""

    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int, m: int = 32, lam0: float = 0.0,
                 a_init_std: float = 0.0, wr_init_std: float = 0.02, u_init_std: float = 0.02):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim, self.m = n_heads, n_kv, head_dim, m
        self.W_r = nn.Linear(hidden, n_kv * m, bias=False)
        self.U = nn.Linear(m, n_kv * head_dim, bias=False)
        self.A = nn.Parameter(torch.randn(n_kv, m, m) * a_init_std if a_init_std > 0 else torch.zeros(n_kv, m, m))
        self.lam = nn.Parameter(torch.full((n_heads,), float(lam0)))
        self.g = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.W_r.weight, std=wr_init_std)
        nn.init.normal_(self.U.weight, std=u_init_std)
        self.state: OAAState | None = None     # plain attribute, not a submodule
        self._r = None

    def capture(self, hidden_states: torch.Tensor) -> None:
        st = self.state
        self._r = self.W_r(hidden_states.to(torch.float32)) if (st is not None and st.active and st.router) else None

    def bias_and_value(self, layer_idx: int, query: torch.Tensor, value: torch.Tensor):
        """Return (bias [b, H, Tq, Tk] float32 or None, value with the origin term)."""
        st = self.state
        b, H, Tq, _ = query.shape
        Tk = value.shape[2]
        G = value.shape[1]
        n_rep = H // G
        key_rows = st.key_rows.to(query.device)
        if key_rows.shape[-1] != Tk:
            raise ValueError(f"OAA: {key_rows.shape[-1]} key origins given for {Tk} keys (layer {layer_idx})")
        key_rows = key_rows.expand(b, Tk)
        query_rows = (st.query_rows if st.query_rows is not None else key_rows[:, -Tq:]).to(query.device).expand(b, Tq)
        bias = None
        if st.router and self._r is not None:
            r = self._r.view(b, Tq, G, self.m)
            Ao = torch.einsum("gij,nj->gni", self.A, st.vec)                       # [G, n+1, m]
            ro = torch.einsum("bqgi,gni->bgqn", r, Ao)                              # [b, G, Tq, n+1]
            router = torch.gather(ro, 3, key_rows[:, None, None, :].expand(b, G, Tq, Tk))
            bias = router.repeat_interleave(n_rep, dim=1)                           # head h -> group h // n_rep
        if st.struct:
            dq = st.dist[query_rows]                                                # [b, Tq, n+1]
            dqk = torch.gather(dq, 2, key_rows[:, None, :].expand(b, Tq, Tk))       # [b, Tq, Tk]
            struct = -self.lam.view(1, H, 1, 1) * dqk[:, None]
            bias = struct if bias is None else bias + struct
        if st.value:
            Uo = self.U(st.vec).view(-1, G, self.head_dim)                          # [n+1, G, d]
            vt = Uo[key_rows].permute(0, 2, 1, 3)                                   # [b, G, Tk, d]
            value = value + (self.g * vt).to(value.dtype)
        return bias, value


def _pre_hook(module, args, kwargs):
    p = getattr(module, "opa", None)
    if p is not None:
        h = kwargs.get("hidden_states", args[0] if args else None)
        p.capture(h)
    return None


def attention_modules(model) -> list[nn.Module]:
    base = model.model if hasattr(model, "model") else model
    return [layer.self_attn for layer in base.layers]


def install_oaa(model, m: int = 32, lam0: float = 0.0, a_init_std: float = 0.0, wr_init_std: float = 0.02,
                u_init_std: float = 0.02, seed: int | None = 0) -> OAAState:
    """Attach OAAParams to every attention layer (as `self_attn.opa`), register the hidden-state pre-hooks and switch
    the model to the "opa_oaa" attention function. Returns the shared state object."""
    use_opa_attention(model)
    cfg = model.config
    gen_state = torch.random.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
    st = OAAState()
    device = next(model.parameters()).device
    for mod in attention_modules(model):
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        p = OAAParams(cfg.hidden_size, n_heads, n_kv, mod.head_dim, m=m, lam0=lam0, a_init_std=a_init_std,
                      wr_init_std=wr_init_std, u_init_std=u_init_std).to(device)
        p.state = st
        mod.opa = p
        if not getattr(mod, "_opa_hooked", False):
            mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            mod._opa_hooked = True
    if seed is not None:
        torch.random.set_rng_state(gen_state)
    model._opa_state = st
    return st


def oaa_parameters(model):
    for mod in attention_modules(model):
        p = getattr(mod, "opa", None)
        if p is not None:
            yield mod.layer_idx, p


@contextlib.contextmanager
def oaa_inputs(model, tables: OriginTables, key_origins, query_origins=None, router: bool = True,
               struct: bool = True, value: bool = True, record_layers=(), evict_layers=(), evict_rows=None, evict_from_query: int = 0):
    """Activate OAA for the forward passes inside the block.

    key_origins    [Tk] or [b, Tk] origin id of every key (cached + new), e.g. token_origins(segments)
    query_origins  [Tq] or [b, Tq]; default = the last Tq entries of key_origins
    router / struct / value switch the three terms (e.g. V1 = struct only)
    record_layers  layer indices whose q, k, v, base mask and bias are kept in state.records (for the
                   factorization check)
    evict_*        paging: in the layers `evict_layers`, query rows >= evict_from_query cannot attend to keys whose origin row is in
                   `evict_rows` (their columns get the mask value in the kernel itself; recorded in record["base"])
    """
    st: OAAState = model._opa_state
    device = next(model.parameters()).device
    key_origins = torch.as_tensor(key_origins)
    if key_origins.dim() == 1:
        key_origins = key_origins[None]
    st.vec, st.dist = tables.padded(device)
    st.key_rows = tables.rows(key_origins).to(device)
    if query_origins is not None:
        q = torch.as_tensor(query_origins)
        st.query_rows = tables.rows(q[None] if q.dim() == 1 else q).to(device)
    else:
        st.query_rows = None
    st.router, st.struct, st.value = router, struct, value
    st.record_layers, st.records = set(record_layers), {}
    st.evict_layers, st.evict_from_query = set(evict_layers), int(evict_from_query)
    st.evict_rows = None if evict_rows is None else {int(r) for r in evict_rows}
    st.active = True
    try:
        yield st
    finally:
        st.active = False
        st.evict_rows, st.evict_layers, st.evict_from_query = None, set(), 0
        for _, p in oaa_parameters(model):
            p._r = None


# ============================================================================ the attention function


def _float_mask(attention_mask, b: int, Tq: int, Tk: int, dtype, device) -> torch.Tensor:
    """Base mask as float [b, 1, Tq, Tk]: 0 where allowed, finfo.min where masked."""
    if attention_mask is None:                         # transformers skipped the mask: plain causal, bottom-right
        i = torch.arange(Tq, device=device)[:, None] + (Tk - Tq)
        j = torch.arange(Tk, device=device)[None, :]
        attention_mask = (j <= i)[None, None].expand(b, 1, Tq, Tk)
    if attention_mask.dtype == torch.bool:
        out = torch.zeros(attention_mask.shape, dtype=dtype, device=device)
        return out.masked_fill(~attention_mask, torch.finfo(dtype).min)
    return attention_mask.to(dtype)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, g, t, d = x.shape
    return x[:, :, None].expand(b, g, n_rep, t, d).reshape(b, g * n_rep, t, d)


def opa_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                          sliding_window=None, **kwargs):
    """transformers attention interface: query [b, H, Tq, d], key/value [b, G, Tk, d] (post-RoPE, post-cache).
    Returns (output [b, Tq, H, d], None). Always SDPA with a dense float mask of shape [b, H, Tq, Tk]."""
    if sliding_window is not None:
        raise NotImplementedError("opa_oaa does not implement sliding-window layers")
    b, H, Tq, d = query.shape
    Tk = key.shape[2]
    n_rep = H // key.shape[1]
    base = _float_mask(attention_mask, b, Tq, Tk, query.dtype, query.device)
    p = getattr(module, "opa", None)
    st = p.state if p is not None else None
    bias = None
    if st is not None and st.active:
        bias, value = p.bias_and_value(module.layer_idx, query, value)
    if st is not None and st.active and st.evict_rows is not None and module.layer_idx in st.evict_layers:
        kr = st.key_rows.to(base.device).expand(b, Tk)
        cols = torch.isin(kr, torch.as_tensor(sorted(st.evict_rows), device=base.device, dtype=kr.dtype))
        qsel = torch.arange(Tq, device=base.device) >= st.evict_from_query
        base = base.masked_fill(cols[:, None, None, :] & qsel[None, None, :, None], torch.finfo(base.dtype).min)
    if bias is None:
        mask = base.expand(b, H, Tq, Tk).contiguous()
    else:
        mask = base.expand(b, H, Tq, Tk) + bias.to(query.dtype)
    k_rep, v_rep = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    if PROBE is not None:                      # additive helper: opa.trackb.metrics.AttentionProbe (read-only)
        PROBE(module, query, k_rep, mask, scaling)
    if st is not None and st.active and module.layer_idx in st.record_layers:
        st.records[module.layer_idx] = {
            "query": query.detach(), "key": k_rep.detach(), "value": v_rep.detach(), "base": base.detach(),
            "bias": None if bias is None else bias.detach(), "scaling": scaling,
            "key_rows": st.key_rows.detach().clone(),
        }
    out = F.scaled_dot_product_attention(query, k_rep, v_rep, attn_mask=mask,
                                         dropout_p=dropout if module.training else 0.0, scale=scaling)
    if st is not None and st.active and module.layer_idx in st.record_layers:
        st.records[module.layer_idx]["output"] = out.detach()
    if PROBE_OUT is not None:
        PROBE_OUT(module, out)
    return out.transpose(1, 2).contiguous(), None


_REGISTERED = False


def register() -> None:
    """Register "opa_oaa" with transformers' attention and mask interfaces (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register(OPA_ATTN, opa_attention_forward)
    AttentionMaskInterface.register(OPA_ATTN, sdpa_mask)
    _REGISTERED = True


def use_opa_attention(model) -> None:
    register()
    if model.config._attn_implementation != OPA_ATTN:
        model.set_attn_implementation(OPA_ATTN)


@contextlib.contextmanager
def attn_implementation(model, name: str):
    """Temporarily switch the model's attention implementation (e.g. to stock "sdpa" or "eager")."""
    old = model.config._attn_implementation
    if name == OPA_ATTN:
        register()
    model.set_attn_implementation(name)
    try:
        yield
    finally:
        model.set_attn_implementation(old)


# ============================================================================ router factorization (Theorem 6)


def factorization_check(record: dict, head: int, query_pos: int, batch: int = 0) -> dict:
    """Paper Theorem 6 on one recorded head and query: a(u, v) = pi_u(c_v) * alpha_u(v | c_v).

    a      full attention weights softmax(s_loc + b) over the allowed keys
    pi     router masses softmax_c(b_u(c) + Lambda_u(c)), Lambda = local log-partition of origin c
    alpha  local attention exp(s_loc - Lambda(c_v))
    Also checks that a @ v reproduces the SDPA output of that head. All in float64 from the recorded fp32 tensors.
    """
    q = record["query"][batch, head, query_pos].double()
    k = record["key"][batch, head].double()
    v = record["value"][batch, head].double()
    scale = record["scaling"] if record["scaling"] is not None else q.shape[-1] ** -0.5
    allowed = record["base"][batch, 0, query_pos] == 0
    s_loc = (k @ q) * scale
    bias = record["bias"][batch, head, query_pos].double() if record["bias"] is not None else torch.zeros_like(s_loc)
    rows = record["key_rows"].expand(record["base"].shape[0], -1)[batch]
    idx = allowed.nonzero().squeeze(-1)
    s_loc, bias, rows_a, v_a = s_loc[idx], bias[idx], rows[idx], v[idx]
    a = torch.softmax(s_loc + bias, 0)
    origins = rows_a.unique()
    lam = torch.stack([torch.logsumexp(s_loc[rows_a == c], 0) for c in origins])
    b_c = torch.stack([bias[rows_a == c][0] for c in origins])
    const = torch.stack([(bias[rows_a == c] - bias[rows_a == c][0]).abs().max() for c in origins]).max()
    pi = torch.softmax(b_c + lam, 0)
    pos = {int(c): i for i, c in enumerate(origins.tolist())}
    ci = torch.tensor([pos[int(c)] for c in rows_a.tolist()], device=s_loc.device)
    alpha = torch.exp(s_loc - lam[ci])
    fact_err = float((a - pi[ci] * alpha).abs().max())
    out_err = float(((a @ v_a) - record["output"][batch, head, query_pos].double()).abs().max())
    return {"factorization_max_abs_err": fact_err, "sdpa_output_max_abs_err": out_err,
            "bias_constant_within_origin_max_dev": float(const), "router_mass": pi.tolist(),
            "origins_rows": origins.tolist(), "n_keys": int(idx.numel())}


# ============================================================================ paging identity (U8)


def attention_weights(record: dict, head: int, query_pos: int, batch: int = 0) -> dict:
    """Float64 attention of one recorded head and query from the recorded q, k, bias and base mask: the full weight vector over all
    keys (0 where masked), the router masses pi per origin row (Theorem 6) and a @ v."""
    q = record["query"][batch, head, query_pos].double()
    k = record["key"][batch, head].double()
    v = record["value"][batch, head].double()
    scale = record["scaling"] if record["scaling"] is not None else q.shape[-1] ** -0.5
    allowed = record["base"][batch, 0, query_pos] == 0
    s_loc = (k @ q) * scale
    bias = record["bias"][batch, head, query_pos].double() if record["bias"] is not None else torch.zeros_like(s_loc)
    rows = record["key_rows"].expand(record["base"].shape[0], -1)[batch]
    idx = allowed.nonzero().squeeze(-1)
    a_sub = torch.softmax((s_loc + bias)[idx], 0)
    a = torch.zeros_like(s_loc)
    a[idx] = a_sub
    rows_a, s_a, b_a = rows[idx], s_loc[idx], bias[idx]
    origins = rows_a.unique()
    lam = torch.stack([torch.logsumexp(s_a[rows_a == c], 0) for c in origins])
    b_c = torch.stack([b_a[rows_a == c][0] for c in origins])
    pi = torch.softmax(b_c + lam, 0)
    return {"a": a, "pi": {int(c): float(p) for c, p in zip(origins.tolist(), pi)}, "out": a @ v, "rows": rows}


def paging_check(rec_full: dict, rec_paged: dict, head: int, query_pos: int, evicted_rows, batch: int = 0) -> dict:
    """Paging identity from TWO real kernel passes. rec_full: a pass with every origin resident. rec_paged: a pass of the same
    forward in which the kernel itself hid the keys of the origins in `evicted_rows` from the query rows (oaa_inputs(evict_*)), in
    the layer being checked, so its q, k, v, bias in that layer are those of the full pass. Then
      TV = 0.5 * sum |a_full - a_paged|  (each from its own recorded pass)   must equal   evicted router mass = sum_{c evicted} pi_c
    with pi from the full pass. Also reported: that the paged pass puts exactly zero weight on the evicted keys, that q, k, v of the two
    passes coincide in this layer (the precondition of the identity), and that the paged pass's SDPA output equals a_paged @ v."""
    full = attention_weights(rec_full, head, query_pos, batch)
    paged = attention_weights(rec_paged, head, query_pos, batch)
    ev = [int(r) for r in evicted_rows]
    tv = float(0.5 * (full["a"] - paged["a"]).abs().sum())
    evicted_mass = float(sum(full["pi"].get(r, 0.0) for r in ev))
    ev_keys = torch.isin(full["rows"], torch.as_tensor(ev, dtype=full["rows"].dtype, device=full["rows"].device))
    inputs_err = max(float((rec_full[n][batch, head] - rec_paged[n][batch, head]).abs().max()) for n in ("query", "key", "value"))
    return {"tv": tv, "evicted_router_mass": evicted_mass, "abs_err": abs(tv - evicted_mass),
            "paged_weight_on_evicted_keys": float(paged["a"][ev_keys].sum()),
            "paged_sdpa_output_max_abs_err": float((paged["out"] - rec_paged["output"][batch, head, query_pos].double()).abs().max()),
            "inputs_max_abs_diff_between_passes": inputs_err, "n_evicted_origins": len(ev), "n_origins": len(full["pi"])}
