"""Training arms (B1, V3, and the untrained V0), the LoRA wrapper and one forward pass for an Example.

Arms (protocol 5.3; PLAN_V3 Section 5: same LoRA budget, data, steps and learning rates for B1 and V3)
  B1  standard attention: archival-style contiguous positions 0..T-1, stock "sdpa" kernel, no origin signal. LoRA only.
  V3  M5 fresh frames + origin-anchored attention with the router term (r_u^T A o(c)) and the value term (g U o(c)); no
      distance penalty (that is V4; lambda is fixed at 0 and not trained unless `struct=True` is requested). LoRA + the new
      parameters W_r, A, U, g.
  V0  M5 only, nothing trained (the model with OAA inactive): the "fresh frames alone" reference of H4.

Training-free variants (protocol 5.3; evaluated by opa.trackb.trainfree)
  V1  M5 + a fixed-lambda hyperbolic penalty: OAA with the structural term only (no router, no value term), lambda = 2 on every head
      (the protocol's initial value, section 5.4), distances = Poincare distance in the lowest-common-ancestor frame on the
      training-free layout of paper Section 7.5 (opa.geometry.TreeLayout: one step outward per origin along its thread's ray).
  B2  APE-style: M5 + one shared prefix at the start of the window (positions 0..p-1) with every origin's fresh frame starting
      at p; no origin signal. (arms.with_shared_prefix transforms the example.)
  B4  Graph-KV-style structured mask: the tokens of an origin attend only to their own origin (causally) and to the whole of their
      graph parent (the previous origin of the thread) if it is co-loaded, whatever the window order; the unanchored tail
      (question / continuation) attends to everything before it. M5 positions, no origin signal. (graph_mask below.)

LoRA: rank r on q_proj, k_proj, v_proj, o_proj of every layer; a minimal in-house implementation (no peft dependency):
y = W x + (alpha / r) * B (A x), A ~ kaiming-uniform, B = 0, A and B kept in float32, cast to the activation dtype in the
forward pass; the base weights are frozen.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from opa.trackb import attention as att
from opa.trackb import data as D


@dataclass(frozen=True)
class Arm:
    name: str
    positions: str                 # "M5" | "contig"
    kernel: str                    # training/eval kernel: "sdpa" (stock) | "opa_oaa"
    oaa: bool = False
    router: bool = True
    value: bool = True
    struct: bool = False
    mask: str = "causal"           # "causal" | "graph" (B4)
    shared_prefix: bool = False    # B2: the example gets a shared prefix before the first origin


ARMS = {
    "B1": Arm("B1", "contig", "sdpa"),
    "V3": Arm("V3", "M5", att.OPA_ATTN, oaa=True),
    "V0": Arm("V0", "M5", att.OPA_ATTN, oaa=False),
    "V1": Arm("V1", "M5", att.OPA_ATTN, oaa=True, router=False, value=False, struct=True),
    "B2": Arm("B2", "M5", att.OPA_ATTN, shared_prefix=True),
    "B4": Arm("B4", "M5", att.OPA_ATTN, mask="graph"),
}
V1_LAMBDA = 2.0                    # fixed lambda of V1 (protocol 5.4: lambda initialised at 2)


def archive_parent(c: int, per_thread: int = D.PER_THREAD) -> int:
    """Graph parent of archive origin c: the previous origin of its thread (0 = root, i.e. none)."""
    return c - 1 if (c - 1) % per_thread else 0


def graph_mask(segments, per_thread: int = D.PER_THREAD) -> torch.Tensor:
    """B4 attention mask [1, 1, T, T] (True = may attend). Rows of an anchored segment: own tokens causally + all tokens of the graph
    parent segment when that origin is in the window (in any window order, so a later-loaded parent is still visible). Rows of an
    unanchored segment (prefix / question / continuation): every earlier token (causal)."""
    seg_id = torch.cat([torch.full((s.length,), i, dtype=torch.long) for i, s in enumerate(segments)])
    T = seg_id.numel()
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    allow = torch.zeros(T, T, dtype=torch.bool)
    seg_of_origin = {s.origin: i for i, s in enumerate(segments) if s.origin != att.NULL_ORIGIN}
    for i, s in enumerate(segments):
        ri = (seg_id == i).nonzero().squeeze(-1)
        if s.origin == att.NULL_ORIGIN:
            allow[ri] = causal[ri]
            continue
        allow[ri[:, None], ri[None, :]] = causal[ri[:, None], ri[None, :]]
        j = seg_of_origin.get(archive_parent(s.origin, per_thread))
        if j is not None:
            ci = (seg_id == j).nonzero().squeeze(-1)
            allow[ri[:, None], ci[None, :]] = True
    return allow[None, None].contiguous()


def with_shared_prefix(ex: D.Example, prefix_ids: list) -> D.Example:
    """B2: put one shared prefix (unanchored, positions 0..p-1) in front of the window; every origin's fresh frame then starts at p
    and the tail follows the largest position. No origin signal is added."""
    p = len(prefix_ids)
    segs = [att.Segment(att.NULL_ORIGIN, p, start=0)]
    o_len = max((s.length for s in ex.segments if s.origin != att.NULL_ORIGIN), default=0)
    for s in ex.segments:
        if s.origin == att.NULL_ORIGIN:
            segs.append(att.Segment(att.NULL_ORIGIN, s.length, start=p + o_len))
        else:
            segs.append(att.Segment(s.origin, s.length, start=p))
    ids = torch.cat([torch.tensor(prefix_ids, dtype=torch.long), ex.ids])
    origin_tok = torch.cat([torch.full((p,), att.NULL_ORIGIN, dtype=torch.long), ex.origin_tok])
    import dataclasses
    return dataclasses.replace(ex, ids=ids, origin_tok=origin_tok, segments=segs, ans_start=ex.ans_start + p)


# ============================================================================ LoRA


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, gen: torch.Generator | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        bound = 1.0 / math.sqrt(base.in_features)
        with torch.no_grad():
            self.A.copy_(((torch.rand(self.A.shape, generator=gen) * 2 - 1) * bound).to(self.A.device)
                         if gen is not None else (torch.rand_like(self.A) * 2 - 1) * bound)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        return y + ((x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)) * self.scale


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def apply_lora(model, r: int = 16, alpha: float = 32.0, seed: int = 0) -> list[nn.Parameter]:
    """Freeze the model and wrap the attention projections. Returns the LoRA parameters (deterministic init)."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    params = []
    for mod in att.attention_modules(model):
        for name in LORA_TARGETS:
            base = getattr(mod, name)
            if isinstance(base, LoRALinear):
                continue
            w = LoRALinear(base, r, alpha, gen)
            setattr(mod, name, w)
            params += [w.A, w.B]
    return params


def lora_state(model) -> dict:
    out = {}
    for i, mod in enumerate(att.attention_modules(model)):
        for n in LORA_TARGETS:
            w = getattr(mod, n)
            if isinstance(w, LoRALinear):
                out[f"{i}.{n}.A"], out[f"{i}.{n}.B"] = w.A.detach().float().cpu(), w.B.detach().float().cpu()
    return out


def load_lora_state(model, st: dict) -> None:
    for i, mod in enumerate(att.attention_modules(model)):
        for n in LORA_TARGETS:
            w = getattr(mod, n)
            if isinstance(w, LoRALinear):
                with torch.no_grad():
                    w.A.copy_(st[f"{i}.{n}.A"].to(w.A.device))
                    w.B.copy_(st[f"{i}.{n}.B"].to(w.B.device))


def load_oaa_state(model, st: dict) -> None:
    for i, p in att.oaa_parameters(model):
        for n, q in p.named_parameters():
            with torch.no_grad():
                q.copy_(st[f"{i}.{n}"].to(q.device))


def oaa_state(model) -> dict:
    return {f"{i}.{n}": q.detach().float().cpu() for i, p in att.oaa_parameters(model) for n, q in p.named_parameters()}


def count_params(params) -> int:
    return int(sum(p.numel() for p in params))


# ============================================================================ forward


def example_kwargs(ex: D.Example, arm: Arm, device, force_mask: bool = False) -> dict:
    """Keyword arguments for model.model(...): position ids and, unless the stock kernel builds its own causal mask,
    an explicit causal mask dict (the "full_attention" key and "sliding_attention" for safety)."""
    pos = D.positions(ex, arm.positions)[None].to(device)
    kw = {"position_ids": pos}
    if arm.kernel != "sdpa" or force_mask:
        mask = (graph_mask(ex.segments) if arm.mask == "graph" else att.layout_mask(ex.segments, "causal")).to(device)
        kw["attention_mask"] = {"full_attention": mask, "sliding_attention": mask}
    return kw


def forward_example(model, ex: D.Example, arm: Arm, tables, backward: bool = False, scale: float = 1.0,
                    probe_kernel: bool = False, oaa_off: bool = False, mapping: str | None = None):
    """Loss and logits over the target tokens of `ex`. With backward=True, (loss * scale).backward() runs INSIDE the OAA
    context (gradient checkpointing re-runs the forward during backward and needs the OAA state still active).
    probe_kernel=True forces the "opa_oaa" kernel with an explicit mask (needed by the attention probe).
    oaa_off=True evaluates without OAA (std-position perplexity); mapping overrides the arm's position map."""
    device = next(model.parameters()).device
    a = arm
    if mapping is not None and mapping != arm.positions:
        import dataclasses
        a = dataclasses.replace(arm, positions=mapping)
    use_oaa = a.oaa and not oaa_off
    kw = example_kwargs(ex, a, device, force_mask=probe_kernel)
    kernel_ctx = att.attn_implementation(model, att.OPA_ATTN) if (probe_kernel and a.kernel != att.OPA_ATTN) \
        else contextlib.nullcontext()
    oaa_ctx = att.oaa_inputs(model, tables, ex.origin_tok, router=a.router, struct=a.struct, value=a.value) \
        if use_oaa else contextlib.nullcontext()
    with kernel_ctx, oaa_ctx:
        out = model.model(input_ids=ex.ids[None].to(device), use_cache=False, **kw)
        if ex.ans_len == 0:                       # probe windows: attention only, no loss
            return None, None
        h = out.last_hidden_state[0, ex.ans_start - 1:ex.ans_start - 1 + ex.ans_len]
        logits = model.lm_head(h).float()
        loss = F.cross_entropy(logits, ex.targets.to(device))
        if backward:
            (loss * scale).backward()
    return loss.detach(), logits.detach()
