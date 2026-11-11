"""Contextual-dependence remedies R0-R7 (protocol 5.2) and the H6 measurements, training-free, on the base model.

Setup. A window is a set of co-loaded origins (the gold fact origins of an MT-LB-MHQA dev item plus other origins of the same item,
archival order). The prompt is Qwen3's chat format, identical for every condition:

    <|im_start|>system ... <|im_end|> <|im_start|>user  Documents:  [doc 0] [doc 1] ...  Question: ...  Answer with the final value
    only, including its unit, and nothing else.<|im_end|> <|im_start|>assistant

The chat prefix up to "Documents:" is a SHARED prefix before document 0 in ALL conditions (R0 included), so conditions differ only in
how the documents are encoded. Every condition builds the document KV differently (the shared prefix is in the cache once, at positions
0..p-1), then the question suffix is prefilled against the composed cache and the answer is GENERATED greedily (max 32 new tokens,
stop at EOS / <|im_end|> / newline). EM is opa.scoring.exact_match against the item's answer and aliases (the Track A scorer). As
secondary metrics a teacher-forced pass over the gold answer gives KL(P_R0 || P_R) and the gold log-probability.

  R0         shared prefix + all documents encoded together in order, contiguous positions (reference; provides the reference KV).
  R1         each document encoded alone after the shared prefix (positions p..p+L-1; the shared prefix KV is kept once), then
             co-loaded (the default M5 reuse; documents alias each other's positions).
  R2oracle   R1 + a fraction f in {5, 15, 30}% of the document tokens, chosen by value deviation from R0 (mean over layers 1.. of
             ||V_R1 - V_R0|| / ||V_R0||), replaced by their R0 key and value. The replaced key is re-rotated (in fp32) from its R0
             position to the token's cached position with the model's own RoPE (Lemma 1, units U1). ORACLE: it reads the full-context
             pass R0 and chooses tokens by their true deviation, so it is an upper bound on CacheBlend's recovery at the same f.
  R3         APE-style: each document encoded after the shared chat prefix plus one more short shared sentence; kept once.
  R4         EPIC-style: R1, then the FIRST token of every document (the attention sink of an isolated encoding) is recomputed in
             context: the cache is viewed in contiguous positions (earlier documents' keys re-rotated to their offsets), the token
             runs at its contiguous position and attends to the preceding documents (with their already recomputed first tokens),
             and its new key is rotated back to its cached position. Deployable (no R0 information).
  R5         red dot: each document encoded with its parent's last r tokens in front (r in {4, 8, 16}); the prefix tokens stay in
             the cache and the document follows them. The parent is the previous origin of the same thread (the graph parent); the
             first origin of a thread has no prefix.
  R6         graph co-encoding: each document encoded after the full text of its k in {1, 2} nearest graph parents; only the
             document's own KV is kept (positions start after the parents).
  R7oracle   R5 (r = 8) + R2oracle (5%) from R0 values. ORACLE for the same reason as R2oracle.
The question suffix starts at max(position in the cache) + 1 in every condition.

What R0 must reach for the pilot to mean anything: R0 (full recompute) well above 0 - at least 20% EM on the real model, ideally 50%+
for a 2-hop question on 6 pages. If R0 < 20% the job still finishes but reports r0_too_low: true and no recovery fraction should
be read (the denominator is noise).

Measures per condition: EM, the generated text, KL(P_R0 || P_R) on the gold answer positions, gold log-probability, time to first token
(cache build + question prefill; for the memoised bases R2oracle/R4/R7oracle include the base build), and for R1/R5_8/R6_2 the cosine of
the cached values and of the cached keys (re-rotated to their R0 positions) with R0 per layer.
  best_remedy_recovery_fraction             = max over R3..R6 (deployable) of acc(R) / acc(R0)
  best_remedy_recovery_fraction_incl_oracle = max over R2oracle..R7oracle as well
The gap closed, (acc(R) - acc(R1)) / (acc(R0) - acc(R1)), is reported when R0 beats R1. R5_vs_R1_accuracy = acc(R5, r = 8) - acc(R1) in
EM points with a paired bootstrap 95% CI and the exact McNemar p (opa.stats.contrast). Every accuracy has a bootstrap 95% CI.
Dev items by default (opa.data checks the split); split="test" reads pre-generated test items from items_uri (see main).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import DynamicCache

from opa import scoring
from opa.trackb import attention as att

CONDITIONS = ["R0", "R1", "R2oracle_5", "R2oracle_15", "R2oracle_30", "R3", "R4", "R5_4", "R5_8", "R5_16", "R6_1", "R6_2",
              "R7oracle"]
DEPLOYABLE = ("R3", "R4", "R5", "R6")
ORACLE = ("R2oracle", "R7oracle")
SHARED_PREFIX = "The following is a document."
CHAT_PREFIX = ("<|im_start|>system\nYou are a careful assistant. Answer using only the documents provided.<|im_end|>\n"
               "<|im_start|>user\nDocuments:\n\n")
DOC_SEP = "\n\n"
QUESTION_SUFFIX = ("Question: {q}\nAnswer with the final value only, including its unit, and nothing else.<|im_end|>\n"
                   "<|im_start|>assistant\n")
R0_MIN_ACC = 0.2
MAX_NEW = 32


def _now(dev) -> float:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def decode(enc, ids) -> str:
    return enc.tok.decode(list(ids)) if hasattr(enc, "tok") else " ".join(f"t{int(i)}" for i in ids)


def stop_ids(enc) -> set:
    if hasattr(enc, "tok"):
        t = enc.tok
        return {i for i in (t.convert_tokens_to_ids("<|im_end|>"), t.eos_token_id) if i is not None}
    return {enc.eos_id}


# ============================================================================ items


@dataclass
class RItem:
    item_id: str
    docs: list                       # [{"origin_id", "tokens", "parents": [tokens of nearest parent, next nearest]}]
    q_ids: list                      # question suffix (question + instruction + assistant turn start)
    a_ids: list                      # gold answer tokens (teacher-forced secondary metrics)
    prefix_ids: list = field(default_factory=list)    # shared chat prefix before document 0
    answers: list = field(default_factory=list)       # accepted answer strings (answer + aliases)
    gold: str = ""
    meta: dict = field(default_factory=dict)


def items_from_mtlbmhqa(items: list, enc, C: int = 6, seed: int = 0, k_parents: int = 2) -> list:
    """Windows from MT-LB-MHQA items (the generator's own dev items): the gold fact origins plus other origins, C in total,
    in archival order; the window must contain every gold fact origin (asserted). Parents of an origin are the previous
    origins of its thread (nearest first)."""
    out = []
    sep = enc.encode(DOC_SEP)
    for it in items:
        rng = np.random.default_rng([seed, int(it["seed"])])
        facts = sorted(set(int(o) for o in it["gold"]["fact_origins"]))
        thread_of = it["gold"]["thread_of"]
        others = [o["origin_id"] for o in it["origins"] if o["origin_id"] not in facts]
        extra = [int(x) for x in rng.choice(others, size=max(0, min(C - len(facts), len(others))), replace=False)]
        win = sorted(facts + extra)
        assert set(facts) <= set(win), "window must contain every gold fact origin"
        docs = []
        for o in win:
            prev = [p for p in range(o - 1, -1, -1) if thread_of[p] == thread_of[o]][:k_parents]
            docs.append({"origin_id": o, "tokens": enc.encode(it["origins"][o]["text"]) + sep,
                         "parents": [enc.encode(it["origins"][p]["text"]) + sep for p in prev]})
        q = enc.encode(QUESTION_SUFFIX.format(q=it["question"]))
        out.append(RItem(it["item_id"], docs, q, enc.encode(it["answer"]), enc.encode(CHAT_PREFIX),
                         scoring.gold_answers(it), it["answer"],
                         {"fact_origins": facts, "window": win, "facts_in_window": True, "h_late": it["gold"].get("h_late"),
                          "seed": int(it["seed"])}))
    return out


# ============================================================================ engine


def _rot(model):
    base = model.model if hasattr(model, "model") else model
    import sys
    fn = sys.modules[type(att.attention_modules(model)[0]).__module__].apply_rotary_pos_emb
    return base.rotary_emb, fn


class Engine:
    def __init__(self, model):
        self.model = model
        self.dev = next(model.parameters()).device
        self.rotary, self.apply_rope = _rot(model)
        self.L = len(att.attention_modules(model))

    def _mask(self, Tq, Tk, vis=None):
        m = torch.ones(Tq, Tk, dtype=torch.bool).tril(Tk - Tq)
        if vis is not None:                                # vis: [Tk - Tq] bool over the cached entries
            m[:, :Tk - Tq] = vis[None, :]
        m = m[None, None].to(self.dev)
        return {"full_attention": m, "sliding_attention": m}

    def _cache(self, keys, values):
        cache = DynamicCache()
        for i in range(self.L):
            cache.update(keys[i], values[i], i)
        return cache

    @torch.no_grad()
    def prefill(self, ids: list, pos: list):
        """Keys/values of one sequence [L x [1, G, T, d]] and the hidden state after the last layer."""
        t = torch.tensor([ids], device=self.dev)
        p = torch.tensor([pos], device=self.dev)
        out = self.model.model(input_ids=t, position_ids=p, attention_mask=self._mask(len(ids), len(ids)), use_cache=True)
        c = out.past_key_values
        return [c.layers[i].keys for i in range(self.L)], [c.layers[i].values for i in range(self.L)], out.last_hidden_state

    @torch.no_grad()
    def answer(self, keys, values, vis, q_ids, a_ids, q_start: int):
        """Teacher-forced logits [len(a), V] at the gold-answer positions given a composed cache."""
        cache = self._cache(keys, values)
        ids = q_ids + a_ids
        Tc = keys[0].shape[2]
        t = torch.tensor([ids], device=self.dev)
        p = torch.arange(q_start, q_start + len(ids), device=self.dev)[None]
        out = self.model.model(input_ids=t, position_ids=p, past_key_values=cache, use_cache=True,
                               attention_mask=self._mask(len(ids), Tc + len(ids), vis.to(self.dev) if vis is not None else None))
        h = out.last_hidden_state[0, len(q_ids) - 1:len(q_ids) - 1 + len(a_ids)]
        return self.model.lm_head(h).float()

    @torch.no_grad()
    def generate(self, keys, values, q_ids, q_start: int, enc, max_new: int = MAX_NEW):
        """Greedy generation from a composed cache. Returns (generated ids, seconds for the question prefill = time to first token
        excluding the cache build). Stops at EOS / <|im_end|> or once a newline follows some text."""
        cache = self._cache(keys, values)
        Tc = keys[0].shape[2]
        stops = stop_ids(enc)
        t0 = _now(self.dev)
        out = self.model.model(input_ids=torch.tensor([q_ids], device=self.dev),
                               position_ids=torch.arange(q_start, q_start + len(q_ids), device=self.dev)[None],
                               past_key_values=cache, use_cache=True, attention_mask=self._mask(len(q_ids), Tc + len(q_ids)))
        logits = self.model.lm_head(out.last_hidden_state[:, -1])
        ttft = _now(self.dev) - t0
        gen, pos, Tk = [], q_start + len(q_ids), Tc + len(q_ids)
        for _ in range(max_new):
            nxt = int(logits[0].argmax())
            if nxt in stops:
                break
            gen.append(nxt)
            if "\n" in decode(enc, gen).lstrip():
                break
            out = self.model.model(input_ids=torch.tensor([[nxt]], device=self.dev), position_ids=torch.tensor([[pos]], device=self.dev),
                                   past_key_values=cache, use_cache=True, attention_mask=self._mask(1, Tk + 1))
            logits = self.model.lm_head(out.last_hidden_state[:, -1])
            pos += 1
            Tk += 1
        return gen, ttft

    @torch.no_grad()
    def append_token(self, keys, values, token: int, pos: int):
        """Keys/values [L x [1, G, 1, d]] of one new token at position `pos` attending to the whole given cache and itself."""
        cache = self._cache(keys, values)
        Tc = keys[0].shape[2]
        out = self.model.model(input_ids=torch.tensor([[token]], device=self.dev), position_ids=torch.tensor([[pos]], device=self.dev),
                               past_key_values=cache, use_cache=True, attention_mask=self._mask(1, Tc + 1))
        c = out.past_key_values
        return [c.layers[i].keys[:, :, -1:] for i in range(self.L)], [c.layers[i].values[:, :, -1:] for i in range(self.L)]

    def rerotate(self, k, delta):
        """Rotate cached keys k [1, G, T, d] by delta [T] positions (RoPE composition, Lemma 1). Pass fp32 keys for fp32 math."""
        d = torch.as_tensor(delta, device=self.dev)[None]
        cos, sin = self.rotary(k, d)
        return self.apply_rope(k, k, cos, sin)[1]


# ============================================================================ conditions


@dataclass
class Cache:
    keys: list
    values: list
    doc: list                         # document index of every entry (-1: shared prefix)
    tok: list                         # token index inside the document (-1: prefix entry)
    pos: list                         # position stored in the key
    q_start: int
    vis: object = None
    abs0: list = None                 # index of the token among the document tokens of R0 (None for prefix entries)
    build_s: float = 0.0              # wall-clock seconds to build this cache (including its memoised base)
    p0: int = 0                       # length of the shared chat prefix: the R0 position of document token i is p0 + abs0


def _cat(parts):
    return [torch.cat([p[i] for p in parts], dim=2) for i in range(len(parts[0]))]


def clone_cache(c: Cache) -> Cache:
    return Cache([x.clone() for x in c.keys], [x.clone() for x in c.values], list(c.doc), list(c.tok), list(c.pos), c.q_start,
                 None if c.vis is None else c.vis.clone(), None if c.abs0 is None else list(c.abs0), c.build_s, c.p0)


def parse_cond(cond: str):
    kind = cond.split("_")[0]
    arg = int(cond.split("_")[1]) if "_" in cond else None
    return kind, arg


def build_cache(eng: Engine, item: RItem, cond: str, enc, r0_kv=None, memo: dict = None) -> Cache:
    """Cache for condition `cond` (not R0). `memo` (dict, created if None) shares the expensive base builds: R2oracle and R4 start
    from the R1 cache, R7oracle from the R5_8 cache (always handed out as clones). r0_kv = document-only R0 keys/values."""
    memo = {} if memo is None else memo
    kind, arg = parse_cond(cond)
    base_key = {"R2oracle": "R1", "R4": "R1", "R7oracle": "R5_8"}.get(kind)
    if base_key is not None:
        if base_key not in memo:
            memo[base_key] = build_cache(eng, item, base_key, enc, r0_kv, memo)
        c = clone_cache(memo[base_key])
        t = _now(eng.dev)
        if kind == "R4":
            recompute_first_tokens(eng, c, item)
        else:
            replace_top_deviation(eng, c, r0_kv, (arg if kind == "R2oracle" else 5) / 100.0)
        c.build_s += _now(eng.dev) - t
        return c
    t0 = _now(eng.dev)
    docs = item.docs
    off, o = [], 0
    for d in docs:
        off.append(o)
        o += len(d["tokens"])
    shared_ids = item.prefix_ids + (enc.encode(SHARED_PREFIX) if kind == "R3" else [])
    ns = len(shared_ids)
    ks, vs, doc_i, tok_i, pos_i, abs0 = [], [], [], [], [], []
    shared = None
    for di, d in enumerate(docs):
        toks = d["tokens"]
        if kind in ("R1", "R3"):
            pre = []
        elif kind == "R5":
            pre = d["parents"][0][-arg:] if d["parents"] else []
        elif kind == "R6":
            pre = [t for par in reversed(d["parents"][:arg]) for t in par]
        else:
            raise ValueError(cond)
        seq = shared_ids + pre + toks
        k, v, _ = eng.prefill(seq, list(range(len(seq))))
        if shared is None and ns:
            shared = ([x[:, :, :ns] for x in k], [x[:, :, :ns] for x in v])
        lo = ns + (len(pre) if kind == "R6" else 0)
        k = [x[:, :, lo:] for x in k]
        v = [x[:, :, lo:] for x in v]
        n_pre_kept = len(pre) if kind == "R5" else 0
        ks.append(k)
        vs.append(v)
        doc_i += [di] * (n_pre_kept + len(toks))
        tok_i += [-1] * n_pre_kept + list(range(len(toks)))
        pos_i += list(range(lo, len(seq)))
        abs0 += [None] * n_pre_kept + [off[di] + j for j in range(len(toks))]
    keys, values = _cat(ks), _cat(vs)
    if shared is not None:
        keys = [torch.cat([shared[0][i], keys[i]], dim=2) for i in range(eng.L)]
        values = [torch.cat([shared[1][i], values[i]], dim=2) for i in range(eng.L)]
        doc_i, tok_i, pos_i, abs0 = [-1] * ns + doc_i, [-1] * ns + tok_i, list(range(ns)) + pos_i, [None] * ns + abs0
    q_start = max(pos_i) + 1                         # the question follows the largest position in the cache
    return Cache(keys, values, doc_i, tok_i, pos_i, q_start, abs0=abs0, build_s=_now(eng.dev) - t0, p0=len(item.prefix_ids))


@torch.no_grad()
def recompute_first_tokens(eng: Engine, c: Cache, item: RItem) -> None:
    """R4: replace the key/value of the first token of every document d >= 1 by a recomputation in context. The token must see
    the earlier documents as they are in a CONTIGUOUS context, so the cache is viewed in contiguous positions (shared prefix at
    0..p-1, document j at p + off_j with off_j = total length of documents 0..j-1; keys re-rotated in fp32, Lemma 1), the token is
    run at P = p + off_d, and its new key is re-rotated back to its cached position (fp32). Values need no rotation. Recomputed
    first tokens of earlier documents move with their documents. Document 0 has no document context and is unchanged."""
    nd = len(item.docs)
    off, o = [], 0
    for d in item.docs:
        off.append(o)
        o += len(d["tokens"])
    contig = torch.tensor([c.pos[i] if dd < 0 else c.p0 + off[dd] + tt for i, (dd, tt) in enumerate(zip(c.doc, c.tok))], device=eng.dev)
    delta = contig - torch.tensor(c.pos, device=eng.dev)
    kc = [eng.rerotate(k.float(), delta).to(k.dtype) for k in c.keys]              # keys in contiguous positions
    starts = [next(i for i, (dd, tt) in enumerate(zip(c.doc, c.tok)) if dd == d and tt == 0) for d in range(nd)]
    for d in range(1, nd):
        s = starts[d]
        P = c.p0 + off[d]
        nk, nv = eng.append_token([x[:, :, :s] for x in kc], [x[:, :, :s] for x in c.values], item.docs[d]["tokens"][0], P)
        for l in range(eng.L):
            kc[l][:, :, s:s + 1] = nk[l]                                            # contiguous view, used by later documents
            back = eng.rerotate(nk[l].float(), [c.pos[s] - P])                      # to the cached position, fp32
            c.keys[l][:, :, s:s + 1] = back.to(c.keys[l].dtype)
            c.values[l][:, :, s:s + 1] = nv[l].to(c.values[l].dtype)


@torch.no_grad()
def replace_top_deviation(eng: Engine, c: Cache, r0_kv, frac: float) -> list:
    """ORACLE: replace the `frac` of document tokens with the largest value deviation from R0 by their R0 key (re-rotated in fp32
    from its R0 position p0 + abs0 to the cached position, then cast) and value, in all layers. Returns the replaced indices."""
    r0k, r0v = r0_kv
    idx = [i for i, a in enumerate(c.abs0) if a is not None]
    a0 = torch.tensor([c.abs0[i] for i in idx], device=eng.dev)
    ci = torch.tensor(idx, device=eng.dev)
    dev = torch.zeros(len(idx), device=eng.dev)
    for l in range(1, eng.L):
        dv = (c.values[l][0][:, ci].float() - r0v[l][0][:, a0].float()).norm(dim=-1)
        dev += (dv / r0v[l][0][:, a0].float().norm(dim=-1).clamp_min(1e-6)).mean(0)
    n = int(math.ceil(frac * len(idx)))
    top = dev.topk(n).indices
    cs, a_sel = ci[top], a0[top]
    delta = torch.tensor([c.pos[int(i)] for i in cs.tolist()], device=eng.dev) - (a_sel + c.p0)
    for l in range(eng.L):
        c.keys[l][:, :, cs] = eng.rerotate(r0k[l][:, :, a_sel].float(), delta).to(c.keys[l].dtype)
        c.values[l][:, :, cs] = r0v[l][:, :, a_sel].to(c.values[l].dtype)
    return cs.tolist()


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def v_cosine_by_layer(c: Cache, r0_kv) -> list:
    """Mean cosine between cached document values and their R0 values, per layer."""
    idx = [i for i, a in enumerate(c.abs0) if a is not None]
    a0 = torch.tensor([c.abs0[i] for i in idx], device=c.values[0].device)
    ci = torch.tensor(idx, device=c.values[0].device)
    return [_cos(c.values[l][0][:, ci].float().transpose(0, 1).reshape(len(idx), -1),
                 r0_kv[1][l][0][:, a0].float().transpose(0, 1).reshape(len(idx), -1)) for l in range(len(c.values))]


def k_cosine_by_layer(eng: Engine, c: Cache, r0_kv) -> list:
    """Mean cosine between cached document keys, re-rotated to their R0 positions (fp32), and the R0 keys, per layer."""
    idx = [i for i, a in enumerate(c.abs0) if a is not None]
    a0 = torch.tensor([c.abs0[i] for i in idx], device=eng.dev)
    ci = torch.tensor(idx, device=eng.dev)
    delta = (a0 + c.p0) - torch.tensor([c.pos[i] for i in idx], device=eng.dev)
    out = []
    for l in range(len(c.keys)):
        k = eng.rerotate(c.keys[l][:, :, ci].float(), delta)
        out.append(_cos(k[0].transpose(0, 1).reshape(len(idx), -1), r0_kv[0][l][0][:, a0].float().transpose(0, 1).reshape(len(idx), -1)))
    return out


def kl_to(ref_logits, logits) -> float:
    lp, lq = torch.log_softmax(ref_logits.float(), -1), torch.log_softmax(logits.float(), -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def gold_logprob(logits, a_ids) -> float:
    lp = torch.log_softmax(logits.float(), -1)
    return float(lp[torch.arange(len(a_ids)), torch.tensor(a_ids, device=lp.device)].sum())


def score_text(item: RItem, text: str) -> int:
    """EM with the Track A scorer (opa.scoring.exact_match: its normalisation and the item's answer aliases) on the first line."""
    line = text.strip().split("\n")[0].strip()
    return int(scoring.exact_match(line, item.answers or [item.gold]))


@torch.no_grad()
def run_item(eng: Engine, item: RItem, enc, conditions=CONDITIONS, cos_for=("R1", "R5_8", "R6_2")) -> dict:
    """{cond: {"em", "text", "kl", "gold_logprob", "ttft_s", ("v_cos", "k_cos")}} for one item."""
    docs_ids = [t for d in item.docs for t in d["tokens"]]
    p = len(item.prefix_ids)
    nd = len(docs_ids)
    ids = item.prefix_ids + docs_ids
    t = _now(eng.dev)
    k0, v0, _ = eng.prefill(ids, list(range(len(ids))))
    t_r0 = _now(eng.dev) - t
    r0_kv = ([x[:, :, p:] for x in k0], [x[:, :, p:] for x in v0])
    memo: dict = {}
    ref = eng.answer(k0, v0, None, item.q_ids, item.a_ids, len(ids))
    gen, tf = eng.generate(k0, v0, item.q_ids, len(ids), enc)
    text = decode(enc, gen)
    res = {"R0": {"em": score_text(item, text), "text": text, "kl": 0.0, "gold_logprob": gold_logprob(ref, item.a_ids),
                  "ttft_s": t_r0 + tf}}
    for cond in conditions:
        if cond == "R0":
            continue
        c = build_cache(eng, item, cond, enc, r0_kv, memo)
        logits = eng.answer(c.keys, c.values, c.vis, item.q_ids, item.a_ids, c.q_start)
        gen, tf = eng.generate(c.keys, c.values, item.q_ids, c.q_start, enc)
        text = decode(enc, gen)
        r = {"em": score_text(item, text), "text": text, "kl": kl_to(ref, logits), "gold_logprob": gold_logprob(logits, item.a_ids),
             "ttft_s": c.build_s + tf}
        if cond in cos_for:
            r["v_cos"] = v_cosine_by_layer(c, r0_kv)
            r["k_cos"] = k_cosine_by_layer(eng, c, r0_kv)
        res[cond] = r
    return res


# ============================================================================ summary


def summarize(per_item: list) -> dict:
    from opa import stats
    conds = list(per_item[0])
    n = len(per_item)
    arr = {c: np.array([r[c]["em"] for r in per_item], float) for c in conds}
    acc = {c: float(arr[c].mean()) for c in conds}
    ci = {c: [float(x) for x in stats.paired_bootstrap(arr[c], np.zeros(n))[1]] for c in conds}
    kl = {c: float(np.mean([r[c]["kl"] for r in per_item])) for c in conds}
    out = {"n_items": n, "R0_correct": int(arr["R0"].sum()), "r0_too_low": bool(acc["R0"] < R0_MIN_ACC), "r0_min_acc": R0_MIN_ACC,
           "acc": acc, "acc_ci95_points": ci, "mean_kl_to_R0": kl,
           "mean_gold_logprob": {c: float(np.mean([r[c]["gold_logprob"] for r in per_item])) for c in conds if "gold_logprob" in per_item[0][c]},
           "ttft_s_mean": {c: float(np.mean([r[c]["ttft_s"] for r in per_item])) for c in conds if "ttft_s" in per_item[0][c]},
           "ttft_s_median": {c: float(np.median([r[c]["ttft_s"] for r in per_item])) for c in conds if "ttft_s" in per_item[0][c]}}
    a0, a1 = acc["R0"], acc.get("R1")
    rec = {c: (acc[c] / a0 if a0 > 0 else None) for c in conds if c not in ("R0", "R1")}
    out["recovery_vs_R0"] = rec
    if a1 is not None:
        out["R1_vs_R0_ratio"] = a1 / a0 if a0 > 0 else None
        out["gap_closed"] = {c: ((acc[c] - a1) / (a0 - a1) if a0 > a1 else None) for c in conds if c not in ("R0", "R1")}
    for key, kinds in (("best_remedy_recovery_fraction", DEPLOYABLE), ("best_remedy_recovery_fraction_incl_oracle", DEPLOYABLE + ORACLE)):
        cand = {c: v for c, v in rec.items() if v is not None and parse_cond(c)[0] in kinds}
        if cand:
            best = max(cand, key=cand.get)
            out[key.replace("_recovery_fraction", "")], out[key] = best, cand[best]
    if "R5_8" in acc and a1 is not None:
        ct = stats.contrast(arr["R5_8"], arr["R1"], key="R5_vs_R1_accuracy")
        out["R5_vs_R1_accuracy"] = ct["value"]
        out["R5_vs_R1"] = {k: ct[k] for k in ("value", "ci_lo", "ci_hi", "n", "n10", "n01", "p_raw")}
    for c in ("R1", "R5_8", "R6_2"):
        for key in ("v_cos", "k_cos"):
            cs = [r[c][key] for r in per_item if c in r and key in r[c]]
            if cs:
                out.setdefault(f"{key}_by_layer", {})[c] = [float(x) for x in np.mean(cs, axis=0)]
    return out


# ============================================================================ job entry point


def build_items(n: int, seed0: int, ftok, enc, filler, n_threads: int = 4, hops: int = 2, length: int = 8192, C: int = 6,
                block_tokens: int = 512, run_blocks: int = 2) -> list:
    """n dev items of MT-LB-MHQA (cell T<n_threads>-h<hops>, no cross-thread hop) windowed to C origins. Generator failures
    (a seed whose schedule cannot be satisfied) are skipped and the next seed is used. block_tokens is passed explicitly so that
    no config file is read."""
    from opa.data import mtlbmhqa
    items, s = [], seed0
    while len(items) < n and s < seed0 + 10 * n + 20:
        try:
            items.append(mtlbmhqa.make_item(s, n_threads, hops, length, False, 0.0, filler, ftok, split="dev",
                                            run_blocks=run_blocks, block_tokens=block_tokens))
        except (RuntimeError, ValueError):
            pass
        s += 1
    return items_from_mtlbmhqa(items, enc, C=C, seed=seed0)


def load_items_uri(uri: str, ctx=None) -> list:
    """Items of a generated cell: items.jsonl.gz at a gs:// uri (gs://{bucket}/... is resolved from the job spec) or a local path."""
    import gzip
    from opa.trackb import train
    uri = train.resolve_bucket(uri, ctx)
    if str(uri).startswith("gs://"):
        from opa import gcs
        data = gcs.get_bytes(uri)
    else:
        data = Path(uri).read_bytes()
    if uri.endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def main(args: dict, ctx) -> dict:
    """Entry point opa.trackb.remedies:main. args: model, tokenizer, filler ("pg19"), hf_dir, n_items, seed0, C, n_threads, hops,
    length, block_tokens, conditions (default all), time_budget_s (no new item is started after the budget minus a margin; what
    is done is summarised and written). Each item's result (generated text and gold per condition) is written to items/<k>.json as
    soon as it is done. The summary flags r0_too_low when R0 < 20% EM.

    split ("dev" by default; "test" needs the prereg-v2 tag: prereg/PREREG_TAG.json or args["prereg_tag"] = {tag, commit}, seed0 >= 100000 and
    items_uri). items_uri (optional for dev, required for test): items.jsonl.gz of a generated MT-LB-MHQA or LB-MHQA cell, with offset / n_items
    selecting the chunk; without it dev items are generated here (the generator refuses test data outside the tagged repository, so test items are
    always read from files). Every item must carry the split's name and seeds from seed0 on."""
    from opa.trackb import data as D
    from opa.trackb import trainfree
    from opa.trackb import units
    t0 = time.time()
    a = dict(model="Qwen/Qwen3-4B-Instruct-2507", tokenizer=None, filler="pg19", hf_dir="/cache/hf/pg19", n_items=48, seed0=0,
             C=6, n_threads=4, hops=2, length=8192, block_tokens=512, conditions=CONDITIONS, time_budget_s=1700.0,
             item_margin_s=90.0, split="dev", items_uri=None, offset=0, prereg_tag=None)
    a.update(args)
    trainfree.check_split(a["split"], a["seed0"], a, int(a["n_items"]))                  # test: tag, seeds >= 100000 - before anything is loaded
    if a["split"] == "test" and not a["items_uri"]:
        raise ValueError("split=test needs items_uri: test items are generated locally and read from GCS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftok, enc = D.make_tokenizers(a["tokenizer"] or a["model"])
    if a["items_uri"]:
        raw = load_items_uri(a["items_uri"], ctx)[int(a["offset"]):int(a["offset"]) + int(a["n_items"])]
        bad = [it["item_id"] for it in raw if it.get("split") != a["split"] or int(it["seed"]) < int(a["seed0"])]
        if bad or not raw:
            raise ValueError(f"items must be non-empty, of split {a['split']!r} and have seeds >= {a['seed0']}: offending {bad[:3]}")
        items = items_from_mtlbmhqa(raw, enc, C=a["C"], seed=a["seed0"])
    else:
        filler = D.make_filler(a["filler"], "dev", hf_dir=a["hf_dir"])
        items = build_items(a["n_items"], a["seed0"], ftok, enc, filler, a["n_threads"], a["hops"], a["length"], a["C"], a["block_tokens"])
    ctx.log({"action": "remedies_items", "n": len(items), "tokens_per_window": [sum(len(d["tokens"]) for d in it.docs) for it in items]})
    model = units.load_model(a["model"], device, dtype=torch.bfloat16)
    model.eval()
    eng = Engine(model)
    per_item, done_ids, stop = [], [], "all_items"
    t_item = 0.0
    for k, it in enumerate(items):
        if time.time() - t0 + max(t_item, a["item_margin_s"]) > a["time_budget_s"]:
            stop = "time_budget"
            break
        ts = time.time()
        r = run_item(eng, it, enc, a["conditions"])
        t_item = max(t_item, time.time() - ts)
        per_item.append(r)
        done_ids.append(it.item_id)
        ctx.put_json(f"items/{k:03d}.json", {"item_id": it.item_id, "global_index": int(a["offset"]) + k, "meta": it.meta, "gold": it.gold,
                                             "answers": it.answers, "result": r})
        ctx.log({"action": "remedy_item", "item_id": it.item_id, "seconds": round(time.time() - ts, 1), "gold": it.gold,
                 "em": {c: v["em"] for c, v in r.items()}, "R0_text": r["R0"]["text"]})
    if not per_item:
        raise RuntimeError("no remedy item finished within the time budget")
    summ = summarize(per_item)
    summ["stop_reason"], summ["n_requested"] = stop, len(items)
    ctx.put_json("remedies.json", {"summary": summ, "per_item": per_item, "item_ids": done_ids, "args": a})
    flat = {"split": a["split"], "offset": int(a["offset"]), "n_items": summ["n_items"], "n_requested": len(items), "stop_reason": stop, "R0_acc": summ["acc"]["R0"],
            "R0_correct": summ["R0_correct"], "r0_too_low": summ["r0_too_low"], "R1_acc": summ["acc"].get("R1"),
            "best_remedy": summ.get("best_remedy"),
            "best_remedy_recovery_fraction": summ.get("best_remedy_recovery_fraction"),
            "best_remedy_incl_oracle": summ.get("best_remedy_incl_oracle"),
            "best_remedy_recovery_fraction_incl_oracle": summ.get("best_remedy_recovery_fraction_incl_oracle"),
            "R5_vs_R1_accuracy": summ.get("R5_vs_R1_accuracy"), "seconds": round(time.time() - t0, 1)}
    flat.update({f"acc_{c}": v for c, v in summ["acc"].items()})
    return {"metrics": flat}
