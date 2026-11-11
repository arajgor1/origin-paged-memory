"""Track B pilot data (dev split only): windows of co-loaded origins and the protocol 5.3 tasks used by the mini-LoRA checks.

What this module builds (PLAN_V3 Section 8 check 4; protocol 5.3 tasks 1, 3, 5 and the own-origin-mass probe)
  Archive      64 abstract origin ids (1..64, root 0) in 8 threads of 8, chained inside a thread (parent = previous origin of
               the thread, the first one hangs off the root), one hyperbolic direction per thread (opa.geometry.TreeLayout,
               the same construction as units.u4_layout). Ids carry fixed addresses o(c) and fixed one-token labels.
  Windows      C co-loaded origins drawn from the 64 ids in RANDOM order. Each id gets freshly drawn PG-19 text of its thread's
               book on every example (a thread = one book, consecutive blocks), so the text of an id is never the same twice:
               the only thing that identifies "origin 17" across examples is its address, never its text (no memorisation).
  Examples     "attrib"   task 3: every window origin holds one invented sentence; the question quotes one of them and asks
                          which page (label word) it came from. Answer = one label token; options = the labels of the window.
               "passkey"  task 1: one origin holds "The access code of <X> is <5 digits>."; the question asks for the code.
               "lm"       task 5: the loss is on the 48 tokens that continue the last window origin's own book.
               "probe"    a window without a tail, used by the attention probe (own-origin mass etc.).
  Splits       books of the filler are split 80/20 into a train pool and an eval pool; the random stream of an example is keyed on
               (example seed, kind, role) with role "train" or "eval": training example i has seed i (any integer, the role
               key keeps it apart from evaluation), evaluation item j has seed 900 + j (inside the dev range 900-999), so
               evaluation windows never equal training windows. No test-split data is ever touched (make_filler refuses
               anything but the dev split).

Position layout of an example
  segments = one Segment(origin_id, length) per window origin + a tail Segment(NULL_ORIGIN, len, start=max origin length).
  M5 (V0, V3): every origin restarts at 0; the tail starts at max(origin lengths) = max(window positions) + 1, so a tail
  token never shares a position with a window key (origins are 480-560 tokens long including a header).
  contiguous (B1): positions 0..T-1 in window order (standard attention, archival-style positions).
  `headers=True` prefixes every origin with "[<label>]\\n" in the text (what a standard system has to do to carry
  provenance); Sampler defaults to False, but TrainConfig.headers is True (PLAN_V3 ruling: B1 and V3 get identical inputs).

Tokenizers: make_tokenizers("hash") gives a whitespace/hash pair for the tiny test model (vocab 128); any other string is a
Hugging Face id (used with the Qwen3 tokenizer on Modal). Importing this module never touches the network or the config.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

from opa.trackb import attention as att

B_REF = 512
N_THREADS, PER_THREAD = 8, 8
N_ORIGINS = N_THREADS * PER_THREAD
CONT_TOKENS = 48                       # LM continuation length
DEV_EVAL_SEED0 = 900                   # pilot evaluation items: seeds 900..999 (dev); training examples use their own index i
# Reserved dev seed ranges (role "eval" windows; all disjoint from each other and from the pilot's 900-999):
MONITOR_SEED0 = 2000                   # learning-curve / stopping-rule monitoring set: seeds 2000..2063 (n_monitor = 64)
G2_SEED0 = 3000                        # fresh G2 evaluation set (PLAN_V3 6a Step 3): seeds 3000.. , never used before G2

LABEL_WORDS = (
    "amber anchor apple arrow aspen badger banner basil beacon birch bison bronze cactus canyon cedar cinder clover cobalt "
    "comet coral cricket crimson dagger dawn delta desert dolphin eagle ember falcon fern flint forest fox garnet glacier "
    "granite harbor hawk hazel heron indigo iron island ivory jade jasper juniper kestrel lantern lark lemon lilac lotus "
    "maple marble meadow mirror nectar oak olive onyx orchid otter panther pearl pepper pine plum quartz raven reef "
    "ridge river saffron sage sparrow spruce summit tiger topaz tulip valley velvet willow wolf zephyr "
    # more common words: with the Qwen3 tokenizer only about half of the rarer words above are a single token
    "table chair window garden bridge castle forest mountain ocean island desert planet star moon cloud storm thunder rain "
    "snow wind fire earth stone gold silver copper glass paper book story song dance horse sheep goat duck goose crow owl bear "
    "deer rabbit mouse snake frog fish whale shark crab bee ant butterfly lake path gate tower wall roof door coin crown sword "
    "shield drum flute bell clock lamp candle ladder wheel boat ship wagon cart basket bottle bucket barrel blanket pillow "
    "ribbon button needle thread").split()


# ============================================================================ tokenizers


class HashEncoder:
    """Test encoder: words and punctuation -> ids in [reserved, vocab) by CRC32. Every label word is one token."""
    name = "hash"

    def __init__(self, vocab: int = 128, reserved: int = 4):
        self.vocab, self.reserved = vocab, reserved
        self.eos_id = 1

    def encode(self, text: str) -> list[int]:
        return [self.reserved + zlib.crc32(w.encode()) % (self.vocab - self.reserved)
                for w in re.findall(r"\w+|[^\w\s]", text)]


class HFEncoder:
    def __init__(self, hf_tokenizer, name: str = "hf"):
        self.tok, self.name = hf_tokenizer, name
        self.eos_id = hf_tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])


def make_tokenizers(spec: str = "hash", vocab: int = 128):
    """(filler tokenizer for opa.data.filler, encoder for model ids)."""
    if spec.startswith("hash"):
        from opa.data.filler import WhitespaceTokenizer
        return WhitespaceTokenizer(), HashEncoder(vocab)
    from transformers import AutoTokenizer
    from opa.data.filler import HFTokenizer
    hf = AutoTokenizer.from_pretrained(spec)
    return HFTokenizer(spec, hf_tokenizer=hf), HFEncoder(hf, spec)


def make_filler(spec: str = "synthetic", split: str = "dev", hf_dir: str | None = None, n_books: int = 24,
                words_per_book: int = 20000):
    """Dev filler only: PG-19 validation (spec "pg19", cached under hf_dir; works without config/project.yaml) or the
    synthetic books used by tests."""
    if split != "dev":
        raise ValueError("opa.trackb.data uses the dev split only")
    from opa.data import filler as fl
    if spec == "synthetic":
        return fl.SyntheticFiller(n_books=n_books, words_per_book=words_per_book)
    if spec == "pg19":
        return fl.PG19Filler("validation", local_dir=hf_dir)
    raise ValueError(spec)


def label_table(enc, n: int = N_ORIGINS) -> list[tuple[str, int]]:
    """The first n label words that are exactly one token with a leading space: [(word, token_id)], index = origin id - 1."""
    out = []
    for w in LABEL_WORDS:
        ids = enc.encode(" " + w)
        if len(ids) == 1 and ids[0] not in {t for _, t in out}:
            out.append((w, ids[0]))
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)} single-token label words for this tokenizer, need {n}")


# ============================================================================ archive (ids, tree, addresses)


def origin_tables(layout, ids, with_dist: bool = False, kind: str = "A2") -> att.OriginTables:
    """OriginTables for `ids`. with_dist=False gives all-zero distances (V3 has no penalty, so the 64x64 hyperbolic
    distances are not computed); True gives the arm's distances (A1/A2/A3) from opa.geometry."""
    if with_dist:
        return att.tables_from_layout(layout, ids, kind=kind)
    from opa.geometry import AddressBookTorch
    tb = AddressBookTorch(layout.book, dtype=torch.float64)
    vec = torch.stack([tb.coords_in_frame(c, 0, work_dtype=torch.float64) if c != 0 else torch.zeros(layout.book.m,
                                                                                                      dtype=torch.float64)
                       for c in ids]).to(torch.float32)
    return att.OriginTables([int(i) for i in ids], vec, torch.zeros(len(ids), len(ids)),
                            meta={"distance": "none", "arm": "V3"})


class Archive:
    """Ids 1..n_threads*per_thread, tree layout, addresses and labels."""

    def __init__(self, enc, m: int = 32, n_threads: int = N_THREADS, per_thread: int = PER_THREAD, seed: int = 0,
                 with_dist: bool = False, kind: str = "A2"):
        from opa.geometry import TreeLayout, spread_directions
        self.n_threads, self.per_thread, self.m = n_threads, per_thread, m
        self.ids = list(range(1, n_threads * per_thread + 1))
        self.layout = TreeLayout(m=m, delta=1.0)
        dirs = spread_directions(n_threads, m, seed=seed)
        for c in self.ids:
            self.layout.add_node(c, self.parent(c), dirs[self.thread(c)])
        self.tables = origin_tables(self.layout, self.ids, with_dist=with_dist, kind=kind)
        self.labels = label_table(enc, len(self.ids))

    def thread(self, c: int) -> int:
        return (c - 1) // self.per_thread

    def parent(self, c: int) -> int:
        return c - 1 if (c - 1) % self.per_thread else 0

    def label(self, c: int) -> str:
        return self.labels[c - 1][0]

    def label_token(self, c: int) -> int:
        return self.labels[c - 1][1]


# ============================================================================ examples


@dataclass
class Example:
    kind: str
    ids: torch.Tensor                       # [T] long
    origin_tok: torch.Tensor                # [T] long, origin id of every token (-1 = tail)
    segments: list
    ans_start: int = 0                      # index of the first target token (predicted from position ans_start - 1)
    ans_len: int = 0
    window: list = field(default_factory=list)      # origin ids in window order
    options: list | None = None             # attrib: option label token ids
    gold_option: int | None = None          # attrib: index into options
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.ids.numel())

    @property
    def targets(self) -> torch.Tensor:
        return self.ids[self.ans_start:self.ans_start + self.ans_len]


class Sampler:
    """Draws windows with fresh text for every example."""

    def __init__(self, archive: Archive, filler, ftok, enc, B: int = B_REF, headers: bool = False):
        self.arch, self.filler, self.ftok, self.enc, self.B, self.headers = archive, filler, ftok, enc, B, headers
        n = filler.n_books()
        n_eval = max(2, n // 5)
        self.pools = {"train": list(range(0, n - n_eval)), "eval": list(range(n - n_eval, n))}

    def window(self, rng: np.random.Generator, C: int, role: str, cont: bool = False):
        """Returns (order, tokens {id: list[int]}, continuation tokens or None). `order` is the random window order."""
        arch = self.arch
        order = [int(c) for c in rng.choice(arch.ids, size=C, replace=False)]
        by_thread: dict[int, list[int]] = {}
        for c in sorted(order):
            by_thread.setdefault(arch.thread(c), []).append(c)
        last = order[-1]
        if cont:                                 # the last window origin takes its thread's last block (it has a continuation)
            lst = by_thread[arch.thread(last)]
            lst.remove(last)
            lst.append(last)
        pool = self.pools[role]
        threads = sorted(by_thread)
        pick = rng.choice(pool, size=len(threads), replace=len(pool) < len(threads))
        tokens, continuation = {}, None
        for t, book in zip(threads, pick):
            ids_t = by_thread[t]
            need = len(ids_t) + (1 if cont and t == arch.thread(last) else 0)
            blocks = self.filler.take_blocks(rng, need, self.ftok, self.B, books=[int(book)])
            for c, blk in zip(ids_t, blocks):
                tokens[c] = self.enc.encode(blk["text"])
            if need > len(ids_t):
                continuation = self.enc.encode(blocks[-1]["text"])[:CONT_TOKENS]
        return order, tokens, continuation

    def _origin_tokens(self, c: int, toks: list[int]) -> list[int]:
        if not self.headers:
            return toks
        return self.enc.encode(f"[{self.arch.label(c)}]\n") + toks

    def _assemble(self, kind, order, tokens, tail, ans_off, ans_len, **kw) -> Example:
        segs = [att.Segment(c, len(tokens[c])) for c in order]
        if tail:
            segs.append(att.Segment(att.NULL_ORIGIN, len(tail), start=max(len(tokens[c]) for c in order)))
        ids = torch.tensor([t for c in order for t in tokens[c]] + list(tail), dtype=torch.long)
        n_win = sum(len(tokens[c]) for c in order)
        return Example(kind, ids, att.token_origins(segs), segs, ans_start=n_win + ans_off, ans_len=ans_len,
                       window=list(order), **kw)

    def make(self, kind: str, seed: int, C: int, role: str = "train") -> Example:
        rng = np.random.default_rng([int(seed), {"attrib": 1, "passkey": 2, "lm": 3, "probe": 4}[kind],
                                     0 if role == "train" else 1])
        enc = self.enc
        order, tokens, cont = self.window(rng, C, role, cont=(kind == "lm"))
        if kind == "probe":
            tokens = {c: self._origin_tokens(c, tokens[c]) for c in order}
            return self._assemble(kind, order, tokens, [], 0, 0)
        if kind == "lm":
            tokens = {c: self._origin_tokens(c, tokens[c]) for c in order}
            return self._assemble(kind, order, tokens, cont, 0, len(cont))
        from opa.data.names import NameGen
        ng = NameGen(rng)
        if kind == "passkey":
            tgt = order[int(rng.integers(len(order)))]
            ent, code = ng.entity(p_suffix=0.0).full, "".join(str(int(x)) for x in rng.integers(0, 10, 5))
            sent = enc.encode(f" The access code of {ent} is {code}.")
            toks = {}
            for c in order:
                t = tokens[c]
                if c == tgt:
                    pos = int(rng.integers(8, max(9, len(t) - 8)))
                    t = t[:pos] + sent + t[pos:]
                toks[c] = self._origin_tokens(c, t)
            q = enc.encode(f"\n\nWhat is the access code of {ent}?\nAnswer:")
            ans = enc.encode(" " + code)
            return self._assemble(kind, order, toks, q + ans, len(q), len(ans), meta={"target_origin": tgt, "code": code})
        if kind == "attrib":
            sents, toks = {}, {}
            for c in order:
                ent = ng.entity(p_suffix=0.0).full
                v = int(rng.integers(100, 1000))
                sents[c] = f"{ent} records a pressure limit of {v} kPa."
                t = tokens[c]
                pos = int(rng.integers(8, max(9, len(t) - 8)))
                toks[c] = self._origin_tokens(c, t[:pos] + enc.encode(" " + sents[c]) + t[pos:])
            tgt = order[int(rng.integers(len(order)))]
            opts = [int(c) for c in rng.permutation(order)]
            words = ", ".join(self.arch.label(c) for c in opts)
            q = enc.encode(f'\n\nWhich page contains the sentence "{sents[tgt]}"? Options: {words}.\nAnswer:')
            ans = [self.arch.label_token(tgt)]
            return self._assemble(kind, order, toks, q + ans, len(q), 1,
                                  options=[self.arch.label_token(c) for c in opts], gold_option=opts.index(tgt),
                                  meta={"target_origin": tgt, "options_origin_ids": opts})
        raise ValueError(kind)


# ============================================================================ inputs for a model


def positions(ex: Example, mapping: str) -> torch.Tensor:
    """Position ids [T]: "M5" fresh frames (tail from max origin length) or "contig" 0..T-1."""
    if mapping == "contig":
        return torch.arange(ex.T, dtype=torch.long)
    if mapping == "M5":
        return att.position_ids(ex.segments, "M5")
    raise ValueError(mapping)
