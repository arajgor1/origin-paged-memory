"""Thrashing tasks (protocol 3.8): joint reasoning over r origins at once.

"Which of these r suppliers quotes the lowest price for part QX-2291: A, B, C, D?" Each of the r quotes sits in a
different origin among `n_origins` (32) filler blocks; prices are distinct (within +-15% of a base price, so the
comparison needs the exact figures) and the minimum is unique. Gold: needed_origins (sorted) and the quotes.
"""
from __future__ import annotations

import re

import numpy as np

from opa.data import GENERATOR_VERSION, check_split, default_block_tokens, item_rng, make_item_id
from opa.data.filler import as_tokenizer, assemble_origins
from opa.data.lbmhqa import check_filler_split
from opa.data.names import SUPPLIER_SUFFIXES, NameGen, part_number

DATASET = "thrash"

QUOTE_TEMPLATES = [
    "{S} quotes {p} per unit for part {P}, with delivery in {w} weeks.",
    "For part {P}, {S} offered a unit price of {p}.",
    "The quotation from {S} for part {P} came in at {p} each.",
    "{S} can supply part {P} at {p} per piece, lead time {w} weeks.",
    "According to the latest bid, {S} charges {p} for each unit of part {P}.",
]


def cell_name(r: int) -> str:
    return f"r{r}"


def parse_cell(cell: str) -> dict:
    m = re.fullmatch(r"r(\d+)", cell)
    if not m:
        raise ValueError(f"bad thrash cell {cell!r} (expected e.g. r4)")
    return {"r": int(m.group(1))}


def fmt_price(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d} USD"


def make_item(seed: int, r: int, filler, tokenizer, split: str = "dev", n_origins: int = 32,
              block_tokens: int | None = None) -> dict:
    block_tokens = default_block_tokens() if block_tokens is None else block_tokens
    check_split(split, seed)
    check_filler_split(filler, split)
    if not 2 <= r <= n_origins:
        raise ValueError("2 <= r <= n_origins")
    tok = as_tokenizer(tokenizer)
    cell = cell_name(r)
    rng = item_rng(DATASET, cell, seed)
    blocks = filler.take_blocks(rng, n_origins, tok, block_tokens, books=list(rng.permutation(filler.n_books())))
    ng = NameGen(rng, vocab=filler.vocab(), text_lower="\n".join(b["text"] for b in blocks).lower())
    suppliers = [ng.entity(SUPPLIER_SUFFIXES, p_suffix=0.85) for _ in range(r)]
    part = part_number(rng)
    base = int(rng.integers(1500, 40000))                       # cents
    while True:
        prices = [int(round(base * (1 + rng.uniform(-0.15, 0.15)))) for _ in range(r)]
        if len(set(prices)) == r:
            break
    origins_needed = [int(x) for x in rng.choice(n_origins, size=r, replace=False)]
    inserts: dict[int, list[dict]] = {}
    sents = []
    for i, (s, p, o) in enumerate(zip(suppliers, prices, origins_needed)):
        tpl = QUOTE_TEMPLATES[int(rng.integers(len(QUOTE_TEMPLATES)))]
        text = tpl.format(S=s.full, p=fmt_price(p), P=part, w=int(rng.integers(2, 13)))
        text = re.sub(r"\.\.$", ".", text)
        sents.append(text)
        inserts.setdefault(o, []).append({"key": ("quote", i), "sentence": text, "kind": "quote"})
    origins, spans = assemble_origins(blocks, [0] * n_origins, inserts, rng, tok)
    quotes = []
    for i, (s, p) in enumerate(zip(suppliers, prices)):
        oid, sp = spans[("quote", i)]
        quotes.append({"supplier": s.full, "price": fmt_price(p), "origin_id": oid, "char_span": sp, "text": sents[i]})
    best = int(np.argmin(prices))
    listed = [suppliers[int(i)].full for i in rng.permutation(r)]
    question = (f"Which of these {r} suppliers quotes the lowest price for part {part}: "
                + ", ".join(listed) + "?")
    ans = suppliers[best]
    return {
        "item_id": make_item_id(DATASET, split, cell, seed),
        "dataset": DATASET, "split": split, "seed": int(seed), "generator_version": GENERATOR_VERSION, "cell": cell,
        "params": {"r": r, "block_tokens": block_tokens, "n_origins": n_origins, "filler": filler.name,
                   "tokenizer": tok.name},
        "origins": origins,
        "question": question,
        "question_thread": 0,
        "answer": ans.full,
        "answer_aliases": [ans.full] + ([ans.stem] if ans.suffix else []),
        "gold": {"needed_origins": sorted(origins_needed), "quotes": quotes, "part": part, "suppliers": listed,
                 "thread_of": [0] * n_origins},
    }
