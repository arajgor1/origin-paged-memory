"""LB-MHQA: single-thread late-binding multi-hop QA (protocol 3.2).

An item is `length_tokens / 512` filler blocks (PG-19) with an h-hop chain of invented entities inserted:
  hop j < h-1:  "<entity j> sources its seals from <entity j+1>."        (varied relation templates)
  hop h-1:      "Seals made by <entity h-1> degrade above 83 °C in ester coolant."   (numeric attribute + unit)
The question names entity 0 and asks for the final attribute.

Placement of the hop facts over origins in distinct 2048-token groups (4 blocks), so placement and h_late are the
same after rechunk to B = 1024 or 2048:
  forward  fact origins increase with the hop          -> h_late = 0
  late     fact origins decrease with the hop          -> every hop after the first is late-bound, h_late = h-1
  mixed    a random order with 0 < h_late < h-1 (for h >= 3; for h = 2 any order)
Distractors: `n_distractor_chains` complete chains of the same templates about other invented entities, with other
values, each sentence in a random origin. Alias variants ("Ostrand Ltd." -> "OSTRAND"): with probability alpha the
subject mention of entity j >= 1 in its own fact uses an alias (same rule in distractor chains).

h_late (paper Definition 6): hop j is late-bound if its fact lies in an origin before the origin where entity j
became relevant. Entity 0 is relevant from origin 0 (the question is known when reading starts); entity j >= 1
becomes relevant at the origin of hop j-1's fact, which reveals it.
"""
from __future__ import annotations

import re

import numpy as np

from opa.data import GENERATOR_VERSION, check_split, default_block_tokens, item_rng, make_item_id
from opa.data.filler import as_tokenizer, assemble_origins
from opa.data.names import Name, NameGen

DATASET = "lbmhqa"
PLACEMENTS = ("forward", "late", "mixed")

COMPONENTS = ["seals", "gaskets", "bearings", "impellers", "couplings", "membranes", "bushings", "diaphragms",
              "valve seats", "O-rings"]

RELATIONS = [
    "{X} sources its {c} from {Y}.",
    "{X} buys all of its {c} from {Y}.",
    "{Y} manufactures the {c} that {X} uses.",
    "{X} gets its {c} from {Y}.",
    "The {c} used by {X} are made by {Y}.",
    "{X} has outsourced production of its {c} to {Y}.",
    "Every shipment of {c} that {X} installs comes from {Y}.",
]

ATTRIBUTES = {
    "temperature": {
        "unit": "°C", "lo": 60, "hi": 141, "step": 1,
        "templates": ["{C} made by {Z} degrade above {v} in ester coolant.",
                      "{Z} reports that its {c} start to fail above {v}.",
                      "Above {v}, the {c} from {Z} begin to degrade."],
        "question": "Above what temperature do the {c} in {E}'s supply chain degrade?",
    },
    "pressure": {
        "unit": "bar", "lo": 6, "hi": 61, "step": 1,
        "templates": ["{C} made by {Z} are rated for at most {v}.",
                      "{Z} certifies its {c} up to {v}.",
                      "The {c} from {Z} hold pressure up to {v}."],
        "question": "Up to what pressure are the {c} in {E}'s supply chain rated?",
    },
    "service_life": {
        "unit": "hours", "lo": 10, "hi": 100, "step": 100,     # 1000..9900 in steps of 100
        "templates": ["{C} made by {Z} last {v} in continuous service.",
                      "{Z} guarantees its {c} for {v} of operation.",
                      "The {c} from {Z} need replacing after {v} of service."],
        "question": "For how many operating hours do the {c} in {E}'s supply chain last?",
    },
}


# ---------------------------------------------------------------- cells
def length_label(n_tokens: int) -> str:
    return f"{n_tokens // 1024}k"


def parse_length(s: str) -> int:
    m = re.fullmatch(r"(\d+)k", s)
    if not m:
        raise ValueError(f"bad length {s!r}")
    return int(m.group(1)) * 1024


def alpha_label(alpha: float) -> str:
    return "a0" if alpha == 0 else "a" + f"{alpha:.2f}".rstrip("0").replace(".", "")


def parse_alpha(s: str) -> float:
    m = re.fullmatch(r"a(\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"bad alpha {s!r}")
    t = m.group(1)
    if "." in t:
        return float(t)
    return 0.0 if t == "0" else float("0." + t[1:]) if t.startswith("0") else float(t)


def cell_name(placement: str, hops: int, length_tokens: int, alpha: float) -> str:
    return f"{placement}-h{hops}-{length_label(length_tokens)}-{alpha_label(alpha)}"


def parse_cell(cell: str) -> dict:
    m = re.fullmatch(r"(forward|late|mixed)-h(\d+)-(\d+k)-(a[\d.]+)", cell)
    if not m:
        raise ValueError(f"bad LB-MHQA cell {cell!r} (expected e.g. late-h3-32k-a0)")
    return {"placement": m.group(1), "hops": int(m.group(2)), "length_tokens": parse_length(m.group(3)),
            "alpha": parse_alpha(m.group(4))}


# ---------------------------------------------------------------- values and sentences
def fmt_value(kind: str, v: int) -> str:
    return f"{v} {ATTRIBUTES[kind]['unit']}"


def answer_aliases(kind: str, v: int) -> list[str]:
    if kind == "temperature":
        return [f"{v} °C", f"{v}°C", f"{v} C", f"{v} degC", f"{v} degrees C", f"{v} degrees Celsius", f"{v}"]
    if kind == "pressure":
        return [f"{v} bar", f"{v}bar", f"{v}"]
    out = [f"{v} hours", f"{v} h", f"{v} operating hours", f"{v}"]
    if v >= 1000:
        out.insert(1, f"{v:,} hours")
    return out


def draw_values(rng: np.random.Generator, kind: str, n: int) -> list[int]:
    a = ATTRIBUTES[kind]
    pool = np.arange(a["lo"], a["hi"]) * (a["step"] if a["step"] > 1 else 1)
    return [int(x) for x in rng.choice(pool, size=n, replace=False)]


def sentence(s: str) -> str:
    s = re.sub(r"\.\.$", ".", s)          # "... Ostrand Ltd.." -> "... Ostrand Ltd."
    return s[0].upper() + s[1:]


def relation_sentence(tpl: str, x: str, y: str, comp: str) -> str:
    return sentence(tpl.format(X=x, Y=y, c=comp))


def attribute_sentence(tpl: str, z: str, comp: str, kind: str, v: int) -> str:
    return sentence(tpl.format(Z=z, c=comp, C=comp[0].upper() + comp[1:], v=fmt_value(kind, v)))


def question_text(kind: str, comp: str, e0: str) -> str:
    return ATTRIBUTES[kind]["question"].format(c=comp, E=e0)


def build_chain(rng: np.random.Generator, names: list[Name], comp: str, kind: str, value: int, alpha: float,
                final_template: int | None = None) -> list[dict]:
    """Sentences of one chain over `names` (len = hops). Subject of hop j >= 1 uses an alias with prob alpha."""
    hops = len(names)
    out = []
    for j in range(hops):
        subj = names[j].full
        if j >= 1 and alpha > 0 and rng.random() < alpha:
            opts = names[j].alias_options()
            surf = opts[int(rng.integers(len(opts)))]
        else:
            surf = subj
        if j < hops - 1:
            tpl = RELATIONS[int(rng.integers(len(RELATIONS)))]
            s = relation_sentence(tpl, surf, names[j + 1].full, comp)
            out.append({"hop": j, "sentence": s, "subject": subj, "subject_surface": surf,
                        "reveals": names[j + 1].full})
        else:
            ti = int(rng.integers(len(ATTRIBUTES[kind]["templates"]))) if final_template is None else final_template
            s = attribute_sentence(ATTRIBUTES[kind]["templates"][ti], surf, comp, kind, value)
            out.append({"hop": j, "sentence": s, "subject": subj, "subject_surface": surf,
                        "reveals": fmt_value(kind, value), "template": ti})
    return out


def late_bound_flags(fact_origins: list[int]) -> list[bool]:
    """Definition 6 with entity 0 relevant from origin 0 and entity j revealed by hop j-1's fact."""
    relevant_at = [0] + [int(o) for o in fact_origins[:-1]]
    return [int(fact_origins[j]) < relevant_at[j] for j in range(len(fact_origins))]


def spread_origins(rng: np.random.Generator, candidates: list[int], k: int, group: int) -> list[int]:
    """k origins from `candidates` lying in k distinct groups of `group` consecutive origins (group = 2048 / 512 = 4),
    so merging to B = 1024 or 2048 never puts two of them in one origin. Returned in ascending order."""
    by_group: dict[int, list[int]] = {}
    for o in candidates:
        by_group.setdefault(int(o) // group, []).append(int(o))
    keys = sorted(by_group)
    if len(keys) < k:
        raise ValueError(f"need {k} distinct groups, have {len(keys)}")
    picked = [keys[int(i)] for i in rng.choice(len(keys), size=k, replace=False)]
    return sorted(by_group[g][int(rng.integers(len(by_group[g])))] for g in picked)


def place_chain(rng: np.random.Generator, n_origins: int, hops: int, placement: str, group: int = 4) -> list[int]:
    slots = spread_origins(rng, list(range(n_origins)), hops, group)
    if placement == "forward":
        return slots
    if placement == "late":
        return slots[::-1]
    if placement == "mixed":
        for _ in range(1000):
            fo = [slots[int(p)] for p in rng.permutation(hops)]
            h = sum(late_bound_flags(fo))
            if hops < 3 or 0 < h < hops - 1:
                return fo
        raise RuntimeError("mixed placement failed")
    raise ValueError(f"unknown placement {placement!r}")


INSERT_TOKENS = 17      # measured mean Qwen3 tokens per inserted sentence (invented names tokenize long)


def filler_block_tokens(block_tokens: int, n_inserts: int, n_origins: int, adjust: int = 0) -> int:
    """Target filler tokens per block so that filler + inserted sentences average `block_tokens` per origin and the
    item totals length_tokens (within 2% even at 8K, where 9-20 inserted sentences are ~3% of the item)."""
    return max(block_tokens // 2, block_tokens - round(n_inserts * INSERT_TOKENS / n_origins) + adjust)


def check_filler_split(filler, split: str) -> None:
    ps = getattr(filler, "pg19_split", None)
    if ps is not None and {"dev": "validation", "test": "test"}[split] != ps:
        raise ValueError(f"{split} items must use the PG-19 {'validation' if split == 'dev' else 'test'} split, "
                         f"filler is {ps}")


# ---------------------------------------------------------------- generator
def correct_length(make, length_tokens: int, tol: float = 0.005, max_passes: int = 4):
    """Length control. `make(adjust)` builds the item with every filler block `adjust` tokens longer. While the item is
    more than `tol` off length_tokens it is rebuilt with the accumulated per-block correction (deterministic: same
    seed, same cell); the closest item is returned. Block ends snap to sentence boundaries, so one pass leaves
    noise; the build check allows 2%."""
    adj, best, best_err = 0, None, None
    for _ in range(max_passes):
        it = make(adj)
        diff = length_tokens - sum(o["n_tokens"] for o in it["origins"])
        if best is None or abs(diff) < best_err:
            best, best_err = it, abs(diff)
        if abs(diff) / length_tokens <= tol:
            break
        step = round(diff / len(it["origins"]))
        adj += step if step else (1 if diff > 0 else -1)
    return best


def make_item(seed: int, hops: int, length_tokens: int, placement: str, alpha: float, filler, tokenizer,
              split: str = "dev", n_distractor_chains: int = 2, block_tokens: int | None = None) -> dict:
    return correct_length(lambda adj: _make_item(seed, hops, length_tokens, placement, alpha, filler, tokenizer, split,
                                                 n_distractor_chains, block_tokens, adj), length_tokens)


def _make_item(seed: int, hops: int, length_tokens: int, placement: str, alpha: float, filler, tokenizer,
               split: str, n_distractor_chains: int, block_tokens: int | None, fbt_adjust: int) -> dict:
    block_tokens = default_block_tokens() if block_tokens is None else block_tokens
    check_split(split, seed)
    check_filler_split(filler, split)
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    if hops < 2:
        raise ValueError("hops >= 2")
    tok = as_tokenizer(tokenizer)
    cell = cell_name(placement, hops, length_tokens, alpha)
    rng = item_rng(DATASET, cell, seed)
    n_origins = length_tokens // block_tokens

    fbt = filler_block_tokens(block_tokens, hops * (1 + n_distractor_chains), n_origins, fbt_adjust)
    blocks = filler.take_blocks(rng, n_origins, tok, fbt, books=list(rng.permutation(filler.n_books())))
    ng = NameGen(rng, vocab=filler.vocab(), text_lower="\n".join(b["text"] for b in blocks).lower())
    comp = COMPONENTS[int(rng.integers(len(COMPONENTS)))]
    kind = list(ATTRIBUTES)[int(rng.integers(len(ATTRIBUTES)))]
    values = draw_values(rng, kind, 1 + n_distractor_chains)
    chain_names = [ng.entity() for _ in range(hops)]
    d_names = [[ng.entity() for _ in range(hops)] for _ in range(n_distractor_chains)]

    chain = build_chain(rng, chain_names, comp, kind, values[0], alpha)
    fact_origins = place_chain(rng, n_origins, hops, placement, group=max(1, 2048 // block_tokens))
    inserts: dict[int, list[dict]] = {}
    for f, o in zip(chain, fact_origins):
        inserts.setdefault(o, []).append({"key": ("fact", f["hop"]), "sentence": f["sentence"], "kind": "fact"})
    d_sents = []
    for k, nm in enumerate(d_names):
        for f in build_chain(rng, nm, comp, kind, values[1 + k], alpha):
            o = int(rng.integers(n_origins))
            d_sents.append((k, f, o))
            inserts.setdefault(o, []).append({"key": ("dist", k, f["hop"]), "sentence": f["sentence"],
                                              "kind": "distractor"})

    origins, spans = assemble_origins(blocks, [0] * n_origins, inserts, rng, tok)
    lb = late_bound_flags(fact_origins)
    facts = []
    for f, o, late in zip(chain, fact_origins, lb):
        oid, sp = spans[("fact", f["hop"])]
        assert oid == o
        facts.append({"hop": f["hop"], "origin_id": o, "char_span": sp, "text": f["sentence"], "subject": f["subject"],
                      "subject_surface": f["subject_surface"], "reveals": f["reveals"], "late_bound": bool(late),
                      "cross_thread": False})
    distractors = []
    for k, f, o in d_sents:
        oid, sp = spans[("dist", k, f["hop"])]
        distractors.append({"origin_id": oid, "char_span": sp, "subject": f["subject"],
                            "subject_surface": f["subject_surface"], "reveals": f["reveals"], "text": f["sentence"],
                            "chain": k, "hop": f["hop"]})
    distractors.sort(key=lambda d: (d["origin_id"], d["char_span"][0]))
    aliases: dict[str, list[str]] = {}
    for f in chain:
        if f["subject_surface"] != f["subject"]:
            aliases.setdefault(f["subject"], []).append(f["subject_surface"])

    return {
        "item_id": make_item_id(DATASET, split, cell, seed),
        "dataset": DATASET, "split": split, "seed": int(seed), "generator_version": GENERATOR_VERSION, "cell": cell,
        "params": {"hops": hops, "length_tokens": length_tokens, "block_tokens": block_tokens, "placement": placement,
                   "alpha": alpha, "n_threads": 1, "cross_thread": False, "n_distractor_chains": n_distractor_chains,
                   "filler": filler.name, "tokenizer": tok.name},
        "origins": origins,
        "question": question_text(kind, comp, chain_names[0].full),
        "question_thread": 0,
        "answer": fmt_value(kind, values[0]),
        "answer_aliases": answer_aliases(kind, values[0]),
        "gold": {
            "entities": [n.full for n in chain_names],
            "aliases": aliases,
            "facts": facts,
            "fact_origins": [int(o) for o in fact_origins],
            "h_late": int(sum(lb)),
            "thread_of": [0] * n_origins,
            "cross_thread_hop": None,
            "confusers": [],
            "distractors": distractors,
            "component": comp,
            "attribute": kind,
            "distractor_values": [fmt_value(kind, v) for v in values[1:]],
        },
    }
