"""MT-LB-MHQA: multi-thread, cross-thread late binding (protocol 3.3).

Extends make_mt_item (opa/reference/protocol_snippets_v2.py) to interleaved arrival:
- T threads, each reading its own PG-19 book continuously. Arrival is a sequence of runs of `run_blocks` (4)
  consecutive blocks; each thread gets an equal share of runs and the run order is a random shuffle, relabelled by
  first appearance, so switching looks like A A B A C B C C A ... Runs of 4 blocks keep rechunk(B=1024, 2048) aligned.
  v3 departure: the question thread (which owns the last run) gets at least hops + 1 runs, because at 32K each thread
  would otherwise have only 2 (T = 8) or 1 (T = 16) runs and the chain's facts need distinct 4-block groups; the
  other threads share the rest. When 4-block runs leave fewer than T + hops runs (T = 16 at 32K: 16 runs), runs are 2
  blocks long (aligned to B = 512 and 1024 only; B = 2048 rechunk is refused for those cells). params.run_blocks
  records it.
- The question belongs to the most recent thread (the thread of the last block). Its chain's facts are placed in
  reverse order inside that thread (hop 0 latest), so every hop after the first is late-bound. Chain facts lie in
  distinct 4-block groups (distinct runs), so h_late is unchanged by rechunk to B = 1024 or 2048.
- cross_thread=True: the last hop's fact is instead stated in a different thread, in an origin before the origin that
  reveals the last entity (so it stays late-bound). The protocol text says "with probability 0.5 per item"; here
  cross-thread is a cell factor (grid: cross-thread in {no, yes}), so each cell is all-no or all-yes.
- Confusers: every other thread (each with probability confuser_rate, default 1.0) contains a near-duplicate of the
  final fact (same template) about the one-character twin of the final entity, with a different value. All values in
  an item (answer, confusers, distractor chains) are distinct, so a wrong value identifies its source.
- Distractor chains and alias variants as in LB-MHQA.
- decoys=True (thread-confusion task for H12, cell suffix "-conf", see opa/data/confusion.py): instead of the single
  shared twin, every other thread holds exactly one decoy of the final fact, of kind twin / suffix (suffix only when the
  final entity has a product-type suffix, never a legal form; see confusion.py). Decoys are stored in
  gold["confusers"] with kind, hop and value_aliases. Always one decoy per other thread (confuser_rate is ignored).
"""
from __future__ import annotations

import re

import numpy as np

from opa.data import GENERATOR_VERSION, check_split, default_block_tokens, item_rng, make_item_id
from opa.data import confusion
from opa.data.filler import as_tokenizer, assemble_origins
from opa.data.lbmhqa import (ATTRIBUTES, COMPONENTS, alpha_label, answer_aliases, attribute_sentence, build_chain,
                             check_filler_split, correct_length, draw_values, filler_block_tokens, fmt_value, late_bound_flags, length_label, parse_alpha,
                             parse_length, question_text, spread_origins)
from opa.data.names import NameGen

DATASET = "mtlbmhqa"


def cell_name(n_threads: int, hops: int, length_tokens: int, cross_thread: bool, alpha: float = 0.0,
              decoys: bool = False) -> str:
    c = f"T{n_threads}-h{hops}-{length_label(length_tokens)}-{'cross' if cross_thread else 'nocross'}"
    c = c if alpha == 0 else f"{c}-{alpha_label(alpha)}"
    return c + "-conf" if decoys else c


def parse_cell(cell: str) -> dict:
    m = re.fullmatch(r"T(\d+)-h(\d+)-(\d+k)-(cross|nocross)(?:-(a[\d.]+))?(-conf)?", cell)
    if not m:
        raise ValueError(f"bad MT-LB-MHQA cell {cell!r} (expected e.g. T8-h3-32k-cross or T8-h3-32k-cross-conf)")
    return {"n_threads": int(m.group(1)), "hops": int(m.group(2)), "length_tokens": parse_length(m.group(3)),
            "cross_thread": m.group(4) == "cross", "alpha": parse_alpha(m.group(5)) if m.group(5) else 0.0,
            "decoys": bool(m.group(6))}


def run_schedule(rng: np.random.Generator, n_runs: int, n_threads: int, min_question_runs: int = 0) -> list[int]:
    """Thread of each run: equal shares in random order, thread ids relabelled by first appearance. The question
    thread (the thread of the last run) gets at least `min_question_runs` runs (the chain's facts need distinct runs);
    the other threads share the rest equally, each with at least one run."""
    base = [n_runs // n_threads + (1 if t < n_runs % n_threads else 0) for t in range(n_threads)]
    if base[0] < min_question_runs:                       # thread 0 is the question thread before relabelling
        extra = min_question_runs - base[0]
        base[0] = min_question_runs
        t = n_threads - 1
        while extra > 0:
            if t == 0:
                raise ValueError(f"{n_runs} runs cannot give {n_threads} threads a run and the question thread "
                                 f"{min_question_runs}")
            if base[t] > 1:
                base[t] -= 1
                extra -= 1
            else:
                t -= 1
    labels = np.repeat(np.arange(n_threads), base)
    labels = labels[rng.permutation(len(labels))]
    if labels[-1] != 0:                                   # the question thread owns the last run
        k = int(np.where(labels == 0)[0][-1])
        labels[k], labels[-1] = labels[-1], labels[k]
    relabel: dict[int, int] = {}
    for x in labels:
        relabel.setdefault(int(x), len(relabel))
    return [relabel[int(x)] for x in labels]


def auto_run_blocks(n_origins: int, n_threads: int, hops: int, group: int = 4) -> int:
    """Runs of 4 blocks (aligned to B = 2048) when every thread can have a run and the question thread keeps
    hops + 1; otherwise runs of 2 blocks (aligned to B = 1024 only; e.g. T = 16 at 32K has just 16 runs of 4)."""
    for rb in (4, 2):
        if n_origins % rb == 0 and n_origins // rb >= n_threads + hops:
            return rb
    raise ValueError(f"{n_origins} origins are too few for {n_threads} threads and {hops} hops")


def make_item(seed: int, n_threads: int, hops: int, length_tokens: int, cross_thread: bool, alpha: float, filler,
              tokenizer, split: str = "dev", confuser_rate: float = 1.0, run_blocks: int | None = None,
              n_distractor_chains: int = 2, block_tokens: int | None = None, decoys: bool = False) -> dict:
    return correct_length(lambda adj: _make_item(seed, n_threads, hops, length_tokens, cross_thread, alpha, filler,
                                                 tokenizer, split, confuser_rate, run_blocks, n_distractor_chains,
                                                 block_tokens, decoys, adj), length_tokens)


def _make_item(seed: int, n_threads: int, hops: int, length_tokens: int, cross_thread: bool, alpha: float, filler,
               tokenizer, split: str, confuser_rate: float, run_blocks: int | None, n_distractor_chains: int,
               block_tokens: int | None, decoys: bool, fbt_adjust: int) -> dict:
    block_tokens = default_block_tokens() if block_tokens is None else block_tokens
    check_split(split, seed)
    check_filler_split(filler, split)
    if n_threads < 2 or hops < 2:
        raise ValueError("n_threads >= 2 and hops >= 2")
    tok = as_tokenizer(tokenizer)
    cell = cell_name(n_threads, hops, length_tokens, cross_thread, alpha, decoys)
    rng = item_rng(DATASET, cell, seed)
    n_origins = length_tokens // block_tokens
    if run_blocks is None:
        run_blocks = auto_run_blocks(n_origins, n_threads, hops)
    if n_origins % run_blocks:
        raise ValueError("length must be a whole number of runs")
    n_runs = n_origins // run_blocks
    if n_runs < n_threads:
        raise ValueError("fewer runs than threads")
    group = max(1, 2048 // block_tokens)
    for _ in range(200):          # the question thread needs `hops` distinct 4-block groups for the chain's facts
        sched = run_schedule(rng, n_runs, n_threads, min_question_runs=hops + 1)
        thread_of = [t for t in sched for _ in range(run_blocks)]
        qt0 = thread_of[-1]
        first_other = min(o for o in range(n_origins) if thread_of[o] != qt0)
        qg = {o // group for o in range(n_origins) if thread_of[o] == qt0}
        # cross cells: hops - 1 question-thread facts must lie after some other-thread origin (the cross-thread fact
        # goes before the earliest of them)
        qg_after = {o // group for o in range(first_other, n_origins) if thread_of[o] == qt0}
        if len(qg) >= hops and (not cross_thread or len(qg_after) >= hops - 1):
            break
    else:
        raise RuntimeError("could not schedule enough question-thread groups")

    # each thread reads its own book (short books continue into the thread's own fallback books)
    nb = filler.n_books()
    if nb < n_threads:
        raise ValueError(f"filler has {nb} books, need one per thread ({n_threads})")
    perm = [int(x) for x in rng.permutation(nb)]
    per_thread = [[perm[t]] + perm[n_threads + t::n_threads] for t in range(n_threads)]
    fbt = filler_block_tokens(block_tokens, hops * (1 + n_distractor_chains) + n_threads - 1, n_origins,
                              fbt_adjust)
    tblocks = [filler.take_blocks(rng, thread_of.count(t), tok, fbt, books=per_thread[t])
               for t in range(n_threads)]
    nxt = [0] * n_threads
    blocks = []
    for t in thread_of:
        blocks.append(tblocks[t][nxt[t]])
        nxt[t] += 1

    ng = NameGen(rng, vocab=filler.vocab(), text_lower="\n".join(b["text"] for b in blocks).lower())
    comp = COMPONENTS[int(rng.integers(len(COMPONENTS)))]
    kind = list(ATTRIBUTES)[int(rng.integers(len(ATTRIBUTES)))]
    chain_names = [ng.entity() for _ in range(hops)]
    qt = thread_of[-1]
    decoy_kinds: list[str] = []
    if decoys:
        decoy_kinds = confusion.assign_kinds(rng, n_threads - 1, confusion.allowed_kinds(chain_names[-1].full))
        twin = ng.twin(chain_names[-1]) if "twin" in decoy_kinds else None
    else:
        twin = ng.twin(chain_names[-1])
    d_names = [[ng.entity() for _ in range(hops)] for _ in range(n_distractor_chains)]

    q_origins = [o for o in range(n_origins) if thread_of[o] == qt]
    for _ in range(1000):
        if cross_thread:
            chosen = spread_origins(rng, [o for o in q_origins if o > first_other], hops - 1, group)[::-1]
            cands = [o for o in range(chosen[-1]) if thread_of[o] != qt]
            if not cands:
                continue
            fact_origins = chosen + [int(cands[int(rng.integers(len(cands)))])]
        else:
            fact_origins = spread_origins(rng, q_origins, hops, group)[::-1]
        break
    else:
        raise RuntimeError("could not place a cross-thread hop")

    other_threads = [t for t in range(n_threads) if t != qt]
    conf_threads = list(other_threads) if decoys else [t for t in other_threads if rng.random() < confuser_rate]
    values = draw_values(rng, kind, 1 + n_distractor_chains + len(conf_threads))
    chain = build_chain(rng, chain_names, comp, kind, values[0], alpha)
    final_tpl = chain[-1]["template"]

    inserts: dict[int, list[dict]] = {}
    for f, o in zip(chain, fact_origins):
        inserts.setdefault(o, []).append({"key": ("fact", f["hop"]), "sentence": f["sentence"], "kind": "fact"})
    conf_recs = []
    for i, t in enumerate(conf_threads):
        t_origins = [o for o in range(n_origins) if thread_of[o] == t]
        o = int(t_origins[int(rng.integers(len(t_origins)))])
        v = values[1 + n_distractor_chains + i]
        if decoys:
            dk = decoy_kinds[i]
            dname = confusion.decoy_name(dk, chain_names[-1], twin, rng)
        else:
            dk, dname = "twin", twin
        s = attribute_sentence(ATTRIBUTES[kind]["templates"][final_tpl], dname.full, comp, kind, v)
        conf_recs.append((t, o, v, s, dk, dname.full))
        inserts.setdefault(o, []).append({"key": ("conf", t), "sentence": s, "kind": "confuser"})
    d_sents = []
    for k, nm in enumerate(d_names):
        for f in build_chain(rng, nm, comp, kind, values[1 + k], alpha):
            o = int(rng.integers(n_origins))
            d_sents.append((k, f, o))
            inserts.setdefault(o, []).append({"key": ("dist", k, f["hop"]), "sentence": f["sentence"],
                                              "kind": "distractor"})

    origins, spans = assemble_origins(blocks, thread_of, inserts, rng, tok)
    lb = late_bound_flags(fact_origins)
    facts = []
    for f, o, late in zip(chain, fact_origins, lb):
        oid, sp = spans[("fact", f["hop"])]
        assert oid == o
        facts.append({"hop": f["hop"], "origin_id": o, "char_span": sp, "text": f["sentence"], "subject": f["subject"],
                      "subject_surface": f["subject_surface"], "reveals": f["reveals"], "late_bound": bool(late),
                      "cross_thread": bool(thread_of[o] != qt), "thread_id": thread_of[o]})
    confusers = []
    for t, o, v, s, dk, dname in conf_recs:
        oid, sp = spans[("conf", t)]
        rec = {"entity": dname, "value": fmt_value(kind, v), "origin_id": oid, "thread_id": t,
               "char_span": sp, "text": s}
        rec["value_aliases"] = answer_aliases(kind, v)          # F8 / EM-symmetric scoring (also in non-conf cells)
        if decoys:
            rec.update({"kind": dk, "hop": hops - 1})
        confusers.append(rec)
    distractors = []
    for k, f, o in d_sents:
        oid, sp = spans[("dist", k, f["hop"])]
        distractors.append({"origin_id": oid, "char_span": sp, "subject": f["subject"],
                            "subject_surface": f["subject_surface"], "reveals": f["reveals"], "text": f["sentence"],
                            "chain": k, "hop": f["hop"], "thread_id": thread_of[oid]})
    distractors.sort(key=lambda d: (d["origin_id"], d["char_span"][0]))
    aliases: dict[str, list[str]] = {}
    for f in chain:
        if f["subject_surface"] != f["subject"]:
            aliases.setdefault(f["subject"], []).append(f["subject_surface"])
    h_late = int(sum(lb))
    assert h_late == hops - 1, (fact_origins, lb)
    question = question_text(kind, comp, chain_names[0].full)

    return {
        "item_id": make_item_id(DATASET, split, cell, seed),
        "dataset": DATASET, "split": split, "seed": int(seed), "generator_version": GENERATOR_VERSION, "cell": cell,
        "params": {"hops": hops, "length_tokens": length_tokens, "block_tokens": block_tokens, "placement": "late",
                   "alpha": alpha, "n_threads": n_threads, "cross_thread": bool(cross_thread),
                   "confuser_rate": 1.0 if decoys else confuser_rate, "run_blocks": run_blocks, "decoys": bool(decoys),
                   "n_distractor_chains": n_distractor_chains, "filler": filler.name, "tokenizer": tok.name},
        "origins": origins,
        "question": question,
        "question_thread": qt,
        "answer": fmt_value(kind, values[0]),
        "answer_aliases": answer_aliases(kind, values[0]),
        "gold": {
            "entities": [n.full for n in chain_names],
            "aliases": aliases,
            "facts": facts,
            "fact_origins": [int(o) for o in fact_origins],
            "h_late": h_late,
            "thread_of": thread_of,
            "cross_thread_hop": hops - 1 if cross_thread else None,
            "confusers": confusers,
            "distractors": distractors,
            "component": comp,
            "attribute": kind,
            "twin": twin.full if twin is not None else None,
            "run_schedule": sched,
            "distractor_values": [fmt_value(kind, v) for v in values[1:1 + n_distractor_chains]],
        },
    }
