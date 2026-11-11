"""Track A systems S1-S18 (protocol 3.5). Each controller runs one item and returns a per-item record
(opa/data/SCHEMA.md). All model calls go through models.Chat; every controller action is logged as one event.

OPA systems share Fold (paper Algorithm 3 at the agent level):
  S6  fold-left, no skeleton            S7  + S-lex skeleton faults        S8  + S-hyb skeleton faults
  S9  + oracle skeleton (ORACLE)        S10 S8 with abstractive notes      S11 S8 + graph faults (five edge types)
  S12 S11 + shell descent + hyperbolic penalty (arm A2)                    S14 S12 with Euclidean distance (A1)
  S15 S12 with exact tree distance (A3) S16 S12 with lambda = 0 (A0)       S12F S12 with fitted addresses (A4)
  S17 S12 with an HNSW-style flat graph index instead of shells          S18 S12 + oracle skeleton and graph (ORACLE)
Baselines: S1 full context, S2 top-k hybrid RAG, S3 IRCoT-style, S4 ReadAgent-style, S5 Chain-of-Agents-style,
S13 RAPTOR-style tree of abstractive summaries.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import CONTEXT_LIMIT, TOKEN_BUDGET
from .models import MISSING, Chat, ContextOverflow, Usage, answer_text, ask_json, parse_json, salvage_answer, salvage_string_field
from .text import QUESTION_STOP, RX_NUM_UNIT, find_exact, lexical_entities, mentions, normalize_key

LEGAL_FORMS = {"ltd", "inc", "gmbh", "corp", "llc", "plc", "ag", "sa", "co", "bv"}
# a pinned answer value: a number with a unit (S-lex pattern) or a service life in hours ("3500 hours", which the S-lex
# pattern does not list; the generator's three attribute kinds are temperature, pressure and service life)
RX_VALUE = re.compile(RX_NUM_UNIT.pattern + r"|\d+(?:[.,]\d+)?\s?hours?\b")
# a `reveals` entry that is a value, not an entity: Qwen3-4B lists the answer value of the final fact as a revealed
# "entity" (micro-pilot shapes: "37 bar", "7700", "3100 hours", "76 °C", "85 c"; units often dropped). Identifiers
# with letters before the digits (QX-2291) are entities.
RX_VALUE_REVEAL = re.compile(r"\s*[\d.,]+\s*[A-Za-z°%]*\s*")


def legal_core(name: str) -> str:
    """normalize_key(name) with trailing legal-form words removed (only those: not pumps, works, systems, ...)."""
    words = normalize_key(name).split()
    while len(words) > 1 and words[-1] in LEGAL_FORMS:
        words.pop()
    return " ".join(words)


def entity_reveals(reveals: list[str]) -> list[str]:
    """The entities among a pin's `reveals`: the values (RX_VALUE_REVEAL) are the answer, not entities, so they never
    become relevant (R), fault targets or chain hops."""
    return [r for r in reveals if not RX_VALUE_REVEAL.fullmatch(r)]


SYSTEMS = {
    #       skeleton   graph  descent  arm   notes  oracle_graph
    "S6": ("none", False, None, "A0", False, False),
    "S7": ("lex", False, None, "A0", False, False),
    "S8": ("hyb", False, None, "A0", False, False),
    "S9": ("oracle", False, None, "A0", False, False),
    "S10": ("hyb", False, None, "A0", True, False),
    "S11": ("hyb", True, None, "A0", False, False),
    "S12": ("hyb", True, "shells", "A2", False, False),
    "S14": ("hyb", True, "shells", "A1", False, False),
    "S15": ("hyb", True, "shells", "A3", False, False),
    "S16": ("hyb", True, "shells", "A0", False, False),
    "S12F": ("hyb", True, "shells", "A4", False, False),
    "S17": ("hyb", True, "hnsw", "A2", False, False),
    "S18": ("oracle", True, "shells", "A2", False, True),
}
BASELINES = ("S1", "S2", "S3", "S4", "S5", "S13")
ORACLE_SYSTEMS = ("S9", "S18")
ARM_OF = {"S16": "A0", "S14": "A1", "S12": "A2", "S15": "A3", "S12F": "A4"}


def session_cfg(system: str, opts: dict) -> dict:
    """Ingest configuration for a system (what Session needs to build)."""
    if system in SYSTEMS:
        sk, graph, _, _, _, oracle_graph = SYSTEMS[system]
    else:
        sk, graph, oracle_graph = "lex", False, False
    return {"skeleton": sk, "graph": graph, "oracle_graph": oracle_graph, **{k: opts[k] for k in
            ("k_max", "tau_thread", "tau_x", "b", "m", "delta", "d_clamp", "use_gold_threads") if k in opts}}


# ---------------------------------------------------------------- prompts

READ_SYSTEM = """You are the reading module of a paged memory system that answers a question about a long archive.
You see the question, the entities the question currently depends on, the evidence pinned so far, and one or more archive origins (text chunks with ids).
Pin every sentence in the shown origins that states a fact about one of the relevant entities that the question may need. Copy each quote exactly, character for character, from the origin text (one or two sentences, no ellipses).
For each pin, name the relevant entity it is about and list any NEW entities that the fact links it to (the question now also depends on them).
Set "answer_found" to true only if the pinned evidence, old and new together, already contains the final answer.
Reply with one JSON object and nothing else:
{"pins": [{"origin_id": <int>, "quote": "<exact text>", "entity": "<relevant entity>", "reveals": ["<new entity>"]}], "answer_found": <true|false>}"""

NOTES_SYSTEM = """You are the reading module of a memory system that answers a question about a long archive.
You see the question, the entities the question currently depends on, your notes so far, and one or more archive origins (text chunks with ids).
For every fact in the shown origins about one of the relevant entities that the question may need, write a short note in your own words.
For each note, name the relevant entity it is about and list any NEW entities that the fact links it to.
Set "answer_found" to true only if your notes, old and new together, already contain the final answer.
Reply with one JSON object and nothing else:
{"notes": [{"origin_id": <int>, "note": "<your note>", "entity": "<relevant entity>", "reveals": ["<new entity>"]}], "answer_found": <true|false>}"""

ANSWER_SYSTEM = """Answer the question using only the evidence given. Reply with one JSON object and nothing else:
{"answer": "<short answer: only the value asked for, with its unit if it has one>", "cited_pin_ids": ["<evidence id>"]}
If the evidence does not contain the answer, give your best guess and cite nothing."""

# S4 (ReadAgent-style) configuration, dev tuning iteration 1 (config version s4-v2):
#  - pages of 1 origin (~512 tokens; ReadAgent pages are ~600 tokens), gist cap 80 words per page (about 4-5x compression,
#    inside ReadAgent's reported 3-20x), gist max_tokens 300 with salvage of a cut-off gist (v1: 4-origin pages, 160 tokens,
#    a cut-off gist was returned as "" - 23.5% empty in the audit);
#  - parallel lookup of up to L = 5 pages (ReadAgent-P), told that it may choose several pages and should take any page that
#    may contain needed facts. No sequential lookup (that would be a further iteration).
S4_CONFIG_VERSION = "s4-v2"
S4_TUNING_ITERATION = 1
S4_GIST_MAX_TOKENS = 300
S4_LOOKUP_MAX_TOKENS = 200
GIST_SYSTEM = """Shorten the page into a gist of at most 80 words. Keep every name, identifier, number and relation; drop description and dialogue. Reply with one JSON object: {"gist": "<text>"}"""
LOOKUP_SYSTEM = """You see a question and numbered gists of the pages of a long document. A gist leaves out details, so the full text of a page may hold facts that its gist does not show. Choose every page whose full text may contain a fact you need to answer the question, up to {L} pages; you may choose several pages. Reply with one JSON object: {{"pages": [<page numbers>]}}"""
COA_SYSTEM = """You are one worker in a chain reading a long document in order. You see the question, the notes passed on by the previous worker, and your part of the document. Update the notes: keep everything relevant to the question from the old notes and add anything relevant from your part (names, relations, numbers, exactly as written). At most 200 words. Reply with one JSON object: {"notes": "<text>"}"""
IRCOT_SYSTEM = """You answer a multi-hop question by reasoning one step at a time over retrieved paragraphs. Write the next single reasoning sentence. If the paragraphs already give the answer, put it in "answer". Reply with one JSON object: {"thought": "<one sentence>", "answer": <null or "short answer">}"""
SUMMARY_SYSTEM = """Summarize the passages in at most 120 words, keeping every name, identifier, number and relation. Reply with one JSON object: {"summary": "<text>"}"""


def fmt_origins(sess, ids) -> str:
    return "\n".join(f'<origin id="{o}">\n{sess.texts[o]}\n</origin>' for o in ids)


@dataclass
class Pin:
    pin_id: str
    origin_id: int
    char_span: list
    quote: str
    entity: str
    reveals: list
    evicted: bool = False
    note: bool = False

    def tokens(self) -> int:
        return max(1, len(self.quote) // 4)


@dataclass
class Ctx:
    item: dict
    sess: object
    chat: Chat
    opts: dict
    log_fn: object
    usage: Usage = field(default_factory=Usage)
    answer_failure: str | None = None        # failure code of the last answer_from call (None = a clean first reply)

    def log(self, action: str, **kw):
        self.log_fn({"item_id": self.item["item_id"], "action": action, **kw})

    def over_budget(self) -> bool:
        return self.usage.tokens_in >= int(self.opts.get("token_budget", TOKEN_BUDGET))


def question_entities(q: str) -> list[str]:
    """The entities the question names, in order of first appearance (lexical_entities is a set: its iteration order
    would follow the process's hash seed, and the order seeds R, the fault order and the chain walk's roots)."""
    keys = [k for k in lexical_entities(q) if k not in QUESTION_STOP and not RX_NUM_UNIT.fullmatch(k)]
    keys = [k for k in keys if not any(k != o and k in o for o in keys)]    # keep the longest form
    nq = normalize_key(q)
    return sorted(keys, key=lambda k: (nq.find(k), k))


def base_record(c: Ctx, system: str) -> dict:
    return {"item_id": c.item["item_id"], "system": system, "model": c.chat.name, "prediction": "", "cited_pin_ids": [],
            "pins": [], "paged_in": [], "evicted": [], "faults": [], "descend_candidates": [], "skeleton": {},
            "status": "ok", "error": None}


REASK = "\n\nYour previous reply was not a single valid JSON object. Reply with JSON only."


def answer_from(c: Ctx, evidence: str, max_tokens: int = 200) -> tuple[str, list]:
    """One answer call. A usable first answer is never discarded and a failed retry never yields a silent empty answer:
      - valid JSON with an "answer" key: used (0 and false are answers; only null / missing / "" are empty);
      - invalid JSON (unescaped quotes elsewhere, a runaway list cut off by max_tokens) but a complete "answer" value
        can be read: that value is used, cited ids are empty, failure code `answer_salvaged`;
      - nothing readable: one retry with the reminder; its answer is used (`answer_retried`), else salvaged from it
        (`answer_salvaged_retry`), else the prediction is empty with failure code `answer_unparseable`;
      - a parsed but empty answer gets `answer_empty`.
    The code goes to c.answer_failure (run_item copies it to the record as `answer_parse`; it is not the scorer's F-code) and to the answer event."""
    user = f"Question: {c.item['question']}\n\nEvidence:\n{evidence}"
    text = c.chat.ask(ANSWER_SYSTEM, user, max_tokens, c.usage, c.log_fn)
    obj, code, ans, cited = parse_json(text), None, MISSING, []
    if obj is not None and "answer" in obj:
        ans, cited = obj["answer"], obj.get("cited_pin_ids") or []
    else:
        ans = salvage_answer(text)
        if ans is not MISSING:
            code = "answer_salvaged"
        else:
            text2 = c.chat.ask(ANSWER_SYSTEM, user + REASK, max_tokens, c.usage, c.log_fn)
            obj2 = parse_json(text2)
            if obj2 is not None and "answer" in obj2:
                ans, cited, code = obj2["answer"], obj2.get("cited_pin_ids") or [], "answer_retried"
            else:
                ans = salvage_answer(text2)
                code = "answer_salvaged_retry" if ans is not MISSING else "answer_unparseable"
                if ans is MISSING:
                    c.log_fn({"item_id": c.item["item_id"], "action": "parse_error", "reply_head": text2[:200]})
    pred = answer_text(ans)
    if not pred.strip() and code is None:
        code = "answer_empty"
    c.answer_failure = code
    c.log("answer", answer=pred[:200], answer_parse=code)
    return pred, [str(x) for x in (cited if isinstance(cited, list) else [cited])]


# ---------------------------------------------------------------- the OPA fold (Algorithm 3)


class Fold:
    def __init__(self, c: Ctx, system: str):
        self.c, self.system = c, system
        self.sk, self.graph, self.descent, arm, self.notes, _ = SYSTEMS[system]
        self.arm = c.opts.get("arm", arm)
        self.lam = float(c.opts.get("lambda", 2.0)) if self.arm != "A0" else 0.0
        self.M = int(c.opts.get("M", 4))
        self.window = int(c.opts.get("read_window", max(1, self.M - 1)))
        self.P_pin = int(c.opts.get("P_pin", 2048))
        self.fault_top = int(c.opts.get("fault_top", 3))
        self.max_faults = int(c.opts.get("max_faults", 16))
        s = c.sess
        self.q_origin = s.N - 1                                  # the question attaches to the latest thread head
        self.fitted = s.fit_addresses(seed=int(c.opts.get("seed", 0))) if self.arm == "A4" else None
        self.slots: list[int] = []                               # LRU order, most recent last
        self.pins: list[Pin] = []
        self.roots: list[str] = question_entities(c.item["question"])   # the entities the question names: R starts here
        self.R: list[str] = []
        self.became_relevant: dict[str, set] = {}
        self.tried: set = set()
        self.rec = base_record(c, system)
        self.rec.update(arm=self.arm, **{"lambda": self.lam})
        self.n_faults = 0

    # ---- memory operations (each logs one event)
    def page_in(self, o: int, via: str = "read"):
        if o in self.slots:
            self.slots.remove(o)
        elif len(self.slots) >= self.M:
            ev = self.slots.pop(0)
            self.rec["evicted"].append({"origin_id": ev, "reason": "lru"})
            self.c.log("evict", origin_id=ev, reason="lru")
        self.slots.append(o)
        self.rec["paged_in"].append(o)
        self.c.log("page_in", origin_id=o, via=via, slot=len(self.slots) - 1, thread_id=self.c.sess.thread_of[o])

    def evict(self, o: int):
        if o in self.slots:
            self.slots.remove(o)
            self.rec["evicted"].append({"origin_id": o, "reason": "explicit"})
            self.c.log("evict", origin_id=o, reason="explicit")

    def live(self):
        return [p for p in self.pins if not p.evicted]

    def covered(self) -> set:
        return {normalize_key(p.entity) for p in self.live()}

    def add_relevant(self, e: str):
        k = normalize_key(e)
        if k and k not in self.R:
            self.R.append(k)
            self.became_relevant[k] = set(self.slots)

    def pin(self, o: int, quote: str, entity: str, reveals: list, note: bool) -> bool:
        if note:
            span, q = [0, 0], quote
        else:
            sp = find_exact(quote, self.c.sess.texts[o])
            if sp is None:                                     # exact-pin rule (protocol 3.1 item 8)
                self.c.log("pin_rejected", origin_id=o, quote=quote[:200])
                return False
            span, q = list(sp), self.c.sess.texts[o][sp[0]:sp[1]]
        if any(p.origin_id == o and p.char_span == span and not p.evicted for p in self.pins) and not note:
            return False
        p = Pin(f"p{len(self.pins) + 1}", o, span, q, entity, list(reveals or []), note=note)
        self.pins.append(p)
        self.c.log("pin", origin_id=o, pins=[{"pin_id": p.pin_id, "origin_id": o, "char_span": span}], entity=entity)
        return True

    def enforce_budget(self):
        while sum(p.tokens() for p in self.live()) > self.P_pin:
            latest = {}
            for p in self.live():
                latest[normalize_key(p.entity)] = p.pin_id
            victims = [p for p in self.live() if p.pin_id not in latest.values()] or self.live()[:1]
            victims[0].evicted = True
            self.c.log("evict_pin", pin_id=victims[0].pin_id, origin_id=victims[0].origin_id, reason="budget")

    # ---- the model-facing read step
    def read(self, ids: list[int], focus: list[str] | None = None) -> bool:
        s = self.c.sess
        ev = "\n".join(f'[{p.pin_id}] (origin {p.origin_id}) "{p.quote}"' for p in self.live()) or "(none)"
        rel = "; ".join(focus or self.R) or "(those named in the question)"
        user = (f"Question: {self.c.item['question']}\n\nRelevant entities: {rel}\n\n"
                f"{'Notes' if self.notes else 'Pinned evidence'} so far:\n{ev}\n\nOrigins:\n{fmt_origins(s, ids)}")
        out = ask_json(self.c.chat, NOTES_SYSTEM if self.notes else READ_SYSTEM, user, 700, self.c.usage, self.c.log_fn)
        items = out.get("notes" if self.notes else "pins") or []
        for it in items if isinstance(items, list) else []:
            try:
                o = int(it.get("origin_id"))
            except (TypeError, ValueError):
                continue
            if o not in ids:
                continue
            text = it.get("note" if self.notes else "quote") or ""
            ent = str(it.get("entity") or "")
            rev = [str(x) for x in (it.get("reveals") or []) if x]
            if self.pin(o, text, ent, rev, self.notes):
                for e in entity_reveals(rev):                     # the pin keeps the raw list; values never enter R
                    self.add_relevant(e)
        self.enforce_budget()
        return bool(out.get("answer_found"))

    # ---- fault sources (skeleton and graph), ranked by relevance - lambda * d(q, c) for geometry arms
    def sources(self, e: str, before: int | None) -> list[tuple[int, str]]:
        s = self.c.sess
        out: dict[int, str] = {}
        if self.sk != "none":
            for o in s.lookup(e, before=before, exclude=set(self.slots)):
                out.setdefault(o, "skeleton")
        if self.graph:
            for r in list(self.slots):
                for o in s.neighbors(r, kinds=("ent", "ref", "x"), mentioning=e):
                    if o not in self.slots and (before is None or o < before):
                        t = next(t for t in ("ent", "ref", "x") if any(d == o for d, _ in s.edges[t].get(r, [])))
                        out.setdefault(o, f"edge:{t}")
        cands = [(o, v) for o, v in out.items() if (e, o) not in self.tried]
        return sorted(cands, key=lambda ov: -self.rank_score(e, ov[0]))[: self.fault_top]

    def rank_score(self, query: str, o: int) -> float:
        s = self.c.sess
        if self.descent is None:                              # pure fold: most recent first (Algorithm 3 order)
            return float(o)
        return s.relevance(query, o) - self.lam * s.distance(self.q_origin, o, self.arm, self.fitted)

    def fault_fixpoint(self, before: int | None):
        pending = [(e, r, v) for e in self.R if e not in self.covered() for r, v in self.sources(e, before)]
        while pending and self.n_faults < self.max_faults and not self.c.over_budget():
            e, r, via = pending.pop(0)
            if (e, r) in self.tried or e in self.covered():
                continue
            self.tried.add((e, r))
            self.n_faults += 1
            self.rec["faults"].append({"origin_id": r, "entity": e, "via": via})
            self.c.log("fault", origin_id=r, reason=f"late_bound:{e}", via=via, thread_id=self.c.sess.thread_of[r])
            before_R = set(self.R)
            self.page_in(r, via=via)
            self.read([r], focus=[e])
            new = [x for x in self.R if x not in before_R]
            self.evict(r)
            if new and self.slots:                            # the current origins are still resident
                self.read(list(self.slots), focus=new)
            pending += [(e2, r2, v2) for e2 in self.R if e2 not in self.covered() for r2, v2 in self.sources(e2, before)
                        if (e2, r2) not in self.tried]

    # ---- the run
    def initial_candidates(self) -> list[int]:
        s, q = self.c.sess, self.c.item["question"]
        if self.descent is None:
            return []
        q_emb = self.c.opts["_q_emb"]
        if self.descent == "shells":
            cands, n_scores = s.descend(q, q_emb, beam=int(self.c.opts.get("beam", 2)))
        else:
            cands, n_scores = s.hnsw_search(q, q_emb, top=int(self.c.opts.get("hnsw_top", 16)))
        if self.graph:
            extra = []
            for o in cands:
                extra += s.neighbors(o, kinds=("ref", "ent", "x"))
            cands = list(dict.fromkeys(cands + extra[:8]))
        self.rec["descend_candidates"] = list(cands)
        self.rec["routing_scores"] = n_scores
        self.c.log("descend", descend={"beam": self.c.opts.get("beam", 2), "scores": n_scores, "candidates": cands[:32]},
                   distance={"arm": self.arm, "lambda": self.lam,
                             "d": {str(o): round(s.distance(self.q_origin, o, self.arm, self.fitted), 4) for o in cands[:32]}})
        ranked = sorted(cands, key=lambda o: -self.rank_score(q, o))
        return ranked[: self.M]

    def run(self) -> dict:
        s = self.c.sess
        for e in self.roots:
            self.add_relevant(e)
        order, done, found = [], set(), False
        first = self.initial_candidates()
        if first:
            order.append(first)
        rest = [o for o in range(s.N) if o not in first]
        order += [rest[i:i + self.window] for i in range(0, len(rest), self.window)]
        for win in order:
            if self.c.over_budget():
                self.c.log("budget_stop", tokens_in=self.c.usage.tokens_in)
                self.rec["status"] = "budget"
                self.rec["stop_reason"] = "budget"
                break
            for o in win:
                self.page_in(o)
            seen_R = set(self.R)
            found = self.read(win)
            for _ in range(3):                                   # newly relevant entities: select again in resident origins
                new = [e for e in self.R if e not in seen_R and e not in self.covered()]
                if found or not new:
                    break
                seen_R = set(self.R)
                found = self.read(win, focus=new)
            done |= set(win)
            self.fault_fixpoint(before=None if self.descent else min(win))
            reason = self.stop_reason(found)
            if reason:
                self.c.log("early_stop", after_origin=max(win), reason=reason, coverage=self.stop_coverage)
                self.rec["stop_reason"] = reason
                break
            for o in win:
                self.evict(o)
        else:
            self.rec.setdefault("stop_reason", "exhausted")
        evidence = "\n".join(f'[{p.pin_id}] (origin {p.origin_id}) "{p.quote}"' for p in self.live()) or "(no evidence)"
        pred, cited = answer_from(self.c, evidence)
        self.rec.update(prediction=pred, cited_pin_ids=cited,
                        pins=[{"pin_id": p.pin_id, "origin_id": p.origin_id, "char_span": p.char_span, "quote": p.quote,
                               "entity": p.entity, "evicted": p.evicted, "note": p.note} for p in self.pins])
        self._gold_diagnostics()
        return self.rec

    def stop_reason(self, found: bool) -> str | None:
        """Early stop (decisions log 2026-11-06, design ruling 2026-11-07): only descent-first systems (S12-S18, S12F) may
        stop before the archive is exhausted, and only when the question's chain is structurally pinned (chain_pinned:
        the walk from the question entity along the pins' reveals ends in leaves whose own pins hold a value:
        `early_chain_pinned`). The model's answer_found claim never ends reading, for any system (Qwen3-4B claims it
        after the first hop of a multi-hop chain); opts early_stop_on_answer_found=true restores the old
        `early_answer_found` stop for descent-first systems. Pure folds (S6-S11) fold-left over every origin."""
        if self.descent is None:
            return None
        if found and self.c.opts.get("early_stop_on_answer_found", False):
            return "early_answer_found"
        ok, self.stop_coverage = self.chain_pinned()
        return "early_chain_pinned" if ok else None

    stop_coverage: dict | None = None

    # ---- the structural chain-pinned condition (descent-first systems only)
    def covering_pins(self, e: str) -> list[tuple[Pin, str]]:
        """The live pins that cover entity `e` (a normalized key), label matches first. A pin covers `e` when (a) its
        entity label is `e` (labels compare after stripping legal forms only: ltd, inc, gmbh, corp, llc, plc, ag, sa,
        co, bv) or (b) its quote contains the full normalized name as whole words; in both cases a pin that lists `e`
        among the entities it reveals (entity_reveals: values are not entities) never covers it (it names the entity
        without having read about it: the pin that reveals hop j+1 is about hop j). Product suffixes (pumps, works,
        systems, ...) stay part of the name and a bare core name is not a match, so "Kelvane Works" never covers
        "Kelvane Pumps" and one-character twins are different entities."""
        ek = legal_core(e)
        label, quote = [], []
        for p in self.live():
            if ek in {legal_core(r) for r in entity_reveals(p.reveals)}:
                continue
            if legal_core(p.entity) == ek:
                label.append((p, "label"))
            elif mentions(p.quote, e):
                quote.append((p, "quote"))
        return label + quote

    def chain_coverage(self) -> dict:
        """For every relevant entity, the first live pin that covers it (covering_pins) or None."""
        out = {}
        for e in self.R:
            cov = self.covering_pins(e)
            out[e] = {"pin_id": cov[0][0].pin_id, "by": cov[0][1]} if cov else None
        return out

    def chain_walk(self) -> dict:
        """Walk the question's chain as the pins describe it (Algorithm 2's late binding, read off the evidence): start
        at the question entities (self.roots: every entity question_entities finds in the question text must be walked
        and covered, a component noun included: "O-rings" is capitalized in MT questions, so those walks have the roots
        ["o-rings", "<question entity>"], and the chain's facts, which all name the component, cover it by quote); from
        every reached entity follow the entities its covering pins reveal
        (entity_reveals: a revealed value such as "37 bar" or "7700" is the answer, not a hop; an entity revealed by a
        covering pin is the next hop; a covering pin never reveals the entity it covers); an entity whose covering
        pins reveal no entity is a leaf. Entities are the same when their names agree after stripping legal forms; a
        reveal of an entity already reached is an edge, not a new hop, so cycles end. Entities that no walked pin
        reveals (a distractor's chain, a twin revealed by a distractor) are never reached, whatever R holds. Returns
        the walk: roots, path (entities in walk order), edges [from, to, pin_id], covered (entity -> first covering pin
        or None), leaves (leaf -> the pin id of a covering pin holding a value, or None) and hops (entities reached
        through a reveal)."""
        names: dict[str, str] = {}

        def node(x: str) -> str:
            k = legal_core(x)
            if k:
                names.setdefault(k, normalize_key(x))                 # the first name form seen is the one matched
            return k

        roots = list(dict.fromkeys(k for k in (node(e) for e in self.roots) if k))
        queue, seen = list(roots), set(roots)
        path, edges, covered, leaves = [], [], {}, {}
        while queue:
            k = queue.pop(0)
            e = names[k]
            path.append(e)
            cov = self.covering_pins(e)
            covered[e] = {"pin_id": cov[0][0].pin_id, "by": cov[0][1]} if cov else None
            if not cov:
                continue
            onward = False
            for p, _ in cov:
                for r in entity_reveals(p.reveals):                   # never e itself: such a pin would not cover e
                    rk = node(r)
                    if not rk:
                        continue
                    onward = True
                    edge = [e, names[rk], p.pin_id]
                    if edge not in edges:
                        edges.append(edge)
                    if rk not in seen:
                        seen.add(rk)
                        queue.append(rk)
            if not onward:
                vp = next((p for p, _ in cov if RX_VALUE.search(p.quote)), None)
                leaves[e] = vp.pin_id if vp else None
        return {"roots": [names[k] for k in roots], "path": path, "edges": edges, "covered": covered, "leaves": leaves,
                "hops": len(path) - len(roots)}

    def chain_pinned(self) -> tuple[bool, dict]:
        """The question's chain is structurally pinned when the walk (chain_walk) from the question entities (1) took at
        least one hop (LB-MHQA / MT-LB-MHQA questions have h >= 2: a question entity with a value of its own is not a
        chain, so a single-hop question never stops early), (2) reached only covered entities (every question entity
        included), and (3) ends in at least one leaf, and every leaf has a covering pin whose quote holds a value
        (RX_VALUE); a leaf's covering pins reveal no entity, a revealed value does not count as one. A value anywhere
        else (in the pin that merely reveals the next entity, in a distractor's pin, in an entity revealed by a
        distractor) does not count, and an entity whose covering pins still reveal an entity is not the end of the
        chain. Identical for every descent-first system. Returns (verdict, details); details carry the walked path for
        the early_stop event."""
        w = self.chain_walk()
        leaves = w["leaves"]
        leaf = next(iter(leaves), None)
        value = bool(leaves) and all(v is not None for v in leaves.values())
        ok = w["hops"] >= 1 and all(v is not None for v in w["covered"].values()) and value
        return ok, {**w, "leaf": leaf, "leaf_pin": leaves.get(leaf), "value_pinned": value}

    def _gold_diagnostics(self):
        """Skeletons of the gold fact origins and graph reachability (gamma_s, gamma_g inputs for the scorer).
        Uses gold only after the run, for measurement."""
        s, g = self.c.sess, self.c.item.get("gold", {})
        self.rec["skeleton"] = {str(f["origin_id"]): sorted(s.skeleton.get(f["origin_id"], ())) for f in g.get("facts", [])}
        if not self.graph:
            return
        reach = {}
        for f in g.get("facts", []):
            o, e = f["origin_id"], normalize_key(f.get("subject", ""))
            resident = self.became_relevant.get(e, set())
            reach[str(o)] = any(o in s.neighbors(r, kinds=("seq", "ent", "ref", "x")) for r in resident)
        self.rec["edges_from_resident"] = reach


# ---------------------------------------------------------------- baselines


def hybrid_rank(c: Ctx, query: str) -> list[int]:
    s = c.sess
    bm = np.array(s.bm25.scores(query))
    q_emb = c.opts["_q_emb"] if query == c.item["question"] else c.opts["_embedder"].embed_query([query])[0]
    cos = s.emb @ q_emb
    norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
    return [int(i) for i in np.argsort(-(norm(bm) + norm(cos)), kind="stable")]


def run_s1(c: Ctx) -> dict:
    rec = base_record(c, "S1")
    total = sum(int(o.get("n_tokens", len(o["text"]) // 4)) for o in c.item["origins"])
    if total > int(c.opts.get("context_limit", CONTEXT_LIMIT)):
        rec.update(status="does_not_fit", prediction="")
        c.log("skip", reason=f"context {total} > limit")
        return rec
    ids = list(range(c.sess.N))
    rec["paged_in"] = ids
    pred, cited = answer_from(c, fmt_origins(c.sess, ids))
    rec.update(prediction=pred, cited_pin_ids=cited)
    return rec


def run_s2(c: Ctx) -> dict:
    rec = base_record(c, "S2")
    k = int(c.opts.get("k", 8))
    top = sorted(hybrid_rank(c, c.item["question"])[:k])
    rec["paged_in"] = top
    c.log("retrieve", candidates=top)
    pred, cited = answer_from(c, fmt_origins(c.sess, top))
    rec.update(prediction=pred, cited_pin_ids=cited)
    return rec


def ircot_step(c: Ctx, user: str, max_tokens: int = 200) -> dict:
    """One IRCoT reply as a dict. Invalid JSON with a readable top-level "answer" is used as that answer (same salvage as
    answer_from, code `answer_salvaged`); otherwise one retry with the reminder, salvaged the same way."""
    text = c.chat.ask(IRCOT_SYSTEM, user, max_tokens, c.usage, c.log_fn)
    out = parse_json(text)
    if out is not None:
        return out
    v = salvage_answer(text)
    if v is not MISSING:
        c.answer_failure = "answer_salvaged"
        return {"answer": v}
    text2 = c.chat.ask(IRCOT_SYSTEM, user + REASK, max_tokens, c.usage, c.log_fn)
    out = parse_json(text2)
    if out is not None:
        return out
    v = salvage_answer(text2)
    if v is not MISSING:
        c.answer_failure = "answer_salvaged_retry"
        return {"answer": v}
    c.log_fn({"item_id": c.item["item_id"], "action": "parse_error", "reply_head": text2[:200]})
    return {}


def run_s3(c: Ctx) -> dict:
    rec = base_record(c, "S3")
    k, steps = int(c.opts.get("k_step", 4)), int(c.opts.get("ircot_steps", 6))
    got, thoughts, pred = [], [], None
    for _ in range(steps):
        # no trailing space when there are no thoughts: the query is then the question itself (precomputed vector)
        query = (c.item["question"] + " " + thoughts[-1]).strip() if thoughts and thoughts[-1] else c.item["question"]
        for o in [o for o in hybrid_rank(c, query) if o not in got][:k]:
            got.append(o)
        c.log("retrieve", candidates=got[-k:])
        out = ircot_step(c, f"Question: {c.item['question']}\n\nParagraphs:\n{fmt_origins(c.sess, sorted(got))}\n\n"
                            f"Reasoning so far: {' '.join(thoughts) or '(none)'}")
        if answer_text(out.get("answer")).strip():
            pred = answer_text(out["answer"])
            break
        thoughts.append(str(out.get("thought", "")))
        if c.over_budget():
            break
    rec["paged_in"] = got
    if pred is None:
        pred, _ = answer_from(c, fmt_origins(c.sess, sorted(got)))
    rec["prediction"] = pred
    return rec


def gist_page(c: Ctx, text: str) -> tuple[str, dict]:
    """One page -> (gist, flags). flags: n_calls, truncated (any request ended with finish_reason length), parse_error
    (a reply was not valid JSON as a whole), empty (the final gist is ""). A cut-off or malformed reply is salvaged
    (the gist text read so far); only a reply with nothing readable is re-asked once, then the gist is ""."""
    fl = {"n_calls": 0, "truncated": False, "parse_error": False, "empty": False}

    def tap(ev):
        if ev.get("action") == "llm_call" and ev.get("finish_reason") == "length":
            fl["truncated"] = True
        c.log_fn(ev)

    gist = ""
    for attempt in range(2):
        reply = c.chat.ask(GIST_SYSTEM, text + (REASK if attempt else ""), S4_GIST_MAX_TOKENS, c.usage, tap)
        fl["n_calls"] += 1
        obj = parse_json(reply)
        if obj is not None and "gist" in obj:
            gist = answer_text(obj["gist"])
            break
        fl["parse_error"] = True
        v = salvage_string_field(reply, "gist")
        if v is not MISSING:
            gist = v[0]
            break
    fl["empty"] = not gist.strip()
    return gist, fl


def gist_harness_failure(item: dict, record: dict) -> bool:
    """True when a page that holds a gold fact has an empty gist because of truncation or a parse failure (a harness
    failure, not the model's choice). Computed from the record and the item after the run; gold never reaches the system."""
    fl, pages = record.get("gist_pages"), record.get("pages")
    if not fl or not pages:
        return False
    fact_origins = {int(o) for o in item.get("gold", {}).get("fact_origins", [])}
    fact_origins |= {int(f["origin_id"]) for f in item.get("gold", {}).get("facts", []) if "origin_id" in f}
    return any(f["empty"] and (f["truncated"] or f["parse_error"]) and fact_origins & set(pg) for f, pg in zip(fl, pages))


def s4_harness_failure(item: dict, record: dict) -> bool:
    """S4 harness failure: a gist failure on a page holding a gold fact (gist_harness_failure) or an unreadable lookup
    reply (lookup_parse_error: not the same as choosing zero pages). Gold is used here, after the run, only."""
    return bool(record.get("lookup_parse_error")) or gist_harness_failure(item, record)


def run_s4(c: Ctx) -> dict:
    rec = base_record(c, "S4")
    size, L = int(c.opts.get("page_size", 1)), int(c.opts.get("lookup_pages", 5))
    pages = [list(range(i, min(i + size, c.sess.N))) for i in range(0, c.sess.N, size)]
    gists, flags = [], []
    for i, pg in enumerate(pages):
        g, fl = gist_page(c, "\n".join(c.sess.texts[o] for o in pg))
        gists.append(g)
        flags.append(fl)
        c.log("gist", page=i, origins=pg, empty=fl["empty"], truncated=fl["truncated"], parse_error=fl["parse_error"])
    listing = "\n".join(f"<page {i}>\n{g}\n</page>" for i, g in enumerate(gists))
    out = ask_json(c.chat, LOOKUP_SYSTEM.format(L=L), f"Question: {c.item['question']}\n\nGists:\n{listing}",
                   S4_LOOKUP_MAX_TOKENS, c.usage, c.log_fn)
    lookup_parse_error = not isinstance(out.get("pages"), list)      # {} after the retry = unreadable, not "no pages"
    chosen = [int(p) for p in (out.get("pages") or []) if str(p).isdigit() and int(p) < len(pages)][:L]
    ids = sorted(o for p in chosen for o in pages[p])
    rec["paged_in"] = ids
    c.log("lookup", pages=chosen)
    pred, cited = answer_from(c, f"Gists:\n{listing}\n\nFull pages:\n{fmt_origins(c.sess, ids)}")
    rec.update(prediction=pred, cited_pin_ids=cited, pages=pages, lookup_pages=chosen, gist_pages=flags,
               n_gist_calls=sum(f["n_calls"] for f in flags), n_gist_truncated=sum(f["truncated"] for f in flags),
               n_gist_empty=sum(f["empty"] for f in flags), n_gist_parse_errors=sum(f["parse_error"] for f in flags),
               lookup_empty=not chosen, lookup_parse_error=lookup_parse_error, config_version=S4_CONFIG_VERSION, tuning_iteration=S4_TUNING_ITERATION)
    return rec


def run_s5(c: Ctx) -> dict:
    rec = base_record(c, "S5")
    size, notes = int(c.opts.get("coa_chunk", 16)), "(none)"
    for i in range(0, c.sess.N, size):
        ids = list(range(i, min(i + size, c.sess.N)))
        out = ask_json(c.chat, COA_SYSTEM, f"Question: {c.item['question']}\n\nNotes from the previous worker:\n{notes}\n\n"
                       f"Your part:\n{fmt_origins(c.sess, ids)}", 400, c.usage, c.log_fn)
        notes = str(out.get("notes", notes))
        c.log("worker", origins=[ids[0], ids[-1]])
        if c.over_budget():
            rec["status"] = "budget"
            break
    rec["paged_in"] = list(range(c.sess.N))
    pred, cited = answer_from(c, f"Notes:\n{notes}")
    rec.update(prediction=pred, cited_pin_ids=cited)
    return rec


def _kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), size=k, replace=False)]
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        lab = np.argmax(x @ cent.T, axis=1)
        for j in range(k):
            if np.any(lab == j):
                v = x[lab == j].mean(0)
                cent[j] = v / (np.linalg.norm(v) or 1)
    return lab


def run_s13(c: Ctx) -> dict:
    rec = base_record(c, "S13")
    s, emb = c.sess, c.sess.emb
    nodes = [("leaf", o, s.texts[o], emb[o]) for o in range(s.N)]
    level = list(range(s.N))
    level_texts, level_emb = [s.texts[o] for o in level], emb
    for depth in range(2):
        k = max(1, math.ceil(len(level_texts) / 8))
        if k <= 1 and depth > 0:
            break
        lab = _kmeans(np.asarray(level_emb), k, seed=depth)
        summaries = []
        for j in range(k):
            members = [t for t, l in zip(level_texts, lab) if l == j]
            if not members:
                continue
            out = ask_json(c.chat, SUMMARY_SYSTEM, "\n\n".join(members)[:24000], 220, c.usage, c.log_fn)
            summaries.append(str(out.get("summary", "")))
        sem = c.opts["_embedder"].embed(summaries)
        nodes += [("summary", depth, t, e) for t, e in zip(summaries, sem)]
        level_texts, level_emb = summaries, sem
    q = c.opts["_q_emb"]
    ranked = sorted(nodes, key=lambda n: -float(n[3] @ q))
    budget, used, chosen = int(c.opts.get("raptor_tokens", 8000)), 0, []
    for n in ranked:
        t = len(n[2]) // 4
        if used + t > budget:
            continue
        chosen.append(n)
        used += t
    rec["paged_in"] = sorted(n[1] for n in chosen if n[0] == "leaf")
    ev = "\n\n".join(f"[{n[0]}] {n[2]}" for n in chosen)
    pred, cited = answer_from(c, ev)
    rec.update(prediction=pred, cited_pin_ids=cited)
    return rec


BASELINE_FN = {"S1": run_s1, "S2": run_s2, "S3": run_s3, "S4": run_s4, "S5": run_s5, "S13": run_s13}


def run_item(system: str, c: Ctx) -> dict:
    t0 = time.time()
    try:
        rec = Fold(c, system).run() if system in SYSTEMS else BASELINE_FN[system](c)
    except Exception as e:                                       # one failed item never kills the run
        import traceback
        rec = base_record(c, system)
        rec.update(status="error", error=traceback.format_exc()[-2000:])
        if type(e).__name__ == "BudgetExceeded":
            rec["status"] = "budget"
        elif isinstance(e, ContextOverflow):                     # prompt exceeded the served window: reported, not an error
            rec.update(status="does_not_fit", prediction="")
            c.log("skip", reason="context overflow (vLLM 400)")
    if c.answer_failure:
        rec["answer_parse"] = c.answer_failure                  # answer_salvaged / _salvaged_retry / _retried / _unparseable / _empty
    rec.update(tokens_in=c.usage.tokens_in, tokens_out=c.usage.tokens_out, n_llm_calls=c.usage.calls,
               wall_ms=int(1000 * (time.time() - t0)), oracle=system in ORACLE_SYSTEMS)
    return rec
