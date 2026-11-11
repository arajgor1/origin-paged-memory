"""Scoring for Track A (protocol Section 3.6) and Track A2 drift (protocol Section 4.3).

Inputs are the item dicts and per-item result records documented in ``opa/data/SCHEMA.md``. Everything here is a
pure function of (item, record): no I/O, no randomness, so a score can always be recomputed from the stored run.

Answer normalization (``normalize_answer``)
-------------------------------------------
SQuAD-style ("standard answer normalization", protocol 3.6) with three documented extensions, applied in this order:

1. Unicode NFKC, lower case, unicode dashes / minus signs mapped to ``-``.
2. Temperatures: ``83 °C``, ``83°C``, ``83 C``, ``83 degC``, ``83 deg C``, ``83 degrees C``, ``83 degrees Celsius``
   all become ``83 degc`` (same for Fahrenheit -> ``degf``). Only when the unit directly follows a number.
3. Numbers that stand alone (not inside an alphanumeric token such as ``QX-2291b`` or ``v2.10``): thousands
   separators are dropped (``1,000`` -> ``1000``) and trailing fractional zeros are removed (``83.0`` -> ``83``,
   ``1.50`` -> ``1.5``). Leading zeros are kept (``007`` stays ``007``) so part numbers do not collide.
4. Punctuation (unicode category P* and ASCII ``string.punctuation``, plus ``°``) is deleted, as in SQuAD, except a
   decimal point between digits and a minus sign directly in front of a stand-alone number. ``±`` is kept.
5. Articles ``a``, ``an``, ``the`` are removed and whitespace is collapsed.

Exact match is 1 if the normalized prediction equals the normalized ``answer`` or any normalized
``answer_aliases`` entry. Token F1 is the SQuAD token-overlap F1, maximized over the same answer set.

Evidence coverage
-----------------
A pin *covers* a gold fact if (a) it is from the fact's origin and its ``char_span`` overlaps at least 50% of the
fact's ``char_span``, or (b) its normalized ``quote`` contains the normalized fact text (route (b) does not require
the origin to match: the harness only accepts quotes that are exact substrings of the named origin, so a quote
holding the fact text verbatim holds the evidence). A *live* pin is one with ``evicted`` false.

- evidence precision = share of the record's pins that cover at least one gold fact (NaN if there are no pins);
- evidence recall = share of gold facts covered by at least one pin (evicted or not).
``cited_precision`` / ``cited_recall`` are the same quantities restricted to ``cited_pin_ids``.

Failure codes (protocol 3.6): exactly one per wrong answer, decided by this precedence
-----------------------------------------------------------------------------------
0. Correct answer (EM = 1): no code. Record with ``status != "ok"``: no code, counted as an error instead.
1. **F8 wrong thread** - the prediction exact-matches a confuser value (and not the gold answer).
2. Otherwise take the **first gold hop, in hop order, whose fact no live pin covers**, and classify it:
   a. **F6 evicted by budget** - a pin covering the fact existed but every such pin was evicted.
   b. The fact origin was **never paged in** (not in ``record.paged_in``); the most specific of, in this order:
      - **F9 descent wrong turn** - the system used descend (``descend_candidates`` non-empty, or
        ``uses_descend=True``) and the fact origin is not among ``descend_candidates``;
      - **F5 alias mismatch** and **F4 skeleton miss** (only for systems with a skeleton: ``record.skeleton``
        non-empty, or ``has_skeleton=True``). Let K be the skeleton keys of the fact origin, kc the normalized key of
        the hop's subject entity, and A the keys of its aliases (``gold.aliases``). If kc is in K, lookup by name
        would hit and neither code applies. Otherwise it is F5 if an alias is involved - the fact text names the
        entity by an alias, or K holds one of the entity's alias keys - and F4 otherwise. (So: fact uses alias and
        the skeleton lacks the key the agent looks up -> F5; fact uses the canonical form but the skeleton only
        holds an alias form -> F5, the "vice versa" case; no alias anywhere and kc missing -> F4.)
      - **F10 needed edge missing** - the system has a graph (``edges_from_resident`` non-empty, or
        ``has_graph=True``) and ``edges_from_resident`` for the fact origin is false or absent;
      - **F1 never paged in** otherwise.
   c. Paged in, but **no pin (live or evicted) from the fact origin**: **F2 paged but not pinned**.
   d. Paged in and some pin from the fact origin exists, but none covers the fact: **F3 pinned but misread**.
3. Every gold fact is covered by a live pin: **F7 reasoning error**.

Coverage factors (protocol 3.6 plot 3)
--------------------------------------
For the late-bound hops of an item (``gold.facts[j].late_bound``):
gamma_s = share whose subject's normalized key is in its fact origin's skeleton (0 for systems without a skeleton);
gamma_g = share whose fact origin has ``edges_from_resident`` true (0 for systems without a graph);
gamma_eff = 1 - (1 - gamma_s)(1 - gamma_g); predicted success = gamma_eff ** h_late (1.0 when h_late = 0, where
gamma_s and gamma_g are NaN).

Faults: a fault in ``record.faults`` is *necessary* if its origin holds a gold fact, else *spurious*.

Drift records (Track A2, protocol 4.3)
--------------------------------------
One record per (condition, session)::

    {"item_id": "drift-dev-000003", "condition": "D4", "run_id": "...",
     "probes": [{"turn": 10, "predicted": {"max_coolant_temp": "80 °C", "...": "..."}},
                {"turn": 25, "predicted": {...}}, {"turn": 50, "predicted": {...}}],
     "status": "ok"}

``predicted`` maps each constraint key to the value string the system gave (missing key, None or "" = no answer).
A predicted value *matches* a value v if its normalized form equals normalize(v) or normalize(v + " " + unit),
ignoring a ``±`` sign on either side (tolerances such as ``±0.16 mm`` vs ``0.16 mm``).
Per probe and constraint the outcome is exactly one of: ``correct`` (matches the current value), ``stale`` (matches
a value that was current earlier and was superseded by a turn <= the probe), ``near_miss`` (matches a near-miss
value stated at a turn <= the probe), ``missing`` (no answer), ``fabricated`` (matches no value of any constraint
stated at a turn <= the probe and does not occur as a token sequence in any turn text up to the probe), or
``other`` (appeared in the session but is none of the above, e.g. another constraint's value).
"""
from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

FAILURE_CODES = [f"F{i}" for i in range(1, 11)]
FAILURE_CODE_NAMES = {
    "F1": "fact never paged in",
    "F2": "paged but not pinned",
    "F3": "pinned but misread",
    "F4": "skeleton miss",
    "F5": "alias mismatch",
    "F6": "evicted by budget",
    "F7": "reasoning error with all facts pinned",
    "F8": "wrong thread (confuser value)",
    "F9": "descent took a wrong turn",
    "F10": "needed edge missing",
}

# ------------------------------------------------------------------ normalization
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹣－"), "-")
_DEG = r"(?:°|º|˚)"
_TEMP_C = re.compile(rf"(\d)\s*{_DEG}?\s*(?:degrees?|deg\.?)?\s*(?:c|celsius|centigrade)\b")
_TEMP_F = re.compile(rf"(\d)\s*{_DEG}?\s*(?:degrees?|deg\.?)?\s*(?:f|fahrenheit)\b")
_THOUSANDS = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+)(?![\w,]|\.\d)")
_NUMBER = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w]|\.\d)")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_NEG_SIGN = re.compile(r"(?<![\w.])-(?=\d)")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"°", "º", "˚"}
_KEEP_DOT, _KEEP_MINUS = "\x00", "\x01"


def _is_punct(ch: str) -> bool:
    return ch in _PUNCT or unicodedata.category(ch).startswith("P")


def _canon_decimal(m: re.Match) -> str:
    frac = m.group(2).rstrip("0")
    return m.group(1) + ("." + frac if frac else "")


def normalize_answer(s: Any) -> str:
    """Normalize an answer string (see module docstring). ``None`` normalizes to ""."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).translate(_DASHES).lower()
    s = _TEMP_C.sub(r"\1 degc", s)
    s = _TEMP_F.sub(r"\1 degf", s)
    s = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), s)
    s = _NUMBER.sub(_canon_decimal, s)
    s = _DECIMAL_POINT.sub(_KEEP_DOT, s)
    s = _NEG_SIGN.sub(_KEEP_MINUS, s)
    s = "".join(ch for ch in s if not _is_punct(ch))
    s = s.replace(_KEEP_DOT, ".").replace(_KEEP_MINUS, "-")
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def entity_key(name: Any) -> str:
    """Normalized entity key used to compare against ``record.skeleton`` (protocol 3.1: lower case, punctuation
    stripped). Same pipeline as ``normalize_answer`` but articles are kept (they can be part of a name)."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).translate(_DASHES).lower()
    s = "".join(ch for ch in s if not _is_punct(ch))
    return " ".join(s.split())


def _answer_list(answers: Any) -> list[str]:
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [a for a in answers if a is not None]


def gold_answers(item: dict) -> list[str]:
    """``answer`` plus ``answer_aliases`` (order kept, duplicates dropped)."""
    out: list[str] = []
    for a in _answer_list(item.get("answer")) + _answer_list(item.get("answer_aliases")):
        if a not in out:
            out.append(a)
    return out


def exact_match(pred: Any, answers: Any) -> float:
    """1.0 if normalize(pred) equals the normalization of any accepted answer, else 0.0. An empty prediction never
    matches."""
    p = normalize_answer(pred)
    if not p:
        return 0.0
    return float(any(p == normalize_answer(a) for a in _answer_list(answers)))


def _token_f1(pred_toks: list[str], gold_toks: list[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pred_toks), same / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: Any, answers: Any) -> float:
    """SQuAD token F1, maximized over the accepted answers (0.0 if there are none)."""
    pt = normalize_answer(pred).split()
    golds = _answer_list(answers)
    return max((_token_f1(pt, normalize_answer(a).split()) for a in golds), default=0.0)


# ------------------------------------------------------------------ evidence
def _overlap(a: Iterable[int], b: Iterable[int]) -> int:
    a0, a1 = a
    b0, b1 = b
    return max(0, min(a1, b1) - max(a0, b0))


def pin_covers_fact(pin: dict, fact: dict) -> bool:
    """True if ``pin`` covers gold ``fact`` (module docstring, "Evidence coverage")."""
    span, fspan = pin.get("char_span"), fact.get("char_span")
    if span is not None and fspan is not None and pin.get("origin_id") == fact.get("origin_id"):
        flen = fspan[1] - fspan[0]
        if flen > 0 and _overlap(span, fspan) >= 0.5 * flen:
            return True
    ftext = normalize_answer(fact.get("text"))
    return bool(ftext) and ftext in normalize_answer(pin.get("quote"))


def _prec_rec(pins: list[dict], facts: list[dict]) -> tuple[float, float]:
    covers = [[pin_covers_fact(p, f) for f in facts] for p in pins]
    prec = sum(any(row) for row in covers) / len(pins) if pins else math.nan
    rec = sum(any(covers[i][j] for i in range(len(pins))) for j in range(len(facts))) / len(facts) if facts else math.nan
    return prec, rec


def evidence_scores(item: dict, record: dict) -> dict:
    facts = item.get("gold", {}).get("facts", [])
    pins = record.get("pins") or []
    cited_ids = set(record.get("cited_pin_ids") or [])
    p, r = _prec_rec(pins, facts)
    cp, cr = _prec_rec([x for x in pins if x.get("pin_id") in cited_ids], facts)
    return {"evidence_precision": p, "evidence_recall": r, "cited_precision": cp, "cited_recall": cr}


# ------------------------------------------------------------------ helpers on records
def _get_by_origin(d: dict | None, origin_id: Any, default=None):
    if not d:
        return default
    if origin_id in d:
        return d[origin_id]
    return d.get(str(origin_id), default)


def thread_confused(item: dict, record: dict) -> bool:
    """Prediction exact-matches a confuser's value or any of its value_aliases (same normalisation as EM, so "4300",
    "4300 h" and "22bar" count when they are aliases of a decoy's value) and not the gold answer."""
    pred = record.get("prediction")
    if exact_match(pred, gold_answers(item)):
        return False
    return any(exact_match(pred, c.get("value_aliases") or [c.get("value")])
               for c in item.get("gold", {}).get("confusers", []) or [])


def _fact_uses_alias(fact: dict, aliases: dict) -> bool:
    """True if the fact names its subject by an alias: from the generator's ``subject_surface`` field when present,
    else by searching the fact text for an alias of the subject."""
    subj = fact.get("subject")
    kc = entity_key(subj)
    if fact.get("subject_surface"):
        return entity_key(fact["subject_surface"]) != kc
    ftext = f" {entity_key(fact.get('text'))} "
    if kc:                                   # "ostrand ltd" contains the alias token "ostrand": mask it first
        ftext = ftext.replace(f" {kc} ", " | ")
    alias_keys = {entity_key(a) for a in (aliases.get(subj) or [])} - {kc, ""}
    return any(f" {k} " in ftext for k in alias_keys)


def _system_flags(record: dict, has_skeleton, has_graph, uses_descend) -> tuple[bool, bool, bool]:
    hs = bool(record.get("skeleton")) if has_skeleton is None else bool(has_skeleton)
    hg = bool(record.get("edges_from_resident")) if has_graph is None else bool(has_graph)
    ud = bool(record.get("descend_candidates")) if uses_descend is None else bool(uses_descend)
    return hs, hg, ud


def classify_failure(item: dict, record: dict, *, has_skeleton: bool | None = None, has_graph: bool | None = None,
                     uses_descend: bool | None = None) -> tuple[str, int | None]:
    """Failure code for a wrong answer and the hop it was decided on (None for F8 / F7). Precedence in the module
    docstring. Does not check correctness; ``score_item`` calls it only for wrong answers."""
    gold = item.get("gold", {})
    if thread_confused(item, record):
        return "F8", None
    hs, hg, ud = _system_flags(record, has_skeleton, has_graph, uses_descend)
    pins = record.get("pins") or []
    live = [p for p in pins if not p.get("evicted")]
    paged = set(record.get("paged_in") or [])
    paged |= {str(o) for o in paged}
    cands = set(record.get("descend_candidates") or [])
    aliases = gold.get("aliases") or {}
    for fact in sorted(gold.get("facts", []), key=lambda f: f.get("hop", 0)):
        if any(pin_covers_fact(p, fact) for p in live):
            continue
        o = fact.get("origin_id")
        hop = fact.get("hop")
        if any(pin_covers_fact(p, fact) for p in pins):
            return "F6", hop
        if o not in paged and str(o) not in paged:
            if ud and o not in cands and str(o) not in cands:
                return "F9", hop
            if hs:
                keys = {entity_key(k) for k in (_get_by_origin(record.get("skeleton"), o, []) or [])}
                kc = entity_key(fact.get("subject"))
                alias_keys = {entity_key(a) for a in (aliases.get(fact.get("subject")) or [])} - {kc, ""}
                if kc not in keys:
                    if _fact_uses_alias(fact, aliases) or (keys & alias_keys):
                        return "F5", hop
                    return "F4", hop
            if hg and not _get_by_origin(record.get("edges_from_resident"), o, False):
                return "F10", hop
            return "F1", hop
        if not any(p.get("origin_id") == o or str(p.get("origin_id")) == str(o) for p in pins):
            return "F2", hop
        return "F3", hop
    return "F7", None


def coverage_factors(item: dict, record: dict, *, has_skeleton: bool | None = None,
                     has_graph: bool | None = None) -> dict:
    """gamma_s, gamma_g, gamma_eff and the predicted success gamma_eff ** h_late (module docstring)."""
    gold = item.get("gold", {})
    hs, hg, _ = _system_flags(record, has_skeleton, has_graph, False)
    late = [f for f in gold.get("facts", []) if f.get("late_bound")]
    h_late = len(late)
    if h_late == 0:
        return {"h_late": 0, "gamma_s": math.nan, "gamma_g": math.nan, "gamma_eff": math.nan, "predicted": 1.0,
                "skeleton_miss": [], "graph_miss": []}
    s_hit, g_hit = [], []
    for f in late:
        o = f.get("origin_id")
        keys = {entity_key(k) for k in (_get_by_origin(record.get("skeleton"), o, []) or [])} if hs else set()
        s_hit.append(entity_key(f.get("subject")) in keys)
        g_hit.append(bool(_get_by_origin(record.get("edges_from_resident"), o, False)) if hg else False)
    gs, gg = sum(s_hit) / h_late, sum(g_hit) / h_late
    ge = 1.0 - (1.0 - gs) * (1.0 - gg)
    return {"h_late": h_late, "gamma_s": gs, "gamma_g": gg, "gamma_eff": ge, "predicted": ge ** h_late,
            "skeleton_miss": [not x for x in s_hit], "graph_miss": [not x for x in g_hit]}


def fault_counts(item: dict, record: dict) -> dict:
    gold_origins = {str(o) for o in item.get("gold", {}).get("fact_origins", [])}
    gold_origins |= {str(f.get("origin_id")) for f in item.get("gold", {}).get("facts", [])}
    nec = sum(1 for f in record.get("faults") or [] if str(f.get("origin_id")) in gold_origins)
    total = len(record.get("faults") or [])
    return {"n_faults": total, "necessary_faults": nec, "spurious_faults": total - nec}


def score_item(item: dict, record: dict, *, has_skeleton: bool | None = None, has_graph: bool | None = None,
               uses_descend: bool | None = None) -> dict:
    """Score one (item, record) pair. Pure function. Returns a flat dict with em, f1, evidence scores, failure
    code (None if correct or errored), thread confusion, coverage factors, fault counts and resource use."""
    pred = record.get("prediction")
    answers = gold_answers(item)
    em = exact_match(pred, answers)
    status = record.get("status", "ok") or "ok"
    out: dict[str, Any] = {
        "item_id": item.get("item_id"), "run_id": record.get("run_id"), "system": record.get("system"),
        "arm": record.get("arm"), "status": status, "error": status != "ok",
        "em": em, "f1": f1_score(pred, answers),
        "thread_confused": thread_confused(item, record),
        "has_confusers": bool(item.get("gold", {}).get("confusers")),
    }
    out.update(evidence_scores(item, record))
    code, hop = (None, None)
    if em < 1 and status == "ok":
        code, hop = classify_failure(item, record, has_skeleton=has_skeleton, has_graph=has_graph,
                                     uses_descend=uses_descend)
    out["failure_code"], out["failure_hop"] = code, hop
    cov = coverage_factors(item, record, has_skeleton=has_skeleton, has_graph=has_graph)
    out.update({k: cov[k] for k in ("h_late", "gamma_s", "gamma_g", "gamma_eff", "predicted")})
    out["skeleton_miss"], out["graph_miss"] = cov["skeleton_miss"], cov["graph_miss"]
    out.update(fault_counts(item, record))
    out.update({
        "tokens_in": float(record.get("tokens_in") or 0), "tokens_out": float(record.get("tokens_out") or 0),
        "page_ins": len(record.get("paged_in") or []), "cost_usd": float(record.get("api_usd") or 0.0),
        "wall_ms": float(record.get("wall_ms") or 0.0), "n_llm_calls": float(record.get("n_llm_calls") or 0),
    })
    return out


# ------------------------------------------------------------------ aggregation
def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def score_records(items: Iterable[dict], records: Iterable[dict], **flags) -> list[dict]:
    """Score every record against its item (matched by item_id). Raises KeyError for an unknown item_id."""
    by_id = {it["item_id"]: it for it in items}
    return [score_item(by_id[r["item_id"]], r, **flags) for r in records]


def aggregate(items: Iterable[dict], records: Iterable[dict], by: str | tuple[str, ...] = "system", **flags) -> dict:
    """Per-system (or per ``by`` key tuple) metrics.

    Means are over all records of the group (errored records count as wrong for EM/F1; NaN evidence values are
    skipped). thread_confusion_rate is over items that have confusers (NaN if none). spurious_fault_rate is
    spurious faults / all faults (NaN if no faults). failure_codes counts F1..F10 over wrong answers of ok records;
    errored records are counted in ``n_error`` / ``error_rate``."""
    rows = score_records(items, records, **flags)
    keys = (by,) if isinstance(by, str) else tuple(by)
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        g = r.get(keys[0]) if len(keys) == 1 else tuple(r.get(k) for k in keys)
        groups[g].append(r)
    out = {}
    for g, rs in groups.items():
        n = len(rs)
        conf = [float(r["thread_confused"]) for r in rs if r["has_confusers"]]
        nf = sum(r["n_faults"] for r in rs)
        hist = {c: 0 for c in FAILURE_CODES}
        for r in rs:
            if r["failure_code"]:
                hist[r["failure_code"]] += 1
        n_err = sum(r["error"] for r in rs)
        out[g] = {
            "n": n, "em": _nanmean([r["em"] for r in rs]), "f1": _nanmean([r["f1"] for r in rs]),
            "evidence_precision": _nanmean([r["evidence_precision"] for r in rs]),
            "evidence_recall": _nanmean([r["evidence_recall"] for r in rs]),
            "mean_tokens_in": _nanmean([r["tokens_in"] for r in rs]),
            "mean_tokens_out": _nanmean([r["tokens_out"] for r in rs]),
            "mean_page_ins": _nanmean([r["page_ins"] for r in rs]),
            "mean_necessary_faults": _nanmean([r["necessary_faults"] for r in rs]),
            "mean_spurious_faults": _nanmean([r["spurious_faults"] for r in rs]),
            "spurious_fault_rate": (sum(r["spurious_faults"] for r in rs) / nf) if nf else math.nan,
            "thread_confusion_rate": _nanmean(conf),
            "failure_codes": hist,
            "cost_usd": sum(r["cost_usd"] for r in rs), "mean_cost_usd": _nanmean([r["cost_usd"] for r in rs]),
            "mean_wall_ms": _nanmean([r["wall_ms"] for r in rs]),
            "n_error": n_err, "error_rate": n_err / n if n else math.nan,
        }
    return out


# ------------------------------------------------------------------ drift (Track A2)
def _strip_pm(s: str) -> str:
    return " ".join(s.replace("±", " ").split())


def _value_matches(pred_norm: str, value: Any, unit: str | None) -> bool:
    if not pred_norm or value is None:
        return False
    p = _strip_pm(pred_norm)
    if p == _strip_pm(normalize_answer(value)):
        return True
    return bool(unit) and p == _strip_pm(normalize_answer(f"{value} {unit}"))


def _value_history(c: dict, turn: int) -> list[str]:
    """Values of constraint c that were current at some turn <= ``turn``, in order; the last one is current."""
    hist = [(c["initial"].get("turn", 1), c["initial"]["value"])]
    hist += sorted(((u["turn"], u["value"]) for u in c.get("updates", []) if u["turn"] <= turn), key=lambda x: x[0])
    return [v for t, v in hist if t <= turn]


def drift_probe_outcomes(session: dict, record: dict, turn: int) -> dict[str, str]:
    """Outcome per constraint key at one probe turn: correct | stale | near_miss | missing | fabricated | other."""
    probe = next((p for p in record.get("probes", []) if p.get("turn") == turn), None)
    pred_map = (probe or {}).get("predicted") or {}
    texts = [f" {_strip_pm(normalize_answer(t.get('text')))} " for t in session.get("turns", [])
             if t.get("turn", 0) <= turn]
    all_vals = []
    for c in session["constraints"]:
        all_vals += [(v, c.get("unit")) for v in _value_history(c, turn)]
        all_vals += [(n["value"], c.get("unit")) for n in c.get("near_misses", []) if n.get("turn", 0) <= turn]
    out = {}
    for c in session["constraints"]:
        unit = c.get("unit")
        p = normalize_answer(pred_map.get(c["key"]))
        hist = _value_history(c, turn)
        if not p:
            out[c["key"]] = "missing"
        elif _value_matches(p, hist[-1], unit):
            out[c["key"]] = "correct"
        elif any(_value_matches(p, v, unit) for v in hist[:-1]):
            out[c["key"]] = "stale"
        elif any(_value_matches(p, n["value"], unit) for n in c.get("near_misses", []) if n.get("turn", 0) <= turn):
            out[c["key"]] = "near_miss"
        elif any(_value_matches(p, v, u) for v, u in all_vals) or any(f" {_strip_pm(p)} " in t for t in texts):
            out[c["key"]] = "other"
        else:
            out[c["key"]] = "fabricated"
    return out


def score_drift_session(session: dict, record: dict) -> list[dict]:
    """Per-probe rates for one session: accuracy (current value), stale, fabricated, near_miss, missing, other."""
    rows = []
    for probe in session.get("probes", []):
        t = probe["turn"]
        oc = drift_probe_outcomes(session, record, t)
        n = len(oc)
        cnt = Counter(oc.values())
        rows.append({"item_id": session.get("item_id"), "condition": record.get("condition"), "turn": t, "n": n,
                     "accuracy": cnt["correct"] / n if n else math.nan,
                     "stale_rate": cnt["stale"] / n if n else math.nan,
                     "fabricated_rate": cnt["fabricated"] / n if n else math.nan,
                     "near_miss_rate": cnt["near_miss"] / n if n else math.nan,
                     "missing_rate": cnt["missing"] / n if n else math.nan,
                     "other_rate": cnt["other"] / n if n else math.nan})
    return rows


def aggregate_drift(sessions: Iterable[dict], records: Iterable[dict]) -> dict:
    """{condition: {turn: {n_sessions, n_constraints, accuracy, stale_rate, fabricated_rate, ...}}}, pooling
    constraint outcomes over sessions (each session contributes its constraints equally)."""
    by_id = {s["item_id"]: s for s in sessions}
    pooled: dict[Any, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    nsess: dict[Any, Counter] = defaultdict(Counter)
    for r in records:
        s = by_id[r["item_id"]]
        for p in s.get("probes", []):
            oc = drift_probe_outcomes(s, r, p["turn"])
            pooled[r.get("condition")][p["turn"]].update(oc.values())
            nsess[r.get("condition")][p["turn"]] += 1
    out = {}
    for cond, turns in pooled.items():
        out[cond] = {}
        for t, cnt in sorted(turns.items()):
            n = sum(cnt.values())
            out[cond][t] = {"n_sessions": nsess[cond][t], "n_constraints": n,
                            **{f"{k}_rate" if k != "correct" else "accuracy": cnt[k] / n if n else math.nan
                               for k in ("correct", "stale", "fabricated", "near_miss", "missing", "other")}}
    return out


def drift_binary_outcomes(sessions: Iterable[dict], records: Iterable[dict], turn: int = 50) -> dict:
    """{(item_id, constraint_key): 1/0 correct at ``turn``} for one condition's records - the paired unit for a
    McNemar / bootstrap contrast between conditions (e.g. D4 vs D2 retention at turn 50)."""
    by_id = {s["item_id"]: s for s in sessions}
    out = {}
    for r in records:
        for k, v in drift_probe_outcomes(by_id[r["item_id"]], r, turn).items():
            out[(r["item_id"], k)] = int(v == "correct")
    return out
