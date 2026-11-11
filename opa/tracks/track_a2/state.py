"""Session state for Track A2 (protocol 3.1 item 10 and 4.2): OPA extractive registers with the exact-pin rule,
inner-shell nodes with abstractive digests (D6), and the string checks behind the per-rewrite loss epsilon (4.3).

Registers (D4/D5/D6)
--------------------
JSON with ``goal``, ``constraints`` and ``decisions``. Each constraint entry is a pin::

    {"pin_id": "p3", "key": "max_coolant_temp", "value": "80", "unit": "°C", "quote": "The maximum ...",
     "origin_id": 1, "thread": "design", "char_span": [120, 161], "superseded_by": null}

Every turn of the session is one origin; ``origin_id`` is the turn number. The model proposes operations and the
harness applies them:

- ``add``       new constraint entry;
- ``update``    new entry for a changed constraint; with ``supersede=True`` (D4, D6) the entry named by ``replaces``
                gets ``superseded_by`` = the new pin id; with ``supersede=False`` (D5) the new entry is added and the
                old one is left untouched (no link), which is the whole D5 ablation;
- ``decision``  an explicit decision (quote only);
- ``goal``      the session goal (quote only).

Exact-pin rule (protocol 3.1 item 8): an operation is rejected if its quote is not an exact substring of the turn
text (``track_a.text.find_exact``: whitespace runs count as one space, nothing else is normalized), or if the value it
registers is not an exact substring of that quote. The stored quote is the matched slice of the turn text, so every
register value is copied by reference from an origin.
"""
from __future__ import annotations

import json
import re
import unicodedata

from ..track_a.text import find_exact
from ... import scoring

CONSTRAINT_OPS = ("add", "update")
QUOTE_OPS = ("decision", "goal")


class Registers:
    def __init__(self, supersede: bool = True):
        self.supersede = supersede
        self.goal: dict | None = None
        self.constraints: list[dict] = []
        self.decisions: list[dict] = []
        self.n_pins = 0
        self.accepted = 0
        self.rejected: list[dict] = []

    # ------------------------------------------------------------ views
    def by_pin(self, pin_id) -> dict | None:
        return next((c for c in self.constraints if c["pin_id"] == str(pin_id)), None)

    def live(self) -> list[dict]:
        return [c for c in self.constraints if c["superseded_by"] is None]

    def all_pin_ids(self) -> set[str]:
        ids = {c["pin_id"] for c in self.constraints} | {d["pin_id"] for d in self.decisions}
        if self.goal:
            ids.add(self.goal["pin_id"])
        return ids

    def to_json(self) -> dict:
        return {"goal": self.goal, "constraints": self.constraints, "decisions": self.decisions}

    def render(self) -> str:
        """Prompt view (compact JSON; char spans omitted)."""
        cons = [{k: c[k] for k in ("pin_id", "key", "value", "unit", "quote", "origin_id", "thread", "superseded_by")}
                for c in self.constraints]
        dec = [{k: d[k] for k in ("pin_id", "quote", "origin_id")} for d in self.decisions]
        goal = {k: self.goal[k] for k in ("pin_id", "quote", "origin_id")} if self.goal else None
        return json.dumps({"goal": goal, "constraints": cons, "decisions": dec}, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------ updates
    def _pin(self, quote: str, text: str):
        span = find_exact(quote, text)
        if span is None:
            return None, None
        return span, text[span[0]:span[1]]

    def apply(self, ops, turn: int, thread: str, text: str) -> tuple[list[dict], list[dict]]:
        """Apply a list of operations proposed for one turn. Returns (accepted events, rejected events); each event
        is a dict ready for the event log."""
        acc, rej = [], []
        if not isinstance(ops, list):
            ops = []
        for op in ops:
            ev = self._apply_one(op, turn, thread, text)
            (acc if ev["ok"] else rej).append(ev)
        self.accepted += len(acc)
        self.rejected += [dict(r, turn=turn) for r in rej]
        return acc, rej

    def _reject(self, op, reason: str) -> dict:
        return {"ok": False, "reason": reason, "op": op}

    def _apply_one(self, op, turn: int, thread: str, text: str) -> dict:
        if not isinstance(op, dict):
            return self._reject(op, "not_an_object")
        kind = str(op.get("op", "")).lower()
        if kind not in CONSTRAINT_OPS + QUOTE_OPS:
            return self._reject(op, "unknown_op")
        quote = str(op.get("quote") or "")
        span, canon = self._pin(quote, text)
        if span is None:
            return self._reject(op, "quote_not_in_origin")
        if kind in QUOTE_OPS:
            self.n_pins += 1
            pin = {"pin_id": f"p{self.n_pins}", "quote": canon, "origin_id": turn, "thread": thread,
                   "char_span": list(span)}
            if kind == "goal":
                self.goal = pin
            else:
                self.decisions.append(pin)
            return {"ok": True, "op": kind, "pin_id": pin["pin_id"], "origin_id": turn, "char_span": list(span)}
        value = op.get("value")
        value = "" if value is None else str(value).strip()
        key = str(op.get("key") or "").strip()
        if not value:
            return self._reject(op, "empty_value")
        if not key:
            return self._reject(op, "empty_key")
        if find_exact(value, canon) is None:
            return self._reject(op, "value_not_in_quote")
        old = None
        replaces_via = None
        if kind == "update":
            old = self.by_pin(op.get("replaces")) if op.get("replaces") is not None else None
            if old is not None and old["superseded_by"] is None:
                replaces_via = "pin_id"
            else:
                same = [c for c in self.live() if c["key"] == key]
                if len(same) == 1:                    # the model named the key but not (or a wrong) pin id
                    old, replaces_via = same[0], "key"
                elif not self.supersede and same:     # D5 never links, so ambiguity does not matter
                    old, replaces_via = same[-1], "key"
                else:
                    return self._reject(op, "bad_replaces")
        dup = kind == "add" and any(c["key"] == key for c in self.live())
        self.n_pins += 1
        pid = f"p{self.n_pins}"
        self.constraints.append({"pin_id": pid, "key": key, "value": value, "unit": str(op.get("unit") or ""),
                                 "quote": canon, "origin_id": turn, "thread": thread, "char_span": list(span),
                                 "superseded_by": None})
        ev = {"ok": True, "op": kind, "pin_id": pid, "key": key, "value": value, "origin_id": turn,
              "char_span": list(span), "duplicate_key": dup}
        if old is not None:
            ev["replaces"], ev["replaces_via"] = old["pin_id"], replaces_via
            if self.supersede:
                old["superseded_by"] = pid
                ev["superseded"] = old["pin_id"]
            else:
                ev["superseded"] = None               # D5: link deliberately not recorded
        return ev


class Shells:
    """D6 inner-shell nodes: one node per ``b`` consecutive turns of a thread (b = 8, as the Track A shell index).
    A node gets its abstractive digest when it closes (its b-th child arrives); open nodes have no digest."""

    def __init__(self, b: int = 8):
        self.b = int(b)
        self.open: dict[str, list[int]] = {}
        self.count: dict[str, int] = {}
        self.nodes: list[dict] = []

    def add_turn(self, thread: str, turn: int) -> dict | None:
        """Add a turn; returns the node that just closed (digest still empty) or None."""
        kids = self.open.setdefault(thread, [])
        kids.append(turn)
        if len(kids) < self.b:
            return None
        k = self.count.get(thread, 0)
        self.count[thread] = k + 1
        node = {"node_id": f"n-{thread}-{k}", "thread": thread, "children": list(kids), "digest": ""}
        self.nodes.append(node)
        self.open[thread] = []
        return node

    def render(self) -> str:
        if not self.nodes:
            return "(no closed inner-shell nodes yet)"
        return "\n".join(f'<node id="{n["node_id"]}" thread="{n["thread"]}" turns="{n["children"][0]}-'
                         f'{n["children"][-1]}">\n{n["digest"]}\n</node>' for n in self.nodes)


# ---------------------------------------------------------------- epsilon (D2, protocol 4.3)
def _boundary_rx(v: str) -> re.Pattern:
    return re.compile(r"(?<![\w.])" + re.escape(v) + r"(?!\w|\.\d)")


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def value_present(value: str, text: str) -> bool:
    """Exact check: the gold value string occurs verbatim in the text as a stand-alone token sequence
    (``46`` matches "46 °C" but not "146" or "46.5"). Values are compared after Unicode NFKC only."""
    if not value:
        return False
    return _boundary_rx(_nfkc(str(value))).search(_nfkc(text)) is not None


def value_present_norm(value: str, unit: str, text: str) -> bool:
    """Lenient check with the scorer's normalization (``opa.scoring.normalize_answer``, ``±`` ignored): the
    normalized value occurs in the normalized text as a whole-token sequence. "83 °C" vs "83°C" match here."""
    if not value:
        return False
    t = f" {scoring._strip_pm(scoring.normalize_answer(text))} "
    v = scoring._strip_pm(scoring.normalize_answer(value))
    return bool(v) and f" {v} " in t


def presence(constraints: list[dict], turn: int, text: str) -> dict:
    """Which constraints' current gold values (after all turns <= ``turn``) are present in ``text``:
    {"exact": [keys], "norm": [keys]}."""
    from ...data.drift import current_values
    gold = current_values(constraints, turn)
    units = {c["key"]: c.get("unit", "") for c in constraints}
    return {"exact": [k for k, v in gold.items() if v is not None and value_present(v, text)],
            "norm": [k for k, v in gold.items() if v is not None and value_present_norm(v, units[k], text)]}


def epsilon_from_rewrites(constraints: list[dict], rewrites: list[dict], field: str = "exact") -> dict:
    """Per-rewrite loss for one D2 session. ``rewrites`` are in turn order, each {"turn", "present": {"exact": [...],
    "norm": [...]}} measured right after that rewrite.

    For rewrite t > first: opportunities = constraints present after the previous rewrite whose gold value did not
    change at turn t (an update turn is a write, not a chance to lose the old value); losses = those not present
    after rewrite t. epsilon = losses / opportunities ("lost or mutated": a value that changed form counts as lost).
    """
    upd_at = {(u["turn"], c["key"]) for c in constraints for u in c.get("updates", [])}
    opp = lost = 0
    prev = None
    for rw in rewrites:
        cur = set(rw["present"][field])
        if prev is not None:
            eligible = {k for k in prev if (rw["turn"], k) not in upd_at}
            opp += len(eligible)
            lost += len(eligible - cur)
        prev = cur
    return {"opportunities": opp, "losses": lost, "epsilon": lost / opp if opp else float("nan")}


def survival_curves(constraints: list[dict], rewrites: list[dict], field: str = "exact") -> dict:
    """Survival of the never-updated (static) turn-1 constraints after each rewrite, for the (1 - eps)^T plot:
    ``present`` = value present at rewrite t; ``continuous`` = present at every rewrite from the first to t.
    Fractions over the static constraints present after the first rewrite (NaN if none)."""
    static = [c["key"] for c in constraints if not c.get("updates")]
    if not rewrites:
        return {"turns": [], "present": [], "continuous": []}
    base = set(rewrites[0]["present"][field]) & set(static)
    alive = set(base)
    turns, pres, cont = [], [], []
    for rw in rewrites:
        cur = set(rw["present"][field])
        alive &= cur
        turns.append(rw["turn"])
        pres.append(len(base & cur) / len(base) if base else float("nan"))
        cont.append(len(alive) / len(base) if base else float("nan"))
    return {"turns": turns, "present": pres, "continuous": cont, "n_static_base": len(base)}
