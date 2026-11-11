"""Track A2 session-state conditions D1-D6 (protocol 4.2) and the probe (4.1).

Every condition reads a drift session (opa/data/SCHEMA.md) turn by turn, keeps its own state, and at each probe turn
(10, 25, 50) asks the model for the current value of every constraint *from that state only*. The probe names the
constraints by key and label (never their values). All model calls go through ``track_a.models.Chat``; every
controller action is one event (with the raw user prompt and reply when ``opts.log_raw`` is true; system prompts are
the fixed constants below, logged by name and version).

Model calls per 50-turn session (b = 8 turns per inner-shell node, 3 probes; retries excluded)
--------------------------------------------------------------------------------------------
  D1  full history ............ 3 probe calls                                      =  3 calls
  D2  rolling summary ......... 50 rewrites (one per turn, turn 1 included) + 3      = 53 calls
  D3  MemGPT-style notes ...... 50 update decisions (one per turn) + 3               = 53 calls
  D4  registers ............... 50 extraction calls + 3 (+ <= 1 retry per turn with a rejected pin)   = 53 calls
  D5  as D4 ................... 53 calls
  D6  as D4 + one digest per closed node (~16-17 turns per thread / 8 -> ~6) ......  ~ 59 calls
Token arithmetic for budgeting lives in ``run.estimate_session`` (measured from the dev sessions: turn 1 is about
530 characters, later turns about 100, a whole session about 5.4K characters, i.e. ~1.4K tokens).
"""
from __future__ import annotations

import json
import time

from ..track_a.models import Chat, Usage, parse_json
from .state import Registers, Shells, presence

PROMPT_VERSION = "a2-v1"
CONDITIONS = ("D1", "D2", "D3", "D4", "D5", "D6")

SUMMARY_SYSTEM = """You maintain a rolling summary of an engineering specification discussion. Only your summary is kept: earlier turns will not be shown again.
You see the current summary and the newest turn. Rewrite the summary so that it keeps everything that may matter later, above all the current value of every constraint exactly as stated (numbers, units, tolerances, part numbers, supplier names, dates).
Keep the summary under {budget} tokens. Reply with the new summary text only."""

NOTES_SYSTEM = """You are the memory manager of an assistant taking part in an engineering specification discussion (MemGPT-style memory). You keep free-text memory notes; only the notes are kept, earlier turns will not be shown again.
After each new turn decide whether the notes need to change. If they do, write the complete new notes (under {budget} tokens), keeping the current value of every constraint exactly as stated.
Reply with one JSON object and nothing else: {{"update": true, "notes": "<the complete new notes>"}} or {{"update": false}}."""

EXTRACT_SYSTEM = """You maintain the session registers of an engineering specification discussion. Each turn is an origin; its origin_id is the turn number.
Registers are JSON: goal, constraints [{pin_id, key, value, unit, quote, origin_id, thread, superseded_by}] and decisions. Every entry is a pin: an exact quote from the turn that stated it.
For the new turn, return the operations that keep the registers current:
- {"op": "add", "key": "<snake_case name>", "value": "<value exactly as written>", "unit": "<unit or empty>", "quote": "<the sentence that states it>"} for a constraint not yet in the registers;
- {"op": "update", "replaces": "<pin_id of the entry being changed>", "key": "<same key>", "value": "<new value exactly as written>", "unit": "...", "quote": "<the sentence that changes it>"} when the turn changes a constraint's value;
- {"op": "decision", "quote": "..."} for an explicit decision that is not a constraint value;
- {"op": "goal", "quote": "..."} to record the goal of the session.
Only binding statements and explicit changes count. Never register values that are merely mentioned: old specs, catalogue values, rejected proposals, earlier product generations, competitors.
Copy every quote exactly, character for character, from the turn text; the value must appear inside its quote.
Reply with one JSON object and nothing else: {"ops": [...]} ({"ops": []} if nothing changes)."""

DIGEST_SYSTEM = """Write a short digest of the following turns of one thread of an engineering specification discussion. The digest is stored in an index node; later readers may use it as evidence. Keep it under {budget} tokens. Reply with the digest text only."""

PROBE_PLAIN_SYSTEM = """You are assisting with an engineering specification discussion. Using only the session state shown, give the current value of every listed constraint: the value that holds now, after all changes so far. Copy values exactly as stated, with their units. If the state does not tell you a constraint's value, give null.
Reply with one JSON object and nothing else: {"answers": {"<key>": "<value or null>", ...}}"""

PROBE_REG_SYSTEM = """You are assisting with an engineering specification discussion. The session state is a set of registers: every constraint entry is an exact quote pinned from the turn (origin_id) that stated it. An entry whose superseded_by is set has been replaced by the entry it names.
Using only the state shown, give the current value of every listed constraint, copied from the state, and cite the pin_id of the entry you used. If the state does not tell you a constraint's value, give null.
Reply with one JSON object and nothing else: {"answers": {"<key>": {"value": "<value or null>", "cite": "<pin_id or null>"}, ...}}"""

PROBE_DIGEST_SYSTEM = """You are assisting with an engineering specification discussion. The session state is a set of registers plus the digests of inner-shell index nodes.
Every register constraint entry is an exact quote pinned from the turn (origin_id) that stated it; an entry whose superseded_by is set has been replaced by the entry it names. A digest is an abstractive summary of a group of turns of one thread; you may also use a digest as evidence and cite its node_id.
Using only the state shown, give the current value of every listed constraint and cite the pin_id or node_id you used. If the state does not tell you a constraint's value, give null.
Reply with one JSON object and nothing else: {"answers": {"<key>": {"value": "<value or null>", "cite": "<pin_id, node_id or null>"}, ...}}"""

DEFAULT_OPTS = {
    "context_limit": 36_000,    # D1 (v3: 40,960 window; was 180_000): approximate tokens (chars / 4) of history that still fit
    "summary_tokens": 1000,     # D2 budget (also the rewrite's max_tokens)
    "notes_tokens": 1000,       # D3 budget
    "extract_tokens": 1200,     # D4-D6 max_tokens per extraction reply (turn 1 has ~10 add ops)
    "digest_b": 8,              # D6: turns per inner-shell node per thread
    "digest_tokens": 200,       # D6 digest budget
    "probe_tokens": 800,        # max_tokens of a probe answer
    "pin_retry": 1,             # D4-D6: re-asks per turn when an operation was rejected
    "json_retry": 1,            # re-asks when a reply is not one JSON object
    "log_raw": True,            # raw user prompts and replies in the event log
}


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def turn_line(t: dict) -> str:
    return f"[turn {t['turn']} | thread {t['thread']}] {t['text']}"


class Controller:
    """Base: one session, one condition. ``run()`` returns the drift record (opa.scoring format plus extras)."""

    condition = "D?"
    probe_system = PROBE_PLAIN_SYSTEM
    probe_system_name = "PROBE_PLAIN_SYSTEM"

    def __init__(self, session: dict, chat: Chat, opts: dict, log_fn):
        self.s, self.chat, self.log_fn = session, chat, log_fn
        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        self.usage = Usage()
        self.step = 0
        self.keys = [(c["key"], c["label"]) for c in session["constraints"]]
        self.probe_turns = sorted(p["turn"] for p in session["probes"])
        self.record: dict = {"item_id": session["item_id"], "condition": self.condition, "model": chat.name,
                             "prompt_version": PROMPT_VERSION, "probes": [], "status": "ok", "error": None}

    # ------------------------------------------------------------ logging and model access
    def log(self, action: str, **kw):
        self.step += 1
        self.log_fn({"item_id": self.s["item_id"], "condition": self.condition, "model": self.chat.name,
                     "step": self.step, "action": action, **kw})

    def ask(self, action: str, system_name: str, system: str, user: str, max_tokens: int, want_json: bool,
            **kw) -> tuple[str, dict | None]:
        """One model call (plus up to ``json_retry`` re-asks when JSON is wanted and unparseable), logged as one
        event per call."""
        tries = 0
        while True:
            tin, tout = self.usage.tokens_in, self.usage.tokens_out
            t0 = time.time()
            text = self.chat.ask(system, user, max_tokens, self.usage)
            obj = parse_json(text) if want_json else None
            ev = {"system_prompt": f"{system_name}@{PROMPT_VERSION}", "tokens_in": self.usage.tokens_in - tin,
                  "tokens_out": self.usage.tokens_out - tout, "latency_ms": int(1000 * (time.time() - t0)),
                  "attempt": tries + 1, **kw}
            if want_json:
                ev["parsed"] = obj is not None
            if self.opts["log_raw"]:
                ev["prompt"], ev["reply"] = user, text
            self.log(action, **ev)
            if not want_json or obj is not None or tries >= int(self.opts["json_retry"]):
                return text, obj
            tries += 1
            user = user + "\n\nYour previous reply was not a single valid JSON object. Reply with JSON only."

    # ------------------------------------------------------------ the session loop
    def run(self) -> dict:
        t0 = time.time()
        turns = sorted(self.s["turns"], key=lambda t: t["turn"])
        probe_at = set(self.probe_turns)
        for t in turns:
            self.observe(t)
            if t["turn"] in probe_at:
                self.do_probe(t["turn"])
        self.finish()
        self.record.update(tokens_in=self.usage.tokens_in, tokens_out=self.usage.tokens_out,
                           n_llm_calls=self.usage.calls, wall_ms=int(1000 * (time.time() - t0)))
        return self.record

    def observe(self, t: dict):
        raise NotImplementedError

    def state_text(self, turn: int) -> str:
        raise NotImplementedError

    def finish(self):
        pass

    # ------------------------------------------------------------ probe
    def probe_user(self, turn: int, state: str) -> str:
        lines = "\n".join(f"- {k}: {label}" for k, label in self.keys)
        return f"{state}\n\nProbe after turn {turn}. Give the current value of each of these constraints:\n{lines}"

    def do_probe(self, turn: int):
        state = self.state_text(turn)
        if state is None:                           # D1 past its context limit
            self.log("probe_skipped", turn=turn, reason="context_limit")
            self.record["probes"].append({"turn": turn, "predicted": {}, "fits": False})
            return
        _, obj = self.ask("probe", self.probe_system_name, self.probe_system, self.probe_user(turn, state),
                          int(self.opts["probe_tokens"]), True, turn=turn, state_tokens=approx_tokens(state))
        predicted, cites = parse_answers(obj, self.keys)
        probe = {"turn": turn, "predicted": predicted, "fits": True, "state_tokens": approx_tokens(state)}
        if cites is not None:
            probe["cites"] = cites
            probe["cite_kind"] = {k: self.cite_kind(v) for k, v in cites.items()}
        self.record["probes"].append(probe)
        self.log("answer", turn=turn, predicted=predicted, cites=cites)

    def cite_kind(self, cite) -> str:
        return "none"


def parse_answers(obj: dict | None, keys: list[tuple[str, str]]) -> tuple[dict, dict | None]:
    """{"answers": {key: value | {"value", "cite"}}} -> (predicted {key: str|None}, cites {key: str|None} or None
    when no answer carried a cite field). Keys may also be given by label."""
    ans = (obj or {}).get("answers")
    if not isinstance(ans, dict):
        ans = {}
    by_label = {label.lower(): k for k, label in keys}
    norm = {}
    for k, v in ans.items():
        kk = str(k).strip()
        kk = kk if kk in dict(keys) else by_label.get(kk.lower(), kk)
        norm[kk] = v
    predicted, cites, any_cite = {}, {}, False
    for k, _ in keys:
        v = norm.get(k)
        cite = None
        if isinstance(v, dict):
            any_cite = any_cite or "cite" in v
            cite = v.get("cite")
            v = v.get("value")
        predicted[k] = None if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none")) else str(v)
        cites[k] = None if cite in (None, "", "null") else str(cite)
    return predicted, (cites if any_cite else None)


# ---------------------------------------------------------------- D1
class FullHistory(Controller):
    condition = "D1"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.history: list[dict] = []

    def observe(self, t):
        self.history.append(t)

    def state_text(self, turn):
        text = "Session history (every turn so far):\n" + "\n".join(turn_line(t) for t in self.history)
        if approx_tokens(text) > int(self.opts["context_limit"]):
            return None
        return text


# ---------------------------------------------------------------- D2
class RollingSummary(Controller):
    condition = "D2"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.summary = ""
        self.rewrites: list[dict] = []

    def observe(self, t):
        b = int(self.opts["summary_tokens"])
        user = f"Current summary:\n{self.summary or '(empty)'}\n\nNew turn:\n{turn_line(t)}"
        text, _ = self.ask("summary_rewrite", "SUMMARY_SYSTEM", SUMMARY_SYSTEM.format(budget=b), user, b, False,
                           turn=t["turn"])
        self.summary = text.strip()
        pres = presence(self.s["constraints"], t["turn"], self.summary)
        self.rewrites.append({"turn": t["turn"], "summary_tokens": approx_tokens(self.summary), "present": pres})
        self.log("rewrite_check", turn=t["turn"], present_exact=pres["exact"], present_norm=pres["norm"])

    def state_text(self, turn):
        return f"Session summary:\n{self.summary or '(empty)'}"

    def finish(self):
        self.record["rewrites"] = self.rewrites


# ---------------------------------------------------------------- D3
class MemoryNotes(Controller):
    condition = "D3"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.notes = ""
        self.updates: list[int] = []
        self.parse_failures = 0

    def observe(self, t):
        b = int(self.opts["notes_tokens"])
        user = f"Current memory notes:\n{self.notes or '(empty)'}\n\nNew turn:\n{turn_line(t)}"
        _, obj = self.ask("notes_decision", "NOTES_SYSTEM", NOTES_SYSTEM.format(budget=b), user, b + 100, True,
                          turn=t["turn"])
        if obj is None:                             # unparseable even after the re-ask: notes are kept unchanged
            self.parse_failures += 1
            self.log("notes_parse_error", turn=t["turn"], notes_tokens=approx_tokens(self.notes))
            return
        upd = obj.get("update") in (True, "true", "yes", 1) and isinstance(obj.get("notes"), str)
        if upd:
            self.notes = obj["notes"].strip()
            self.updates.append(t["turn"])
        self.log("notes_update" if upd else "notes_keep", turn=t["turn"], notes_tokens=approx_tokens(self.notes))

    def state_text(self, turn):
        return f"Memory notes:\n{self.notes or '(empty)'}"

    def finish(self):
        self.record["notes_updates"] = self.updates
        self.record["notes_parse_failures"] = self.parse_failures


# ---------------------------------------------------------------- D4 / D5 / D6
class RegisterState(Controller):
    condition = "D4"
    supersede = True
    probe_system = PROBE_REG_SYSTEM
    probe_system_name = "PROBE_REG_SYSTEM"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reg = Registers(supersede=self.supersede)

    def observe(self, t):
        text, turn, thread = t["text"], t["turn"], t["thread"]
        user = (f"Registers:\n{self.reg.render()}\n\nNew turn (origin_id {turn}, thread {thread}):\n{text}")
        _, obj = self.ask("extract", "EXTRACT_SYSTEM", EXTRACT_SYSTEM, user, int(self.opts["extract_tokens"]), True,
                          turn=turn)
        rej = self._apply((obj or {}).get("ops"), t)
        tries = 0
        while rej and tries < int(self.opts["pin_retry"]):
            tries += 1
            why = json.dumps([{"op": r["op"], "reason": r["reason"]} for r in rej], ensure_ascii=False)
            retry_user = (f"Registers:\n{self.reg.render()}\n\nNew turn (origin_id {turn}, thread {thread}):\n{text}\n\n"
                          f"The harness rejected these operations: {why}\nA quote must be an exact substring of the "
                          f"turn text and must contain the value. Reply with corrected operations for these only, as "
                          f'{{"ops": [...]}}.')
            _, obj = self.ask("extract_retry", "EXTRACT_SYSTEM", EXTRACT_SYSTEM, retry_user,
                              int(self.opts["extract_tokens"]), True, turn=turn)
            rej = self._apply((obj or {}).get("ops"), t)

    def _apply(self, ops, t) -> list[dict]:
        acc, rej = self.reg.apply(ops, t["turn"], t["thread"], t["text"])
        for ev in acc:
            self.log("pin", turn=t["turn"], origin_id=t["turn"], thread=t["thread"],
                     pins=[{"pin_id": ev["pin_id"], "origin_id": t["turn"], "char_span": ev["char_span"]}],
                     op=ev["op"], key=ev.get("key"), value=ev.get("value"), duplicate_key=ev.get("duplicate_key"))
            if ev.get("replaces"):
                self.log("supersede" if ev.get("superseded") else "update_unlinked", turn=t["turn"],
                         new_pin=ev["pin_id"], old_pin=ev["replaces"], via=ev.get("replaces_via"))
        for ev in rej:
            op = ev["op"] if isinstance(ev["op"], dict) else {"raw": str(ev["op"])}
            self.log("pin_rejected", turn=t["turn"], origin_id=t["turn"], reason=ev["reason"],
                     quote=str(op.get("quote", ""))[:300], value=op.get("value"))
        return rej

    def state_text(self, turn):
        return f"Registers:\n{self.reg.render()}"

    def cite_kind(self, cite) -> str:
        if cite is None:
            return "none"
        c = str(cite).strip()
        e = self.reg.by_pin(c)
        if e is not None:
            return "register_superseded" if e["superseded_by"] else "register"
        if c in self.reg.all_pin_ids():
            return "register"
        if c.isdigit() or (c[:1] in "tT" and c[1:].isdigit()):
            return "origin"
        return "unknown"

    def finish(self):
        self.record["registers"] = self.reg.to_json()
        self.record["pins_accepted"] = self.reg.accepted
        self.record["pins_rejected"] = [{"turn": r["turn"], "reason": r["reason"]} for r in self.reg.rejected]


class RegisterNoLinks(RegisterState):
    condition = "D5"
    supersede = False


class RegisterDigests(RegisterState):
    condition = "D6"
    probe_system = PROBE_DIGEST_SYSTEM
    probe_system_name = "PROBE_DIGEST_SYSTEM"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shells = Shells(int(self.opts["digest_b"]))
        self.turn_text = {}

    def observe(self, t):
        super().observe(t)
        self.turn_text[t["turn"]] = t
        node = self.shells.add_turn(t["thread"], t["turn"])
        if node is not None:
            b = int(self.opts["digest_tokens"])
            user = "\n".join(turn_line(self.turn_text[i]) for i in node["children"])
            text, _ = self.ask("digest", "DIGEST_SYSTEM", DIGEST_SYSTEM.format(budget=b), user, b, False,
                               turn=t["turn"], node_id=node["node_id"], children=node["children"])
            node["digest"] = text.strip()

    def state_text(self, turn):
        return f"Registers:\n{self.reg.render()}\n\nInner-shell digests:\n{self.shells.render()}"

    def cite_kind(self, cite) -> str:
        if cite is not None and any(str(cite).strip() == n["node_id"] for n in self.shells.nodes):
            return "digest"
        return super().cite_kind(cite)

    def finish(self):
        super().finish()
        self.record["digests"] = [{"node_id": n["node_id"], "thread": n["thread"], "children": n["children"],
                                   "digest": n["digest"]} for n in self.shells.nodes]


CONTROLLERS = {"D1": FullHistory, "D2": RollingSummary, "D3": MemoryNotes, "D4": RegisterState,
               "D5": RegisterNoLinks, "D6": RegisterDigests}


def run_session(condition: str, session: dict, chat: Chat, opts: dict, log_fn) -> dict:
    if condition not in CONTROLLERS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return CONTROLLERS[condition](session, chat, opts, log_fn).run()
