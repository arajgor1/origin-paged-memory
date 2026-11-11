"""H10 thrashing controller (protocol 3.8, paper Proposition 15).

Entrypoint `opa.tracks.track_a:run_thrash` runs one (M, pinning) arm on thrash items (opa/data/thrash.py: "which of
these r suppliers quotes the lowest price for part X?", the r quotes in r different origins among 32).

The step needs the r quotes at once. The model works through a paged memory of M slots (opa.tracks.track_a.systems.Fold
is reused for the slot LRU, the exact-pin rule and the pin budget, so the mechanics and event names are the same as
in the Track A OPA systems). Each step is one model call returning one JSON action:

  {"action": "page_in", "origin_id": <int>}          load an origin; when the M slots are full the least recently
                                                      used origin is evicted (logged `evict`, reason lru). Paging in
                                                      an origin that is already resident is an LRU touch (logged
                                                      `touch`), not a page-in.
  {"action": "pin", "origin_id": <int>, "quote": ..} pinning arm only: pin an exact quote from a RESIDENT origin; pins
                                                      stay visible after eviction, within a pin budget of P_pin tokens
                                                      (oldest pin dropped first, logged `evict_pin`); quotes longer
                                                      than max_pin_chars are rejected (pinning is extraction, not a
                                                      way to keep a whole origin)
  {"action": "answer", "answer": "<name>"}           ends the item

Design choices (both arms identical except for pinning):
- No side memory. The model sees only the question, the index, the resident origins' text, the pins (pinning arm),
  and a log of past actions rendered by the controller from the parsed actions (kind + origin ids, never the model's
  own text). Its earlier replies are not shown, so an evicted origin's content is really gone unless it was pinned.
- Fair candidate finding: both arms get the same index, built before the run from the question alone: the r supplier
  names are parsed from the question and each is mapped to the origins whose text mentions it (normalized whole-word
  match, the S-lex skeleton rule; text.mentions). No gold is used to build it. This isolates memory management (what
  H10 measures) from search; without it, scanning 32 origins with M = 2 would confound thrashing with search cost.
- Step cap: max_steps = 4r + 4 model actions by default (opts.max_steps overrides). The last step's prompt tells the
  model to answer; if it still does not, the prediction is empty and status = "step_cap" (`used_final_step` marks
  items that reached the last step either way). An unparseable reply is
  re-asked once within the same step; if still unparseable, the step is spent as an invalid action.
- H10 outcome: repeated page-ins = sum over origins of (loads - 1), counting only real loads (not touches).

Per-item record: prediction, em (scoring.exact_match), paged_in, evicted, touches, pins, actions, page_ins,
repeated_page_ins, repeated_needed_page_ins and needed_visible_max (gold used only after the run, for diagnostics),
steps, max_steps, tokens_in/out, n_llm_calls, status. metrics.json: overall and by cell "r<r>-M<M>-<pin|nopin>".
`h10_contrast(unpinned_records, pinned_records)` gives the pre-registered key pinned_vs_unpinned_repeat_pagein_reduction
(paired by item, value = mean(unpinned - pinned) repeated page-ins, Wilcoxon signed-rank, scale 1; also sd_diff,
the SD of the per-item differences).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from .models import Usage, answer_text, parse_json
from .run import make_chat, select_items, stops_vllm
from .systems import Ctx, Fold
from .text import find_exact, mentions, normalize_key

PROMPT_VERSION = "thrash-v2"   # v1 asked for a "<short answer>"; Qwen3-4B wrote sentences, so exact match scored 0 for correct picks.
# v2 also adds the pin-arm policy sentence (tuning iteration 1) and makes the two arms' prompts identical except the pin lines.
SYSTEM_NAME = "THRASH"
CONTRAST_KEY = "pinned_vs_unpinned_repeat_pagein_reduction"

# The two arms share every line of the system prompt except the pin-specific ones: SYSTEM_HEAD + [PIN_RULES] +
# SYSTEM_BODY + ACTION_PAGE_IN + [ACTION_PIN] + ACTION_ANSWER. Nothing else differs (tests/test_thrash.py checks it).
SYSTEM_HEAD = """You answer a question about a long archive split into numbered origins (text chunks), through a small paged memory.
Memory: you have {M} slots. Only the origins currently in the slots are visible to you. Loading an origin when all slots are full evicts the least recently used one, and its text is then gone. Loading an origin that is already in a slot only marks it as most recently used.
You have no other memory: your earlier replies are not shown to you again, only a log of the actions taken.
"""

# Pin-arm block 1 (tuning iteration 1, thrash-v2): the rules plus the intended policy.
PIN_RULES = """Pinning: you may pin an exact quote (one or two sentences, at most {C} characters, copied character for character) from an origin that is currently in a slot. Pins stay visible after that origin is evicted, next to the origins in your slots. Pin budget: {P} tokens; when it is exceeded the oldest pin is dropped. A quote that is not an exact substring of a resident origin is rejected.
Policy: pin each quote you will need to answer (for this question: each candidate supplier's price quote) as soon as you see it, before you page in the next origin. Once all the quotes you need are pinned, answer from the pinned quotes.
"""

SYSTEM_BODY = """An index lists, for each name in the question, the origins whose text mentions it.
Answer only from what is visible now. Do not page in an origin again if the text you need from it is still visible to you (in a slot or, if you have them, in your pinned quotes). Each reply is one step, and you have at most {S} steps.
Reply with exactly one JSON object and nothing else, one of:
"""

ACTION_PAGE_IN = """{"action": "page_in", "origin_id": <int>}
"""
ACTION_PIN = """{"action": "pin", "origin_id": <int>, "quote": "<exact text>"}
"""
ACTION_ANSWER = """{"action": "answer", "answer": "<the supplier name only, copied from the question, no other words>"}"""

TUNING_ITERATION = {"pin": 1, "nopin": 1}      # system-prompt tuning iterations used on dev so far (budget: 10 per system); the shared
# SYSTEM_BODY sentence changed both arms in iteration 1


def build_system_prompt(M: int, S: int, pinning: bool, C: int = 300, P: int = 512) -> str:
    return (SYSTEM_HEAD.format(M=M) + (PIN_RULES.format(C=C, P=P) if pinning else "") + SYSTEM_BODY.format(S=S)
            + ACTION_PAGE_IN + (ACTION_PIN if pinning else "") + ACTION_ANSWER)


_Q_NAMES = re.compile(r":\s*(.+?)\?\s*$", re.S)


# ---------------------------------------------------------------- helpers


def question_names(question: str) -> list[str]:
    """The supplier names printed in a thrash question ("...for part X: A, B, C?")."""
    m = _Q_NAMES.search(question)
    if not m:
        return []
    return [n.strip() for n in m.group(1).split(",") if n.strip()]


def build_index(item: dict) -> dict[str, list[int]]:
    """name -> origin ids whose text mentions it (normalized whole-word match). Built from the question only."""
    out = {}
    for name in question_names(item["question"]):
        k = normalize_key(name)
        out[name] = [o["origin_id"] for o in item["origins"] if mentions(o["text"], k)]
    return out


def default_max_steps(r: int, opts: dict) -> int:
    if opts.get("max_steps") is not None:
        return int(opts["max_steps"])
    return int(opts.get("step_mult", 4)) * r + int(opts.get("step_add", 4))


def repeats(paged_in: list[int], only: set | None = None) -> int:
    c = Counter(o for o in paged_in if only is None or o in only)
    return sum(v - 1 for v in c.values())


# ---------------------------------------------------------------- the controller


class ThrashController:
    def __init__(self, item: dict, chat, M: int, pinning: bool, opts: dict, log_fn, usage: Usage | None = None):
        self.item, self.chat, self.M, self.pinning = item, chat, int(M), bool(pinning)
        self.opts = dict(opts)
        self.r = int(item.get("params", {}).get("r") or len(question_names(item["question"])))
        self.max_steps = default_max_steps(self.r, self.opts)
        self.P_pin = int(self.opts.get("P_pin", 512))
        self.max_pin_chars = int(self.opts.get("max_pin_chars", 300))
        self.max_tokens = int(self.opts.get("max_tokens", 300))
        self.log_prompts = bool(self.opts.get("log_prompts", True))
        sess = SimpleNamespace(N=len(item["origins"]), texts=[o["text"] for o in item["origins"]],
                               thread_of=[o.get("thread_id", 0) for o in item["origins"]])
        self.c = Ctx(item=item, sess=sess, chat=chat, opts={"M": self.M, "P_pin": self.P_pin}, log_fn=log_fn,
                     usage=usage or Usage())
        self.fold = Fold(self.c, "S6")           # slots (LRU), exact-pin rule, pin budget; no skeleton/graph used
        self.index = build_index(item)
        self.system_prompt = build_system_prompt(self.M, self.max_steps, self.pinning, self.max_pin_chars, self.P_pin)
        self.history: list[str] = []
        self.actions: list[dict] = []
        self.touches: list[int] = []
        self.trace: list[dict] = []
        self.invalid = 0
        self.pins_rejected = 0

    # ---- prompt
    def user_prompt(self, step: int) -> str:
        s, f = self.c.sess, self.fold
        idx = "\n".join(f"- {n}: {ids}" for n, ids in self.index.items()) or "(no names found)"
        resident = "\n".join(f'<origin id="{o}">\n{s.texts[o]}\n</origin>' for o in f.slots) or "(empty)"
        parts = [f"Question: {self.item['question']}",
                 f"Index (name -> origins that mention it):\n{idx}",
                 f"Slots ({len(f.slots)}/{self.M} used, least recently used first): {list(f.slots)}",
                 f"Resident origins:\n{resident}"]
        if self.pinning:
            pins = "\n".join(f'[{p.pin_id}] (origin {p.origin_id}) "{p.quote}"' for p in f.live()) or "(none)"
            parts.append(f"Pinned quotes:\n{pins}")
        parts.append("Actions so far:\n" + ("\n".join(f"{i + 1}. {h}" for i, h in enumerate(self.history)) or "(none)"))
        tail = f"Step {step} of {self.max_steps}."
        if step == self.max_steps:
            tail += " This is your final step: reply with an answer action."
        parts.append(tail)
        return "\n\n".join(parts)

    def ask(self, step: int) -> tuple[dict | None, str]:
        user = self.user_prompt(step)
        text = self.chat.ask(self.system_prompt, user, self.max_tokens, self.c.usage, self.c.log_fn)
        obj = parse_json(text)
        ev = {"step": step, "reply": text[:2000], "prompt_sha256": hashlib.sha256(user.encode()).hexdigest()[:16],
              "prompt_chars": len(user)}
        if self.log_prompts:
            ev["prompt"] = user
        if obj is None:
            self.c.log("llm_step", retry=0, **ev)
            text = self.chat.ask(self.system_prompt, user + "\n\nYour previous reply was not a single valid JSON object. "
                                 "Reply with JSON only.", self.max_tokens, self.c.usage, self.c.log_fn)
            obj = parse_json(text)
            ev.update(reply=text[:2000])
            self.c.log("llm_step", retry=1, **{k: v for k, v in ev.items() if k != "prompt"})
        else:
            self.c.log("llm_step", retry=0, **ev)
        return obj, text

    # ---- actions
    def _origin(self, a: dict):
        try:
            o = int(a.get("origin_id"))
        except (TypeError, ValueError):
            return None
        return o if 0 <= o < self.c.sess.N else None

    def _invalid(self, step: int, reason: str, a):
        self.invalid += 1
        self.history.append(f"invalid action ({reason})")
        self.c.log("invalid_action", step=step, reason=reason, got=json.dumps(a, default=str)[:300] if a else None)

    def do_page_in(self, step: int, a: dict):
        o = self._origin(a)
        if o is None:
            return self._invalid(step, "bad origin_id", a)
        f = self.fold
        if o in f.slots:                                     # LRU touch: no load, no eviction
            f.slots.remove(o)
            f.slots.append(o)
            self.touches.append(o)
            self.c.log("touch", step=step, origin_id=o, slots=list(f.slots))
            self.history.append(f"page_in {o} (already resident; marked most recently used)")
            return
        before = list(f.slots)
        n_prev = f.rec["paged_in"].count(o)
        f.page_in(o, via="thrash")
        evicted = [x for x in before if x not in f.slots]
        if n_prev:
            self.c.log("repeat_page_in", step=step, origin_id=o, nth_load=n_prev + 1)
        self.history.append(f"page_in {o}" + (f" (evicted {evicted[0]})" if evicted else ""))

    def do_pin(self, step: int, a: dict):
        if not self.pinning:
            return self._invalid(step, "pinning disabled", a)
        o = self._origin(a)
        quote = str(a.get("quote") or "")
        f = self.fold
        reason = None
        if o is None:
            reason = "bad origin_id"
        elif o not in f.slots:
            reason = "origin not resident"
        elif len(quote.strip()) > self.max_pin_chars:
            reason = "quote too long"
        elif find_exact(quote, self.c.sess.texts[o]) is None:
            reason = "not an exact quote"
        if reason:
            self.pins_rejected += 1
            if reason != "not an exact quote":
                self.c.log("pin_rejected", step=step, origin_id=o, reason=reason, quote=quote[:200])
            else:
                f.pin(o, quote, "", [], False)                # logs pin_rejected (exact-pin rule)
            self.history.append(f"pin from origin {o} rejected ({reason})")
            return
        live0 = [p for p in f.live()]
        ok = f.pin(o, quote, "", [], False)
        if not ok:
            self.pins_rejected += 1
            self.c.log("pin_rejected", step=step, origin_id=o, reason="duplicate", quote=quote[:200])
            self.history.append(f"pin from origin {o} rejected (duplicate)")
            return
        f.enforce_budget()
        dropped = [p.pin_id for p in live0 if p.evicted]
        self.history.append(f"pin {f.pins[-1].pin_id} from origin {o}" + (f" (dropped {', '.join(dropped)})" if dropped else ""))

    # ---- run
    def run(self) -> dict:
        f, rec = self.fold, self.fold.rec
        rec.update(system=f"{SYSTEM_NAME}-{'pin' if self.pinning else 'nopin'}", arm="pin" if self.pinning else "nopin",
                   r=self.r, M=self.M, pinning=self.pinning, max_steps=self.max_steps, prompt_version=PROMPT_VERSION,
                   tuning_iteration=TUNING_ITERATION["pin" if self.pinning else "nopin"], index=self.index)
        self.c.log("thrash_start", r=self.r, M=self.M, pinning=self.pinning, max_steps=self.max_steps, P_pin=self.P_pin,
                   prompt_version=PROMPT_VERSION, tuning_iteration=TUNING_ITERATION["pin" if self.pinning else "nopin"],
                   index=self.index, system_prompt=self.system_prompt)
        pred, status, steps = "", "step_cap", 0
        for step in range(1, self.max_steps + 1):
            steps = step
            a, _ = self.ask(step)
            kind = str((a or {}).get("action", "")).strip().lower()
            self.actions.append({"step": step, "action": kind or None,
                                 **({k: a[k] for k in ("origin_id", "quote", "answer") if k in a} if a else {})})
            if a is None:
                self._invalid(step, "unparseable reply", None)
            elif kind == "page_in":
                self.do_page_in(step, a)
            elif kind == "pin":
                self.do_pin(step, a)
            elif kind == "answer":
                pred, status = answer_text(a.get("answer")), "ok"
                self.c.log("answer", step=step, answer=pred[:200], slots=list(f.slots),
                           pins=[p.pin_id for p in f.live()])
            else:
                self._invalid(step, f"unknown action {kind!r}", a)
            self.trace.append({"step": step, "slots": list(f.slots), "pin_origins": sorted({p.origin_id for p in f.live()})})
            if status == "ok":
                break
        if status != "ok":
            self.c.log("step_cap", steps=steps, max_steps=self.max_steps)
        rec.update(prediction=pred, status=status, steps=steps, used_final_step=steps == self.max_steps,
                   actions=self.actions, touches=self.touches,
                   invalid_actions=self.invalid, pins_rejected=self.pins_rejected, trace=self.trace,
                   pins=[{"pin_id": p.pin_id, "origin_id": p.origin_id, "char_span": p.char_span, "quote": p.quote,
                          "evicted": p.evicted} for p in f.pins])
        self._outcomes()
        return rec

    def _outcomes(self):
        rec = self.fold.rec
        loads = rec["paged_in"]
        rec.update(page_ins=len(loads), distinct_page_ins=len(set(loads)), repeated_page_ins=repeats(loads),
                   repeated_by_origin={str(o): n - 1 for o, n in sorted(Counter(loads).items()) if n > 1},
                   n_evictions=sum(e["reason"] == "lru" for e in rec["evicted"]))
        # diagnostics with gold, after the run only
        from ... import scoring
        needed = set(self.item.get("gold", {}).get("needed_origins", []))
        rec["em"] = scoring.exact_match(rec["prediction"], scoring.gold_answers(self.item))
        rec["repeated_needed_page_ins"] = repeats(loads, needed)
        rec["needed_paged"] = len(needed & set(loads))
        rec["needed_visible_max"] = max((len(needed & (set(t["slots"]) | set(t["pin_origins"]))) for t in self.trace),
                                        default=0)


def run_thrash_item(item: dict, chat, M: int, pinning: bool, opts: dict, log_fn) -> dict:
    usage = Usage()
    t0 = time.time()
    ctl = None
    try:
        ctl = ThrashController(item, chat, M, pinning, opts, log_fn, usage)
        rec = ctl.run()
    except Exception as e:                                   # one failed item never kills the run
        rec = {"item_id": item["item_id"], "system": f"{SYSTEM_NAME}-{'pin' if pinning else 'nopin'}",
               "arm": "pin" if pinning else "nopin", "r": int(item.get("params", {}).get("r", 0)), "M": int(M),
               "pinning": bool(pinning), "model": getattr(chat, "name", ""), "prediction": "", "em": 0.0,
               "status": ("budget" if type(e).__name__ == "BudgetExceeded" else
                          "does_not_fit" if type(e).__name__ == "ContextOverflow" else "error"),
               "error": traceback.format_exc()[-2000:], "prompt_version": PROMPT_VERSION,
               "tuning_iteration": TUNING_ITERATION["pin" if pinning else "nopin"]}
        if ctl is not None:                                   # keep what was measured before the failure
            loads = ctl.fold.rec["paged_in"]
            rec.update(paged_in=loads, page_ins=len(loads), repeated_page_ins=repeats(loads),
                       steps=len(ctl.actions), max_steps=ctl.max_steps)
    rec.update(tokens_in=usage.tokens_in, tokens_out=usage.tokens_out, n_llm_calls=usage.calls,
               wall_ms=int(1000 * (time.time() - t0)), oracle=False)
    return rec


# ---------------------------------------------------------------- metrics and the H10 contrast


def cell_key(r: int, M: int, pinning: bool) -> str:
    return f"r{r}-M{M}-{'pin' if pinning else 'nopin'}"


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _block(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("status") in ("ok", "step_cap")]
    return {"n": len(recs), "n_scored": len(ok),
            "em": _mean(r.get("em", 0.0) for r in ok),
            "page_ins_mean": _mean(r.get("page_ins") for r in ok),
            "repeated_page_ins_mean": _mean(r.get("repeated_page_ins") for r in ok),
            "repeated_needed_page_ins_mean": _mean(r.get("repeated_needed_page_ins") for r in ok),
            "repeat_any_rate": _mean(float((r.get("repeated_page_ins") or 0) > 0) for r in ok),
            "needed_all_visible_rate": _mean(float(r.get("needed_visible_max", 0) >= r.get("r", 0)) for r in ok),
            "steps_mean": _mean(r.get("steps") for r in ok),
            "step_cap_rate": _mean(float(r.get("status") == "step_cap") for r in ok),
            "final_step_rate": _mean(float(bool(r.get("used_final_step"))) for r in ok),
            "invalid_actions_mean": _mean(r.get("invalid_actions", 0) for r in ok),
            "tokens_in": sum(r.get("tokens_in", 0) for r in recs), "tokens_out": sum(r.get("tokens_out", 0) for r in recs),
            "tokens_in_per_item": _mean(r.get("tokens_in", 0) for r in recs),
            "tokens_out_per_item": _mean(r.get("tokens_out", 0) for r in recs),
            "calls_per_item": _mean(r.get("n_llm_calls", 0) for r in recs),
            "errors": sum(r.get("status") in ("error", "budget", "does_not_fit") for r in recs)}


def summarize_thrash(records: list[dict], M: int, pinning: bool, model: str = "") -> dict:
    """Errors (status error/budget) are counted, never silently dropped; outcome means are over scored items."""
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(cell_key(r.get("r", 0), M, pinning), []).append(r)
    out = {"system": f"{SYSTEM_NAME}-{'pin' if pinning else 'nopin'}", "M": int(M), "pinning": bool(pinning),
           "model": model, "prompt_version": PROMPT_VERSION, "tuning_iteration": TUNING_ITERATION["pin" if pinning else "nopin"],
           "oracle": False, **_block(records)}
    out["by_cell"] = {k: dict(r=v[0].get("r"), **_block(v)) for k, v in sorted(by.items())}
    return out


H10_EXTRA_COLUMNS = ["n10", "n01", "split", "sd_diff", "n_error_unpinned", "n_error_pinned", "n_budget_unpinned",
                     "n_budget_pinned", "n_not_fit_unpinned", "n_not_fit_pinned", "n_excluded",
                     "prompt_version", "tuning_iteration_unpinned", "tuning_iteration_pinned"]


def _arm(records: list[dict], arm: str) -> dict:
    out = {}
    for r in records:
        k = (r["item_id"], r["M"])
        if k in out:
            raise ValueError(f"duplicate record for item {k[0]} (M={k[1]}) in arm {arm}")
        out[k] = r
    return out


def h10_contrast(unpinned: list[dict], pinned: list[dict], *, split: str, cells: str = "r_gt_M",
                 n_boot: int = 10_000, seed: int = 0, run_ids: list[str] | None = None) -> dict:
    """pinned_vs_unpinned_repeat_pagein_reduction: per-item repeated page-ins, unpinned minus pinned, paired by
    (item_id, M); Wilcoxon signed-rank, scale 1 (stats.contrast). cells="r_gt_M" keeps pairs with r > M (where
    Proposition 15 predicts thrashing), "all" keeps every pair. ``split`` ("dev" | "test") is required.

    Intention-to-treat (draft prereg Section 11): the outcome is repeated page-ins, where LOWER IS BETTER, so an arm
    record with status error / budget / does_not_fit is scored with the WORST value observed: the largest
    repeated_page_ins among the non-failed records of the same (r, M) cell, over both arms and all items. The pair
    is kept. The imputed value per cell is returned in ``imputed_worst`` ({"r<r>-M<M>": value}) and the number of
    imputed records per arm in n_error_*, n_budget_*, n_not_fit_* (unpinned / pinned). A failed pinned record whose
    unpinned partner did not fail never contributes a positive difference (its imputed value is at least the
    partner's); the imputed value is the cell's observed maximum, not the unknown counterfactual.
    Excluded (listed in ``excluded`` as {"item_id", "M", "reason"}, counted in ``n_excluded``): a pair whose item is
    absent from one arm ("missing unpinned" / "missing pinned") and a pair in a cell with no non-failed record
    ("no observed value in cell"). Duplicate (item_id, M) records within an arm raise ValueError. ``sd_diff`` is the
    sample SD (ddof 1) of the per-item differences."""
    import numpy as np
    from ... import stats
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    failed = ("error", "budget", "does_not_fit")
    every = list(unpinned) + list(pinned)
    scored = [r for r in every if r.get("status") not in failed] or every                       # failure records never decide the version (unless nothing was scored)
    versions = {r.get("prompt_version") or "thrash-v1" for r in scored}                        # records before v2 carry none
    if len(versions) != 1:
        raise ValueError(f"h10_contrast mixes prompt versions {sorted(versions)}: both arms must come from the same prompt version")
    tuning = lambda recs: sorted({r.get("tuning_iteration") for r in recs if r.get("status") not in failed} or {None},
                                 key=lambda x: (x is None, x))
    if cells not in ("r_gt_M", "all"):
        raise ValueError("cells must be 'r_gt_M' or 'all'")
    a_by, b_by = _arm(unpinned, "unpinned"), _arm(pinned, "pinned")
    bad = ("error", "budget", "does_not_fit")
    worst: dict[tuple[int, int], int] = {}
    for r in list(a_by.values()) + list(b_by.values()):
        if r.get("status") not in bad:
            c = (int(r["r"]), int(r["M"]))
            worst[c] = max(worst.get(c, 0), int(r["repeated_page_ins"]))

    def cell_of(k):
        r = a_by.get(k) or b_by[k]
        return (int(r["r"]), int(r["M"]))

    allk = sorted(set(a_by) | set(b_by))
    if cells == "r_gt_M":
        allk = [k for k in allk if cell_of(k)[0] > k[1]]
    keys, excluded = [], []
    for k in allk:
        if k not in a_by or k not in b_by:
            excluded.append({"item_id": k[0], "M": k[1], "reason": "missing unpinned" if k not in a_by else "missing pinned"})
        elif cell_of(k) not in worst:
            excluded.append({"item_id": k[0], "M": k[1], "reason": "no observed value in cell"})
        else:
            keys.append(k)
    if not keys:
        raise ValueError("no item is present in both arms with an observed value in its cell")

    def score(r, k):
        return worst[cell_of(k)] if r.get("status") in bad else r["repeated_page_ins"]

    a = [score(a_by[k], k) for k in keys]
    b = [score(b_by[k], k) for k in keys]
    cnt = lambda by, st: sum(by[k].get("status") == st for k in keys)
    out = stats.contrast(a, b, CONTRAST_KEY, test="wilcoxon", scale=1.0, n_boot=n_boot, seed=seed, run_ids=run_ids,
                         split=split)
    tu, tp = tuning(unpinned), tuning(pinned)
    out.update(prompt_version=next(iter(versions)), tuning_iteration_unpinned=tu[0] if len(tu) == 1 else tu,
               tuning_iteration_pinned=tp[0] if len(tp) == 1 else tp)
    out.update(cells=cells, excluded=excluded, n_excluded=len(excluded),
               imputed_worst={f"r{c[0]}-M{c[1]}": worst[c] for c in sorted({cell_of(k) for k in keys})},
               sd_diff=float(np.std(np.asarray(a, float) - np.asarray(b, float), ddof=1)) if len(a) > 1 else float("nan"),
               n_error_unpinned=cnt(a_by, "error"), n_error_pinned=cnt(b_by, "error"),
               n_budget_unpinned=cnt(a_by, "budget"), n_budget_pinned=cnt(b_by, "budget"),
               n_not_fit_unpinned=cnt(a_by, "does_not_fit"), n_not_fit_pinned=cnt(b_by, "does_not_fit"),
               unpinned_mean=sum(a) / len(a), pinned_mean=sum(b) / len(b))
    return out


def write_h10_contrast(path, row: dict):
    """Contrast table with the per-arm count columns (so [[ref:]] tags and claims can reach them)."""
    from ...analysis.tables import write_contrast_table
    return write_contrast_table(path, [row], extra_columns=H10_EXTRA_COLUMNS)


# ---------------------------------------------------------------- entrypoint


@stops_vllm
def run_thrash(args: dict, ctx) -> dict:
    """Job args: items_uri (one URI or a list, e.g. the r2..r6 cells), n_items, offset, M (2 or 3), pinning (bool),
    model ("qwen" default | "gemma" | {"kind": "open_local"|"open", ...} | {"kind": "fake", "chat": ...}; "api" is refused), workers (default 8),
    opts {max_steps | step_mult, step_add; P_pin (512); max_pin_chars (300); max_tokens (300); log_prompts (true)}."""
    M, pinning = int(args["M"]), bool(args.get("pinning", False))
    opts = dict(args.get("opts", {}))
    items = select_items(args, ctx)
    chat = make_chat(args, ctx)
    arm = "pin" if pinning else "nopin"
    ctx.log({"action": "thrash_config", "system": f"{SYSTEM_NAME}-{arm}", "M": M, "pinning": pinning, "model": chat.name,
             "prompt_version": PROMPT_VERSION, "opts": opts, "n_items": len(items)})
    lock = threading.Lock()
    records = []

    def one(item):
        rec = run_thrash_item(item, chat, M, pinning, opts,
                              lambda ev: ctx.log({"system": f"{SYSTEM_NAME}-{arm}", "model": chat.name, **ev}))
        rec["model"] = chat.name
        with lock:
            records.append(rec)
        return rec

    with ThreadPoolExecutor(int(args.get("workers", 8))) as ex:
        list(ex.map(one, items))
    records.sort(key=lambda r: r["item_id"])
    for r in records:
        r["run_id"] = getattr(ctx, "run_id", "")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in records:
            gz.write((json.dumps(r, default=str) + "\n").encode())
    ctx.put_bytes("records.jsonl.gz", buf.getvalue(), "application/gzip")
    metrics = summarize_thrash(records, M, pinning, chat.name)
    ctx.put_json("metrics.json", metrics)
    return {"metrics": metrics}
