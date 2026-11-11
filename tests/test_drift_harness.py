"""Offline tests of the Track A2 session-drift harness (protocol Section 4) with scripted models (no API, no money).

The scripted policies know the session (they are test doubles, not baselines): they identify the session from the
project name in the turn-1 text, and use the generator's ground truth to decide which statements are binding. What
they exercise is the harness: state bookkeeping, the exact-pin rule, supersession, probes, records and metrics.
"""
import gzip
import json
import random
import re

import pytest

from opa import scoring
from opa.data.drift import current_values, make_session, value_text
from opa.tracks.track_a.models import FakeChat
from opa.tracks.track_a.run import LocalCtx
from opa.tracks.track_a2 import estimate_session, job_spec, run_condition
from opa.tracks.track_a2.conditions import (CONDITIONS, DIGEST_SYSTEM, EXTRACT_SYSTEM, NOTES_SYSTEM,
                                            PROBE_DIGEST_SYSTEM, PROBE_PLAIN_SYSTEM, PROBE_REG_SYSTEM,
                                            SUMMARY_SYSTEM, parse_answers, run_session)
from opa.tracks.track_a2.state import (Registers, Shells, epsilon_from_rewrites, survival_curves,
                                       value_present, value_present_norm)

SEEDS = [0, 1, 2, 3, 4, 5]
PRICES = {"modal": {"gpu_per_second": {"L4": 0.000222}, "cpu_core_per_second": 0.0000131,
                    "memory_gib_per_second": 0.00000222}}


@pytest.fixture(scope="module")
def sessions():
    out = [make_session(s) for s in SEEDS]
    assert len({s["project"] for s in out}) == len(out)            # policies identify sessions by project name
    return out


def _first_line(s):
    return s.split("\n", 1)[0].split("{", 1)[0][:80]


def _vt(c, v):
    return value_text(v, c["unit"])


def _quote(text, c, v):
    """Minimal exact span from the constraint's label to the stated value (what a careful extractor pins)."""
    i = text.lower().find(c["label"])
    j = text.find(_vt(c, v), i)
    assert i >= 0 and j >= 0, (c["label"], v, text)
    return text[i:j + len(_vt(c, v))]


class Policy:
    """Scripted model. mode flags:
      lossy_every     D2/D3: every k-th rewrite drops the first static constraint line still present (0 = never)
      corrupt_key     D4-D6: extractor paraphrases this constraint's quote at turn 1 (exact-pin rule must reject it)
      fix_on_retry    D4-D6: the extractor copies the quote correctly when re-asked
      digest_trust    D6: probability the reader cites a digest that states a constraint (gives the digest's value)
    """

    def __init__(self, sessions, lossy_every=0, corrupt_key=None, fix_on_retry=True, digest_trust=0.0, seed=0):
        self.by_project = {s["project"]: s for s in sessions}
        self.cur = None
        self.lossy_every, self.corrupt_key, self.fix_on_retry = lossy_every, corrupt_key, fix_on_retry
        self.digest_trust = digest_trust
        self.rng = random.Random(seed)

    def __call__(self, system, user):
        m = re.search(r"Kick-off for the (\w+) pump", user)
        if m and m.group(1) in self.by_project:
            self.cur = self.by_project[m.group(1)]
        s = self.cur
        head = _first_line(system)
        if head == _first_line(SUMMARY_SYSTEM):
            return self.rewrite(s, user, "Current summary:\n")
        if head == _first_line(NOTES_SYSTEM):
            n = int(re.search(r"\[turn (\d+) \|", user).group(1))
            if not self.binding(s, n) and not (self.lossy_every and n % self.lossy_every == 0):
                return json.dumps({"update": False})
            return json.dumps({"update": True, "notes": self.rewrite(s, user, "Current memory notes:\n")})
        if head == _first_line(EXTRACT_SYSTEM):
            return self.extract(s, user)
        if head == _first_line(DIGEST_SYSTEM):
            return "Digest: " + " ".join(re.sub(r"^\[turn \d+ \| thread \w+\] ", "", ln) for ln in user.split("\n"))[:700]
        if system == PROBE_PLAIN_SYSTEM:
            return self.probe_plain(s, user)
        if system in (PROBE_REG_SYSTEM, PROBE_DIGEST_SYSTEM):
            return self.probe_reg(s, user)
        raise AssertionError("unexpected prompt: " + head)

    # binding statements at a turn: [(constraint, value)]
    @staticmethod
    def binding(s, n):
        out = [(c, c["initial"]["value"]) for c in s["constraints"]] if n == 1 else []
        out += [(c, u["value"]) for c in s["constraints"] for u in c["updates"] if u["turn"] == n]
        return out

    def rewrite(self, s, user, marker):
        prev = user.split(marker, 1)[1].split("\n\nNew turn:\n", 1)[0]
        lines = dict(ln.split(" = ", 1) for ln in prev.split("\n") if " = " in ln)
        n = int(re.search(r"\[turn (\d+) \|", user).group(1))
        for c, v in self.binding(s, n):
            lines[c["key"]] = _vt(c, v)
        if self.lossy_every and n > 1 and n % self.lossy_every == 0:
            static = [c["key"] for c in s["constraints"] if not c["updates"] and c["key"] in lines]
            if static:
                del lines[static[0]]
        return "\n".join(f"{k} = {v}" for k, v in lines.items())

    def probe_plain(self, s, user):
        keys = re.findall(r"^- (\w+): ", user, re.M)
        if user.startswith("Session history"):
            last = max(int(x) for x in re.findall(r"\[turn (\d+) \|", user))
            cur = current_values(s["constraints"], last)
            units = {c["key"]: c["unit"] for c in s["constraints"]}
            return json.dumps({"answers": {k: value_text(cur[k], units[k]) for k in keys}})
        lines = dict(ln.split(" = ", 1) for ln in user.split("\n") if " = " in ln)
        return json.dumps({"answers": {k: lines.get(k) for k in keys}})

    @staticmethod
    def registers(user):
        return json.loads(re.search(r"Registers:\n(\{.*?\n\})", user, re.S).group(1))

    def extract(self, s, user):
        reg = self.registers(user)
        m = re.search(r"New turn \(origin_id (\d+), thread (\w+)\):\n(.*?)(?:\n\nThe harness rejected|$)", user, re.S)
        n, text = int(m.group(1)), m.group(3)
        retry = "The harness rejected" in user
        ops = []
        if n == 1 and not retry:
            ops.append({"op": "goal", "quote": text.split(" Baseline")[0]})
        for c, v in self.binding(s, n):
            q = _quote(text, c, v)
            if c["key"] == self.corrupt_key and (not retry or not self.fix_on_retry):
                q = q.replace(" is ", " equals ")                     # a paraphrase, not a copy
            if retry and c["key"] != self.corrupt_key:
                continue
            op = {"op": "add", "key": c["key"], "value": v, "unit": c["unit"], "quote": q}
            if n > 1:
                live = [e for e in reg["constraints"] if e["key"] == c["key"] and e["superseded_by"] is None]
                op.update(op="update", replaces=live[-1]["pin_id"] if live else None)
            ops.append(op)
        return json.dumps({"ops": ops}, ensure_ascii=False)

    def probe_reg(self, s, user):
        reg = self.registers(user)
        keys = re.findall(r"^- (\w+): ", user, re.M)
        nodes = re.findall(r'<node id="([^"]+)"[^>]*>\n(.*?)\n</node>', user, re.S)
        ans = {}
        for k in keys:
            c = next(x for x in s["constraints"] if x["key"] == k)
            # a fair reader: uniform choice among entries that are not marked superseded
            live = [e for e in reg["constraints"] if e["key"] == k and e["superseded_by"] is None]
            pick = self.rng.choice(live) if live else None
            ans[k] = {"value": pick["value"], "cite": pick["pin_id"]} if pick else {"value": None, "cite": None}
            if self.digest_trust:
                for nid, dg in nodes:
                    stated = f"{c['label']} is {_vt(c, c['initial']['value'])}"
                    if stated in dg and self.rng.random() < self.digest_trust:
                        ans[k] = {"value": _vt(c, c["initial"]["value"]), "cite": nid}
                        break
        return json.dumps({"answers": ans}, ensure_ascii=False)


def run(cond, sessions, out, **pol):
    ctx = LocalCtx(str(out), run_id=f"test-{cond}")
    res = run_condition({"items_uri": "unused", "condition": cond, "workers": 1,
                         "model": {"kind": "fake", "chat": FakeChat(Policy(sessions, **pol))}}, _Ctx(ctx, sessions))
    with gzip.open(out / "records.jsonl.gz", "rt", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return res["metrics"], recs, ctx.events


class _Ctx:
    """LocalCtx whose items come from memory (run_condition calls select_items(args, ctx) -> load_items)."""

    def __init__(self, inner, sessions):
        self.inner, self.sessions = inner, sessions
        self.run_id, self.spec = inner.run_id, inner.spec

    def log(self, ev):
        self.inner.log(ev)

    def put_json(self, *a):
        return self.inner.put_json(*a)

    def put_bytes(self, *a):
        return self.inner.put_bytes(*a)


@pytest.fixture(autouse=True)
def _mem_items(monkeypatch, sessions):
    import opa.tracks.track_a2.run as r2
    monkeypatch.setattr(r2, "select_items", lambda args, ctx: list(getattr(ctx, "sessions", sessions)))


# ---------------------------------------------------------------- unit checks
def test_value_present_boundaries():
    assert value_present("46", "max_coolant_temp = 46 °C")
    assert not value_present("46", "146 °C") and not value_present("46", "46.5 °C")
    assert value_present("±0.10", "tolerance ±0.10 mm.") and value_present("2027-04-04", "on 2027-04-04.")
    assert not value_present("±0.10", "tolerance 0.10 mm")             # exact: the ± sign was lost
    assert value_present_norm("±0.10", "mm", "tolerance 0.10 mm")      # lenient
    assert value_present_norm("83", "°C", "limit 83°C")


def test_epsilon_known_value():
    cons = [{"key": "a", "updates": []}, {"key": "b", "updates": []}, {"key": "c", "updates": [{"turn": 3, "value": "x"}]}]
    rw = [{"turn": 1, "present": {"exact": ["a", "b", "c"]}}, {"turn": 2, "present": {"exact": ["a", "b", "c"]}},
          {"turn": 3, "present": {"exact": ["a"]}}, {"turn": 4, "present": {"exact": ["a", "c"]}}]
    e = epsilon_from_rewrites(cons, rw)
    assert (e["opportunities"], e["losses"]) == (6, 1) and e["epsilon"] == pytest.approx(1 / 6)
    sc = survival_curves(cons, rw)
    assert sc["present"] == [1.0, 1.0, 0.5, 0.5] and sc["continuous"] == [1.0, 1.0, 0.5, 0.5]


def test_exact_pin_rule_and_supersession():
    text = "Update to the spec: the bore diameter tolerance is now ±0.05 mm (it was ±0.10 mm)."
    for sup in (True, False):
        r = Registers(supersede=sup)
        acc, rej = r.apply([{"op": "add", "key": "bore", "value": "±0.10", "unit": "mm",
                             "quote": "bore diameter tolerance is ±0.10 mm"}], 1, "design",
                           "(7) The bore diameter tolerance is ±0.10 mm.")
        assert len(acc) == 1 and not rej
        acc, rej = r.apply([
            {"op": "update", "replaces": "p1", "key": "bore", "value": "±0.05", "quote": "bore diameter tolerance is now ±0.05 mm"},
            {"op": "update", "replaces": "p1", "key": "bore", "value": "±0.05", "quote": "bore tolerance is now ±0.05 mm"},
            {"op": "add", "key": "x", "value": "±0.07", "quote": "bore diameter tolerance is now ±0.05 mm"},
            {"op": "frobnicate", "quote": "Update"}], 9, "testing", text)
        assert [x["reason"] for x in rej] == ["quote_not_in_origin", "value_not_in_quote", "unknown_op"]
        assert len(acc) == 1 and r.constraints[1]["origin_id"] == 9 and r.constraints[1]["quote"] in text
        assert r.constraints[0]["superseded_by"] == ("p2" if sup else None)
        assert len(r.live()) == (1 if sup else 2)
    # whitespace runs count as one space (find_exact), nothing else is normalized
    r = Registers()
    acc, _ = r.apply([{"op": "add", "key": "k", "value": "80", "quote": "is  80 °C"}], 1, "design", "The limit is 80 °C.")
    assert acc and r.constraints[0]["quote"] == "is 80 °C"


def test_shells_close_every_b_turns():
    sh = Shells(b=3)
    closed = [sh.add_turn(th, t) for t, th in enumerate(["a", "b", "a", "a", "b", "a", "a", "a"], 1)]
    ids = [c["node_id"] for c in closed if c]
    assert ids == ["n-a-0", "n-a-1"] and sh.nodes[0]["children"] == [1, 3, 4]


def test_parse_answers_by_label_and_cites():
    keys = [("max_coolant_temp", "maximum coolant temperature"), ("seal_supplier", "seal supplier")]
    pred, cites = parse_answers({"answers": {"maximum coolant temperature": {"value": "80 °C", "cite": "p3"},
                                             "seal_supplier": {"value": "null", "cite": None}}}, keys)
    assert pred == {"max_coolant_temp": "80 °C", "seal_supplier": None} and cites["max_coolant_temp"] == "p3"
    pred, cites = parse_answers({"answers": {"max_coolant_temp": 80}}, keys)
    assert pred["max_coolant_temp"] == "80" and cites is None


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("cond", CONDITIONS)
def test_every_condition_runs_end_to_end(cond, sessions, tmp_path):
    metrics, recs, events = run(cond, sessions[:2], tmp_path)
    assert len(recs) == 2 and all(r["status"] == "ok" for r in recs), [r.get("error") for r in recs]
    for r in recs:
        assert [p["turn"] for p in r["probes"]] == [10, 25, 50] and r["condition"] == cond
        assert r["run_id"] == f"test-{cond}"
    # the stored record is exactly what opa.scoring scores
    rows = [row for r, s in zip(recs, sessions) for row in scoring.score_drift_session(s, r)]
    assert len(rows) == 6
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["condition"] == cond
    assert all(e["condition"] == cond and "action" in e and "step" in e for e in events if e["action"] != "session_error")
    calls = {"D1": 3, "D2": 53, "D3": 53, "D4": 53, "D5": 53}
    if cond in calls:
        assert all(r["n_llm_calls"] == calls[cond] for r in recs)
    else:
        assert all(r["n_llm_calls"] == 53 + len(r["digests"]) for r in recs)
    assert sum(e["action"] == "probe" for e in events) == 6


@pytest.mark.parametrize("cond", ["D1", "D2", "D3", "D4", "D6"])
def test_correct_state_keeping_reaches_100(cond, sessions, tmp_path):
    m, recs, _ = run(cond, sessions, tmp_path)
    for t in (10, 25, 50):
        assert m["per_probe"][str(t)]["accuracy"] == 1.0, (cond, t, m["per_probe"][str(t)])
        assert m["per_probe"][str(t)]["stale_rate"] == 0.0


def test_d4_registers_and_events(sessions, tmp_path):
    m, recs, events = run("D4", sessions, tmp_path)
    assert m["registers"]["n_pins_rejected"] == 0
    assert m["registers"]["mean_superseded"] == 3.0                   # three updates per session
    for r, s in zip(recs, sessions):
        live = {c["key"]: c for c in r["registers"]["constraints"] if c["superseded_by"] is None}
        assert {k: v["value"] for k, v in live.items()} == s["probes"][-1]["gold"]
        for c in r["registers"]["constraints"]:                       # every value is a pinned exact span of its turn
            turn = next(t for t in s["turns"] if t["turn"] == c["origin_id"])
            assert turn["text"][c["char_span"][0]:c["char_span"][1]] == c["quote"] and c["value"] in c["quote"]
        assert all(k == "register" for p in r["probes"] for k in p["cite_kind"].values())
    acts = {e["action"] for e in events}
    assert {"extract", "pin", "supersede", "probe", "answer"} <= acts
    assert all("prompt" in e and "reply" in e for e in events if e["action"] == "extract")


def test_d5_shows_stale_values_where_d4_does_not(sessions, tmp_path):
    """Same extractor, same reader for both: the reader picks uniformly among entries not marked superseded. With
    links (D4) there is one such entry per key; without them (D5) the old and new entries both qualify."""
    m4, _, _ = run("D4", sessions, tmp_path / "d4", seed=7)
    m5, r5, _ = run("D5", sessions, tmp_path / "d5", seed=7)
    assert m4["stale_rate_t50"] == 0.0 and m4["accuracy_t50"] == 1.0
    assert m5["stale_rate_t50"] > 0.0 and m5["accuracy_t50"] < 1.0
    assert m5["registers"]["mean_superseded"] == 0.0
    assert m5["accuracy_by_kind"]["50"]["static"] == 1.0                  # only updated constraints go stale
    assert m5["accuracy_by_kind"]["50"]["updated"] < 1.0
    assert m5["per_probe"]["10"]["stale_rate"] <= m5["per_probe"]["50"]["stale_rate"]


def test_exact_pin_rejection_end_to_end(sessions, tmp_path):
    s = sessions[0]
    key = next(c["key"] for c in s["constraints"] if not c["updates"])
    # rejected at turn 1, fixed on the retry: nothing is lost
    m, recs, ev = run("D4", [s], tmp_path / "fix", corrupt_key=key, fix_on_retry=True)
    assert m["registers"]["pins_rejected"] == {"quote_not_in_origin": 1} and m["accuracy_t50"] == 1.0
    assert sum(e["action"] == "extract_retry" for e in ev) == 1
    assert recs[0]["n_llm_calls"] == 54
    # never fixed: rejected twice (turn 1 and its retry), the constraint is absent from the registers -> missing
    m, recs, ev = run("D4", [s], tmp_path / "nofix", corrupt_key=key, fix_on_retry=False)
    assert m["registers"]["pins_rejected"] == {"quote_not_in_origin": 2}
    assert key not in {c["key"] for c in recs[0]["registers"]["constraints"]}
    oc = scoring.drift_probe_outcomes(s, recs[0], 50)
    assert oc[key] == "missing" and sum(v == "correct" for v in oc.values()) == 9
    rej = [e for e in ev if e["action"] == "pin_rejected"]
    assert len(rej) == 2 and all(e["reason"] == "quote_not_in_origin" and " equals " in e["quote"] for e in rej)


def test_d2_epsilon(sessions, tmp_path):
    m, recs, ev = run("D2", sessions, tmp_path / "perfect")
    assert m["d2"]["epsilon"] == 0.0 and m["d2"]["exact"]["opportunities"] > 0
    assert all(len(r["rewrites"]) == 50 for r in recs)
    assert sum(e["action"] == "summary_rewrite" for e in ev) == 50 * len(sessions)
    m, recs, _ = run("D2", sessions, tmp_path / "lossy", lossy_every=10)
    d2 = m["d2"]
    # every 10th rewrite (turns 10..50) drops one static constraint: 5 losses per session at most
    assert 0 < d2["exact"]["losses"] <= 5 * len(sessions)
    assert d2["epsilon"] == pytest.approx(d2["exact"]["losses"] / d2["exact"]["opportunities"])
    surv = d2["exact"]["survival"]
    assert len(surv["observed_present"]) == 50 and surv["observed_present"][0] == 1.0
    assert surv["observed_continuous"][-1] < 1.0 and surv["predicted"][0] == 1.0
    assert surv["predicted"][-1] == pytest.approx((1 - d2["epsilon"]) ** 49)
    assert m["accuracy_t50"] < 1.0 and m["per_probe"]["50"]["missing_rate"] > 0
    # the per-rewrite presence check is a string check against the current gold value
    s, r = sessions[0], recs[0]
    rw = next(x for x in r["rewrites"] if x["turn"] == 50)
    assert set(rw["present"]["exact"]) <= set(s["probes"][-1]["gold"])


def test_d3_notes_update_only_when_chosen(sessions, tmp_path):
    m, recs, ev = run("D3", sessions[:2], tmp_path)
    for r, s in zip(recs, sessions):
        upd_turns = sorted({1} | {u["turn"] for c in s["constraints"] for u in c["updates"]})
        assert r["notes_updates"] == upd_turns
    assert m["d3_update_rate"] == pytest.approx(4 / 50)
    assert sum(e["action"] == "notes_keep" for e in ev) == 2 * 46


def test_d6_digest_citations(sessions, tmp_path):
    m, recs, ev = run("D6", sessions, tmp_path, digest_trust=0.5, seed=3)
    assert all(len(r["digests"]) >= 3 for r in recs) and m["mean_digests"] >= 3
    kinds = {k for r in recs for p in r["probes"] for k in p["cite_kind"].values()}
    assert {"register", "digest"} <= kinds
    assert 0 < m["digest_citation_share"] < 1
    assert m["digest_citation_share_by_probe"]["10"] == 0.0             # no node has closed by turn 10
    # digests restate turn-1 values, so every wrong answer here comes from a digest citation
    assert m["wrong_digest_citation_share"] == 1.0 and m["stale_rate_t50"] > 0
    assert sum(e["action"] == "digest" for e in ev) == sum(len(r["digests"]) for r in recs)


def test_d1_context_limit(sessions, tmp_path):
    ctx = LocalCtx(str(tmp_path))
    rec = run_session("D1", sessions[0], FakeChat(Policy(sessions)), {"context_limit": 600}, ctx.log)
    fits = [p["fits"] for p in rec["probes"]]
    assert fits == [True, False, False] and rec["n_llm_calls"] == 1
    assert any(e["action"] == "probe_skipped" for e in ctx.events)


def test_error_in_one_session_does_not_kill_batch(sessions, tmp_path):
    pol = Policy(sessions)

    def bad(system, user):
        if sessions[1]["project"] in user:
            raise RuntimeError("boom")
        return pol(system, user)
    ctx = LocalCtx(str(tmp_path), run_id="t")
    res = run_condition({"items_uri": "unused", "condition": "D4", "workers": 1, "model": {"kind": "fake", "chat": FakeChat(bad)}},
                        _Ctx(ctx, sessions[:3]))
    assert res["metrics"]["n_error"] == 1 and res["metrics"]["n_sessions"] == 3


# ---------------------------------------------------------------- budgeting
def test_estimate_and_job_spec():
    calls = {c: estimate_session(c)["typical"]["calls"] for c in CONDITIONS}
    assert calls == {"D1": 3, "D2": 53, "D3": 53, "D4": 53, "D5": 53, "D6": 59}
    for c in CONDITIONS:
        e = estimate_session(c)
        assert e["worst"]["tokens_in"] >= e["typical"]["tokens_in"] and e["worst"]["tokens_out"] >= e["typical"]["tokens_out"]
    spec = job_spec("p2-drift-d2", "D2", "gs://{bucket}/opa/data/drift/dev/v1/s50/items.jsonl.gz", n_items=200,
                    envelope="E2", prices=PRICES)
    assert spec["entrypoint"] == "opa.tracks.track_a2:run_condition" and spec["args"]["condition"] == "D2"
    assert spec["track"] == "A2" and spec["hypotheses"] == ["H2"]
    assert 0 < spec["estimate_usd"] < spec["max_usd"] and spec["max_api_usd"] == 0
    assert spec["compute"]["gpu"] == "L4" and spec["models"]["subject"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "sessions x $" in spec["estimate_formula"]
    with pytest.raises(Exception):
        job_spec("x", "D9", "u", prices=PRICES)
