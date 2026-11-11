"""Track A2 (session drift, H2, protocol Section 4) job entrypoint and budgeting helpers.

  opa.tracks.track_a2:run_condition   one condition D1..D6 on a batch of drift sessions

Job args (run_condition):
  items_uri    gs://.../drift/<split>/v1/s50/items.jsonl.gz or a local path, or a list ("{bucket}" is replaced by
               the job's bucket; same helper as Track A)
  n_items      sessions to run (default all), ``offset`` (default 0)
  condition    D1 | D2 | D3 | D4 | D5 | D6
  model        v3, open models only: "qwen" (default, Qwen3-4B-Instruct-2507 on a local vLLM server) | "gemma" |
               {"kind": "open_local", "name", "max_model_len"} | {"kind": "open", "base_url", "name"} |
               {"kind": "fake", "chat": FakeChat(...)} (tests); {"kind": "api"} is refused (see track_a.run.make_chat)
  workers      sessions in parallel (default 8); each session is processed sequentially, turn by turn
  opts         see conditions.DEFAULT_OPTS (summary_tokens 1000, notes_tokens 1000, digest_b 8, context_limit, ...)

Outputs (run folder): records.jsonl.gz (one drift record per session, the format opa.scoring scores, plus the
condition's state traces) and metrics.json (accuracy / stale / fabricated / near-miss / missing / other rates per
probe turn; static vs updated constraints; epsilon and survival curves for D2; cite kinds and the digest-citation
share for D6; calls and tokens per session).

``job_spec`` builds (does not write) a vLLM-image GPU job spec; estimate = GPU hours x rate, hours = start-up +
sessions x (calls x sec_per_call + prefill tokens / prefill_tok_per_s + out tokens / out_tok_per_s) / workers, with
token counts from ``estimate_session`` and PLACEHOLDER throughputs until the P0 micro-pilot.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from ... import scoring
from ..track_a.run import make_chat, select_items, stops_vllm
from .conditions import (CONDITIONS, DEFAULT_OPTS, DIGEST_SYSTEM, EXTRACT_SYSTEM, NOTES_SYSTEM, PROBE_DIGEST_SYSTEM,
                         PROBE_PLAIN_SYSTEM, PROBE_REG_SYSTEM, PROMPT_VERSION, SUMMARY_SYSTEM, run_session)
from .state import epsilon_from_rewrites, survival_curves


@stops_vllm
def run_condition(args: dict, ctx) -> dict:
    condition = str(args["condition"]).upper()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    opts = {**DEFAULT_OPTS, **dict(args.get("opts", {}))}
    sessions = select_items(args, ctx)
    bad = [s.get("item_id") for s in sessions if s.get("dataset") != "drift"]
    if bad:
        raise ValueError(f"not drift sessions: {bad[:3]}")
    chat = make_chat(args, ctx)
    lock = threading.Lock()
    records: list[dict] = []
    stop: dict = {}

    def log(ev):
        ctx.log(ev)

    def one(sess):
        if stop:
            rec = {"item_id": sess["item_id"], "condition": condition, "model": chat.name, "probes": [],
                   "status": "budget", "error": "not run: budget stop"}
        else:
            try:
                rec = run_session(condition, sess, chat, opts, log)
            except Exception as e:                       # one failed session must not kill the batch
                if type(e).__name__ == "BudgetExceeded":
                    stop["exc"] = e
                    status = "budget"
                else:
                    status = "error"
                rec = {"item_id": sess["item_id"], "condition": condition, "model": chat.name, "probes": [],
                       "status": status, "error": repr(e)[:2000]}
                ctx.log({"item_id": sess["item_id"], "condition": condition, "action": "session_error",
                         "status": status, "error": repr(e)[:2000]})
        with lock:
            records.append(rec)

    with ThreadPoolExecutor(int(args.get("workers", 8))) as ex:
        list(ex.map(one, sessions))
    records.sort(key=lambda r: r["item_id"])
    for r in records:
        r["run_id"] = getattr(ctx, "run_id", "")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in records:
            gz.write((json.dumps(r, default=str, ensure_ascii=False) + "\n").encode())
    ctx.put_bytes("records.jsonl.gz", buf.getvalue(), "application/gzip")
    metrics = summarize(sessions, records, condition)
    metrics.update(model=chat.name, prompt_version=PROMPT_VERSION, opts={k: v for k, v in opts.items()})
    ctx.put_json("metrics.json", _clean(metrics))
    if stop:
        raise stop["exc"]                               # outputs are written; the dispatcher records stopped_budget
    return {"metrics": _clean(metrics)}


# ---------------------------------------------------------------- metrics
def _nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def summarize(sessions: list[dict], records: list[dict], condition: str) -> dict:
    """Metrics for one condition. Every record counts (an errored session's probes score as missing)."""
    by_id = {s["item_id"]: s for s in sessions}
    n = len(records)
    agg = scoring.aggregate_drift(sessions, records).get(condition, {})
    m: dict = {"condition": condition, "track": "A2", "n_sessions": n,
               "n_error": sum(r.get("status") != "ok" for r in records),
               "per_probe": agg,
               "accuracy_t50": agg.get(50, {}).get("accuracy", math.nan),
               "stale_rate_t50": agg.get(50, {}).get("stale_rate", math.nan),
               "fabricated_rate_t50": agg.get(50, {}).get("fabricated_rate", math.nan)}
    # static vs updated constraints, per probe
    split: dict = defaultdict(lambda: defaultdict(list))
    cite_counts: dict = defaultdict(Counter)
    wrong_cite: dict = defaultdict(Counter)
    for r in records:
        s = by_id[r["item_id"]]
        for p in s["probes"]:
            t = p["turn"]
            oc = scoring.drift_probe_outcomes(s, r, t)
            rp = next((q for q in r.get("probes", []) if q.get("turn") == t), {})
            kinds = rp.get("cite_kind") or {}
            for c in s["constraints"]:
                upd = any(u["turn"] <= t for u in c.get("updates", []))
                split[t]["updated" if upd else "static"].append(float(oc[c["key"]] == "correct"))
                if "cite_kind" in rp and oc[c["key"]] != "missing":
                    ck = kinds.get(c["key"], "none")
                    cite_counts[t][ck] += 1
                    if oc[c["key"]] != "correct":
                        wrong_cite[t][ck] += 1
    m["accuracy_by_kind"] = {t: {k: _nanmean(v) for k, v in d.items()} | {f"n_{k}": len(v) for k, v in d.items()}
                             for t, d in sorted(split.items())}
    # resource use
    ok = [r for r in records if r.get("status") == "ok"]
    m.update(mean_calls=_nanmean([r.get("n_llm_calls", 0) for r in ok]),
             mean_tokens_in=_nanmean([r.get("tokens_in", 0) for r in ok]),
             mean_tokens_out=_nanmean([r.get("tokens_out", 0) for r in ok]),
             tokens_in=sum(r.get("tokens_in", 0) for r in records),
             tokens_out=sum(r.get("tokens_out", 0) for r in records),
             calls=sum(r.get("n_llm_calls", 0) for r in records),
             mean_wall_ms=_nanmean([r.get("wall_ms", 0) for r in ok]))
    m["probes_not_fit"] = sum(1 for r in records for p in r.get("probes", []) if p.get("fits") is False)
    if condition == "D2":
        m["d2"] = d2_metrics(by_id, ok)
    if condition == "D3":
        m["d3_update_rate"] = _nanmean([len(r.get("notes_updates", [])) / max(1, len(by_id[r["item_id"]]["turns"]))
                                        for r in ok])
    if condition in ("D4", "D5", "D6"):
        rej = Counter(x["reason"] for r in ok for x in r.get("pins_rejected", []))
        m["registers"] = {"mean_pins_accepted": _nanmean([r.get("pins_accepted", 0) for r in ok]),
                          "pins_rejected": dict(rej), "n_pins_rejected": sum(rej.values()),
                          "mean_superseded": _nanmean([sum(c.get("superseded_by") is not None
                                                           for c in (r.get("registers") or {}).get("constraints", []))
                                                       for r in ok])}
        m["cite_kinds"] = {t: dict(c) for t, c in sorted(cite_counts.items())}
        m["wrong_cite_kinds"] = {t: dict(c) for t, c in sorted(wrong_cite.items())}
        tot = sum(sum(c.values()) for c in cite_counts.values())
        wtot = sum(sum(c.values()) for c in wrong_cite.values())
        m["digest_citation_share"] = (sum(c["digest"] for c in cite_counts.values()) / tot) if tot else math.nan
        m["wrong_digest_citation_share"] = (sum(c["digest"] for c in wrong_cite.values()) / wtot) if wtot else math.nan
        m["digest_citation_share_by_probe"] = {t: (c["digest"] / sum(c.values())) if sum(c.values()) else math.nan
                                               for t, c in sorted(cite_counts.items())}
        if condition == "D6":
            m["mean_digests"] = _nanmean([len(r.get("digests", [])) for r in ok])
    return m


def d2_metrics(by_id: dict, ok: list[dict]) -> dict:
    """Per-rewrite loss epsilon (pooled and per session), and observed vs predicted survival of the static turn-1
    constraints: predicted(T) = (1 - eps)^T with T = rewrites after the first (protocol 4.3)."""
    out = {}
    for field in ("exact", "norm"):
        opp = lost = 0
        per = []
        curves = []
        for r in ok:
            cons = by_id[r["item_id"]]["constraints"]
            e = epsilon_from_rewrites(cons, r.get("rewrites", []), field)
            opp, lost = opp + e["opportunities"], lost + e["losses"]
            per.append(e["epsilon"])
            curves.append(survival_curves(cons, r.get("rewrites", []), field))
        eps = lost / opp if opp else math.nan
        L = max((len(c["turns"]) for c in curves), default=0)
        obs = [_nanmean([c["present"][i] for c in curves if len(c["present"]) > i]) for i in range(L)]
        cont = [_nanmean([c["continuous"][i] for c in curves if len(c["continuous"]) > i]) for i in range(L)]
        turns = next((c["turns"] for c in curves if len(c["turns"]) == L), list(range(1, L + 1)))
        out[field] = {"epsilon": eps, "opportunities": opp, "losses": lost,
                      "epsilon_session_mean": _nanmean(per),
                      "survival": {"turns": turns, "observed_present": obs, "observed_continuous": cont,
                                   "predicted": [(1 - eps) ** i if not math.isnan(eps) else math.nan for i in range(L)]}}
    out["epsilon"] = out["exact"]["epsilon"]
    out["mean_summary_tokens"] = _nanmean([rw["summary_tokens"] for r in ok for rw in r.get("rewrites", [])])
    return out


# ---------------------------------------------------------------- budgeting
# Measured on the 200 dev sessions (drift/dev/v1/s50): turn 1 ~528 chars, later turns ~99 chars (max 208).
TURN1_TOK, TURN_TOK, LINE_TOK = 132, 25, 8      # chars/4; LINE_TOK = "[turn n | thread x] " header
N_TURNS, PROBE_TURNS, N_CONSTRAINTS, N_UPDATES = 50, (10, 25, 50), 10, 3
KEYLIST_TOK = 10 * 12 + 30                       # "- key: label" lines + probe header
REG_ENTRY_TOK = 70                               # one rendered constraint register entry (JSON, indent 1)
ADD_OP_TOK, UPDATE_OP_TOK, EMPTY_OPS_TOK = 55, 70, 8


def _p(s: str) -> int:
    return len(s) // 4


def estimate_session(condition: str, opts: dict | None = None, typical_state_tok: int = 400,
                     d3_update_rate: float = 0.3, digest_out_tok: int = 120) -> dict:
    """Calls and tokens for one 50-turn session, typical and worst case. Arithmetic (tokens = chars / 4):

    history(t)   = 132 + (t-1) * 25 + t * 8                       (D1 state at probe t; 445 / 970 / 1845 tokens)
    probe        = system prompt + state + 150 (key list); out 170 (plain) or 270 (with cites)
    D2 rewrite   = SUMMARY_SYSTEM + S + 33 (turn) in, S out; S = 400 typical, summary_tokens (1000) worst
    D3 decision  = NOTES_SYSTEM + N + 33 in; out N + 20 on an update (rate 0.3 typical, 1.0 worst), else 10
    D4 extract   = EXTRACT_SYSTEM + registers + 33 in; registers ~70 tokens per entry (0 at turn 1, ~11.5 later);
                   out 10 x 55 at turn 1, 70 per update turn, 8 otherwise; worst = x1.5 (pin / JSON retries)
    D6           = D4 + 6 digests (8 turns x 33 + DIGEST_SYSTEM in, 120 out typical, digest_tokens worst) and the
                   digests (6 x out) in each probe state
    """
    o = {**DEFAULT_OPTS, **(opts or {})}
    cond = condition.upper()
    hist = {t: TURN1_TOK + (t - 1) * TURN_TOK + t * LINE_TOK for t in PROBE_TURNS}
    tturn = TURN_TOK + LINE_TOK

    def probe(system: str, states: list[int], out_tok: int):
        return len(states), sum(_p(system) + s + KEYLIST_TOK for s in states), len(states) * out_tok

    res = {}
    for case in ("typical", "worst"):
        worst = case == "worst"
        if cond == "D1":
            c, i, u = probe(PROBE_PLAIN_SYSTEM, list(hist.values()), 170)
        elif cond == "D2":
            S = int(o["summary_tokens"]) if worst else typical_state_tok
            sys_t = _p(SUMMARY_SYSTEM)
            c, i, u = N_TURNS, N_TURNS * (sys_t + S + tturn), N_TURNS * S
            pc, pi, pu = probe(PROBE_PLAIN_SYSTEM, [S] * 3, 170)
            c, i, u = c + pc, i + pi, u + pu
        elif cond == "D3":
            Nn = int(o["notes_tokens"]) if worst else typical_state_tok
            rate = 1.0 if worst else d3_update_rate
            c, i = N_TURNS, N_TURNS * (_p(NOTES_SYSTEM) + Nn + tturn)
            u = int(N_TURNS * (rate * (Nn + 20) + (1 - rate) * 10))
            pc, pi, pu = probe(PROBE_PLAIN_SYSTEM, [Nn] * 3, 170)
            c, i, u = c + pc, i + pi, u + pu
        elif cond in ("D4", "D5", "D6"):
            reg_late = int(11.5 * REG_ENTRY_TOK) + 50
            c = N_TURNS
            i = (_p(EXTRACT_SYSTEM) + 20 + TURN1_TOK) + (N_TURNS - 1) * (_p(EXTRACT_SYSTEM) + reg_late + tturn)
            u = N_CONSTRAINTS * ADD_OP_TOK + N_UPDATES * UPDATE_OP_TOK + (N_TURNS - 1 - N_UPDATES) * EMPTY_OPS_TOK
            if worst:
                c, i, u = int(c * 1.5), int(i * 1.5), int(u * 1.5)
            psys, dig = PROBE_REG_SYSTEM, 0
            if cond == "D6":
                n_nodes = 6
                dout = int(o["digest_tokens"]) if worst else digest_out_tok
                c += n_nodes
                i += n_nodes * (_p(DIGEST_SYSTEM) + int(o["digest_b"]) * tturn)
                u += n_nodes * dout
                psys, dig = PROBE_DIGEST_SYSTEM, n_nodes * (dout + 15)
            pc, pi, pu = probe(psys, [reg_late + dig] * 3, 270)
            c, i, u = c + pc, i + pi, u + pu
        else:
            raise ValueError(cond)
        res[case] = {"calls": int(c), "tokens_in": int(i), "tokens_out": int(u)}
    return {"condition": cond, **res}


# PLACEHOLDER throughput constants (not measured). Replace with the P0 micro-pilot's measured values
# (state/unit_costs.md) before any paid A2 job is requested; job_spec returns them in "throughput_placeholders".
PLACEHOLDER_THROUGHPUT = {"sec_per_call": 1.0, "out_tok_per_s": 80.0, "prefill_tok_per_s": 6000.0, "startup_h": 0.1}


def session_gpu_hours(est: dict, case: str = "typical", workers: int = 8, sec_per_call: float = 1.0,
                      out_tok_per_s: float = 80.0, prefill_tok_per_s: float = 6000.0) -> float:
    """Wall-clock GPU hours of one session's share of a batch:
    (calls x sec_per_call + prefill tokens / prefill_tok_per_s + out tokens / out_tok_per_s) / workers.
    Placeholders until the P0 micro-pilot measures real unit costs (state/unit_costs.md)."""
    e = est[case]
    return (e["calls"] * sec_per_call + e["tokens_in"] / prefill_tok_per_s + e["tokens_out"] / out_tok_per_s) / max(1, workers) / 3600


def _load_prices() -> dict:
    import yaml
    from ... import PROJECT_DIR
    return yaml.safe_load((PROJECT_DIR / "budget" / "prices.yaml").read_text(encoding="utf-8"))


def job_spec(name: str, condition: str, items_uri, *, n_items: int = 200, envelope: str = "E2",
             model: str = "qwen", split: str = "dev", cell: str = "s50", seed: int = 0, workers: int = 8,
             opts: dict | None = None, prices: dict | None = None, gpu: str = "L4", sec_per_call: float = 1.0,
             out_tok_per_s: float = 80.0, prefill_tok_per_s: float = 6000.0, startup_h: float = 0.1,
             cpu: float = 4.0, memory_gib: int = 16) -> dict:
    """A job spec dict (jobs/TEMPLATE.yaml fields) for one condition on an open model served by vLLM in the job.
    Does not write a file. Estimate = GPU hours x the GPU's per-second rate (rate card) with
    hours = start-up + n_items x (calls x sec_per_call + prefill tokens / prefill_tok_per_s +
    out tokens / out_tok_per_s) / workers; the placeholders
    are replaced by measured values after the P0 micro-pilot. max_api_usd is 0 (no API in v3); max_usd covers the
    worst-case token arithmetic x 1.5 plus start-up. Refuses split="test" before the prereg-v2 tag."""
    cond = condition.upper()
    if split == "test":
        from ... import launch
        if not launch.prereg_tagged():
            raise RuntimeError("test-split jobs only after the prereg-v2 tag (project rules, Section 6)")
    from ..track_a import config
    model_id = config.resolve_model(model)
    prices = prices or _load_prices()
    m = prices.get("modal", {})
    rates = m.get("gpu_per_second") or {}
    if gpu not in rates:
        raise ValueError(f"unknown gpu {gpu!r}; the rate card has {sorted(rates)}")
    gpu_rate = rates[gpu]
    rate_h = 3600 * (gpu_rate + cpu * m.get("cpu_core_per_second", 0.0000131) + memory_gib * m.get("memory_gib_per_second", 0.00000222))
    est = estimate_session(cond, opts)
    hours = startup_h + n_items * session_gpu_hours(est, "typical", workers, sec_per_call, out_tok_per_s, prefill_tok_per_s)
    hours_w = startup_h + n_items * session_gpu_hours(est, "worst", workers, sec_per_call, out_tok_per_s, prefill_tok_per_s)
    estimate = round(hours * rate_h, 2)
    t = est["typical"]
    formula = (f"{n_items} sessions x ({cond}: {t['calls']} calls x {sec_per_call}s + {t['tokens_in']} prefill tokens / "
               f"{prefill_tok_per_s} tok/s + {t['tokens_out']} out tokens / {out_tok_per_s} tok/s) / {workers} workers + {startup_h} h vLLM start-up = {hours:.2f} h x ${rate_h:.3f}/h "
               f"({gpu} + {cpu} cpu + {memory_gib} GiB) = ${estimate:.2f}; "
               f"sessions x $ per session = ${estimate / max(1, n_items):.4f} "
               f"(opa.tracks.track_a2.run.estimate_session; PLACEHOLDER throughputs until the P0 micro-pilot)")
    return {
        "name": name, "envelope": envelope, "track": "A2", "hypotheses": ["H2"], "system": cond, "dataset": "drift",
        "cell": cell, "split": split, "seed": seed, "entrypoint": "opa.tracks.track_a2:run_condition",
        "args": {"items_uri": items_uri, "n_items": n_items, "condition": cond,
                 "model": {"kind": "open_local", "name": model}, "workers": workers, "opts": dict(opts or {})},
        "models": {"subject": model_id},
        "compute": {"kind": "gpu", "gpu": gpu, "image": "vllm", "cpu": cpu, "memory_gib": memory_gib},
        "throughput_placeholders": {"sec_per_call": sec_per_call, "out_tok_per_s": out_tok_per_s,
                                    "prefill_tok_per_s": prefill_tok_per_s, "startup_h": startup_h,
                                    "status": "placeholder: replace with P0 micro-pilot measurements"},
        "estimate_formula": formula, "estimate_usd": estimate,
        "max_usd": round(max(hours_w * 1.5, hours * 2) * rate_h + 1, 2), "max_api_usd": 0,
    }
