"""Budget accounting: approvals (human-owned), prices (human-owned), ledger (append-only).

Ledger kinds:
  reserve  +max_usd held for a running job (counts against the envelope until released)
  release  the same amount returned when the job finishes (ledger amount is positive; it cancels the reserve)
  actual   money actually spent
  reconcile  informational line written by `reconcile` (amount = provider figure, not added to spend)

CLI:
  python -m opa.budget status
  python -m opa.budget check <ENVELOPE> <max_usd>
  python -m opa.budget reconcile [--days 7]
  python -m opa.budget annotate <reconciliation file> --explanation "<text>"   (only if the explanation is empty)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import PROJECT_DIR

APPROVALS = PROJECT_DIR / "budget" / "approvals.yaml"
PRICES = PROJECT_DIR / "budget" / "prices.yaml"
LEDGER = PROJECT_DIR / "budget" / "ledger.csv"
RECON_DIR = PROJECT_DIR / "budget" / "reconciliation"
LEDGER_FIELDS = ["ts_utc", "envelope", "job", "kind", "amount_usd", "provider", "run_id", "note"]
OVERRUN_TOLERANCE = 1.05   # verifier allows actual <= 1.05 x approved; the launcher itself enforces <= 1.0


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_approvals(path: Path | None = None) -> dict:
    data = yaml.safe_load((path or APPROVALS).read_text()) or {}
    data.setdefault("envelopes", [])
    return data


def approved_envelopes(approvals: dict) -> dict[str, float]:
    """Envelopes that are actually approved: positive amount, initials and a date."""
    out = {}
    for e in approvals.get("envelopes", []):
        if float(e.get("approved_usd") or 0) > 0 and str(e.get("approved_by") or "").strip() and str(e.get("approved_on") or "").strip():
            out[str(e["id"])] = float(e["approved_usd"])
    return out


def load_prices(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PRICES).read_text())


def read_ledger(path: Path | None = None) -> list[dict]:
    path = path or LEDGER
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_ledger(row: dict, path: Path | None = None) -> None:
    path = path or LEDGER
    row = {k: row.get(k, "") for k in LEDGER_FIELDS}
    row["ts_utc"] = row["ts_utc"] or now()
    row["amount_usd"] = f"{float(row['amount_usd']):.6f}"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


@dataclass
class EnvelopeState:
    id: str
    approved: float
    spent: float
    reserved: float

    @property
    def remaining(self) -> float:
        return self.approved - self.spent - self.reserved


def envelope_states(approvals: dict | None = None, ledger: list[dict] | None = None) -> dict[str, EnvelopeState]:
    approvals = approvals if approvals is not None else load_approvals()
    ledger = ledger if ledger is not None else read_ledger()
    ok = approved_envelopes(approvals)
    ids = set(ok) | {r["envelope"] for r in ledger}
    st = {i: EnvelopeState(i, ok.get(i, 0.0), 0.0, 0.0) for i in ids}
    for r in ledger:
        amt = float(r["amount_usd"] or 0)
        s = st[r["envelope"]]
        if r["kind"] == "actual":
            s.spent += amt
        elif r["kind"] == "reserve":
            s.reserved += amt
        elif r["kind"] == "release":
            s.reserved -= amt
    for s in st.values():
        s.reserved = max(0.0, round(s.reserved, 6))
    return st


def check(envelope: str, max_usd: float, approvals: dict | None = None, ledger: list[dict] | None = None) -> tuple[bool, str]:
    """Can a job with worst-case cost max_usd run against this envelope (and the total cap)?"""
    approvals = approvals if approvals is not None else load_approvals()
    ledger = ledger if ledger is not None else read_ledger()
    ok = approved_envelopes(approvals)
    if envelope not in ok:
        return False, (f"Envelope {envelope} is not approved in budget/approvals.yaml. Write budget/requests/{envelope}.md "
                       f"from budget/requests/TEMPLATE.md, set state/STATE to AWAITING_BUDGET_APPROVAL, and stop.")
    st = envelope_states(approvals, ledger)
    s = st[envelope]
    if max_usd > s.remaining + 1e-9:
        return False, (f"Envelope {envelope}: approved ${s.approved:.2f}, spent ${s.spent:.2f}, reserved ${s.reserved:.2f}, "
                       f"remaining ${s.remaining:.2f} < job max ${max_usd:.2f}. Request more in budget/requests/{envelope}.md "
                       f"and set state/STATE to AWAITING_BUDGET_APPROVAL.")
    cap = float(approvals.get("total_cap_usd") or 0)
    total = sum(x.spent + x.reserved for x in st.values())
    if cap <= 0 or total + max_usd > cap + 1e-9:
        return False, (f"Total cap ${cap:.2f} would be exceeded (committed ${total:.2f} + job ${max_usd:.2f}). "
                       f"The human must raise total_cap_usd in budget/approvals.yaml.")
    return True, f"ok: {envelope} remaining ${s.remaining:.2f} after reservations; job max ${max_usd:.2f}"


# ---------------------------------------------------------------- cost formulas
def modal_cost(seconds: float, gpu: str | None, cpu_cores: float = 1.0, mem_gib: float = 4.0, prices: dict | None = None) -> float:
    prices = prices or load_prices()
    p = prices["modal"] if "modal" in prices else prices
    rate = p["cpu_core_per_second"] * max(cpu_cores, 0.125) + p["memory_gib_per_second"] * mem_gib
    if gpu:
        rate += p["gpu_per_second"][gpu]
    return seconds * rate


def anthropic_cost(model: str, usage: dict, batch: bool = False, prices: dict | None = None) -> float:
    """usage: input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens (Anthropic API names)."""
    prices = prices or load_prices()
    p = prices["anthropic"] if "anthropic" in prices else prices   # accept the full rate card or its anthropic section
    table = p[model]
    prompt_len = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    if "over_100k" in table and prompt_len > 100_000:
        table = {**table, **table["over_100k"]}
    m = 1e-6
    cost = (usage.get("input_tokens", 0) * table["input"] + usage.get("output_tokens", 0) * table["output"]
            + usage.get("cache_creation_input_tokens", 0) * table["cache_write_5m"]
            + usage.get("cache_read_input_tokens", 0) * table["cache_read"]) * m
    return cost * (p.get("batch_discount", 0.5) if batch else 1.0)


# ---------------------------------------------------------------- reconciliation with Modal's billing report
def _app_name() -> str:
    try:
        return str((yaml.safe_load((PROJECT_DIR / "config" / "project.yaml").read_text()) or {}).get("modal", {}).get("app_name") or "opa-experiments")
    except Exception:
        return "opa-experiments"


def _ratio_text(a: float, b: float) -> str:
    return f"{a / b:.2f}x" if b > 0 else "n/a"


def _next_day(day: str) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()


def _utc_day(v) -> str | None:
    """interval_start (datetime or ISO string, any offset) as a UTC calendar date, or None if unparseable."""
    try:
        d = v if isinstance(v, dt.datetime) else dt.datetime.fromisoformat(str(v))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc).date().isoformat()
    except (ValueError, TypeError):
        return None


def analyse_days(provider_by_day: dict, ledger_by_day: dict, provider_total: float | None = None,
                 period_end: dt.datetime | None = None, floor_usd: float | None = None) -> dict:
    """Matched totals over the days the provider reports a row for, the ledger days it has no row for yet, flags
    that need a human, and an automatic explanation only when no flag is raised.

    Ledger days are collect-time bookings (a run is booked when it is collected, possibly a day after it ran), so
    they can only add flags, never prove a provider figure wrong. floor_usd is the ledger's matched-day spend re-priced
    at the container requests (no limit pricing); the rate-card cause is given only if the provider bill reaches 90% of it."""
    matched_days = sorted(provider_by_day)
    unmatched = sorted(d for d in ledger_by_day if d not in provider_by_day and ledger_by_day[d] > 0)
    mp = sum(provider_by_day[d] for d in matched_days)
    ml = sum(ledger_by_day.get(d, 0.0) for d in matched_days)
    pt = sum(provider_by_day.values()) if provider_total is None else provider_total
    lt = sum(ledger_by_day.values())
    out = {"matched_provider_usd": round(mp, 4), "matched_ledger_usd": round(ml, 4), "unmatched_ledger_days": unmatched,
           "explanation": "", "flag": ""}
    flags = []
    if abs(sum(provider_by_day.values()) - pt) > 0.01:
        flags.append(f"PROVIDER FILTER MISMATCH: investigate (provider total ${pt:.2f} vs per-day sum ${sum(provider_by_day.values()):.2f})")
    bad = []
    for d in matched_days:
        comb = ledger_by_day.get(d, 0.0) + ledger_by_day.get(_next_day(d), 0.0)      # +1 day: the ledger books at collect time
        if provider_by_day[d] - comb > 0.10 * max(provider_by_day[d], comb) + 0.05:
            bad.append(f"{d} (provider ${provider_by_day[d]:.2f} vs ledger ${comb:.2f} over that day and the next)")
    if bad:
        flags.append("PROVIDER DAY EXCEEDS LEDGER: investigate: " + "; ".join(bad))
    lag_cutoff = ((period_end - dt.timedelta(days=1)).date().isoformat()) if period_end else None
    lag, gaps = [], []
    for d in unmatched:
        (lag if lag_cutoff is not None and d >= lag_cutoff else gaps).append(d)
    if gaps:
        flags.append("LEDGER DAY WITHOUT PROVIDER ROW: investigate: " + ", ".join(f"{d} (ledger ${ledger_by_day[d]:.2f})" for d in gaps))
    differs = max(pt, lt) > 0 and abs(pt - lt) > 0.10 * max(pt, lt)
    if (max(mp, ml) > 0 and mp - ml > 0.10 * max(mp, ml)) or (differs and pt > lt):
        flags.append(f"PROVIDER EXCEEDS LEDGER: investigate (matched days: provider ${mp:.2f} vs ledger ${ml:.2f}; "
                     f"all days: provider ${pt:.2f} vs ledger ${lt:.2f})")
    ledger_high = ml - mp > 0.10 * max(mp, ml)
    if ledger_high and floor_usd is None:
        flags.append("LEDGER EXCEEDS PROVIDER BEYOND LIMIT PRICING: investigate (no request-priced floor available, so the rate-card cause cannot be checked)")
    elif ledger_high and mp < 0.9 * floor_usd:
        flags.append(f"LEDGER EXCEEDS PROVIDER BEYOND LIMIT PRICING: investigate (matched provider ${mp:.2f} < 90% of the ledger re-priced "
                     f"at the container requests, ${floor_usd:.2f})")
    if flags:
        out["flag"] = " | ".join(flags) + ". Not auto-explained."
        return out
    if not differs:
        return out
    parts = []
    if lag:
        parts.append("provider has no row yet for " + ", ".join(f"{d} (ledger ${ledger_by_day[d]:.2f})" for d in lag)
                     + " (Modal's daily billing rows lag about a day)")
    if ledger_high:
        extra = (f"; the ledger re-priced at the container requests is ${floor_usd:.2f} (ledger/floor {_ratio_text(ml, floor_usd)}), "
                 "which the provider bill reaches at least 90% of") if floor_usd else ""
        parts.append(f"on the days the provider reports ({', '.join(matched_days) or 'none'}): provider ${mp:.2f}, ledger ${ml:.2f}, "
                     f"ledger/provider {_ratio_text(ml, mp)}; the ledger prices containers at the launcher rate with CPU/memory at their "
                     "limits (2x the request), while Modal bills max(request, usage) up to the limit" + extra)
    if parts:
        out["explanation"] = ("Automatic: " + "; ".join(parts) + ". Ledger days are collect-time bookings, so a run's cost can land "
                              "on the day after it ran; this can only raise flags.")
    return out


def _request_ratio(run_id: str) -> float:
    """request-rate / limit-rate of a run's container (the cheapest the launcher's pricing could be); 1.0 when unknown."""
    from . import launch      # lazy: launch imports this module
    try:
        o = json.loads((launch.RUNS / run_id / "call.json").read_text())["options"]
        cpu, mem = o["cpu"], o["memory"]
        kind = {"kind": "gpu", "gpu": o["gpu"]} if o.get("gpu") else {"kind": "cpu"}
        prices = load_prices()
        req = launch.rate_per_second({**kind, "cpu": cpu[0], "cpu_limit": cpu[0], "memory_gib": mem[0] / 1024, "memory_gib_limit": mem[0] / 1024}, prices)
        lim = launch.rate_per_second({**kind, "cpu": cpu[0], "cpu_limit": cpu[1], "memory_gib": mem[0] / 1024, "memory_gib_limit": mem[1] / 1024}, prices)
        return min(1.0, req / lim) if lim > 0 else 1.0
    except Exception:
        return 1.0


def reconcile(days: int = 7, now: dt.datetime | None = None, app_name: str | None = None) -> Path:
    import modal  # imported here so the rest of the module works without modal installed
    from modal.billing import workspace_billing_report

    app = app_name or _app_name()
    end = (now or dt.datetime.now(dt.timezone.utc)).replace(minute=0, second=0, microsecond=0)
    start = (end - dt.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)      # whole UTC days
    rows = workspace_billing_report(start=start, end=end, resolution="d", tag_names=["project"])
    provider_total = 0.0
    provider_by_day: dict[str, float] = {}
    items = []
    for r in rows:
        rd = dict(r) if not isinstance(r, dict) else r
        tags = rd.get("tags") or {}
        cost = float(rd.get("cost", 0) or 0)
        if tags.get("project") == "opa" or rd.get("description") == app:      # one predicate for the total and the per-day figures
            provider_total += cost
            day = _utc_day(rd.get("interval_start"))
            if day is not None:
                provider_by_day[day] = provider_by_day.get(day, 0.0) + cost
        items.append({k: (str(v) if not isinstance(v, (int, float, str, dict, list, type(None))) else v) for k, v in rd.items()})
    in_window = [r for r in read_ledger()
                 if r["kind"] == "actual" and r["provider"] == "modal" and start.isoformat()[:19] <= r["ts_utc"].replace("Z", "") <= end.isoformat()[:19]]
    ledger_total = sum(float(r["amount_usd"]) for r in in_window)
    ledger_by_day: dict[str, float] = {}
    for r in in_window:
        ledger_by_day[r["ts_utc"][:10]] = ledger_by_day.get(r["ts_utc"][:10], 0.0) + float(r["amount_usd"])
    floor = sum(float(r["amount_usd"]) * (_request_ratio(r["run_id"]) if r.get("run_id") else 1.0)
                for r in in_window if r["ts_utc"][:10] in provider_by_day)
    an = analyse_days(provider_by_day, ledger_by_day, provider_total, end, floor)
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    out = RECON_DIR / f"modal_{start:%Y%m%d}_{end:%Y%m%dT%H}.json"
    n = 1
    while out.exists():      # never overwrite a reconciliation file (it may carry a human annotation)
        n += 1
        out = RECON_DIR / f"modal_{start:%Y%m%d}_{end:%Y%m%dT%H}-{n}.json"
    rec = {"provider": "modal", "period_start": start.isoformat(), "period_end": end.isoformat(),
           "provider_total_usd": round(provider_total, 4), "ledger_total_usd": round(ledger_total, 4),
           "app_name": app,
           "provider_by_day": {d: round(v, 4) for d, v in sorted(provider_by_day.items())},
           "ledger_by_day": {d: round(v, 4) for d, v in sorted(ledger_by_day.items())},
           "matched_provider_usd": an["matched_provider_usd"], "matched_ledger_usd": an["matched_ledger_usd"],
           "matched_ledger_at_requests_usd": round(floor, 4),
           "unmatched_ledger_days": an["unmatched_ledger_days"],
           "explanation": an["explanation"], "rows": items}
    if an["flag"]:
        rec["flag"] = an["flag"]
        print(an["flag"])
    out.write_text(json.dumps(rec, indent=2, default=str))
    append_ledger({"envelope": "RECON", "job": "reconcile", "kind": "reconcile", "amount_usd": provider_total,
                   "provider": "modal", "note": f"provider {provider_total:.2f} vs ledger {ledger_total:.2f} for {start:%Y-%m-%d}..{end:%Y-%m-%d}"})
    return out


def annotate(path: str | Path, explanation: str) -> int:
    """Set the explanation of a reconciliation file in RECON_DIR, only if it is empty. Changes no number; adds annotated_at."""
    p = Path(path)
    if not p.exists():
        print(f"Refused: {p} does not exist.")
        return 2
    if p.resolve().parent != Path(RECON_DIR).resolve():
        print(f"Refused: {p} is not directly inside {RECON_DIR} (bundle copies are not annotated).")
        return 2
    if not explanation.strip():
        print("Refused: the explanation is empty.")
        return 2
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"Refused: {p.name} is not valid JSON ({e}).")
        return 2
    if not isinstance(rec, dict) or not all(k in rec for k in ("provider", "period_start", "period_end", "provider_total_usd", "ledger_total_usd")):
        print("Refused: not a reconciliation file.")
        return 2
    if str(rec.get("explanation", "")).strip():
        print(f"Refused: {p.name} already has an explanation.")
        return 2
    rec["explanation"] = explanation.strip()
    rec["annotated_at"] = now()
    p.write_text(json.dumps(rec, indent=2, default=str))
    print(f"annotated {p.name}")
    return 0


def status_table() -> str:
    approvals = load_approvals()
    st = envelope_states(approvals)
    cap = float(approvals.get("total_cap_usd") or 0)
    lines = [f"{'envelope':10} {'approved':>10} {'spent':>10} {'reserved':>10} {'remaining':>10}"]
    for k in sorted(st):
        if k == "RECON":
            continue
        s = st[k]
        lines.append(f"{k:10} {s.approved:10.2f} {s.spent:10.2f} {s.reserved:10.2f} {s.remaining:10.2f}")
    committed = sum(s.spent + s.reserved for k, s in st.items() if k != "RECON")
    lines.append(f"total cap ${cap:.2f}; committed ${committed:.2f}; headroom ${cap - committed:.2f}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "status":
        print(status_table())
        return 0
    if argv[0] == "check":
        ok, msg = check(argv[1], float(argv[2]))
        print(msg)
        return 0 if ok else 3
    if argv[0] == "reconcile":
        days = int(argv[argv.index("--days") + 1]) if "--days" in argv else 7
        print(reconcile(days))
        return 0
    if argv[0] == "annotate":
        if len(argv) < 4 or "--explanation" not in argv or argv.index("--explanation") + 1 >= len(argv):
            print('usage: python -m opa.budget annotate <reconciliation file> --explanation "<text>"')
            return 2
        return annotate(argv[1], argv[argv.index("--explanation") + 1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
