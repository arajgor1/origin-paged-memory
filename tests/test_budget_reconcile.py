"""reconcile(): per-day matched totals, flags, automatic explanation, annotate. Fake billing report, temp ledger/runs/reconciliation (conftest redirect)."""
import datetime as dt
import json
import sys
import types

import pytest

from opa import budget, launch

NOW = dt.datetime(2026, 11, 9, 7, 0, tzinfo=dt.timezone.utc)
PRICES = {"modal": {"cpu_core_per_second": 0.00003, "memory_gib_per_second": 0.000004, "gpu_per_second": {"L4": 0.0002, "T4": 0.0001}}}
OPTIONS = {"cpu": [1.0, 2.0], "memory": [4096, 8192], "gpu": "L4"}      # request rate / limit rate = 0.8425


@pytest.fixture(autouse=True)
def prices(monkeypatch):
    monkeypatch.setattr(budget, "load_prices", lambda path=None: PRICES)


def fake_report(monkeypatch, rows):
    mod = types.ModuleType("modal")
    bill = types.ModuleType("modal.billing")
    bill.workspace_billing_report = lambda **kw: rows
    mod.billing = bill
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.billing", bill)


def actual(day, usd, run_id="r", options=OPTIONS):
    if options is not None and run_id:
        d = launch.RUNS / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "call.json").write_text(json.dumps({"options": options}))
    budget.append_ledger({"ts_utc": f"{day}T12:00:00Z", "envelope": "E2", "job": "j", "kind": "actual", "amount_usd": usd,
                          "provider": "modal", "run_id": run_id})


def prow(day, cost, app="opa-experiments", tags=None):
    return {"description": app, "interval_start": f"{day} 00:00:00+00:00", "cost": str(cost),
            "tags": {"project": "opa"} if tags is None and app == "opa-experiments" else (tags or {})}


def run(monkeypatch, rows, **kw):
    fake_report(monkeypatch, rows)
    return json.loads(budget.reconcile(7, now=NOW, app_name="opa-experiments", **kw).read_text())


def test_real_case_unmatched_day_and_limit_ratio(monkeypatch):
    actual("2026-11-07", 33.38)
    actual("2026-11-08", 57.69, run_id="r2")
    rec = run(monkeypatch, [prow("2026-11-07", 29.168), prow("2026-11-01", 0.01, app="other-app", tags={})])
    assert rec["provider_total_usd"] == 29.168 and rec["ledger_total_usd"] == pytest.approx(91.07)
    assert rec["provider_by_day"] == {"2026-11-07": 29.168}
    assert rec["ledger_by_day"] == {"2026-11-07": 33.38, "2026-11-08": 57.69}
    assert rec["matched_provider_usd"] == 29.168 and rec["matched_ledger_usd"] == 33.38
    assert rec["unmatched_ledger_days"] == ["2026-11-08"]
    assert rec["matched_ledger_at_requests_usd"] == pytest.approx(33.38 * 0.8425, rel=1e-3)
    e = rec["explanation"]
    assert "provider has no row yet for 2026-11-08" in e and "1.14x" in e and "limits" in e
    assert "max(request, usage)" in e and "re-priced at the container requests" in e and "collect-time bookings" in e
    assert "flag" not in rec
    assert rec["period_start"] == "2026-11-02T00:00:00+00:00"       # whole UTC days


def test_within_ten_percent_has_no_explanation(monkeypatch):
    actual("2026-11-07", 10.0)
    rec = run(monkeypatch, [prow("2026-11-07", 9.5)])
    assert rec["explanation"] == "" and "flag" not in rec and rec["unmatched_ledger_days"] == []


def test_provider_exceeds_ledger_is_flagged_never_explained(monkeypatch, capsys):
    actual("2026-11-07", 10.0)
    actual("2026-11-08", 5.0, run_id="r2")    # an unmatched lag day must not hide it
    rec = run(monkeypatch, [prow("2026-11-07", 20.0)])
    assert rec["explanation"] == "" and "PROVIDER EXCEEDS LEDGER: investigate" in rec["flag"]
    assert "PROVIDER EXCEEDS LEDGER" in capsys.readouterr().out


def test_provider_only_day_is_flagged(monkeypatch):
    actual("2026-11-07", 10.0)
    rec = run(monkeypatch, [prow("2026-11-07", 10.0), prow("2026-11-04", 5.0)])
    assert "PROVIDER DAY EXCEEDS LEDGER: investigate" in rec["flag"] and "2026-11-04" in rec["flag"] and rec["explanation"] == ""


def test_provider_day_allows_next_day_booking(monkeypatch):
    actual("2026-11-08", 10.0)       # the run of 11-07 was collected on 11-08
    rec = run(monkeypatch, [prow("2026-11-07", 9.0), prow("2026-11-08", 0.5)])
    assert "PROVIDER DAY EXCEEDS" not in rec.get("flag", "")


def test_mid_window_gap_is_flagged_not_lag(monkeypatch):
    actual("2026-11-03", 4.0, run_id="r3")
    actual("2026-11-07", 10.0)
    rec = run(monkeypatch, [prow("2026-11-07", 9.5)])
    assert "LEDGER DAY WITHOUT PROVIDER ROW: investigate" in rec["flag"] and "2026-11-03" in rec["flag"]
    assert rec["explanation"] == ""


def test_limit_pricing_cause_is_bounded(monkeypatch):
    actual("2026-11-07", 33.38)
    rec = run(monkeypatch, [prow("2026-11-07", 10.0)])        # far below even the request-priced floor (28.1)
    assert "LEDGER EXCEEDS PROVIDER BEYOND LIMIT PRICING: investigate" in rec["flag"] and rec["explanation"] == ""


def test_row_without_call_json_counts_in_full_in_floor(monkeypatch):
    actual("2026-11-07", 30.0, run_id="nocall", options=None)
    rec = run(monkeypatch, [prow("2026-11-07", 20.0)])
    assert rec["matched_ledger_at_requests_usd"] == 30.0
    assert "BEYOND LIMIT PRICING" in rec["flag"]               # 20 < 0.9 x 30


def test_other_apps_are_not_counted(monkeypatch):
    actual("2026-11-07", 10.0)
    rec = run(monkeypatch, [prow("2026-11-07", 10.0), prow("2026-11-07", 99.0, app="other", tags={})])
    assert rec["provider_total_usd"] == 10.0 and rec["provider_by_day"] == {"2026-11-07": 10.0} and rec["explanation"] == ""


def test_second_tagged_app_counts_in_total_and_by_day(monkeypatch):
    actual("2026-11-07", 20.0)
    rec = run(monkeypatch, [prow("2026-11-07", 10.0), prow("2026-11-07", 9.0, app="opa-other", tags={"project": "opa"})])
    assert rec["provider_total_usd"] == 19.0 and rec["provider_by_day"] == {"2026-11-07": 19.0}
    assert "FILTER MISMATCH" not in rec.get("flag", "")


def test_filter_mismatch_flag(monkeypatch):
    actual("2026-11-07", 10.0)
    bad = {**prow("2026-11-07", 3.0), "interval_start": "not a date"}
    rec = run(monkeypatch, [prow("2026-11-07", 10.0), bad])
    assert "PROVIDER FILTER MISMATCH: investigate" in rec["flag"]


def test_interval_start_is_normalised_to_utc(monkeypatch):
    actual("2026-11-07", 10.0)
    r = {**prow("2026-11-06", 9.5), "interval_start": "2026-11-06 20:00:00-08:00"}          # = 2026-11-07 04:00 UTC
    rec = run(monkeypatch, [r])
    assert rec["provider_by_day"] == {"2026-11-07": 9.5}
    assert budget._utc_day(dt.datetime(2026, 11, 7, 1, tzinfo=dt.timezone(dt.timedelta(hours=3)))) == "2026-11-06"      # 22:00 UTC the day before


def test_reconcile_line_written(monkeypatch):
    actual("2026-11-07", 10.0)
    run(monkeypatch, [prow("2026-11-07", 10.0)])
    assert [r["kind"] for r in budget.read_ledger()][-1] == "reconcile"


def write_rec(explanation="", name="modal_x.json", **extra):
    budget.RECON_DIR.mkdir(parents=True, exist_ok=True)
    p = budget.RECON_DIR / name
    p.write_text(json.dumps({"provider": "modal", "period_start": "a", "period_end": "b", "provider_total_usd": 29.17,
                             "ledger_total_usd": 84.57, "explanation": explanation, "rows": [], **extra}))
    return p


def test_annotate_sets_empty_explanation_only():
    p = write_rec()
    assert budget.annotate(p, "known cause") == 0
    rec = json.loads(p.read_text())
    assert rec["explanation"] == "known cause" and "annotated_at" in rec
    assert rec["provider_total_usd"] == 29.17 and rec["ledger_total_usd"] == 84.57
    before = p.read_text()
    assert budget.annotate(p, "another") == 2 and p.read_text() == before


def test_annotate_refusals(tmp_path):
    assert budget.annotate(write_rec("already"), "x") == 2
    q = write_rec(name="q.json")
    assert budget.annotate(q, "  ") == 2 and json.loads(q.read_text())["explanation"] == ""
    assert budget.annotate(budget.RECON_DIR / "nope.json", "x") == 2
    outside = tmp_path / "bundle" / "modal_x.json"
    outside.parent.mkdir()
    outside.write_text(q.read_text())
    assert budget.annotate(outside, "x") == 2 and json.loads(outside.read_text())["explanation"] == ""
    bad = budget.RECON_DIR / "bad.json"
    bad.write_text("{not json")
    assert budget.annotate(bad, "x") == 2
    lst = budget.RECON_DIR / "list.json"
    lst.write_text("[1, 2]")
    assert budget.annotate(lst, "x") == 2
    nokeys = budget.RECON_DIR / "nokeys.json"
    nokeys.write_text(json.dumps({"provider_total_usd": 1, "ledger_total_usd": 2, "explanation": ""}))
    assert budget.annotate(nokeys, "x") == 2


def test_annotate_cli():
    p = write_rec()
    assert budget.main(["annotate", str(p), "--explanation", "why"]) == 0
    assert json.loads(p.read_text())["explanation"] == "why"
    assert budget.main(["annotate", str(p)]) == 2
    q = write_rec(name="q.json")
    assert budget.main(["annotate", str(q), "x", "--explanation"]) == 2       # --explanation is the last argument
    assert json.loads(q.read_text())["explanation"] == ""


def test_second_run_never_overwrites_the_first(monkeypatch):
    actual("2026-11-07", 10.0)
    fake_report(monkeypatch, [prow("2026-11-07", 10.0)])
    first = budget.reconcile(7, now=NOW, app_name="opa-experiments")
    assert first.name == "modal_20261102_20261109T07.json"
    assert budget.annotate(first, "human note") == 0
    before = first.read_text()
    second = budget.reconcile(7, now=NOW, app_name="opa-experiments")
    third = budget.reconcile(7, now=NOW, app_name="opa-experiments")
    assert second != first and second.name.endswith("-2.json") and third.name.endswith("-3.json")
    assert first.read_text() == before and json.loads(first.read_text())["explanation"] == "human note"


def test_no_floor_means_flag_not_cause():
    out = budget.analyse_days({"2026-11-07": 10.0}, {"2026-11-07": 30.0}, 10.0, NOW, None)
    assert out["explanation"] == "" and "BEYOND LIMIT PRICING: investigate" in out["flag"]


def test_closing_sentence_wording(monkeypatch):
    actual("2026-11-07", 33.38)
    e = run(monkeypatch, [prow("2026-11-07", 29.168)])["explanation"] + budget.analyse_days(
        {"2026-11-07": 29.168}, {"2026-11-07": 33.38}, 29.168, NOW, 28.0)["explanation"]
    assert "reaches at least 90% of" in e
    assert "collect-time bookings, so a run's cost can land on the day after it ran; this can only raise flags." in e
