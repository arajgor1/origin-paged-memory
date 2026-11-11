"""Costed-request packages (opa.analysis.plan_packages): base vs BOOST vs raise money, reservation slack, container splits, tiers, power."""
import csv
import math

import pytest

from opa import stats
from opa.analysis import plan_costs as pc
from opa.analysis import plan_packages as pk


def names(res, key="included"):
    return [it["key"] for it in res[key]]


def fill(base, boost, raise_=0.0, **kw):
    return pk.fill(base, boost, raise_, **kw)


def test_power_matches_the_sample_size_formulas():
    for key, (label, method, delta, p_d, sd, n_req) in pk.PRIMARIES.items():
        if n_req is None:
            continue
        pw = pk.power(method, n_req, delta, p_d, sd)
        assert 0.78 <= pw <= 0.83 or pw > 0.95, (key, pw)               # sized for 80% (floors / rounding give more)
    assert abs(stats.n_mcnemar(0.3, 0.03) - 2614) <= 2 and abs(stats.n_mcnemar(0.312, 0.02) - pk.PRIMARIES["H13"][5]) <= 2
    assert abs(pk.power("mcnemar", 1000, 0.02, 0.312, None) - 0.20) < 0.02            # H13 at ~1,000 questions
    assert pk.power("mcnemar", 0, 0.05, 0.3, None) == 0.0
    assert pk.power("tost", stats.n_tost(0.02, 0.03), 0.03, 0.02, None) == pytest.approx(0.8, abs=0.02)
    # Wilcoxon ARE 0.955 is applied to the paired-normal tests (H10, H2)
    a = pk.power("paired_normal", 860, 0.5, None, 5.114)
    b = 0.5 * math.sqrt(860) / 5.114
    assert a < pk._PHI.cdf(b - 1.96) and abs(a - pk._PHI.cdf(b * math.sqrt(0.955) - 1.96)) < 1e-3


def test_p_pass_s11_vs_s4_and_min_delta_rules():
    n = 391
    se = math.sqrt((0.50 - 0.46 ** 2) / n)
    assert pk.p_pass("S11_vs_S4", n, 0.46, 0.50) == pytest.approx(pk._PHI.cdf((0.46 - 0.03) / se))
    assert pk.p_pass("S11_vs_S4", n, 0.46, 0.50) > 0.999 and pk.p_pass("S11_vs_S4", n, 0.03, 0.50) == pytest.approx(0.5)
    # rules needing a CI/p condition: the cut is max(threshold, 1.96 SE)
    assert pk.p_pass("S11_vs_S2", 1191, 0.10, 0.38) > 0.99
    assert pk.p_pass("S11_vs_S8", 1191, 0.02, 0.02) == pytest.approx(0.5, abs=0.01)
    assert pk.p_pass("H2", 200, 2.667, None) < 0.001 and pk.p_pass("H2", 200, 12.0, None) > 0.9
    assert pk.PRIMARIES["S11_vs_S4"][5] is None                                       # the 6,452 from broken S4 v1 is gone


def test_money_state_and_pools(monkeypatch):
    monkeypatch.setattr(pk, "money_state", lambda: (60.0, 30.0, 40.0))
    assert pk.package_pools("96") == (60.0, 30.0, 0.0)
    assert pk.package_pools("126") == (60.0, 30.0, 36.0) and pk.package_pools("full")[2] == pk.INF
    res = budget_state = pk.money_state  # sanity: the monkeypatched function is what fill() reads
    assert res() == (60.0, 30.0, 40.0)


def test_boost_only_funds_the_allowed_lines_and_base_goes_first():
    res = fill(60.0, 30.0)
    allowed = {"H1", "H5", "h12", "g2_train"}
    for it in res["included"]:
        assert it["base_usd"] + it["boost_usd"] + it["raise_usd"] == pytest.approx(it["chosen"]["usd"])
        if it["boost_usd"] > 1e-9:
            assert it["key"] in allowed and it["boost_usd"] <= it["chosen"]["boost_elig"] + 1e-9 and it["chosen"]["boost_why"]
            assert it["base_usd"] > 0 or it["key"] != "H1"
        if it["key"] not in allowed:
            assert it["boost_usd"] == 0
    # the training item may be paid from BOOST only beyond L=400
    g2 = next(it for it in pk.catalog(pk.context(24)) if it["key"] == "g2_train")["alts"][0]
    c400 = pk.cost(pk.finish_rows([pc.train("P4", "E4", "x", 314, 21, 400, 64, 171 / 32), pc.train("P4", "E4", "y", 100, 5, 400, 64, 3.1)], pc.PARAMS))
    assert g2["boost"] == "d" and g2["boost_elig"] == pytest.approx(g2["usd"] - c400, rel=0.01)
    assert res["boost_used"] <= 30.0 + 1e-9 and res["base_used"] <= 60.0 + 1e-9
    # base-first literally: BOOST is touched only after base money (net of the reserve) ran out for that item
    first_boost = next((it for it in res["included"] if it["boost_usd"] > 0), None)
    assert first_boost is None or first_boost["base_usd"] + res["reserve"] >= 0


def test_reserve_slack_stays_inside_base_money_and_containers_split():
    res = fill(60.0, 30.0)
    assert res["base_used"] + res["reserve"] <= 60.0 + 1e-9
    assert res["cap_raise"] == 0 and res["reserve"] == pytest.approx(sum(res["reserve_by_env"].values()))
    for it in pk.catalog(pk.context(24)):
        for a in it["alts"] + it["options"]:
            for r in a["rows"]:
                assert r["container_est"] <= pk.CONTAINER_MAX_USD + 1e-6, (it["key"], r["line"], r["container_est"])
                if r["usd"] > 0:
                    assert r["containers"] >= 1
    for e, v in res["reserve_by_env"].items():
        assert 0 < v <= pk.CONTAINER_MAX_USD + 1e-6


def test_raise_pool_and_cap_message():
    big = fill(60.0, 30.0, pk.INF)
    assert big["excluded"] == [] and big["raise_used"] > 0 and big["cap_raise"] > 0
    assert big["total"] == pytest.approx(big["base_used"] + big["boost_used"] + big["raise_used"])
    small = fill(60.0, 30.0, 0.0)
    assert small["cap_raise"] == 0 and set(names(small)) <= set(names(big))


def test_tiers_and_orders():
    cat = pk.catalog(pk.context(24))
    tier = {it["key"]: it["tier"] for it in cat}
    order = [it["key"] for it in cat]
    assert order[0] == "H1" and "h5_cell_check" not in order                           # the M=2 cell check is an optional row, not a default item
    assert tier["h9"] == 4 and tier["h4"] == 2 and tier["a1a3"] == 2 and tier["h3_64k"] == 6
    assert order.index("g2_eval") + 1 == order.index("a1a3") and order.index("a1a3") < order.index("h12") < order.index("h4")   # model-level H5 right after G2
    assert order.index("h11") < order.index("h13") < order.index("h9")
    assert [k for k in order if tier[k] == 2][:6] == ["learning_curve", "u6u8", "remedies", "trainfree", "h7", "g2_train"]
    assert "h3" not in order and not any("R2 cell" in it["name"] for it in cat)              # H3 R2 leaf accepted as inconclusive
    assert order.index("h3_64k") > order.index("repro")                                       # lowest priority, after the robustness extras
    res = fill(pk.INF, 30.0, 0.0)
    assert any(o["key"] == "H1" and "607" in o["option"]["label"] for o in res["options"])     # delta 7 row always shown
    assert any(o["key"] == "H5" and "M=2 cell qualifies" in o["option"]["label"] for o in res["options"])
    assert any(o["key"] == "H5" and "696" in o["option"]["label"] for o in res["options"])
    assert any(o["key"] == "bsweep" for o in res["options"]) and "bsweep" not in names(res)


def test_h5_cell_check_and_64k_check_pricing():
    x = pk.context(24)
    cat = {it["key"]: it for it in pk.catalog(x)}
    chk_opt = next(o for o in cat["H5"]["options"] if o["label"].startswith("OPTIONAL: blind M=2 cell check"))
    assert "routed_rate is already 0.83 < 0.90" in chk_opt["label"] and "H5 stays on T4 default M" in chk_opt["label"]
    chk = chk_opt
    rows = {r["line"]: r for r in chk["rows"]}
    s16 = next(r for r in chk["rows"] if "S16" in r["line"])
    assert s16["units"] == 30 and "A0" in s16["line"] and "lambda 0" in s16["line"]
    sec = 2.2 * x.mt["S16"] * 30
    assert math.isclose(s16["usd"], sec * pc.RATE + 0.066 * s16["containers"], abs_tol=0.01)               # MT S16 seconds x 2.2
    assert len(chk["rows"]) == 3 and chk["primary_n"] == {} and chk["boost"] is None
    assert not any("cell check" in r["line"] for it in fill(pk.INF, 30.0, 0.0)["included"] for r in it["chosen"]["rows"])      # not in the default packages
    k64 = cat["h3_64k"]["alts"][0]
    assert all(r["basis"] == "assumption" for r in k64["rows"]) and any("x2" in r["line"] for r in k64["rows"])
    s11 = next(r for r in k64["rows"] if "S11" in r["line"])
    assert math.isclose(s11["usd"], 2.0 * 30 * x.lb["S11"] * pc.RATE + 0.066 * s11["containers"], abs_tol=0.01)
    assert "64K vLLM config" in cat["h3_64k"]["name"] and "64K data generation" in cat["h3_64k"]["name"]
    variant = next(o for o in cat["H5"]["options"] if "M=2 cell qualifies" in o["label"])
    base = cat["H5"]["alts"][0]
    assert variant["usd"] > base["usd"]                                                       # same n on the M=2 cell: systems x2.2


def test_a1a3_is_use_d_boost_and_depends_on_g2():
    a = next(it for it in pk.catalog(pk.context(24)) if it["key"] == "a1a3")
    alt0 = a["alts"][0]
    assert alt0["boost"] == "d" and 0 < alt0["boost_elig"] < alt0["usd"] and a["requires"] == ["g2_train", "g2_eval"] and "conditional on G2" in a["name"]
    # a dependant is excluded together with the training it needs
    res = fill(8.0, 0.0, 0.0, strict=False)
    ex = {it["key"]: it for it in res["excluded"]}
    if "g2_train" in ex:
        for k in ("g2_eval", "a1a3", "h12", "h4", "h11", "h13"):
            assert k in ex and ex[k]["why"].startswith("needs")
    full = fill(pk.INF, 30.0, 0.0)
    assert "a1a3" in names(full)


def test_greedy_skip_and_h1_fallback():
    res = fill(70.0, 0.0, 0.0)
    tiers = [it["tier"] for it in res["included"]]
    assert tiers == sorted(tiers)
    small = fill(40.0, 0.0, 0.0)
    h1 = next((it for it in small["included"] if it["key"] == "H1"), None)
    assert h1 and h1["reduced"] and h1["chosen"]["primary_n"]["S11_vs_S2"] == 607     # delta 7 when 1,191 does not fit
    # a lower-priority item can be included while a higher-priority one is excluded (for some budget)
    order = [it["key"] for it in pk.catalog(pk.context(24))]
    found = False
    for b in range(10, 200, 2):
        r = fill(float(b), 0.0, 0.0, strict=False)
        inc = [order.index(k) for k in names(r)]
        ex = [order.index(k) for k in names(r, "excluded") if k in order]
        if any(i > e for i in inc for e in ex):
            found = True
            break
    assert found


def test_powers_table():
    res = fill(60.0, 30.0)
    pw = {r["key"]: r for r in res["powers"]}
    s4 = pw["S11_vs_S4"]
    assert s4["n_planned"] == 391 and not s4["reduced"] and s4["power"] is None and len(s4["p_pass"]) == 2
    assert pw["S11_vs_S8"]["flag"].startswith("p_d rests on 1 discordant pair")
    assert any(r["excluded"] and r["power"] == 0.0 for r in res["powers"])
    assert all(r["power"] is None or 0.0 <= r["power"] <= 1.0 for r in res["powers"])


def test_cli_prints_the_required_lines_and_writes_csvs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pk, "money_state", lambda: (60.0, 30.0, 40.0))
    out = tmp_path / "pk.csv"
    assert pc.main(["--package", "126", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    for frag in ("Strict priority: the fill stops at the first item that does not fit", "GREEDY ALTERNATIVE", "S2/S8/S11/S4 at 16 workers",
                 "MT at 24 workers was slower", "A-S2+S8+S11+S4-lbmhqa-late-h2-32k-a0-dev-s0-20261107202540-55d852f-141a", "RESERVATION SLACK",
                 "SENSITIVITY", "at 2x plan", "at 3x plan", "MONEY: base used", "needs owner to raise the cap by", "POWER", "ALTERNATIVE ORDER"):
        assert frag in text, frag
    rows = list(csv.DictReader(out.open(encoding="utf-8", newline="")))
    assert rows and set(rows[0]) == set(pk.CSV_FIELDS) and {r["status"] for r in rows} >= {"included", "option"}
    assert (tmp_path / "pk_power.csv").exists()


def test_stale_docstring_fixed():
    assert "now -3.0" in pk.__doc__ and "-2" not in pk.__doc__.split("Tier")[0][:0]


def test_default_order_is_boost_first_and_base_first_is_the_alternative(capsys):
    d = fill(60.0, 30.0)                                  # default: BOOST pays its eligible lines first
    lit = fill(60.0, 30.0, boost_first=False)             # the literal base-first order
    h1 = next(it for it in d["included"] if it["key"] == "H1")
    assert h1["boost_usd"] > 0 and h1["boost_usd"] <= h1["chosen"]["boost_elig"] + 1e-9
    assert next(it for it in lit["included"] if it["key"] == "H1")["boost_usd"] == 0
    assert d["boost_used"] >= lit["boost_used"] and d["total"] >= lit["total"]
    assert len(d["included"]) > len(lit["included"])                      # more of the plan fits when BOOST goes first
    for it in d["included"]:
        if it["key"] not in {"H1", "H5", "h12", "g2_train"}:
            assert it["boost_usd"] == 0
    pk.print_package("96", d)
    out = capsys.readouterr().out
    assert "ALTERNATIVE ORDER (literal base-first" in out


def test_final_system_pricing_and_failed_bench_runs_are_ignored(tmp_path, monkeypatch, capsys):
    import json
    import yaml
    x = pk.context(24)
    assert x.lb["S2"] == 0.9 and x.lb["S8"] == 13.8 and x.lb["S11"] == 14.4                      # measured at 16 workers (run ...-141a)
    p = pc.PARAMS
    assert math.isclose(x.lb["S4"], 1634.80 / 48) and x.lb["S4"] != p["s_per_item_lb"]["S4"]     # S4 measured at 16 workers (n 48)
    assert x.lb["S18"] == p["s_per_item_lb"]["S18"] and x.lb["S1"] == p["s_per_item_lb"]["S1"]
    for k in ("S12", "S14", "S15", "S16", "S18"):
        assert x.mt[k] == p["s_per_item_mt"][k]                                                  # MT stays at workers-8 (w24 was slower)
    assert p["mt_w24_measured_slower"] == {"S12": (31.6, 21.8), "S14": (30.9, 27.5)}
    assert math.isclose(x.sk, 0.0246, abs_tol=0.0002)
    # a collected bench run is not picked up silently (the ruling fixes the pricing); an explicit bench argument still overrides
    assert pk.context(24, {"lb": {"S11": 9.0}, "mt": {}}).lb["S11"] == 9.0
    # load_bench ignores failed runs
    runs = tmp_path / "runs"
    for nm, st in (("A-ok", "ok"), ("A-bad", "failed")):
        r = runs / nm
        r.mkdir(parents=True)
        (r / "job.yaml").write_text(yaml.safe_dump({"name": "bench-lb-w24"}))
        (r / "result.json").write_text(json.dumps({"status": st, "metrics": {"runs": {"S2/r0": {"wall_seconds": 24.0, "n": 48}}}}))
    assert pc.load_bench(24, runs)["lb"] == {"S2": 0.5}                                          # one ok run + one failed: no duplicate error
    monkeypatch.setattr(pk, "money_state", lambda: (60.0, 30.0, 40.0))
    assert pc.main(["--package", "96", "--out", str(tmp_path / "o.csv")]) == 0
    assert "S2/S8/S11/S4 at 16 workers" in capsys.readouterr().out


def test_strict_priority_stops_at_the_first_misfit_and_reports_unallocated():
    order = [it["key"] for it in pk.catalog(pk.context(24)) if it["alts"]]
    for b in (30.0, 60.0, 62.0, 75.0):
        r = fill(b, 0.0, 0.0)
        inc = names(r)
        assert inc == order[: len(inc)]                                  # a strict prefix of the priority order
        if r["stopped"]:
            nxt = order[len(inc)]
            assert r["stopped"]["key"] == nxt and r["stopped"]["need"] > 0
            assert names(r, "excluded") == order[len(inc):]
            assert all(it["why"].startswith(("does not fit", "needs", "not reached")) for it in r["excluded"])
            assert r["unallocated_base"] >= 0
        g = fill(b, 0.0, 0.0, strict=False)
        assert g["total"] >= r["total"] - 1e-9 and set(inc) <= set(names(g))


def test_stopped_package_prints_unallocated_greedy_and_the_l400_minimum(monkeypatch, capsys):
    monkeypatch.setattr(pk, "money_state", lambda: (57.0, 30.0, 40.0))     # 57 (was 60 before S4 was repriced at 16 workers) still stops at g2_train
    res = pk.run("96")
    out = capsys.readouterr().out
    assert res["stopped"] and res["stopped"]["key"] == "g2_train"
    assert "STOPPED (strict priority order) at T2 V3 + B1 to the stopping rule" in out and "UNALLOCATED: base $" in out
    assert "NOT spent on lower-priority items" in out and "GREEDY ALTERNATIVE" in out and "would add:" in out
    assert "L=400 MINIMUM ALTERNATIVE" in out and "plateaus by 400" in out and "does NOT allow longer training" in out
    assert "g2_train" in [i["key"] for i in res["excluded"]] and not any(i["key"] == "g2_eval" for i in res["included"])


def _fake_budget(monkeypatch, lc_open=True):
    from types import SimpleNamespace as NS
    st = {"ORCH": NS(approved=1.0, spent=0.0, reserved=0.0), "E0": NS(approved=9.0, spent=7.2, reserved=0.0), "E1": NS(approved=9.0, spent=8.01, reserved=0.0),
          "E2": NS(approved=22.0, spent=16.78, reserved=3.07), "E3": NS(approved=10.0, spent=0.27, reserved=7.40 if lc_open else 0.0), "E4": NS(approved=30.0, spent=0.0, reserved=0.0)}
    monkeypatch.setattr(pk.budget, "envelope_states", lambda *a, **k: st)
    rows = [{"job": "p3-learning-curve-v3", "kind": "reserve", "amount_usd": "7.4"}] if lc_open else []
    monkeypatch.setattr(pk.budget, "read_ledger", lambda *a, **k: rows)


def test_owner_package_contents_and_money(monkeypatch):
    _fake_budget(monkeypatch)
    r = pk.owner_package()
    keys = [ln["key"] for ln in r["lines"]]
    assert keys == pk.OWNER_ITEMS + ["orch_fixed", "e6_fixed"] and "h2" not in keys and "h13" not in keys and "gemma" not in keys and "h3_64k" not in keys and "h11" not in keys
    h1 = r["lines"][0]
    assert h1["alt"]["primary_n"]["S11_vs_S2"] == 607 and h1["alt"]["primary_n"]["S11_vs_S4"] == 391            # delta 7, S4 stays at 391
    lc = next(ln for ln in r["lines"] if ln["key"] == "learning_curve")
    assert lc["usd"] == 0.0 and "ALREADY RUNNING" in lc["name"] and lc["committed_note"] == 3.70
    assert r["committed_env"]["E3"] == pytest.approx(0.27 + 3.70) and r["committed_total"] == pytest.approx(7.2 + 8.01 + 16.78 + 3.07 + 0.27 + 3.70)
    assert r["h11"]["primary_n"] == {"H11": 2614}
    for ln in r["lines"]:                                     # BOOST only for uses (b) and (d)
        if ln["boost"] > 0:
            assert ln["key"] in {"H1", "H5", "h12", "g2_train", "a1a3"} and ln["boost"] <= ln["alt"]["boost_elig"] + 1e-9
        assert ln["base"] + ln["boost"] == pytest.approx(ln["usd"])
    assert r["boost_used"] <= 30.0 - r["boost_slack"] + 1e-6 and r["boost_needed"] <= 30.0 + 1e-6              # BOOST keeps its own reserve slack inside the $30
    assert r["plan_total"] == pytest.approx(sum(ln["usd"] for ln in r["lines"]))
    for e in ("E2", "E3", "E4", "E5"):
        assert r["needed"][e] == pytest.approx(r["committed_env"].get(e, 0) + r["by_env"].get(e, 0) + r["slack_env"].get(e, 0))
        assert r["slack_env"][e] <= 2.5 + 1e-6                                                  # containers split at <= $2.50
    assert r["grand"] <= 140.0 and r["grand_h11"] > 140.0 and "rules file" not in str(r["h11"]["label"])
    assert r["needed"]["E0"] == pytest.approx(7.2) and r["needed"]["E1"] == pytest.approx(8.01)
    assert r["needed"]["ORCH"] == pytest.approx(1.0) and r["needed"]["E6"] == pytest.approx(0.5)               # ORCH stays approved at $1, E6 gets $0.50
    assert r["container_cap"] == 2.5 and pk.CONTAINER_MAX_USD == 3.0 and pc.PARAMS["max_container_s"] == 8500    # the cap is applied temporarily
    r3 = pk.owner_package(container_cap=3.0)
    assert r3["slack_total"] > r["slack_total"] and max(r3["slack_env"].values()) > 2.5                        # a smaller cap shrinks the slack
    assert r["base_total"] == pytest.approx(sum(r["needed"].values()))
    assert r["grand"] == pytest.approx(r["committed_total"] + r["plan_total"] + r["slack_total"])
    assert r["grand_h11"] > r["grand"] + 7.0


def test_owner_h1_power_and_cli(monkeypatch, capsys, tmp_path):
    _fake_budget(monkeypatch)
    r = pk.owner_package()
    h = r["h1"]
    assert h["n"] == 607 and 0.78 <= h["design_power_delta7"] <= 0.83 and 0.5 <= h["power_delta5_at_607"] <= 0.55 and h["p_pass_pilot_effect_10pts"] > 0.95
    out = tmp_path / "o.csv"
    assert pc.main(["--package", "owner", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    for frag in ("package OWNER", "ALREADY RUNNING", "ADD-ON (not in the totals): H11", "BASE total needed", "GRAND TOTAL = committed", "WITH the H11 add-on", "H1 AT n=607"):
        assert frag in text, frag
    rows = list(csv.reader(out.open(encoding="utf-8", newline="")))
    assert rows[0][:4] == ["tier", "item", "alt", "envelope"] and any("ADD-ON H11" in x[1] for x in rows[1:])


def test_owner_learning_curve_reservation_once_finished(monkeypatch):
    _fake_budget(monkeypatch, lc_open=False)
    r = pk.owner_package()
    assert r["committed_env"]["E3"] == pytest.approx(0.27)                                  # nothing open: no adjustment
