"""opa.analysis.plan_costs: prices the remaining PLAN_V3 work from measured unit costs only (no Modal)."""
import csv
import json
import math
from pathlib import Path

import pytest

from opa import launch
from opa.analysis import plan_costs as pc

ROOT = Path(__file__).resolve().parents[1]
ROWS = pc.build_lines()


def test_rate_and_startup():
    assert math.isclose(pc.RATE, launch.rate_per_second({"kind": "gpu", "gpu": "L4", "image": "vllm", "cpu": 2.0, "memory_gib": 16},
                                                         pc.budget.load_prices()))
    assert abs(pc.RATE - 0.000345) < 2e-6 and pc.PARAMS["start_usd"] == 0.066


def test_rows_well_formed_and_grouped():
    assert [r for r in ROWS if not set(pc.FIELDS) <= set(r)] == []
    phases = []
    for r in ROWS:
        if not phases or phases[-1] != r["phase"]:
            phases.append(r["phase"])
        assert r["basis"] in ("measured", "assumption", "TBD") and r["counted"] in ("yes", "alt") and r["usd"] >= 0
        assert r["envelope"] in ("E2", "E3", "E4", "E5", "E6") and r["formula"]
    assert phases == ["P1 remaining", "P2 test split", "P3", "P4", "P5", "P6"]
    env = {"P1 remaining": "E2", "P2 test split": "E2", "P3": "E3", "P4": "E4", "P5": "E5", "P6": "E6"}
    assert all(env[r["phase"]] == r["envelope"] for r in ROWS)


def test_required_lines_present():
    names = " | ".join(r["line"] for r in ROWS)
    for frag in ("H5 T8 oracle", "conditional arms", "H3 qualifying-cell top-up", "H1 cell late h2: S11, S2, S8 at n_H1", "S4 at n_S4", "H3 R2 cell",
                 "H3 S11 - S8", "S12, S14 at n_H5", "S15 at n_tost", "H8", "H10 thrash", "H2 drift D2", "H2 drift D4", "natural benchmarks",
                 "test-data generation", "learning-curve check", "U6-U8", "remedies at full n", "training-free variants", "H7",
                 "V3 to L=400", "V3 to L=800", "V3 to L=1200", "V3 to L=2000", "A1 to L=", "A3 to L=", "G2 fresh dev eval", "evals V3", "Track C",
                 "Gemma replication", "embedder robustness", "B sweep", "reproducibility", "bundle assembly"):
        assert frag in names, frag
    tbd = [r for r in ROWS if r["line"].startswith("natural benchmarks")][0]
    assert tbd["basis"] == "TBD" and tbd["usd"] == 0


def test_measured_inputs_match_runs_when_present():
    p = pc.PARAMS
    for frag, subs in pc.MEASURED_RUNS.items():
        d = next(iter((ROOT / "runs").glob(f"*{frag}")), None)
        if d is None or not (d / "result.json").exists():
            continue
        runs = json.loads((d / "result.json").read_text(encoding="utf-8"))["metrics"]["runs"]
        for sub, (wall, n) in subs.items():
            assert abs(runs[sub]["wall_seconds"] - wall) < 1.0 and runs[sub]["n"] == n
    assert math.isclose(p["s_per_item_lb"]["S11"], 921 / 50) and math.isclose(p["eval_s_item"]["V3"], 171 / 32)


def test_sample_sizes_consistent_with_csv():
    assert pc.check_sample_sizes() == []
    assert pc.PARAMS["n"]["H1"] == 1191 and pc.PARAMS["n"]["S4"] == 391 and pc.PARAMS["n"]["H5_delta3"] == 696
    assert pc.PARAMS["n"]["H5_delta5"] == 249 and pc.PARAMS["n"]["tost"] == 191 and pc.PARAMS["n"]["H10"] == 860 and pc.PARAMS["n"]["H2"] == 200


def test_arithmetic_of_a_few_lines():
    n = pc.PARAMS["n"]["H1"]
    sec = n * (921 / 50 + 46 / 50 + 854 / 50)
    r = next(x for x in ROWS if x["line"].startswith("H1 cell late h2: S11, S2, S8"))
    assert math.isclose(r["usd"], sec * pc.RATE + math.ceil(sec / pc.PARAMS["max_container_s"]) * 0.066, abs_tol=1e-3)
    sk = next(x for x in ROWS if x["line"] == "H1 cell late h2: skeletons")
    assert math.isclose(sk["usd"], n * 0.0336, abs_tol=1e-3)
    lc = next(x for x in ROWS if x["line"].startswith("learning-curve check"))
    assert math.isclose(lc["usd"], (314 + 400 * 21 + 4 * 64 * 171 / 32) * pc.RATE, abs_tol=1e-3)           # about $3.5
    v3 = {L: next(x for x in ROWS if x["line"] == f"V3 to L={L} steps")["usd"] for L in (400, 800, 1200, 2000)}
    assert v3[400] < v3[800] < v3[1200] < v3[2000]


def test_alt_rows_are_not_counted_and_totals():
    t = pc.totals(ROWS)
    assert all(r["counted"] == "alt" for r in ROWS if r["scenario"] in ("H5 delta=5 (n 249)", "L=400", "L=1200", "L=2000"))
    assert [r["counted"] for r in ROWS if r["line"].startswith("B sweep")] == ["alt"]                 # dropped from the base (owner)
    assert all(r["counted"] == "yes" for r in ROWS if r["scenario"] == "L=800")
    assert math.isclose(sum(t["base"].values()), sum(r["usd"] for r in ROWS if r["counted"] == "yes"), abs_tol=1e-6)
    assert t["per_L"][800] == t["base"]
    assert t["per_L"][400]["E4"] < t["base"]["E4"] < t["per_L"][1200]["E4"] < t["per_L"][2000]["E4"]
    assert math.isclose(t["per_L"][400]["E2"], t["base"]["E2"])


def test_cli_writes_csv(tmp_path, capsys):
    out = tmp_path / "plan.csv"
    assert pc.main(["--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8", newline="")))
    assert len(rows) == len(ROWS) and list(rows[0]) == pc.FIELDS and b"\r" not in out.read_bytes()
    text = capsys.readouterr().out
    assert "BASE totals per envelope" in text and "committed so far" in text and "Per training length L" in text


def test_current_state_reads_ledger_only():
    s = pc.current_state()
    assert "E2" in s["approved"] and s["committed"] >= 0


def test_skeleton_rate_uses_all_five_w24_runs():
    runs = pc.SKELETON_RUNS_W24
    assert len(runs) == 5
    want = (sum(u for u, _ in runs.values()) - 5 * 0.066) / sum(i for _, i in runs.values())
    assert math.isclose(pc.PARAMS["skeleton_net_usd_per_item_w24"], want, rel_tol=1e-3)
    for frag, (usd, items) in runs.items():
        d = next(iter((ROOT / "runs").glob(f"A-sllm-*{frag}")), None)
        if d is None:
            continue
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        assert abs(r["compute_usd"] - usd) < 0.01 and r["metrics"]["items"] == items


def test_bsweep_startups_net_vs_gross():
    n = next(r for r in pc.build_lines(concurrency=24, bench={"lb": {}, "mt": {}}) if r["line"].startswith("B sweep"))
    p = pc.PARAMS
    want = 2 * p["n_bsweep"] * (p["embed_usd_per_item"] + p["skeleton_net_usd_per_item_w24"] + p["s_per_item_lb"]["S11"] * pc.RATE) + 4 * 0.066
    assert math.isclose(n["usd"], want, abs_tol=1e-3)            # 2 S11 start-ups + 2 skeleton start-ups (skeletons priced net of start-up)


def test_load_bench_refuses_duplicates(tmp_path):
    import pytest
    import yaml
    for nm in ("a", "b"):
        r = tmp_path / f"A-{nm}"
        r.mkdir()
        (r / "job.yaml").write_text(yaml.safe_dump({"name": "bench-lb-w16"}))
        (r / "result.json").write_text(json.dumps({"status": "ok", "metrics": {"runs": {"S2/r0": {"wall_seconds": 24.0, "n": 48}}}}))
    with pytest.raises(ValueError, match="duplicate bench runs"):
        pc.load_bench(16, tmp_path)
