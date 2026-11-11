"""Throughput-benchmark specs (jobs/make_bench_jobs.py) and plan_costs' concurrency parameter."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
spec_ = importlib.util.spec_from_file_location("make_bench_jobs", ROOT / "jobs" / "make_bench_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import launch  # noqa: E402
from opa.analysis import plan_costs as pc  # noqa: E402

E = mk.jobs()
BY = {m["name"]: (s, m) for s, m in E}


def test_specs():
    assert [m["name"] for _, m in E] == mk.ORDER
    for s, _ in E:
        assert launch.validate(s) == [] and s["split"] == "dev" and s["max_api_usd"] == 0 and s["seed"] == 0
        assert s["compute"].get("gpu") == "L4" and s["args"]["opts"] == {"lambda": 2.0}
        assert any("throughput benchmark (owner request 2026-11-07); outcomes already seen; not a result" in n for n in s["notes"])
        assert "{bucket}" in yaml.safe_dump(s["args"]) and "RUN_ID" not in yaml.safe_dump(s)


def test_items_systems_workers_and_inputs():
    lb16, lb24, mt = (BY[n][0]["args"] for n in ("bench-lb-w16", "bench-lb-w24", "bench-mt-w24"))
    for a, w in ((lb16, 16), (lb24, 24)):
        assert a["systems"] == ["S2", "S8", "S11", "S4"] and a["workers"] == w
        assert a["items_uri"] == [{"uri": a["items_uri"][0]["uri"], "n_items": 48, "offset": 0}] and "late-h2-32k-a0" in a["items_uri"][0]["uri"]
        assert mk.LB_VEC in a["embedder"] and mk.LB_SK in a["skeleton_llm_uri"]
    assert mt["systems"] == ["S12", "S14"] and mt["workers"] == 24 and mt["items_uri"][0]["n_items"] == 24 and mt["items_uri"][0]["offset"] == 0
    assert "T4-h3-32k-cross" in mt["items_uri"][0]["uri"] and mk.MT_VEC in mt["embedder"] and mk.MT_SK in mt["skeleton_llm_uri"]
    # same inputs as the launched pilot jobs (when present)
    for name, ref in (("bench-lb-w16", "p1-lb-late-h2"), ("bench-mt-w24", "p1-mt-t4-arms")):
        f = ROOT / "jobs" / "launch" / f"{ref}.yaml"
        if f.exists():
            r = yaml.safe_load(f.read_text(encoding="utf-8"))["args"]
            a = BY[name][0]["args"]
            assert r["embedder"] == a["embedder"] and r["skeleton_llm_uri"] == a["skeleton_llm_uri"] and r["opts"] == a["opts"] and r["model"] == a["model"]
    for s, _ in E:                       # never the blind H3 candidate items 50-79
        assert all(u["offset"] + u["n_items"] <= 50 for u in s["args"]["items_uri"])


def test_pricing_is_workers8_seconds_plus_startup_max_2x():
    lb = pc.PARAMS["s_per_item_lb"]
    sec = 48 * sum(lb[s] for s in ("S2", "S8", "S11", "S4"))
    for n in ("bench-lb-w16", "bench-lb-w24"):
        s, m = BY[n]
        assert math.isclose(m["raw"], sec * pc.RATE + 0.066, abs_tol=1e-6) and s["max_usd"] == math.ceil(2 * m["raw"] * 100) / 100
    mt = pc.PARAMS["s_per_item_mt"]
    assert math.isclose(BY["bench-mt-w24"][1]["raw"], 24 * (mt["S12"] + mt["S14"]) * pc.RATE + 0.066, abs_tol=1e-6)


def test_reservation_and_envelopes():
    rows = mk.reservation_check(E)
    assert [r["envelope"] for r in rows] == ["E0", "E2", "E3"] and all(r["ok"] for r in rows)
    assert mk.BUDGET["E3"] == 10.00 and any("logged reallocation ruling" in n for n in BY["bench-lb-w24"][0]["notes"])
    for s, _ in E:
        assert (ROOT / "jobs" / f"{s['name']}.yaml").read_bytes() == mk.dump(s)


def test_plan_costs_concurrency(tmp_path):
    base = {r["line"]: r["usd"] for r in pc.build_lines(concurrency=8)}
    # no bench runs: only the skeleton price changes at 24 workers (net 24-worker measurement + one start-up per line)
    w24 = {r["line"]: r["usd"] for r in pc.build_lines(concurrency=24, bench={"lb": {}, "mt": {}})}
    assert w24["H1 cell late h2: skeletons"] < base["H1 cell late h2: skeletons"]
    net = pc.PARAMS["skeleton_net_usd_per_item_w24"]
    assert 0.024 < net < 0.027 and math.isclose(w24["H1 cell late h2: skeletons"], 1191 * net + 0.066, abs_tol=1e-3)
    assert w24["H1 cell late h2: S11, S2, S8 at n_H1"] == base["H1 cell late h2: S11, S2, S8 at n_H1"]
    # bench seconds per item are used when present
    b = {"lb": {"S11": 9.0, "S2": 0.5, "S8": 8.0}, "mt": {"S12": 11.0}}
    w16 = {r["line"]: r["usd"] for r in pc.build_lines(concurrency=16, bench=b)}
    assert w16["H1 cell late h2: S11, S2, S8 at n_H1"] < base["H1 cell late h2: S11, S2, S8 at n_H1"] / 1.5
    assert w16["H1 cell late h2: skeletons"] == base["H1 cell late h2: skeletons"]


def test_load_bench_reads_runs(tmp_path):
    run = tmp_path / "A-S2+S8-x"
    run.mkdir()
    (run / "job.yaml").write_text(yaml.safe_dump({"name": "bench-lb-w16"}))
    (run / "result.json").write_text(json.dumps({"status": "ok", "metrics": {"runs": {"S2/r0": {"wall_seconds": 24.0, "n": 48},
                                                                                      "S8/r0": {"wall_seconds": 480.0, "n": 48}}}}))
    other = tmp_path / "A-other"
    other.mkdir()
    (other / "job.yaml").write_text(yaml.safe_dump({"name": "bench-lb-w24"}))
    (other / "result.json").write_text(json.dumps({"status": "ok", "metrics": {"runs": {"S2/r0": {"wall_seconds": 1.0, "n": 48}}}}))
    b = pc.load_bench(16, tmp_path)
    assert b["lb"] == {"S2": 0.5, "S8": 10.0} and b["mt"] == {}
    assert pc.load_bench(24, tmp_path)["lb"] == {"S2": 1.0 / 48}
