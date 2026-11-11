"""Dev-only job specs for the two candidate H3 cells (jobs/make_h3cells_jobs.py)."""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
spec_ = importlib.util.spec_from_file_location("make_h3cells_jobs", ROOT / "jobs" / "make_h3cells_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import budget, launch  # noqa: E402
from opa.tracks.track_a.systems import Fold, SYSTEMS  # noqa: E402

E = mk.jobs()
BY = {m["name"]: (s, m) for s, m in E}
RX = re.compile(r"\b[A-Z][A-Z0-9_]*_RUN_ID\b")


def test_names_order_and_validity():
    assert [m["name"] for _, m in E] == mk.ORDER
    for s, _ in E:
        assert launch.validate(mk.filled(s)) == []
        assert s["split"] == "dev" and s["max_api_usd"] == 0 and s["hypotheses"] == ["H3"] and s["seed"] == 0
        assert s["compute"].get("gpu", "L4") == "L4" and s["envelope"] in ("E1", "E2")
        assert not RX.search(yaml.safe_dump({k: v for k, v in s.items() if k not in ("args",)}))
        assert "{bucket}" in yaml.safe_dump(s["args"])


def test_fresh_items_50_to_79():
    for s, _ in E:
        uris = s["args"]["items_uri"]
        assert isinstance(uris, list)
        for u in uris:
            assert u["offset"] == 50 and u["n_items"] == 30


def test_embed_covers_both_cells_and_runs_read_it():
    e = BY["h3cells-embed"][0]
    assert e["entrypoint"] == "opa.tracks.track_a:embed_items"
    assert sorted(u["uri"].split("/")[-2] for u in e["args"]["items_uri"]) == ["late-h2-32k-a0", "late-h3-32k-a0"]
    for n in ("h3cells-late-h2-m2-s11-s18", "h3cells-late-h3-s11-s18"):
        assert BY[n][0]["args"]["embedder"] == mk.VEC_URI


def test_skeleton_jobs_sllm_v2_workers_and_price():
    for n, cell in (("h3cells-sk-late-h2", "late-h2-32k-a0"), ("h3cells-sk-late-h3", "late-h3-32k-a0")):
        s, m = BY[n]
        assert s["entrypoint"] == "opa.tracks.track_a:build_skeletons" and s["args"]["workers"] == 24
        assert [u["uri"].split("/")[-2] for u in s["args"]["items_uri"]] == [cell]
        assert math.isclose(m["raw"], 30 * 0.026 + 0.066)
        assert s["max_usd"] == math.ceil(2 * (30 * 0.026 + 0.066) * 100) / 100
        assert any("sllm-v2" in x for x in s["notes"])


def test_run_jobs_cells_systems_and_m():
    a, b = (BY[n][0] for n in ("h3cells-late-h2-m2-s11-s18", "h3cells-late-h3-s11-s18"))
    assert a["args"]["systems"] == ["S18", "S11"] and a["args"]["opts"] == {"lambda": 2.0, "M": 2}       # oracle under the same M as the cell
    assert b["args"]["systems"] == ["S11", "S18"] and "M" not in b["args"]["opts"]
    assert "late-h2-32k-a0" in a["args"]["items_uri"][0]["uri"] and a["args"]["skeleton_llm_uri"] == mk.SK2_URI
    assert "late-h3-32k-a0" in b["args"]["items_uri"][0]["uri"] and b["args"]["skeleton_llm_uri"] == mk.SK3_URI
    assert not any(n in BY for n in ("h3cells-late-h2-s11-m2", "h3cells-late-h2-s18"))
    import inspect
    assert 'c.opts.get("M", 4)' in inspect.getsource(Fold.__init__) and SYSTEMS["S11"][0] == "hyb"
    assert math.isclose(BY["h3cells-late-h3-s11-s18"][1]["raw"], 0.066 + 30 * (0.0064 + 0.0039)) and BY["h3cells-late-h3-s11-s18"][1]["items"] == 60
    am = BY["h3cells-late-h2-m2-s11-s18"]
    assert math.isclose(am[1]["raw"], 0.066 + 30 * (0.014 + 0.008)) and am[1]["items"] == 60 and am[0]["max_usd"] == 1.46
    assert any("read window from 3 origins to 1" in n for n in am[0]["notes"])
    rate = launch.rate_per_second(am[0]["compute"], budget.load_prices())
    assert am[0]["max_usd"] / rate > 2000 * 1.5                       # the derived timeout covers ~2,000 s with margin
    assert math.isclose(BY["h3cells-embed"][1]["raw"], 60 * 0.0018)


def test_reservation_rule():
    rows = mk.reservation_check(E)
    assert all(r["ok"] for r in rows)
    envs = [r["envelope"] for r in rows]
    assert envs == sorted(envs) and envs[0] == "E1"
    spent = {"E1": 0.0, "E2": 0.0}
    for s, m in E:
        assert spent[m["envelope"]] + s["max_usd"] <= mk.BUDGET[m["envelope"]] + 1e-9
        spent[m["envelope"]] += s["estimate_usd"]


def test_templates_on_disk_and_launch_layout(tmp_path):
    for s, _ in E:
        assert (ROOT / "jobs" / f"{s['name']}.yaml").read_bytes() == mk.dump(s, {})
    out = tmp_path / "launch"
    w, sk = mk.write_launch(E, {mk.PH_VEC: "V1", mk.PH_SK2: "K2"}, out)
    assert "h3cells-late-h2-m2-s11-s18" in w and sk == {"h3cells-late-h3-s11-s18": [mk.PH_SK3]}
    w2, sk2 = mk.write_launch(E, {mk.PH_SK3: "K3"}, out)
    assert not sk2 and len(w2) == len(E)
    assert json.loads((out / "run_ids.json").read_text()) == {mk.PH_SK2: "K2", mk.PH_SK3: "K3", mk.PH_VEC: "V1"}
