"""Dev-only job specs for the H5 T8 h3 cross candidate cell (jobs/make_h5cell_jobs.py)."""
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
spec_ = importlib.util.spec_from_file_location("make_h5cell_jobs", ROOT / "jobs" / "make_h5cell_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import budget, launch  # noqa: E402

E = mk.jobs()
BY = {m["name"]: (s, m) for s, m in E}
RX = re.compile(r"\b[A-Z][A-Z0-9_]*_RUN_ID\b")


def test_all_specs():
    assert [m["name"] for _, m in E] == mk.ORDER
    for s, _ in E:
        assert launch.validate(mk.filled(s)) == []
        assert s["split"] == "dev" and s["envelope"] == "E2" and s["max_api_usd"] == 0 and s["hypotheses"] == ["H5"] and s["seed"] == 0
        assert s["compute"].get("gpu", "L4") == "L4"
        assert not RX.search(yaml.safe_dump({k: v for k, v in s.items() if k != "args"})) and "{bucket}" in yaml.safe_dump(s["args"])
        assert [(u["n_items"], u["offset"]) for u in s["args"]["items_uri"]] == [(50, 5)]
        assert s["args"]["items_uri"][0]["uri"].split("/")[-2] == "T8-h3-32k-cross"
        assert s["max_usd"] >= 2 * BY[s["name"]][1]["raw"] - 0.011


def test_embed_and_skeleton():
    e, sk = BY["h5cell-embed"][0], BY["h5cell-sk-t8"][0]
    assert e["entrypoint"].endswith("embed_items") and e["args"]["embedder"] == "harrier"
    assert math.isclose(BY["h5cell-embed"][1]["raw"], 50 * 0.0018)
    assert sk["entrypoint"].endswith("build_skeletons") and sk["args"]["workers"] == 24 and any("sllm-v2" in n for n in sk["notes"])
    assert math.isclose(BY["h5cell-sk-t8"][1]["raw"], 50 * 0.026 + 0.066) and sk["max_usd"] == math.ceil(2 * (50 * 0.026 + 0.066) * 100) / 100


def test_oracle_and_conditional_arms():
    o, a = BY["h5cell-t8-oracle"], BY["h5cell-t8-arms"]
    for s in (o[0], a[0]):
        assert s["args"]["opts"] == {"lambda": 2.0} and s["args"]["workers"] == 8
        assert s["args"]["embedder"] == mk.VEC_URI and s["args"]["skeleton_llm_uri"] == mk.SK_URI
    assert o[0]["args"]["systems"] == ["S18"] and a[0]["args"]["systems"] == ["S12", "S14", "S15", "S16"]
    assert not o[1]["conditional"] and a[1]["conditional"] and any("CONDITIONAL" in n for n in a[0]["notes"])
    # same opts and structure as the launched T4 arms job (when present)
    t4 = ROOT / "jobs" / "launch" / "p1-mt-t4-arms.yaml"
    if t4.exists():
        t4s = yaml.safe_load(t4.read_text(encoding="utf-8"))
        assert t4s["args"]["opts"] == a[0]["args"]["opts"] and t4s["args"]["workers"] == a[0]["args"]["workers"]
    rate = launch.rate_per_second(o[0]["compute"], budget.load_prices())
    assert math.isclose(o[1]["raw"], 0.066 + 1142.2 * rate) and 0.0075 < (o[1]["raw"] - 0.066) / 50 < 0.0085        # ~ $0.0079/item
    assert o[0]["max_usd"] / rate > 1142 * 1.5


def test_reservation_and_files(tmp_path):
    rows = mk.reservation_check(E)
    assert [r["name"] for r in rows] == mk.ORDER and all(r["ok"] for r in rows)       # worst case incl. the conditional job
    spent = 0.0
    for s, m in E:
        assert spent + s["max_usd"] <= mk.E2_HEADROOM + 1e-9
        spent += s["estimate_usd"]
    for s, _ in E:
        assert (ROOT / "jobs" / f"{s['name']}.yaml").read_bytes() == mk.dump(s, {})
    w, sk = mk.write_launch(E, {mk.PH_VEC: "V1"}, tmp_path / "l")
    assert w == ["h5cell-embed", "h5cell-sk-t8"] and set(sk) == {"h5cell-t8-oracle", "h5cell-t8-arms"}
    w2, sk2 = mk.write_launch(E, {mk.PH_SK: "K1"}, tmp_path / "l")
    assert not sk2 and len(w2) == 4
    assert json.loads((tmp_path / "l" / "run_ids.json").read_text()) == {mk.PH_SK: "K1", mk.PH_VEC: "V1"}
