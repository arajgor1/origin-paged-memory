"""The P1 Track A pilot job specs (jobs/make_p1_jobs.py): two stages, both --primary-h variants, dev only, every S-llm reader points
at a rebuilt (sllm-v2) skeleton placeholder, frozen headroom, reservation rule in launch order for E1 and E2."""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
spec_ = importlib.util.spec_from_file_location("make_p1_jobs", ROOT / "jobs" / "make_p1_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import launch  # noqa: E402
from opa.data.build import v3_dev_cells  # noqa: E402
from opa.tracks.track_a.systems import SYSTEMS  # noqa: E402

VARIANTS = {None: mk.plan(None), 2: mk.plan(2), 3: mk.plan(3)}
REAL = [(h, s, m) for h, es in VARIANTS.items() for s, m in es if s is not None]
RX = re.compile(r"\b[A-Z][A-Z0-9_]*_RUN_ID\b")
SK_PLACEHOLDERS = {mk.PH_SK_H3, mk.PH_SK_LATE_H2, mk.PH_SK_LB, mk.PH_SK_MT}


def by(h):
    return {m["name"]: (s, m) for s, m in VARIANTS[h]}


def test_frozen_headroom_constants():
    assert mk.E1_HEADROOM == 2.33 and mk.E2_APPROVED == 22.00 and mk.BUDGET == {"E1": 2.33, "E2": 22.0}
    assert mk.E1_AT_STAGE_A == 5.65
    assert "budget" not in vars(mk) and "headroom" not in vars(mk)               # the live ledger is not read
    for h, s, _ in REAL:
        want = "E1 5.65, E2 22.00" if h is None else "E1 2.33, E2 22.00"
        assert any(f"frozen at generation: {want}" in n for n in s["notes"]), s["name"]


def test_stage_a_alone():
    names = [m["name"] for _, m in VARIANTS[None]]
    assert names == ["p1-embed-pilot", "p1-skeleton-lateh3", "p1-g0-s18-h3-recheck"] == mk.order(None)
    assert all(r["envelope"] == "E1" and r["ok"] for r in mk.reservation_check(VARIANTS[None], mk.BUDGET_STAGE_A))     # the record
    assert sum(m["est"] for _, m in VARIANTS[None]) < mk.E1_AT_STAGE_A
    g = by(None)["p1-g0-s18-h3-recheck"][0]
    assert g["args"]["system"] == "S18" and g["args"]["n_items"] == 50 and "late-h3-32k" in g["args"]["items_uri"]
    assert g["args"]["skeleton_llm_uri"] == mk.SK_H3_URI and g["hypotheses"] == ["G0"] and g["seed"] == 0


def test_embed_pilot_covers_every_variant():
    e = by(None)["p1-embed-pilot"][0]
    cells = sorted(u["uri"].split("/")[-2] for u in e["args"]["items_uri"])
    assert cells == sorted(["forward-h2-32k-a0", "mixed-h2-32k-a0", "forward-h3-32k-a0", "mixed-h3-32k-a0", "T4-h3-32k-cross"])
    assert all(u["n_items"] == 50 for u in e["args"]["items_uri"])
    assert not any("late-h" in c or "T8" in c or "T16" in c for c in cells)            # late cells come from p0-embed-dev
    # every pilot job's items have vectors: late cells in p0-embed-dev (50), the rest in the pilot npz
    for h in (2, 3):
        for name, (s, m) in by(h).items():
            if s is not None and s["entrypoint"].endswith("run_system"):
                for u in s["args"]["items_uri"] if isinstance(s["args"]["items_uri"], list) else [{"uri": s["args"]["items_uri"]}]:
                    cell = u["uri"].split("/")[-2]
                    if cell.startswith("late-"):
                        assert s["args"]["embedder"] == mk.p0.EMBED_URI, name
                    else:
                        assert s["args"]["embedder"] == mk.VEC_URI and cell in cells, (name, cell)


@pytest.mark.parametrize("h", [2, 3])
def test_variant_structure(h):
    b = by(h)
    names = [m["name"] for _, m in VARIANTS[h]]
    assert names == mk.order(h)
    late = b[f"p1-lb-late-h{h}"][0]
    want = ["S1", "S2", "S4", "S8", "S11"] + (["S18"] if h == 2 else [])
    assert late["args"]["systems"] == want and all(f"-h{h}-32k" in u["uri"] for u in late["args"]["items_uri"])
    for r in (1, 2):
        rep = b[f"p1-lb-late-h{h}-rep{r}"][0]
        assert rep["args"]["systems"] == ["S1", "S2", "S8"] and rep["seed"] == 0
        assert rep["args"]["skeleton_llm_uri"] == late["args"]["skeleton_llm_uri"]
    fm = b[f"p1-lb-forward-mixed-h{h}"][0]
    assert fm["args"]["systems"] == ["S11"] and [u["uri"].split("/")[-2] for u in fm["args"]["items_uri"]] == \
        [f"forward-h{h}-32k-a0", f"mixed-h{h}-32k-a0"]
    sk = {n: {u["uri"].split("/")[-2]: u["n_items"] for u in s["args"]["items_uri"]} for n, (s, m) in b.items() if n.startswith("p1-skeleton")}
    assert not any(n.startswith("trackb") for n in b) and not any(n in b for n in mk.STAGE_A_ORDER)      # stage A / Track B are in the ledger
    assert sk["p1-skeleton-lb"] == {f"forward-h{h}-32k-a0": 50, f"mixed-h{h}-32k-a0": 50}
    assert sk["p1-skeleton-mt"] == {"T4-h3-32k-cross": 50}
    if h == 2:
        assert sk["p1-skeleton-lateh2"] == {"late-h2-32k-a0": 50} and late["args"]["skeleton_llm_uri"] == mk.SK_LATE_H2_URI
    else:
        assert "p1-skeleton-lateh2" not in b and late["args"]["skeleton_llm_uri"] == mk.SK_H3_URI    # reuses the stage-A skeletons
    assert "p1-mt-oracle-t4" not in b and "p1-mt-t8-arms" not in b


def test_mt_arms_identical_across_variants_and_on_t4_h3():
    a2, a3 = by(2)["p1-mt-t4-arms"][0], by(3)["p1-mt-t4-arms"][0]
    for a in (a2, a3):
        assert a["args"]["systems"] == ["S12", "S14", "S15", "S16", "S18"]
        assert [u["uri"].split("/")[-2] for u in a["args"]["items_uri"]] == ["T4-h3-32k-cross"] and a["args"]["skeleton_llm_uri"] == mk.SK_MT_URI
    assert {k: v for k, v in a2.items() if k not in ("envelope", "notes")} == {k: v for k, v in a3.items() if k not in ("envelope", "notes")}
    assert [SYSTEMS[s][3] for s in ("S12", "S14", "S15", "S16")] == ["A2", "A1", "A3", "A0"]
    have = {(d, c) for d, c, _ in v3_dev_cells()}
    assert ("mtlbmhqa", "T4-h3-32k-cross") in have and not any(d == "mtlbmhqa" and "-h2-" in c for d, c in have)      # MT has h3 cells only


@pytest.mark.parametrize("h,spec,meta", REAL, ids=lambda x: str(x) if not isinstance(x, dict) else "")
def test_every_spec(h, spec, meta):
    assert launch.validate(mk.filled(spec)) == []
    assert spec["envelope"] in ("E1", "E2") and spec["split"] == "dev" and spec["max_api_usd"] == 0 and spec["seed"] == 0
    assert spec["compute"].get("gpu", "L4") in ("L4", "T4") and "$" in spec["estimate_formula"]
    text = yaml.safe_dump({k: v for k, v in spec.items() if k != "args"})
    assert not RX.search(text), spec["name"]
    assert "{bucket}" in yaml.safe_dump(spec["args"]) or spec["name"].startswith("p1-drift")
    for k, v in spec["args"].items():
        found = set(RX.findall(yaml.safe_dump(v)))
        if found:
            assert k in ("embedder", "skeleton_llm_uri"), (spec["name"], k)
            assert found <= SK_PLACEHOLDERS | {mk.PH_VEC, mk.p0.PH_EMBED}, (spec["name"], found)
    # every S-llm reader points at a rebuilt (sllm-v2) skeleton placeholder, never at an old run
    systems = spec["args"].get("systems") or ([spec["args"]["system"]] if "system" in spec["args"] else [])
    needs = any(s in SYSTEMS and SYSTEMS[s][0] in ("llm", "hyb", "oracle") for s in systems)
    if needs:
        uri = spec["args"]["skeleton_llm_uri"]
        assert isinstance(uri, str) and any(ph in uri for ph in SK_PLACEHOLDERS), spec["name"]
        assert "SKELETON_H2_RUN_ID" not in uri.replace("PILOT_SKELETON_LATEH2_RUN_ID", "")
    else:
        assert "skeleton_llm_uri" not in spec["args"]
    if spec["entrypoint"].endswith("build_skeletons"):
        assert any("sllm-v2" in n for n in spec["notes"])
        items = meta["items"]
        if h is None:                                           # stage A: exactly what was launched (workers 12, $0.028)
            assert math.isclose(meta["raw"], items * 0.028 + 0.066) and spec["max_usd"] == math.ceil(2 * (items * 0.028 + 0.066) * 100) / 100
            assert spec["args"]["workers"] == 12
        else:                                                   # stage B: measured $0.034, workers 24
            assert math.isclose(meta["raw"], items * 0.034 + 0.066) and spec["max_usd"] == math.ceil(2 * (items * 0.034 + 0.066) * 100) / 100
            assert mk.SKELETON_SOURCE_RUN in spec["estimate_formula"] and spec["args"]["workers"] == 24
            assert spec["estimate_formula"].count("max_usd = 2 x the estimate") == 1 and spec["estimate_formula"].rstrip().endswith(f"${2 * meta['raw']:.3f}")
            assert any("12-worker rate" in n for n in spec["notes"])
        assert all(u["n_items"] == 50 for u in spec["args"]["items_uri"])


@pytest.mark.parametrize("h", [2, 3])
def test_reservation_rule_both_envelopes(h):
    rows = mk.reservation_check(VARIANTS[h])
    assert [r["name"] for r in rows] == mk.order(h)
    assert all(r["ok"] and r["need"] <= mk.BUDGET[r["envelope"]] + 1e-9 for r in rows)
    envs = [r["envelope"] for r in rows]
    assert envs == sorted(envs)
    assert envs[0] == "E2"                                    # nothing in stage B fits E1's remaining $2.33
    first = VARIANTS[h][0][1]
    assert first["max"] > mk.E1_HEADROOM
    spent = {"E1": 0.0, "E2": 0.0}                            # independent recomputation from the specs
    for s, m in VARIANTS[h]:
        assert spent[m["envelope"]] + s["max_usd"] <= mk.BUDGET[m["envelope"]] + 1e-9, m["name"]
        spent[m["envelope"]] += s["estimate_usd"]


def test_drift_and_thrash():
    b = by(3)
    for c in ("d2", "d4"):
        s = b[f"p1-drift-{c}"][0]
        assert s["entrypoint"] == "opa.tracks.track_a2:run_condition" and s["args"]["condition"] == c.upper()
        assert s["args"]["n_items"] == 30 and s["args"]["workers"] == 30 and s["hypotheses"] == ["H2"]
    d2 = b["p1-drift-d2"]
    assert d2[0]["max_usd"] >= 2.5 * d2[1]["raw"] - 0.011
    for arm, pin in (("pin", True), ("nopin", False)):
        s = b[f"p1-thrash-{arm}"][0]
        assert s["entrypoint"] == "opa.tracks.track_a:run_thrash" and s["args"]["pinning"] is pin and s["args"]["M"] == 2
        assert [u["uri"].split("/")[-2] for u in s["args"]["items_uri"]] == ["r3", "r4", "r6"] and all(u["n_items"] == 30 for u in s["args"]["items_uri"])


def test_templates_on_disk_are_stage_a():
    on_disk = sorted(p.stem for p in (ROOT / "jobs").glob("p1-*.yaml"))
    assert on_disk == sorted(mk.STAGE_A_ORDER) or set(mk.STAGE_A_ORDER) <= set(on_disk)
    for s, _ in VARIANTS[None]:
        assert (ROOT / "jobs" / f"{s['name']}.yaml").read_bytes() == mk.dump(s, {})
        assert b"\r" not in mk.dump(s, {})


def test_deterministic():
    assert [mk.dump(s, {}) for s, _ in mk.plan(2) if s] == [mk.dump(s, {}) for s, _ in mk.plan(2) if s]


def test_launch_layout_merges_ids(tmp_path):
    out = tmp_path / "launch"
    out.mkdir()
    (out / "run_ids.json").write_text(json.dumps({"EMBED_RUN_ID": "E0"}))
    es = [e for e in VARIANTS[2] if e[0] is not None]
    w, sk = mk.write_launch(es, {mk.PH_VEC: "V1", mk.PH_SK_H3: "H3"}, out)
    assert "p1-skeleton-lateh2" in w and "p1-lb-late-h2" not in w
    assert sk["p1-lb-late-h2"] == [mk.PH_SK_LATE_H2]
    w2, sk2 = mk.write_launch(es, {mk.PH_SK_LB: "L1", mk.PH_SK_MT: "M1", mk.PH_SK_LATE_H2: "H2"}, out)
    assert not sk2 and len(w2) == len(es)
    txt = (out / "p1-mt-t4-arms.yaml").read_text()
    assert "runs/V1/embeddings.npz" in txt and "runs/M1/skeleton_llm" in txt and not mk.unresolved(txt)
    assert "runs/H2/skeleton_llm" in (out / "p1-lb-late-h2.yaml").read_text()
    assert json.loads((out / "run_ids.json").read_text())["EMBED_RUN_ID"] == "E0"


def test_stage_a_templates_equal_what_was_launched():
    ids_file = ROOT / "jobs" / "launch" / "run_ids.json"
    if not ids_file.exists():
        pytest.skip("no jobs/launch/run_ids.json")
    ids = json.loads(ids_file.read_text())
    subs = {k: ids.get(k) for k in mk.ALL_PH}
    for spec, _ in VARIANTS[None]:
        launched = ROOT / "jobs" / "launch" / f"{spec['name']}.yaml"
        if not launched.exists():
            continue
        mine = yaml.safe_load(mk.dump(spec, subs))
        assert not mk.unresolved(mk.dump(spec, subs).decode()), spec["name"]
        assert mine == yaml.safe_load(launched.read_text(encoding="utf-8")), spec["name"]          # field for field
        assert (ROOT / "jobs" / f"{spec['name']}.yaml").read_bytes() == mk.dump(spec, {})
