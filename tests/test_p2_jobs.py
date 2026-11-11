"""P2 test-split job templates (jobs/make_p2_jobs.py): templates only, no test data read, containers <= $3, exact item ranges, envelopes per package."""
import builtins
import csv
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
spec_ = importlib.util.spec_from_file_location("make_p2_jobs", ROOT / "jobs" / "make_p2_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import launch  # noqa: E402

SIZES = mk.load_sizes(None)
PK = {p: mk.load_package(p) for p in ("owner", "96", "126", "full")}
PLANS = {p: mk.plan(SIZES, PK[p]) for p in PK}
PLANS["none"] = mk.plan(SIZES, None)


def by_prefix(entries, prefix):
    return [(s, m) for s, m in entries if s["name"].startswith(prefix)]


def ranges(entries, key_count=True):
    """(offset, count) of each spec's single items_uri entry, sorted."""
    return sorted((s["args"]["items_uri"][0]["offset"], s["args"]["items_uri"][0]["n_items"]) for s, _ in entries)


def assert_partition(rs, n):
    off = 0
    for o, c in rs:
        assert o == off and c > 0, (rs, n)
        off += c
    assert off == n


def test_defaults_are_the_96_package_values():
    assert SIZES == {"H1": 1191, "S4": 391, "H5": 249, "tost": 191, "H2": 200, "H10": 860}
    assert set(PK["96"]) == {"H1", "H5"}                                            # H2 / H10 are not in the "fits $96" package
    assert {"H1", "H5", "H2", "H10"} <= set(PK["full"])


def test_sizes_file_overrides(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("H1: 607\nH5: 696\nbogus: 1\n")
    s = mk.load_sizes(str(f))
    assert s["H1"] == 607 and s["H5"] == 696 and s["S4"] == 391 and "bogus" not in s
    j = tmp_path / "s.json"
    j.write_text(json.dumps({"H2": 150}))
    assert mk.load_sizes(str(j))["H2"] == 150


def test_no_test_data_is_read_or_generated(monkeypatch):
    real_open = builtins.open
    seen = []

    def guard(file, *a, **k):
        seen.append(str(file))
        low = str(file).replace("\\", "/")
        assert "opa_data" not in low and "/test/" not in low and "items.jsonl" not in low, low
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(launch, "prereg_tagged", lambda: (_ for _ in ()).throw(AssertionError("the tag state must not matter for templates")))
    mk.plan(SIZES, PK["96"])
    for s, _ in PLANS["full"]:
        for u in (s["args"]["items_uri"] if isinstance(s["args"].get("items_uri"), list) else []):
            assert "/test/" in u["uri"] and u["uri"].startswith("gs://{bucket}/opa/data/") and u["uri"].endswith("/items.jsonl.gz")
    assert not [e for e in PLANS["full"] if e[0]["entrypoint"].endswith(("datagen", "build"))]       # generation is a documented local command


@pytest.mark.parametrize("p", ["96", "126", "full", "none"])
def test_every_spec_is_a_valid_test_template(p):
    for s, m in PLANS[p]:
        assert launch.validate(mk.filled(s)) == [], s["name"]
        assert s["split"] == "test" and s["max_api_usd"] == 0 and s["compute"].get("gpu", "L4") == "L4" and s["seed"] >= 100000
        assert s["envelope"] in ("E2", "BOOST") and any(n.startswith("package line:") or n.startswith("all items") for n in s["notes"])
        assert m["est"] <= mk.MAX_CONTAINER_USD + 1e-9, (s["name"], m["est"])              # every container <= $3
        assert s["max_usd"] >= 2 * m["est_raw"] - 0.011


def test_item_ranges_partition_each_cell_exactly():
    for p in ("96", "full"):
        E = PLANS[p]
        for kind in ("embed-lb", "sk-lb", "h1-lb"):
            assert_partition(ranges(by_prefix(E, f"p2-{kind}-")), SIZES["H1"])
        assert_partition(ranges(by_prefix(E, "p2-h1-s4-")), SIZES["S4"])
        for kind in ("embed-mt", "sk-mt", "h5-mt"):
            assert_partition(ranges(by_prefix(E, f"p2-{kind}-")), SIZES["H5"])
        s15 = [(s["args"]["items_uri"][0]["offset"], s["args"]["items_uri"][0]["n_items"]) for s, _ in by_prefix(E, "p2-h5-mt-") if "S15" in s["args"]["systems"]]
        assert_partition(sorted(s15), SIZES["tost"])                                          # S15 on exactly the first n_tost items
    E = PLANS["full"]
    for cond in ("d2", "d4"):
        assert_partition(ranges(by_prefix(E, f"p2-h2-{cond}-")), SIZES["H2"])
    for arm in ("pin", "nopin"):
        tot = {}
        for s, _ in by_prefix(E, f"p2-h10-{arm}-"):
            for u in s["args"]["items_uri"]:
                tot.setdefault(u["uri"].split("/")[-2], []).append((u["offset"], u["n_items"]))
        assert sorted(tot) == ["r3", "r4", "r6"]
        for rs in tot.values():
            assert_partition(sorted(rs), sum(c for _, c in rs))
        assert sum(c for rs in tot.values() for _, c in rs) == SIZES["H10"]


def test_runs_read_exactly_one_vector_file_and_one_skeleton_directory():
    E = PLANS["96"]
    prep = {}
    for s, _ in E:
        n = s["name"]
        if n.startswith(("p2-embed-", "p2-sk-")):
            kind, cell, tag = n.split("-")[1:4]
            prep[(kind, cell, tag)] = (s["args"]["items_uri"][0]["offset"], s["args"]["items_uri"][0]["n_items"])
    for s, _ in E:
        if not s["entrypoint"].endswith("run_system"):
            continue
        u = s["args"]["items_uri"][0]
        emb = re.search(r"(P2_EMBED_[A-Z]+_C\d\d)_RUN_ID", s["args"]["embedder"]).group(1)
        cell, idx = emb.split("_")[2].lower(), emb[-2:]
        o, c = prep[("embed", cell, f"c{idx}")]
        assert o <= u["offset"] and u["offset"] + u["n_items"] <= o + c, s["name"]              # inside its embed chunk
        if "skeleton_llm_uri" in s["args"]:
            sk = re.search(r"(P2_SK_[A-Z]+_C\d\d)_RUN_ID", s["args"]["skeleton_llm_uri"]).group(1)
            assert sk[-2:] == idx and prep[("sk", cell, f"c{idx}")] == (o, c)                  # same chunk partition for skeletons


def test_workers_systems_and_opts():
    E = PLANS["96"]
    for s, _ in by_prefix(E, "p2-h1-lb-"):
        assert s["args"]["systems"] == ["S11", "S2", "S8"] and s["args"]["workers"] == 16 and s["args"]["opts"] == {"lambda": 2.0}
        assert "skeleton_llm_uri" in s["args"]
    for s, _ in by_prefix(E, "p2-h1-s4-"):
        assert s["args"]["systems"] == ["S4"] and s["args"]["workers"] == 16 and "skeleton_llm_uri" not in s["args"]
    for s, _ in by_prefix(E, "p2-h5-mt-"):
        assert s["args"]["systems"][:2] == ["S12", "S14"] and s["args"]["workers"] == 8 and s["args"]["opts"] == {"lambda": 2.0}
    for s, _ in by_prefix(E, "p2-sk-"):
        assert s["args"]["workers"] == 24 and s["entrypoint"].endswith("build_skeletons") and any("sllm-v2" in n for n in s["notes"])
    F = PLANS["full"]
    assert {s["args"]["condition"] for s, _ in by_prefix(F, "p2-h2-")} == {"D2", "D4"}
    for s, _ in by_prefix(F, "p2-h10-"):
        assert s["args"]["M"] == 2 and any("thrash-v2" in n for n in s["notes"]) and s["args"]["pinning"] == s["name"].startswith("p2-h10-pin-")


def test_package_drops_items_and_funds_boost_first():
    assert not by_prefix(PLANS["96"], "p2-h2-") and not by_prefix(PLANS["96"], "p2-h10-")
    assert by_prefix(PLANS["full"], "p2-h2-") and by_prefix(PLANS["full"], "p2-h10-")
    pk96 = PK["96"]
    boost = [m for _, m in PLANS["96"] if m["item_key"] == "H1" and m["envelope"] == "BOOST"]
    assert boost and sum(m["est"] for m in boost) <= pk96["H1"]["boost"] + 1e-6
    assert sum(m["est"] for m in boost) > pk96["H1"]["boost"] - mk.MAX_CONTAINER_USD            # the BOOST dollars are used up to one container
    h1 = [m for _, m in PLANS["96"] if m["item_key"] == "H1"]
    first_e2 = next(i for i, m in enumerate(h1) if m["envelope"] == "E2")
    assert all(m["envelope"] == "E2" for m in h1[first_e2:])                                    # BOOST containers come first
    assert all(m["envelope"] == "E2" for _, m in PLANS["96"] if m["item_key"] == "H5")           # package 96: no BOOST on H5
    assert all(s["envelope"] == "E2" for s, _ in PLANS["none"])
    boosted = [s for s, _ in PLANS["96"] if s["envelope"] == "BOOST"]
    assert boosted and all(any("use (b)" in n for n in s["notes"]) for s in boosted)


def test_totals_track_the_package_lines():
    for k in ("H1", "H5"):
        tot = sum(m["est"] for _, m in PLANS["96"] if m["item_key"] == k)
        assert abs(tot - PK["96"][k]["usd"]) / PK["96"][k]["usd"] < 0.03                         # containers add only a few start-ups


def test_reduced_h1_label_sets_n(tmp_path):
    f = tmp_path / "pk.csv"
    rows = [{"package": "x", "status": "reduced", "tier": 1, "item": "H1 (+ H3 S11 - S8 leaf)", "alt": "delta 7, n=607", "envelope": "E2", "line": "l", "usd": 10,
             "base_usd": 10, "boost_usd": 0, "raise_usd": 0, "containers": 1, "basis": "m", "formula": ""}]
    with f.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    pk = mk.load_package("x", f)
    assert pk["H1"]["n"] == 607
    E = mk.plan(SIZES, pk)
    assert_partition(ranges(by_prefix(E, "p2-h1-lb-")), 607) and True
    assert not by_prefix(E, "p2-h5-")


def test_datagen_commands_are_local_and_use_test_seeds():
    cmds = mk.datagen_commands(SIZES)
    assert any("--dataset lbmhqa --split test --cell late-h2-32k-a0 --n 1191 --seed-start 100000" in c for c in cmds)
    assert any("--dataset mtlbmhqa --split test --cell T4-h3-32k-cross --n 249" in c for c in cmds)
    assert any("--dataset drift --split test --cell s50 --n 200" in c for c in cmds)
    assert sum("--dataset thrash" in c and "--split test" in c and "build" in c for c in cmds) == 3
    assert any(c.startswith("python -m opa.data.upload --dataset lbmhqa --split test --version v2") for c in cmds)


def test_deterministic_and_launch_layout(tmp_path):
    a = [mk.dump(s, {}) for s, _ in mk.plan(SIZES, PK["96"])]
    b = [mk.dump(s, {}) for s, _ in mk.plan(SIZES, PK["96"])]
    assert a == b and all(b"\r" not in x for x in a)
    E = PLANS["96"]
    phs = mk.placeholders(E)
    assert len(phs) == 2 * (len(by_prefix(E, "p2-embed-lb-")) + len(by_prefix(E, "p2-embed-mt-"))) and all(p.startswith("P2_") for p in phs)
    ids = {p: f"RUN-{i}" for i, p in enumerate(sorted(phs))}
    half = dict(list(ids.items())[: len(ids) // 2])
    w, sk = mk.write_launch(E, half, tmp_path / "l")
    assert sk and len(w) + len(sk) == len(E)
    w2, sk2 = mk.write_launch(E, ids, tmp_path / "l")
    assert not sk2 and len(w2) == len(E)
    assert json.loads((tmp_path / "l" / "run_ids.json").read_text()) == ids
    assert not mk.p0.unresolved((tmp_path / "l" / "p2-h1-lb-c01.yaml").read_text())


def test_cli_writes_templates_to_a_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mk, "OUT_DIR", tmp_path / "p2")
    assert mk.main(["--package", "96"]) == 0
    files = sorted(p.name for p in (tmp_path / "p2").glob("p2-*.yaml"))
    assert len(files) == len(PLANS["96"])
    out = capsys.readouterr().out
    assert "TOTAL estimate" in out and "P2.1 local data generation" in out and "package line" in out


def test_owner_package_jobs(tmp_path, monkeypatch, capsys):
    E = PLANS["owner"]
    assert mk.MAX_CONTAINER_USD == 2.5 and all(m["est"] <= 2.5 + 1e-9 for _, m in E)
    assert set(PK["owner"]) == {"H1", "H5", "H10"}                                       # H2 deferred, not included
    assert not by_prefix(E, "p2-h2-")
    n1, n5, n10 = PK["owner"]["H1"]["n"], PK["owner"]["H5"]["n"], 860
    assert n1 == 607 and n5 == 249
    for kind in ("embed-lb", "sk-lb", "h1-lb"):
        assert_partition(ranges(by_prefix(E, f"p2-{kind}-")), 607)
    assert_partition(ranges(by_prefix(E, "p2-h1-s4-")), 391)                              # S4 stays on 391 items
    for kind in ("embed-mt", "sk-mt", "h5-mt"):
        assert_partition(ranges(by_prefix(E, f"p2-{kind}-")), 249)
    s15 = [(s["args"]["items_uri"][0]["offset"], s["args"]["items_uri"][0]["n_items"]) for s, _ in by_prefix(E, "p2-h5-mt-") if "S15" in s["args"]["systems"]]
    assert_partition(sorted(s15), 191)
    for arm in ("pin", "nopin"):
        tot = sum(u["n_items"] for s, _ in by_prefix(E, f"p2-h10-{arm}-") for u in s["args"]["items_uri"])
        assert tot == n10
    for s, _ in by_prefix(E, "p2-h1-lb-"):
        assert s["args"]["workers"] == 16 and s["args"]["systems"] == ["S11", "S2", "S8"]
    for s, _ in by_prefix(E, "p2-h1-s4-"):
        assert s["args"]["workers"] == 16
    for s, _ in by_prefix(E, "p2-h5-mt-"):
        assert s["args"]["workers"] == 8 and "T4-h3-32k-cross" in s["args"]["items_uri"][0]["uri"]
    for s, _ in by_prefix(E, "p2-sk-"):
        assert s["args"]["workers"] == 24
    boost = [m for _, m in E if m["envelope"] == "BOOST"]
    assert boost and all(m["item_key"] == "H1" for m in boost)                           # BOOST-funded H1 containers carry envelope BOOST
    assert sum(m["est"] for m in boost) <= PK["owner"]["H1"]["boost"] + 1e-6
    assert all(m["envelope"] == "E2" for _, m in E if m["item_key"] in ("H5", "H10"))
    cmds = mk.datagen_commands({**SIZES, "H1": 607}, ("H1", "H5", "H10"))
    assert any("--n 607 --seed-start 100000" in c for c in cmds) and not any("drift" in c for c in cmds)
    monkeypatch.setattr(mk, "OUT_DIR", tmp_path / "p2")
    assert mk.main([]) == 0                                                              # default package is owner
    out = capsys.readouterr().out
    assert "P2 templates for package owner" in out and "P2.1 local data generation" in out


def test_main_prints_datagen_with_the_effective_sizes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mk, "OUT_DIR", tmp_path / "p2")
    assert mk.main(["--package", "owner"]) == 0
    out = capsys.readouterr().out
    assert "--cell late-h2-32k-a0 --n 607 --seed-start 100000" in out and "--n 1191" not in out and "drift" not in out


def test_seeds_are_distinct_across_series_and_start_at_100000():
    for p in ("owner", "96", "full", "none"):
        E = PLANS[p]
        seeds = [s["seed"] for s, _ in E]
        assert len(seeds) == len(set(seeds)) and min(seeds) >= 100000, p                    # every container has its own seed
        for s, _ in E:
            kind, idx = s["name"].rsplit("-c", 1)
            assert s["seed"] == 100000 + 100 * mk.SERIES.index(kind) + int(idx) - 1
    assert len({100000 + 100 * i for i in range(len(mk.SERIES))}) == len(mk.SERIES)


def test_s4_p2_specs_use_workers_16_and_measured_price():
    for name in PLANS:
        specs = by_prefix(PLANS[name], "p2-h1-s4-")
        assert specs
        for s, _ in specs:
            assert s["args"]["workers"] == 16
            assert "workers-16" in s["estimate_formula"] and "workers-8" not in s["estimate_formula"]
            assert "S4 at workers 16" in " ".join(s["notes"])
