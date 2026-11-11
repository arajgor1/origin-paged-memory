"""The v3 P0 job specs (jobs/make_p0_jobs.py): valid for the launcher, routed to the intended Modal function, priced
from the human-owned rate card, deterministic, and consistent with the dataset v2 cell names."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec_ = importlib.util.spec_from_file_location("make_p0_jobs", ROOT / "jobs" / "make_p0_jobs.py")
mk = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(mk)

from opa import budget, launch  # noqa: E402

JOBS = mk.all_jobs()
BY_NAME = {s["name"]: (s, m) for s, m in JOBS}


def test_names_unique_and_expected_set():
    names = [s["name"] for s, _ in JOBS]
    assert len(names) == len(set(names))
    for n in ["p0-embed-dev", "p0-embed-dev-aux", "p0-embedcheck", "p0-smoke-vllm", "p0-smoke-vllm-gemma", "p0-g0-s18",
              "p0-g0-s18-h2", "p0-g0-s1-rep1", "p0-g0-s1-rep2", "p0-g0-s1-rep3", "p0-skeleton-sllm",
              "p0-micropilot-g1", "p0-micropilot-g2", "p0-micropilot-g3"]:
        assert n in names


@pytest.mark.parametrize("spec", [s for s, _ in JOBS], ids=lambda s: s["name"])
def test_spec_valid_and_priced(spec):
    assert launch.validate(mk.filled(spec)) == []            # templates carry run-id placeholders on purpose
    assert spec["envelope"] == "E0" and spec["split"] == "dev" and spec["max_api_usd"] == 0
    assert spec["compute"].get("gpu", "L4") in ("L4", "T4")
    assert "seed" in spec and isinstance(spec["seed"], int)
    assert spec["estimate_formula"] and "$/s" in spec["estimate_formula"] and "L4" in spec["estimate_formula"]
    assert spec["max_usd"] >= spec["estimate_usd"]
    rate = launch.rate_per_second(spec["compute"], budget.load_prices())
    m = BY_NAME[spec["name"]][1]
    assert math.isclose(m["raw"], m["seconds"] * rate)
    assert spec["estimate_usd"] >= m["raw"] - 1e-9 and spec["estimate_usd"] - m["raw"] < 0.0101
    assert spec["max_usd"] >= (1.5 if spec["name"] == "p0-g0-s18-h2" else 2) * m["raw"] - 0.0101
    # the timeout the launcher derives leaves room for the estimated run
    assert spec["max_usd"] / rate >= m["seconds"] * 1.5
    for k in ("models",):
        assert spec[k]


def test_routing():
    for s, _ in JOBS:
        fn = launch.function_name(s["compute"])
        if s["entrypoint"] in ("opa.tracks.track_a:run_system", "opa.tracks.track_a:build_skeletons"):
            assert fn == "run_job_vllm", s["name"]
        else:
            assert fn == "run_job_gpu", s["name"]
    units = yaml.safe_load((ROOT / "jobs" / "p0_trackb_units.yaml").read_text())
    assert launch.validate(units) == [] and launch.function_name(units["compute"]) == "run_job_gpu"


def test_models_from_config():
    cfg = yaml.safe_load((ROOT / "config" / "project.yaml").read_text())["models"]
    assert mk.QWEN == cfg["subject"] and mk.GEMMA == cfg["track_a_replication"]
    assert mk.EMBEDDER == cfg["embedder"] and mk.EMBEDDER_BACKUP == cfg["embedder_backup"]
    for s, _ in JOBS:
        text = yaml.safe_dump(s)
        assert "claude" not in text.lower() and "anthropic" not in text.lower().replace("no anthropic", "")


def test_micropilot_shape():
    from opa.tracks.track_a.systems import SYSTEMS
    groups = mk.micropilot_groups()
    flat = [s for g in groups for s in g]
    assert sorted(flat) == sorted(mk.SYSTEM_ORDER) and len(flat) == 19          # every system exactly once, incl. S12F
    assert set(mk.SYSTEM_ORDER) == set(SYSTEMS) | {"S1", "S2", "S3", "S4", "S5", "S13"}
    for i, g in enumerate(groups, 1):
        s, m = BY_NAME[f"p0-micropilot-g{i}"]
        a = s["args"]
        assert a["systems"] == g and "system" not in a
        assert [(e["n_items"]) for e in a["items_uri"]] == [5, 5]             # 10 items per system: 5 LB + 5 MT
        assert "late-h3-32k" in a["items_uri"][0]["uri"] and "T8-h3-32k-cross" in a["items_uri"][1]["uri"]
        assert m["items"] == 10 * len(g)
        assert a["embedder"].startswith("precomputed:") and "EMBED_RUN_ID" in a["embedder"]
        assert a["model"]["name"] == "qwen" and a["model"]["max_model_len"] == 40960
        assert ("skeleton_llm_uri" in a) == any(mk.PROFILE[x][5] for x in g)
        assert m["seconds"] < 3000                                            # each stays far below its derived timeout
    # profile's skeleton need matches the real controllers
    for name, (sk, *_rest) in SYSTEMS.items():
        assert mk.PROFILE[name][5] == (sk in ("llm", "hyb", "oracle")), name
    loads = [sum(mk.system_time(x, 10)[0] for x in g) for g in groups]
    assert max(loads) < 1.15 * min(loads)                                     # balanced


def test_skeleton_per_uri_n():
    a = BY_NAME["p0-skeleton-sllm"][0]["args"]
    assert [(e["n_items"], e["uri"].split("/")[-2]) for e in a["items_uri"]] == [(50, "late-h3-32k-a0"), (5, "T8-h3-32k-cross")]


def test_g0_and_smokes():
    s, _ = BY_NAME["p0-g0-s18"]
    assert s["args"]["system"] == "S18" and s["args"]["n_items"] == 50 and "late-h3-32k" in s["args"]["items_uri"]
    assert "late-h2-32k" in BY_NAME["p0-g0-s18-h2"][0]["args"]["items_uri"] and BY_NAME["p0-g0-s18-h2"][1]["conditional"]
    seeds = [BY_NAME[f"p0-g0-s1-rep{r}"][0]["seed"] for r in (1, 2, 3)]
    assert seeds == [0, 0, 0]          # greedy decoding ignores the seed; the reps are a harness determinism check
    for r in (1, 2, 3):
        sp = BY_NAME[f"p0-g0-s1-rep{r}"][0]
        assert sp["hypotheses"] == ["G0"] and "harness determinism check: separate containers, same inputs" in sp["notes"]
        assert "repeats" not in sp["args"] and sp["args"]["system"] == "S1"
    g = BY_NAME["p0-smoke-vllm-gemma"][0]["args"]["model"]
    assert g["name"] == "gemma" and "--limit-mm-per-prompt" in g["extra_args"] and g["max_model_len"] == 40960
    for n in ("p0-smoke-vllm", "p0-smoke-vllm-gemma"):
        assert BY_NAME[n][0]["args"]["n_items"] == 2 and BY_NAME[n][0]["args"]["system"] == "S1"


def test_embed_jobs_cover_cells():
    core, aux = BY_NAME["p0-embed-dev"][0], BY_NAME["p0-embed-dev-aux"][0]
    assert core["entrypoint"] == "opa.tracks.track_a:embed_items" and core["args"]["embedder"] == "harrier"
    uris = " ".join(e["uri"] for e in core["args"]["items_uri"] + aux["args"]["items_uri"])
    for cell in ("late-h3-32k-a0", "late-h2-32k-a0", "T8-h3-32k-cross", "T4-h3-32k-cross-conf", "T16-h3-32k-nocross-conf"):
        assert f"/{cell}/" in uris
    ec = BY_NAME["p0-embedcheck"][0]
    assert ec["entrypoint"] == "opa.tracks.track_a:embedder_check" and ec["args"]["M"] == 4 and ec["args"]["offset"] == 0
    assert ec["args"]["baseline"] == "harrier-0.6b" and ec["args"]["challenger"] == "qwen3-embedding-4b"
    assert ec["compute"]["memory_gib"] >= 24                                  # fp32 CPU reference for the self-check


def test_cells_exist_in_dataset_v2_preset():
    from opa.data.build import v3_dev_cells
    have = {(d, c) for d, c, _ in v3_dev_cells()}
    for key, dc in mk.CELLS.items():
        assert dc in have, (key, dc)


def test_deterministic_and_lf(tmp_path):
    a = [mk.dump(s, {}) for s, _ in mk.all_jobs()]
    b = [mk.dump(s, {}) for s, _ in mk.all_jobs()]
    assert a == b and all(b"\r" not in x for x in a)
    for s, _ in JOBS:
        on_disk = (ROOT / "jobs" / f"{s['name']}.yaml").read_bytes()
        assert on_disk == mk.dump(s, {}), f"{s['name']}.yaml is stale: rerun python jobs/make_p0_jobs.py"


def test_placeholders_substituted():
    import copy
    s = copy.deepcopy(BY_NAME["p0-micropilot-g1"][0])
    s["args"]["skeleton_llm_uri"] = mk.SKELETON_URI
    out = mk.dump(s, {mk.PH_SK: "SK1", mk.PH_EMBED: "EM1"}).decode()
    assert "gs://{bucket}/opa/runs/EM1/embeddings.npz" in out and "SK1/skeleton_llm" in out


def test_hypothesis_tags():
    for s, _ in JOBS:
        n = s["name"]
        want = ["G0"] if n.startswith("p0-g0-") else ["P0-pilot-check"] if n == "p0-embedcheck" or n.startswith("p0-detprobe-") else ["P0-unit-costs"]
        assert s["hypotheses"] == want, n


def test_reservation_rule_in_launch_order():
    rows = mk.reservation_check()
    assert [r["name"] for r in rows] == mk.LAUNCH_ORDER
    assert mk.LAUNCH_ORDER[:6] == ["p0-smoke-vllm", "p0-smoke-vllm-gemma", "p0-embed-dev", "p0-embed-dev-aux", "p0_trackb_units",
                                   "p0-skeleton-sllm"]
    assert mk.LAUNCH_ORDER[-2:] == ["p0-skeleton-sllm-h2", "p0-g0-s18-h2"]
    assert not any("qwen3" in n for n in mk.LAUNCH_ORDER)       # a switch means an E0 top-up request, not a job
    for r in rows:
        assert r["prior_est"] + r["max"] <= 9.00 + 1e-9, r
    # independent recomputation from the generated specs
    by = {s["name"]: s for s, _ in JOBS}
    units = yaml.safe_load((ROOT / "jobs" / "p0_trackb_units.yaml").read_text())
    by["p0_trackb_units"] = units
    spent = 0.0
    for n in mk.LAUNCH_ORDER:
        assert spent + by[n]["max_usd"] <= 9.00 + 1e-9, n
        spent += by[n]["estimate_usd"]


def test_launch_dir_layout_and_run_ids(tmp_path):
    out = tmp_path / "launch"
    w, skipped = mk.write_launch(JOBS, {mk.PH_EMBED: "EMB1"}, out)
    assert "p0-embed-dev" in w and "p0-smoke-vllm" in w
    assert skipped["p0-micropilot-g1"] == ["SKELETON_RUN_ID"] and "p0-g0-s18-h2" in skipped        # refused, not written
    assert not (out / "p0-micropilot-g1.yaml").exists()
    w2, skipped2 = mk.write_launch(JOBS, {mk.PH_SK: "SK1"}, out)                                # a later call keeps EMB1
    assert "p0-micropilot-g1" in w2 and "p0-g0-s18" in w2
    txt = (out / "p0-micropilot-g1.yaml").read_text()
    assert "gs://{bucket}/opa/runs/EMB1/embeddings.npz" in txt and "SK1/skeleton_llm" in txt and not mk.unresolved(txt)
    assert skipped2 == {"p0-skeleton-sllm-h2": [], "p0-g0-s18-h2": ["SKELETON_H2_RUN_ID"]} or "p0-g0-s18-h2" in skipped2
    import json
    assert json.loads((out / "run_ids.json").read_text()) == {"EMBED_RUN_ID": "EMB1", "SKELETON_RUN_ID": "SK1"}
    assert mk.unresolved("a EMBED_RUN_ID b SKELETON_H2_RUN_ID") == ["EMBED_RUN_ID", "SKELETON_H2_RUN_ID"]
    # templates in jobs/ keep their placeholders
    assert "EMBED_RUN_ID" in (ROOT / "jobs" / "p0-micropilot-g1.yaml").read_text()


def test_placeholders_only_in_run_id_fields():
    import re
    rx = re.compile(r"[A-Z][A-Z0-9_]*_RUN_ID")
    for s, _ in JOBS:
        assert not rx.search(yaml.safe_dump(s.get("notes", []))), s["name"]
        assert not rx.search(s["estimate_formula"]), s["name"]
        text = yaml.safe_dump({k: v for k, v in s.items() if k != "args"})
        assert not rx.search(text), s["name"]
        # in args only the embedder / skeleton URIs may carry one, and never a substituted bucket
        for k, v in s["args"].items():
            if rx.search(yaml.safe_dump(v)):
                assert k in ("embedder", "embedders", "skeleton_llm_uri"), (s["name"], k)


def test_detprobes():
    for mode in ("bi", "w1"):
        for r in (1, 2):
            sp, m = BY_NAME[f"p0-detprobe-{mode}-r{r}"]
            a = sp["args"]
            assert a["system"] == "S1" and a["n_items"] == 10 and a["offset"] == 0 and sp["seed"] == 0 and sp["envelope"] == "E0"
            assert "late-h3-32k" in a["items_uri"] and sp["hypotheses"] == ["P0-pilot-check"]
            if mode == "bi":
                assert a["workers"] == 8 and a["model"]["env"] == {"VLLM_BATCH_INVARIANT": "1"}
                assert a["model"]["extra_args"] == ["--attention-backend", "FLASH_ATTN"]
                rate = launch.rate_per_second(sp["compute"], budget.load_prices())
                assert sp["max_usd"] / rate >= 180 + 10 * 12.4 * 4        # a 4x slowdown still fits
            else:
                assert a["workers"] == 1 and a["model"]["prefix_caching"] is False and "env" not in a["model"]
    order = mk.LAUNCH_ORDER
    i = order.index("p0-embedcheck")
    assert order[i + 1:i + 5] == ["p0-detprobe-bi-r1", "p0-detprobe-bi-r2", "p0-detprobe-w1-r1", "p0-detprobe-w1-r2"]
    assert order[i + 5] == "p0-micropilot-g1"


def test_reservation_from_now():
    rows = mk.reservation_check(None, mk.COMMITTED_NOW, mk.FROM_NOW)
    nxt = [r for r in rows if r["name"] in ("p0-detprobe-bi-r1", "p0-detprobe-bi-r2", "p0-detprobe-w1-r1", "p0-detprobe-w1-r2",
                                            "p0-micropilot-g1", "p0-micropilot-g2", "p0-micropilot-g3", "p0-g0-s18")]
    assert len(nxt) == 8 and all(r["need"] <= 9.00 + 1e-9 for r in nxt)   # the next tranche fits; re-running the 3 S1 reps does not
