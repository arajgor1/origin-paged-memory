"""opa.analysis.trackb_test on synthetic runs (MemStore): pooling, pairing, rows, guards."""
import json
import random

import numpy as np
import pytest
import yaml

from opa import stats
from opa.analysis import pilot as pl
from opa.analysis import tables
from opa.analysis import trackb_test as TB
from opa.analysis.p2_results import RunStatusError

BUCKET = "b"
SPLIT = "test"
V3_SHA = TB.PINNED_SHA["V3"]
B1_SHA = "bb" * 32


def _put(files, rid, rel, obj):
    files[f"gs://{BUCKET}/opa/runs/{rid}/{rel}"] = json.dumps(obj).encode()


def _get(files, rid, rel):
    return json.loads(files[f"gs://{BUCKET}/opa/runs/{rid}/{rel}"])


def _mkrun(runs_dir, rid, split=SPLIT, status="ok"):
    (runs_dir / rid).mkdir(parents=True, exist_ok=True)
    (runs_dir / rid / "job.yaml").write_text(yaml.safe_dump({"split": split}))
    (runs_dir / rid / "result.json").write_text(json.dumps({"run_id": rid, "status": status}))


def _meta(s):
    return {"seed": s, "window": [s % 7, 1], "target_origin": s % 5, "options_origin_ids": [1, 2, 3], "gold_option": s % 3, "T": 900}


def _h4_run(files, runs_dir, rid, arm, seeds, em, rank, probe=None, sha=None, steps=600, split=SPLIT):
    _mkrun(runs_dir, rid)
    fin = {"tag": "final", "opt_step": steps}
    if probe:
        fin["probe"] = probe
    _put(files, rid, "final_eval.json", {"status": "complete", "final": fin, "per_item": {"final": {
        "arm": arm, "per_item": {"attrib_em": [em[s] for s in seeds], "attrib_rank": [rank[s] for s in seeds]},
        "meta": {"attrib": [_meta(s) for s in seeds]}}}})
    sha = sha or (V3_SHA if arm == "V3" else B1_SHA)
    _put(files, rid, "config.json", {"status": "complete", "arm": arm, "opt_steps": steps, "train_loss_mean": None, "resumed_from_sha256": sha,
                                     "config": {"arm": arm, "lr": 1e-4, "max_opt_steps": steps, "eval_split": split, "expected_resume_sha256": sha,
                                                "monitor": {"attrib": len(seeds), "passkey": 0, "probe": 16 if probe else 0}}})


PROBE = {"own_origin_mass_fraction": 0.314, "own_origin_mass_fraction_excl_prefix": 0.3, "n_windows": 16, "C": 8}
PLAN = range(100000, 100040)


def _h4_setup(tmp_path, n=40, seed=1):
    rng = random.Random(seed)
    files, runs = {}, tmp_path / "runs"
    seeds = list(range(100000, 100000 + n))
    v3 = {s: int(rng.random() < 0.8) for s in seeds}
    b1 = {s: int(rng.random() < 0.5) for s in seeds}
    half = n // 2
    ids = {"h4-v3": ["v3a", "v3b"], "h4-b1": ["b1a", "b1b"]}
    for arm, em, arm_ids in (("V3", v3, ids["h4-v3"]), ("B1", b1, ids["h4-b1"])):
        for k, rid in enumerate(arm_ids):
            _h4_run(files, runs, rid, arm, seeds[:half] if k == 0 else seeds[half:], em, em, probe=PROBE if (arm == "V3" and k == 0) else None)
    (tmp_path / "tag.json").write_text("{}")
    return files, runs, ids, seeds, v3, b1


def _go(tmp_path, files, runs, ids, **kw):
    kw.setdefault("expect_sha", {"B1": B1_SHA})
    kw.setdefault("planned_seeds", PLAN)
    return TB.run_trackb(pl.MemStore(files), ids, SPLIT, tmp_path / "out", runs_dir=runs, tag_file=tmp_path / "tag.json", **kw)


def _edit(files, rid, rel, fn):
    d = _get(files, rid, rel)
    fn(d)
    _put(files, rid, rel, d)


def test_h4_pools_pairs_and_matches_stats_contrast(tmp_path):
    files, runs, ids, seeds, v3, b1 = _h4_setup(tmp_path)
    res = _go(tmp_path, files, runs, ids)
    row = {r["key"]: r for r in res["rows"]}
    ref = stats.contrast([v3[s] for s in seeds], [b1[s] for s in seeds], "x", test="mcnemar", n_boot=10_000, seed=0)
    r = row["V3_vs_B1_source_attribution"]
    assert r["n"] == 40 and r["value"] == pytest.approx(ref["value"]) and r["ci_lo"] == pytest.approx(ref["ci_lo"]) and r["p_raw"] == pytest.approx(ref["p_raw"])
    assert r["hypothesis"] == "H4" and r["boot_seed"] == 0 and r["n_boot"] == 10_000 and r["n_excluded_missing"] == 0
    assert row["V3_own_origin_mass_fraction"]["value"] == pytest.approx(0.314) and row["V3_own_origin_mass_fraction"]["n"] == 16
    assert "V3_vs_B1_source_attribution_rank" in row
    t = tables.read_table(tmp_path / "out" / "trackb_contrasts_test.csv")
    assert {x["key"] for x in t} >= {"V3_vs_B1_source_attribution", "V3_own_origin_mass_fraction"}
    ser = {x["key"]: x for x in tables.read_table(tmp_path / "out" / "trackb_series_test.csv")}
    assert ser["h4-v3"]["n_requested"] == "40" and ser["h4-v3"]["resume_sha256"] == V3_SHA and ser["h4-b1"]["resume_sha256"] == B1_SHA


def test_h4_unpaired_and_never_evaluated_seeds_are_exclusions(tmp_path):
    files, runs, ids, seeds, v3, b1 = _h4_setup(tmp_path)
    ids["h4-b1"] = ["b1a"]                                        # B1 missing its second container: 20 seeds are V3-only
    res = _go(tmp_path, files, runs, ids, planned_seeds=range(100000, 100045))      # and 5 planned seeds neither arm evaluated
    r = {x["key"]: x for x in res["rows"]}["V3_vs_B1_source_attribution"]
    assert r["n"] == 20 and r["n_excluded_missing"] == 25 and len(res["exclusions"]) == 25
    assert sum("neither arm" in e["reason"] for e in res["exclusions"]) == 5


def test_h4_refuses_duplicates_mixed_config_sha_and_different_items(tmp_path):
    files, runs, ids, *_ = _h4_setup(tmp_path / "a")
    ids["h4-v3"] = ["v3a", "v3a"]
    with pytest.raises(ValueError, match="duplicate attribution"):
        _go(tmp_path / "a", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "b")
    _edit(files, "v3b", "config.json", lambda d: d["config"].update(lr=2e-4))
    with pytest.raises(ValueError, match="mixed training configurations"):
        _go(tmp_path / "b", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "c")
    _edit(files, "b1b", "config.json", lambda d: d.update(resumed_from_sha256="zz"))
    with pytest.raises(ValueError, match="resumed sha256"):
        _go(tmp_path / "c", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "d")
    _edit(files, "b1a", "final_eval.json", lambda d: d["per_item"]["final"]["meta"]["attrib"][0].update(gold_option=99))
    with pytest.raises(ValueError, match="not the same item"):
        _go(tmp_path / "d", files, runs, ids)


def test_h4_arm_split_steps_and_sha_guards(tmp_path):
    files, runs, ids, *_ = _h4_setup(tmp_path / "a")
    ids["h4-v3"], ids["h4-b1"] = ids["h4-b1"], ids["h4-v3"]                                   # swapped arms
    with pytest.raises(ValueError, match="arms swapped"):
        _go(tmp_path / "a", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "b")
    _edit(files, "b1a", "config.json", lambda d: d["config"].update(eval_split="dev"))
    with pytest.raises(ValueError, match="eval_split"):
        _go(tmp_path / "b", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "c")
    for rid in ids["h4-b1"]:
        _edit(files, rid, "config.json", lambda d: (d.update(opt_steps=400), d["config"].update(max_opt_steps=400)))
    with pytest.raises(ValueError, match="different step counts"):
        _go(tmp_path / "c", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "d")
    _edit(files, "b1a", "config.json", lambda d: d.update(opt_steps=3))                      # trained: opt_steps != max_opt_steps
    with pytest.raises(ValueError, match="evaluation-only"):
        _go(tmp_path / "d", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "e")
    _edit(files, "b1a", "config.json", lambda d: d.update(resumed_from_sha256=None))
    with pytest.raises(ValueError, match="missing"):
        _go(tmp_path / "e", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "f")
    with pytest.raises(ValueError, match="expected B1 checkpoint"):
        _go(tmp_path / "f", files, runs, ids, expect_sha={"B1": "cc" * 32})
    for rid in ids["h4-b1"]:
        _edit(files, rid, "config.json", lambda d: (d.update(resumed_from_sha256=V3_SHA), d["config"].update(expected_resume_sha256=V3_SHA)))
    with pytest.raises(ValueError, match="same checkpoint"):
        _go(tmp_path / "f", files, runs, ids, expect_sha={"B1": V3_SHA})
    files, runs, ids, *_ = _h4_setup(tmp_path / "g")
    _edit(files, "v3a", "config.json", lambda d: d["config"].update(expected_resume_sha256="dd" * 32))
    with pytest.raises(ValueError, match="expected_resume_sha256"):
        _go(tmp_path / "g", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "h")
    _edit(files, "b1a", "final_eval.json", lambda d: d["per_item"]["final"]["meta"]["attrib"][0].update(seed=5))
    with pytest.raises(ValueError, match="not a test seed"):
        _go(tmp_path / "h", files, runs, ids)


def test_non_ok_run_split_mismatch_and_missing_probe(tmp_path):
    files, runs, ids, *_ = _h4_setup(tmp_path / "a")
    (runs / "b1a" / "result.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RunStatusError):
        _go(tmp_path / "a", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "b")
    (runs / "v3a" / "job.yaml").write_text(yaml.safe_dump({"split": "dev"}))
    with pytest.raises(ValueError, match="job split"):
        _go(tmp_path / "b", files, runs, ids)
    files, runs, ids, *_ = _h4_setup(tmp_path / "c")
    _edit(files, "v3a", "final_eval.json", lambda d: d["final"].pop("probe"))
    with pytest.raises(pl.MissingInput, match="probe"):
        _go(tmp_path / "c", files, runs, ids)


def test_refuses_test_without_the_prereg_tag_and_dev_names(tmp_path):
    files, runs, ids, *_ = _h4_setup(tmp_path)
    with pytest.raises(PermissionError):
        TB.run_trackb(pl.MemStore(files), ids, "test", tmp_path / "o", runs_dir=runs, tag_file=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="dev"):
        TB.run_trackb(pl.MemStore(files), ids, "test", tmp_path / "dev_out", runs_dir=runs, tag_file=tmp_path / "tag.json")


# ---------------------------------------------------------------- trainfree + H6
def _trainfree(files, runs, rid="tf", split=SPLIT, val=1.2, inconsistent=False, n_ok=16):
    _mkrun(runs, rid)
    _put(files, rid, "metrics.json", {"split": split, "predictions_to_read": {"V0_own_origin_mass_ratio_to_uniform": val},
                                      "variants": {"V0": {"probe": {"own_origin_mass_ratio_to_uniform": val + (0.5 if inconsistent else 0), "n_windows": 16, "C": 8}}}})
    _put(files, rid, "probe_V0.json", {"n_ok": n_ok, "summary": {"n_windows": n_ok}})


CONDS = ["R0", "R1", "R2oracle", "R3_1", "R4", "R5_8", "R6_2", "R7oracle"]
PROBS = {"R0": 0.8, "R1": 0.3, "R2oracle": 0.85, "R3_1": 0.5, "R4": 0.7, "R5_8": 0.78, "R6_2": 0.6, "R7oracle": 0.9}


def _h6_run(files, runs, rid, ids, rng, n_requested=None, probs=None, offset=0, split=SPLIT, seed0=100000):
    _mkrun(runs, rid)
    probs = probs or PROBS
    per_item = [{c: {"em": int(rng.random() < probs[c])} for c in CONDS} for _ in ids]
    _put(files, rid, "remedies.json", {"summary": {"n_requested": n_requested or len(ids), "stop_reason": "all_items"}, "per_item": per_item,
                                       "item_ids": ids, "args": {"split": split, "seed0": seed0, "offset": offset}})
    return per_item


def _h6(tmp_path, files, runs, ids):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tag.json").write_text("{}")
    return TB.run_trackb(pl.MemStore(files), ids, SPLIT, tmp_path / "out", runs_dir=runs, tag_file=tmp_path / "tag.json")


def test_h6_pools_both_cells_and_reports_ratio_and_oracle_separately(tmp_path):
    rng = random.Random(3)
    files, runs = {}, tmp_path / "runs"
    allp, ids = [], {"h6-lb": ["l1", "l2"], "h6-mt": ["m1", "m2"], "trainfree": ["tf"]}
    for rid in ids["h6-lb"] + ids["h6-mt"]:
        allp += _h6_run(files, runs, rid, [f"{rid}-{k}" for k in range(50)], rng)
    _trainfree(files, runs)
    res = _h6(tmp_path, files, runs, ids)
    row = {r["key"]: r for r in res["rows"]}
    r5, r1 = (np.array([p[c]["em"] for p in allp], float) for c in ("R5_8", "R1"))
    ref = stats.contrast(r5, r1, "x", test="mcnemar", n_boot=10_000, seed=0)
    r = row["R5_vs_R1_accuracy"]
    assert r["n"] == 200 and r["value"] == pytest.approx(ref["value"]) and r["p_raw"] == pytest.approx(ref["p_raw"]) and r["hypothesis"] == "H6"
    accs = {c: np.mean([p[c]["em"] for p in allp]) for c in CONDS}
    dep = max(accs[c] for c in ("R3_1", "R4", "R5_8", "R6_2")) / accs["R0"]
    assert row["best_remedy_recovery_fraction"]["value"] == pytest.approx(dep)
    inc = row["best_remedy_recovery_fraction_incl_oracle"]
    assert inc["value"] == pytest.approx(max(accs[c] for c in CONDS if c not in ("R0", "R1")) / accs["R0"]) and "ORACLE" in inc["note"]
    assert row["best_remedy_recovery_fraction"]["ci_lo"] <= dep <= row["best_remedy_recovery_fraction"]["ci_hi"]
    assert row["V0_own_origin_mass_ratio_to_uniform"]["value"] == pytest.approx(1.2) and row["V0_own_origin_mass_ratio_to_uniform"]["n_excluded_missing"] == 0
    assert "oracle" not in row["best_remedy_recovery_fraction"]["note"].lower() and "r0_too_low" not in row["best_remedy_recovery_fraction"]["note"]


def test_h6_one_exclusion_row_per_missing_item_and_duplicates(tmp_path):
    rng = random.Random(4)
    files, runs = {}, tmp_path / "runs"
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(10)], rng)
    _h6_run(files, runs, "l2", [f"a{k}" for k in range(5, 12)], rng)                     # a5..a9 repeat
    with pytest.raises(ValueError, match="duplicate remedy item"):
        _h6(tmp_path / "a", files, runs, {"h6-lb": ["l1", "l2"]})
    files, runs = {}, tmp_path / "runs2"
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(10)], rng, n_requested=13, offset=100)     # three items not finished
    res = _h6(tmp_path / "b", files, runs, {"h6-lb": ["l1"]})
    r = {x["key"]: x for x in res["rows"]}["R5_vs_R1_accuracy"]
    assert r["n"] == 10 and r["n_excluded_missing"] == 3 and len(res["exclusions"]) == 3
    assert [e["item_id"] for e in res["exclusions"]] == ["h6-lb#110", "h6-lb#111", "h6-lb#112"]


def test_h6_split_comes_from_remedies_args_and_test_needs_test_seeds(tmp_path):
    rng = random.Random(5)
    files, runs = {}, tmp_path / "runs"
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(5)], rng, split="dev")
    with pytest.raises(ValueError, match="args.split"):
        _h6(tmp_path / "a", files, runs, {"h6-lb": ["l1"]})
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(5)], rng, seed0=900)
    with pytest.raises(ValueError, match="seed0"):
        _h6(tmp_path / "b", files, runs, {"h6-lb": ["l1"]})
    _edit(files, "l1", "remedies.json", lambda d: d.pop("args"))
    with pytest.raises(ValueError, match="args.split"):
        _h6(tmp_path / "c", files, runs, {"h6-lb": ["l1"]})


def test_h6_low_r0_flag_and_skipped_messages(tmp_path):
    rng = random.Random(6)
    files, runs = {}, tmp_path / "runs"
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(40)], rng, probs={**PROBS, "R0": 0.05})
    res = _h6(tmp_path / "a", files, runs, {"h6-lb": ["l1"]})
    best = [r for r in res["rows"] if r["key"].startswith("best_remedy")]
    assert best and all("r0_too_low" in r["note"] for r in best)
    files, runs = {}, tmp_path / "runs2"
    _h6_run(files, runs, "l1", [f"a{k}" for k in range(20)], rng, probs={**PROBS, "R0": 0.0})
    res = _h6(tmp_path / "b", files, runs, {"h6-lb": ["l1"]})
    assert any("undefined" in s for s in res["skipped"]) and not any(r["key"].startswith("best_remedy") for r in res["rows"])


def test_trainfree_guards_and_probe_shortfall(tmp_path):
    files, runs = {}, tmp_path / "runs"
    _trainfree(files, runs, n_ok=14)
    res = _h6(tmp_path / "a", files, runs, {"trainfree": ["tf"]})
    r = res["rows"][0]
    assert r["n"] == 14 and r["n_excluded_missing"] == 2 and "14 of 16" in r["note"]
    _trainfree(files, runs, "tf", inconsistent=True)
    with pytest.raises(ValueError, match="inconsistent"):
        _h6(tmp_path / "b", files, runs, {"trainfree": ["tf"]})
    _trainfree(files, runs, "tf", split="dev")
    with pytest.raises(ValueError, match="split"):
        _h6(tmp_path / "c", files, runs, {"trainfree": ["tf"]})


def test_unknown_series_refused(tmp_path):
    (tmp_path / "tag.json").write_text("{}")
    with pytest.raises(ValueError, match="unknown series"):
        TB.run_trackb(pl.MemStore({}), {"h9": ["x"]}, SPLIT, tmp_path / "o", runs_dir=tmp_path, tag_file=tmp_path / "tag.json")


def test_trainfree_series_row_is_written(tmp_path):
    files, runs = {}, tmp_path / "runs"
    _trainfree(files, runs, n_ok=14)
    (runs / "tf" / "manifest.json").write_text(json.dumps({"git_sha": "abc1234"}))
    _h6(tmp_path / "a", files, runs, {"trainfree": ["tf"]})
    ser = {x["key"]: x for x in tables.read_table(tmp_path / "a" / "out" / "trackb_series_test.csv")}
    t = ser["trainfree"]
    assert (t["split"], t["status"], t["git_sha"], t["n_items"], t["n_requested"], t["resume_sha256"], t["run_ids"]) == ("test", "ok", "abc1234", "14", "16", "n/a", "tf")
