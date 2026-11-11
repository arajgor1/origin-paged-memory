"""P2 results pipeline (opa.analysis.p2_results) on synthetic multi-container runs held in an in-memory store.

The opt-in test at the end (OPA_GCS_TESTS=1) runs the pipeline on the dev pilot runs in GCS (read-only).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from opa import stats
from opa.analysis import p2_results as p2
from opa.analysis import pilot as pl
from opa.analysis import tables as tb
from opa.tracks.track_a.embed import compute_item_embeddings, embeddings_to_bytes
from opa.tracks.track_a.index import HashEmbedder

sys.path.insert(0, str(Path(__file__).parent))
import test_pilot as tp  # noqa: E402  (lb_items, lb_records, put_job, uri)
import test_routing_recall as trr  # noqa: E402

B = "b"
SPLIT = "dev"
K_H1, K_S4, K_H3, K_H5, K_TOST = (p2.CONTRASTS[i][0] for i in range(5))


def put_run(runs_dir, files, rid, args, status="ok", split=SPLIT, manifest_split=None):
    tp.put_job(runs_dir, rid, args, split=split)
    (runs_dir / rid / "result.json").write_text(json.dumps({"run_id": rid, "status": status}))
    if manifest_split:
        (runs_dir / rid / "manifest.json").write_text(json.dumps({"split": manifest_split}))


def put_multi(runs_dir, files, rid, items_file, offset, n, systems, recs_by_sys, extra_args=None, **kw):
    args = {"items_uri": [{"uri": f"gs://{{bucket}}/{items_file}", "n_items": n, "offset": offset}], "systems": systems,
            **(extra_args or {})}
    put_run(runs_dir, files, rid, args, **kw)
    return args


def put_records(files, rid, system, recs, items, metrics_extra=None):
    files[tp.uri(rid, f"{system}/r0/records.jsonl.gz")] = pl.gz_jsonl(recs)
    by = {i["item_id"]: i for i in items}
    em = float(np.mean([pl.harness_em(by[r["item_id"]], r) for r in recs])) if recs else 0.0
    files[tp.uri(rid, f"{system}/r0/metrics.json")] = json.dumps({"em": em, "n": len(recs), **(metrics_extra or {})}).encode()


def put_thrash(files, rid, recs, prompt_version="thrash-v2"):
    files[tp.uri(rid, "records.jsonl.gz")] = pl.gz_jsonl(recs)
    sc = [r["repeated_page_ins"] for r in recs if r.get("status") in pl.THRASH_SCORED]
    files[tp.uri(rid, "metrics.json")] = json.dumps({"prompt_version": prompt_version,
                                                     "repeated_page_ins_mean": float(np.mean(sc))}).encode()


def build(tmp_path, *, p=None, seed=1):
    """h1-lb (2 containers x 20 items; S11, S2, S8), h1-s4 (S4 on the first 20 items), h5-mt (24 routing items; S12,
    S14, S15 on the first 12), h10 pin / nopin (40 thrash items over r4)."""
    runs_dir = tmp_path / "runs"
    files: dict = {}
    rng = np.random.default_rng(seed)
    p = p or {"S11": 0.7, "S2": 0.5, "S8": 0.65, "S4": 0.4, "S12": 0.7, "S14": 0.5, "S15": 0.68}
    lb = tp.lb_items("lb", 40)
    files[f"gs://{B}/d/lb.jsonl.gz"] = pl.gz_jsonl(lb)
    ids = {}
    for c in (1, 2):
        rid = f"R-h1-lb-c0{c}"
        part = lb[(c - 1) * 20:c * 20]
        put_multi(runs_dir, files, rid, "d/lb.jsonl.gz", (c - 1) * 20, 20, ["S11", "S2", "S8"], None)
        for s in ("S11", "S2", "S8"):
            put_records(files, rid, s, tp.lb_records(part, s, p[s], rng), part)
        ids.setdefault("h1-lb", []).append(rid)
    rid = "R-h1-s4-c01"
    put_multi(runs_dir, files, rid, "d/lb.jsonl.gz", 0, 20, ["S4"], None)
    put_records(files, rid, "S4", tp.lb_records(lb[:20], "S4", p["S4"], rng), lb[:20])
    ids["h1-s4"] = [rid]
    # ---- MT: routing items double as the agent items
    mt = trr.make_items(24)
    for it in mt:
        it["answer"] = f"mt-{it['item_id']}"
    files[f"gs://{B}/d/mt.jsonl.gz"] = pl.gz_jsonl(mt)
    files[tp.uri("R-embed-mt", "embeddings.npz")] = embeddings_to_bytes(compute_item_embeddings(mt, HashEmbedder()))
    for iid, payload in trr.make_skeletons(mt).items():
        files[tp.uri("R-sk-mt", f"skeleton_llm/{iid}.json")] = json.dumps(payload).encode()
    ids["embed-mt"] = ["R-embed-mt"]
    put_run(runs_dir, files, "R-embed-mt", {"items_uri": [{"uri": f"gs://{{bucket}}/d/mt.jsonl.gz", "n_items": 24}]})
    ids["sk-mt"] = ["R-sk-mt"]
    put_run(runs_dir, files, "R-sk-mt", {"items_uri": [{"uri": f"gs://{{bucket}}/d/mt.jsonl.gz", "n_items": 24}]})
    emb = {"embedder": "precomputed:gs://{bucket}/opa/runs/R-embed-mt/embeddings.npz",
           "skeleton_llm_uri": "gs://{bucket}/opa/runs/R-sk-mt/skeleton_llm"}
    ids["h5-mt"] = []
    for c, (off, n, systems) in enumerate([(0, 12, ["S12", "S14", "S15"]), (12, 12, ["S12", "S14"])], 1):
        rid = f"R-h5-mt-c0{c}"
        part = mt[off:off + n]
        put_multi(runs_dir, files, rid, "d/mt.jsonl.gz", off, n, systems, None, extra_args=emb)
        for s in systems:
            put_records(files, rid, s, tp.lb_records(part, s, p[s], rng), part)
        ids["h5-mt"].append(rid)
    # ---- thrash
    th = [{"item_id": f"th{i:03d}", "answer": "x", "params": {"r": 4}} for i in range(40)]
    files[f"gs://{B}/d/th.jsonl.gz"] = pl.gz_jsonl(th)
    for name, hi in (("h10-nopin", 8), ("h10-pin", 2)):
        rid = f"R-{name}-c01"
        put_run(runs_dir, files, rid, {"items_uri": [{"uri": f"gs://{{bucket}}/d/th.jsonl.gz", "n_items": 40}], "M": 2})
        recs = [{"item_id": t["item_id"], "M": 2, "r": 4, "repeated_page_ins": int(rng.integers(0, hi)), "status": "ok"} for t in th]
        put_thrash(files, rid, recs)
        ids[name] = [rid]
    return runs_dir, files, ids


def run(files, ids, runs_dir, out, **kw):
    return p2.run_p2(pl.MemStore(files, B), ids, kw.pop("split", SPLIT), out, runs_dir=runs_dir, rules=kw.pop("rules", {}), **kw)


def rows_by_key(res):
    return {r["key"]: r for r in res["rows"]}


def get_records(files, rid, system):
    return pl.read_jsonl(files[tp.uri(rid, f"{system}/r0/records.jsonl.gz")])


def set_records(files, rid, system, recs):
    files[tp.uri(rid, f"{system}/r0/records.jsonl.gz")] = pl.gz_jsonl(recs)
    by = {i["item_id"]: i for i in tp.lb_items("lb", 40)}
    files[tp.uri(rid, f"{system}/r0/metrics.json")] = json.dumps(            # keep metrics.json consistent with the records
        {"em": float(np.mean([pl.harness_em(by[r["item_id"]], r) for r in recs])), "n": len(recs)}).encode()


@pytest.fixture()
def world(tmp_path):
    return build(tmp_path)


# ---------------------------------------------------------------------------------------------- merging and contrasts
def test_container_merge_matches_direct_computation(world, tmp_path):
    runs_dir, files, ids = world
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    rows = rows_by_key(res)
    # direct: concatenate both containers by hand
    recs = {s: {} for s in ("S11", "S2")}
    for rid in ids["h1-lb"]:
        for s in recs:
            recs[s].update({r["item_id"]: r for r in get_records(files, rid, s)})
    items = {i["item_id"]: i for i in tp.lb_items("lb", 40)}
    order = sorted(items)
    a = np.array([pl.harness_em(items[i], recs["S11"][i]) for i in order])
    b = np.array([pl.harness_em(items[i], recs["S2"][i]) for i in order])
    want = stats.contrast(a, b, K_H1, seed=p2.BOOT_SEED)
    r = rows[K_H1]
    assert r["n"] == 40 and r["n_pairs"] == 40 and r["n_excluded_missing"] == 0 and r["n_itt_imputed"] == 0
    assert r["value"] == want["value"] and (r["ci_lo"], r["ci_hi"]) == (want["ci_lo"], want["ci_hi"]) and r["p_raw"] == want["p_raw"]
    assert r["boot_seed"] == 0 and r["n_boot"] == 10_000 and r["split"] == SPLIT and r["p_holm"] is None
    assert set(r["run_ids"]) == set(ids["h1-lb"])
    # S4 exists on the first 20 items only: 20 pairs, 20 unpaired by design, no exclusion
    s4 = rows[K_S4]
    assert s4["n"] == 20 and s4["n_unpaired_by_design"] == 20 and s4["n_excluded_missing"] == 0
    assert rows[K_H3]["n"] == 40
    # H5: S15 on the first 12 only
    assert rows[K_H5]["n"] == 24 and rows[K_TOST]["n"] == 12 and rows[K_TOST]["n_unpaired_by_design"] == 12
    assert rows[p2.K_H8]["n"] == 24 and rows[p2.K_H10]["n"] == 40
    assert "R2_predicted_vs_observed_success" not in rows and not res["skipped"]
    assert "R2_predicted_vs_observed_success" not in (tmp_path / "p2_dev" / "p2_contrasts_dev.csv").read_text()
    assert not hasattr(p2, "r2_row") and not hasattr(p2, "K_R2")


def test_duplicate_item_across_containers(world, tmp_path):
    runs_dir, files, ids = world
    job = yaml.safe_load((runs_dir / ids["h1-lb"][1] / "job.yaml").read_text())
    job["args"]["items_uri"][0]["offset"] = 10                      # overlaps container 1 (items 0-19)
    (runs_dir / ids["h1-lb"][1] / "job.yaml").write_text(yaml.safe_dump(job))
    with pytest.raises(ValueError, match="more than one container"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_duplicate_record_within_system(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][0]
    recs = get_records(files, rid, "S2")
    set_records(files, rid, "S2", recs + [recs[0]])
    with pytest.raises(ValueError, match="duplicate record"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


@pytest.mark.parametrize("field_name", ["config_version", "prompt_version"])
def test_mixed_versions_error(world, tmp_path, field_name):
    runs_dir, files, ids = world
    for k, rid in enumerate(ids["h1-lb"]):
        recs = get_records(files, rid, "S11")
        for r in recs:
            r[field_name] = f"v{k}"
        set_records(files, rid, "S11", recs)
    with pytest.raises(ValueError, match=f"mixed {field_name}"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_mixed_versions_ok_when_equal(world, tmp_path):
    runs_dir, files, ids = world
    for rid in ids["h1-lb"]:
        recs = get_records(files, rid, "S11")
        for r in recs:
            r["config_version"] = "cfg1"
        set_records(files, rid, "S11", recs)
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    ser = tb.read_table(tmp_path / "p2_dev" / "p2_series_dev.csv")
    assert "S11=cfg1" in {r["key"]: r for r in ser}["h1-lb"]["config_version"]


def test_run_status_not_ok_is_reported(world, tmp_path):
    runs_dir, files, ids = world
    (runs_dir / ids["h1-lb"][1] / "result.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(p2.RunStatusError) as e:
        run(files, ids, runs_dir, tmp_path / "p2_dev")
    assert ids["h1-lb"][1] in str(e.value) and "failed" in str(e.value)
    assert not (tmp_path / "p2_dev").exists()


def test_split_mismatch(world, tmp_path):
    runs_dir, files, ids = world
    with pytest.raises(ValueError, match="manifest split"):
        (runs_dir / ids["h1-lb"][0] / "manifest.json").write_text(json.dumps({"split": "test"}))
        run(files, ids, runs_dir, tmp_path / "p2_dev")
    (runs_dir / ids["h1-lb"][0] / "manifest.json").unlink()
    job = yaml.safe_load((runs_dir / ids["h1-s4"][0] / "job.yaml").read_text())
    job["split"] = "test"
    (runs_dir / ids["h1-s4"][0] / "job.yaml").write_text(yaml.safe_dump(job))
    with pytest.raises(ValueError, match="job split"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_missing_system_records(world, tmp_path):
    runs_dir, files, ids = world
    del files[tp.uri(ids["h1-lb"][0], "S8/r0/records.jsonl.gz")]
    with pytest.raises(ValueError, match="no records for S8"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_em_metrics_mismatch(world, tmp_path):
    runs_dir, files, ids = world
    u = tp.uri(ids["h1-lb"][0], "S2/r0/metrics.json")
    m = json.loads(files[u])
    m["em"] += 0.05
    files[u] = json.dumps(m).encode()
    with pytest.raises(ValueError, match="metrics.json em"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


# ---------------------------------------------------------------------------------------------- ITT and exclusions
def test_itt_failure_scores_em_zero(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][0]
    recs = get_records(files, rid, "S2")
    recs[0]["status"] = "error"                                       # a correct-looking answer must still score 0
    recs[0]["prediction"] = next(i for i in tp.lb_items("lb", 40) if i["item_id"] == recs[0]["item_id"])["answer"]
    set_records(files, rid, "S2", recs)
    r = rows_by_key(run(files, ids, runs_dir, tmp_path / "p2_dev"))[K_H1]
    assert r["n"] == 40 and r["n_itt_imputed"] == 1 and r["n_error_b"] == 1 and r["n_error_a"] == 0
    # same failure on the S11 side: counted once per pair even if both arms failed
    recs11 = get_records(files, rid, "S11")
    recs11[0]["status"] = "error"
    set_records(files, rid, "S11", recs11)
    r = rows_by_key(run(files, ids, runs_dir, tmp_path / "p2_dev2"))[K_H1]
    assert r["n_itt_imputed"] == 1 and r["n_error_a"] == 1 and r["n_error_b"] == 1


def test_missing_pair_is_excluded_and_listed(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][1]
    recs = get_records(files, rid, "S2")
    gone = recs.pop(3)["item_id"]
    set_records(files, rid, "S2", recs)
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    r = rows_by_key(res)[K_H1]
    assert r["n"] == 39 and r["n_excluded_missing"] == 1
    assert [e for e in res["exclusions"] if e["contrast"] == K_H1] == [
        {"key": f"{K_H1}|{gone}", "contrast": K_H1, "item_id": gone, "reason": "missing S2", "split": SPLIT}]
    ex = tb.read_table(tmp_path / "p2_dev" / "p2_exclusions_dev.csv")
    assert any(e["item_id"] == gone and e["reason"] == "missing S2" for e in ex)
    assert rows_by_key(res)[K_H3]["n_excluded_missing"] == 0        # S8 pair unaffected


def test_h10_itt_uses_worst_observed(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h10-pin"][0]
    recs = pl.read_jsonl(files[tp.uri(rid, "records.jsonl.gz")])
    worst = max(r["repeated_page_ins"] for r in recs + pl.read_jsonl(files[tp.uri(ids["h10-nopin"][0], "records.jsonl.gz")]))
    recs[0] = {"item_id": recs[0]["item_id"], "M": 2, "r": 4, "status": "error"}      # failed pinned record, no count
    put_thrash(files, rid, recs)
    r = rows_by_key(run(files, ids, runs_dir, tmp_path / "p2_dev"))[p2.K_H10]
    assert r["n"] == 40 and r["n_itt_imputed"] == 1 and r["n_error_b"] == 1 and r["imputed_worst"] == f"r4-M2={worst}"
    nop = {x["item_id"]: x["repeated_page_ins"] for x in pl.read_jsonl(files[tp.uri(ids["h10-nopin"][0], "records.jsonl.gz")])}
    pin = {x["item_id"]: (worst if x.get("status") == "error" else x["repeated_page_ins"]) for x in recs}
    assert r["value"] == pytest.approx(np.mean([nop[i] - pin[i] for i in nop]))     # unpinned minus pinned
    assert r["pinned_mean"] == pytest.approx(np.mean(list(pin.values())))


def test_h10_prompt_version_enforced(world, tmp_path):
    runs_dir, files, ids = world
    put_thrash(files, ids["h10-pin"][0], pl.read_jsonl(files[tp.uri(ids["h10-pin"][0], "records.jsonl.gz")]), "thrash-v1")
    with pytest.raises(ValueError, match="thrash-v2"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


# ---------------------------------------------------------------------------------------------- TOST / non-inferiority
def mk_series(name, system, a_correct, n=None):
    """Series with one system whose records are correct (1) or wrong (0) as given; items are lb_items."""
    n = len(a_correct)
    items = {i["item_id"]: i for i in tp.lb_items(name, n)}
    recs = {iid: {"item_id": iid, "system": system, "status": "ok", "prediction": it["answer"] if c else "wrong"}
            for (iid, it), c in zip(items.items(), a_correct)}
    return p2.Series(name, [f"R-{name}"], items, {system: set(items)}, {system: recs})


def paired_series(ca, cb):
    sa, sb = mk_series("x", "S12", ca), mk_series("x", "S15", cb)
    sa.records["S15"], sa.scheduled["S15"], sa.items = sb.records["S15"], sb.scheduled["S15"], {**sa.items}
    return sa


def test_tost_identical_arms():
    n = 200
    c = [i % 2 for i in range(n)]
    s = paired_series(c, c)
    r = p2.pair_contrast(K_TOST, "H5", s, "S12", s, "S15", SPLIT, {})
    assert r["value"] == 0.0 and r["n10"] == r["n01"] == 0
    want = stats.tost_paired_p(np.array(c), np.array(c), 3.0)
    assert r["tost_equivalent"] is True and r["p_raw"] == want["p"] and r["p_raw"] < 0.05
    assert r["tost_margin"] == 3.0 and "TOST" in r["test"]


def test_tost_not_equivalent_when_all_discordant():
    n = 60
    s = paired_series([1] * n, [0] * n)
    r = p2.pair_contrast(K_TOST, "H5", s, "S12", s, "S15", SPLIT, {})
    assert r["value"] == 100.0 and r["tost_equivalent"] is False and r["p_raw"] > 0.05


def test_tost_small_n_inconclusive():
    s = paired_series([1, 0, 1, 1], [1, 0, 1, 1])
    r = p2.pair_contrast(K_TOST, "H5", s, "S12", s, "S15", SPLIT, {})
    assert r["n"] == 4 and r["tost_equivalent"] is False             # zero discordance but far too few pairs


def test_ni_met_is_strict():
    assert p2.ni_met(-2.99, -3.0) is True
    assert p2.ni_met(-3.0, -3.0) is False                           # lower bound exactly on the margin: not met
    assert p2.ni_met(-3.01, -3.0) is False


def test_h8_row_is_ci_only_and_merges_containers(world, tmp_path):
    runs_dir, files, ids = world
    res = run(files, ids, runs_dir, tmp_path / "p2_dev", M=4)
    r = rows_by_key(res)[p2.K_H8]
    assert r["n"] == 24 and r["ni_margin"] == -3.0 and r["ni_met"] == (r["ci_lo"] > -3.0)
    # one-shot equivalent over all 24 items
    mt = trr.make_items(24)
    direct = rr_rows(mt)
    assert r["value"] == pytest.approx(direct["value"]) and r["n10"] == direct["n10"] and r["n01"] == direct["n01"]
    row = {x["key"]: x for x in tb.read_table(tmp_path / "p2_dev" / "p2_contrasts_dev.csv")}[p2.K_H8]
    assert row["p_raw"] == "" and row["p_holm"] == "" and "lower bound only" in row["test"]


def rr_rows(items):
    from opa.analysis import routing_recall as rr
    rows = rr.routing_recall_rows(items, embeddings_to_bytes(compute_item_embeddings(items, HashEmbedder())),
                                  trr.make_skeletons(items), M=4, split=SPLIT, n_boot=500)
    return rows[0]


def test_h8_skeleton_missing_is_excluded(world, tmp_path):
    runs_dir, files, ids = world
    del files[tp.uri("R-sk-mt", "skeleton_llm/it05.json")]
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    r = rows_by_key(res)[p2.K_H8]
    assert r["n"] == 23 and r["n_excluded_missing"] == 1
    assert any(e["contrast"] == p2.K_H8 and e["item_id"] == "it05" and e["reason"] == "no skeleton" for e in res["exclusions"])


# ---------------------------------------------------------------------------------------------- min n, guards, determinism
def test_min_n_flags_short_cells(world, tmp_path):
    runs_dir, files, ids = world
    f = tmp_path / "sample_sizes.yaml"
    f.write_text(yaml.safe_dump({"min_n": {K_H1: 100, K_S4: 20, K_H3: None}, "equivalence": {"A2_vs_A3_agent_min_n": 191}}))
    assert p2.load_min_n(f) == {K_H1: 100, K_S4: 20, K_TOST: 191}
    res = run(files, ids, runs_dir, tmp_path / "p2_dev", min_n=f)
    rows = rows_by_key(res)
    assert rows[K_H1]["n_short"] is True and rows[K_H1]["min_n"] == 100
    assert rows[K_S4]["n_short"] is False and rows[K_H3]["n_short"] is None and rows[K_TOST]["n_short"] is True
    assert sorted(s.split(":")[0] for s in res["short"]) == sorted([K_H1, K_TOST])
    csv_rows = {x["key"]: x for x in tb.read_table(tmp_path / "p2_dev" / "p2_contrasts_dev.csv")}
    assert csv_rows[K_H1]["n_short"] == "true" and csv_rows[K_H3]["n_short"] == ""
    assert p2.load_min_n({K_H1: 7}) == {K_H1: 7}


def test_table_name_guard(world, tmp_path):
    runs_dir, files, ids = world
    with pytest.raises(ValueError, match="'test'"):
        run(files, ids, runs_dir, tmp_path / "p2_test_results")      # dev split, a "test" directory
    tag = tmp_path / "PREREG_TAG.json"
    tag.write_text("{}")
    with pytest.raises(ValueError, match="'dev'"):
        run(files, ids, runs_dir, tmp_path / "p2_dev_results", split="test", tag_file=tag)
    p2.check_table_names(tmp_path / "pytest-of-someone" / "p2_dev", ["p2_contrasts_dev.csv"], "dev")   # the parents are not checked


def test_test_split_needs_the_prereg_tag(world, tmp_path):
    runs_dir, files, ids = world
    with pytest.raises(PermissionError, match="prereg-v2 tag"):
        run(files, ids, runs_dir, tmp_path / "p2_final", split="test", tag_file=tmp_path / "missing.json")


def test_unknown_series(world, tmp_path):
    runs_dir, files, ids = world
    with pytest.raises(ValueError, match="unknown series"):
        run(files, {**ids, "h1-xx": ["a"]}, runs_dir, tmp_path / "p2_dev")


def test_deterministic_bytes(world, tmp_path):
    runs_dir, files, ids = world
    run(files, ids, runs_dir, tmp_path / "a_dev")
    run(files, ids, runs_dir, tmp_path / "b_dev")
    for name in ("p2_contrasts_dev.csv", "p2_series_dev.csv", "p2_exclusions_dev.csv"):
        assert (tmp_path / "a_dev" / name).read_bytes() == (tmp_path / "b_dev" / name).read_bytes()
    # container order does not matter
    ids2 = copy.deepcopy(ids)
    ids2["h1-lb"].reverse()
    ids2["h5-mt"].reverse()
    run(files, ids2, runs_dir, tmp_path / "c_dev")
    a = tb.read_table(tmp_path / "a_dev" / "p2_contrasts_dev.csv")
    c = tb.read_table(tmp_path / "c_dev" / "p2_contrasts_dev.csv")
    drop = lambda rows: [{k: v for k, v in r.items() if k != "run_ids"} for r in rows]
    assert drop(a) == drop(c)


def test_skipped_when_series_absent(world, tmp_path):
    runs_dir, files, ids = world
    res = run(files, {"h1-lb": ids["h1-lb"]}, runs_dir, tmp_path / "p2_dev")
    assert set(rows_by_key(res)) == {K_H1, K_H3}
    assert any(K_S4 in s for s in res["skipped"]) and any(p2.K_H10 in s for s in res["skipped"])


def test_oracle_systems_never_pair():
    s = mk_series("o", "S18", [1, 0, 1])
    with pytest.raises(ValueError, match="oracle"):
        p2.paired_arrays(s, "S18", s, "S18")


# ---------------------------------------------------------------------------------------------- checker fixes
def test_stray_record_raises(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][0]
    recs = get_records(files, rid, "S2")
    recs[0] = {**recs[0], "item_id": "lb-039"}                     # belongs to container 2
    set_records(files, rid, "S2", recs)
    with pytest.raises(ValueError, match="outside the run.*lb-039"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_missing_metrics_raises(world, tmp_path):
    runs_dir, files, ids = world
    del files[tp.uri(ids["h1-lb"][0], "S2/r0/metrics.json")]
    with pytest.raises(pl.MissingInput, match="no metrics.json em"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_em_mismatch_counts_every_record(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][0]
    recs = get_records(files, rid, "S2")
    ans = next(i for i in tp.lb_items("lb", 40) if i["item_id"] == recs[0]["item_id"])["answer"]
    recs[0]["prediction"] = "wrong" if recs[0]["prediction"] == ans else ans
    files[tp.uri(rid, "S2/r0/records.jsonl.gz")] = pl.gz_jsonl(recs)     # metrics.json is now stale
    with pytest.raises(ValueError, match="metrics.json em"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_thrash_consistency(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h10-pin"][0]
    good = files[tp.uri(rid, "metrics.json")]
    m = json.loads(good)
    files[tp.uri(rid, "metrics.json")] = json.dumps({**m, "repeated_page_ins_mean": m["repeated_page_ins_mean"] + 0.5}).encode()
    with pytest.raises(ValueError, match="repeated_page_ins_mean"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")
    files[tp.uri(rid, "metrics.json")] = json.dumps({"prompt_version": "thrash-v2"}).encode()
    with pytest.raises(pl.MissingInput, match="repeated_page_ins_mean"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")
    files[tp.uri(rid, "metrics.json")] = good
    run(files, ids, runs_dir, tmp_path / "p2_dev")
    ser = {r["key"]: r for r in tb.read_table(tmp_path / "p2_dev" / "p2_series_dev.csv")}
    assert ser["h10-pin"]["em_check"] == "ok"


def test_h10_stray_and_schedule_mismatch(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h10-pin"][0]
    recs = pl.read_jsonl(files[tp.uri(rid, "records.jsonl.gz")])
    put_thrash(files, rid, recs + [{**recs[0], "item_id": "zzz"}])
    with pytest.raises(ValueError, match="outside the run"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")
    put_thrash(files, rid, recs)
    job = yaml.safe_load((runs_dir / rid / "job.yaml").read_text())
    job["args"]["items_uri"][0]["n_items"] = 39
    (runs_dir / rid / "job.yaml").write_text(yaml.safe_dump(job))
    put_thrash(files, rid, recs[:39])
    with pytest.raises(ValueError, match="different items"):
        run(files, ids, runs_dir, tmp_path / "p2_dev")


def test_h10_missing_both_and_pair_counts(world, tmp_path):
    runs_dir, files, ids = world
    pin_id, nop_id = ids["h10-pin"][0], ids["h10-nopin"][0]
    pin = pl.read_jsonl(files[tp.uri(pin_id, "records.jsonl.gz")])
    nop = pl.read_jsonl(files[tp.uri(nop_id, "records.jsonl.gz")])
    pin = [r for r in pin if r["item_id"] != "th005"]
    nop = [r for r in nop if r["item_id"] not in ("th005", "th006")]           # th005 missing in both, th006 only unpinned
    pin[0] = {**pin[0], "status": "budget"}
    nop[0] = {**nop[0], "status": "error"}                                       # same pair, both arms failed
    pin[1] = {**pin[1], "status": "error"}                                       # a second imputed pair
    put_thrash(files, pin_id, pin)
    put_thrash(files, nop_id, nop)
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    r = rows_by_key(res)[p2.K_H10]
    assert r["n"] == 38 and r["n_excluded_missing"] == 2
    assert r["n_itt_imputed"] == 2                                   # pairs, although three records failed
    assert (r["n_error_a"], r["n_error_b"], r["n_budget_a"], r["n_budget_b"], r["n_not_fit_a"]) == (1, 1, 0, 1, 0)
    reasons = {e["item_id"]: e["reason"] for e in res["exclusions"] if e["contrast"] == p2.K_H10}
    assert reasons == {"th005@M2": "missing both", "th006@M2": "missing unpinned"}


def test_budget_and_not_fit_columns(world, tmp_path):
    runs_dir, files, ids = world
    rid = ids["h1-lb"][0]
    recs = get_records(files, rid, "S2")
    recs[0]["status"], recs[1]["status"], recs[2]["status"] = "budget", "does_not_fit", "error"
    set_records(files, rid, "S2", recs)
    r = rows_by_key(run(files, ids, runs_dir, tmp_path / "p2_dev"))[K_H1]
    assert (r["n_error_b"], r["n_budget_b"], r["n_not_fit_b"], r["n_itt_imputed"]) == (1, 1, 1, 3)
    row = {x["key"]: x for x in tb.read_table(tmp_path / "p2_dev" / "p2_contrasts_dev.csv")}[K_H1]
    assert row["n_budget_b"] == "1" and row["n_not_fit_b"] == "1"
    assert "imputed_worst" in row


def test_exclusions_sorted(world, tmp_path):
    runs_dir, files, ids = world
    for rid, k in ((ids["h1-lb"][1], 7), (ids["h1-lb"][0], 9)):
        recs = get_records(files, rid, "S2")
        recs.pop(k)
        set_records(files, rid, "S2", recs)
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    mine = [e["item_id"] for e in res["exclusions"] if e["contrast"] == K_H1]
    assert mine == sorted(mine) and len(mine) == 2


def test_h8_missing_embeddings_is_missing_input(world, tmp_path):
    runs_dir, files, ids = world
    del files[tp.uri("R-embed-mt", "embeddings.npz")]
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    assert p2.K_H8 not in rows_by_key(res) and any("embeddings" in s for s in res["skipped"])


def test_h8_row_dict_is_ci_only(world, tmp_path):
    runs_dir, files, ids = world
    r = rows_by_key(run(files, ids, runs_dir, tmp_path / "p2_dev"))[p2.K_H8]
    assert r["p_raw"] is None and r["test"] == tb.NI_TEST_LABEL


def test_s4_config_version_enforced_on_test_split(tmp_path):
    from opa.tracks.track_a.systems import S4_CONFIG_VERSION
    runs_dir, files, ids = build(tmp_path)
    for rid in ids["h1-lb"] + ids["h1-s4"]:
        jp = runs_dir / rid / "job.yaml"
        job = yaml.safe_load(jp.read_text())
        job["split"] = "test"
        jp.write_text(yaml.safe_dump(job))
    tag = tmp_path / "PREREG_TAG.json"
    tag.write_text("{}")
    sub = {k: ids[k] for k in ("h1-lb", "h1-s4")}
    with pytest.raises(ValueError, match="S4: config_version"):
        run(files, sub, runs_dir, tmp_path / "p2_final", split="test", tag_file=tag)
    recs = get_records(files, ids["h1-s4"][0], "S4")
    for r in recs:
        r["config_version"] = S4_CONFIG_VERSION
    set_records(files, ids["h1-s4"][0], "S4", recs)
    assert K_S4 in rows_by_key(run(files, sub, runs_dir, tmp_path / "p2_final", split="test", tag_file=tag))


def test_series_table_lists_git_sha(world, tmp_path):
    runs_dir, files, ids = world
    for k, rid in enumerate(ids["h1-lb"]):
        (runs_dir / rid / "manifest.json").write_text(json.dumps({"split": SPLIT, "git_sha": f"sha{k}"}))
    run(files, ids, runs_dir, tmp_path / "p2_dev")
    ser = {r["key"]: r for r in tb.read_table(tmp_path / "p2_dev" / "p2_series_dev.csv")}
    assert ser["h1-lb"]["git_sha"] == "sha0;sha1"


# ---------------------------------------------------------------------------------------------- pages read before gold hit
def pages_series(a_lists, b_lists, a_status=None, drop_b=()):
    """S12 / S14 records with the given paged_in lists; the lb_items gold origins are {0, 1, 2}."""
    items = {i["item_id"]: i for i in tp.lb_items("p", len(a_lists))}
    ids = list(items)
    mk = lambda sy, lists, st: {iid: {"item_id": iid, "system": sy, "status": (st or {}).get(k, "ok"), "paged_in": pl_}
                                for k, (iid, pl_) in enumerate(zip(ids, lists)) if iid not in drop_b or sy == "S12"}
    return p2.Series("h5-mt", ["R"], items, {"S12": set(ids), "S14": set(ids)},
                     {"S12": mk("S12", a_lists, a_status), "S14": mk("S14", b_lists, None)})


def test_pages_before_gold_hand_computed():
    s = pages_series([[5, 0, 6], [1], [4, 5], [0]], [[5, 6, 0], [7, 8, 9, 2], [3, 1], [0]])
    r, ex = p2.pages_row(s, SPLIT, {})
    # pages: A2 = 1, 0, censored, 0; A1 = 2, 3, 1, 0; worst observed = 3; censored A2 = max(len 2, 3) = 3
    assert r["key"] == "A2_vs_A1_pages_read_before_gold_hit" and r["n"] == 4 and ex == []
    assert r["value"] == pytest.approx(np.mean([1 - 2, 0 - 3, 3 - 1, 0 - 0]))            # -0.5
    assert r["n_censored_a"] == 1 and r["n_censored_b"] == 0 and r["n_itt_imputed"] == 1
    assert r["em_a"] == pytest.approx(1.0) and r["em_b"] == pytest.approx(1.5)
    assert "no verdict" in r["test"] and "Wilcoxon" in r["test"] and r["p_holm"] is None


def test_pages_censored_uses_record_length_when_larger_and_failed_is_censored():
    s = pages_series([[0], [1], [9, 9, 9, 9, 9, 9]], [[7, 2], [7, 7, 1], [0]], a_status={1: "error"})
    r, _ = p2.pages_row(s, SPLIT, {})
    # observed: A2 item0 = 0; A1 = 1, 2, 0 -> worst 2. A2 item1 failed -> max(len 1, 2) = 2; A2 item2 no gold -> max(6, 2) = 6
    assert r["n_censored_a"] == 2 and r["n_censored_b"] == 0
    assert r["value"] == pytest.approx(np.mean([0 - 1, 2 - 2, 6 - 0]))


def test_pages_ordering_and_missing_pair():
    a = [[5, 0], [1], [2, 0], [9]]
    b = [[0], [5, 5, 1], [7, 7, 7, 0], [0]]
    r1, _ = p2.pages_row(pages_series(a, b), SPLIT, {})
    s = pages_series(a, b, drop_b={"p-003"})
    r2, ex = p2.pages_row(s, SPLIT, {})
    assert r2["n"] == 3 and r2["n_excluded_missing"] == 1 and ex == [{"item_id": "p-003", "reason": "missing S14"}]
    # record order is irrelevant: shuffle the dict insertion order
    for sy in s.records:
        s.records[sy] = dict(reversed(list(s.records[sy].items())))
    r3, _ = p2.pages_row(s, SPLIT, {})
    assert (r3["value"], r3["ci_lo"], r3["ci_hi"], r3["p_raw"]) == (r2["value"], r2["ci_lo"], r2["ci_hi"], r2["p_raw"])
    assert r1["n"] == 4


def test_pages_row_in_pipeline_and_table(world, tmp_path):
    runs_dir, files, ids = world
    res = run(files, ids, runs_dir, tmp_path / "p2_dev")
    # synthetic records carry no paged_in: every item is censored in both arms, so the contrast is 0 and well-formed
    r = rows_by_key(res)["A2_vs_A1_pages_read_before_gold_hit"]
    assert r["n"] == 24 and r["n_censored_a"] == 24 and r["value"] == 0.0
    row = {x["key"]: x for x in tb.read_table(tmp_path / "p2_dev" / "p2_contrasts_dev.csv")}["A2_vs_A1_pages_read_before_gold_hit"]
    assert row["n_censored_a"] == "24" and row["p_holm"] == ""


# ---------------------------------------------------------------------------------------------- opt-in GCS check
@pytest.mark.skipif(os.environ.get("OPA_GCS_TESTS") != "1", reason="set OPA_GCS_TESTS=1 to read the dev pilot runs from GCS")
def test_dev_pilot_matches_discordance_csv(tmp_path):
    from opa.analysis.pilot import RUNS
    creds = pl.credentials_path()
    if not os.environ.get("GCP_SA_JSON") and creds and Path(creds).exists():
        os.environ["GCP_SA_JSON"] = Path(creds).read_text(encoding="utf-8")
    ids = {"h1-lb": [RUNS["lb_main"]], "h1-s4": [RUNS["lb_main"]], "h5-mt": [RUNS["mt_arms"]],
           "h10-pin": [RUNS["thrash_pin"]], "h10-nopin": [RUNS["thrash_nopin"]]}
    res = p2.run_p2(pl.GcsStore(), ids, "dev", tmp_path / "p2_dev_check")
    rows = rows_by_key(res)
    ref = {r["key"]: r for r in tb.read_table(Path(__file__).resolve().parent.parent / "bundle" / "tables" / "pilot_dev" / "discordance.csv")}
    for k in (K_H1, K_S4, K_H3, K_H5, K_TOST, p2.K_H8):
        assert float(ref[k]["value"]) == pytest.approx(rows[k]["value"], abs=1e-9), k
        assert int(ref[k]["n"]) == rows[k]["n"], k
