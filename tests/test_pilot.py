"""P1.4 pilot analysis (opa.analysis.pilot) on synthetic runs held in an in-memory store."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from opa import stats
from opa.analysis import pilot as pl
from opa.analysis import tables as tb
from opa.tracks.track_a.embed import compute_item_embeddings, embeddings_to_bytes
from opa.tracks.track_a.index import HashEmbedder

sys.path.insert(0, str(Path(__file__).parent))
import test_drift_contrast as tdc  # noqa: E402  (synthetic drift sessions / records)
import test_routing_recall as trr  # noqa: E402  (synthetic routing items, skeletons)

B = "b"
RNG = np.random.default_rng(3)


def lb_items(prefix: str, n: int, h_late: int = 2):
    items = []
    for i in range(n):
        facts = [{"origin_id": h, "late_bound": h < h_late, "subject": f"Ent{h}"} for h in range(3)]
        items.append({"item_id": f"{prefix}-{i:03d}", "answer": f"ans{i}", "answer_aliases": [],
                      "gold": {"facts": facts, "fact_origins": [0, 1, 2], "confusers": []}})
    return items


def lb_records(items, system, p_correct, rng, run_flip=None):
    recs = []
    for it in items:
        ok = rng.random() < p_correct
        recs.append({"item_id": it["item_id"], "system": system, "prediction": it["answer"] if ok else "wrong",
                     "status": "ok", "skeleton": {str(h): [f"Ent{h}"] if rng.random() < 0.7 else [] for h in range(3)},
                     "edges_from_resident": {str(h): bool(rng.random() < 0.4) for h in range(3)}})
    return recs


def put_job(runs_dir: Path, run_id: str, args: dict, split="dev"):
    d = runs_dir / run_id
    d.mkdir(parents=True)
    (d / "job.yaml").write_text(yaml.safe_dump({"split": split, "args": args}))


def uri(run_id, rest):
    return f"gs://{B}/opa/runs/{run_id}/{rest}"


def build_world(tmp_path, *, with_main=True):
    runs_dir = tmp_path / "runs"
    files = {}
    rng = np.random.default_rng(1)
    # ---- LB late h2: main + two repeats + forward/mixed
    items = lb_items("lb-late", 40)
    files[f"gs://{B}/data/lb/items.jsonl.gz"] = pl.gz_jsonl(items)
    arg = {"items_uri": [{"uri": f"gs://{{bucket}}/data/lb/items.jsonl.gz", "n_items": 40}]}
    for name, systems in (("lb_main", ["S1", "S2", "S4", "S8", "S11"]), ("lb_rep2", ["S1", "S2", "S8"]),
                          ("lb_rep3", ["S1", "S2", "S8"])):
        put_job(runs_dir, pl.RUNS[name], arg)
        if name == "lb_main" and not with_main:
            continue
        for s in systems:
            p = {"S1": 0.5, "S2": 0.45, "S4": 0.5, "S8": 0.55, "S11": 0.7}[s]
            files[uri(pl.RUNS[name], f"{s}/r0/records.jsonl.gz")] = pl.gz_jsonl(lb_records(items, s, p, rng))
    fitems = lb_items("lb-fwd", 15, h_late=0) + lb_items("lb-mixed", 15, h_late=1)
    files[f"gs://{B}/data/lb/fm.jsonl.gz"] = pl.gz_jsonl(fitems)
    put_job(runs_dir, pl.RUNS["lb_fwd_mixed"], {"items_uri": [{"uri": f"gs://{{bucket}}/data/lb/fm.jsonl.gz", "n_items": 30}]})
    files[uri(pl.RUNS["lb_fwd_mixed"], "S11/r0/records.jsonl.gz")] = pl.gz_jsonl(lb_records(fitems, "S11", 0.8, rng))
    # ---- MT T4 arms + routing inputs
    mt_items = trr.make_items(24)
    sk = trr.make_skeletons(mt_items)
    for it in mt_items:
        it["answer"] = f"mt-{it['item_id']}"
    files[f"gs://{B}/data/mt/items.jsonl.gz"] = pl.gz_jsonl(mt_items)
    embed_run = "A-embed-test"
    files[uri(embed_run, "embeddings.npz")] = embeddings_to_bytes(compute_item_embeddings(mt_items, HashEmbedder()))
    sk_run = "A-sllm-test"
    for iid, payload in sk.items():
        files[uri(sk_run, f"skeleton_llm/{iid}.json")] = json.dumps(payload).encode()
    put_job(runs_dir, pl.RUNS["mt_arms"], {"items_uri": [{"uri": f"gs://{{bucket}}/data/mt/items.jsonl.gz", "n_items": 24}],
                                          "embedder": f"precomputed:gs://{{bucket}}/opa/runs/{embed_run}/embeddings.npz",
                                          "skeleton_llm_uri": f"gs://{{bucket}}/opa/runs/{sk_run}/skeleton_llm"})
    for s, p in (("S12", 0.6), ("S14", 0.5), ("S15", 0.58), ("S16", 0.4), ("S18", 0.9)):
        files[uri(pl.RUNS["mt_arms"], f"{s}/r0/records.jsonl.gz")] = pl.gz_jsonl(lb_records(mt_items, s, p, rng))
    # ---- drift
    sessions, r2, r4, _, _ = tdc.build(n=30)
    files[f"gs://{B}/data/drift/items.jsonl.gz"] = pl.gz_jsonl(sessions)
    darg = {"items_uri": [{"uri": f"gs://{{bucket}}/data/drift/items.jsonl.gz", "n_items": 30}]}
    for name, recs in (("drift_d2", r2), ("drift_d4", r4)):
        put_job(runs_dir, pl.RUNS[name], darg)
        files[uri(pl.RUNS[name], "records.jsonl.gz")] = pl.gz_jsonl(recs)
    # ---- thrash
    targ = {"items_uri": [{"uri": f"gs://{{bucket}}/data/th/items.jsonl.gz", "n_items": 40}]}
    titems = [{"item_id": f"th{i}", "answer": "x"} for i in range(40)]
    files[f"gs://{B}/data/th/items.jsonl.gz"] = pl.gz_jsonl(titems)
    for name, hi in (("thrash_nopin", 8), ("thrash_pin", 2)):
        put_job(runs_dir, pl.RUNS[name], targ)
        recs = [{"item_id": f"th{i}", "M": 2, "r": 4, "repeated_page_ins": int(rng.integers(0, hi)), "status": "ok"}
                for i in range(40)]
        files[uri(pl.RUNS[name], "records.jsonl.gz")] = pl.gz_jsonl(recs)
    for name in ("thrash_pin", "thrash_nopin"):
        recs = pl.read_jsonl(files[uri(pl.RUNS[name], "records.jsonl.gz")])
        files[uri(pl.RUNS[name], "metrics.json")] = json.dumps(
            {"prompt_version": "thrash-v2", "tuning_iteration": 1,
             "repeated_page_ins_mean": float(np.mean([r["repeated_page_ins"] for r in recs]))}).encode()
    for name, recs in (("drift_d2", r2), ("drift_d4", r4)):
        acc = float(np.mean([pl.dcm.session_retention(next(x for x in sessions if x["item_id"] == r["item_id"]), r) for r in recs]))
        files[uri(pl.RUNS[name], "metrics.json")] = json.dumps({"prompt_version": "drift-v3", "accuracy_t50": acc}).encode()
    # sub-run metrics.json: em as the harness reports it (mean of the per-item harness EM)
    all_items = {it["item_id"]: it for it in items + fitems + mt_items}
    for name in ("lb_main", "lb_rep2", "lb_rep3", "lb_fwd_mixed", "mt_arms"):
        for u in [k for k in list(files) if k.startswith(uri(pl.RUNS[name], "")) and k.endswith("/r0/records.jsonl.gz")]:
            recs = pl.read_jsonl(files[u])
            em = float(np.mean([pl.harness_em(all_items[r["item_id"]], r) for r in recs]))
            files[u.replace("records.jsonl.gz", "metrics.json")] = json.dumps({"em": em, "n": len(recs)}).encode()
    # Track B mini-LoRA pilot: 32 attribution items, identities = seeds 900..931, B1 listed in a different order
    seeds = list(range(900, 932))
    v3 = [int(rng.random() < 0.4) for _ in seeds]
    b1 = [int(rng.random() < 0.2) for _ in seeds]
    order = list(reversed(range(32)))

    def ident(i):
        return {"seed": seeds[i], "window": [i, i + 1, i + 2], "target_origin": i, "options_origin_ids": [i, i + 1],
                "gold_option": i % 2, "T": 3000 + i}

    def fe(arm, em, idx):
        return json.dumps({"status": "complete", "final": {"tag": "final", "arm": arm, "opt_step": 39},
                           "per_item": {"final": {"per_item": {"attrib_em": [em[i] for i in idx]},
                                                  "meta": {"attrib": [ident(i) for i in idx]}}}}).encode()

    files[uri(pl.TRACKB_RUNS["V3"], "final_eval.json")] = fe("V3", v3, list(range(32)))
    files[uri(pl.TRACKB_RUNS["B1"], "final_eval.json")] = fe("B1", b1, order)
    for k in range(20):
        files[uri(pl.REMEDIES_RUN, f"items/{k:03d}.json")] = json.dumps(
            {"item_id": f"r{k}", "result": {"R1": {"em": int(rng.random() < 0.3)}, "R5_8": {"em": int(rng.random() < 0.6)}}}).encode()
    files[uri(pl.REMEDIES_RUN, "result.json")] = b"{}"
    return pl.MemStore(files, B), runs_dir


def test_full_pipeline_writes_every_table(tmp_path):
    store, runs_dir = build_world(tmp_path)
    out = tmp_path / "out"
    res = pl.run_all(store, out, runs_dir, log=lambda s: None)
    assert res["skipped"] == []
    names = {Path(w).name for w in res["written"]}
    assert names == {"discordance.csv", "sd_paired.csv", "h3_fit.csv", "h3_bins.csv", "noise.csv", "sample_sizes_proposed.csv"}
    disc = {r["key"]: r for r in tb.read_table(out / "discordance.csv")}
    assert set(disc) == {pl.K_S11_S2, pl.K_S11_S4, pl.K_S11_S8, pl.K_A2_A1, pl.K_A2_A3, pl.K_H8, pl.K_V3_B1,
                         pl.K_R5_R1 + "_remedies_pilot_info"}
    tbrow = disc[pl.K_V3_B1]
    assert tbrow["n"] == "32" and tbrow["note"] == "mini-LoRA pilot, 39 steps, n=32"
    for r in disc.values():
        assert r["split"] == "dev" and r["run_ids"] and r["oracle"] == "false"
        assert float(r["p_d"]) == pytest.approx((int(r["n10"]) + int(r["n01"])) / int(r["n"]))
    assert int(disc[pl.K_S11_S2]["n"]) == 40 and int(disc[pl.K_H8]["n"]) == 24
    assert disc[pl.K_H8]["p_raw"] == ""                               # NI primary: CI only
    sd = {r["key"]: r for r in tb.read_table(out / "sd_paired.csv")}
    assert float(sd[pl.K_H2]["sd_diff"]) > 0 and float(sd[pl.K_H10]["sd_diff"]) > 0 and sd[pl.K_H2]["n"] == "30"
    for col in ("n_error_d2", "n_error_d4", "n_not_fit_d2", "n_not_fit_d4", "n_excluded"):
        assert sd[pl.K_H2][col] != ""
    for col in ("n_error_unpinned", "n_budget_pinned", "n_not_fit_unpinned", "n_excluded", "imputed_worst"):
        assert sd[pl.K_H10][col] != ""
    assert sd[pl.K_H10]["prompt_version"] == "thrash-v2" and sd[pl.K_H10]["tuning_iteration"] == "1;1"
    assert sd[pl.K_H2]["prompt_version"] == "drift-v3"
    h3 = {r["key"]: r for r in tb.read_table(out / "h3_fit.csv")}
    assert {pl.K_R2, "R2_fitted_line", "logistic_h_late", "phi_skeleton_miss_graph_miss"} <= set(h3)
    assert all(r["split"] == "dev" for r in h3.values()) and h3[pl.K_R2]["n"] == "70"
    noise = {r["key"]: r for r in tb.read_table(out / "noise.csv")}
    assert set(noise) == {"noise_S1", "noise_S2", "noise_S8"} and 0 <= float(noise["noise_S1"]["flip_rate"]) <= 1
    assert noise["noise_S1"]["n_repeats"] == "3"


def test_sample_sizes_follow_the_formulas(tmp_path):
    store, runs_dir = build_world(tmp_path)
    out = tmp_path / "out"
    pl.run_all(store, out, runs_dir, log=lambda s: None)
    disc = {r["key"]: r for r in tb.read_table(out / "discordance.csv")}
    ss = {r["key"]: r for r in tb.read_table(out / "sample_sizes_proposed.csv")}
    pd1 = float(disc[pl.K_S11_S2]["p_d"])
    assert int(ss[pl.K_S11_S2]["n_required"]) == stats.n_mcnemar(pd1, 0.05) and ss[pl.K_S11_S2]["p_d_source"] == "pilot"
    pd3 = float(disc[pl.K_A2_A3]["p_d"])
    assert int(ss[pl.K_A2_A3]["n_required"]) == stats.n_tost(pd3, 0.03)
    assert int(ss[pl.K_A2_A1 + "|delta=5"]["n_required"]) == stats.n_mcnemar(float(disc[pl.K_A2_A1]["p_d"]), 0.05)
    h8 = ss[pl.K_H8]
    if h8["n_required"] == "":
        assert "undefined" in h8["note"] and float(disc[pl.K_H8]["p_d"]) <= 0.03 ** 2
    else:
        assert int(h8["n_required"]) == stats.n_mcnemar(float(disc[pl.K_H8]["p_d"]), 0.03)
    sd = {r["key"]: r for r in tb.read_table(out / "sd_paired.csv")}
    n_h2 = pl.n_paired_normal(float(sd[pl.K_H2]["sd_diff"]), 10.0)
    assert int(ss[pl.K_H2]["n_required"]) == n_h2 and int(ss[pl.K_H2]["n_proposed"]) == max(n_h2, 200)
    n_h10 = pl.n_paired_normal(float(sd[pl.K_H10]["sd_diff"]), 0.5)
    assert int(ss[pl.K_H10]["n_proposed"]) == max(n_h10, 100)
    v3 = ss[pl.K_V3_B1]
    assert v3["p_d_source"] == "pilot" and "mini-LoRA pilot, 39 steps, n=32" in v3["note"]
    assert float(v3["p_d"]) == pytest.approx(float(disc[pl.K_V3_B1]["p_d"]))
    other = ss["A2_vs_A1_model_thread_confusion_task"]
    assert float(other["p_d"]) == 0.30 and int(other["n_required"]) == stats.n_mcnemar(0.30, 0.05)
    assert other["p_d_source"].startswith("default 0.30")
    assert ss["R5_vs_R1_accuracy"]["n_proposed"] == "400"
    assert not (tmp_path / "out" / "sample_sizes.yaml").exists()


def test_paired_normal_formula_value():
    # ((1.95996 + 0.84162) * 20 / 10)^2 / 0.955 = 32.87 -> 33
    assert pl.n_paired_normal(20.0, 10.0) == 33
    assert pl.n_paired_normal(2.0, 10.0) == 1


def test_missing_inputs_are_skipped_with_messages(tmp_path):
    store, runs_dir = build_world(tmp_path, with_main=False)
    out = tmp_path / "out"
    res = pl.run_all(store, out, runs_dir, log=lambda s: None)
    sk = " | ".join(res["skipped"])
    assert "lb_main" in sk and "h3_fit" in sk and "noise" in sk
    names = {Path(w).name for w in res["written"]}
    assert {"discordance.csv", "sd_paired.csv", "sample_sizes_proposed.csv"} <= names and "noise.csv" not in names
    disc = {r["key"] for r in tb.read_table(out / "discordance.csv")}
    assert pl.K_S11_S2 not in disc and pl.K_A2_A1 in disc
    ss = {r["key"]: r for r in tb.read_table(out / "sample_sizes_proposed.csv")}
    assert ss[pl.K_S11_S2]["p_d_source"].startswith("pending") and ss[pl.K_S11_S2]["n_required"] == ""
    # a store with nothing at all: no crash, every section skipped, sample-size table still written
    res = pl.run_all(pl.MemStore({}, B), tmp_path / "out2", runs_dir, log=lambda s: None)
    assert [Path(w).name for w in res["written"]] == ["sample_sizes_proposed.csv"] and len(res["skipped"]) >= 5


def test_non_dev_job_is_refused(tmp_path):
    runs_dir = tmp_path / "runs"
    put_job(runs_dir, "X-run", {"items_uri": []}, split="test")
    with pytest.raises(ValueError, match="dev runs only"):
        pl.load_job("X-run", runs_dir)


def test_oracle_systems_never_pair(tmp_path):
    store, runs_dir = build_world(tmp_path)
    run = pl.load_multi_run(store, pl.RUNS["mt_arms"], runs_dir)
    with pytest.raises(ValueError, match="oracle"):
        pl.pair_row("k", run, "S12", "S18")


def test_errors_score_zero_and_are_counted(tmp_path):
    items = {"i": {"item_id": "i", "answer": "a"}}
    em, n_err, unk = pl.em_by_item(items, [{"item_id": "i", "prediction": "a", "status": "error"}])
    assert em == {"i": 0.0} and n_err == 1 and unk == set()
    em, n_err, _ = pl.em_by_item(items, [{"item_id": "i", "prediction": "a"}])           # missing status = error
    assert em == {"i": 0.0} and n_err == 1
    with pytest.raises(ValueError, match="duplicate record for item i"):
        pl.em_by_item(items, [{"item_id": "i", "status": "ok"}, {"item_id": "i", "status": "ok"}])


def test_pair_row_reports_excluded_items_with_reasons(tmp_path):
    items = {f"i{k}": {"item_id": f"i{k}", "answer": "a"} for k in range(6)}
    ok = lambda i: {"item_id": i, "prediction": "a", "status": "ok"}
    run = pl.Run("R", {}, items, {"S11": {0: [ok("i0"), ok("i1"), ok("i2"), ok("zz")]},
                                  "S2": {0: [ok("i0"), ok("i1"), ok("i3")]}})
    row = pl.pair_row("k", run, "S11", "S2", verify_em=False)
    why = {e["item_id"]: e["reason"] for e in row["excluded"]}
    assert row["n"] == 2 and why == {"i2": "missing S2", "i3": "missing S11", "i4": "missing both",
                                     "i5": "missing both", "zz": "not in item list"}
    assert row["n_excluded"] == 5
    run.records["S2"][0].append(ok("i0"))
    with pytest.raises(ValueError, match="duplicate record for item i0"):
        pl.pair_row("k", run, "S11", "S2", verify_em=False)


def test_layout_mismatch_vs_not_finished(tmp_path):
    runs_dir = tmp_path / "runs"
    rid = "A-S1-layout"
    put_job(runs_dir, rid, {"items_uri": []})
    with pytest.raises(pl.MissingInput, match="not finished"):
        pl.load_multi_run(pl.MemStore({}, B), rid, runs_dir)
    files = {uri(rid, "result.json"): b"{}", uri(rid, "records.jsonl.gz"): pl.gz_jsonl([])}
    with pytest.raises(pl.MissingInput, match=r"layout mismatch.*records\.jsonl\.gz"):
        pl.load_multi_run(pl.MemStore(files, B), rid, runs_dir)
    with pytest.raises(pl.MissingInput, match="layout mismatch"):
        pl.load_multi_run(pl.MemStore({uri(rid, "result.json"): b"{}"}, B), rid, runs_dir)
    with pytest.raises(pl.MissingInput, match="layout mismatch"):
        pl.load_single_run(pl.MemStore({uri(rid, "result.json"): b"{}"}, B), rid, runs_dir)


def test_h8_passes_job_opts(tmp_path, monkeypatch):
    store, runs_dir = build_world(tmp_path)
    mt = pl.load_multi_run(store, pl.RUNS["mt_arms"], runs_dir)
    mt.job["args"]["opts"] = {"b": 4}
    seen = {}
    orig = pl.rr.routing_recall_rows

    def spy(*a, **k):
        seen.update(k)
        return orig(*a, **k)

    monkeypatch.setattr(pl.rr, "routing_recall_rows", spy)
    pl.h8_row(store, mt)
    assert seen["opts"] == {"b": 4}


def test_thrash_v1_runs_are_refused(tmp_path):
    store, runs_dir = build_world(tmp_path)
    store.files[uri(pl.RUNS["thrash_pin"], "metrics.json")] = json.dumps({"prompt_version": "thrash-v1"}).encode()
    d2, d4 = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("drift_d2", "drift_d4"))
    pin, nopin = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("thrash_pin", "thrash_nopin"))
    with pytest.raises(ValueError, match="thrash-v2"):
        pl.sd_rows(d2, d4, pin, nopin)


def test_trackb_pairs_by_item_identity_and_remedies_info(tmp_path):
    store, runs_dir = build_world(tmp_path)
    row = pl.trackb_attribution_row(store)
    assert row["n"] == 32 and row["p_d"] == pytest.approx((row["n10"] + row["n01"]) / 32)
    # pairing by seed, not position: shuffling B1's stored order must not change the discordance
    d = json.loads(store.files[uri(pl.TRACKB_RUNS["B1"], "final_eval.json")])
    blk = d["per_item"]["final"]
    assert [m["seed"] for m in blk["meta"]["attrib"]][0] == 931          # stored in reverse order
    by_seed = {m["seed"]: e for m, e in zip(blk["meta"]["attrib"], blk["per_item"]["attrib_em"])}
    d3 = json.loads(store.files[uri(pl.TRACKB_RUNS["V3"], "final_eval.json")])["per_item"]["final"]
    v3_by_seed = {m["seed"]: e for m, e in zip(d3["meta"]["attrib"], d3["per_item"]["attrib_em"])}
    assert row["n10"] == sum(v3_by_seed[k] == 1 and by_seed[k] == 0 for k in by_seed)
    assert row["n01"] == sum(v3_by_seed[k] == 0 and by_seed[k] == 1 for k in by_seed)
    info = pl.remedies_info_row(store)
    assert info["key"] == "R5_vs_R1_accuracy_remedies_pilot_info" and info["n"] == 20 and "information only" in info["note"]
    bad = pl.MemStore({k: v for k, v in store.files.items() if "final_eval" not in k}, B)
    with pytest.raises(pl.MissingInput):
        pl.trackb_attribution_row(bad)


def test_credentials_path_comes_from_the_project_records(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "credentials.md").write_text("- **Key file (primary)**: `C:/keys/sa.json`\n- **Key file (alt)**: `C:/keys/alt.json`\n")
    (tmp_path / "config" / "project.yaml").write_text("gcp: {bucket: x}\n")
    assert pl.credentials_path(tmp_path) == "C:/keys/sa.json"
    (tmp_path / "config" / "project.yaml").write_text("gcp: {credentials_file: D:/other.json}\n")
    assert pl.credentials_path(tmp_path) == "D:/other.json"
    assert pl.credentials_path(tmp_path / "nowhere") is None


def test_cli_reads_credentials_file_without_printing(tmp_path, monkeypatch, capsys):
    cred = tmp_path / "sa.json"
    cred.write_text('{"secret": "do-not-print"}')
    monkeypatch.delenv("GCP_SA_JSON", raising=False)
    monkeypatch.setattr(pl, "GcsStore", lambda: pl.MemStore({}, B))
    monkeypatch.setattr(pl, "run_all", lambda store, out, runs_dir=None, log=print: {"written": ["x"], "skipped": []})
    assert pl.main(["--out", str(tmp_path / "o"), "--creds", str(cred)]) == 0
    import os
    assert "do-not-print" in os.environ["GCP_SA_JSON"]
    assert "do-not-print" not in capsys.readouterr().out
    monkeypatch.delenv("GCP_SA_JSON", raising=False)


def test_gcs_store_translates_not_found_to_key_error():
    class NotFound(Exception):                          # same class name as google.api_core.exceptions.NotFound
        pass

    class Fake:
        def get_bytes(self, uri):
            raise NotFound(uri)

    st = pl.GcsStore.__new__(pl.GcsStore)
    st.bucket, st._gcs = B, Fake()
    with pytest.raises(KeyError):
        st.get("gs://b/x")
    # and the "not there yet" paths then skip instead of crashing
    assert pl._metrics(st, "whatever") == {}

    class Boom:
        def get_bytes(self, uri):
            raise RuntimeError("network")

    st._gcs = Boom()
    with pytest.raises(RuntimeError):
        st.get("gs://b/x")


def test_thrash_run_without_prompt_version_is_refused(tmp_path):
    store, runs_dir = build_world(tmp_path)
    store.files[uri(pl.RUNS["thrash_nopin"], "metrics.json")] = b"{}"
    d2, d4 = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("drift_d2", "drift_d4"))
    pin, nopin = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("thrash_pin", "thrash_nopin"))
    with pytest.raises(ValueError, match="missing or wrong"):
        pl.sd_rows(d2, d4, pin, nopin)


def test_trackb_item_identity_mismatch_raises(tmp_path):
    store, _ = build_world(tmp_path)
    key = uri(pl.TRACKB_RUNS["B1"], "final_eval.json")
    for field, value in (("T", 1), ("window", [0]), ("target_origin", 99), ("gold_option", 7), ("options_origin_ids", [])):
        d = json.loads(store.files[key])
        d["per_item"]["final"]["meta"]["attrib"][5][field] = value
        bad = pl.MemStore({**store.files, key: json.dumps(d).encode()}, B)
        with pytest.raises(ValueError, match="not the same item"):
            pl.trackb_attribution_row(bad)


def test_consistency_check_raises_on_a_metrics_mismatch(tmp_path):
    store, runs_dir = build_world(tmp_path)
    run = pl.load_multi_run(store, pl.RUNS["lb_main"], runs_dir)
    for sy in ("S1", "S2", "S4", "S8", "S11"):
        assert pl.check_em_consistency(run, sy) == pytest.approx(run.sub_metrics[(sy, 0)]["em"], abs=1e-9)
    run.sub_metrics[("S4", 0)]["em"] += 0.02                     # the harness reported something else
    with pytest.raises(ValueError, match="S4 r0.*metrics.json em"):
        pl.pair_row("k", run, "S11", "S4")
    run.sub_metrics[("S8", 0)]["em"] += 0.02
    with pytest.raises(ValueError, match="S8 r0.*metrics.json em"):
        pl.noise_rows([run, run, run])
    del run.sub_metrics[("S4", 0)]
    with pytest.raises(pl.MissingInput, match="cannot verify"):
        pl.check_em_consistency(run, "S4")


def test_per_item_em_is_the_harness_score_item(monkeypatch):
    """em_by_item must read what scoring.score_item reads, not re-score the raw prediction itself."""
    calls = []
    real = pl.scoring.score_item

    def spy(item, record, **kw):
        calls.append(record["item_id"])
        return real(item, record, **kw)

    monkeypatch.setattr(pl.scoring, "score_item", spy)
    items = {"i": {"item_id": "i", "answer": "x1"}}
    em, _, _ = pl.em_by_item(items, [{"item_id": "i", "prediction": "x1", "status": "ok"}])
    assert em == {"i": 1.0} and calls == ["i"]


def test_drift_and_thrash_consistency_checks(tmp_path):
    store, runs_dir = build_world(tmp_path)
    d2, d4 = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("drift_d2", "drift_d4"))
    pin, nopin = (pl.load_single_run(store, pl.RUNS[n], runs_dir) for n in ("thrash_pin", "thrash_nopin"))
    pl.sd_rows(d2, d4, pin, nopin)
    d4.metrics["accuracy_t50"] += 0.01
    with pytest.raises(ValueError, match="accuracy_t50"):
        pl.sd_rows(d2, d4, pin, nopin)
    d4.metrics["accuracy_t50"] -= 0.01
    nopin.metrics["repeated_page_ins_mean"] += 0.1
    with pytest.raises(ValueError, match="repeated_page_ins_mean"):
        pl.sd_rows(d2, d4, pin, nopin)
    del nopin.metrics["repeated_page_ins_mean"]
    with pytest.raises(pl.MissingInput):
        pl.sd_rows(d2, d4, pin, nopin)
