"""Blind H3 cell selection (opa.analysis.h3_cells) and the tie-deterministic r2_binned."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from opa import scoring, stats
from opa.analysis import h3_cells as hc
from opa.analysis import pilot as pl
from opa.analysis import tables as tb

B = "b"


class SpyStore(pl.MemStore):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fetched: list[str] = []

    def get(self, uri):
        self.fetched.append(uri)
        return super().get(uri)


def make_cell(runs_dir: Path, run_id: str, cell: str, covered_p: float, n=30, h_late=2, oracle=0.8, seed=0):
    """Items with h_late late-bound facts; S11 skeleton covers each fact with probability covered_p. The records also
    carry outcome fields (prediction, status) that the tool must never touch."""
    rng = np.random.default_rng(seed)
    items, recs = [], []
    for i in range(n):
        iid = f"{cell}-{i:03d}"
        facts = [{"origin_id": h, "late_bound": h < h_late, "subject": f"Ent{h}"} for h in range(3)]
        items.append({"item_id": iid, "answer": f"ans{i}", "gold": {"facts": facts, "fact_origins": [0, 1, 2]}})
        recs.append({"item_id": iid, "system": "S11", "prediction": f"ans{i}", "status": "ok", "em": 1.0,
                     "skeleton": {str(h): [f"Ent{h}"] if rng.random() < covered_p else ["zzz"] for h in range(3)},
                     "edges_from_resident": {}})
    d = runs_dir / run_id
    d.mkdir(parents=True)
    (d / "job.yaml").write_text(yaml.safe_dump({"split": "dev", "cell": cell, "args": {
        "items_uri": [{"uri": f"gs://{{bucket}}/data/{cell}/items.jsonl.gz", "n_items": n}]}}))
    pre = f"gs://{B}/opa/runs/{run_id}"
    return {f"gs://{B}/data/{cell}/items.jsonl.gz": pl.gz_jsonl(items),
            f"{pre}/S11/r0/records.jsonl.gz": pl.gz_jsonl(recs),
            f"{pre}/S11/r0/metrics.json": json.dumps({"em": 0.123456}).encode(),       # must stay unread
            f"{pre}/S18/r0/metrics.json": json.dumps({"em": oracle}).encode()}


def test_predictor_stats_and_qualification(tmp_path):
    runs_dir = tmp_path / "runs"
    files = {}
    files |= make_cell(runs_dir, "R-hard", "hard", covered_p=0.5, oracle=0.8, seed=1)       # varied predictions
    files |= make_cell(runs_dir, "R-easy", "easy", covered_p=1.0, oracle=0.9, seed=2)       # predicts 1.0 everywhere
    files |= make_cell(runs_dir, "R-weak", "weak", covered_p=0.5, oracle=0.4, seed=3)       # oracle too low
    store = SpyStore(files, B)
    res = hc.run_cells(store, ["R-hard", "R-easy", "R-weak"], tmp_path / "out", runs_dir, log=lambda s: None)
    rows = {r["key"]: r for r in res["rows"]}
    hard, easy, weak = rows["hard"], rows["easy"], rows["weak"]
    assert hard["n"] == 30 and hard["pred_sd"] >= 0.15 and hard["share_le_0_8"] >= 0.30 and hard["qualifies"] is True
    assert easy["pred_sd"] == 0.0 and easy["share_le_0_8"] == 0.0 and easy["qualifies"] is False
    assert weak["oracle_em"] == 0.4 and weak["qualifies"] is False and weak["pred_sd"] >= 0.15
    assert hard["n_h_late_0"] == 0
    # the table: predictor columns and oracle em only
    t = tb.read_table(tmp_path / "out" / "h3_cells.csv")
    assert list(t[0].keys()) == hc.OUTPUT_COLUMNS and {r["key"] for r in t} == {"hard", "easy", "weak"}


def test_predictor_matches_coverage_factors_and_h_late_zero_count():
    items = {"a": {"item_id": "a", "gold": {"facts": [{"origin_id": 0, "late_bound": False, "subject": "E0"}]}},
             "b": {"item_id": "b", "gold": {"facts": [{"origin_id": 0, "late_bound": True, "subject": "E0"},
                                                      {"origin_id": 1, "late_bound": True, "subject": "E1"}]}}}
    recs = [{"item_id": "a", "skeleton": {"0": ["E0"]}, "edges_from_resident": {}},
            {"item_id": "b", "skeleton": {"0": ["E0"], "1": ["x"]}, "edges_from_resident": {}}]
    st = hc.predictor_stats(items, recs)
    # a: h_late 0 -> predicted 1.0; b: gamma_s = 0.5, h_late 2 -> 0.25
    assert st["n_h_late_0"] == 1 and st["pred_mean"] == pytest.approx(0.625)
    assert st["pred_sd"] == pytest.approx(np.std([1.0, 0.25], ddof=1)) and st["share_le_0_8"] == 0.5


def test_blindness_no_outcome_is_read_or_computed(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    files = make_cell(runs_dir, "R-hard", "hard", covered_p=0.5, seed=1)
    store = SpyStore(files, B)

    def forbidden(*a, **k):
        raise AssertionError("an outcome scorer was called on the blind path")

    for fn in ("exact_match", "score_item", "f1_score", "score_records", "aggregate", "classify_failure"):
        monkeypatch.setattr(scoring, fn, forbidden)
    out = []
    res = hc.run_cells(store, ["R-hard"], tmp_path / "out", runs_dir, log=out.append)
    assert res["rows"]
    # S11's metrics.json is never fetched; only its records (projected) and S18's metrics.json
    assert not any(u.endswith("/S11/r0/metrics.json") for u in store.fetched)
    assert any(u.endswith("/S18/r0/metrics.json") for u in store.fetched)
    # no outcome column, and no outcome value (S11's em 0.123456, the records' predictions) in the output or the log
    cols = set(res["rows"][0]) | set(tb.read_table(tmp_path / "out" / "h3_cells.csv")[0])
    assert not cols & {"em", "f1", "prediction", "status", "failure_code", "accuracy", "s11_em"}
    text = (tmp_path / "out" / "h3_cells.csv").read_text() + "\n".join(out)
    assert "0.123456" not in text and "ans0" not in text


def test_projection_drops_every_outcome_field():
    rec = {"item_id": "x", "prediction": "p", "status": "ok", "skeleton": {"0": ["E"]}, "edges_from_resident": {},
           "tokens_in": 5, "failure_code": "F2", "cited_pin_ids": [1]}
    assert hc.project_record(rec) == {"item_id": "x", "skeleton": {"0": ["E"]}, "edges_from_resident": {},
                                      "budget_stop": False}
    assert hc.project_record({**rec, "status": "budget"})["budget_stop"] is True
    assert "status" not in hc.project_record({**rec, "status": "budget"})


def test_n_budget_stop_is_a_count_only(tmp_path):
    runs_dir = tmp_path / "runs"
    files = make_cell(runs_dir, "R-hard", "hard", covered_p=0.5, seed=1)
    key = f"gs://{B}/opa/runs/R-hard/S11/r0/records.jsonl.gz"
    recs = pl.read_jsonl(files[key])
    for r in recs[:4]:
        r["status"] = "budget"
    recs[4]["status"] = "error"                                   # not a budget stop
    files[key] = pl.gz_jsonl(recs)
    res = hc.run_cells(pl.MemStore(files, B), ["R-hard"], tmp_path / "out", runs_dir, log=lambda s: None)
    row = res["rows"][0]
    assert row["n_budget_stop"] == 4
    t = tb.read_table(tmp_path / "out" / "h3_cells.csv")[0]
    assert t["n_budget_stop"] == "4" and "status" not in t and "budget_stop" not in t


def test_missing_s11_records_are_skipped(tmp_path):
    runs_dir = tmp_path / "runs"
    files = make_cell(runs_dir, "R-hard", "hard", covered_p=0.5)
    del files[f"gs://{B}/opa/runs/R-hard/S11/r0/records.jsonl.gz"]
    res = hc.run_cells(pl.MemStore(files, B), ["R-hard"], tmp_path / "out", runs_dir, log=lambda s: None)
    assert res["rows"] == [] and "no S11 records yet" in res["skipped"][0] and res["path"] is None


def test_oracle_missing_means_not_qualified(tmp_path):
    runs_dir = tmp_path / "runs"
    files = make_cell(runs_dir, "R-hard", "hard", covered_p=0.5)
    del files[f"gs://{B}/opa/runs/R-hard/S18/r0/metrics.json"]
    res = hc.run_cells(pl.MemStore(files, B), ["R-hard"], tmp_path / "out", runs_dir, log=lambda s: None)
    assert res["rows"][0]["oracle_em"] == "" and res["rows"][0]["qualifies"] is False


# ---------------------------------------------------------------- r2_binned ties
def test_r2_binned_is_deterministic_under_ties_with_ids():
    rng = np.random.default_rng(0)
    n = 60
    pred = np.where(rng.random(n) < 0.6, 1.0, rng.random(n))            # many ties at 1.0 straddling bin edges
    obs = (rng.random(n) < 0.7).astype(float)
    ids = [f"i{k:03d}" for k in range(n)]
    base = stats.r2_binned(pred, obs, bins=10, ids=ids)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(n)
        r = stats.r2_binned(pred[perm], obs[perm], bins=10, ids=[ids[k] for k in perm])
        assert r["r2"] == base["r2"] and r["r2_linear"] == base["r2_linear"]
        assert [b["mean_observed"] for b in r["bins"]] == [b["mean_observed"] for b in base["bins"]]
    # without ids the old behaviour (stable in input order) is kept, and it does depend on the order
    plain = stats.r2_binned(pred, obs, bins=10)
    assert plain["n"] == base["n"]
    outs = {stats.r2_binned(pred[p], obs[p], bins=10)["r2"] for p in (np.random.default_rng(s).permutation(n) for s in range(8))}
    assert len(outs) > 1
    with pytest.raises(ValueError):
        stats.r2_binned(pred, obs, ids=ids[:5])


def test_pilot_h3_fit_ties_do_not_depend_on_record_order():
    rng = np.random.default_rng(1)
    per = [{"item_id": f"i{k:03d}", "predicted": 1.0 if k % 3 else 0.5 + 0.01 * (k % 7), "observed": float(rng.random() < 0.6),
            "h_late": 1, "gamma_s": 1.0, "gamma_g": 0.0, "skeleton_miss": [False], "graph_miss": [True]} for k in range(50)]
    a, _ = pl.h3_fit(per, ["r"])
    b, _ = pl.h3_fit(list(reversed(per)), ["r"])
    assert {r["key"]: r["value"] for r in a}[pl.K_R2] == {r["key"]: r["value"] for r in b}[pl.K_R2]
