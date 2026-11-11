"""Blind routing-hit check (opa.analysis.routing_hits) on synthetic runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from opa import scoring
from opa.analysis import pilot as pl
from opa.analysis import routing_hits as rh
from opa.analysis import tables as tb

B = "b"
ENT = ["Urnek Fluidics", "Blonvar Group", "Trivembat GmbH"]


class SpyStore(pl.MemStore):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fetched: list[str] = []

    def get(self, uri):
        self.fetched.append(uri)
        return super().get(uri)


def item(i: int, cell="t4"):
    facts = [{"hop": 0, "origin_id": 10, "subject": ENT[0], "reveals": ENT[1]},
             {"hop": 1, "origin_id": 20, "subject": ENT[1], "reveals": ENT[2]},
             {"hop": 2, "origin_id": 30, "subject": ENT[2], "reveals": None}]
    return {"item_id": f"{cell}-{i:03d}", "answer": f"secret{i}", "gold": {"facts": facts, "fact_origins": [10, 20, 30]}}


def record(it, *, hit1=True, hit2=True, paged=(10, 20, 30), status="ok"):
    sk = {"10": ["urnek fluidics"], "20": ["blonvar group"] if hit1 else ["other"],
          "30": ["trivembat gmbh"] if hit2 else ["other"]}
    return {"item_id": it["item_id"], "system": "S16", "skeleton": sk, "paged_in": list(paged), "status": status,
            "prediction": it["answer"], "em": 1.0}


def make_run(tmp_path, run_id="R-t4", cell="t4", n=10, misses=0, unpaged=0, budget=0):
    items = [item(i, cell) for i in range(n)]
    recs = []
    for k, it in enumerate(items):
        recs.append(record(it, hit2=k >= misses, paged=(10, 20) if k < unpaged else (10, 20, 30),
                           status="budget" if k < budget else "ok"))
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    (d / "job.yaml").write_text(yaml.safe_dump({"split": "dev", "cell": cell, "args": {
        "items_uri": [{"uri": f"gs://{{bucket}}/data/{cell}/items.jsonl.gz", "n_items": n}]}}))
    pre = f"gs://{B}/opa/runs/{run_id}"
    return {f"gs://{B}/data/{cell}/items.jsonl.gz": pl.gz_jsonl(items),
            f"{pre}/S16/r0/records.jsonl.gz": pl.gz_jsonl(recs),
            f"{pre}/S16/r0/metrics.json": json.dumps({"em": 0.424242}).encode()}      # must stay unread


def test_hit_rates_per_hop_and_threshold(tmp_path):
    files = make_run(tmp_path, n=10, misses=3, unpaged=3)
    res = rh.run_cells(pl.MemStore(files, B), ["R-t4"], "S16", tmp_path / "out", tmp_path / "runs", log=lambda s: None)
    r = res["rows"][0]
    assert r["n"] == 10 and r["n_hops"] == 20
    assert r["hop1_rate"] == 1.0 and r["hop2_rate"] == pytest.approx(0.7) and r["hop3_rate"] == ""
    assert r["hit_rate"] == pytest.approx(17 / 20) and r["paged_rate"] == pytest.approx(17 / 20)
    assert r["routed_rate"] == pytest.approx(17 / 20) and r["qualifies"] is True         # routed_rate < 0.90 selects
    assert r["M"] == 4 and "qualifies_paged" not in r
    full = rh.run_cells(pl.MemStore(make_run(tmp_path / "x", n=10), B), ["R-t4"], "S16", tmp_path / "o2",
                        tmp_path / "x" / "runs", log=lambda s: None)["rows"][0]
    assert full["hit_rate"] == 1.0 and full["routed_rate"] == 1.0 and full["qualifies"] is False
    t = tb.read_table(tmp_path / "out" / "routing_hits.csv")
    assert list(t[0].keys()) == rh.OUTPUT_COLUMNS and t[0]["split"] == "dev"


def test_lookup_uses_the_session_key_logic():
    from opa.tracks.track_a.text import alias_keys, normalize_key
    # an alias key of the revealed entity counts as a hit, a different entity does not
    ent = "Ostrand Ltd."
    ak = sorted(alias_keys(normalize_key(ent)))
    assert rh.lookup_hit({"7": [normalize_key(ent)]}, ent, 7) and rh.lookup_hit({"7": ak[:1]}, ent, 7) == bool(ak)
    assert not rh.lookup_hit({"7": ["kelvane"]}, ent, 7) and not rh.lookup_hit({}, ent, 7)
    assert rh.lookup_hit({7: ["blonvar group"]}, "Blonvar Group", 7)                  # int-keyed skeleton too


def test_routed_rate_uses_the_initial_window_and_faults(tmp_path):
    its = [item(i) for i in range(4)]
    recs = [
        # initial window (M = 3) holds 10, 20, 30: both hops in the window
        {**record(its[0], paged=(10, 20, 30, 7)), "faults": []},
        # 20 only reached by a fault later; 30 never
        {**record(its[1], paged=(10, 7, 8, 20, 30)), "faults": [{"origin_id": 20, "entity": "x", "via": "skeleton"}]},
        # nothing routed except the first fact
        {**record(its[2], paged=(10, 7, 8, 9, 20, 30)), "faults": []},
        # string origin ids in paged_in and faults
        {**record(its[3], paged=("7", "8", "9", "20", "30")), "faults": [{"origin_id": "20"}, {"origin_id": "30"}]},
    ]
    proj = [rh.project_record(r) for r in recs]
    st = rh.hit_stats({i["item_id"]: i for i in its}, proj, M=3)
    # hops (20, 30) per item: window -> [2,0,0,0]; routed -> [2, 1, 0, 2]
    assert st["n_hops"] == 8 and st["initial_window_rate"] == pytest.approx(2 / 8) and st["routed_rate"] == pytest.approx(5 / 8)
    assert st["paged_rate"] == 1.0 and st["hit_rate"] == 1.0 and st["qualifies"] is True
    # M comes from the job's opts
    files = make_run(tmp_path, n=3)
    jp = tmp_path / "runs" / "R-t4" / "job.yaml"
    job = yaml.safe_load(jp.read_text())
    job["args"]["opts"] = {"M": 2}
    jp.write_text(yaml.safe_dump(job))
    r = rh.run_cells(pl.MemStore(files, B), ["R-t4"], "S16", tmp_path / "out", tmp_path / "runs", log=lambda s: None)["rows"][0]
    assert r["M"] == 2 and r["initial_window_rate"] == 0.5          # window (10, 20): hop to 20 in, hop to 30 out


def test_list_valued_reveals_and_string_ids():
    it = item(0)
    it["gold"]["facts"][0]["reveals"] = ["Nothing Known", "Blonvar Group"]       # any entity of the list may hit
    it["gold"]["facts"][1]["reveals"] = ["Trivembat GmbH"]
    rec = {**record(it), "paged_in": ["10", "20", "30"], "faults": [{"origin_id": "30"}]}
    hops = rh.item_hops(it, rh.project_record(rec), M=2)
    assert [h["lookup_hit"] for h in hops] == [True, True]
    assert [h["in_window"] for h in hops] == [True, False] and [h["routed"] for h in hops] == [True, True]
    it["gold"]["facts"][0]["reveals"] = None                                     # nothing revealed: no hit, no crash
    assert rh.item_hops(it, rh.project_record(rec), M=2)[0]["lookup_hit"] is False


def test_budget_stop_count_only(tmp_path):
    files = make_run(tmp_path, n=10, budget=3)
    r = rh.run_cells(pl.MemStore(files, B), ["R-t4"], "S16", tmp_path / "out", tmp_path / "runs", log=lambda s: None)["rows"][0]
    assert r["n_budget_stop"] == 3


def test_blindness(tmp_path, monkeypatch):
    files = make_run(tmp_path, n=10, budget=2)
    store = SpyStore(files, B)

    def forbidden(*a, **k):
        raise AssertionError("an outcome scorer was called on the blind path")

    for fn in ("exact_match", "score_item", "f1_score", "score_records", "aggregate", "classify_failure",
               "coverage_factors"):
        monkeypatch.setattr(scoring, fn, forbidden)
    out = []
    res = rh.run_cells(store, ["R-t4"], "S16", tmp_path / "out", tmp_path / "runs", log=out.append)
    assert res["rows"]
    assert not any(u.endswith("/S16/r0/metrics.json") for u in store.fetched)
    cols = set(res["rows"][0]) | set(tb.read_table(tmp_path / "out" / "routing_hits.csv")[0])
    assert not cols & {"em", "f1", "prediction", "status", "failure_code", "accuracy"}
    text = (tmp_path / "out" / "routing_hits.csv").read_text() + "\n".join(out)
    assert "0.424242" not in text and "secret0" not in text


def test_projection_drops_outcomes():
    rec = {"item_id": "x", "skeleton": {"1": ["a"]}, "paged_in": [1], "prediction": "p", "status": "error",
           "failure_code": "F2", "em": 1.0, "tokens_in": 4}
    p = rh.project_record(rec)
    assert set(p) == {"item_id", "skeleton", "paged_in", "fault_origins", "budget_stop"} and p["budget_stop"] is False
    assert rh.project_record({**rec, "faults": [{"origin_id": 5, "entity": "e", "via": "x"}]})["fault_origins"] == [5]
    assert rh.project_record({**rec, "status": "budget"})["budget_stop"] is True


def test_missing_records_skip_and_oracle_refused(tmp_path):
    files = make_run(tmp_path)
    del files[f"gs://{B}/opa/runs/R-t4/S16/r0/records.jsonl.gz"]
    res = rh.run_cells(pl.MemStore(files, B), ["R-t4"], "S16", tmp_path / "out", tmp_path / "runs", log=lambda s: None)
    assert res["rows"] == [] and "no S16 records yet" in res["skipped"][0]
    with pytest.raises(SystemExit):
        rh.main(["--run", "x", "--system", "S18", "--out", str(tmp_path / "o")])
