"""H2 / H10 contrast builders (opa.analysis.drift_contrast, track_a.thrash.h10_contrast) on synthetic records."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from opa import claims as cl
from opa import stats
from opa.analysis import drift_contrast as dc
from opa.analysis import tables as tb
from opa.tracks.track_a.thrash import CONTRAST_KEY, h10_contrast

WORDS = ["amber", "cobalt", "violet", "scarlet", "ivory", "indigo", "teal", "crimson", "ochre", "mauve"]
NEW = ["jade", "slate", "coral", "olive", "navy", "peach", "plum", "rust", "sand", "lilac"]


def session(i: int) -> dict:
    cons = [{"key": f"c{k}", "unit": None, "initial": {"turn": 1, "value": WORDS[k]},
             "updates": [{"turn": 20, "value": NEW[k]}] if k % 2 == 0 else [], "near_misses": []} for k in range(10)]
    return {"item_id": f"drift-{i:03d}", "dataset": "drift", "constraints": cons,
            "turns": [{"turn": 1, "text": "start"}, {"turn": 20, "text": "update"}], "probes": [{"turn": 50}]}


def current(c):
    return c["updates"][-1]["value"] if c["updates"] else c["initial"]["value"]


def record(sess: dict, cond: str, n_correct: int) -> dict:
    pred = {}
    for k, c in enumerate(sess["constraints"]):
        pred[c["key"]] = current(c) if k < n_correct else c["initial"]["value"] if c["updates"] else "zzz-unknown"
    return {"item_id": sess["item_id"], "condition": cond, "status": "ok", "probes": [{"turn": 50, "predicted": pred}]}


def build(n=40, seed=0, d4_gain=2):
    rng = np.random.default_rng(seed)
    sessions = [session(i) for i in range(n)]
    c2 = rng.integers(3, 8, n)
    c4 = np.clip(c2 + rng.integers(0, d4_gain + 1, n), 0, 10)
    return (sessions, [record(s, "D2", int(k)) for s, k in zip(sessions, c2)],
            [record(s, "D4", int(k)) for s, k in zip(sessions, c4)], c2, c4)


def test_session_retention_counts_current_values_only():
    s = session(0)
    assert dc.session_retention(s, record(s, "D4", 10)) == 1.0
    assert dc.session_retention(s, record(s, "D4", 4)) == pytest.approx(0.4)
    assert dc.session_retention(s, {"item_id": s["item_id"], "probes": []}) == 0.0     # errored session: all missing


def test_contrast_matches_stats_contrast_and_reports_sd():
    sessions, r2, r4, c2, c4 = build()
    row = dc.drift_retention_contrast(sessions, r2, list(reversed(r4)), run_ids=["r-d2", "r-d4"], split="test")
    ref = stats.contrast(c4 / 10, c2 / 10, dc.KEY, test="wilcoxon", scale=100.0, run_ids=["r-d2", "r-d4"], split="test")
    for k in ("value", "ci_lo", "ci_hi", "n", "p_raw", "n10", "n01"):
        assert row[k] == pytest.approx(ref[k]), k
    assert row["key"] == "D4_vs_D2_retention_t50" and row["n"] == 40 and row["split"] == "test"
    assert row["sd_diff"] == pytest.approx(np.std((c4 - c2) * 10.0, ddof=1))
    assert row["value"] == pytest.approx(np.mean(c4 - c2) * 10.0)
    assert row["d4_mean"] >= row["d2_mean"] and "Wilcoxon" in row["test"]


def test_unpaired_sessions_are_excluded_and_zero_differences_dropped_in_p():
    sessions, r2, r4, _, _ = build(n=12)
    row = dc.drift_retention_contrast(sessions, r2[:-2], r4[1:], split="dev")
    assert row["n"] == 9 and len(row["excluded"]) == 3
    same = dc.drift_retention_contrast(sessions, r2, r2, split="dev")
    assert same["value"] == 0.0 and same["p_raw"] == 1.0 and same["sd_diff"] == 0.0
    with pytest.raises(ValueError):
        dc.drift_retention_contrast(sessions, r2[:1], r4[1:2], split="dev")


def test_excluded_reasons_including_missing_from_both_arms():
    sessions, r2, r4, _, _ = build(n=8)
    sessions.append(session(99))                                   # in the session list, in neither arm
    r2 = [r for r in r2 if r["item_id"] != "drift-001"]            # missing D2
    r4 = [r for r in r4 if r["item_id"] != "drift-002"]            # missing D4
    r2.append(record(session(77), "D2", 5))                        # record without a listed session
    row = dc.drift_retention_contrast(sessions, r2, r4, split="dev")
    why = {e["item_id"]: e["reason"] for e in row["excluded"]}
    assert why == {"drift-099": "missing both", "drift-001": "missing D2", "drift-002": "missing D4",
                   "drift-077": "not in session list"}
    assert row["n"] == 6 and row["n_excluded"] == 4


def test_duplicate_records_raise():
    sessions, r2, r4, _, _ = build(n=5)
    with pytest.raises(ValueError, match="duplicate record for session drift-000 in arm D2"):
        dc.drift_retention_contrast(sessions, r2 + [r2[0]], r4, split="dev")
    with pytest.raises(ValueError, match="arm D4"):
        dc.drift_retention_contrast(sessions, r2, r4 + [r4[3]], split="dev")


def test_error_counts_in_row_and_table(tmp_path):
    sessions, r2, r4, _, _ = build(n=10)
    r2[0] = {**r2[0], "status": "error", "probes": []}
    r4[1] = {**r4[1], "probes": [{"turn": 25, "predicted": {}}]}      # ok status but no probe at turn 50
    r4[2] = {**r4[2], "status": "budget"}
    row = dc.drift_retention_contrast(sessions, r2, r4, split="dev")
    assert (row["n_error_d2"], row["n_error_d4"], row["n_excluded"], row["n"]) == (1, 2, 0, 10)
    dc.write_drift_contrast(tmp_path / "t.csv", row)
    assert [cl.verifier().resolve_ref(f"t.csv#row={dc.KEY}&cols={c}", tmp_path)[0]
            for c in ("n_error_d2", "n_error_d4", "n_excluded")] == [1.0, 2.0, 0.0]


def test_not_fit_probes_are_scored_zero_and_counted():
    sessions, r2, r4, _, _ = build(n=6)
    r4[0] = {**r4[0], "probes": [{**r4[0]["probes"][0], "fits": False, "predicted": {}}]}
    r2[1] = {**r2[1], "probes": [{**r2[1]["probes"][0], "fits": False}]}
    row = dc.drift_retention_contrast(sessions, r2, r4, split="dev")
    assert (row["n_not_fit_d4"], row["n_not_fit_d2"], row["n_error_d4"], row["n_error_d2"]) == (1, 1, 0, 0)
    assert "n_not_fit_d2" in dc.EXTRA_COLUMNS and "n_not_fit_d4" in dc.EXTRA_COLUMNS


def test_split_is_required_and_restricted():
    sessions, r2, r4, _, _ = build(n=5)
    with pytest.raises(TypeError):
        dc.drift_retention_contrast(sessions, r2, r4)
    for bad in ("n/a", "train", None):
        with pytest.raises(ValueError, match="split"):
            dc.drift_retention_contrast(sessions, r2, r4, split=bad)
    assert dc.drift_retention_contrast(sessions, r2, r4, split="test")["split"] == "test"


def test_table_row_is_readable_by_claims(tmp_path):
    sessions, r2, r4, _, _ = build()
    row = dc.drift_retention_contrast(sessions, r2, r4, run_ids=["r-d2"], split="test")
    dc.write_drift_contrast(tmp_path / "tables" / "H2.csv", row)
    r = cl.result_from_table(tmp_path, "tables/H2.csv", dc.KEY, level="agent", primary=True)
    assert r["value"] == pytest.approx(row["value"]) and r["n"] == 40 and r["ci95"] is not None
    sd = tb.read_table(tmp_path / "tables" / "H2.csv")[0]["sd_diff"]
    assert float(sd) == pytest.approx(row["sd_diff"])


def test_read_records_gz(tmp_path):
    sessions, r2, _, _, _ = build(n=3)
    p = tmp_path / "records.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in r2) + "\n")
    assert dc.read_records(p) == r2


def test_h10_contrast_emits_sd_of_differences():
    unp = [{"item_id": f"i{k}", "M": 2, "r": 3, "repeated_page_ins": int(v), "status": "ok"}
           for k, v in enumerate([3, 4, 2, 5, 3, 6])]
    pin = [{"item_id": f"i{k}", "M": 2, "r": 3, "repeated_page_ins": int(v), "status": "ok"}
           for k, v in enumerate([0, 1, 0, 2, 0, 1])]
    c = h10_contrast(unp, pin, split="dev", n_boot=500)
    assert c["key"] == CONTRAST_KEY and c["n"] == 6
    assert c["sd_diff"] == pytest.approx(np.std(np.array([3, 3, 2, 3, 3, 5]), ddof=1))
