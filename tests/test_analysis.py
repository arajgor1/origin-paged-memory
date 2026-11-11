"""P0.12: opa.analysis tables and figures in the bundle formats (bundle_spec/BUNDLE_SPEC.md), checked with the
verifier's own reference resolution (read-only import of the human-owned verifier)."""
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opa import PROJECT_DIR
from opa import stats as st
from opa.analysis import figures as fg
from opa.analysis import tables as tb


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_bundle_ro", PROJECT_DIR / "tools" / "verify_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ tables
def test_format_cell():
    assert tb.format_cell(None) == ""
    assert tb.format_cell(True) == "true" and tb.format_cell(np.bool_(False)) == "false"
    assert tb.format_cell(np.int64(312)) == "312"
    assert tb.format_cell(5.0) == "5.0" and tb.format_cell(-0.0) == "0.0"
    assert tb.format_cell(1e-7) == "0.0000001"                     # never scientific notation
    assert tb.format_cell(float("nan")) == "nan" and tb.format_cell(float("-inf")) == "-inf"
    assert tb.format_cell(["a", "b"]) == "a;b"
    for x in (0.1, 1 / 3, 7.234567891234567, 1e-12, 123456789.123, np.float64(2) ** -30):
        assert float(tb.format_cell(x)) == float(x)                 # lossless round trip


def _contrast_rows():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    r1 = st.contrast(a, b, "S11_vs_S2_lbmhqa_late_h3_128k", run_ids=["run-a", "run-b"], n_boot=2000)
    r2 = st.contrast(b, a, "S11_vs_S4_lbmhqa_late_h3_128k", run_ids=["run-c"], n_boot=2000)
    holm = st.holm({r1["key"]: r1["p_raw"], r2["key"]: r2["p_raw"]})
    for r in (r1, r2):
        r["p_holm"] = holm[r["key"]]
    return [r1, r2]


def test_contrast_table_round_trip_and_determinism(tmp_path):
    rows = _contrast_rows()
    p = tb.write_contrast_table(tmp_path / "tables" / "H1_main.csv", rows, extra_columns=["n10", "n01"])
    text = p.read_bytes()
    assert text.splitlines()[0] == b"key,value,ci_lo,ci_hi,n,p_raw,p_holm,test,run_ids,n10,n01"
    assert b"\r\n" not in text
    tb.write_contrast_table(tmp_path / "again.csv", rows, extra_columns=["n10", "n01"])
    assert (tmp_path / "again.csv").read_bytes() == text            # byte-identical
    back = tb.read_table(p)
    for r, b in zip(rows, back):
        assert b["key"] == r["key"]
        for c in ("value", "ci_lo", "ci_hi", "p_raw", "p_holm"):
            assert float(b[c]) == r[c]                              # exact, as the verifier compares claims
        assert int(b["n"]) == r["n"] and b["run_ids"].split(";") == r["run_ids"]
    df = tb.read_table(p, as_dataframe=True)
    assert df.loc[0, "value"] == rows[0]["value"] and df.loc[0, "run_ids"] == ["run-a", "run-b"]


def test_tables_resolve_with_verifier(tmp_path):
    vb = _verifier()
    rows = _contrast_rows()
    tb.write_contrast_table(tmp_path / "tables" / "H1_main.csv", rows)
    key = rows[0]["key"]
    vals = vb.resolve_ref(f"tables/H1_main.csv#row={key}&cols=value,ci_lo,ci_hi", tmp_path)
    assert vals == [rows[0]["value"], rows[0]["ci_lo"], rows[0]["ci_hi"]]
    shown = f"{rows[0]['value']:.1f}"                                # what PAPER_INPUT.md would display
    assert vb.number_matches(shown, 1, False, vals[0])


def test_write_table_generic(tmp_path):
    df = pd.DataFrame({"em": [0.5, 0.25], "key": ["S11", "S2"], "n": [50, 50], "note": [None, "x"]})
    p = tb.write_table(tmp_path / "t.csv", df)
    rows = list(csv.reader(p.open(newline="")))
    assert rows[0] == ["key", "em", "n", "note"] and rows[1] == ["S11", "0.5", "50", ""]
    p2 = tb.write_table(tmp_path / "t2.csv", [{"key": "a", "x": 1.5}, {"key": "b", "y": 2}])
    assert p2.read_text().splitlines() == ["key,x,y", "a,1.5,", "b,,2"]
    with pytest.raises(ValueError):
        tb.write_table(tmp_path / "bad.csv", [{"x": 1}])               # no key column
    with pytest.raises(ValueError):
        tb.write_table(tmp_path / "dup.csv", [{"key": "a"}, {"key": "a"}])


# ------------------------------------------------------------------ figures
def _bundle(tmp_path):
    root = tmp_path / "bundle"
    (root / "code").mkdir(parents=True)
    (root / "code" / "plot_h3.py").write_text("# plotting script\n")
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    items = pd.DataFrame({"key": [f"i{i}" for i in range(200)], "predicted": p,
                          "em": (rng.uniform(size=200) < p).astype(float)})
    tb.write_table(root / "tables" / "H3_items.csv", items)
    return root, items


def test_save_figure_sidecar(tmp_path):
    root, items = _bundle(tmp_path)
    fig = fg.plot_observed_vs_predicted(items)
    side = fg.save_figure(fig, "h3_observed_vs_predicted", root / "figures", root / "code" / "plot_h3.py",
                          [root / "tables" / "H3_items.csv"])
    png, js = root / "figures" / "h3_observed_vs_predicted.png", root / "figures" / "h3_observed_vs_predicted.json"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    on_disk = json.loads(js.read_text())
    assert on_disk == side and set(side) == {"script", "inputs", "created_at"}
    assert side["script"] == "code/plot_h3.py"
    assert side["inputs"] == [{"path": "tables/H3_items.csv",
                               "sha256": hashlib.sha256((root / "tables" / "H3_items.csv").read_bytes()).hexdigest()}]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", side["created_at"])
    # relative paths are accepted too, and anything outside the bundle is refused
    fg.save_figure(fig, "again", root / "figures", "code/plot_h3.py", ["tables/H3_items.csv"])
    with pytest.raises(ValueError):
        fg.save_figure(fig, "bad", root / "figures", tmp_path / "elsewhere.py", ["tables/H3_items.csv"])
    with pytest.raises(FileNotFoundError):
        fg.save_figure(fig, "bad", root / "figures", "code/plot_h3.py", ["tables/missing.csv"])


def test_sidecar_passes_verifier_figure_checks(tmp_path):
    """Run the verifier's D8 loop on our output (other checks fail on this partial bundle and are ignored)."""
    root, items = _bundle(tmp_path)
    fg.save_figure(fg.plot_observed_vs_predicted(items), "h3", root / "figures", "code/plot_h3.py",
                   ["tables/H3_items.csv"])
    rep = _verifier().verify(root, None)
    assert not [f for f in rep.fails if f.startswith("[D8]")]


def _legend_texts(fig):
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg:
            out += [t.get_text() for t in leg.get_texts()]
        out += [t.get_text() for t in ax.get_xticklabels()] + [t.get_text() for t in ax.texts]
    return out


def test_all_plots_render_headless(tmp_path):
    systems = ["S2", "S8", "S9", "S11", "S12", "S18"]
    lengths = [32768, 131072, 524288]
    em_len = pd.DataFrame([{"system": s, "length_tokens": L, "em": 0.5 + 0.01 * i - 0.05 * j,
                            "ci_lo": 0.4, "ci_hi": 0.6} for i, s in enumerate(systems) for j, L in enumerate(lengths)])
    em_h = pd.DataFrame([{"system": s, "h_late": h, "em": 0.9 ** h, "predicted": 0.85 ** h}
                         for s in systems for h in range(4)])
    threads = pd.DataFrame([{"system": s, "n_threads": T, "em": 0.6, "thread_confusion": 0.1 * k}
                            for s in ["S11", "S12", "S14", "S15", "S16"] for k, T in enumerate([4, 8, 16])])
    routing = pd.DataFrame([{"method": m, "scores_per_query": sc, "recall": r, "ci_lo": r - 0.02, "ci_hi": r + 0.02}
                            for m, sc, r in [("descent b1", 40, 0.8), ("descent b2", 72, 0.9),
                                             ("descent b4", 136, 0.95), ("S17", 300, 0.96), ("flat", 2048, 0.97)]])
    rng = np.random.default_rng(0)
    faults = pd.DataFrame({"system": rng.choice(["S7", "S8", "S9"], 120), "necessary_faults": rng.integers(0, 4, 120),
                           "spurious_faults": rng.integers(0, 3, 120)})
    cost = pd.DataFrame({"system": systems, "cost_usd": [0.01, 0.05, 0.06, 0.08, 0.1, 0.12],
                         "em": [0.3, 0.6, 0.9, 0.7, 0.65, 0.95]})
    items = pd.DataFrame({"predicted": rng.uniform(size=300)})
    items["em"] = (rng.uniform(size=300) < items["predicted"]).astype(float)
    figs = {
        "len": fg.plot_em_vs_length(em_len), "hlate": fg.plot_em_vs_hlate(em_h),
        "obs": fg.plot_observed_vs_predicted(items), "threads": fg.plot_threads(threads),
        "routing": fg.plot_routing_recall(routing), "faults": fg.plot_faults(faults),
        "cost": fg.plot_cost_frontier(cost),
    }
    for name, fig in figs.items():
        out = tmp_path / f"{name}.png"
        fig.savefig(out)
        assert out.stat().st_size > 1000, name
    for name in ("len", "hlate", "faults", "cost"):
        texts = _legend_texts(figs[name])
        assert "S9 (oracle)" in texts, name
        if name != "faults":
            assert "S18 (oracle)" in texts, name
    assert "S11" in _legend_texts(figs["len"]) and "S11 (oracle)" not in _legend_texts(figs["len"])


def test_oracle_labels_and_pareto():
    assert fg.system_label("S9") == "S9 (oracle)" and fg.system_label("S18") == "S18 (oracle)"
    assert fg.system_label("S1") == "S1" and fg.system_label("S19") == "S19"
    assert fg.system_label("S9 oracle skeleton") == "S9 oracle skeleton"
    assert fg.is_oracle("S18/A2") and not fg.is_oracle("S8")
    assert fg.pareto_front([1, 2, 3, 2], [0.5, 0.7, 0.6, 0.4]) == [0, 1]
    assert fg._ordered(["S12", "S2", "S10", "S9"]) == ["S2", "S9", "S10", "S12"]
