"""Figures of the interim Track A package (bundle/figures/*.pdf + *.png + sidecar *.json + source *.csv).

(a) fig_h1_depth_context   H1 contrasts on the primary cell. The funded test split has ONE H1 cell (LB-MHQA late h=2, 32K), so
                           there is no depth or context-length axis in test data; the dev pilot has the same single cell.
                           Test points are filled circles, dev pilot points hollow diamonds, and every legend entry says which.
(b) fig_h5_thread_depth    H5: agent-level A2 - A1 at the one funded thread depth (T4; test and dev pilot) and the model-free
                           distortion ratio at T = 4 / 8 / 16 (dev only: no test-split distortion run exists).
(c) fig_v3_learning_curve  V3 training curve on the dev monitoring set: own-origin mass fraction, attribution, passkey, loss.
                           Always dev; G2 (own-origin fraction >= 0.90) is judged later on fresh dev seeds.

In test mode the H1 and H5 source tables (and so the figure files) are named ``..._test_with_dev_pilot`` because they hold both
splits, told apart by their ``split`` column; in dev mode they are ``..._dev``.

Each figure is built from a table written to tables/ (first column ``key``), copied next to the figure, and saved with
opa.analysis.figures.save_figure (sidecar with the script and the SHA-256 of the input). The module itself is copied to
code/interim_figures.py inside the bundle because the sidecar names its script.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .figures import save_figure
from .tables import read_table, write_table

BLUE, ORANGE, GREY, VERM = "#0072B2", "#E69F00", "#6b6b6b", "#D55E00"
SPLIT_STYLE = {"test": dict(marker="o", color=BLUE, mfc=BLUE, label="test split"),
               "dev": dict(marker="D", color=ORANGE, mfc="none", label="dev pilot (not test)")}
H1_ORDER = [("S11_vs_S2_lbmhqa_late_32k_primary", "S11 - S2 (primary)", 5.0),
            ("S11_vs_S4_lbmhqa_late_32k_primary", "S11 - S4", 3.0),
            ("S11_vs_S8_late_bound", "S11 - S8 (H3 leg)", 2.0)]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def plot_h1(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    fig = Figure(figsize=(7.0, 3.6), layout="constrained")
    ax = fig.subplots()
    ys = {k: i for i, (k, _, _) in enumerate(reversed(H1_ORDER))}
    off = {"test": 0.12, "dev": -0.12}
    for split in ("test", "dev"):
        d = df[df["split"] == split]
        if d.empty:
            continue
        st = SPLIT_STYLE[split]
        first = True
        for _, r in d.iterrows():
            y = ys[r["contrast"]] + off[split]
            ax.errorbar([r["value"]], [y], xerr=[[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]], fmt=st["marker"],
                        color=st["color"], mfc=st["mfc"], capsize=3, ms=6,
                        label=f"{st['label']} (n = {int(r['n'])} on the primary pair)" if first else None)
            first = False
    ax.axvline(0, color=GREY, lw=1)
    for k, _, thr in H1_ORDER:
        ax.plot([thr, thr], [ys[k] - 0.3, ys[k] + 0.3], color=VERM, lw=1.5, ls="--")
    ax.plot([], [], color=VERM, ls="--", label="frozen rule threshold")
    ax.set_yticks(list(ys.values()), [lab for _, lab, _ in reversed(H1_ORDER)])
    ax.set_xlabel("EM difference (points), paired bootstrap 95% CI")
    ax.set_title("H1 on the one funded cell: LB-MHQA late h=2, 32K" + title_suffix, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_h5(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    fig = Figure(figsize=(8.0, 3.6), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    ag = df[df["panel"] == "agent"]
    for split in ("test", "dev"):
        d = ag[ag["split"] == split]
        if d.empty:
            continue
        st = SPLIT_STYLE[split]
        r = d.iloc[0]
        a1.errorbar([r["threads_T"] + (0.15 if split == "test" else -0.15)], [r["value"]],
                    yerr=[[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]], fmt=st["marker"], color=st["color"],
                    mfc=st["mfc"], capsize=3, ms=6, label=f"{st['label']} (n = {int(r['n'])})")
    a1.axhline(0, color=GREY, lw=1)
    a1.set_xticks([4], ["T4"])
    a1.set_xlim(3, 5)
    lo, hi = float(np.nanmin(ag["ci_lo"])) if len(ag) else -1.0, float(np.nanmax(ag["ci_hi"])) if len(ag) else 1.0
    a1.set_ylim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
    a1.set_ylabel("A2 - A1 agent EM (points), 95% CI")
    a1.set_title("agent level (only T4 funded)", fontsize=9)
    a1.legend(fontsize=8)
    dist = df[df["panel"] == "distortion"].sort_values("threads_T")
    if not dist.empty:
        a2.plot(dist["threads_T"], dist["value"], marker="D", mfc="none", color=ORANGE, label="dev (model-free, 32K h=3)")
        a2.axhline(1.0, color=GREY, lw=1, ls=":")
        a2.set_xscale("log", base=2)
        a2.set_xticks(dist["threads_T"].tolist(), [f"T{int(t)}" for t in dist["threads_T"]])
        a2.set_ylabel("Euclidean / hyperbolic tree distortion")
        a2.set_title("distortion ratio by thread depth (dev only)", fontsize=9)
        a2.legend(fontsize=8)
    fig.suptitle("H5 by thread depth" + title_suffix, fontsize=10)
    return fig


def plot_learning_curve(df: pd.DataFrame, g2_status: str | None = None) -> Figure:
    fig = Figure(figsize=(8.0, 5.0), layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    v3 = df[df["arm"] == "V3"].sort_values("opt_step")
    panels = [("own_origin_mass_fraction", "own-origin mass fraction", 0.90, "G2 threshold 0.90"),
              ("attrib_em", "source attribution EM", None, None),
              ("passkey_em", "passkey EM", None, None),
              ("train_loss", "training loss", None, None)]
    for ax, (col, lab, thr, thr_lab) in zip(axes, panels):
        for arm, d in df.groupby("arm"):
            d = d.sort_values("opt_step")
            ax.plot(d["opt_step"], d[col], marker="D", mfc="none", label=f"{arm} (dev monitoring)")
        if thr is not None:
            ax.axhline(thr, color=VERM, ls="--", lw=1.2, label=thr_lab)
        if col == "attrib_em" and "attrib_chance" in v3 and v3["attrib_chance"].notna().any():
            ax.axhline(float(v3["attrib_chance"].dropna().iloc[0]), color=GREY, ls=":", label="chance")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"V3 learning curve on the dev monitoring set (Track B; G2 {g2_status or 'pending'})", fontsize=10)
    return fig


def _save(fig: Figure, name: str, root: Path, script: Path, inputs: list[Path]) -> list[str]:
    fdir = root / "figures"
    save_figure(fig, name, fdir, script, inputs, bundle_root=root)
    pdf = fdir / f"{name}.pdf"
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    return [str(fdir / f"{name}.png"), str(pdf), str(fdir / f"{name}.json")]


def make_figures(out: Path | str, split: str, test_rows: Mapping[str, Mapping], dev_rows: Mapping[str, Mapping],
                 repo: Path | str | None = None) -> list[str]:
    """Write the three figures and their source tables. ``test_rows`` / ``dev_rows``: {key: row} of p2_contrasts_test.csv /
    p2_contrasts_dev.csv (test_rows empty in dev mode)."""
    from . import interim as itm
    out = Path(out).resolve()
    tdir, fdir, cdir = out / "tables", out / "figures", out / "code"
    cdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    script = cdir / "interim_figures.py"
    shutil.copyfile(Path(__file__), script)
    sfx = itm.suffix(split) if split == "dev" else "_test_with_dev_pilot"      # test-mode H1/H5 tables also hold dev pilot rows
    written: list[str] = []
    note = "  [DEV PILOT]" if split == "dev" else ""

    # ---- (a) H1
    dev_src = itm.DEV_SRC_TEST_MODE if split == "test" else "tables/p2_contrasts_dev.csv"
    rows = itm.h1_figure_rows(test_rows if split == "test" else {}, dev_rows, dev_src)
    if rows:
        name = f"fig_h1_depth_context{sfx}"
        itm.check_name(split, name + ".csv", dev_data=True)
        csv_path = write_table(tdir / f"{name}.csv", rows, columns=["key", "split", "cell", "depth_h", "context_tokens", "contrast",
                                                                    "value", "ci_lo", "ci_hi", "n", "em_a", "em_b", "systems", "source"])
        shutil.copyfile(csv_path, fdir / csv_path.name)
        df = pd.DataFrame(read_table(csv_path))
        for c in ("value", "ci_lo", "ci_hi", "n"):
            df[c] = df[c].map(_num)
        written += _save(plot_h1(df, note), name, out, script, [csv_path])
        written.append(str(fdir / csv_path.name))

    # ---- (b) H5
    dist_path = tdir / "h5_distortion_dev.csv"
    if not dist_path.exists() and repo is not None:
        dist_path = Path(repo) / "bundle" / "tables" / "h5_distortion_dev.csv"
    dist = {r["key"]: r for r in read_table(dist_path)} if dist_path.exists() else {}
    rows = itm.h5_figure_rows(test_rows if split == "test" else {}, dev_rows, dist, dev_src)
    if rows:
        name = f"fig_h5_thread_depth{sfx}"
        itm.check_name(split, name + ".csv", dev_data=True)
        csv_path = write_table(tdir / f"{name}.csv", rows, columns=["key", "panel", "split", "threads_T", "metric", "value", "ci_lo",
                                                                    "ci_hi", "n", "source"])
        shutil.copyfile(csv_path, fdir / csv_path.name)
        df = pd.DataFrame(read_table(csv_path))
        for c in ("value", "ci_lo", "ci_hi", "n", "threads_T"):
            df[c] = df[c].map(_num)
        written += _save(plot_h5(df, note), name, out, script, [csv_path])
        written.append(str(fdir / csv_path.name))

    # ---- (c) V3 learning curve (always dev data)
    lc_path = tdir / "v3_learning_curve_dev.csv"
    if lc_path.exists():
        name = "fig_v3_learning_curve_dev"
        shutil.copyfile(lc_path, fdir / f"{name}.csv")
        df = pd.DataFrame(read_table(lc_path))
        for c in df.columns:
            if c not in ("key", "arm", "split", "run_id"):
                df[c] = df[c].map(_num)
        written += _save(plot_learning_curve(df, itm.g2_state(itm.load_g2(out))), name, out, script, [lc_path])
        written.append(str(fdir / f"{name}.csv"))
    return written
