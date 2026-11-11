"""Figures for the proof bundle (protocol 3.6 plots 1-7) and their sidecars (bundle_spec/BUNDLE_SPEC.md, "Figures").

Every plot function takes a pandas DataFrame (long format, one row per point) and returns a
``matplotlib.figure.Figure`` built with the object API (no pyplot, no global backend switch), so it renders
headless. ``save_figure`` writes ``<out_dir>/<name>.png`` and the sidecar ``<out_dir>/<name>.json``::

    {"script": "code/plot_h3.py", "inputs": [{"path": "tables/H3_items.csv", "sha256": "..."}],
     "created_at": "2027-01-10T12:00:00Z"}

with paths relative to the bundle root (default: the parent of ``out_dir``, i.e. ``bundle/`` for
``bundle/figures``). Oracle systems (S9, S18, or any system whose name contains "oracle") are labelled
"(oracle)" in every legend and drawn dashed / hollow.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from matplotlib.figure import Figure

ORACLE_SYSTEMS = {"S9", "S18"}


# ------------------------------------------------------------------ labelling
def is_oracle(system) -> bool:
    s = str(system).strip()
    head = re.split(r"[\s/(]", s, maxsplit=1)[0] if s else ""
    return head in ORACLE_SYSTEMS or "oracle" in s.lower()


def system_label(system) -> str:
    """'S9' -> 'S9 (oracle)'; other names unchanged (never doubles the tag)."""
    s = str(system)
    return f"{s} (oracle)" if is_oracle(s) and "oracle" not in s.lower() else s


def _style(system) -> dict:
    return {"linestyle": "--", "markerfacecolor": "none"} if is_oracle(system) else {"linestyle": "-"}


def _ordered(values) -> list:
    """Stable, natural ordering of systems: S2 before S10."""
    def k(v):
        s = str(v)
        m = re.match(r"([A-Za-z]*)(\d+)", s)
        return (m.group(1), int(m.group(2)), s) if m else (s, -1, s)
    return sorted(set(values), key=k)


# ------------------------------------------------------------------ sidecar
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path | str, root: Path) -> str:
    p = Path(p)
    full = (p if p.is_absolute() else root / p).resolve()
    try:
        return full.relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"{p} is not inside the bundle root {root}") from None


def save_figure(fig: Figure, name: str, out_dir: Path | str, script_path: Path | str, inputs: Iterable[Path | str],
                bundle_root: Path | str | None = None, dpi: int = 150, created_at: str | None = None) -> dict:
    """Write ``<out_dir>/<name>.png`` and its sidecar ``<out_dir>/<name>.json``; return the sidecar dict.

    ``script_path`` and ``inputs`` are absolute paths or paths relative to the bundle root, and must exist inside
    it (the verifier re-hashes them). Input hashes are computed from the files as they are now, so write the input
    tables before calling this and do not change them afterwards. ``created_at`` defaults to now (UTC, ISO 8601
    with a ``Z`` suffix)."""
    out_dir = Path(out_dir)
    root = Path(bundle_root) if bundle_root is not None else out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    script_rel = _rel(script_path, root)
    if not (root / script_rel).is_file():
        raise FileNotFoundError(f"script {script_rel} not found under {root}")
    ins = []
    for p in inputs:
        rel = _rel(p, root)
        f = root / rel
        if not f.is_file():
            raise FileNotFoundError(f"input {rel} not found under {root}")
        ins.append({"path": rel, "sha256": _sha256(f)})
    if not ins:
        raise ValueError("a figure needs at least one input file")
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=dpi, metadata={"Software": None})
    side = {"script": script_rel, "inputs": ins,
            "created_at": created_at or dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            .strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out_dir / f"{name}.json").write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return side


# ------------------------------------------------------------------ helpers
def _need(df, cols: Sequence[str], fn: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fn}: DataFrame lacks columns {missing}")


def _lines(ax, df, x: str, y: str, group: str, yerr: bool = True):
    for g in _ordered(df[group]):
        d = df[df[group] == g].sort_values(x)
        st = _style(g)
        ax.plot(d[x], d[y], marker="o", label=system_label(g), **st)
        if yerr and {"ci_lo", "ci_hi"} <= set(d.columns) and d["ci_lo"].notna().any():
            ax.fill_between(d[x], d["ci_lo"], d["ci_hi"], alpha=0.12)


def _fig(ncols: int = 1, width: float = 6.0, height: float = 4.0):
    fig = Figure(figsize=(width * ncols, height), layout="constrained")
    axes = fig.subplots(1, ncols)
    return fig, (np.atleast_1d(axes).tolist())


# ------------------------------------------------------------------ protocol 3.6 plots
def plot_em_vs_length(df, x: str = "length_tokens", y: str = "em", group: str = "system", title: str | None = None):
    """Plot 1: EM vs context length, one line per system. Columns: system, length_tokens, em (+ ci_lo, ci_hi)."""
    _need(df, [x, y, group], "plot_em_vs_length")
    fig, (ax,) = _fig()
    _lines(ax, df, x, y, group)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("EM")
    ax.set_title(title or "EM vs context length")
    ax.legend(fontsize=8)
    return fig


def plot_em_vs_hlate(df, x: str = "h_late", y: str = "em", group: str = "system", predicted: str | None = "predicted",
                     title: str | None = None):
    """Plot 2: EM vs h_late per system; if a ``predicted`` column is present (gamma_eff ** h_late per row) it is
    drawn dotted in the same colour."""
    _need(df, [x, y, group], "plot_em_vs_hlate")
    fig, (ax,) = _fig()
    _lines(ax, df, x, y, group)
    if predicted and predicted in df.columns:
        for g in _ordered(df[group]):
            d = df[df[group] == g].sort_values(x)
            ax.plot(d[x], d[predicted], linestyle=":", color="grey", alpha=0.7)
        ax.plot([], [], linestyle=":", color="grey", label="predicted gamma_eff^h_late")
    ax.set_xlabel("h_late (late-bound hops)")
    ax.set_ylabel("EM")
    ax.set_title(title or "EM vs h_late")
    ax.legend(fontsize=8)
    return fig


def plot_observed_vs_predicted(df, predicted: str = "predicted", observed: str = "em", bins: int = 10,
                               method: str = "quantile", title: str | None = None):
    """Plot 3: binned observed success vs predicted gamma_eff^h_late (per-item rows), identity line and R^2 from
    opa.stats.r2_binned (identity-line R^2; the fitted-line R^2 is shown in brackets)."""
    from opa.stats import r2_binned
    _need(df, [predicted, observed], "plot_observed_vs_predicted")
    res = r2_binned(df[predicted].to_numpy(float), df[observed].to_numpy(float), bins=bins, method=method)
    fig, (ax,) = _fig(width=5.0, height=5.0)
    bp = [b["mean_predicted"] for b in res["bins"]]
    bo = [b["mean_observed"] for b in res["bins"]]
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="identity")
    ax.scatter(bp, bo, s=[12 + 2 * b["n"] ** 0.5 for b in res["bins"]], zorder=3, label=f"{res['n_bins']} {method} bins")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("predicted success gamma_eff^h_late")
    ax.set_ylabel("observed success (EM)")
    ax.set_title(title or f"Observed vs predicted (R^2 = {res['r2']:.3f} [linear {res['r2_linear']:.3f}], n = {res['n']})")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_threads(df, x: str = "n_threads", em: str = "em", confusion: str = "thread_confusion",
                 group: str = "system", title: str | None = None):
    """Plot 4: MT-LB-MHQA EM (left) and thread-confusion rate (right) vs number of threads T, one line per system."""
    _need(df, [x, em, confusion, group], "plot_threads")
    fig, (a1, a2) = _fig(ncols=2, width=5.0)
    for ax, y, lab in ((a1, em, "EM"), (a2, confusion, "thread-confusion rate")):
        for g in _ordered(df[group]):
            d = df[df[group] == g].sort_values(x)
            ax.plot(d[x], d[y], marker="o", label=system_label(g), **_style(g))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("threads T")
        ax.set_ylabel(lab)
    a1.legend(fontsize=8)
    fig.suptitle(title or "MT-LB-MHQA: EM and thread confusion vs T")
    return fig


def plot_routing_recall(df, x: str = "scores_per_query", y: str = "recall", group: str = "method",
                        title: str | None = None):
    """Plot 5: routing recall vs scores per query for descent (beams 1, 2, 4), S17 and a flat scan. One point (or
    line, if several archive sizes) per method; optional ci_lo / ci_hi drawn as error bars."""
    _need(df, [x, y, group], "plot_routing_recall")
    fig, (ax,) = _fig()
    for g in _ordered(df[group]):
        d = df[df[group] == g].sort_values(x)
        kw = _style(g)
        if {"ci_lo", "ci_hi"} <= set(d.columns):
            ax.errorbar(d[x], d[y], yerr=[d[y] - d["ci_lo"], d["ci_hi"] - d[y]], marker="o", capsize=3,
                        label=system_label(g), **kw)
        else:
            ax.plot(d[x], d[y], marker="o", label=system_label(g), **kw)
    ax.set_xscale("log")
    ax.set_xlabel("scores per query")
    ax.set_ylabel("routing recall")
    ax.set_title(title or "Routing recall vs scores per query")
    ax.legend(fontsize=8)
    return fig


def plot_faults(df, group: str = "system", necessary: str = "necessary_faults", spurious: str = "spurious_faults",
                title: str | None = None):
    """Plot 6: per-item fault histogram (left, total faults per item, one step histogram per group) and the
    spurious-fault rate per group (right; group by skeleton type or edge type by passing ``group``). Per-item rows
    with columns ``group``, necessary_faults, spurious_faults."""
    _need(df, [group, necessary, spurious], "plot_faults")
    fig, (a1, a2) = _fig(ncols=2, width=5.0)
    groups = _ordered(df[group])
    tot = (df[necessary] + df[spurious]).to_numpy(int)
    edges = np.arange(-0.5, max(1, int(tot.max()) if len(tot) else 1) + 1.5, 1.0)
    rates = []
    for g in groups:
        d = df[df[group] == g]
        t = (d[necessary] + d[spurious]).to_numpy(int)
        a1.hist(t, bins=edges, histtype="step", linewidth=1.5, label=system_label(g),
                linestyle="--" if is_oracle(g) else "-")
        s, n = float(d[spurious].sum()), float((d[necessary] + d[spurious]).sum())
        rates.append(s / n if n else np.nan)
    a1.set_xlabel("faults per item")
    a1.set_ylabel("items")
    a1.legend(fontsize=8)
    xs = np.arange(len(groups))
    bars = a2.bar(xs, rates)
    for b, g in zip(bars, groups):
        if is_oracle(g):
            b.set_hatch("//")
            b.set_facecolor("none")
    a2.set_xticks(xs, [system_label(g) for g in groups], rotation=45, ha="right", fontsize=8)
    a2.set_ylabel("spurious faults / all faults")
    fig.suptitle(title or "Faults per item and spurious-fault rate")
    return fig


def pareto_front(cost: Sequence[float], em: Sequence[float]) -> list[int]:
    """Indices of Pareto-optimal points (lower cost, higher EM), sorted by cost. A point is dominated if another
    point has cost <= and EM >= with at least one strict."""
    c, e = np.asarray(cost, float), np.asarray(em, float)
    out = []
    for i in range(len(c)):
        dom = (c <= c[i]) & (e >= e[i]) & ((c < c[i]) | (e > e[i]))
        if not dom.any():
            out.append(i)
    return sorted(out, key=lambda i: (c[i], -e[i]))


def plot_cost_frontier(df, x: str = "cost_usd", y: str = "em", group: str = "system", title: str | None = None):
    """Plot 7: cost vs EM per system with the Pareto frontier (non-oracle systems only; oracle points are shown
    hollow and labelled but cannot be on the frontier)."""
    _need(df, [x, y, group], "plot_cost_frontier")
    fig, (ax,) = _fig()
    d = df.reset_index(drop=True)
    for i, r in d.iterrows():
        orc = is_oracle(r[group])
        ax.scatter([r[x]], [r[y]], facecolors="none" if orc else None, edgecolors="C0", zorder=3)
        ax.annotate(system_label(r[group]), (r[x], r[y]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    real = d[[not is_oracle(s) for s in d[group]]].reset_index(drop=True)
    if len(real):
        front = pareto_front(real[x], real[y])
        ax.plot(real.loc[front, x], real.loc[front, y], color="C1", label="Pareto frontier (non-oracle)")
        ax.legend(fontsize=8)
    ax.set_xlabel("cost per item (USD)")
    ax.set_ylabel("EM")
    ax.set_title(title or "Cost vs EM")
    return fig


__all__ = ["ORACLE_SYSTEMS", "is_oracle", "system_label", "save_figure", "pareto_front", "plot_em_vs_length",
           "plot_em_vs_hlate", "plot_observed_vs_predicted", "plot_threads", "plot_routing_recall", "plot_faults",
           "plot_cost_frontier"]
