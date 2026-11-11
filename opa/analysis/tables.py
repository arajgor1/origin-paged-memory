"""Deterministic CSV tables for bundle/tables/ (bundle_spec/BUNDLE_SPEC.md, "Tables").

Format rules (the same input always gives byte-identical output):

- UTF-8, header row, ``\\n`` line endings, RFC-4180 quoting only where needed (csv.QUOTE_MINIMAL).
- First column is ``key`` (the row id used by ``[[ref:tables/<file>.csv#row=<key>&cols=...]]`` and by claim
  ``table_ref``); keys must be unique and non-empty.
- Floats are written with the shortest representation that round-trips exactly
  (``numpy.format_float_positional(x, unique=True, trim="0")``, never scientific notation): ``float(cell) == x``
  bit for bit. The verifier compares a claim's ``value`` with the table row to 1e-9 relative and matches a number
  shown in PAPER_INPUT.md against the source rounded to the displayed decimals, so tables must not be rounded:
  rounding happens only where a number is displayed. NaN is written ``nan``, +-inf ``inf`` / ``-inf``.
- None is written as an empty cell; bools as ``true`` / ``false``; ints as integers; lists (e.g. run ids) are
  joined with ``;``.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

CONTRAST_COLUMNS = ["key", "value", "ci_lo", "ci_hi", "n", "p_raw", "p_holm", "test", "run_ids"]
LIST_SEP = ";"


def format_cell(v: Any) -> str:
    """Deterministic, lossless text for one CSV cell (module docstring)."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0:
            return "0.0"                                   # also maps -0.0 to 0.0
        return np.format_float_positional(x, unique=True, trim="0")
    if isinstance(v, (list, tuple, set, np.ndarray)):
        seq = sorted(v) if isinstance(v, set) else list(v)
        return LIST_SEP.join(format_cell(x) for x in seq)
    return str(v)


def _rows_from(df_or_rows: Any) -> tuple[list[dict], list[str]]:
    """Accept a pandas DataFrame or an iterable of dicts; return (rows, column order of first appearance)."""
    try:
        import pandas as pd
        if isinstance(df_or_rows, pd.DataFrame):
            cols = [str(c) for c in df_or_rows.columns]
            # missing values in non-float columns (None/NaN/NA in text or object columns) become empty cells;
            # NaN in a float column is a value and is written "nan"
            floaty = {str(c) for c in df_or_rows.columns if pd.api.types.is_float_dtype(df_or_rows[c].dtype)}
            rows = []
            for r in df_or_rows.to_dict(orient="records"):
                rows.append({str(k): (None if str(k) not in floaty and not isinstance(v, (list, tuple, np.ndarray))
                                      and pd.isna(v) else v) for k, v in r.items()})
            return rows, cols
    except ImportError:  # pragma: no cover
        pass
    rows = [dict(r) for r in df_or_rows]
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return rows, cols


def _write(path: Path | str, rows: list[dict], columns: Sequence[str]) -> Path:
    if not columns or columns[0] != "key":
        raise ValueError("the first column of a bundle table must be 'key'")
    seen = set()
    for r in rows:
        k = r.get("key")
        if k is None or str(k) == "":
            raise ValueError("every row needs a non-empty key")
        if str(k) in seen:
            raise ValueError(f"duplicate key {k!r}")
        seen.add(str(k))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(columns)
        for r in rows:
            w.writerow([format_cell(r.get(c)) for c in columns])
    return path


def write_table(path: Path | str, df_or_rows: Any, columns: Sequence[str] | None = None) -> Path:
    """Write a generic bundle table. ``df_or_rows`` is a DataFrame or a list of dicts and must have a ``key``
    column, which is moved to the front. ``columns`` fixes the column order (default: key, then the input order);
    missing values become empty cells. Returns the path."""
    rows, cols = _rows_from(df_or_rows)
    if columns is None:
        if "key" not in cols:
            raise ValueError("table needs a 'key' column")
        columns = ["key"] + [c for c in cols if c != "key"]
    return _write(path, rows, list(columns))


NI_TEST_LABEL = "paired bootstrap 95% CI (10,000 resamples); non-inferiority: CI lower bound only"


def write_contrast_table(path: Path | str, rows: Iterable[dict], extra_columns: Sequence[str] = (),
                         ci_only_keys: Iterable[str] = ()) -> Path:
    """Write contrast rows (e.g. from opa.stats.contrast) with columns
    key,value,ci_lo,ci_hi,n,p_raw,p_holm,test,run_ids (+ ``extra_columns``, e.g. n10, n01, split).
    A row may give its interval as ``ci95`` = [lo, hi] instead of ci_lo / ci_hi. Rows whose key is in
    ``ci_only_keys`` (non-inferiority primaries decided from the CI lower bound: opa.claims.non_inferiority_keys)
    get p_raw / p_holm blank and test = NI_TEST_LABEL, so the table agrees with a claim that reports p_raw null."""
    ci_only = set(ci_only_keys)
    out = []
    for r in rows:
        r = dict(r)
        if r.get("key") in ci_only:
            r["p_raw"], r["p_holm"], r["test"] = None, None, NI_TEST_LABEL
        if ("ci_lo" not in r or "ci_hi" not in r) and r.get("ci95") is not None:
            r["ci_lo"], r["ci_hi"] = r["ci95"][0], r["ci95"][1]
        out.append(r)
    return _write(path, out, CONTRAST_COLUMNS + [c for c in extra_columns if c not in CONTRAST_COLUMNS])


def read_table(path: Path | str, as_dataframe: bool = False, list_columns: Sequence[str] = ("run_ids",)):
    """Read a table written by this module. Returns a list of dicts of strings (as the verifier sees them), or with
    as_dataframe=True a DataFrame with numeric columns parsed and ``list_columns`` split back into lists."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not as_dataframe:
        return rows
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if c in list_columns:
            df[c] = [v.split(LIST_SEP) if v else [] for v in df[c]]
            continue
        conv = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
        if conv.notna().sum() == (df[c] != "").sum():
            df[c] = conv
    return df
