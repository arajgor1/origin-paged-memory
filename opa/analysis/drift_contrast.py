"""H2 primary contrast D4_vs_D2_retention_t50 (prereg-v2.DRAFT.md Section 6; protocol Section 7).

Drift sessions (10 constraints, 50 turns, 3 threads); unit = session; conditions D4 (extractive state) vs D2
(rolling summary). Per session, retention at turn 50 = share of the session's constraints whose CURRENT value the
probe answered correctly (opa.scoring.drift_probe_outcomes, outcome "correct"). Contrast row: value = mean over
paired sessions of (D4 - D2) x 100, 95% paired bootstrap over sessions (10,000 resamples, seed 0), p = two-sided
Wilcoxon signed-rank on the per-session differences with zero differences dropped (all via opa.stats.contrast,
test="wilcoxon", scale=100); n = number of paired sessions. The row also carries ``sd_diff``, the sample SD
(ddof = 1) of the per-session differences in the same units as ``value``, which the paired-normal sample-size
formula of protocol Section 7 needs.

Records are the rows of a drift run's records.jsonl.gz ({"item_id", "condition", "probes": [{"turn", "predicted"}]});
scoring a record needs its session (constraints and turns), passed as ``sessions``. A session id seen anywhere
(session list, D2 records, D4 records) but not paired is excluded and listed in ``excluded`` with a reason
("missing D2", "missing D4", "missing both", "not in session list"); ``n_excluded`` counts them. Duplicate
item_ids within one arm raise ValueError. A record with status != "ok" or without a probe at ``turn`` still scores
0 retention (the pinned rule, as in the run metrics) and is counted in ``n_error_d2`` / ``n_error_d4``; a probe
at ``turn`` with fits False is also scored 0 and counted in ``n_not_fit_d2`` / ``n_not_fit_d4``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import scoring, stats
from .tables import write_contrast_table

KEY = "D4_vs_D2_retention_t50"
EXTRA_COLUMNS = ["n10", "n01", "split", "sd_diff", "n_error_d2", "n_error_d4", "n_not_fit_d2", "n_not_fit_d4",
                 "n_excluded"]
SPLITS = ("dev", "test")


def read_records(path: Path | str) -> list[dict]:
    """Rows of a records.jsonl.gz (or plain .jsonl) file."""
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def session_retention(session: dict, record: dict, turn: int = 50) -> float:
    """Share of the session's constraints whose current value is answered correctly at ``turn``."""
    oc = scoring.drift_probe_outcomes(session, record, turn)
    return sum(v == "correct" for v in oc.values()) / len(oc) if oc else float("nan")


def sd_of_differences(a: Sequence[float], b: Sequence[float], scale: float = 1.0) -> float:
    """Sample SD (ddof = 1) of the paired differences (a - b) * scale; nan for fewer than two pairs."""
    d = (np.asarray(a, float) - np.asarray(b, float)) * scale
    return float(d.std(ddof=1)) if d.size > 1 else float("nan")


def _by_id(records: Iterable[dict], arm: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        if r["item_id"] in out:
            raise ValueError(f"duplicate record for session {r['item_id']} in arm {arm}")
        out[r["item_id"]] = r
    return out


def _not_fit(r: dict, turn: int) -> bool:
    """A probe at ``turn`` exists but its prompt did not fit the context (fits is False): scored 0, counted."""
    return any(p.get("turn") == turn and p.get("fits") is False for p in r.get("probes", []))


def _is_error(r: dict, turn: int) -> bool:
    return r.get("status") != "ok" or not any(p.get("turn") == turn for p in r.get("probes", []))


def drift_retention_contrast(sessions: Iterable[dict], d2_records: Iterable[dict], d4_records: Iterable[dict], *,
                             split: str, turn: int = 50, key: str = KEY, n_boot: int = 10_000, seed: int = 0,
                             run_ids: Sequence[str] | None = None) -> dict:
    """The D4 vs D2 retention contrast row (see module docstring) plus sd_diff, d2_mean / d4_mean (retention,
    0-1), ``turn`` and ``excluded`` (session ids missing from one arm or from ``sessions``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    by_s = _by_id(sessions, "sessions")
    r2, r4 = _by_id(d2_records, "D2"), _by_id(d4_records, "D4")
    ids = sorted(set(r2) & set(r4) & set(by_s))
    excluded = []
    for i in sorted((set(by_s) | set(r2) | set(r4)) - set(ids)):
        if i not in by_s:
            why = "not in session list"
        elif i not in r2 and i not in r4:
            why = "missing both"
        else:
            why = "missing D2" if i not in r2 else "missing D4"
        excluded.append({"item_id": i, "reason": why})
    if not ids:
        raise ValueError("no session is present in both arms (and in the session list)")
    a = [session_retention(by_s[i], r4[i], turn) for i in ids]      # D4
    b = [session_retention(by_s[i], r2[i], turn) for i in ids]      # D2
    out = stats.contrast(a, b, key, test="wilcoxon", scale=100.0, n_boot=n_boot, seed=seed,
                         run_ids=list(run_ids or []), split=split)
    out.update(sd_diff=sd_of_differences(a, b, 100.0), d2_mean=float(np.mean(b)), d4_mean=float(np.mean(a)),
               turn=turn, excluded=excluded, n_excluded=len(excluded), session_ids=ids,
               n_error_d2=sum(_is_error(r2[i], turn) for i in ids), n_error_d4=sum(_is_error(r4[i], turn) for i in ids),
               n_not_fit_d2=sum(_not_fit(r2[i], turn) for i in ids), n_not_fit_d4=sum(_not_fit(r4[i], turn) for i in ids))
    return out


def write_drift_contrast(path: Path | str, row: dict) -> Path:
    """Write the contrast row to a contrast table (key,value,ci_lo,ci_hi,n,p_raw,p_holm,test,run_ids,n10,n01,split,
    sd_diff,n_error_d2,n_error_d4,n_not_fit_d2,n_not_fit_d4,n_excluded) so claims.result_from_table can read it."""
    return write_contrast_table(path, [row], extra_columns=EXTRA_COLUMNS)


__all__ = ["KEY", "read_records", "session_retention", "sd_of_differences", "drift_retention_contrast",
           "write_drift_contrast"]
