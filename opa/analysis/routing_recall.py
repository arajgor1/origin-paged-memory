"""H8 routing recall: shell descent vs flat scan at M candidate slots (CPU only, precomputed vectors).

Key `descent_beam4_vs_flat_recall_at_M` (prereg-v2.DRAFT.md Section 6): per query, recall = 1 if every gold-fact
origin is among the M candidates the router returns, else 0. Descent is the beam descent over the shell tree
(branching b = 8, beam 4); the comparator is the flat scan over all origins. Both routines are the ones the agent
systems use (``opa.tracks.track_a.index.Session.descend`` / ``flat_scan``, same routing key score), run on the item's
precomputed vectors (the npz of ``opa.tracks.track_a.embed``), so no model and no GPU is needed. The session is
built exactly as the S12 run builds it: ``session_cfg("S12", opts)`` from opa.tracks.track_a.systems (derived, never
copied) and the shared S-llm skeletons (prompt_version must equal run.SKELETON_PROMPT_VERSION, status not "error",
as in run_system). The flat comparator is ``Session.flat_scan`` on the same session (no system maps to the
protocol's flat scan; S17 is the HNSW index). value = mean over
items of (descent recall - flat recall) x 100 points, 95% paired bootstrap (opa.stats.contrast, 10,000 resamples,
seed 0), exact McNemar p (reported, not used: H8 is a non-inferiority decision from the CI lower bound; write the
table with ``write_routing_table``, which blanks p_raw for the primary key). Beams 1 and 2 are reported as extra rows.

Items without gold-fact origins, or without precomputed vectors, cannot be scored and are excluded with a reason
(``excluded``, ``n_excluded``): there is no model run behind them, so this is not an intention-to-treat case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import stats
from .tables import write_contrast_table

PRIMARY_BEAM = 4
EXTRA_COLUMNS = ["n10", "n01", "split", "sd_diff", "M", "beam", "flat_recall", "descent_recall", "scores_flat",
                 "scores_descent", "n_excluded"]
SYSTEM = "S12"


def s12_cfg(opts: dict | None = None) -> dict:
    """The session configuration the A2 arm S12 uses (systems.session_cfg), derived at call time."""
    from ..tracks.track_a.systems import session_cfg
    return session_cfg(SYSTEM, dict(opts or {}))


def load_skeleton(item_id: str, skeletons) -> dict:
    """Entities dict of one item's S-llm skeleton from ``skeletons``: a {item_id: payload} mapping or a directory /
    URI holding <item_id>.json (as skeleton_llm_uri). Refuses a missing payload, a prompt_version other than the
    current one, or status "error" (the checks run_system makes)."""
    from ..tracks.track_a.run import SKELETON_PROMPT_VERSION, load_json
    if hasattr(skeletons, "get"):
        got = skeletons.get(item_id)
    else:
        got = load_json(f"{str(skeletons).rstrip('/')}/{item_id}.json")
    if not got:
        raise RuntimeError(f"no S-llm skeleton for {item_id}")
    if got.get("prompt_version") != SKELETON_PROMPT_VERSION:
        raise RuntimeError(f"skeleton for {item_id} has prompt_version {got.get('prompt_version')!r}, expected "
                           f"{SKELETON_PROMPT_VERSION!r}: rebuild it with build_skeletons")
    if got.get("status") == "error":
        raise RuntimeError(f"skeleton for {item_id} failed its quality check: {got.get('error')}")
    return got["entities"]


def key_for(beam: int) -> str:
    return f"descent_beam{beam}_vs_flat_recall_at_M"


def gold_origin_ids(item: dict) -> set[int]:
    """Origins of the gold facts (gold.fact_origins plus each fact's origin_id)."""
    g = item.get("gold", {})
    ids = {int(o) for o in g.get("fact_origins", [])}
    ids |= {int(f["origin_id"]) for f in g.get("facts", []) if f.get("origin_id") is not None}
    return ids


def _embedders(embeddings):
    """A PrecomputedEmbedder from bytes / a path / an instance; anything with embed() and embed_query() but no
    for_item() (a live or test embedder) is used as it is."""
    from ..tracks.track_a.embed import PrecomputedEmbedder
    if isinstance(embeddings, (bytes, bytearray, str, Path)):
        src = bytes(embeddings) if isinstance(embeddings, (bytes, bytearray)) else str(embeddings)
        return PrecomputedEmbedder(src)
    return embeddings


def route_item(item: dict, embedder, M: int, skeleton: dict, beams: Sequence[int] = (1, 2, 4),
               opts: dict | None = None) -> dict:
    """{"gold": sorted ids, "flat": {"cands", "n_scores", "recall"}, "beam<k>": {...}} for one item. ``embedder`` is
    already bound to the item (PrecomputedEmbedder.for_item) or a plain embedder."""
    from ..tracks.track_a.index import Session
    sess = Session(item, s12_cfg(opts), embedder, skeleton_llm=skeleton)
    q = str(item["question"])
    q_emb = embedder.embed_query([q])[0]
    gold = gold_origin_ids(item)
    flat, n_flat = sess.flat_scan(q, q_emb, M)
    out = {"gold": sorted(gold), "flat": {"cands": flat, "n_scores": n_flat, "recall": int(gold <= set(flat))}}
    for b in beams:
        cands, n = sess.descend(q, q_emb, beam=b, top=M)
        out[f"beam{b}"] = {"cands": cands, "n_scores": n, "recall": int(gold <= set(cands))}
    return out


def routing_recall_rows(items: Iterable[dict], embeddings, skeletons, *, M: int, split: str,
                        beams: Sequence[int] = (1, 2, 4), opts: dict | None = None, n_boot: int = 10_000, seed: int = 0,
                        run_ids: Sequence[str] | None = None) -> list[dict]:
    """``skeletons``: {item_id: skeleton payload} or a skeleton_llm directory / URI (see load_skeleton; a stale or
    failed skeleton raises). ``opts`` are the S12 run's opts (b, tau_thread, ... as session_cfg reads them).
    One contrast row per beam (descent beam k minus flat, recall in points), the beam-4 row first and primary.
    Each row also has sd_diff (SD, ddof 1, of the per-item differences in points), flat_recall / descent_recall
    (0-1 means), scores_flat / scores_descent (mean routing scores per query), M, beam, excluded, n_excluded and
    per_item (item_id -> recalls) for the figure scripts."""
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if PRIMARY_BEAM not in beams:
        raise ValueError(f"beams must include {PRIMARY_BEAM}")
    emb = _embedders(embeddings)
    per, excluded = {}, []
    for it in items:
        iid = it["item_id"]
        if not gold_origin_ids(it):
            excluded.append({"item_id": iid, "reason": "no gold-fact origins"})
            continue
        try:
            bound = emb.for_item(it) if hasattr(emb, "for_item") else emb
        except (KeyError, ValueError) as e:
            excluded.append({"item_id": iid, "reason": f"no usable embeddings: {e}"})
            continue
        per[iid] = route_item(it, bound, M, load_skeleton(iid, skeletons), beams, opts)
    if not per:
        raise ValueError("no item could be routed")
    ids = sorted(per)
    flat = np.array([per[i]["flat"]["recall"] for i in ids])
    rows = []
    for b in sorted(beams, key=lambda x: (x != PRIMARY_BEAM, x)):
        d = np.array([per[i][f"beam{b}"]["recall"] for i in ids])
        row = stats.contrast(d, flat, key_for(b), test="mcnemar", scale=100.0, n_boot=n_boot, seed=seed,
                             run_ids=list(run_ids or []), split=split)
        diff = (d - flat) * 100.0
        row.update(sd_diff=float(diff.std(ddof=1)) if len(ids) > 1 else float("nan"), M=M, beam=b,
                   flat_recall=float(flat.mean()), descent_recall=float(d.mean()),
                   scores_flat=float(np.mean([per[i]["flat"]["n_scores"] for i in ids])),
                   scores_descent=float(np.mean([per[i][f"beam{b}"]["n_scores"] for i in ids])),
                   excluded=excluded, n_excluded=len(excluded),
                   per_item={i: {"flat": int(flat[k]), "descent": int(d[k])} for k, i in enumerate(ids)})
        rows.append(row)
    return rows


def write_routing_table(path: Path | str, rows: list[dict]) -> Path:
    """Contrast table of the rows; the primary (beam 4) row is a non-inferiority key, so p_raw is blank and the
    test label says CI lower bound only (opa.analysis.tables.write_contrast_table, ci_only_keys)."""
    return write_contrast_table(path, rows, extra_columns=EXTRA_COLUMNS, ci_only_keys=[key_for(PRIMARY_BEAM)])


__all__ = ["PRIMARY_BEAM", "EXTRA_COLUMNS", "key_for", "gold_origin_ids", "route_item", "routing_recall_rows",
           "write_routing_table"]
