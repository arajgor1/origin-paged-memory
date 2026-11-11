"""Thread-confusion task for H12 (plan v3, Section 7): MT-LB-MHQA items with look-alike decoys.

It is MT-LB-MHQA (mtlbmhqa.make_item(..., decoys=True), cell suffix "-conf") in which every thread other than the
question's thread carries exactly one decoy of the final fact: same template, same component and attribute type, a
different value, and a near-identical entity surface form. A system that answers with a decoy's value took the
answer from the wrong thread. Decoy kinds (no arm sees thread labels, so an identical name would make the item
ambiguous and is not used):

  twin    the shared one-character twin of the final entity (hamming distance 1, e.g. "Kelvane" / "Kelvine").
  suffix  the same stem with another product-type suffix ("Kelvane Pumps" / "Kelvane Works"). Only when the final
          entity already carries a product-type suffix; never a legal form (Ltd., GmbH, Corp., Group, ...) on either
          side, because a different legal form is a legitimately different company.

Items whose final entity has no product-type suffix get twin decoys only. A decoy's entity is never an alias of the
final entity (STEM, Stem): validate_item enforces it.

Gold: the decoys are stored in gold["confusers"] (so opa.scoring.thread_confused / failure code F8 work unchanged),
each with entity, value, value_aliases, kind, hop (the hop it imitates), thread_id, origin_id, char_span and text.
Departures from the protocol text are logged in mtlbmhqa.py and SCHEMA.md.
"""
from __future__ import annotations

import numpy as np

from opa.data.names import Name

KINDS = ("twin", "suffix")
# product-type suffixes; legal forms (Ltd., GmbH, Corp., Group, Inc., AG, plc, LLC, S.A., ...) are deliberately absent
PRODUCT_SUFFIXES = (" Pumps", " Valves", " Works", " Systems", " Engineering", " Fluidics", " Metals", " Polymers",
                    " Hydraulics")


def suffix_allowed(final_full: str) -> bool:
    """True if the final entity's name ends in a product-type suffix, so that a suffix decoy is possible."""
    return any(final_full.endswith(s) and len(final_full) > len(s) and " " not in final_full[:-len(s)]
               for s in PRODUCT_SUFFIXES)


def allowed_kinds(final_full: str) -> tuple[str, ...]:
    return KINDS if suffix_allowed(final_full) else ("twin",)


def assign_kinds(rng: np.random.Generator, n: int, allowed: tuple[str, ...]) -> list[str]:
    """n decoy kinds: every allowed kind occurs when n >= len(allowed); order random."""
    base = [allowed[i % len(allowed)] for i in range(n)]
    return [base[int(i)] for i in rng.permutation(n)]


def suffix_variant(rng: np.random.Generator, final: Name) -> Name:
    opts = [s for s in PRODUCT_SUFFIXES if s != final.suffix]
    return Name(final.stem, opts[int(rng.integers(len(opts)))])


def decoy_name(kind: str, final: Name, twin: Name | None, rng: np.random.Generator) -> Name:
    if kind == "twin":
        assert twin is not None
        return twin
    if kind == "suffix":
        return suffix_variant(rng, final)
    raise ValueError(f"unknown decoy kind {kind!r}")


def final_aliases(final_full: str) -> set[str]:
    """Surface forms that mean the final entity itself: its stem, the upper-case stem and the full name."""
    stem = final_full.split(" ")[0]
    return {stem, stem.upper(), final_full}


def decoy_entity_ok(kind: str, decoy: str, final: str) -> bool:
    """The surface-form relation that a decoy of `kind` must have to the final entity."""
    from opa.data.names import hamming1
    if decoy in final_aliases(final):
        return False
    if kind == "twin":
        return hamming1(decoy, final)
    if kind == "suffix":
        return (suffix_allowed(final) and suffix_allowed(decoy) and decoy != final
                and decoy.split(" ")[0] == final.split(" ")[0])
    return False


def classify_prediction(item: dict, prediction) -> dict:
    """Where did a prediction's value come from? {"outcome": "correct" | "wrong_thread" | "other",
    "decoy_thread": thread id of the matching decoy or None}. Uses opa.scoring.exact_match (normalised EM)."""
    from opa.scoring import exact_match
    if exact_match(prediction, item["answer_aliases"]):
        return {"outcome": "correct", "decoy_thread": None}
    for c in item["gold"]["confusers"]:
        if exact_match(prediction, c.get("value_aliases") or [c["value"]]):
            return {"outcome": "wrong_thread", "decoy_thread": c["thread_id"]}
    return {"outcome": "other", "decoy_thread": None}
