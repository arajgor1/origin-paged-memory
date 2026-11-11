"""Invented, pronounceable, company-like names; alias variants; one-character twins.

Names are built from syllables (onset + vowel + coda) and an optional company suffix ("Varrel Pumps",
"Ostrand Ltd.", "Kelvane"). A candidate stem is rejected if it is a word of the filler vocabulary, occurs as a
substring of the item's filler text, or is within edit distance < 3 of a stem already used in the item, so the
only near-duplicates in an item are the deliberate one-character twins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# stem = onset + vowel + junction + vowel [+ junction + vowel] + coda, e.g. V-a-rr-e-l, O-str-a-nd, K-e-lv-a-ne
ONSETS = ["", "", "b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st",
          "t", "tr", "v", "z", "bl", "cl", "fl", "sk", "sl", "th", "w", "c"]
VOWELS = ["a", "e", "i", "o", "u"]
JUNCTIONS = ["r", "l", "n", "m", "s", "t", "d", "v", "k", "p", "b", "g", "rr", "ll", "rv", "rl", "lv", "nd", "nt",
             "rd", "rt", "st", "sk", "lk", "rk", "rn", "lm", "ld", "mb", "ng", "str", "lt", "rb", "rg", "lb", "sp",
             "nv", "rm", "lf", "rth"]
CODAS = ["n", "r", "l", "s", "x", "m", "nd", "rn", "st", "ll", "rr", "ne", "ve", "re", "le", "se", "ck", "t", "d",
         "k", "", "", "nt", "rd", "rk", "lt", "sh", "th", "dge", "mble"]

COMPANY_SUFFIXES = [" Ltd.", " Pumps", " Valves", " Works", " Industries", " Systems", " Engineering", " Group",
                    " GmbH", " Corp.", " Fluidics", " Metals", " Polymers", " Hydraulics"]
SUPPLIER_SUFFIXES = [" Supply", " Trading", " Components", " Industrial", " Ltd.", " GmbH", " & Sons", " Parts"]

_BAD = re.compile(r"[aeiou]{2,}|[^aeiou]{4,}|(.)\1\1|^c[eiy]|q|y|^(.)\2")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hamming1(a: str, b: str) -> bool:
    """True if a and b have equal length and differ in exactly one character."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


@dataclass(frozen=True)
class Name:
    stem: str
    suffix: str = ""

    @property
    def full(self) -> str:
        return self.stem + self.suffix

    def alias_options(self) -> list[str]:
        """Alias variants of the canonical name ("Ostrand Ltd." -> "OSTRAND", "Ostrand")."""
        opts = [self.stem.upper()]
        if self.suffix:
            opts.append(self.stem)
        return opts


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


class NameGen:
    """Per-item name generator. vocab: set of lower-case words of the filler; text_lower: the item's filler text."""

    def __init__(self, rng: np.random.Generator, vocab: set[str] | frozenset[str] | None = None,
                 text_lower: str | None = None, min_dist: int = 3):
        self.rng = rng
        self.vocab = vocab or frozenset()
        self.text_lower = text_lower or ""
        self.min_dist = min_dist
        self.used: list[str] = []          # lower-case stems in use (including twins)
        self.n_rejected = 0

    def forbidden(self, stem: str) -> bool:
        s = stem.lower()
        return s in self.vocab or (s in self.text_lower)

    def _clear_of_used(self, s: str, ignore: str | None = None) -> bool:
        return all(levenshtein(s, u) >= self.min_dist for u in self.used if u != ignore)

    def _candidate(self) -> str:
        rng = self.rng
        s = _choice(rng, ONSETS) + _choice(rng, VOWELS) + _choice(rng, JUNCTIONS) + _choice(rng, VOWELS)
        if rng.random() < 0.2:
            s += _choice(rng, JUNCTIONS) + _choice(rng, VOWELS)
        return s + _choice(rng, CODAS)

    def stem(self) -> str:
        for _ in range(100000):
            s = self._candidate()
            if not 5 <= len(s) <= 9 or _BAD.search(s):
                continue
            if self.forbidden(s) or not self._clear_of_used(s):
                self.n_rejected += 1
                continue
            self.used.append(s)
            return s.capitalize()
        raise RuntimeError("could not find a free name")

    def entity(self, suffixes=COMPANY_SUFFIXES, p_suffix: float = 0.6) -> Name:
        st = self.stem()
        suf = _choice(self.rng, suffixes) if self.rng.random() < p_suffix else ""
        return Name(st, suf)

    def twin(self, name: Name) -> Name:
        """A name that differs from `name` in exactly one character: one vowel of the stem (not the first letter)
        replaced by another vowel, so the twin stays pronounceable. Same suffix."""
        st = name.stem
        pos = [i for i in range(1, len(st)) if st[i] in VOWELS]
        order = [pos[i] for i in self.rng.permutation(len(pos))]
        for i in order:
            for v in [VOWELS[k] for k in self.rng.permutation(len(VOWELS))]:
                if v == st[i]:
                    continue
                cand = st[:i] + v + st[i + 1:]
                c = cand.lower()
                if _BAD.search(c) or self.forbidden(cand) or not self._clear_of_used(c, ignore=st.lower()):
                    self.n_rejected += 1
                    continue
                self.used.append(c)
                out = Name(cand, name.suffix)
                assert hamming1(out.full, name.full)
                return out
        raise RuntimeError(f"no twin for {name.full}")


def part_number(rng: np.random.Generator, used: set[str] | None = None) -> str:
    """Invented part number like 'QX-2291' (letters without I and O)."""
    letters = "ABCDEFGHJKLMNPRSTUVWXZ"
    used = used if used is not None else set()
    for _ in range(10000):
        p = f"{letters[int(rng.integers(len(letters)))]}{letters[int(rng.integers(len(letters)))]}-{int(rng.integers(1000, 10000))}"
        if p not in used:
            used.add(p)
            return p
    raise RuntimeError("part numbers exhausted")
