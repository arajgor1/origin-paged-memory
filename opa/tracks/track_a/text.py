"""Text utilities for the Track A harness: entity keys, the lexical skeleton (S-lex), BM25, exact-pin matching."""
from __future__ import annotations

import math
import re
from collections import Counter

_PUNCT = re.compile(r"[^\w\s/.\-]", re.UNICODE)
_WS = re.compile(r"\s+")
CORP_SUFFIX = {"ltd", "inc", "co", "corp", "gmbh", "plc", "llc", "sa", "ag", "bv", "group", "industries", "works",
               "systems", "supply", "pumps", "valves", "seals", "labs", "holdings"}
SENT_START_STOP = set("""a an the he she it they we i you his her their its our my your this that these those there here
then when while if but and or so yet for nor as at by in on of to from with without after before during since until
upon into onto over under again once all any each every no not one two three some such what which who whom whose why
how where mr mrs miss dr sir lord lady st chapter book part volume yes oh ah well now perhaps indeed""".split())

# S-lex patterns (protocol 3.1 item 3): identifiers, part numbers, numbers with units, dates, file paths, capitalized names
RX_IDENT = re.compile(r"\b(?=[A-Za-z0-9\-]*\d)(?=[A-Za-z0-9\-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b|\b[A-Z]{2,}\d+[A-Z0-9]*\b")
RX_NUM_UNIT = re.compile(r"[±+\-]?\d+(?:[.,]\d+)?\s?(?:°\s?[CF]|deg\s?[CF]|degrees?\s?[CF]?|mm|cm|m|kg|g|bar|psi|kPa|MPa|V|A|W|kW|Hz|%|USD|EUR)\b")
RX_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{4}\b|"
                     r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{1,2},?\s\d{4}\b")
RX_PATH = re.compile(r"(?:\b[\w.\-]+/)+[\w.\-]+\.\w+\b")
RX_CAPNAME = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+(?:[A-Z][A-Za-z'\-]+\.?|&|of|de|van|von))*(?:\s+(?:Ltd|Inc|Co|Corp|GmbH|PLC|LLC)\.?)?")
RX_ALLCAPS = re.compile(r"\b[A-Z]{3,}\b")
RX_WORD = re.compile(r"\w+", re.UNICODE)


QUESTION_STOP = set("""above below give do does did is are was were how what which who whom whose where when why can
could should would will name list tell find state report according""".split())


def normalize_key(s: str) -> str:
    s = s.replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = s.replace(".", " ")
    return _WS.sub(" ", s).strip()


def alias_keys(key: str) -> set[str]:
    """Keys under which an entity may also be looked up: its first word (OSTRAND -> ostrand for "Ostrand Ltd."),
    and the name without corporate suffixes."""
    words = key.split()
    out = set()
    if len(words) > 1:
        core = [w for w in words if w not in CORP_SUFFIX]
        if core and core != words:
            out.add(" ".join(core))
        if len(words[0]) >= 4:
            out.add(words[0])
    return out - {key}


def lexical_entities(text: str) -> set[str]:
    """S-lex: deterministic pass over identifiers, part numbers, numbers with units, dates, file paths and capitalized
    names. Returns normalized keys."""
    keys: set[str] = set()
    for rx in (RX_IDENT, RX_NUM_UNIT, RX_DATE, RX_PATH):
        keys.update(normalize_key(m.group(0)) for m in rx.finditer(text))
    for m in RX_CAPNAME.finditer(text):
        name = m.group(0).strip()
        words = name.split()
        while words and normalize_key(words[0]) in SENT_START_STOP:
            words = words[1:]
        if not words:
            continue
        k = normalize_key(" ".join(words))
        if k and k not in SENT_START_STOP and len(k) > 2:
            keys.add(k)
    keys.update(normalize_key(m.group(0)) for m in RX_ALLCAPS.finditer(text))
    return {k for k in keys if k}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in RX_WORD.findall(text)]


class BM25:
    """Okapi BM25 over a fixed list of documents (k1 = 1.5, b = 0.75)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: str, idx: int) -> float:
        tf, dl, s = self.tf[idx], self.len[idx], 0.0
        for w in tokenize(query):
            f = tf.get(w)
            if f:
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def scores(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(len(self.tf))]


def find_exact(quote: str, text: str) -> tuple[int, int] | None:
    """Exact-pin rule: the quote must be an exact substring of the origin text. Whitespace runs are treated as one
    space on both sides (PG-19 line breaks); nothing else is normalized. Returns the char span in `text` or None."""
    q = quote.strip()
    if not q:
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q)
    parts = [re.escape(p) for p in q.split()]
    if not parts:
        return None
    m = re.search(r"\s+".join(parts), text)
    return (m.start(), m.end()) if m else None


def mentions(text: str, key: str) -> bool:
    """True if the normalized text contains the normalized entity key as whole words."""
    if not key:
        return False
    return re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", normalize_key(text)) is not None
