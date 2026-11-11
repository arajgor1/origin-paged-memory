"""Filler text and tokenizers.

PG19Filler   emozilla/pg19 on Hugging Face. Dev items use the VALIDATION split only (50 books); the test split's
             books are reserved for test items after the prereg-v2 tag. Only the one parquet file of the split is
             downloaded (validation: 10.8 MB), cached under <local_data>/_cache/hf.
SyntheticFiller  deterministic pseudo-English books for unit tests (no network).

Blocks: about `block_tokens` (512) tokens of contiguous book text, cut at a sentence boundary when one lies within
48 tokens of the target, else at a word boundary. Book and offset are drawn from the item's generator, which is
keyed on (dataset, cell, seed). The first 1% (at least 1500 characters) and last 1% of each book are skipped
(Project Gutenberg front and back matter).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from opa import PROJECT_DIR

PG19_REPO = "emozilla/pg19"
_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z])|\n\n")
_WS_TOKEN = re.compile(r"\S+")


# ---------------------------------------------------------------- config
@lru_cache(maxsize=1)
def project_cfg() -> dict:
    return yaml.safe_load((PROJECT_DIR / "config" / "project.yaml").read_text())


def local_data_dir() -> Path:
    return (PROJECT_DIR / project_cfg()["paths"]["local_data"]).resolve()


def cache_dir() -> Path:
    d = local_data_dir() / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- tokenizers
class HFTokenizer:
    """Adapter around a Hugging Face fast tokenizer: offsets() and count()."""

    def __init__(self, name: str, hf_tokenizer=None):
        self.name = name
        if hf_tokenizer is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=str(cache_dir() / "hf"))
        self.tok = hf_tokenizer
        self.tok.model_max_length = 10 ** 9      # counting only; silences the long-sequence warning

    def offsets(self, text: str) -> np.ndarray:
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        return [len(x) for x in self.tok(list(texts), add_special_tokens=False)["input_ids"]]


class WhitespaceTokenizer:
    """Test stub: one token per whitespace-separated word."""
    name = "whitespace-stub"

    def offsets(self, text: str) -> np.ndarray:
        return np.asarray([m.span() for m in _WS_TOKEN.finditer(text)], dtype=np.int64).reshape(-1, 2)

    def count(self, text: str) -> int:
        return len(_WS_TOKEN.findall(text))

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]


def as_tokenizer(tok):
    """Accept our adapters or a raw Hugging Face tokenizer."""
    if hasattr(tok, "offsets") and hasattr(tok, "count_many"):
        return tok
    return HFTokenizer(getattr(tok, "name_or_path", "hf-tokenizer"), hf_tokenizer=tok)


@lru_cache(maxsize=2)
def load_tokenizer(name: str | None = None) -> HFTokenizer:
    return HFTokenizer(name or project_cfg()["paths"]["tokenizer"])


# ---------------------------------------------------------------- text helpers
def normalize_text(raw: str) -> str:
    """Join hard-wrapped lines inside paragraphs; paragraphs separated by one blank line."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paras = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw))
    return "\n\n".join(p for p in paras if p)


_ABBR = {"mr", "mrs", "ms", "dr", "st", "mt", "messrs", "jr", "sr", "prof", "capt", "col", "gen", "rev", "no", "vol",
         "lt", "sgt", "hon", "esq", "co", "ch", "chap", "fig", "pp", "viz", "vs", "etc", "ibid", "cf"}
_PREV_WORD = re.compile(r"(\w+)\.[\"')\]]*\s*$")


def sentence_starts(text: str) -> list[int]:
    """Character positions where a sentence (or paragraph) starts, excluding 0 and len(text).
    A full stop after a common abbreviation ("Dr.", "St.") or a single capital initial is not a boundary."""
    out = []
    for m in _SENT.finditer(text):
        e = m.end()
        if not 0 < e < len(text):
            continue
        if m.group(0) != "\n\n":
            w = _PREV_WORD.search(text, max(0, m.start() - 12), m.end())
            if w and (w.group(1).lower() in _ABBR or (len(w.group(1)) == 1 and w.group(1).isupper())):
                continue
        out.append(e)
    return out


def insert_positions(text: str, k: int, rng: np.random.Generator) -> list[int]:
    """k insertion points at sentence starts (with replacement); falls back to the start/end of the block."""
    cands = sentence_starts(text) or [0, len(text)]
    return [int(cands[int(rng.integers(len(cands)))]) for _ in range(k)]


def insert_sentences(text: str, inserts: list[tuple[int, str]]) -> tuple[str, list[list[int]]]:
    """Insert sentences at character positions of `text` (sentence starts, 0 or len(text)).
    Returns the new text and, for each insert in input order, its [start, end) span in the new text."""
    order = sorted(range(len(inserts)), key=lambda k: (inserts[k][0], k))
    pieces: list[str] = []
    spans: list[list[int] | None] = [None] * len(inserts)
    cur, prev = 0, 0
    for k in order:
        pos, s = inserts[k]
        if not 0 <= pos <= len(text):
            raise ValueError(f"insert position {pos} outside text of length {len(text)}")
        seg = text[prev:pos]
        pieces.append(seg)
        cur += len(seg)
        prev = pos
        if pos == len(text) and len(text) > 0:
            pieces.append(" " + s)
            spans[k] = [cur + 1, cur + 1 + len(s)]
            cur += 1 + len(s)
        else:
            pieces.append(s + " ")
            spans[k] = [cur, cur + len(s)]
            cur += len(s) + 1
    pieces.append(text[prev:])
    return "".join(pieces), spans  # type: ignore[return-value]


# ---------------------------------------------------------------- fillers
@dataclass
class Book:
    index: int
    book_id: str
    title: str
    text: str
    url: str = ""


class Filler:
    """Base class: subclasses provide n_books(), _raw(i) -> (book_id, title, url, raw_text), vocab(), source()."""

    name = "filler"
    pg19_split: str | None = None

    def __init__(self):
        self._books: dict[int, Book] = {}
        self._tokinfo: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def n_books(self) -> int:
        raise NotImplementedError

    def _raw(self, i: int) -> tuple[str, str, str, str]:
        raise NotImplementedError

    def book(self, i: int) -> Book:
        if i not in self._books:
            bid, title, url, raw = self._raw(i)
            self._books[i] = Book(i, bid, title, normalize_text(raw), url)
        return self._books[i]

    def vocab(self) -> frozenset[str]:
        raise NotImplementedError

    def source(self) -> dict:
        raise NotImplementedError

    def _info(self, i: int, tok) -> tuple[np.ndarray, np.ndarray]:
        key = (i, tok.name)
        if key not in self._tokinfo:
            text = self.book(i).text
            starts = tok.offsets(text)[:, 0].copy()
            sst = np.asarray(sentence_starts(text), dtype=np.int64)
            self._tokinfo[key] = (starts, sst)
        return self._tokinfo[key]

    def take_blocks(self, rng: np.random.Generator, n_blocks: int, tok, block_tokens: int = 512,
                    books: list[int] | None = None, window: int = 48) -> list[dict]:
        """n_blocks consecutive blocks of about block_tokens tokens. Starts at a random offset of books[0]
        (so that the run fits if the book is long enough) and continues into books[1:], from their start."""
        tok = as_tokenizer(tok)
        books = list(range(self.n_books())) if books is None else [int(b) for b in books]
        blocks: list[dict] = []
        first = True
        for b in books:
            if len(blocks) >= n_blocks:
                break
            bk = self.book(b)
            text = bk.text
            starts, sst = self._info(b, tok)
            n_tok = len(starts)
            lo_c = max(1500, int(0.01 * len(text))) if len(text) > 20000 else 0
            hi_c = len(text) - (int(0.01 * len(text)) if len(text) > 20000 else 0)
            lo_t = int(np.searchsorted(starts, lo_c))
            hi_t = int(np.searchsorted(starts, hi_c))
            need = (n_blocks - len(blocks)) * block_tokens
            if hi_t - lo_t < block_tokens + window:
                continue
            slack = int(need * 1.05) + window
            if first and hi_t - lo_t > slack:
                start_t = int(rng.integers(lo_t, hi_t - slack + 1))
            else:
                start_t = lo_t
            first = False
            c0 = int(starts[start_t])
            j = int(np.searchsorted(sst, c0))           # snap to the next sentence start
            if j < len(sst) and sst[j] < (starts[min(start_t + window, n_tok - 1)]):
                c0 = int(sst[j])
            while len(blocks) < n_blocks:
                t0 = max(0, int(np.searchsorted(starts, c0, side="right")) - 1)
                tt = t0 + block_tokens
                if tt + 1 >= hi_t:
                    break
                ct = int(starts[tt])
                lo_w, hi_w = int(starts[max(t0 + 1, tt - window)]), int(starts[min(tt + window, n_tok - 1)])
                a, z = int(np.searchsorted(sst, lo_w)), int(np.searchsorted(sst, hi_w, side="right"))
                if z > a:
                    cand = sst[a:z]
                    c1 = int(cand[np.argmin(np.abs(cand - ct))])
                else:
                    ws = text.rfind(" ", c0 + 1, ct)
                    c1 = ws + 1 if ws > c0 else ct
                piece = text[c0:c1].strip()
                if piece:
                    blocks.append({"text": piece, "doc_title": bk.title, "book_id": bk.book_id, "book_index": b})
                c0 = c1
        if len(blocks) < n_blocks:
            raise RuntimeError(f"filler exhausted: got {len(blocks)} of {n_blocks} blocks from books {books}")
        return blocks


class PG19Filler(Filler):
    """emozilla/pg19, one split (validation for dev items; test only after the prereg-v2 tag)."""

    name = "pg19"

    def __init__(self, split: str = "validation", local_dir: str | Path | None = None):
        super().__init__()
        if split not in ("validation", "test"):
            raise ValueError("PG19Filler uses the validation split for dev items and the test split for test items")
        self.pg19_split = split
        self.path = self._locate(split, local_dir)
        import pandas as pd
        df = pd.read_parquet(self.path, columns=["short_book_title", "url", "text"])
        self._rows = [(str(t), str(u), str(x)) for t, u, x in zip(df["short_book_title"], df["url"], df["text"])]
        self._vocab: frozenset[str] | None = None
        m = re.search(r"snapshots[\\/]+([0-9a-f]+)", str(self.path))
        self.revision = m.group(1) if m else None

    @staticmethod
    def _locate(split: str, local_dir) -> Path:
        hf_cache = Path(local_dir) if local_dir else cache_dir() / "hf"
        pat = str(hf_cache / "datasets--emozilla--pg19" / "snapshots" / "*" / "data" / f"{split}-*.parquet")
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[-1])
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files(PG19_REPO, repo_type="dataset")
                 if f.startswith(f"data/{split}-") and f.endswith(".parquet")]
        if len(files) != 1:
            raise RuntimeError(f"expected one {split} parquet in {PG19_REPO}, found {files}")
        return Path(hf_hub_download(PG19_REPO, files[0], repo_type="dataset", cache_dir=str(hf_cache)))

    def n_books(self) -> int:
        return len(self._rows)

    def _raw(self, i: int):
        t, u, x = self._rows[i]
        return f"pg19-{self.pg19_split}-{i:03d}", t, u, x

    def vocab(self) -> frozenset[str]:
        if self._vocab is None:
            words: set[str] = set()
            for i in range(self.n_books()):
                words.update(re.findall(r"[a-z]+", self.book(i).text.lower()))
            self._vocab = frozenset(words)
        return self._vocab

    def source(self) -> dict:
        from opa.data.io import sha256_file
        return {"name": "pg19", "hf_repo": PG19_REPO, "split": self.pg19_split, "revision": self.revision,
                "file": Path(self.path).name, "sha256": sha256_file(self.path), "n_books": self.n_books()}


_SYN_WORDS = ("the of and to in that was he it with as for his on is at by had not be but from they this which "
              "she or have an were her all one we there their been would so if when who what out up into them more "
              "about could some time very than then now only other made over little such upon man before great old "
              "came well good after any long should way most these through people down day must much where those "
              "under never same while last many house water light place again world night still room every along "
              "river garden window morning letter evening village road winter summer mother father friend stone "
              "hill field door table voice face hand heart mind story ship town street wood fire green quiet "
              "slowly early often across behind toward between without nothing something everything always").split()


class SyntheticFiller(Filler):
    """Deterministic pseudo-English books for tests: sentences of common words, paragraphs of 3-6 sentences."""

    name = "synthetic"

    def __init__(self, n_books: int = 8, words_per_book: int = 30000, seed: int = 0):
        super().__init__()
        self._n, self._w, self._seed = n_books, words_per_book, seed
        self._vocab = frozenset(_SYN_WORDS)

    def n_books(self) -> int:
        return self._n

    def _raw(self, i: int):
        rng = np.random.default_rng([self._seed, i])
        paras, n = [], 0
        while n < self._w:
            sents = []
            for _ in range(int(rng.integers(3, 7))):
                k = int(rng.integers(8, 21))
                ws = [_SYN_WORDS[int(j)] for j in rng.integers(0, len(_SYN_WORDS), k)]
                sents.append(" ".join(ws).capitalize() + ".")
                n += k
            paras.append(" ".join(sents))
        return f"synthetic-{i:03d}", f"Synthetic Book {i}", "", "\n\n".join(paras)

    def vocab(self) -> frozenset[str]:
        return self._vocab

    def source(self) -> dict:
        return {"name": "synthetic", "n_books": self._n, "words_per_book": self._w, "seed": self._seed}


# ---------------------------------------------------------------- assembling origins
def assemble_origins(blocks: list[dict], thread_of: list[int], inserts: dict[int, list[dict]],
                     rng: np.random.Generator, tok) -> tuple[list[dict], dict]:
    """Insert sentences into blocks and build SCHEMA origins.
    inserts[o] = [{"key", "sentence", "kind"}, ...]. Returns (origins, spans) with spans[key] = (origin_id, [s, e])."""
    tok = as_tokenizer(tok)
    texts, ins_meta, spans = [], [], {}
    for o, blk in enumerate(blocks):
        lst = inserts.get(o, [])
        text = blk["text"]
        meta = []
        if lst:
            pos = insert_positions(text, len(lst), rng)
            text, sp = insert_sentences(text, [(p, d["sentence"]) for p, d in zip(pos, lst)])
            for d, s in zip(lst, sp):
                spans[d["key"]] = (o, s)
                meta.append({"kind": d["kind"], "char_span": s})
            meta.sort(key=lambda m: m["char_span"][0])
        texts.append(text)
        ins_meta.append(meta)
    counts = tok.count_many(texts)
    origins = [{"origin_id": o, "text": texts[o], "thread_id": int(thread_of[o]), "doc_title": blocks[o]["doc_title"],
                "book_id": blocks[o]["book_id"], "n_tokens": int(counts[o]), "inserted": ins_meta[o]}
               for o in range(len(blocks))]
    return origins, spans
