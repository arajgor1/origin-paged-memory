"""Dense embedder for Track A (v3): microsoft/harrier-oss-v1-0.6b, plus precomputed embedding files.

harrier: Qwen3 architecture, 1024-dim, last-token pooling, L2-normalised. Queries get the instruction prefix
"Instruct: <task>\\nQuery: <text>"; documents (archive origins) get no prefix.

Tokenization follows the model card exactly: tok(texts, truncation=True, max_length=L, padding=True), right padding,
last real token picked with the attention mask. The tokenizer's post-processor already appends <|endoftext|>
(151643); its `eos_token` is <|im_end|> (151645) and is NOT forced. Only a tokenizer that adds no terminator at all
gets <|endoftext|> (else the pad id) appended by us (`append_id`).

Numerics: embeddings are computed in fp32 (0.6B is cheap) and the live fallback also runs fp32 on CPU, so precomputed
and live vectors come from the same numeric path. `max_length` defaults to max(1024, B + 64) (harrier supports 32768).

Embeddings are computed once per item by one job (`embed_items`) and read by every system, so all systems see
identical vectors. File format (npz, no pickles):
  doc_keys   "<item_id>/<origin_id>"        doc_emb    float32 (N, dim)       doc_sha   sha256(text)[:16]
  query_keys "<item_id>"                    query_emb  float32 (Q, dim)       query_sha sha256(question)[:16]
  selfcheck_emb float32 (3, dim): embeddings of SELFCHECK_TEXTS; the live fallback must reproduce them (cosine > 0.999)
               when it is first built, because it may run under a different transformers version than the precompute job
  meta       JSON string {model, revision, task, query_style, dim, max_length, dtype, pooling, version, B}
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path

import numpy as np

# One task string for every query in the study (Track A questions and the agent's sub-queries).
QUERY_TASK = "Given a question about an archive of documents, retrieve the archive passages that state the facts needed to answer it"
EMBED_FORMAT_VERSION = "emb-v2"
DEFAULT_EMBEDDER = "microsoft/harrier-oss-v1-0.6b"
SELFCHECK_TEXTS = ["The quick brown fox jumps over the lazy dog.",
                   "Varrel Pumps sources seals from Ostrand Ltd. under part QX-2291 rated to 83 degrees.",
                   "Which supplier of the company that makes the pump seals changed its price in March?"]
SELFCHECK_MIN_COS = 0.999


def default_max_length(B: int = 512) -> int:
    return max(1024, int(B) + 64)


def format_query(text: str, task: str = QUERY_TASK, style: str = "harrier") -> str:
    """harrier: "Instruct: <task>\nQuery: <text>". Qwen3-Embedding ("qwen3") writes "Query:" with no space after it."""
    return f"Instruct: {task}\nQuery:{text}" if style == "qwen3" else f"Instruct: {task}\nQuery: {text}"


def query_style_for(model: str) -> str:
    return "qwen3" if "qwen3-embedding" in model.lower() else "harrier"


def last_token_pool(hidden, attention_mask):
    """hidden (B, T, D), attention_mask (B, T) -> (B, D): the hidden state of the last non-padding token."""
    import torch
    if bool((attention_mask[:, -1] == 1).all()):          # left padding (or no padding)
        return hidden[:, -1]
    idx = attention_mask.sum(dim=1) - 1                    # right padding
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_sha = text_sha


def vec_sha(v) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()[:16]


def resolve_revision(model: str, revision: str | None = None, encoder=None) -> str | None:
    """The commit hash actually used: the pinned one, else the loaded config's, else the Hub's current head."""
    if revision:
        return revision
    h = getattr(getattr(encoder, "config", None), "_commit_hash", None)
    if h:
        return h
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model).sha
    except Exception:
        return None


class HFEmbedder:
    """harrier-oss-v1-0.6b by default. `tokenizer` / `encoder` can be injected (tests use a tiny stub)."""

    def __init__(self, model: str = DEFAULT_EMBEDDER, device: str | None = None, batch: int = 16,
                 max_length: int | None = None, task: str = QUERY_TASK, tokenizer=None, encoder=None,
                 revision: str | None = None, B: int = 512, dtype: str = "float32", query_style: str | None = None):
        import torch
        self.name, self.batch, self.task = model, batch, task
        self.query_style = query_style or query_style_for(model)
        self.max_length = int(max_length or default_max_length(B))
        self.B = B
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError(f"unsupported embedder dtype {dtype!r}")
        self.dtype = dtype
        if tokenizer is None or encoder is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model, revision=revision)
            encoder = encoder or AutoModel.from_pretrained(model, revision=revision, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        self.tok, self.model = tokenizer, encoder
        self.revision = resolve_revision(model, revision, encoder)
        self.dim = None
        self.append_id = self._terminator_to_append()

    def _terminator_to_append(self):
        """None if the tokenizer's post-processor adds a terminator (harrier: <|endoftext|>); else the id to append."""
        with_sp = self.tok(["probe"])["input_ids"][0]
        without = self.tok(["probe"], add_special_tokens=False)["input_ids"][0]
        if len(with_sp) > len(without):
            return None
        cand = None
        conv = getattr(self.tok, "convert_tokens_to_ids", None)
        if conv is not None:
            try:
                cand = conv("<|endoftext|>")
            except Exception:
                cand = None
        unk = getattr(self.tok, "unk_token_id", None)
        if cand is None or cand == unk:
            cand = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        return cand

    def _tensors(self, texts: list[str]):
        import torch
        if self.append_id is None:                         # the model card's call
            enc = self.tok(list(texts), truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
            return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        ids = [list(x)[: self.max_length - 1] + [self.append_id]
               for x in self.tok(list(texts), truncation=True, max_length=self.max_length - 1, add_special_tokens=False)["input_ids"]]
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.append_id
        T = max(len(i) for i in ids)
        x = torch.full((len(ids), T), pad, dtype=torch.long)
        m = torch.zeros((len(ids), T), dtype=torch.long)
        for r, i in enumerate(ids):
            x[r, :len(i)], m[r, :len(i)] = torch.tensor(i), 1
        return x.to(self.device), m.to(self.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Documents: no prefix."""
        import torch
        texts = list(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res: list[np.ndarray | None] = [None] * len(texts)
        with torch.no_grad():
            for s in range(0, len(order), self.batch):
                sel = order[s:s + self.batch]
                x, m = self._tensors([texts[i] for i in sel])
                h = last_token_pool(self.model(input_ids=x, attention_mask=m).last_hidden_state, m)
                h = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
                for r, i in enumerate(sel):
                    res[i] = h[r]
        if not res:
            return np.zeros((0, self.dim or 1024), np.float32)
        self.dim = len(res[0])
        return np.stack(res).astype(np.float32)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Queries: instruction prefix."""
        return self.embed([format_query(t, self.task, self.query_style) for t in texts])

    def provenance(self) -> dict:
        return {"model": self.name, "revision": self.revision, "task": self.task, "max_length": self.max_length,
                "dtype": self.dtype, "pooling": "last_token_l2", "B": self.B, "device": self.device,
                "query_style": self.query_style}


# ---------------------------------------------------------------- startup self-check against a reference implementation


def reference_embeddings(model: str, texts: list[str], max_length: int, revision: str | None = None) -> np.ndarray:
    """Reference vectors: sentence-transformers if installed, else the model card's snippet (AutoModel, padding=True,
    last-token pooling with the left-padding test, L2). CPU, fp32."""
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model, device="cpu", revision=revision)
        st.max_seq_length = max_length
        return np.asarray(st.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    except ImportError:
        pass
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    mdl = AutoModel.from_pretrained(model, revision=revision, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        b = tok(list(texts), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        h = mdl(**b).last_hidden_state
        am = b["attention_mask"]
        if bool(am[:, -1].sum() == am.shape[0]):
            v = h[:, -1]
        else:
            v = h[torch.arange(h.shape[0]), am.sum(dim=1) - 1]
        return torch.nn.functional.normalize(v.float(), dim=-1).numpy()


def check_against_reference(ours: np.ndarray, ref: np.ndarray, min_cos: float = SELFCHECK_MIN_COS) -> list[float]:
    """Row-wise cosine of two (n, d) matrices; raises RuntimeError if any is below `min_cos` or the shapes differ."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if ours.shape != ref.shape:
        raise RuntimeError(f"embedder self-check: shape {ours.shape} != reference {ref.shape}")
    cos = (ours * ref).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(ref, axis=1))
    if not bool((cos > min_cos).all()):
        raise RuntimeError(f"embedder self-check failed: cosines {np.round(cos, 5).tolist()} (need > {min_cos})")
    return [float(c) for c in cos]


def selfcheck(embedder, reference_fn=None, texts: list[str] | None = None, min_cos: float = SELFCHECK_MIN_COS) -> list[float]:
    texts = texts or SELFCHECK_TEXTS
    ref_fn = reference_fn or (lambda t: reference_embeddings(embedder.name, t, embedder.max_length, embedder.revision))
    return check_against_reference(embedder.embed(texts), ref_fn(texts), min_cos)


# ---------------------------------------------------------------- precomputed embeddings


def compute_item_embeddings(items: list[dict], embedder) -> dict:
    """Embed every origin (documents) and question (query) of `items`; returns the arrays of the npz format."""
    dk, dsha, dtexts, qk, qsha, qtexts = [], [], [], [], [], []
    for it in items:
        for o in it["origins"]:
            dk.append(f"{it['item_id']}/{o['origin_id']}"), dsha.append(_sha(o["text"])), dtexts.append(o["text"])
        qk.append(str(it["item_id"])), qsha.append(_sha(it["question"])), qtexts.append(it["question"])
    demb = embedder.embed(dtexts) if dtexts else np.zeros((0, 1), np.float32)
    qemb = embedder.embed_query(qtexts) if qtexts else np.zeros((0, 1), np.float32)
    prov = embedder.provenance() if hasattr(embedder, "provenance") else {"model": getattr(embedder, "name", "?")}
    sc = embedder.embed(SELFCHECK_TEXTS).astype(np.float32)        # lets the live fallback prove it matches these vectors
    meta = {"task": getattr(embedder, "task", QUERY_TASK), **prov, "dim": int(demb.shape[1]), "version": EMBED_FORMAT_VERSION}
    return {"doc_keys": np.array(dk, dtype=str), "doc_emb": demb.astype(np.float32), "doc_sha": np.array(dsha, dtype=str),
            "query_keys": np.array(qk, dtype=str), "query_emb": qemb.astype(np.float32), "query_sha": np.array(qsha, dtype=str),
            "selfcheck_emb": sc, "meta": np.array(json.dumps(meta))}


def embeddings_to_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def write_embeddings(path, arrays: dict):
    Path(path).write_bytes(embeddings_to_bytes(arrays))


class PrecomputedEmbedder:
    """Reads an npz written by `embed_items`. `for_item(item)` binds it to one item; the bound embedder returns the
    stored vectors in origin order and the stored question vector. Text hashes are checked, so a rechunked item
    (different B) or an edited text cannot silently pick up stale vectors. Texts that were not precomputed (agent
    sub-queries) go to `fallback` (a live embedder or a zero-argument factory, built once under a lock) or raise.
    Every live embedding is logged with the sha256 of the text and of the vector, so runs can be replayed."""

    def __init__(self, source, fallback=None):
        data = np.load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source, allow_pickle=False)
        self.meta = json.loads(str(data["meta"]))
        self.name = self.meta["model"] + "+precomputed"
        self.task = self.meta.get("task", QUERY_TASK)
        self.doc = {k: (i, s) for i, (k, s) in enumerate(zip(data["doc_keys"].tolist(), data["doc_sha"].tolist()))}
        self.doc_emb = data["doc_emb"]
        self.selfcheck_emb = data["selfcheck_emb"] if "selfcheck_emb" in data.files else None
        self.q = {k: (i, s) for i, (k, s) in enumerate(zip(data["query_keys"].tolist(), data["query_sha"].tolist()))}
        self.query_emb = data["query_emb"]
        self.fallback = fallback
        self._lock = threading.Lock()

    def provenance(self) -> dict:
        return dict(self.meta)

    def for_item(self, item: dict) -> "_ItemEmbedder":
        rows = []
        for o in item["origins"]:
            key = f"{item['item_id']}/{o['origin_id']}"
            if key not in self.doc:
                raise KeyError(f"no precomputed embedding for {key}")
            i, sha = self.doc[key]
            if sha != _sha(o["text"]):
                raise ValueError(f"precomputed embedding for {key} was made from different text (rechunked item?)")
            rows.append(i)
        qi = None
        if str(item["item_id"]) in self.q:
            i, sha = self.q[str(item["item_id"])]
            if sha != _sha(item["question"]):
                raise ValueError(f"precomputed query embedding for {item['item_id']} was made from a different question")
            qi = i
        return _ItemEmbedder(self, rows, qi, item["question"], [_sha(o["text"]) for o in item["origins"]])

    # unbound use: only through the fallback
    def embed(self, texts):
        return self._fb().embed(texts)

    def embed_query(self, texts):
        return self._fb().embed_query(texts)

    def _fb(self):
        with self._lock:
            if self.fallback is None:
                raise RuntimeError("PrecomputedEmbedder has no live fallback; this text was not precomputed")
            if callable(self.fallback) and not hasattr(self.fallback, "embed"):
                fb = self.fallback()
                d = int(fb.embed(["dimension probe"]).shape[1])
                if d != int(self.doc_emb.shape[1]):
                    raise RuntimeError(f"live fallback has dim {d}, precomputed vectors have {self.doc_emb.shape[1]}")
                if self.selfcheck_emb is not None:     # same vectors for the fixed strings under this library version?
                    check_against_reference(fb.embed(SELFCHECK_TEXTS), self.selfcheck_emb,
                                            0.999 if self.meta.get("dtype", "float32") == "float32" else 0.99)
                self.fallback = fb
            return self.fallback


class _ItemEmbedder:
    def __init__(self, parent: PrecomputedEmbedder, rows: list[int], qrow, question: str, shas: list[str]):
        self.p, self.rows, self.qrow, self.question, self.shas = parent, rows, qrow, question, shas
        self.name, self.task = parent.name, parent.task
        self.log = None                                       # set by the runner: log(event_dict)

    def _logged(self, kind: str, texts, vecs):
        if self.log:
            for t, v in zip(texts, vecs):
                self.log({"action": "live_embed", "kind": kind, "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
                          "vec_sha256": vec_sha(v), "text_head": t[:80]})

    def embed(self, texts):
        if [_sha(t) for t in texts] == self.shas:           # the Session call: all origin texts of the item, in order
            return self.p.doc_emb[self.rows].copy()
        out = self.p._fb().embed(texts)
        self._logged("document", texts, out)
        return out

    def embed_query(self, texts):
        out = []
        for t in texts:
            if self.qrow is not None and t == self.question:
                out.append(self.p.query_emb[self.qrow])
            else:
                v = self.p._fb().embed_query([t])[0]
                self._logged("query", [t], [v])
                out.append(v)
        return np.stack(out).astype(np.float32) if out else np.zeros((0, self.p.doc_emb.shape[1]), np.float32)
