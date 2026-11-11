"""v3 changes to Tracks A / A2: open models only, harrier pooling and prefixes, precomputed embeddings, 32K budgets."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_track_a import item, make_item, policy  # noqa: E402,F401

from opa.tracks.track_a import config
from opa.tracks.track_a.embed import SELFCHECK_TEXTS  # noqa: E402
from opa.tracks.track_a.embed import (QUERY_TASK, HFEmbedder, PrecomputedEmbedder, check_against_reference,
                                      compute_item_embeddings, embeddings_to_bytes, format_query, last_token_pool, selfcheck,
                                      vec_sha, write_embeddings)
from opa.tracks.track_a.index import HashEmbedder
from opa.tracks.track_a.models import ApiChat, ContextOverflow, FakeChat, OpenAIChat, Usage, ask_json
from opa.tracks.track_a.run import LocalCtx, embed_items, make_chat, make_embedder, run_system


# ---------------------------------------------------------------- models
def test_api_chat_refused():
    with pytest.raises(RuntimeError, match="disabled in v3"):
        ApiChat(object())
    with pytest.raises(RuntimeError, match="refused in v3"):
        make_chat({"model": {"kind": "api"}}, LocalCtx("."))


def test_model_resolution_and_system_fold():
    assert config.resolve_model(None) == "Qwen/Qwen3-4B-Instruct-2507" == config.resolve_model("qwen")
    assert config.resolve_model("gemma") == "google/gemma-4-E4B-it"
    assert config.resolve_model("org/other") == "org/other"
    assert not config.needs_system_fold("Qwen/Qwen3-4B-Instruct-2507") and config.needs_system_fold("google/gemma-4-E4B-it")
    q = OpenAIChat("http://x", "Qwen/Qwen3-4B-Instruct-2507")
    g = OpenAIChat("http://x", "google/gemma-4-E4B-it", fold_system=True)
    assert [m["role"] for m in q.messages("S", "U")] == ["system", "user"]
    gm = g.messages("S", "U")
    assert [m["role"] for m in gm] == ["user"] and gm[0]["content"] == "S\n\nU"


def test_make_chat_open_dict_folds_for_gemma():
    c = make_chat({"model": {"kind": "open", "base_url": "http://x", "name": "gemma"}}, LocalCtx("."))
    assert c.name == "google/gemma-4-E4B-it" and c.fold_system
    c = make_chat({"model": {"kind": "open", "base_url": "http://x", "name": "qwen"}}, LocalCtx("."))
    assert c.name == "Qwen/Qwen3-4B-Instruct-2507" and not c.fold_system


def test_vllm_args_for_l4():
    a = config.vllm_args("Qwen/Qwen3-4B-Instruct-2507")
    assert a[a.index("--max-model-len") + 1] == "40960" and a[a.index("--dtype") + 1] == "bfloat16"
    assert a[a.index("--gpu-memory-utilization") + 1] == "0.9" and "--enable-prefix-caching" in a


def test_budgets_for_32k():
    assert config.TOKEN_BUDGET == 100_000 and config.CONTEXT_LIMIT + 4096 <= config.MAX_MODEL_LEN
    assert config.CONTEXT_LIMIT > 32768 + 64 * 20                    # a 32K item with origin tags fits


# ---------------------------------------------------------------- harrier pooling / prefix (stub model, no download)
ENDOFTEXT, IM_END, PAD = 5, 7, 0        # harrier: post-processor appends <|endoftext|>; eos_token is <|im_end|>


class StubTok:
    """Mimics harrier's tokenizer: add_special_tokens=True appends <|endoftext|> (5) although eos_token_id is 7."""
    eos_token_id, pad_token_id, unk_token_id = IM_END, PAD, None
    adds_terminator = True

    def convert_tokens_to_ids(self, t):
        return ENDOFTEXT

    def __call__(self, texts, truncation=False, max_length=None, padding=False, return_tensors=None, add_special_tokens=True):
        out = []
        for t in texts:
            ids = [10 + (ord(c) % 50) for c in t]
            term = [ENDOFTEXT] if (add_special_tokens and self.adds_terminator) else []
            if truncation and max_length:
                ids = ids[: max_length - len(term)]
            out.append(ids + term)
        if not return_tensors:
            return {"input_ids": out}
        T = max(len(i) for i in out)
        ids_t = torch.full((len(out), T), PAD, dtype=torch.long)
        m = torch.zeros((len(out), T), dtype=torch.long)
        for r, i in enumerate(out):
            ids_t[r, :len(i)], m[r, :len(i)] = torch.tensor(i), 1
        return {"input_ids": ids_t, "attention_mask": m}


class NoTermTok(StubTok):
    adds_terminator = False


def _hid(prefix_sum: torch.Tensor, dim: int = 8) -> torch.Tensor:
    return torch.stack([((prefix_sum * (d + 1) + d * 7) % 13).float() + 0.1 for d in range(dim)], dim=-1)


class StubModel:
    """Causal toy encoder: the hidden state at t depends on the sum of token ids up to t (so the pooled position matters)."""

    def __init__(self, dim=8):
        self.dim, self.calls = dim, []

    def __call__(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        return type("O", (), {"last_hidden_state": _hid(torch.cumsum(input_ids * attention_mask, dim=1), self.dim)})()


def _emb(tok=None, **kw):
    return HFEmbedder("stub", device="cpu", tokenizer=tok or StubTok(), encoder=StubModel(), batch=2, revision="abc123", **kw)


def _ref(text, term=ENDOFTEXT):
    v = _hid(torch.tensor(sum(10 + (ord(c) % 50) for c in text) + term))
    return (v / v.norm()).numpy()


def test_last_token_pool_right_and_left_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float).reshape(2, 4, 3)
    right = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert torch.equal(last_token_pool(h, right), torch.stack([h[0, 2], h[1, 3]]))
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(last_token_pool(h, left), h[:, -1])


def test_harrier_official_tokenization_tail_is_endoftext_only():
    e = _emb()
    assert e.append_id is None                                          # tokenizer already terminates: nothing forced
    texts = ["a longer document text", "short", "mid length"]
    v = e.embed(texts)
    assert v.shape == (3, 8) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    for ids, m in e.model.calls:                                        # tail is [..., <|endoftext|>], never <|im_end|>
        for row, mm in zip(ids, m):
            real = row[mm.bool()]
            assert real[-1].item() == ENDOFTEXT and IM_END not in real.tolist() and real.tolist().count(ENDOFTEXT) == 1
    # batching and sorting by length must not change a text's vector or the output order
    assert np.allclose(e.embed(["short"])[0], v[1], atol=1e-6) and np.allclose(e.embed([texts[2]])[0], v[2], atol=1e-6)
    for t, row in zip(texts, v):                                        # pooled at <|endoftext|>
        assert np.allclose(row, _ref(t), atol=1e-5)


def test_tokenizer_without_terminator_gets_endoftext_appended():
    e = _emb(NoTermTok())
    assert e.append_id == ENDOFTEXT
    v = e.embed(["short", "a somewhat longer text"])
    assert np.allclose(v[0], _ref("short"), atol=1e-5) and np.allclose(v[1], _ref("a somewhat longer text"), atol=1e-5)


def test_max_length_truncation_keeps_terminator_and_default():
    from opa.tracks.track_a.embed import default_max_length
    assert default_max_length(512) == 1024 and default_max_length(2048) == 2112
    e = _emb(max_length=6)
    e.embed(["x" * 50])
    ids, m = e.model.calls[0]
    assert ids.shape[1] == 6 and ids[0, -1].item() == ENDOFTEXT
    assert _emb(B=2048).max_length == 2112


def test_harrier_query_prefix_and_task():
    assert format_query("who?") == f"Instruct: {QUERY_TASK}\nQuery: who?"
    e = _emb()
    q = e.embed_query(["who?"])[0]
    assert np.allclose(q, e.embed([format_query("who?")])[0]) and not np.allclose(q, e.embed(["who?"])[0])


def test_make_embedder_names_and_refusals(monkeypatch):
    assert isinstance(make_embedder("hash"), HashEmbedder)
    with pytest.raises(ValueError, match="bge-small"):
        make_embedder("bge-small")
    with pytest.raises(ValueError):
        make_embedder("BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)      # a GPU container: live harrier is refused
    with pytest.raises(RuntimeError, match="GPU container"):
        make_embedder("harrier")


# ---------------------------------------------------------------- self-check against a reference implementation
def test_check_against_reference_logic():
    a = np.eye(3, dtype=np.float32)
    assert check_against_reference(a, a * 2.0) == [1.0, 1.0, 1.0]
    bad = a.copy()
    bad[1] = [0.2, 0.98, 0.0]
    with pytest.raises(RuntimeError, match="self-check failed"):
        check_against_reference(a, bad)
    with pytest.raises(RuntimeError, match="shape"):
        check_against_reference(a, a[:2])


def test_selfcheck_uses_reference_and_embed_items_fails_on_mismatch(tmp_path, item, monkeypatch):
    e = _emb()
    texts = ["t1", "t2", "t3"]
    assert selfcheck(e, reference_fn=lambda t: np.stack([_ref(x) for x in t]), texts=texts) == pytest.approx([1.0] * 3)
    with pytest.raises(RuntimeError):
        selfcheck(e, reference_fn=lambda t: np.stack([_ref(x, term=IM_END) for x in t]), texts=texts)
    import opa.tracks.track_a.embed as emb_mod
    import opa.tracks.track_a.run as run_mod
    ip = tmp_path / "items.jsonl.gz"
    with gzip.open(ip, "wt") as f:
        f.write(json.dumps(item) + "\n")
    monkeypatch.setattr(run_mod, "make_embedder", lambda *a, **k: _emb())
    monkeypatch.setattr(emb_mod, "reference_embeddings", lambda m, t, ml, rev=None: np.stack([_ref(x, term=IM_END) for x in t]))
    with pytest.raises(RuntimeError, match="self-check failed"):
        embed_items({"items_uri": str(ip)}, LocalCtx(str(tmp_path / "bad")))
    monkeypatch.setattr(emb_mod, "reference_embeddings", lambda m, t, ml, rev=None: np.stack([_ref(x) for x in t]))
    out = embed_items({"items_uri": str(ip)}, LocalCtx(str(tmp_path / "good")))
    assert out["metrics"]["selfcheck_cosines"] and out["metrics"]["provenance"]["max_length"] == 1024
    assert out["metrics"]["provenance"]["revision"] == "abc123" and out["metrics"]["provenance"]["dtype"] == "float32"


# ---------------------------------------------------------------- raw I/O logging, seed, overflow
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _Http:
    def __init__(self, resp):
        self.resp, self.bodies = resp, []

    def post(self, url, json):
        self.bodies.append(json)
        return self.resp


def _chat(resp, **kw):
    c = OpenAIChat("http://x", kw.pop("model", "Qwen/Qwen3-4B-Instruct-2507"), **kw)
    c._http = _Http(resp)
    return c


def test_llm_call_logged_with_fold_seed_usage():
    ok = _Resp(200, {"choices": [{"message": {"content": "{\"a\": 1}"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 3,
                               "prompt_tokens_details": {"cached_tokens": 8}}})
    c = _chat(ok, model="google/gemma-4-E4B-it", fold_system=True, seed=7)
    ev, u = [], Usage()
    assert ask_json(c, "SYS", "USR", 50, u, ev.append) == {"a": 1}
    assert c.http.bodies[0]["seed"] == 7 and c.http.bodies[0]["temperature"] == 0
    e = ev[0]
    assert e["action"] == "llm_call" and e["messages"] == [{"role": "user", "content": "SYS\n\nUSR"}]      # after the fold
    assert (e["model"], e["max_tokens"], e["reply"], e["finish_reason"], e["seed"]) == \
        ("google/gemma-4-E4B-it", 50, "{\"a\": 1}", "stop", 7)
    assert e["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "cached_tokens": 8}
    assert (u.tokens_in, u.tokens_out, u.cached_tokens) == (11, 3, 8)


def test_context_overflow_is_not_retried_and_maps_to_does_not_fit(item):
    bad = _Resp(400, {"message": "This model's maximum context length is 40960 tokens. However, you requested 41500"})
    c = _chat(bad)
    with pytest.raises(ContextOverflow):
        c.ask("s", "u")
    assert len(c.http.bodies) == 1                                       # no retry
    other = _chat(_Resp(400, {"message": "bad field"}))
    with pytest.raises(RuntimeError):
        other.ask("s", "u")
    assert len(other.http.bodies) == 1

    def boom(system, user):
        raise ContextOverflow("too long")
    rec, events = _run_with(FakeChat(boom), "S2", item)
    assert rec["status"] == "does_not_fit" and rec["prediction"] == ""


def _run_with(chat, system, item, emb=None, opts=None):
    from opa.tracks.track_a.index import Session
    from opa.tracks.track_a.systems import Ctx, run_item, session_cfg
    emb = emb or HashEmbedder()
    sess = Session(item, session_cfg(system, opts or {}), emb, skeleton_llm={})
    events = []
    c = Ctx(item=item, sess=sess, chat=chat, opts={**(opts or {}), "_q_emb": emb.embed_query([item["question"]])[0],
                                                     "_embedder": emb}, log_fn=events.append)
    return run_item(system, c), events


class SpyEmb(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.queries = []

    def embed_query(self, texts):
        self.queries += list(texts)
        return super().embed_query(texts)


def test_s3_first_query_is_the_question_without_trailing_space(item):
    spy = SpyEmb()
    rec, _ = _run_with(FakeChat(policy), "S3", item, spy)
    assert spy.queries[0] == item["question"] and len(spy.queries) >= 1 and all(q == q.strip() for q in spy.queries)


# ---------------------------------------------------------------- A2 job spec routing
def test_a2_job_spec_uses_vllm_image_and_checks_gpu():
    from opa.launch import function_name
    from opa.tracks.track_a2 import job_spec
    prices = {"modal": {"gpu_per_second": {"L4": 0.000222, "T4": 0.000164}, "cpu_core_per_second": 0.0000131,
                        "memory_gib_per_second": 0.00000222}}
    spec = job_spec("x", "D2", "u", prices=prices)
    assert function_name(spec["compute"]) == "run_job_vllm" and spec["compute"]["image"] == "vllm"
    assert "prefill tokens" in spec["estimate_formula"] and "PLACEHOLDER" in spec["estimate_formula"]
    assert spec["throughput_placeholders"]["status"].startswith("placeholder")
    with pytest.raises(ValueError, match="unknown gpu"):
        job_spec("x", "D2", "u", prices=prices, gpu="H900")


# ---------------------------------------------------------------- precomputed embeddings
def test_embedding_cache_round_trip(tmp_path, item):
    items = [item, make_item(seed=1)]
    arrays = compute_item_embeddings(items, HashEmbedder())
    assert arrays["doc_emb"].shape == (sum(len(i["origins"]) for i in items), 512)
    p = tmp_path / "e.npz"
    write_embeddings(p, arrays)
    pe = PrecomputedEmbedder(p.read_bytes(), fallback=HashEmbedder())
    h = HashEmbedder()
    for it in items:
        b = pe.for_item(it)
        assert np.array_equal(b.embed([o["text"] for o in it["origins"]]), h.embed([o["text"] for o in it["origins"]]))
        assert np.array_equal(b.embed_query([it["question"]])[0], h.embed_query([it["question"]])[0])
        assert np.allclose(b.embed_query(["another query"]), h.embed_query(["another query"]))      # fallback
    stale = json.loads(json.dumps(item))
    stale["origins"][0]["text"] += " edited"
    with pytest.raises(ValueError, match="different text"):
        pe.for_item(stale)
    with pytest.raises(RuntimeError):
        PrecomputedEmbedder(p.read_bytes()).for_item(item).embed_query(["unseen"])


def test_embed_items_entrypoint_and_run_with_precomputed(tmp_path, item):
    items = [item, make_item(seed=1)]
    ip = tmp_path / "items.jsonl.gz"
    with gzip.open(ip, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    ctx = LocalCtx(str(tmp_path / "emb"))
    out = embed_items({"items_uri": str(ip), "embedder": "hash"}, ctx)
    assert out["metrics"]["items"] == 2 and out["metrics"]["dim"] == 512
    npz = tmp_path / "emb" / "embeddings.npz"
    assert npz.exists()
    results = {}
    for tag, emb in (("hash", "hash"), ("prec", f"precomputed:{npz}")):
        c2 = LocalCtx(str(tmp_path / ("o_" + tag)))
        run_system({"items_uri": str(ip), "system": "S2", "embedder": emb, "workers": 1,
                    "model": {"kind": "fake", "chat": FakeChat(policy)}}, c2)
        recs = gzip.decompress((tmp_path / ("o_" + tag) / "records.jsonl.gz").read_bytes()).decode().splitlines()
        results[tag] = [(json.loads(x)["item_id"], json.loads(x)["prediction"], json.loads(x).get("paged_in")) for x in recs]
    assert results["hash"] == results["prec"]                       # identical vectors give identical runs


def test_fallback_logging_dim_check_and_meta(tmp_path, item):
    h = HashEmbedder()
    arrays = compute_item_embeddings([item], h)
    meta = json.loads(str(arrays["meta"]))
    assert meta["model"] == h.name and meta["dim"] == 512 and meta["version"] == "emb-v2"
    pe = PrecomputedEmbedder(embeddings_to_bytes(arrays), fallback=lambda: HashEmbedder())
    b = pe.for_item(item)
    ev = []
    b.log = ev.append
    v = b.embed_query(["live sub query"])[0]
    assert np.array_equal(v, h.embed_query(["live sub query"])[0])
    import hashlib
    assert ev[0]["action"] == "live_embed" and ev[0]["text_sha256"] == hashlib.sha256(b"live sub query").hexdigest()
    assert ev[0]["vec_sha256"] == vec_sha(v)
    b.embed_query([item["question"]])                                  # precomputed question: not logged
    assert len(ev) == 1
    wrong = PrecomputedEmbedder(embeddings_to_bytes(arrays), fallback=lambda: HashEmbedder(dim=64))
    with pytest.raises(RuntimeError, match="dim"):
        wrong.for_item(item).embed_query(["x"])


def test_run_system_logs_provenance_and_live_embeds(tmp_path, item):
    ip = tmp_path / "items.jsonl.gz"
    with gzip.open(ip, "wt") as f:
        f.write(json.dumps(item) + "\n")
    ctx = LocalCtx(str(tmp_path / "o"))
    run_system({"items_uri": str(ip), "system": "S3", "embedder": "hash", "workers": 1,
                "model": {"kind": "fake", "chat": FakeChat(policy)}}, ctx)
    assert any(e["action"] == "provenance" for e in ctx.events)
    m = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert m["provenance"]["chat"]["model"] == "fake" or "model" in m["provenance"]["chat"]


def test_fallback_selfcheck_on_first_build(item):
    class Shifted(HashEmbedder):                       # a fallback whose numerics differ from the precompute job's
        def embed(self, texts):
            return HashEmbedder.embed(self, [t + " extra words" for t in texts])

    arrays = compute_item_embeddings([item], HashEmbedder())
    assert arrays["selfcheck_emb"].shape == (3, 512)
    ok = PrecomputedEmbedder(embeddings_to_bytes(arrays), fallback=lambda: HashEmbedder())
    ok.for_item(item).embed_query(["unseen"])          # first build passes the check
    bad = PrecomputedEmbedder(embeddings_to_bytes(arrays), fallback=lambda: Shifted())
    with pytest.raises(RuntimeError, match="self-check failed"):
        bad.for_item(item).embed_query(["unseen"])


# ---------------------------------------------------------------- multi-system jobs, per-URI items, bucket, embedder_check
def _items_file(tmp_path, items, name="items.jsonl.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return p


def test_run_system_systems_and_repeats_layout(tmp_path, item):
    ip = _items_file(tmp_path, [item, make_item(seed=1)])
    ctx = LocalCtx(str(tmp_path / "o"))
    out = run_system({"items_uri": str(ip), "systems": ["S1", "S2"], "repeats": 2, "embedder": "hash", "workers": 2,
                      "model": {"kind": "fake", "chat": FakeChat(policy)}}, ctx)
    assert set(out["metrics"]["runs"]) == {"S1/r0", "S1/r1", "S2/r0", "S2/r1"}
    for s in ("S1", "S2"):
        for r in (0, 1):
            d = tmp_path / "o" / s / f"r{r}"
            recs = [json.loads(x) for x in gzip.decompress((d / "records.jsonl.gz").read_bytes()).decode().splitlines()]
            m = json.loads((d / "metrics.json").read_text())
            assert len(recs) == 2 and {x["repeat"] for x in recs} == {r} and m["system"] == s and m["repeat"] == r
    assert {(e["system"], e["repeat"]) for e in ctx.events if "item_id" in e and "repeat" in e} == \
        {("S1", 0), ("S1", 1), ("S2", 0), ("S2", 1)}
    assert json.loads((tmp_path / "o" / "metrics.json").read_text())["repeats"] == 2


def test_single_system_form_keeps_root_layout(tmp_path, item):
    ip = _items_file(tmp_path, [item])
    ctx = LocalCtx(str(tmp_path / "o"))
    out = run_system({"items_uri": str(ip), "system": "S2", "embedder": "hash", "model": {"kind": "fake", "chat": FakeChat(policy)}}, ctx)
    assert (tmp_path / "o" / "records.jsonl.gz").exists() and out["metrics"]["system"] == "S2"


def test_per_uri_n_items_in_select_items_and_build_skeletons(tmp_path):
    a = _items_file(tmp_path, [make_item(seed=s) for s in range(3)], "a.jsonl.gz")
    b = _items_file(tmp_path, [make_item(seed=s) for s in range(10, 14)], "b.jsonl.gz")
    from opa.tracks.track_a.run import build_skeletons, select_items
    ctx = LocalCtx(str(tmp_path / "o"))
    got = select_items({"items_uri": [{"uri": str(a), "n_items": 2}, {"uri": str(b), "n_items": 1, "offset": 2}]}, ctx)
    assert len(got) == 3
    assert len(select_items({"items_uri": [str(a), str(b)], "n_items": 1}, ctx)) == 2          # old form
    out = build_skeletons({"items_uri": [{"uri": str(a), "n_items": 1}, str(b)], "n_items": 2, "workers": 1,
                           "model": {"kind": "fake", "chat": SkChat()}}, ctx)
    assert out["metrics"]["items"] == 3


def test_precomputed_bucket_substitution(monkeypatch, tmp_path, item):
    import opa.gcs as gcs
    arrays = compute_item_embeddings([item], HashEmbedder())
    seen = []
    monkeypatch.setattr(gcs, "get_bytes", lambda uri: seen.append(uri) or embeddings_to_bytes(arrays))
    make_embedder("precomputed:gs://{bucket}/opa/x.npz", bucket="my-bkt")
    import opa.tracks.track_a.run as run_mod
    monkeypatch.setattr(run_mod, "config_bucket", lambda: "cfg-bkt")
    make_embedder("precomputed:gs://{bucket}/opa/x.npz")
    assert seen == ["gs://my-bkt/opa/x.npz", "gs://cfg-bkt/opa/x.npz"]


def test_query_style_per_embedder():
    from opa.tracks.track_a.embed import query_style_for
    assert query_style_for("Qwen/Qwen3-Embedding-4B") == "qwen3" and query_style_for("microsoft/harrier-oss-v1-0.6b") == "harrier"
    assert format_query("q", "T", "qwen3") == "Instruct: T\nQuery:q" and format_query("q", "T") == "Instruct: T\nQuery: q"
    e = _emb(query_style="qwen3")
    assert np.allclose(e.embed_query(["q"])[0], e.embed([format_query("q", e.task, "qwen3")])[0])


def test_recall_at_k_and_embedder_check_decision(tmp_path, item):
    from opa.tracks.track_a.run import embedder_check, recall_at_k
    assert recall_at_k([0.1, 0.9, 0.5, 0.2], [1, 2], 1) == 0.5 and recall_at_k([0.1, 0.9, 0.5, 0.2], [1, 2], 2) == 1.0
    ip = _items_file(tmp_path, [item, make_item(seed=1), make_item(late=False, seed=2)])
    emb_path = tmp_path / "emb"
    embed_items({"items_uri": str(ip), "embedder": "hash"}, LocalCtx(str(emb_path)))
    ctx = LocalCtx(str(tmp_path / "chk"))
    out = embedder_check({"items_uri": str(ip), "M": 4, "ks": [1, 4, 8],
                          "embedders": [{"label": "harrier", "name": "hash"},
                                        {"label": "pre", "name": f"precomputed:{emb_path / 'embeddings.npz'}"}]}, ctx)
    m = out["metrics"]
    assert m["ks"] == [1, 4, 8] and set(m["embedders"]) == {"harrier", "pre"}
    assert m["embedders"]["harrier"]["recall"] == m["embedders"]["pre"]["recall"]       # identical vectors
    assert m["decision"]["decision"] == "keep_baseline" and m["decision"]["delta_recall_at_M"] == 0.0
    assert (tmp_path / "chk" / "embedder_check.csv").read_text().startswith("embedder,k,recall")
    assert json.loads((tmp_path / "chk" / "metrics.json").read_text())["decision"]["M"] == 4


class _Fixed:
    """Per-item embedder: gold-fact origins score highest if `good`, else the question matches nothing in particular."""

    name = "fixed"

    def __init__(self, good):
        self.good = good

    def for_item(self, it):
        gold, good = set(it["gold"]["fact_origins"]), self.good

        class B:
            def embed(s, texts):
                d = np.zeros((len(texts), 512), np.float32)
                for i in range(len(texts)):
                    d[i, 0 if (good and i in gold) else 1 + i % 500] = 1.0
                return d

            def embed_query(s, texts):
                q = np.zeros((len(texts), 512), np.float32)
                q[:, 0] = 1.0
                return q
        return B()


def test_embedder_check_switch_rule_and_selfcheck(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.embed as emb_mod
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    made = {"base": _Fixed(False), "chal": _Fixed(True)}
    monkeypatch.setattr(run_mod, "make_embedder", lambda name, **k: made[name])
    out = run_mod.embedder_check({"items_uri": str(ip), "M": 4, "embedders": [{"label": "base", "name": "base", "selfcheck": False},
                                                                          {"label": "chal", "name": "chal", "selfcheck": False}]},
                                 LocalCtx(str(tmp_path / "c")))
    d = out["metrics"]["decision"]
    assert d["decision"] == "switch_to_challenger" and d["delta_recall_at_M"] >= 0.03
    # a live HF embedder is self-checked: a reference that disagrees fails the job
    monkeypatch.setattr(run_mod, "make_embedder", lambda name, **k: _emb())
    monkeypatch.setattr(run_mod, "reference_embeddings", lambda m, t, ml, rev=None: np.stack([_ref(x, term=IM_END) for x in t]))
    with pytest.raises(RuntimeError, match="self-check failed"):
        run_mod.embedder_check({"items_uri": str(ip), "embedders": [{"label": "x", "name": "x"}]}, LocalCtx(str(tmp_path / "d")))


class ResetChat(FakeChat):
    def __init__(self, policy):
        super().__init__(policy)
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1
        return {"success": True, "attempts": 1}


def test_sub_runs_reset_cache_manifest_and_ids(tmp_path, item):
    ip = _items_file(tmp_path, [item, make_item(seed=1)])
    ctx = LocalCtx(str(tmp_path / "o"), run_id="RUN1")
    chat = ResetChat(policy)
    out = run_system({"items_uri": str(ip), "systems": ["S2", "S1"], "repeats": 2, "embedder": "hash", "workers": 1,
                      "model": {"kind": "fake", "chat": chat}}, ctx)
    assert chat.resets == 4                                              # before every (system, repeat)
    resets = [e for e in ctx.events if e.get("action") == "prefix_cache_reset"]
    assert [e["sub_run_id"] for e in resets] == ["RUN1~S2~r0", "RUN1~S2~r1", "RUN1~S1~r0", "RUN1~S1~r1"]
    root = out["metrics"]
    assert root["repeat_kind"] == "same_container" and root["run_order"] == ["S2~r0", "S2~r1", "S1~r0", "S1~r1"]
    assert root["provenance"]["run_order"] == root["run_order"] and root["provenance"]["repeat_kind"] == "same_container"
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert man["run_id"] == "RUN1" and [r["sub_run_id"] for r in man["sub_runs"]] == ["RUN1~S2~r0", "RUN1~S2~r1", "RUN1~S1~r0", "RUN1~S1~r1"]
    for r in man["sub_runs"]:
        d = tmp_path / "o" / r["dir"]
        recs = [json.loads(x) for x in gzip.decompress((tmp_path / "o" / r["records"]).read_bytes()).decode().splitlines()]
        m = json.loads((tmp_path / "o" / r["metrics"]).read_text())
        assert {x["sub_run_id"] for x in recs} == {r["sub_run_id"]} == {m["sub_run_id"]} and m["repeat_kind"] == "same_container"
        assert {x["run_id"] for x in recs} == {"RUN1"} and r["n_records"] == len(recs)
        ev = [e for e in ctx.events if e.get("sub_run_id") == r["sub_run_id"] and "item_id" in e]
        assert ev and all(e["system"] == r["system"] and e["repeat"] == r["repeat"] for e in ev)


class _SeqHttp:
    def __init__(self, resps):
        self.resps, self.n = list(resps), 0

    def post(self, url, json):
        r = self.resps[min(self.n, len(self.resps) - 1)]
        self.n += 1
        return r


def test_reset_prefix_cache_parses_success_retries_and_raises():
    sleeps = []
    c = _chat(_Resp(200, '{"success": false}'))
    c._http = _SeqHttp([_Resp(200, '{"success": false}'), _Resp(200, '{"success": false}'), _Resp(200, '{"success": true}')])
    assert c.reset_prefix_cache(_sleep=sleeps.append) == {"success": True, "attempts": 3}
    assert sleeps == [1.0, 1.0] and c._http.n == 3
    allfalse = _chat(_Resp(200, '{"success": false}'))
    sleeps.clear()
    with pytest.raises(RuntimeError, match="did not succeed after 5 attempts"):
        allfalse.reset_prefix_cache(_sleep=sleeps.append)
    assert allfalse.http.bodies == [None] * 5 and len(sleeps) == 4
    with pytest.raises(RuntimeError, match="did not succeed"):                   # 200 without a success field is not success
        _chat(_Resp(200, "{}")).reset_prefix_cache(attempts=2, _sleep=lambda s: None)
    ok = _chat(_Resp(200, '{"success": true}'))
    assert ok.reset_prefix_cache()["attempts"] == 1 and ok.http.bodies == [None]
    with pytest.raises(RuntimeError, match="VLLM_SERVER_DEV_MODE"):
        _chat(_Resp(404, "not found")).reset_prefix_cache()


def test_vllm_started_in_dev_mode(monkeypatch):
    import subprocess

    import opa.tracks.track_a.run as run_mod
    seen = {}

    class P:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.update(kw=kw) or P())
    monkeypatch.setattr(run_mod, "resolve_revision", lambda *a, **k: "rev")
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: real_open("NUL" if str(f) == "/tmp/vllm.log" else f, *a, **k))
    import sys
    import types
    fake_httpx = types.SimpleNamespace(get=lambda *a, **k: type("R", (), {"status_code": 200})())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    run_mod._VLLM.pop("m-dev", None)
    run_mod.start_local_vllm("m-dev")
    assert seen["kw"]["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert seen["kw"]["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"             # no nvcc in the image: PyTorch sampler
    assert "vllm_ready_at" in run_mod._VLLM_PROV["m-dev"] and run_mod._VLLM_PROV["m-dev"]["vllm_start_seconds"] >= 0
    assert run_mod.stop_local_vllm() == ["m-dev"] and "m-dev" not in run_mod._VLLM


def test_n_items_zero_means_zero(tmp_path):
    from opa.tracks.track_a.run import select_items
    a = _items_file(tmp_path, [make_item(seed=s) for s in range(3)], "a.jsonl.gz")
    ctx = LocalCtx(str(tmp_path / "o"))
    assert select_items({"items_uri": str(a), "n_items": 0}, ctx) == []
    assert select_items({"items_uri": [{"uri": str(a), "n_items": 0}, str(a)], "n_items": 2}, ctx).__len__() == 2
    assert len(select_items({"items_uri": str(a)}, ctx)) == 3


def test_query_style_in_provenance_and_npz_meta(item):
    e = _emb(query_style="qwen3")
    assert e.provenance()["query_style"] == "qwen3"
    meta = json.loads(str(compute_item_embeddings([item], e)["meta"]))
    assert meta["query_style"] == "qwen3"


def test_embedder_check_dev_only_and_reference_before_model_load(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.embed as emb_mod
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    order = []
    monkeypatch.setattr(emb_mod, "reference_embeddings", lambda m, t, ml, rev=None: order.append("reference") or np.stack([_ref(x) for x in t]))
    monkeypatch.setattr(run_mod, "reference_embeddings", emb_mod.reference_embeddings)
    monkeypatch.setattr(run_mod, "make_embedder", lambda name, **k: order.append("load") or _emb())
    out = run_mod.embedder_check({"items_uri": str(ip), "embedders": [{"label": "x", "name": "x"}]}, LocalCtx(str(tmp_path / "c")))
    assert order == ["reference", "load"] and out["metrics"]["embedders"]["x"]["selfcheck_cosines"]
    test_item = dict(item, split="test")
    tp = _items_file(tmp_path, [test_item], "t.jsonl.gz")
    with pytest.raises(RuntimeError, match="dev split only"):
        run_mod.embedder_check({"items_uri": str(tp), "embedders": [{"label": "x", "name": "x"}]}, LocalCtx(str(tmp_path / "d")))
    with pytest.raises(RuntimeError, match="dev split only"):
        run_mod.embedder_check({"items_uri": str(ip), "embedders": [{"label": "x", "name": "x"}]},
                               LocalCtx(str(tmp_path / "e"), spec={"seed": 0, "split": "test"}))


class CounterChat(ResetChat):
    """Fake chat that logs llm_call events: the first request after a reset is uncached, later ones are cached."""

    def __init__(self, policy):
        super().__init__(policy)
        self.fresh = True

    def reset_prefix_cache(self):
        self.fresh = True
        return super().reset_prefix_cache()

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        out = self.policy(system, user)
        cached = 0 if self.fresh else 40
        self.fresh = False
        if usage is not None:
            usage.add(100, 5, cached)
        if log:
            log({"action": "llm_call", "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": cached}})
        return out


def test_cost_attribution_startup_and_reset_evidence(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    import time as _t
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: (_t.sleep(0.05), CounterChat(policy))[1])
    monkeypatch.setattr(run_mod, "_vllm_log_reset_lines", lambda n=3: ["INFO Resetting prefix cache"])
    ip = _items_file(tmp_path, [item, make_item(seed=1)])
    ctx = LocalCtx(str(tmp_path / "o"), run_id="R9")
    out = run_system({"items_uri": str(ip), "systems": ["S1", "S2"], "repeats": 2, "embedder": "hash", "workers": 1}, ctx)
    root = out["metrics"]
    assert root["server_startup_seconds"] >= 0.05
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert man["server_startup_seconds"] == root["server_startup_seconds"]
    for row in man["sub_runs"]:
        m = json.loads((tmp_path / "o" / row["metrics"]).read_text())
        for k in run_mod.COST_FIELDS:
            assert k in m and m[k] is not None and row[k] == m[k], k
        assert m["gpu_seconds_share"] == m["wall_seconds"] >= 0 and m["finished_at"] >= m["started_at"]
        recs = [json.loads(x) for x in gzip.decompress((tmp_path / "o" / row["records"]).read_bytes()).decode().splitlines()]
        assert m["tokens_in"] == sum(r["tokens_in"] for r in recs) and m["n_llm_calls"] == sum(r["n_llm_calls"] for r in recs)
        assert m["first_request_cached_tokens"] == 0
    resets = [e for e in ctx.events if e.get("action") == "prefix_cache_reset"]
    assert len(resets) == 4 and all(e["success"] is True and e["attempts"] == 1 for e in resets)
    ev = [e for e in ctx.events if e.get("action") == "prefix_cache_reset_evidence"]
    assert len(ev) == 4 and all(e["first_request_cached_tokens"] == 0 for e in ev)
    assert [e["sub_run_id"] for e in ev] == [e["sub_run_id"] for e in resets]
    assert ev[0]["server_log_lines"] == ["INFO Resetting prefix cache"] and all("server_log_lines" not in e for e in ev[1:])
    assert all("counters_before" not in e for e in ev)


def test_multi_requires_skeleton_uri_and_never_recomputes(tmp_path, item):
    ip = _items_file(tmp_path, [item])
    base = {"items_uri": str(ip), "embedder": "hash", "workers": 1, "model": {"kind": "fake", "chat": FakeChat(policy)}}
    with pytest.raises(ValueError, match="skeleton_llm_uri"):
        run_system({**base, "systems": ["S11", "S2"], "repeats": 1}, LocalCtx(str(tmp_path / "a")))
    # S2 needs no skeleton: fine without a uri. A missing file inside a sub-run errors even with compute_llm_skeleton.
    run_system({**base, "systems": ["S2"], "repeats": 1}, LocalCtx(str(tmp_path / "b")))
    with pytest.raises(RuntimeError, match="no S-llm skeleton"):
        run_system({**base, "systems": ["S11"], "repeats": 1, "skeleton_llm_uri": str(tmp_path / "none"),
                    "compute_llm_skeleton": True}, LocalCtx(str(tmp_path / "c")))


def test_host_flag():
    a = config.vllm_args("m")
    assert a[a.index("--host") + 1] == "127.0.0.1"


class CachedChat(CounterChat):
    """Fake chat that reports cached prompt tokens through Usage, like OpenAIChat does."""

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        out = self.policy(system, user)
        if usage is not None:
            usage.add(100, 5, 60)
        return out


def test_cached_tokens_per_record_and_sub_run(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: CachedChat(policy))
    ip = _items_file(tmp_path, [item, make_item(seed=1)])
    ctx = LocalCtx(str(tmp_path / "o"), run_id="RC")
    out = run_system({"items_uri": str(ip), "systems": ["S2"], "repeats": 2, "embedder": "hash", "workers": 1}, ctx)
    for r in ("S2/r0", "S2/r1"):
        m = out["metrics"]["runs"][r]
        assert m["cached_tokens"] == 60 * m["n_llm_calls"] > 0
        assert m["uncached_prompt_tokens"] == m["tokens_in"] - m["cached_tokens"]
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert all(row["cached_tokens"] > 0 and "uncached_prompt_tokens" in row for row in man["sub_runs"])
    recs = [json.loads(x) for x in gzip.decompress((tmp_path / "o" / "S2/r0/records.jsonl.gz").read_bytes()).decode().splitlines()]
    assert all(r["cached_tokens"] > 0 for r in recs)


def test_server_stopped_after_run_system_and_build_skeletons_even_on_error(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod

    class Proc:
        terminated = 0

        def poll(self):
            return None

        def terminate(self):
            Proc.terminated += 1

        def wait(self, timeout=None):
            pass

    ip = _items_file(tmp_path, [item])

    def fake_make_chat(a, c):
        run_mod._PROCS["fake-model"], run_mod._VLLM["fake-model"] = Proc(), "http://x"
        return FakeChat(policy)

    monkeypatch.setattr(run_mod, "make_chat", fake_make_chat)
    run_system({"items_uri": str(ip), "system": "S2", "embedder": "hash"}, LocalCtx(str(tmp_path / "a")))
    assert Proc.terminated == 1 and not run_mod._PROCS and "fake-model" not in run_mod._VLLM
    run_mod.build_skeletons({"items_uri": str(ip), "workers": 1}, LocalCtx(str(tmp_path / "b")))
    assert Proc.terminated == 2 and not run_mod._PROCS
    with pytest.raises(RuntimeError):                                    # a failing run still stops the server
        run_system({"items_uri": str(ip), "system": "S11", "embedder": "hash"}, LocalCtx(str(tmp_path / "c")))
    assert Proc.terminated == 3 and not run_mod._PROCS


def test_vllm_args_enable_prompt_tokens_details():
    assert "--enable-prompt-tokens-details" in config.vllm_args("m")


def test_every_make_chat_entrypoint_stops_the_server(tmp_path, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    import opa.tracks.track_a.thrash as thrash_mod
    import opa.tracks.track_a2.run as a2_mod

    class Proc:
        n = 0

        def poll(self):
            return None

        def terminate(self):
            Proc.n += 1

        def wait(self, timeout=None):
            pass

    def started(a, c):
        run_mod._PROCS["m"], run_mod._VLLM["m"] = Proc(), "http://x"
        return FakeChat(lambda s, u: "{}")

    monkeypatch.setattr(thrash_mod, "make_chat", started)
    monkeypatch.setattr(a2_mod, "make_chat", started)
    # run_thrash: items load fails after/before the server starts -> still stopped; here items are empty so it runs through
    ip = _items_file(tmp_path, [], "empty.jsonl.gz")
    try:
        thrash_mod.run_thrash({"items_uri": str(ip), "M": 2, "model": {"kind": "fake", "chat": FakeChat(policy)}}, LocalCtx(str(tmp_path / "t")))
    except Exception:
        pass
    assert Proc.n == 1 and not run_mod._PROCS
    try:
        a2_mod.run_condition({"items_uri": str(ip), "condition": "D1", "model": "qwen"}, LocalCtx(str(tmp_path / "a")))
    except Exception:
        pass
    assert Proc.n == 2 and not run_mod._PROCS
    with pytest.raises(ValueError):                                       # an error before any server start is harmless too
        a2_mod.run_condition({"items_uri": str(ip), "condition": "D9"}, LocalCtx(str(tmp_path / "b")))
    assert Proc.n == 2


def test_embed_items_passes_dtype_batch_and_threshold(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.embed as emb_mod
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    seen = {}

    def fake_make(name, **kw):
        seen.update(kw)
        e = _emb(dtype=kw["dtype"], query_style="qwen3")
        if kw.get("batch"):
            e.batch = kw["batch"]
        return e

    monkeypatch.setattr(run_mod, "make_embedder", fake_make)

    def noisy_ref(m, t, ml, rev=None):                        # cosine ~0.995 to ours: passes at 0.99, fails at 0.999
        r = np.stack([_ref(x) for x in t])
        n = np.zeros_like(r)
        n[:, 0] = 0.1
        r = r + n
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    monkeypatch.setattr(emb_mod, "reference_embeddings", noisy_ref)
    out = run_mod.embed_items({"items_uri": str(ip), "embedder": "Qwen/Qwen3-Embedding-4B", "dtype": "bfloat16", "batch": 4},
                              LocalCtx(str(tmp_path / "bf")))
    assert seen["dtype"] == "bfloat16" and seen["batch"] == 4
    prov = out["metrics"]["provenance"]
    assert prov["dtype"] == "bfloat16" and prov["query_style"] == "qwen3"
    with pytest.raises(RuntimeError, match="self-check failed"):            # float32 keeps the strict 0.999
        run_mod.embed_items({"items_uri": str(ip), "embedder": "x", "dtype": "float32"}, LocalCtx(str(tmp_path / "f32")))
    with pytest.raises(ValueError, match="dtype"):
        HFEmbedder("stub", device="cpu", tokenizer=StubTok(), encoder=StubModel(), dtype="int8")


def test_precomputed_fallback_uses_meta_dtype_and_query_style(monkeypatch, tmp_path, item):
    import opa.tracks.track_a.run as run_mod
    arrays = compute_item_embeddings([item], _emb(dtype="bfloat16", query_style="qwen3"))
    p = tmp_path / "e.npz"
    write_embeddings(p, arrays)
    built = {}

    class Spy(HFEmbedder):
        def __init__(self, model, **kw):
            built.update(kw)
            super().__init__(model, tokenizer=StubTok(), encoder=StubModel(), **kw)

    monkeypatch.setattr(run_mod, "HFEmbedder", Spy)
    pe = run_mod.make_embedder(f"precomputed:{p}")
    pe.fallback()
    assert built["dtype"] == "bfloat16" and built["query_style"] == "qwen3" and built["device"] == "cpu"


def test_stop_local_vllm_closes_the_log_handle():
    import opa.tracks.track_a.run as run_mod

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    class FH:
        closed = False

        def close(self):
            FH.closed = True

    run_mod._PROCS["lm"], run_mod._LOGS["lm"], run_mod._VLLM["lm"] = Proc(), FH(), "u"
    assert run_mod.stop_local_vllm() == ["lm"] and FH.closed and "lm" not in run_mod._LOGS


def test_bucket_is_normalised_from_config_and_job(monkeypatch, tmp_path, item):
    import yaml

    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    monkeypatch.setattr(yaml, "safe_load", lambda text: {"gcp": {"bucket": "gs://YOUR_BUCKET/"}})
    assert run_mod.config_bucket() == "YOUR_BUCKET"
    arrays = compute_item_embeddings([item], HashEmbedder())
    seen = []
    monkeypatch.setattr(gcs, "get_bytes", lambda uri: seen.append(uri) or embeddings_to_bytes(arrays))
    make_embedder("precomputed:gs://{bucket}/opa/x.npz")                         # from config
    make_embedder("precomputed:gs://{bucket}/opa/x.npz", bucket="gs://YOUR_BUCKET/")   # from a job spec
    assert seen == ["gs://YOUR_BUCKET/opa/x.npz"] * 2


class ProbeChat(CounterChat):
    """Fake server cache: probe_cache reports cached tokens while the probe prefix is cached; reset clears it unless broken."""

    def __init__(self, policy, broken=False, confirm=True):
        super().__init__(policy)
        self.cached, self.broken, self.confirm, self.probes = False, broken, confirm, 0

    def probe_cache(self):
        self.probes += 1
        got = 96 if self.cached else 0
        self.cached = True
        return {"prompt_tokens": 120, "cached_tokens": got}

    def reset_prefix_cache(self):
        super().reset_prefix_cache()
        if not self.broken:
            self.cached = False
        return {"success": self.confirm, "attempts": 2}


def test_probe_evidence_in_reset_event_and_raises_when_reset_did_not_take(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    base = {"items_uri": str(ip), "systems": ["S2"], "repeats": 2, "embedder": "hash", "workers": 1}
    good = ProbeChat(policy)
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: good)
    ctx = LocalCtx(str(tmp_path / "ok"))
    run_system(base, ctx)
    resets = [e for e in ctx.events if e.get("action") == "prefix_cache_reset"]
    assert len(resets) == 2 and good.probes == 6                                # 3 probes per reset
    assert all(e["probe_informative"] is True and e["probe_cached_tokens_before"] == 96 and e["probe_cached_tokens_after"] == 0 and e["probe_prompt_tokens"] >= 64
               and e["success"] is True and e["attempts"] == 2 for e in resets)
    ev = [e for e in ctx.events if e.get("action") == "prefix_cache_reset_evidence"]
    assert all(e["probe_cached_tokens_after"] == 0 and e["first_request_prompt_tokens"] == 100
               and e["first_request_cached_tokens"] == 0 for e in ev) and len(ev) == 2
    bad = ProbeChat(policy, broken=True)
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: bad)
    with pytest.raises(RuntimeError, match="reset did not take"):
        run_system(base, LocalCtx(str(tmp_path / "bad")))


def test_probe_cache_http_request_shape():
    ok = _Resp(200, {"choices": [], "usage": {"prompt_tokens": 150, "completion_tokens": 1,
                                              "prompt_tokens_details": {"cached_tokens": 144}}})
    c = _chat(ok)
    assert c.probe_cache() == {"prompt_tokens": 150, "cached_tokens": 144}
    b = c.http.bodies[0]
    assert b["max_tokens"] == 1 and b["temperature"] == 0 and len(b["messages"][1]["content"].split()) >= 64
    assert b["messages"][0]["content"].startswith("PROBE-")
    with pytest.raises(RuntimeError, match="probe failed"):
        _chat(_Resp(500, "x")).probe_cache()


def test_fallback_selfcheck_threshold_follows_meta_dtype():
    class Slightly(HashEmbedder):
        def embed(self, texts):
            v = HashEmbedder.embed(self, texts)
            v[:, 0] += 0.1
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    for dtype, should_pass in (("float32", False), ("bfloat16", True)):
        e = _emb(dtype=dtype)
        arrays = compute_item_embeddings([make_item(seed=3)], HashEmbedder())
        meta = json.loads(str(arrays["meta"]))
        meta["dtype"] = dtype
        arrays["meta"] = np.array(json.dumps(meta))
        stored = HashEmbedder().embed(SELFCHECK_TEXTS)
        cos = float((stored * Slightly().embed(SELFCHECK_TEXTS)).sum(1).min())
        assert 0.99 < cos < 0.999, cos
        pe = PrecomputedEmbedder(embeddings_to_bytes(arrays), fallback=lambda: Slightly())
        pe.selfcheck_emb = stored
        if should_pass:
            pe._fb()
        else:
            with pytest.raises(RuntimeError, match="self-check failed"):
                pe._fb()


class FlakyChat(ProbeChat):
    """The server stops accepting cache resets after `ok_resets` of them (a failure that propagates out of the loop)."""

    def __init__(self, policy, ok_resets):
        super().__init__(policy)
        self.left = ok_resets

    def reset_prefix_cache(self):
        if self.left <= 0:
            raise ConnectionError("server died")
        self.left -= 1
        return super().reset_prefix_cache()


def test_partial_manifest_written_on_failure(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    # the first sub-run (S2 r0) completes; the reset before S2 r1 fails
    chat = FlakyChat(policy, ok_resets=1)
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: chat)
    ctx = LocalCtx(str(tmp_path / "o"), run_id="RF")
    with pytest.raises(ConnectionError):
        run_system({"items_uri": str(ip), "systems": ["S2"], "repeats": 3, "embedder": "hash", "workers": 1}, ctx)
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert man["complete"] is False and [r["sub_run_id"] for r in man["sub_runs"]] == ["RF~S2~r0"]
    assert man["failure"]["sub_run_id"] == "RF~S2~r1" and man["failure"]["system"] == "S2" and man["failure"]["repeat"] == 1
    assert "ConnectionError" in man["failure"]["error"] and "server died" in man["failure"]["error"]
    root = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert root["complete"] is False and list(root["runs"]) == ["S2/r0"] and root["failure"] == man["failure"]
    assert (tmp_path / "o" / "S2" / "r0" / "metrics.json").exists()


def test_successful_multi_run_is_marked_complete(tmp_path, item):
    ip = _items_file(tmp_path, [item])
    run_system({"items_uri": str(ip), "systems": ["S2"], "repeats": 2, "embedder": "hash",
                "model": {"kind": "fake", "chat": FakeChat(policy)}}, LocalCtx(str(tmp_path / "o")))
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert man["complete"] is True and man["failure"] is None


def test_uninformative_probe_logs_event_and_raises(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod

    class NoCache(ProbeChat):
        def probe_cache(self):
            self.probes += 1
            return {"prompt_tokens": 120, "cached_tokens": 0}

    ip = _items_file(tmp_path, [item])
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: NoCache(policy))
    ctx = LocalCtx(str(tmp_path / "o"))
    with pytest.raises(RuntimeError, match="not informative"):
        run_system({"items_uri": str(ip), "systems": ["S2"], "repeats": 1, "embedder": "hash", "workers": 1}, ctx)
    ev = [e for e in ctx.events if e.get("action") == "prefix_cache_reset"]
    assert len(ev) == 1 and ev[0]["probe_informative"] is False and ev[0]["probe_cached_tokens_before"] == 0
    man = json.loads((tmp_path / "o" / "sub_runs.json").read_text())
    assert man["complete"] is False and man["failure"]["sub_run_id"].endswith("~S2~r0") and man["sub_runs"] == []


def _fake_boot(monkeypatch, tmp_path, log_text, exit_code=1):
    """Stub the vLLM start: Popen writes `log_text` to the (redirected) log and the process has already exited."""
    import builtins
    import subprocess
    import sys
    import types

    import opa.tracks.track_a.run as run_mod
    log_path = tmp_path / "vllm.log"
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: real_open(log_path if str(f) == "/tmp/vllm.log" else f, *a, **k))
    monkeypatch.setattr(run_mod, "vllm_failure_message", lambda why, log_path_=None, _f=run_mod.vllm_failure_message:
                        _f(why, str(log_path)))

    class P:
        def poll(self):
            return exit_code

    def popen(cmd, stdout=None, **kw):
        stdout.write(log_text)
        stdout.flush()
        return P()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(run_mod, "resolve_revision", lambda *a, **k: "rev")
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=lambda *a, **k: None))
    run_mod._VLLM.pop("boom-model", None)
    return run_mod


def test_vllm_start_failure_uploads_full_log_and_reports_root_cause(tmp_path, monkeypatch):
    noise = "".join(f"INFO line {i}\n" for i in range(500))
    log = noise + "ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n" + noise
    run_mod = _fake_boot(monkeypatch, tmp_path, log)
    ctx = LocalCtx(str(tmp_path / "o"))
    with pytest.raises(RuntimeError) as ei:
        run_mod.start_local_vllm("boom-model", ctx=ctx)
    msg = str(ei.value)
    assert "exited with code 1" in msg and "CUDA out of memory" in msg and msg.count("OutOfMemoryError") >= 1
    assert "last 1500 characters" in msg and len(msg) < 1500 + 40 * 420 + 500
    saved = (tmp_path / "o" / "vllm.log").read_text()
    assert saved == log                                                     # the FULL log, not a tail
    kinds = [e["action"] for e in ctx.events]
    assert kinds == ["vllm_preflight", "vllm_failed"] and not run_mod._PROCS and not run_mod._LOGS


def test_error_lines_capped_at_40(tmp_path, monkeypatch):
    run_mod = _fake_boot(monkeypatch, tmp_path, "".join(f"Error number {i}\n" for i in range(100)))
    with pytest.raises(RuntimeError) as ei:
        run_mod.start_local_vllm("boom-model")
    assert "Error number 39" in str(ei.value) and "Error number 40" not in str(ei.value).split("last 1500")[0]
    head = str(ei.value).split("--- last 1500")[0]
    assert head.count("Error number") == 40


def test_preflight_info_and_events_exist_before_a_failing_server_start(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    info = run_mod.gpu_env_info()
    assert {"vllm_version", "torch_version", "device_name", "free_bytes", "total_bytes"} <= set(info)

    def boom(args, ctx):
        ctx.log({"action": "vllm_preflight", "model": "m"})
        raise RuntimeError("Engine core initialization failed")

    monkeypatch.setattr(run_mod, "make_chat", boom)
    ip = _items_file(tmp_path, [item])
    ctx = LocalCtx(str(tmp_path / "o"))
    with pytest.raises(RuntimeError, match="Engine core"):
        run_system({"items_uri": str(ip), "system": "S2", "embedder": "hash"}, ctx)
    assert [e["action"] for e in ctx.events][:2] == ["job_start", "vllm_preflight"]      # events exist for the flush in dispatch


def test_server_env_overrides_and_preflight_records_them(tmp_path, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    assert run_mod.VLLM_ENV == {"VLLM_SERVER_DEV_MODE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    r = _fake_boot(monkeypatch, tmp_path, "boom\n")
    ctx = LocalCtx(str(tmp_path / "o"))
    with pytest.raises(RuntimeError):
        r.start_local_vllm("boom-model", ctx=ctx)
    assert ctx.events[0]["action"] == "vllm_preflight" and ctx.events[0]["server_env"] == run_mod.VLLM_ENV


# ---------------------------------------------------------------- G0 findings: answer parsing (real replies from the logs)
BAD_QUOTES = '{"answer": 0, "cited_pin_ids": ["<origin id="16>"]}'
RUNAWAY = '{"answer": "57 bar", "cited_pin_ids": [' + ", ".join(f'"<origin id=\\"{i}\\">"' for i in range(38, 80))[:700]


class Replies(FakeChat):
    def __init__(self, replies):
        super().__init__(lambda s, u: "")
        self.replies, self.calls = list(replies), []

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        self.calls.append(user)
        out = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if usage is not None:
            usage.add(10, 5)
        return out


def _s1(item, replies):
    chat = Replies(replies)
    rec, events = _run_with(chat, "S1", item)
    return rec, chat, events


def test_unescaped_quotes_reply_is_salvaged_not_retried(item):
    rec, chat, _ = _s1(item, [BAD_QUOTES, '{"answer": "WRONG", "cited_pin_ids": []}'])
    assert len(chat.calls) == 1                                   # the usable first answer is kept, no second call
    assert rec["prediction"] == "0" and rec["answer_parse"] == "answer_salvaged"


def test_truncated_runaway_reply_keeps_its_answer(item):
    rec, chat, _ = _s1(item, [RUNAWAY, '{"answer": "other"}'])
    assert len(chat.calls) == 1 and rec["prediction"] == "57 bar" and rec["answer_parse"] == "answer_salvaged"


def test_numeric_zero_and_false_are_answers_not_empty(item):
    rec, chat, _ = _s1(item, ['{"answer": 0, "cited_pin_ids": []}'])
    assert rec["prediction"] == "0" and "answer_parse" not in rec and len(chat.calls) == 1
    rec, _, _ = _s1(item, ['{"answer": false}'])
    assert rec["prediction"] == "False"


def test_retry_used_when_nothing_is_readable_and_empty_is_a_failure_code(item):
    rec, chat, _ = _s1(item, ["I think it is 42", '{"answer": 0, "cited_pin_ids": []}'])
    assert len(chat.calls) == 2 and "valid JSON object" in chat.calls[1]
    assert rec["prediction"] == "0" and rec["answer_parse"] == "answer_retried"
    rec, chat, ev = _s1(item, ["no json", "still none"])
    assert rec["prediction"] == "" and rec["answer_parse"] == "answer_unparseable" and len(chat.calls) == 2
    assert any(e.get("action") == "parse_error" for e in ev)
    rec, chat, _ = _s1(item, ['{"answer": null, "cited_pin_ids": []}'])
    assert rec["prediction"] == "" and rec["answer_parse"] == "answer_empty" and len(chat.calls) == 1
    rec, chat, _ = _s1(item, ["garbage", BAD_QUOTES])
    assert rec["prediction"] == "0" and rec["answer_parse"] == "answer_salvaged_retry"


def test_salvage_answer_values():
    from opa.tracks.track_a.models import MISSING, salvage_answer
    assert salvage_answer('{"answer": "9\\" pipe", "x": [') == '9" pipe'
    assert salvage_answer('{"answer": 12.5, "cited') == 12.5 and salvage_answer('{"answer": "cut') is MISSING
    assert salvage_answer("no field") is MISSING and salvage_answer('{"answer": null') is MISSING     # no delimiter yet
    assert salvage_answer('{"answer": null, "x"') is None


def test_server_env_and_prefix_caching_options_reach_the_server(tmp_path, monkeypatch):
    a = config.vllm_args("m", prefix_caching=False)
    assert "--no-enable-prefix-caching" in a and "--enable-prefix-caching" not in a
    assert "--enable-prefix-caching" in config.vllm_args("m")
    r = _fake_boot(monkeypatch, tmp_path, "boom\n")
    ctx = LocalCtx(str(tmp_path / "o"))
    seen = {}
    import subprocess
    orig = subprocess.Popen

    def spy(cmd, **kw):
        seen.update(cmd=cmd, env=kw["env"])
        return orig(cmd, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    with pytest.raises(RuntimeError):
        r.start_local_vllm("boom-model", ctx=ctx, env_extra={"VLLM_BATCH_INVARIANT": 1}, prefix_caching=False)
    assert seen["env"]["VLLM_BATCH_INVARIANT"] == "1" and "--no-enable-prefix-caching" in seen["cmd"]
    pre = ctx.events[0]
    assert pre["server_env"]["VLLM_BATCH_INVARIANT"] == "1" and pre["prefix_caching"] is False


def test_salvage_strictness_cases():
    from opa.tracks.track_a.models import MISSING, parse_json, salvage_answer
    ok = {
        '{"answer": "Ostrand Ltd.", "cited_pin_ids": ["<origin id="16>"]}': "Ostrand Ltd.",
        '{"answer": "57 bar", "cited_pin_ids": ["p1", "p2", "p3", "p': "57 bar",
        '{"note": "he said \\"answer\\": \\"wrong\\"", "answer": "right", "x": [': "right",
        '{"meta": {"answer": "inner"}, "answer": "outer", "x": "a"b"}': "outer",
        '{"answer": null, "cited_pin_ids": ["<origin id="3>"]}': None,
        'The answer: 5. {"answer": "6", "cited_pin_ids": [': "6",
        '{"answer": 57, "cited_pin_ids": ["a"': 57,
        '{"answer": true}': True,
    }
    for text, want in ok.items():
        assert salvage_answer(text) == want and salvage_answer(text) is not MISSING, text
    refuse = ['{"answer": 57 bar, "cited_pin_ids": []}', '{"cited_pin_ids": ["p1","p2"], "answer": 57',
              '{"answer_found": true, "x": "y"', '{"note": "quote "answer": "wrong" end", "answer": "right"}',
              '{"answer": "57 b', '{"answer": 12abc}', '{"answer": nullable}', '{"answer": "a", "answer": "b", "x": [',
              '{"answer": 5', '{"meta": {"answer": "inner"}, "x": "a"b"}', "no braces: answer 5", '{"answer": }']
    for text in refuse:
        assert salvage_answer(text) is MISSING, text


def test_unsalvageable_unit_answer_is_retried(item):
    rec, chat, _ = _s1(item, ['{"answer": 57 bar, "cited_pin_ids": []}', '{"answer": "57 bar", "cited_pin_ids": []}'])
    assert len(chat.calls) == 2 and rec["prediction"] == "57 bar" and rec["answer_parse"] == "answer_retried"


def test_s3_in_loop_answer_is_salvaged_and_summarize_counts(item):
    from opa.tracks.track_a.run import summarize
    chat = Replies([BAD_QUOTES])
    rec, _ev = _run_with(chat, "S3", item)
    assert len(chat.calls) == 1 and rec["prediction"] == "0" and rec["answer_parse"] == "answer_salvaged"
    recs = [rec, dict(rec, answer_parse="answer_retried", cited_pin_ids=["1"]), dict(rec, answer_parse="answer_unparseable", prediction=""),
            dict(rec, answer_parse=None), dict(rec, answer_parse="answer_empty", prediction="")]
    m = summarize([item] * 0 + [dict(item, item_id=r["item_id"]) for r in recs[:1]], recs[:1], "S3")
    assert m["answer_parse"] == {"salvaged": 1, "salvaged_retry": 0, "retried": 0, "unparseable": 0, "empty": 0}
    assert m["answer_salvaged_without_citations"] == 1
    m = summarize([dict(item)], [dict(r, item_id=item["item_id"]) for r in recs], "S3")
    assert m["answer_parse"] == {"salvaged": 1, "salvaged_retry": 0, "retried": 1, "unparseable": 1, "empty": 1}
    assert m["answer_salvaged_without_citations"] == 1


def test_prefix_caching_off_skips_reset_and_probe_and_provenance(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    chat = ProbeChat(policy)
    chat.provenance = {"prefix_caching": False, "server_env": {"VLLM_BATCH_INVARIANT": "1"}}
    monkeypatch.setattr(run_mod, "make_chat", lambda a, c: chat)
    ip = _items_file(tmp_path, [item])
    ctx = LocalCtx(str(tmp_path / "o"))
    out = run_system({"items_uri": str(ip), "systems": ["S2"], "repeats": 2, "embedder": "hash", "workers": 1}, ctx)
    assert chat.probes == 0 and chat.resets == 0
    ev = [e for e in ctx.events if e.get("action") == "prefix_cache_reset"]
    assert len(ev) == 2 and all(e["skipped"] == "prefix caching disabled" for e in ev)
    assert not [e for e in ctx.events if e.get("action") == "prefix_cache_reset_evidence"]
    prov = out["metrics"]["provenance"]["chat"]
    assert prov["prefix_caching"] is False and prov["server_env"]["VLLM_BATCH_INVARIANT"] == "1"


def test_vllm_prov_records_prefix_caching_env_and_extra_args(tmp_path, monkeypatch):
    r = _fake_boot(monkeypatch, tmp_path, "boom\n")
    with pytest.raises(RuntimeError):
        r.start_local_vllm("boom-model", env_extra={"VLLM_BATCH_INVARIANT": 1}, prefix_caching=False, extra_args=["--x"])
    pv = r._VLLM_PROV["boom-model"]
    assert pv["prefix_caching"] is False and pv["server_env"]["VLLM_BATCH_INVARIANT"] == "1" and pv["extra_args"] == ["--x"]


# ---------------------------------------------------------------- micro-pilot finding: early stop on answer_found
def _always_found(system, user):
    if "Copy each quote" in system or "notes" in system.lower() and "answer_found" in system:
        return json.dumps({"pins": [], "notes": [], "answer_found": True})
    return json.dumps({"answer": "x", "cited_pin_ids": []})


def test_pure_folds_read_every_origin_even_if_the_model_claims_answer_found(item):
    n = len(item["origins"])
    for system in ("S6", "S7", "S8", "S9", "S10", "S11"):
        rec, events = _run_with(FakeChat(_always_found), system, item)
        assert sorted(set(rec["paged_in"])) == list(range(n)), (system, len(set(rec["paged_in"])), n)
        assert not [e for e in events if e["action"] == "early_stop"] and rec["stop_reason"] == "exhausted", system


def test_answer_found_never_stops_reading_by_default_but_the_old_behaviour_is_an_option(item):
    n = len(item["origins"])
    for system in ("S12", "S16", "S18", "S11"):
        rec, events = _run_with(FakeChat(_always_found), system, item)
        assert rec["stop_reason"] == "exhausted" and sorted(set(rec["paged_in"])) == list(range(n)), system
        assert not [e for e in events if e["action"] == "early_stop"]
    rec, events = _run_with(FakeChat(_always_found), "S12", item, opts={"early_stop_on_answer_found": True})
    assert rec["stop_reason"] == "early_answer_found" and len(set(rec["paged_in"])) < n
    assert [e["reason"] for e in events if e["action"] == "early_stop"] == ["early_answer_found"]
    # pure folds ignore the option
    rec, _ = _run_with(FakeChat(_always_found), "S7", item, opts={"early_stop_on_answer_found": True})
    assert rec["stop_reason"] == "exhausted"


CHAIN = [("Varrel Pumps sources seals from Ostrand Ltd.", "Varrel Pumps", ["Ostrand Ltd."]),
         ("Ostrand Ltd. makes its seals from Kelvane.", "Ostrand Ltd.", ["Kelvane"]),
         ("Kelvane seals degrade above 83 \u00b0C in ester coolant.", "Kelvane", [])]


def _pins_a_value(system, user):
    """A reader that pins the three chain sentences of the test item as soon as they are shown, with their reveals."""
    import re as _re
    if "Copy each quote" in system:
        pins = []
        for m in _re.finditer(r'<origin id="(\d+)">\n(.*?)\n</origin>', user, _re.S):
            for quote, ent, rev in CHAIN:
                if quote in m.group(2):
                    pins.append({"origin_id": int(m.group(1)), "quote": quote, "entity": ent, "reveals": rev})
        return json.dumps({"pins": pins, "answer_found": False})
    return json.dumps({"answer": "x", "cited_pin_ids": []})


def test_structural_chain_pinned_stop_and_stop_reason_counts(item):
    from opa.tracks.track_a.run import summarize
    n = len(item["origins"])
    rec, events = _run_with(FakeChat(_pins_a_value), "S12", item)
    assert rec["stop_reason"] == "early_chain_pinned" and len(set(rec["paged_in"])) < n
    assert [e["reason"] for e in events if e["action"] == "early_stop"] == ["early_chain_pinned"]
    rec7, _ = _run_with(FakeChat(_pins_a_value), "S7", item)                     # a pure fold ignores it
    assert rec7["stop_reason"] == "exhausted" and sorted(set(rec7["paged_in"])) == list(range(n))
    recs = [dict(rec, stop_reason="exhausted"), dict(rec, stop_reason="exhausted"), dict(rec, stop_reason="budget"),
            dict(rec, stop_reason="early_chain_pinned"), {k: v for k, v in rec.items() if k != "stop_reason"}]
    m = summarize([dict(item)], [dict(r, item_id=item["item_id"]) for r in recs], "S12")
    assert m["stop_reasons"] == {"exhausted": 2, "budget": 1, "early_chain_pinned": 1}


# ---------------------------------------------------------------- structural chain-pinned condition
def _fold_with_pins(item, pins, R):
    """A descent-first Fold whose live pins and relevant entities are set directly (no model calls).
    pins: [(quote, label, reveals)]"""
    from opa.tracks.track_a.index import Session
    from opa.tracks.track_a.systems import Ctx, Fold, Pin, session_cfg
    emb = HashEmbedder()
    sess = Session(item, session_cfg("S12", {}), emb, skeleton_llm={})
    c = Ctx(item=item, sess=sess, chat=FakeChat(lambda s, u: "{}"), opts={"_q_emb": emb.embed_query([item["question"]])[0],
                                                                         "_embedder": emb}, log_fn=lambda ev: None)
    f = Fold(c, "S12")
    f.R = list(R)
    from opa.tracks.track_a.systems import Pin
    f.pins = [Pin(f"p{i + 1}", 0, [0, 1], q, ent, list(rev)) for i, (q, ent, rev) in enumerate(pins)]
    return f


GENUINE = [("Varrel Pumps sources seals from Ostrand Ltd.", "Varrel Pumps", ["Ostrand Ltd."]),
           ("Ostrand Ltd. makes its seals from Kelvane.", "Ostrand Ltd.", ["Kelvane"]),
           ("Kelvane seals degrade above 83 \u00b0C in ester coolant.", "Kelvane", [])]
GENUINE_R = ["varrel pumps", "ostrand ltd", "kelvane"]


def test_chain_pinned_fires_on_a_genuine_full_chain(item):
    f = _fold_with_pins(item, GENUINE, GENUINE_R)
    ok, det = f.chain_pinned()
    assert ok and det["value_pinned"] and det["leaf"] == "kelvane" and det["leaf_pin"] == "p3"
    assert det["covered"]["varrel pumps"] == {"pin_id": "p1", "by": "label"}
    assert det["covered"]["ostrand ltd"] == {"pin_id": "p2", "by": "label"}
    assert f.stop_reason(False) == "early_chain_pinned" and f.stop_coverage["covered"] == det["covered"]
    # the legal form may be dropped from a label; covering by quote also works when the pin does not reveal the entity
    g = _fold_with_pins(item, [GENUINE[0], ("Ostrand makes seals. Seals from Kelvane.", "seals", ["Kelvane"]),
                               ("Kelvane seals degrade above 83 \u00b0C.", "Kelvane", [])], GENUINE_R)
    assert g.chain_coverage()["ostrand ltd"] is None                     # "Ostrand" alone is a bare core name: no
    h = _fold_with_pins(item, [GENUINE[0], ("Ostrand Ltd. makes seals. Seals from Kelvane.", "seals", ["Kelvane"]),
                               ("Kelvane seals degrade above 83 \u00b0C.", "Kelvane", [])], GENUINE_R)
    assert h.chain_coverage()["ostrand ltd"] == {"pin_id": "p2", "by": "quote"} and h.chain_pinned()[0]
    k = _fold_with_pins(item, [GENUINE[0], ("Ostrand makes seals.", "Ostrand", ["Kelvane"]), GENUINE[2]], GENUINE_R)
    assert k.chain_coverage()["ostrand ltd"] == {"pin_id": "p2", "by": "label"}      # label "Ostrand" = Ostrand Ltd


def test_revealer_pin_with_a_value_does_not_stop_after_hop_one(item):
    f = _fold_with_pins(item, [("Varrel Pumps sources its seals from Ostrand Ltd. under part QX-2291 rated to 83 degrees.",
                                "Varrel Pumps", ["Ostrand Ltd."])], ["varrel pumps", "ostrand ltd"])
    ok, det = f.chain_pinned()
    assert not ok and det["covered"]["ostrand ltd"] is None            # the revealer mentions it but may not cover it
    assert f.stop_reason(False) is None


def test_distractor_pin_with_a_value_does_not_stop(item):
    # all entities covered, but the only number sits in a distractor pin, not in the pin of the leaf
    f = _fold_with_pins(item, [GENUINE[0], GENUINE[1],
                               ("Kelvane was founded in a 12 km valley.", "Kelvane", ["Hallum Works"]),
                               ("Hallum Works is far away.", "Hallum Works", [])],
                        ["varrel pumps", "ostrand ltd", "kelvane", "hallum works"])
    ok, det = f.chain_pinned()
    assert not ok and not det["value_pinned"] and det["leaf"] == "hallum works"
    # a question-entity-only R with any value never fires (single entity cannot show a chain)
    g = _fold_with_pins(item, [("Varrel Pumps was founded in 1990 and employs 40 people at a 12 km site.", "Varrel Pumps", [])],
                        ["varrel pumps"])
    assert not g.chain_pinned()[0]
    # value only in a non-leaf pin
    h = _fold_with_pins(item, [("Varrel Pumps sources seals from Ostrand Ltd. rated at 5 bar.", "Varrel Pumps", ["Ostrand Ltd."]),
                               ("Ostrand Ltd. makes its seals.", "Ostrand Ltd.", [])], ["varrel pumps", "ostrand ltd"])
    assert not h.chain_pinned()[0]


def test_suffix_decoy_and_one_character_twin_are_not_the_entity(item):
    chain = [("Varrel Pumps buys from Zorrin Works.", "Varrel Pumps", ["Zorrin Works"])]
    decoy = _fold_with_pins(item, chain + [("Zorrin Works rates its seal at 57 bar.", "Zorrin Works", [])],
                            ["varrel pumps", "zorrin works"])
    assert decoy.chain_pinned()[0]                                            # a genuine two-hop chain fires
    wrong = _fold_with_pins(item, [("Kelvane Works rates its seal at 57 bar.", "Kelvane Works", [])], ["kelvane pumps"])
    assert wrong.chain_coverage()["kelvane pumps"] is None                    # "Kelvane Works" is not "Kelvane Pumps"
    mid = _fold_with_pins(item, [("Varrel Pumps sources seals from Kelvane Pumps.", "Varrel Pumps", ["Kelvane Pumps"]),
                                 ("Kelvane Works rates its seal at 57 bar.", "Kelvane Works", [])],
                          ["varrel pumps", "kelvane pumps"])
    assert not mid.chain_pinned()[0] and mid.chain_coverage()["kelvane pumps"] is None
    twin = _fold_with_pins(item, [("Varrel Pumps sources seals from Kelvane Pumps.", "Varrel Pumps", ["Kelvane Pumps"]),
                                  ("Kelvine Pumps rates its seal at 57 bar.", "Kelvine Pumps", [])],
                           ["varrel pumps", "kelvane pumps"])
    assert not twin.chain_pinned()[0] and twin.chain_coverage()["kelvane pumps"] is None
    # a bare core name in a quote is not a match, nor is a one-character-longer word
    bare = _fold_with_pins(item, [("Ostrand rates its seal at 57 bar.", "x", [])], ["ostrand ltd"])
    assert bare.chain_coverage()["ostrand ltd"] is None
    longer = _fold_with_pins(item, [("Ostrande Ltd. rates its seal at 57 bar.", "x", [])], ["ostrand ltd"])
    assert longer.chain_coverage()["ostrand ltd"] is None


def test_early_stop_event_carries_the_coverage_details(item):
    rec, events = _run_with(FakeChat(_pins_a_value), "S12", item)
    ev = [e for e in events if e["action"] == "early_stop"]
    assert len(ev) == 1 and ev[0]["reason"] == "early_chain_pinned" and rec["stop_reason"] == "early_chain_pinned"
    cov = ev[0]["coverage"]
    assert cov["value_pinned"] and cov["leaf"] == "kelvane" and all(v and v["pin_id"] for v in cov["covered"].values())


# ---------------------------------------------------------------- the chain walk from the question entity (rescuer fix)
TRUE3 = [("Varrel Pumps buys its seals from Ostrand Ltd.", "Varrel Pumps", ["Ostrand Ltd."]),
         ("Ostrand Ltd. sources the seal compound from Kelvane Fluidics.", "Ostrand Ltd.", ["Kelvane Fluidics"]),
         ("Kelvane Fluidics rates the compound to 57 bar.", "Kelvane Fluidics", [])]
TRUE3_R = ["varrel pumps", "ostrand ltd", "kelvane fluidics"]


def test_off_chain_leaf_revealed_by_a_distractor_does_not_stop(item):
    """Checker case G: the true leaf is covered only by a value-less pin; a late distractor pin reveals Morvane Pumps
    and Morvane's value is pinned. The last entity added to R is off the chain: no stop."""
    f = _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics is based in Leeds.", "Kelvane Fluidics", []),
                                           ("Zentro Ltd. resells parts from Morvane Pumps.", "Zentro Ltd.", ["Morvane Pumps"]),
                                           ("Morvane Pumps rates its seal to 12 bar.", "Morvane Pumps", [])],
                        TRUE3_R + ["morvane pumps"])
    ok, det = f.chain_pinned()
    assert not ok and f.stop_reason(False) is None
    assert det["path"] == TRUE3_R and "morvane pumps" not in det["path"]           # the walk never reaches Morvane
    assert det["leaves"] == {"kelvane fluidics": None} and not det["value_pinned"]


def test_genuine_chains_stop_with_the_walked_path(item):
    ok, det = _fold_with_pins(item, TRUE3, TRUE3_R).chain_pinned()                 # 3 hops
    assert ok and det["roots"] == ["varrel pumps"] and det["path"] == TRUE3_R and det["hops"] == 2
    assert det["edges"] == [["varrel pumps", "ostrand ltd", "p1"], ["ostrand ltd", "kelvane fluidics", "p2"]]
    assert det["leaves"] == {"kelvane fluidics": "p3"} and det["leaf"] == "kelvane fluidics" and det["leaf_pin"] == "p3"
    ok, det = _fold_with_pins(item, [TRUE3[0], ("Ostrand Ltd. rates the seal to 83 degrees C.", "Ostrand Ltd.", [])],
                              TRUE3_R[:2]).chain_pinned()                             # 2 hops
    assert ok and det["path"] == ["varrel pumps", "ostrand ltd"] and det["leaves"] == {"ostrand ltd": "p2"}
    # F3: the final pin's label differs ("seal compound") but its quote names the leaf and reveals nothing
    ok, det = _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics rates the compound to 57 bar.", "seal compound", [])],
                              TRUE3_R).chain_pinned()
    assert ok and det["covered"]["kelvane fluidics"] == {"pin_id": "p3", "by": "quote"}
    # a fully pinned distractor chain that the walk cannot reach neither fires alone nor blocks the true chain
    far = [("Zentro Ltd. resells parts from Morvane Pumps.", "Zentro Ltd.", ["Morvane Pumps"]),
           ("Morvane Pumps rates its seal to 12 bar.", "Morvane Pumps", [])]
    assert not _fold_with_pins(item, TRUE3[:2] + far, TRUE3_R + ["morvane pumps"]).chain_pinned()[0]
    ok, det = _fold_with_pins(item, TRUE3 + far, TRUE3_R + ["morvane pumps"]).chain_pinned()
    assert ok and det["path"] == TRUE3_R
    # the service-life unit: "hours" is a value too (RX_NUM_UNIT alone never matched it)
    life = TRUE3[:2] + [("Seals made by Kelvane Fluidics last 3500 hours in continuous service.", "Kelvane Fluidics", [])]
    assert _fold_with_pins(item, life, TRUE3_R).chain_pinned()[0]
    year = TRUE3[:2] + [("Kelvane Fluidics was founded in 1990.", "Kelvane Fluidics", [])]      # a bare number is no value
    assert not _fold_with_pins(item, year, TRUE3_R).chain_pinned()[0]


def test_decoy_branch_from_hop_two_and_twin_via_distractor_do_not_stop(item):
    # a decoy chain branches from the true chain's hop-2 entity: Ostrand's pins reveal Kelvane (true) and Morvane
    # (decoy); Morvane's value is pinned, Kelvane's is not
    branch = TRUE3[:2] + [("Ostrand Ltd. also resells parts from Morvane Pumps.", "Ostrand Ltd.", ["Morvane Pumps"]),
                          ("Morvane Pumps rates its seal to 12 bar.", "Morvane Pumps", []),
                          ("Kelvane Fluidics is based in Leeds.", "Kelvane Fluidics", [])]
    ok, det = _fold_with_pins(item, branch, TRUE3_R + ["morvane pumps"]).chain_pinned()
    assert not ok and det["leaves"] == {"morvane pumps": "p4", "kelvane fluidics": None}
    assert set(det["path"]) == set(TRUE3_R + ["morvane pumps"])
    # with Kelvane's value pinned too, both leaves hold a value and the chain is pinned
    both = branch + [("Kelvane Fluidics rates the compound to 57 bar.", "Kelvane Fluidics", [])]
    assert _fold_with_pins(item, both, TRUE3_R + ["morvane pumps"]).chain_pinned()[0]
    # a one-character twin revealed by a distractor pin, its value pinned: the true leaf Kelvane Pumps is uncovered
    twin = [("Varrel Pumps sources its seals from Kelvane Pumps.", "Varrel Pumps", ["Kelvane Pumps"]),
            ("Zentro Ltd. buys from Kelvine Pumps.", "Zentro Ltd.", ["Kelvine Pumps"]),
            ("Kelvine Pumps rates its seal at 57 bar.", "Kelvine Pumps", [])]
    ok, det = _fold_with_pins(item, twin, ["varrel pumps", "kelvane pumps", "kelvine pumps"]).chain_pinned()
    assert not ok and det["covered"]["kelvane pumps"] is None and "kelvine pumps" not in det["path"]
    # a pin that reveals an entity never covers it, even when the model also labels the pin with that entity
    mislabel = [("Varrel Pumps sources its seals from Ostrand Ltd. rated to 83 degrees.", "Ostrand Ltd.", ["Ostrand Ltd."])]
    ok, det = _fold_with_pins(item, mislabel, ["varrel pumps", "ostrand ltd"]).chain_pinned()
    assert not ok and det["covered"] == {"varrel pumps": {"pin_id": "p1", "by": "quote"}, "ostrand ltd": None}


def test_chain_walk_handles_cycles_and_self_reveals(item):
    # a back-reveal of the root on the hop-2 pin is an edge, not a hop: the walk still ends at Kelvane
    back = [TRUE3[0], ("Ostrand Ltd. sells to Varrel Pumps and buys from Kelvane Fluidics.", "Ostrand Ltd.",
                       ["Kelvane Fluidics", "Varrel Pumps"]), TRUE3[2]]
    ok, det = _fold_with_pins(item, back, TRUE3_R).chain_pinned()
    assert ok and det["path"] == TRUE3_R and ["ostrand ltd", "varrel pumps", "p2"] in det["edges"]
    # a pure cycle has no leaf: no stop even though Ostrand's pin holds a value
    cycle = [TRUE3[0], ("Ostrand Ltd. sells 57 bar seals to Varrel Pumps.", "Ostrand Ltd.", ["Varrel Pumps"])]
    ok, det = _fold_with_pins(item, cycle, TRUE3_R[:2]).chain_pinned()
    assert not ok and det["leaves"] == {} and det["hops"] == 1
    # a pin that lists its own entity in reveals covers nothing (the same rule that stops the mislabelled hop-1 pin
    # from firing after hop 1): the leaf stays uncovered and the fold reads on
    selfrev = TRUE3[:2] + [("Kelvane Fluidics rates the compound to 57 bar.", "Kelvane Fluidics", ["Kelvane Fluidics"])]
    ok, det = _fold_with_pins(item, selfrev, TRUE3_R).chain_pinned()
    assert not ok and det["covered"]["kelvane fluidics"] is None and det["leaves"] == {}
    # entity identity ignores legal forms: a reveal of "Ostrand" and a label "Ostrand Ltd." are one entity
    short = [("Varrel Pumps buys its seals from Ostrand.", "Varrel Pumps", ["Ostrand"]),
             ("Ostrand Ltd. rates the seal to 83 degrees C.", "Ostrand Ltd.", [])]
    ok, det = _fold_with_pins(item, short, ["varrel pumps", "ostrand"]).chain_pinned()
    assert ok and det["path"] == ["varrel pumps", "ostrand"]


def test_two_question_entities_both_need_a_pinned_chain(item):
    two = dict(item, question="Above what temperature do the seals that Varrel Pumps and Dorrin Valves share degrade?")
    from opa.tracks.track_a.systems import question_entities
    assert question_entities(two["question"]) == ["varrel pumps", "dorrin valves"]      # order of appearance, not hash order
    assert question_entities("Above what temperature do the seals that Dorrin Valves and Varrel Pumps share degrade?") == \
        ["dorrin valves", "varrel pumps"]
    f = _fold_with_pins(two, TRUE3, TRUE3_R)
    assert f.roots == ["varrel pumps", "dorrin valves"]
    ok, det = f.chain_pinned()
    assert not ok and det["roots"] == ["varrel pumps", "dorrin valves"] and det["covered"]["dorrin valves"] is None
    dorrin = [("Dorrin Valves sources seals from Hallum Works.", "Dorrin Valves", ["Hallum Works"]),
              ("Hallum Works rates its seals to 40 bar.", "Hallum Works", [])]
    ok, det = _fold_with_pins(two, TRUE3 + dorrin, TRUE3_R + ["dorrin valves", "hallum works"]).chain_pinned()
    assert ok and det["hops"] == 3 and det["leaves"] == {"kelvane fluidics": "p3", "hallum works": "p5"}
    # a question entity that a pin reveals (a distractor pointing back at it) stays a root of the walk
    g = _fold_with_pins(item, TRUE3 + [("Zentro Ltd. supplies Varrel Pumps.", "Zentro Ltd.", ["Varrel Pumps"])], TRUE3_R)
    ok, det = g.chain_pinned()
    assert ok and det["roots"] == ["varrel pumps"] and det["path"] == TRUE3_R


@pytest.mark.parametrize("value_reveal", ["37 bar", "7700", "3100 hours", "76 °C", "85 c", "6,200", "41 bar "])
def test_a_revealed_value_is_the_answer_not_a_hop(item, value_reveal):
    """Micro-pilot reply shapes: Qwen3-4B lists the final fact's value in the final pin's reveals (with or without
    its unit). The value is not an entity to cover: the final entity stays the leaf and the chain is pinned."""
    quote = f"O-rings made by Kelvane Fluidics are rated for at most {value_reveal.strip()}."
    if "hours" in value_reveal:
        quote = "O-rings made by Kelvane Fluidics last 3100 hours in continuous service."
    if value_reveal in ("7700", "6,200"):
        quote = "Kelvane Fluidics guarantees its o-rings for 7,700 hours of operation."
    if value_reveal == "85 c":                                   # the reveal dropped the degree sign; the quote has it
        quote = "Above 85 °C, the o-rings from Kelvane Fluidics begin to degrade."
    ok, det = _fold_with_pins(item, TRUE3[:2] + [(quote, "Kelvane Fluidics", [value_reveal])], TRUE3_R).chain_pinned()
    assert ok and det["path"] == TRUE3_R and det["leaves"] == {"kelvane fluidics": "p3"}, det
    assert not any(v in e for e in det["path"] for v in ("bar", "7700", "hours", "76", "85"))


def test_value_reveals_elsewhere_and_identifier_reveals(item):
    # a value reveal on a hop-1 pin is dropped too: the pin still reveals the next entity and still covers only hop 1
    f = _fold_with_pins(item, [("Varrel Pumps buys its seals from Ostrand Ltd. at 5 bar.", "Varrel Pumps", ["5 bar", "Ostrand Ltd."]),
                               ("Ostrand Ltd. rates the seal to 83 degrees C.", "Ostrand Ltd.", ["83"])], TRUE3_R[:2])
    ok, det = f.chain_pinned()
    assert ok and det["edges"] == [["varrel pumps", "ostrand ltd", "p1"]] and det["leaves"] == {"ostrand ltd": "p2"}
    # a value-only reveal does not turn a value-less leaf into a chain end with a value, nor exclude coverage
    g = _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics is based in Leeds.", "Kelvane Fluidics", ["12"])], TRUE3_R)
    ok, det = g.chain_pinned()
    assert not ok and det["leaves"] == {"kelvane fluidics": None} and det["covered"]["kelvane fluidics"]["pin_id"] == "p3"
    # an identifier with letters before its digits is an entity (a hop), as is a noun: both keep the fold reading
    from opa.tracks.track_a.text import normalize_key
    for rev in ("QX-2291", "seal service life"):
        h = _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics rates the compound to 57 bar.", "Kelvane Fluidics", [rev])], TRUE3_R)
        ok, det = h.chain_pinned()
        assert not ok and det["covered"][normalize_key(rev)] is None, rev
    # case G and case H keep their verdicts with value reveals ignored
    assert not _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics is based in Leeds.", "Kelvane Fluidics", []),
                                                  ("Zentro Ltd. resells parts from Morvane Pumps.", "Zentro Ltd.", ["Morvane Pumps"]),
                                                  ("Morvane Pumps rates its seal to 12 bar.", "Morvane Pumps", ["12 bar"])],
                               TRUE3_R + ["morvane pumps"]).chain_pinned()[0]
    assert not _fold_with_pins(item, TRUE3[:2] + [("Kelvane Fluidics rates the compound to 57 bar per Norsk Test Labs.",
                                                   "Kelvane Fluidics", ["Norsk Test Labs", "57 bar"])], TRUE3_R).chain_pinned()[0]


def test_value_reveals_never_enter_r_faults_or_the_read_prompt(item):
    """Fold.read adds only entity reveals to R: a revealed value ("37 bar") never becomes a relevant entity, a fault
    target or part of the read prompt's "Relevant entities" line (the pin itself keeps the raw list)."""
    import re as _re
    from opa.tracks.track_a.index import Session
    from opa.tracks.track_a.systems import Ctx, Fold, session_cfg
    o = item["gold"]["fact_origins"][0]                                  # holds "Varrel Pumps sources seals from Ostrand Ltd."

    def fold_after_read(reveals):
        emb = HashEmbedder()
        sess = Session(item, session_cfg("S12", {}), emb, skeleton_llm={})
        reply = json.dumps({"pins": [{"origin_id": o, "quote": GENUINE[0][0], "entity": "Varrel Pumps", "reveals": reveals}],
                            "answer_found": False})
        prompts, events = [], []
        c = Ctx(item=item, sess=sess, chat=FakeChat(lambda s, u: prompts.append(u) or reply),
                opts={"_q_emb": emb.embed_query([item["question"]])[0], "_embedder": emb}, log_fn=events.append)
        f = Fold(c, "S12")
        for e in f.roots:
            f.add_relevant(e)
        f.page_in(o)
        f.read([o])
        f.fault_fixpoint(before=None)
        rel = [_re.search(r"Relevant entities: (.*)", u).group(1) for u in prompts]
        return f, events, rel

    f, events, rel = fold_after_read(["37 bar"])
    assert f.R == ["varrel pumps"] and list(f.became_relevant) == ["varrel pumps"] and f.pins[0].reveals == ["37 bar"]
    assert not [e for e in events if e["action"] == "fault"] and all("37 bar" not in r for r in rel)
    g, events, rel = fold_after_read(["Ostrand Ltd.", "37 bar"])
    assert g.R == ["varrel pumps", "ostrand ltd"] and list(g.became_relevant) == ["varrel pumps", "ostrand ltd"]
    faults = [e for e in events if e["action"] == "fault"]
    assert faults and all(e["reason"] == "late_bound:ostrand ltd" for e in faults) and all("37 bar" not in r for r in rel)
    assert g.pins[0].reveals == ["Ostrand Ltd.", "37 bar"]


def test_component_noun_in_the_question_is_a_root_that_the_chain_facts_cover(item):
    """MT questions capitalize "O-rings": question_entities then yields the component as a second root. The chain's
    facts all name the component, so it is covered by quote and walks the same chain; pins that never name it leave
    it uncovered and the fold reads on."""
    orings = dict(item, question="Up to what pressure are the O-rings in Varrel Pumps's supply chain rated?")
    chain = [("Varrel Pumps sources its O-rings from Ostrand Ltd.", "Varrel Pumps", ["Ostrand Ltd."]),
             ("O-rings made by Ostrand Ltd. are rated for at most 41 bar.", "Ostrand Ltd.", ["41 bar"])]
    f = _fold_with_pins(orings, chain, ["o-rings", "varrel pumps", "ostrand ltd"])
    assert f.roots == ["o-rings", "varrel pumps"]
    ok, det = f.chain_pinned()
    assert ok and det["path"] == ["o-rings", "varrel pumps", "ostrand ltd"] and det["leaves"] == {"ostrand ltd": "p2"}
    assert det["covered"]["o-rings"] == {"pin_id": "p1", "by": "quote"}
    g = _fold_with_pins(orings, [("Varrel Pumps buys its seals from Ostrand Ltd.", "Varrel Pumps", ["Ostrand Ltd."]),
                                 ("Ostrand Ltd. rates the seal to 41 bar.", "Ostrand Ltd.", [])], ["o-rings", "varrel pumps", "ostrand ltd"])
    ok, det = g.chain_pinned()
    assert not ok and det["covered"]["o-rings"] is None


def test_early_stop_event_logs_the_walked_path(item):
    rec, events = _run_with(FakeChat(_pins_a_value), "S12", item)
    cov = [e for e in events if e["action"] == "early_stop"][0]["coverage"]
    assert cov["roots"] == ["varrel pumps"] and cov["path"] == ["varrel pumps", "ostrand ltd", "kelvane"] and cov["hops"] == 2
    assert [e[:2] for e in cov["edges"]] == [["varrel pumps", "ostrand ltd"], ["ostrand ltd", "kelvane"]]
    assert cov["leaves"] == {"kelvane": cov["leaf_pin"]} and cov["leaf_pin"] and cov["value_pinned"]
    # the same reader on every descent-first system stops for the same structural reason
    for system in ("S14", "S15", "S16", "S17", "S18", "S12F"):
        rec, events = _run_with(FakeChat(_pins_a_value), system, item)
        assert rec["stop_reason"] == "early_chain_pinned", system
        assert [e["coverage"]["path"] for e in events if e["action"] == "early_stop"] == [["varrel pumps", "ostrand ltd", "kelvane"]]


# ---------------------------------------------------------------- transport retries
class RemoteProtocolError(Exception):
    pass


class ConnectError(Exception):
    pass


class ReadTimeout(Exception):
    pass


class _Flaky:
    """Fake transport: raises / returns the scripted outcomes in order, then repeats the last one."""

    def __init__(self, outcomes):
        self.outcomes, self.n = list(outcomes), 0

    def post(self, url, json):
        o = self.outcomes[min(self.n, len(self.outcomes) - 1)]
        self.n += 1
        if isinstance(o, BaseException):
            raise o
        return o


OKR = _Resp(200, {"choices": [{"message": {"content": '{"answer": "1", "cited_pin_ids": []}'}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 3}})


def _flaky_chat(outcomes):
    c = OpenAIChat("http://x", "m")
    c._http = _Flaky(outcomes)
    sleeps = []
    c._sleep = sleeps.append
    return c, sleeps


def test_transport_error_then_success_is_retried_with_backoff_logged_and_counted():
    c, sleeps = _flaky_chat([RemoteProtocolError("Server disconnected without sending a response"), ConnectError("x"),
                             _Resp(503, "busy"), OKR])
    ev, u = [], Usage()
    assert c.ask("s", "u", 50, u, ev.append) == '{"answer": "1", "cited_pin_ids": []}'
    assert sleeps == [1.0, 3.0, 9.0] and c.http.n == 4 and u.n_transport_retries == 3 and u.calls == 1
    retries = [e for e in ev if e["action"] == "llm_transport_retry"]
    assert [(e["attempt"], e["error"], e["status"]) for e in retries] == \
        [(1, "RemoteProtocolError", None), (2, "ConnectError", None), (3, None, 503)]
    assert [e["action"] for e in ev][-1] == "llm_call"


def test_four_transport_failures_raise_and_the_item_is_an_error(item):
    c, sleeps = _flaky_chat([ReadTimeout("t")])
    u = Usage()
    with pytest.raises(ReadTimeout):
        c.ask("s", "u", 50, u)
    assert c.http.n == 4 and sleeps == [1.0, 3.0, 9.0] and u.n_transport_retries == 3
    c2, _ = _flaky_chat([RemoteProtocolError("gone")])
    rec, events = _run_with(c2, "S1", item)
    assert rec["status"] == "error" and "RemoteProtocolError" in rec["error"]
    assert len([e for e in events if e["action"] == "llm_transport_retry"]) == 3
    c3, _ = _flaky_chat([_Resp(504, "gateway")])                                  # persistent 5xx: raise_for_status error
    with pytest.raises(RuntimeError, match="http 504"):
        c3.ask("s", "u", 50, Usage())
    assert c3.http.n == 4


def test_4xx_and_other_exceptions_are_never_retried():
    for resp in (_Resp(429, "slow down"), _Resp(401, "no"), _Resp(404, "nf")):
        c, sleeps = _flaky_chat([resp])
        with pytest.raises(RuntimeError):
            c.ask("s", "u", 50, Usage())
        assert c.http.n == 1 and sleeps == [], resp.status_code
    c, sleeps = _flaky_chat([ValueError("bug")])
    with pytest.raises(ValueError):
        c.ask("s", "u", 50, Usage())
    assert c.http.n == 1 and sleeps == []
    c, sleeps = _flaky_chat([_Resp(400, {"message": "This model's maximum context length is 40960 tokens"})])
    with pytest.raises(ContextOverflow):
        c.ask("s", "u", 50, Usage())
    assert c.http.n == 1


def test_retries_are_counted_per_record_and_in_summarize(tmp_path, item, monkeypatch):
    import opa.tracks.track_a.run as run_mod
    c, _ = _flaky_chat([ConnectError("x"), OKR])
    monkeypatch.setattr(run_mod, "make_chat", lambda a, ctx: c)
    ip = _items_file(tmp_path, [item])
    out = run_system({"items_uri": str(ip), "system": "S1", "embedder": "hash", "workers": 1}, LocalCtx(str(tmp_path / "o")))
    assert out["metrics"]["n_transport_retries"] == 1


def test_http_500_is_retried_then_succeeds():
    c, sleeps = _flaky_chat([_Resp(500, "transient"), OKR])
    u = Usage()
    assert c.ask("s", "u", 50, u).startswith("{")
    assert c.http.n == 2 and sleeps == [1.0] and u.n_transport_retries == 1


# ---------------------------------------------------------------- S-llm skeletons (sllm-v2)
def _origin_ids(user):
    import re as _re
    return _re.findall(r'<origin id="(\d+)">', user)


class SkChat(FakeChat):
    """Answers the skeleton prompt with 3 names per origin, compact JSON (like the model is asked to), never truncated."""

    def __init__(self, cap_tokens=None):
        super().__init__(lambda s, u: "")
        self.cap, self.calls_log = cap_tokens, []

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        ids = _origin_ids(user)
        full = json.dumps({"entities": {i: [f"Name{i}a", f"Name{i}b", f"Name{i}c"] for i in ids}}, separators=(",", ":"))
        cap = self.cap if self.cap is not None else 10 ** 9
        reply, finish = (full[:cap], "length") if len(full) > cap else (full, "stop")
        self.calls_log.append((ids, max_tokens, finish))
        if usage is not None:
            usage.add(len(user) // 4, len(reply) // 4)
        if log:
            log({"action": "llm_call", "finish_reason": finish, "usage": {"prompt_tokens": len(user) // 4,
                                                                          "completion_tokens": len(reply) // 4, "cached_tokens": 0}})
        return reply


def test_salvage_members_keeps_every_complete_origin_before_the_cut():
    from opa.tracks.track_a.models import salvage_members
    cut = '{"entities":{"0":["A","B"],"1":["C"],"2":["D","E'
    assert salvage_members(cut, "entities") == {"0": ["A", "B"], "1": ["C"]}
    pretty = '{\n "entities": {\n  "0": [\n   "A"\n  ],\n  "1": [\n   "B",\n   "Ste'
    assert salvage_members(pretty, "entities") == {"0": ["A"]}
    assert salvage_members('{"entities":{"0":["A"],"1":["B"]', "entities") == {"0": ["A"], "1": ["B"]}     # cut right after a list
    assert salvage_members('{"entities":{"0":["A"],"1":"B","2":["C"]}}', "entities") == {"0": ["A"], "2": ["C"]}
    assert salvage_members("no json", "entities") == {} and salvage_members('{"other":{"0":["A"]}}', "entities") == {}
    assert salvage_members('{"entities":{"0":["A"],"x":5,"2":["C"]}}', "entities") == {"0": ["A"], "2": ["C"]}


def test_llm_skeleton_chunks_caps_and_stats(item):
    from opa.tracks.track_a.run import SKELETON_MAX_TOKENS, SKELETON_PER_CALL, SKELETON_PROMPT_VERSION, llm_skeleton
    assert (SKELETON_PER_CALL, SKELETON_MAX_TOKENS, SKELETON_PROMPT_VERSION) == (4, 1800, "sllm-v2")
    chat, st = SkChat(), {}
    sk = llm_skeleton(item, chat, Usage(), None, stats=st)
    n = len(item["origins"])
    assert len(sk) == n and all(len(v) == 3 for v in sk.values())
    assert all(len(ids) <= 4 and mt == 1800 for ids, mt, _ in chat.calls_log) and len(chat.calls_log) == (n + 3) // 4
    assert (st["n_calls"], st["n_parse_errors"], st["n_truncated"], st["n_missing"], st["n_empty"]) == ((n + 3) // 4, 0, 0, 0, 0)
    assert st["empty_origin_fraction"] == 0.0


def test_truncated_reply_keeps_complete_origins_and_asks_the_lost_ones_again(item):
    from opa.tracks.track_a.run import llm_skeleton
    full_len = len(json.dumps({"entities": {str(i): ["Name%sa" % i, "Name%sb" % i, "Name%sc" % i] for i in range(4)}}, separators=(",", ":")))
    chat, st = SkChat(cap_tokens=full_len - 25), {}                       # 4-origin chunks are cut inside the last entry
    sk = llm_skeleton(item, chat, Usage(), None, stats=st)
    assert all(len(v) == 3 for v in sk.values()) and st["n_missing"] == 0 and st["empty_origin_fraction"] == 0.0
    assert st["n_truncated"] >= 1 and st["n_parse_errors"] >= 1
    # the first call of a chunk lost only its last origin; the second call asked for just that origin
    assert len(chat.calls_log[0][0]) == 4 and chat.calls_log[1][0] == [chat.calls_log[0][0][-1]]
    # when even the single-origin retry is cut, those origins stay empty and are counted as missing
    hopeless, st2 = SkChat(cap_tokens=5), {}
    sk2 = llm_skeleton(item, hopeless, Usage(), None, stats=st2)
    assert all(v == [] for v in sk2.values()) and st2["n_missing"] == len(sk2) and st2["empty_origin_fraction"] == 1.0


def test_build_skeletons_records_stats_and_fails_loudly_above_half_empty(tmp_path, item):
    from opa.tracks.track_a.run import build_skeletons
    ip = _items_file(tmp_path, [item, make_item(seed=1)])
    ctx = LocalCtx(str(tmp_path / "ok"))
    out = build_skeletons({"items_uri": str(ip), "workers": 1, "model": {"kind": "fake", "chat": SkChat()}}, ctx)
    m = out["metrics"]
    assert m["prompt_version"] == "sllm-v2" and m["empty_origin_fraction"] == 0.0 and m["n_item_errors"] == 0
    f = json.loads((tmp_path / "ok" / "skeleton_llm" / f"{item['item_id']}.json").read_text())
    assert f["prompt_version"] == "sllm-v2" and f["status"] == "ok" and f["stats"]["n_calls"] > 0
    bctx = LocalCtx(str(tmp_path / "bad"))
    with pytest.raises(RuntimeError, match="more than 50% empty"):
        build_skeletons({"items_uri": str(ip), "workers": 1, "model": {"kind": "fake", "chat": SkChat(cap_tokens=5)}}, bctx)
    f = json.loads((tmp_path / "bad" / "skeleton_llm" / f"{item['item_id']}.json").read_text())
    assert f["status"] == "error" and "empty skeleton" in f["error"]
    mm = json.loads((tmp_path / "bad" / "metrics.json").read_text())
    assert mm["n_item_errors"] == 2 and mm["empty_origin_fraction"] == 1.0 and mm["n_missing"] > 0


def test_run_system_refuses_old_or_failed_skeletons(tmp_path, item):
    ip = _items_file(tmp_path, [item])
    sk = tmp_path / "sk"
    sk.mkdir()
    base = {"items_uri": str(ip), "system": "S11", "embedder": "hash", "workers": 1, "skeleton_llm_uri": str(sk),
            "model": {"kind": "fake", "chat": FakeChat(policy)}}
    ents = {str(o["origin_id"]): [] for o in item["origins"]}
    (sk / f"{item['item_id']}.json").write_text(json.dumps({"prompt_version": "sllm-v1", "entities": ents}))
    with pytest.raises(RuntimeError, match="expected 'sllm-v2'"):
        run_system(base, LocalCtx(str(tmp_path / "a")))
    (sk / f"{item['item_id']}.json").write_text(json.dumps({"entities": ents}))
    with pytest.raises(RuntimeError, match="prompt_version None"):
        run_system(base, LocalCtx(str(tmp_path / "b")))
    (sk / f"{item['item_id']}.json").write_text(json.dumps({"prompt_version": "sllm-v2", "status": "error", "error": "70% empty", "entities": ents}))
    with pytest.raises(RuntimeError, match="quality check"):
        run_system(base, LocalCtx(str(tmp_path / "c")))
    (sk / f"{item['item_id']}.json").write_text(json.dumps({"prompt_version": "sllm-v2", "status": "ok", "entities": ents}))
    run_system(base, LocalCtx(str(tmp_path / "d")))


# ---------------------------------------------------------------- S4 (ReadAgent-style) tuning iteration 1
class GistChat(FakeChat):
    """Answers the S4 prompts. Gist replies: `mode` per call index -> "ok" | "cut" (truncated mid-string, finish length) |
    "garbage" | "cut_empty" (cut before any gist text)."""

    def __init__(self, modes=None, lookup_pages=(0,)):
        super().__init__(lambda s, u: "")
        self.modes, self.lookup_pages, self.gist_calls, self.max_tokens_seen = modes or {}, list(lookup_pages), 0, []

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        finish = "stop"
        if system.startswith("Shorten the page"):
            self.max_tokens_seen.append(max_tokens)
            if user.endswith("Reply with JSON only."):
                mode = self.modes.get(("retry", self.cur), "ok")
            else:
                self.cur = self.gist_calls
                self.gist_calls += 1
                mode = self.modes.get(self.cur, "ok")
            reply = {"ok": '{"gist": "short gist"}', "cut": '{"gist": "Varrel Pumps sources seals from Ostrand Ltd. and mor',
                     "garbage": "no json at all", "cut_empty": '{"gi'}[mode]
            finish = "length" if mode in ("cut", "cut_empty") else "stop"
        elif system.startswith("You see a question"):
            reply = json.dumps({"pages": self.lookup_pages})
        else:
            reply = '{"answer": "83 \u00b0C", "cited_pin_ids": []}'
        if usage is not None:
            usage.add(10, 5)
        if log:
            log({"action": "llm_call", "finish_reason": finish, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}})
        return reply


def test_salvage_string_field_cases():
    from opa.tracks.track_a.models import MISSING, salvage_string_field
    assert salvage_string_field('{"gist": "all done"}', "gist") == ("all done", True)
    assert salvage_string_field('{"gist": "cut off in the mid', "gist") == ("cut off in the mid", False)
    assert salvage_string_field('{"gist": "a \\"quoted\\" word and a dangling esc\\', "gist") == ('a "quoted" word and a dangling esc', False)
    assert salvage_string_field('{"gist": "', "gist") is MISSING and salvage_string_field('{"gi', "gist") is MISSING
    assert salvage_string_field('{"x": 1, "gist": "after a member"}', "gist") == ("after a member", True)
    assert salvage_string_field('{"x": 1 "gist": "bad member before"}', "gist") is MISSING
    assert salvage_string_field("no json", "gist") is MISSING and salvage_string_field('{"other": "v"}', "gist") is MISSING
    # sloppy model JSON: a literal newline in a complete gist, a fenced reply, an unescaped inner quote, closers on a cut value
    assert salvage_string_field('{"gist": "line one\nline two"}', "gist") == ("line one\nline two", True)
    assert salvage_string_field('```json\n{"gist": "fenced text"}\n```', "gist") == ("fenced text", True)
    assert salvage_string_field('{"gist": "He said "hi" then left."}', "gist") == ('He said "hi" then left.', False)
    assert salvage_string_field('```json\n{"gist": "He said "hi" then left."}\n```', "gist") == ('He said "hi" then left.', False)
    assert salvage_string_field('{"gist": "bad \\q escape here"}\n```', "gist") == ("bad q escape here", False)       # invalid escape, closers stripped
    assert salvage_string_field('{"gist": "cut after text"}', "gist") == ("cut after text", True)


def test_s4_defaults_page_size_one_and_gist_cap_300(item):
    chat = GistChat()
    rec, events = _run_with(chat, "S4", item)
    n = len(item["origins"])
    assert len(rec["pages"]) == n and all(len(pg) == 1 for pg in rec["pages"])            # page_size 1
    assert chat.max_tokens_seen == [300] * n                                              # gist cap 300
    assert rec["n_gist_calls"] == n and rec["n_gist_truncated"] == rec["n_gist_empty"] == rec["n_gist_parse_errors"] == 0
    assert rec["config_version"] == "s4-v2" and rec["tuning_iteration"] == 1
    assert rec["paged_in"] == [0] and rec["lookup_pages"] == [0]


def test_s4_lookup_prompt_allows_several_pages_and_up_to_L():
    from opa.tracks.track_a.systems import LOOKUP_SYSTEM
    p = LOOKUP_SYSTEM.format(L=5)
    assert "up to 5 pages" in p and "you may choose several pages" in p and "may contain a fact you need" in p
    chat = GistChat(lookup_pages=[0, 2, 3, 5, 7, 9, 11])
    rec, _ = _run_with(chat, "S4", make_item())
    assert rec["lookup_pages"] == [0, 2, 3, 5, 7] and rec["paged_in"] == [0, 2, 3, 5, 7]       # parallel lookup, capped at L=5


def test_s4_truncated_gist_is_salvaged_not_emptied_and_flags_are_recorded(item):
    chat = GistChat(modes={0: "cut", 1: "garbage", 2: "cut_empty", ("retry", 1): "ok", ("retry", 2): "garbage"})
    rec, events = _run_with(chat, "S4", item)
    fl = rec["gist_pages"]
    assert fl[0] == {"n_calls": 1, "truncated": True, "parse_error": True, "empty": False}      # cut mid-string: salvaged, no retry
    assert fl[1]["n_calls"] == 2 and fl[1]["parse_error"] and not fl[1]["empty"]               # garbage then a good retry
    assert fl[2]["n_calls"] == 2 and fl[2]["truncated"] and fl[2]["empty"]                    # nothing readable twice: empty
    assert rec["n_gist_truncated"] == 2 and rec["n_gist_empty"] == 1 and rec["n_gist_parse_errors"] == 3
    gist_events = [e for e in events if e["action"] == "gist"]
    assert gist_events[0]["truncated"] and not gist_events[0]["empty"] and gist_events[2]["empty"]
    # the salvaged text reached the lookup prompt, the empty gist is an empty page
    # (the cut gist is the model's partial text, not "")
    assert rec["n_gist_calls"] == len(rec["pages"]) + 2          # pages 1 and 2 were re-asked once


def test_gist_harness_failure_uses_gold_after_the_run_only(item):
    from opa.tracks.track_a.run import summarize
    from opa.tracks.track_a.systems import gist_harness_failure
    fact_origin = item["gold"]["fact_origins"][0]
    n = len(item["origins"])
    modes = {fact_origin: "cut_empty", ("retry", fact_origin): "cut_empty"}
    rec, _ = _run_with(GistChat(modes=modes), "S4", item)
    assert rec["gist_pages"][fact_origin]["empty"] and gist_harness_failure(item, rec) is True
    other = next(i for i in range(n) if i not in item["gold"]["fact_origins"])
    rec2, _ = _run_with(GistChat(modes={other: "cut_empty", ("retry", other): "cut_empty"}), "S4", item)
    assert rec2["n_gist_empty"] == 1 and gist_harness_failure(item, rec2) is False        # empty gist on a page without a gold fact
    rec3, _ = _run_with(GistChat(), "S4", item)
    assert gist_harness_failure(item, rec3) is False
    m = summarize([item] * 0 + [dict(item, item_id="a"), dict(item, item_id="b")],
                  [dict(rec, item_id="a"), dict(rec3, item_id="b")], "S4")
    assert m["gist_harness_failure_rate"] == 0.5 and m["n_gist_empty"] == 1 and m["config_version"] == "s4-v2"
    assert m["tuning_iteration"] == 1 and m["n_pages"] == 2 * n
    # the system never saw gold: the gold-dependent flag is not in the record the run produced
    assert "gist_harness_failure" not in rec


class LookupGarbageChat(GistChat):
    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        if system.startswith("You see a question"):
            self.lookup_calls = getattr(self, "lookup_calls", 0) + 1
            if usage is not None:
                usage.add(10, 5)
            return "I would look at page three, I think."
        return super().ask(system, user, max_tokens, usage, log)


def test_unreadable_lookup_is_a_recorded_harness_failure_not_zero_pages(item):
    from opa.tracks.track_a.run import summarize
    from opa.tracks.track_a.systems import s4_harness_failure
    chat = LookupGarbageChat()
    rec, events = _run_with(chat, "S4", item)
    assert chat.lookup_calls == 2                                          # asked once, re-asked once
    assert rec["lookup_parse_error"] is True and rec["lookup_empty"] is True and rec["paged_in"] == []
    assert s4_harness_failure(item, rec) is True
    ok_rec, _ = _run_with(GistChat(lookup_pages=[]), "S4", item)             # a readable {"pages": []} is a choice, not a failure
    assert ok_rec["lookup_parse_error"] is False and ok_rec["lookup_empty"] is True and s4_harness_failure(item, ok_rec) is False
    good, _ = _run_with(GistChat(), "S4", item)
    m = summarize([dict(item, item_id="a"), dict(item, item_id="b"), dict(item, item_id="c")],
                  [dict(rec, item_id="a"), dict(ok_rec, item_id="b"), dict(good, item_id="c")], "S4")
    assert m["n_lookup_parse_errors"] == 1 and abs(m["s4_harness_failure_rate"] - 1 / 3) < 1e-9
    assert m["gist_harness_failure_rate"] == 0.0 and "tokens_in" in m and "tokens_out" in m


# ---------------------------------------------------------------- storage reads: NotFound vs transient errors, retries, cache
class NotFound(Exception):                       # same class name as google.api_core.exceptions.NotFound
    pass


class FakeStore:
    """Fake gs:// store. script[uri] = list of outcomes consumed in order (an exception instance raises, bytes returns);
    the last outcome repeats. Objects in `files` are returned directly."""

    def __init__(self, files=None, script=None):
        self.files, self.script, self.calls = dict(files or {}), dict(script or {}), collections.Counter()

    def get_bytes(self, uri):
        self.calls[uri] += 1
        if uri in self.script:
            seq = self.script[uri]
            o = seq[min(self.calls[uri] - 1, len(seq) - 1)]
            if isinstance(o, BaseException):
                raise o
            return o
        if uri in self.files:
            return self.files[uri]
        raise NotFound(uri)


import collections  # noqa: E402


def _skeleton_bytes(it):
    return json.dumps({"item_id": it["item_id"], "prompt_version": "sllm-v2", "status": "ok",
                       "entities": {str(o["origin_id"]): [] for o in it["origins"]}}).encode()


def test_read_bytes_retries_transient_errors_with_backoff_and_counts(monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    store = FakeStore(script={"gs://b/x.json": [ConnectionError("reset"), TimeoutError("slow"), b'{"ok": 1}']})
    monkeypatch.setattr(gcs, "get_bytes", store.get_bytes)
    sleeps = []
    monkeypatch.setattr(run_mod, "_gcs_sleep", sleeps.append)
    stats = {}
    assert run_mod.load_json("gs://b/x.json", stats) == {"ok": 1}
    assert sleeps == [0.5, 1.0] and stats == {"retries": 2} and store.calls["gs://b/x.json"] == 3


def test_not_found_is_not_retried_and_other_errors_are_never_reported_as_missing(monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    store = FakeStore(script={"gs://b/bad.json": [ConnectionError("down")]})
    monkeypatch.setattr(gcs, "get_bytes", store.get_bytes)
    sleeps = []
    monkeypatch.setattr(run_mod, "_gcs_sleep", sleeps.append)
    assert run_mod.load_json("gs://b/missing.json") is None                   # true NotFound -> None, no retry
    assert store.calls["gs://b/missing.json"] == 1 and sleeps == []
    with pytest.raises(ConnectionError):                                      # persistent transient error: raised, not None
        run_mod.load_json("gs://b/bad.json")
    assert store.calls["gs://b/bad.json"] == 5 and sleeps == [0.5, 1.0, 2.0, 4.0]
    assert run_mod.load_json(str(Path("no") / "such" / "file.json")) is None  # local missing file


def test_run_system_survives_a_transient_skeleton_read_error_and_counts_retries(tmp_path, item, monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    uri = f"gs://bkt/sk/{item['item_id']}.json"
    store = FakeStore(script={uri: [ConnectionError("429 rate limit"), _skeleton_bytes(item)]})
    monkeypatch.setattr(gcs, "get_bytes", store.get_bytes)
    monkeypatch.setattr(run_mod, "_gcs_sleep", lambda s: None)
    ip = _items_file(tmp_path, [item])
    out = run_system({"items_uri": str(ip), "system": "S8", "embedder": "hash", "workers": 1, "skeleton_llm_uri": "gs://bkt/sk",
                      "model": {"kind": "fake", "chat": FakeChat(policy)}}, LocalCtx(str(tmp_path / "o")))
    assert out["metrics"]["n_gcs_retries"] == 1 and out["metrics"]["n"] == 1
    recs = [json.loads(x) for x in gzip.decompress((tmp_path / "o" / "records.jsonl.gz").read_bytes()).decode().splitlines()]
    assert recs[0]["n_gcs_retries"] == 1


def test_only_a_true_not_found_raises_the_no_skeleton_error(tmp_path, item, monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    ip = _items_file(tmp_path, [item])
    base = {"items_uri": str(ip), "system": "S8", "embedder": "hash", "workers": 1, "skeleton_llm_uri": "gs://bkt/sk",
            "model": {"kind": "fake", "chat": FakeChat(policy)}}
    monkeypatch.setattr(run_mod, "_gcs_sleep", lambda s: None)
    monkeypatch.setattr(gcs, "get_bytes", FakeStore().get_bytes)              # 404 for everything
    with pytest.raises(RuntimeError, match="no S-llm skeleton"):
        run_system(base, LocalCtx(str(tmp_path / "a")))
    uri = f"gs://bkt/sk/{item['item_id']}.json"
    monkeypatch.setattr(gcs, "get_bytes", FakeStore(script={uri: [ConnectionError("down")]}).get_bytes)   # never recovers
    with pytest.raises(ConnectionError):
        run_system(base, LocalCtx(str(tmp_path / "b")))                       # surfaces the real error, not "missing"


def test_skeleton_is_read_once_per_item_per_run_across_sub_runs(tmp_path, item, monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    uri = f"gs://bkt/sk/{item['item_id']}.json"
    store = FakeStore(files={uri: _skeleton_bytes(item)})
    monkeypatch.setattr(gcs, "get_bytes", store.get_bytes)
    ip = _items_file(tmp_path, [item])
    run_system({"items_uri": str(ip), "systems": ["S8", "S11"], "repeats": 2, "embedder": "hash", "workers": 1,
                "skeleton_llm_uri": "gs://bkt/sk", "model": {"kind": "fake", "chat": FakeChat(policy)}},
               LocalCtx(str(tmp_path / "o")))
    assert store.calls[uri] == 1                                              # 4 sub-runs, one read


def test_retry_events_and_total_in_root_metrics_and_provenance(tmp_path, item, monkeypatch):
    import opa.gcs as gcs
    import opa.tracks.track_a.run as run_mod
    uri = f"gs://bkt/sk/{item['item_id']}.json"
    items_uri = "gs://bkt/items.jsonl.gz"
    items_blob = gzip.compress((json.dumps(item) + "\n").encode())
    store = FakeStore(files={uri: _skeleton_bytes(item)},
                      script={items_uri: [TimeoutError("read timed out"), items_blob],         # a retry that never reaches n_gcs_retries
                              uri: [ConnectionError("429 Too Many Requests"), _skeleton_bytes(item)]})
    monkeypatch.setattr(gcs, "get_bytes", store.get_bytes)
    monkeypatch.setattr(run_mod, "_gcs_sleep", lambda s: None)
    ctx = LocalCtx(str(tmp_path / "o"))
    out = run_system({"items_uri": items_uri, "systems": ["S8"], "repeats": 1, "embedder": "hash", "workers": 1,
                      "skeleton_llm_uri": "gs://bkt/sk", "model": {"kind": "fake", "chat": FakeChat(policy)}}, ctx)
    root = out["metrics"]
    assert root["n_gcs_retries_total"] == 2 and root["provenance"]["n_gcs_retries_total"] == 2         # items read + skeleton read
    assert root["runs"]["S8/r0"]["n_gcs_retries"] == 1                                                # per-item reads only
    assert root["runs"]["S8/r0"]["n_gcs_retries_total"] in (1, 2)
    ev = [e for e in ctx.events if e.get("action") == "gcs_retry"]
    assert [(e["uri"], e["attempt"], e["exc_type"]) for e in ev] == [(items_uri, 1, "TimeoutError"), (uri, 1, "ConnectionError")]
    assert "429" in ev[1]["exc"] and ev[0]["exc"] == "read timed out"
    single = run_system({"items_uri": items_uri, "system": "S8", "embedder": "hash", "workers": 1, "skeleton_llm_uri": "gs://bkt/sk",
                         "model": {"kind": "fake", "chat": FakeChat(policy)}}, LocalCtx(str(tmp_path / "s")))
    assert single["metrics"]["n_gcs_retries_total"] >= 0 and "n_gcs_retries_total" in single["metrics"]["provenance"]
