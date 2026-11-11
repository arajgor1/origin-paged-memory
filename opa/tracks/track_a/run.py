"""Track A job entrypoints.

  opa.tracks.track_a:run_system       one system, or several systems x repeats, on a batch of items (one vLLM start)
  opa.tracks.track_a:embed_items      embeddings of origins and questions, once (npz)
  opa.tracks.track_a:embedder_check   PLAN_V3 Section 8 check 1: recall@k of gold-fact origins per embedder
  opa.tracks.track_a:build_skeletons  S-llm skeletons for a dataset cell, once, shared by every hybrid-skeleton system

Job args (run_system):
  items_uri        gs://.../items.jsonl.gz or a local path
  n_items, offset  which items (default all, 0). items_uri may be a list whose entries are a uri string or
                   {"uri": ..., "n_items": n, "offset": o} (per-URI values override the job-level n_items/offset)
  system           S1..S18 (S12F = S12 with fitted addresses, arm A4), one system; outputs at the run root
  systems, repeats `systems: ["S1", "S2", ...]` and/or `repeats: k` (default 1): one model server and embedder for
                   all, each (system, repeat) runs over the same selected items. Outputs per run in
                   `<system>/r<rep>/records.jsonl.gz` and `<system>/r<rep>/metrics.json` (readable as separate runs);
                   the root metrics.json holds {"runs": {"S1/r0": metrics, ...}, "provenance": ...}. Events carry
                   `system`, `repeat` and `sub_run_id`. With neither key beyond a single `system`, the old single-run
                   layout is kept (no sub-runs).
                   Sub-runs: sub_run_id = f"{run_id}~{system}~r{rep}", written into every record, every sub-run
                   metrics.json and every event line; `sub_runs.json` at the root lists them. Before each (system,
                   repeat) the server's prefix cache is reset (POST /reset_prefix_cache; the server runs with
                   VLLM_SERVER_DEV_MODE=1) and a `prefix_cache_reset` event is logged; provenance records `run_order`.
                   Cost attribution: each sub-run's metrics.json and sub_runs.json row carry started_at, finished_at,
                   wall_seconds, tokens_in, tokens_out, n_llm_calls and gpu_seconds_share (= wall_seconds: the container
                   is dedicated to one sub-run at a time); the root metrics.json / sub_runs.json carry
                   server_startup_seconds so start-up is not charged to the first system. After the first cache reset
                   a `prefix_cache_reset` event logs the server's answer (success, attempts; the reset is retried while
                   blocks are held and the job fails if it never succeeds) and a deterministic probe: a fixed prompt
                   (>= 64 tokens, max_tokens 1) is sent twice before the reset (`probe_cached_tokens_before`, > 0 when
                   prefix caching works), the cache is reset, and the probe is sent again
                   (`probe_cached_tokens_after`); the job raises if that is > 0 after a server-confirmed success, i.e.
                   the reset did not take. After each sub-run a `prefix_cache_reset_evidence` event repeats
                   `probe_cached_tokens_after` and adds the supporting `first_request_cached_tokens` with
                   `first_request_prompt_tokens` (the sub-run's first completed request) and, in the first one, reset
                   lines from the vLLM log. Multi jobs require skeleton_llm_uri for hybrid-skeleton systems.
                   Residual effects of sharing one container: the reset clears cached KV blocks but not the CUDA
                   graphs, allocator state or GPU clocks that earlier runs warmed, so later sub-runs can be faster, and
                   non-batch-invariant kernels can make results depend on the concurrent load. Repeats here are
                   `repeat_kind: "same_container"`: they measure only within-server nondeterminism. Run-to-run repeats
                   for G0 and the pilot are separate jobs (separate containers).
  model            v3: open models only. Default (omitted) = "qwen" = Qwen/Qwen3-4B-Instruct-2507 on a local vLLM server
                   in the job. "gemma" = google/gemma-4-E4B-it (system prompt folded into the user turn). Dict forms:
                   {"kind": "open_local", "name": "qwen"|"gemma"|<hf id>, "max_model_len": 40960, "extra_args": [...],
                    "env": {"VLLM_BATCH_INVARIANT": "1"} (extra server environment), "prefix_caching": false},
                   {"kind": "open", "base_url": ..., "name": ...}, {"kind": "fake", "chat": ...} (tests).
                   {"kind": "api"} is refused (the Anthropic API is not used in v3).
  embedder         paid runs: "precomputed:<npz path or gs:// uri>" (written by opa.tracks.track_a:embed_items;
                   texts that are not precomputed, e.g. agent sub-queries, fall back to harrier on CPU in fp32, built
                   from the file's meta and checked against the file's stored self-check vectors). "harrier" (live
                   microsoft/harrier-oss-v1-0.6b) is REFUSED on a GPU container (only embed_items may use it there);
                   "hash" is for tests; "bge-small" raises.
  skeleton_llm_uri gs:// or local directory holding <item_id>.json from build_skeletons (required for S-llm/S-hyb
                   systems unless `compute_llm_skeleton: true`)
  B                origin size (512 default; 1024/2048 via opa.data.schema.rechunk)
  opts             controller options: M, read_window, P_pin, token_budget, lambda, arm, beam, k, fault_top,
                   max_faults, tau_thread, tau_x, k_max, b, m, delta, context_limit, ...
  workers          items in parallel (default 8)
"""
from __future__ import annotations

import collections
import gzip
import io
import json
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .index import HashEmbedder, Session
from . import config
from .embed import (HFEmbedder, PrecomputedEmbedder, compute_item_embeddings, default_max_length, embeddings_to_bytes,
                    resolve_revision, selfcheck)
from .embed import SELFCHECK_TEXTS, check_against_reference, reference_embeddings
from .models import OpenAIChat, Usage, ask_json, parse_json, salvage_members
from .systems import ORACLE_SYSTEMS, SYSTEMS, Ctx, run_item, session_cfg

SKELETON_SYSTEM = """List the named entities in each archive origin: people, organisations, companies, products, places, identifiers, part numbers, and other proper names. Copy each name exactly as written. Reply with one JSON object and nothing else, written compactly on a single line without spaces or line breaks between tokens: {"entities":{"<origin id>":["<name>","<name>"],"<origin id>":[...]}}"""
# sllm-v2: 4 origins per call, max_tokens 1800, compact JSON, per-origin salvage of a cut-off reply, strict run_system check.
# v1 (8 origins, 900 tokens, pretty-printed JSON) was truncated in 82% of calls and left 68% of origins empty.
SKELETON_PROMPT_VERSION = "sllm-v2"
SKELETON_PER_CALL = 4
SKELETON_MAX_TOKENS = 1800
SKELETON_MAX_EMPTY_FRACTION = 0.5


# ---------------------------------------------------------------- items and caches


GCS_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)       # 4 retries after the first attempt, for transient read errors only
_gcs_sleep = time.sleep
_gcs_lock = threading.Lock()
_gcs_total = {"retries": 0}
_gcs_run = {"log": None, "base": 0}          # set by select_items: the running job's event logger and the retry total at its start


class NotFound(Exception):
    """The object (or local file) does not exist. The only condition that callers may treat as "missing"."""


def is_not_found(e: BaseException) -> bool:
    return isinstance(e, (NotFound, FileNotFoundError)) or any(c.__name__ == "NotFound" for c in type(e).__mro__) \
        or getattr(e, "code", None) == 404


def read_bytes(uri: str, stats: dict | None = None) -> bytes:
    """Bytes of a gs:// object or a local file. A missing object raises NotFound at once; any other exception (rate limit,
    timeout, connection reset) is retried after 0.5, 1, 2 and 4 s and then raised as itself. Every retry is counted in
    stats["retries"] (if given) and in the process-wide total."""
    for attempt in range(len(GCS_RETRY_DELAYS) + 1):
        try:
            if uri.startswith("gs://"):
                from ... import gcs
                return gcs.get_bytes(uri)
            return Path(uri).read_bytes()
        except Exception as e:
            if is_not_found(e):
                raise NotFound(uri) from e
            if attempt == len(GCS_RETRY_DELAYS):
                raise
            with _gcs_lock:
                _gcs_total["retries"] += 1
                if stats is not None:
                    stats["retries"] = stats.get("retries", 0) + 1
            log = _gcs_run["log"]
            if log:                                              # one event per retry, so a transient cause can be found later
                log({"action": "gcs_retry", "uri": uri, "attempt": attempt + 1, "sleep_s": GCS_RETRY_DELAYS[attempt],
                     "exc_type": type(e).__name__, "exc": str(e)[:300]})
            _gcs_sleep(GCS_RETRY_DELAYS[attempt])


def load_items(uri, bucket: str | None = None) -> list[dict]:
    """One URI or a list of URIs; `{bucket}` in a URI is replaced by the job's bucket."""
    if isinstance(uri, (list, tuple)):
        return [it for u in uri for it in load_items(u, bucket)]
    if bucket and "{bucket}" in uri:
        uri = uri.replace("{bucket}", bucket)
    data = read_bytes(uri)
    if uri.endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _bucket(ctx) -> str:
    return (getattr(ctx, "spec", None) or {}).get("bucket", "")


def select_items(args: dict, ctx) -> list[dict]:
    """`items_uri` may be one URI or a list; `offset` and `n_items` apply to each URI separately."""
    uris = args["items_uri"] if isinstance(args["items_uri"], (list, tuple)) else [args["items_uri"]]
    _gcs_run["log"], _gcs_run["base"] = getattr(ctx, "log", None), _gcs_total["retries"]       # retries counted from here
    out = []
    for u in uris:
        off, n = int(args.get("offset", 0)), args.get("n_items")
        if isinstance(u, dict):                                  # per-URI {uri, n_items, offset}
            off, n, u = int(u.get("offset", off)), u.get("n_items", n), u["uri"]
        its = load_items(u, _bucket(ctx))[off:]
        out += its[: int(n)] if n is not None else its           # n_items: 0 means zero items
    return out


def load_json(uri: str, stats: dict | None = None):
    """Parsed JSON, or None when the object does not exist. Transient read errors are retried (read_bytes) and, if they
    persist, raised: they are never reported as "missing"."""
    try:
        return json.loads(read_bytes(uri, stats))
    except NotFound:
        return None


def config_bucket() -> str:
    try:
        import yaml
        p = Path(__file__).resolve().parents[3] / "config" / "project.yaml"
        from ... import gcs
        return gcs.normalize_bucket(str(((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("gcp") or {}).get("bucket", "")))
    except Exception:
        return ""


def make_embedder(name: str, *, max_length: int | None = None, B: int = 512, allow_live_gpu: bool = False,
                  bucket: str | None = None, dtype: str = "float32", batch: int | None = None):
    """"hash" (tests) | "precomputed:<path-or-gs-uri>" (paid runs) | "harrier" (alias "default") | a Hugging Face id.
    Live harrier on a GPU container is refused unless `allow_live_gpu` (only the embedding job sets it): paid runs
    read precomputed vectors, and the live fallback inside them runs fp32 on CPU from the file's own meta."""
    if name == "hash":
        return HashEmbedder()
    if name.startswith("precomputed:"):
        uri = name.split(":", 1)[1]
        if "{bucket}" in uri:                                  # same substitution as select_items; config as fallback
            from ... import gcs
            uri = uri.replace("{bucket}", gcs.normalize_bucket(bucket) if bucket else config_bucket())
        data = read_bytes(uri)
        pe = PrecomputedEmbedder(data)

        def live():                                        # same model, revision, task, max_length, fp32 as the file
            mt = pe.meta
            return HFEmbedder(mt["model"], device="cpu", max_length=mt.get("max_length"), task=mt.get("task", pe.task),
                              revision=mt.get("revision"), B=int(mt.get("B", 512)), dtype=mt.get("dtype", "float32"),
                              query_style=mt.get("query_style"))

        pe.fallback = live
        return pe
    if "bge-small" in name:
        raise ValueError("bge-small was replaced by microsoft/harrier-oss-v1-0.6b in v3; use 'harrier' or 'precomputed:...'")
    if name in ("harrier", "default"):
        name = config.models()["embedder"]
    import torch
    if torch.cuda.is_available() and not allow_live_gpu:
        raise RuntimeError("live embedder on a GPU container is refused: compute embeddings once with "
                           "opa.tracks.track_a:embed_items and pass embedder='precomputed:<uri>'")
    kw = {"batch": batch} if batch else {}
    return HFEmbedder(name, max_length=max_length or default_max_length(B), B=B, revision=config.pinned_revision(name),
                      dtype=dtype, **kw)


_VLLM = {}
_VLLM_PROV = {}
_PROCS = {}
_LOGS = {}


def stop_local_vllm(model: str | None = None, grace_s: float = 30.0) -> list[str]:
    """Terminate the vLLM server(s) this process started (all of them if `model` is None) and forget them, so a later
    job in a reused container can never inherit a warm server (prefix cache, CUDA graphs). Returns the models stopped."""
    stopped = []
    for m in [model] if model else list(_PROCS):
        proc = _PROCS.pop(m, None)
        _VLLM.pop(m, None)
        fh = _LOGS.pop(m, None)
        if proc is None:
            if fh:
                fh.close()
            continue
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_s)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=10)
        finally:
            if fh:
                fh.close()
            stopped.append(m)
    return stopped


def vllm_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return None


# Environment overrides for the vLLM server.
#  VLLM_SERVER_DEV_MODE=1           exposes POST /reset_prefix_cache (between sub-runs).
#  VLLM_USE_FLASHINFER_SAMPLER=0    use vLLM's PyTorch sampler. vLLM 0.32.0 JIT-compiles FlashInfer's
#     top_k_top_p_sampling_from_logits during warm-up, which needs nvcc, and the inference image has none (the first paid
#     smoke died there). v0.32.0 reads the variable in envs.py (default True) via flashinfer_sampler_supported() in
#     vllm/v1/sample/ops/topk_topp_sampler.py, which the GPU-worker sampler (v1/worker/gpu/sample/sampler.py) also uses
#     for its fused path. We decode at temperature 0, and greedy requests never take the fused path, so this cannot
#     change outputs.
VLLM_ENV = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}

LOG_ERROR_PATTERN = r"Error|error|Traceback|CUDA|memory|OOM|ValueError|not supported|Exception"


def gpu_env_info() -> dict:
    """vLLM / torch versions, CUDA device name and free / total memory (best effort; fields are None when unavailable)."""
    info = {"vllm_version": vllm_version(), "torch_version": None, "cuda_available": None, "device_name": None,
            "free_bytes": None, "total_bytes": None}
    try:
        import torch
        info.update(torch_version=torch.__version__, cuda_available=bool(torch.cuda.is_available()))
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["free_bytes"], info["total_bytes"] = (int(x) for x in torch.cuda.mem_get_info())
    except Exception as e:
        info["error"] = repr(e)[:200]
    return info


def vllm_failure_message(why: str, log_path: str = "/tmp/vllm.log") -> tuple[str, bytes]:
    """(exception text, full log bytes): the first ~40 lines that look like errors plus the last 1500 characters."""
    import re
    try:
        data = open(log_path, "rb").read()
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace")
    pat = re.compile(LOG_ERROR_PATTERN)
    hits = [ln.rstrip()[:400] for ln in text.splitlines() if pat.search(ln)][:40]
    msg = (f"{why}\n--- first {len(hits)} error-like lines of the vLLM log ---\n" + "\n".join(hits) +
           "\n--- last 1500 characters ---\n" + text[-1500:])
    return msg, data


def _fail_vllm(model: str, why: str, ctx=None):
    """Upload the FULL vLLM log to the run folder (vllm.log) and raise with the root-cause lines in the message."""
    fh = _LOGS.get(model)
    try:
        if fh:
            fh.flush()
    except Exception:
        pass
    msg, data = vllm_failure_message(why)
    if ctx is not None:
        try:
            ctx.put_bytes("vllm.log", data, "text/plain")
            ctx.log({"action": "vllm_failed", "model": model, "why": why, "log_bytes": len(data)})
        except Exception:
            pass
    stop_local_vllm(model)
    raise RuntimeError(msg)


def start_local_vllm(model: str, max_len: int = config.MAX_MODEL_LEN, port: int = 8000, wait_s: int = 1500,
                     extra_args: list[str] | None = None, ctx=None, env_extra: dict | None = None,
                     prefix_caching: bool = True) -> str:
    """Start `vllm serve` inside this (GPU) job container and wait until it answers. Returns the base URL.
    Sized for an L4 (24 GB): bf16, window 40960, 90% memory, prefix caching on (config.vllm_args). The model
    revision is resolved first (pinned in config/project.yaml `model_revisions`, else the Hub head) and passed to
    vLLM, so the commit recorded in `_VLLM_PROV[model]` is the one served."""
    import subprocess
    import time
    import httpx
    if model in _VLLM:
        return _VLLM[model]
    if ctx is not None:                                           # flushed even if the start fails: environment first
        ctx.log({"action": "vllm_preflight", "model": model, "max_model_len": max_len, "server_env": {**VLLM_ENV, **{k: str(v) for k, v in (env_extra or {}).items()}},
                 "prefix_caching": prefix_caching, "extra_args": list(extra_args or []),
                 **gpu_env_info()})
    rev = resolve_revision(model, config.pinned_revision(model))
    _VLLM_PROV[model] = {"model": model, "revision": rev, "vllm_version": vllm_version(), "dtype": "bfloat16",
                         "max_model_len": max_len, "seed": 0, "prefix_caching": prefix_caching,
                         "server_env": {**VLLM_ENV, **{k: str(v) for k, v in (env_extra or {}).items()}},
                         "extra_args": list(extra_args or [])}
    cmd = config.vllm_args(model, port, max_len, extra_args, revision=rev, prefix_caching=prefix_caching)
    import os
    env = dict(os.environ, **VLLM_ENV, **{k: str(v) for k, v in (env_extra or {}).items()})   # e.g. VLLM_BATCH_INVARIANT
    logfh = open("/tmp/vllm.log", "w")
    proc = subprocess.Popen(cmd, stdout=logfh, stderr=subprocess.STDOUT, env=env)
    _PROCS[model], _LOGS[model] = proc, logfh
    url, t0 = f"http://127.0.0.1:{port}", time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            _fail_vllm(model, f"vLLM exited with code {proc.poll()} before becoming ready", ctx)
        try:
            if httpx.get(url + "/health", timeout=5).status_code == 200:
                _VLLM[model] = url
                _VLLM_PROV[model].update(vllm_ready_at=datetime.now(timezone.utc).isoformat(),
                                         vllm_start_seconds=time.time() - t0)
                return url
        except Exception:
            pass
        time.sleep(5)
    _fail_vllm(model, f"vLLM did not become ready within {wait_s} s", ctx)


def make_chat(args: dict, ctx):
    """Open models only (v3). Default: Qwen3-4B-Instruct-2507 on a local vLLM server, temperature 0, seed per request."""
    m = args.get("model", "qwen")
    if isinstance(m, str):
        m = {"kind": "open_local", "name": m}
    kind = m.get("kind", "open_local")
    if kind == "api":
        raise RuntimeError("model kind 'api' is refused in v3: every experiment call goes to an open model "
                           "(use 'qwen' or 'gemma'); the Anthropic API is not used")
    if kind == "fake":
        return m["chat"]
    name = config.resolve_model(m.get("name"))
    fold = bool(m.get("fold_system", config.needs_system_fold(name)))
    seed = int(m.get("seed", (getattr(ctx, "spec", None) or {}).get("seed", 0)))
    if kind == "open":
        return OpenAIChat(m["base_url"], name, fold_system=fold, seed=seed, provenance={"model": name, **m.get("provenance", {})})
    if kind == "open_local":
        url = start_local_vllm(name, int(m.get("max_model_len", config.MAX_MODEL_LEN)), extra_args=m.get("extra_args"), ctx=ctx,
                               env_extra=m.get("env"), prefix_caching=bool(m.get("prefix_caching", True)))
        ctx.log({"action": "vllm_ready", "model": name, **_VLLM_PROV.get(name, {})})
        return OpenAIChat(url, name, fold_system=fold, seed=seed, provenance=dict(_VLLM_PROV.get(name, {"model": name})))
    raise ValueError(f"unknown model kind {kind!r}")


def _entities_for(chat, chunk, usage, tap) -> tuple[dict, str]:
    """One request for `chunk`; returns ({origin_id: [names]} for every origin that could be read, reply text)."""
    user = "\n".join(f'<origin id="{o["origin_id"]}">\n{o["text"]}\n</origin>' for o in chunk)
    text = chat.ask(SKELETON_SYSTEM, user, SKELETON_MAX_TOKENS, usage, tap)
    j = parse_json(text)
    ents = j.get("entities") if isinstance(j, dict) and isinstance(j.get("entities"), dict) else None
    if ents is None:
        ents = salvage_members(text, "entities")                 # keep every complete origin entry before the cut
    got = {}
    for o in chunk:
        v = ents.get(str(o["origin_id"]), ents.get(o["origin_id"]))
        if isinstance(v, list):
            got[str(o["origin_id"])] = [str(x) for x in v if isinstance(x, (str, int))][:64]
    return got, text


def llm_skeleton(item: dict, chat, usage: Usage, log, per_call: int = SKELETON_PER_CALL, stats: dict | None = None) -> dict:
    """S-llm: the model lists the entities of each origin with a fixed prompt, `per_call` origins per request. Never
    shown the question. A cut-off or malformed reply keeps every complete origin entry; the origins it lost are asked
    again once, alone. `stats` (filled in place) records n_calls, n_parse_errors (replies that were not valid JSON),
    n_truncated (finish_reason length), n_missing (origins never read) and empty_origin_fraction (origins whose
    list is empty, including missing ones)."""
    st = stats if stats is not None else {}
    st.update(n_calls=0, n_parse_errors=0, n_truncated=0, n_missing=0, n_origins=0, n_empty=0)

    def tap(ev):
        if ev.get("action") == "llm_call":
            st["n_calls"] += 1
            st["n_truncated"] += ev.get("finish_reason") == "length"
        if log:
            log(ev)

    out: dict[str, list[str]] = {}
    origins = item["origins"]
    for i in range(0, len(origins), per_call):
        chunk = origins[i:i + per_call]
        got, text = _entities_for(chat, chunk, usage, tap)
        if parse_json(text) is None:
            st["n_parse_errors"] += 1
        missing = [o for o in chunk if str(o["origin_id"]) not in got]
        if missing:                                              # the cut lost these origins: ask for them alone, once
            got2, text2 = _entities_for(chat, missing, usage, tap)
            if parse_json(text2) is None:
                st["n_parse_errors"] += 1
            got.update(got2)
            missing = [o for o in chunk if str(o["origin_id"]) not in got]
        st["n_missing"] += len(missing)
        for o in chunk:
            out[str(o["origin_id"])] = got.get(str(o["origin_id"]), [])
    st["n_origins"] = len(out)
    st["n_empty"] = sum(1 for v in out.values() if not v)
    st["empty_origin_fraction"] = st["n_empty"] / max(1, len(out))
    return out


# ---------------------------------------------------------------- entrypoints


def stops_vllm(fn):
    """Entrypoint decorator: whatever happens, terminate the vLLM server(s) this job started (never leave a warm
    server for a later job in a reused container). Use on every entrypoint that calls make_chat."""
    import functools

    @functools.wraps(fn)
    def wrapper(args, ctx):
        try:
            return fn(args, ctx)
        finally:
            stop_local_vllm()
    return wrapper


def build_skeletons(args: dict, ctx) -> dict:
    try:
        return _build_skeletons(args, ctx)
    finally:
        stop_local_vllm()                                        # never leave a warm server for a later job


def _build_skeletons(args: dict, ctx) -> dict:
    items = select_items(args, ctx)
    chat = make_chat(args, ctx)
    usage = Usage()
    lock = threading.Lock()
    per_item: dict[str, dict] = {}

    def one(item):
        st: dict = {}
        sk = llm_skeleton(item, chat, usage, ctx.log, stats=st)
        bad = st["empty_origin_fraction"] > SKELETON_MAX_EMPTY_FRACTION
        st.update(status="error" if bad else "ok")
        if bad:
            st["error"] = (f"{st['n_empty']}/{st['n_origins']} origins have an empty skeleton "
                           f"(> {SKELETON_MAX_EMPTY_FRACTION:.0%}); n_truncated={st['n_truncated']}, n_missing={st['n_missing']}")
        with lock:
            per_item[item["item_id"]] = st
            ctx.put_json(f"skeleton_llm/{item['item_id']}.json", {"item_id": item["item_id"], "model": chat.name,
                                                                   "prompt_version": SKELETON_PROMPT_VERSION, "status": st["status"],
                                                                   "error": st.get("error"), "stats": st, "entities": sk})
        return len(sk)

    with ThreadPoolExecutor(int(args.get("workers", 8))) as ex:
        n = sum(ex.map(one, items))
    tot = lambda k: sum(st[k] for st in per_item.values())
    errors = [k for k, st in per_item.items() if st["status"] == "error"]
    metrics = {"items": len(items), "origins": n, "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
               "calls": usage.calls, "model": chat.name, "prompt_version": SKELETON_PROMPT_VERSION,
               "per_call": SKELETON_PER_CALL, "max_tokens": SKELETON_MAX_TOKENS, "n_calls": tot("n_calls"),
               "n_parse_errors": tot("n_parse_errors"), "n_truncated": tot("n_truncated"), "n_missing": tot("n_missing"),
               "empty_origin_fraction": tot("n_empty") / max(1, tot("n_origins")), "n_item_errors": len(errors),
               "item_errors": errors[:20], "per_item": per_item}
    if errors:                                                   # fail loudly, but keep every output and the metrics
        ctx.put_json("metrics.json", metrics)
        raise RuntimeError(f"{len(errors)}/{len(items)} items have more than {SKELETON_MAX_EMPTY_FRACTION:.0%} empty "
                           f"origin skeletons, e.g. {errors[:3]}; see metrics.json / skeleton_llm/<item>.json")
    return {"metrics": metrics}


def embed_items(args: dict, ctx) -> dict:
    """Embed every origin and question of a set of items once (harrier by default) and write `embeddings.npz`
    (format in embed.py) so every system reads identical vectors (`embedder: "precomputed:<uri>"`).
    Args: items_uri, n_items, offset, embedder ("harrier" | "hash" | an HF id such as "Qwen/Qwen3-Embedding-4B"),
    B (rechunk size, default 512), batch, dtype ("float32" default | "bfloat16" | "float16"; recorded in the npz meta),
    max_length (default max(1024, B + 64)). A live HF embedder is checked against the reference implementation
    (sentence-transformers, else the model card's snippet) on 3 fixed strings: cosine > 0.999 for float32, > 0.99 for
    bfloat16/float16, or the job fails."""
    items = select_items(args, ctx)
    if int(args.get("B", 512)) != 512:
        from ...data.schema import rechunk
        items = [rechunk(it, int(args["B"])) for it in items]
    B = int(args.get("B", 512))
    dtype = args.get("dtype", "float32")
    emb = make_embedder(args.get("embedder", "harrier"), max_length=args.get("max_length"), B=B, allow_live_gpu=True,
                        bucket=_bucket(ctx), dtype=dtype, batch=args.get("batch"))
    check = None
    if isinstance(emb, HFEmbedder):
        # raises RuntimeError below the cosine threshold
        check = selfcheck(emb, min_cos=0.999 if dtype == "float32" else 0.99)
        ctx.log({"action": "embedder_selfcheck", "cosines": check, **emb.provenance()})
    arrays = compute_item_embeddings(items, emb)
    ctx.put_bytes("embeddings.npz", embeddings_to_bytes(arrays), "application/octet-stream")
    return {"metrics": {"items": len(items), "origins": int(arrays["doc_emb"].shape[0]), "dim": int(arrays["doc_emb"].shape[1]),
                        "model": getattr(emb, "name", "?"), "task": str(getattr(emb, "task", "")),
                        "selfcheck_cosines": check, "provenance": json.loads(str(arrays["meta"]))}}


def run_system(args: dict, ctx) -> dict:
    try:
        return _run_system(args, ctx)
    finally:
        stop_local_vllm()                                        # never leave a warm server for a later job


def _run_system(args: dict, ctx) -> dict:
    """One system (`system`) or several systems x repeats (`systems`, `repeats`) in one job; see the module docstring."""
    systems = list(args["systems"]) if args.get("systems") else [args["system"]]
    repeats = int(args.get("repeats", 1))
    multi = bool(args.get("systems")) or repeats > 1
    opts = dict(args.get("opts", {}))
    opts.setdefault("seed", ctx.spec.get("seed", 0) if hasattr(ctx, "spec") else 0)
    ctx.log({"action": "job_start", "entrypoint": "run_system", "systems": systems, "repeats": repeats,
             "embedder": args.get("embedder", "harrier"), "model": args.get("model", "qwen"), "n_items": args.get("n_items")})
    items = select_items(args, ctx)
    if int(args.get("B", 512)) != 512:
        from ...data.schema import rechunk
        items = [rechunk(it, int(args["B"])) for it in items]
    embedder = make_embedder(args.get("embedder", "harrier"), B=int(args.get("B", 512)), bucket=_bucket(ctx))
    if multi and not args.get("skeleton_llm_uri") and any(s in SYSTEMS and SYSTEMS[s][0] in ("llm", "hyb", "oracle")
                                                           for s in systems):
        raise ValueError("multi-system / repeat jobs need skeleton_llm_uri for hybrid-skeleton systems "
                         "(no per-sub-run skeleton recompute); build it once with build_skeletons")
    t_start = time.time()
    chat = make_chat(args, ctx)                                  # one server start for every (system, repeat)
    server_startup_seconds = time.time() - t_start                 # includes revision lookup + vLLM ready wait
    provenance = {"chat": getattr(chat, "provenance", {}) or {"model": chat.name},
                  "embedder": embedder.provenance() if hasattr(embedder, "provenance") else {"model": getattr(embedder, "name", "?")},
                  "seed": opts["seed"]}
    run_id = getattr(ctx, "run_id", "")
    order = [f"{system}~r{rep}" for system in systems for rep in range(repeats)]
    provenance["run_order"] = order
    provenance["repeat_kind"] = "same_container" if multi else "single"
    ctx.log({"action": "provenance", "systems": systems, "repeats": repeats, **provenance})
    all_metrics, manifest, evidence_logged = {}, [], False
    reset_res = None
    skel_cache: dict = {}                                        # skeleton files read once per item per run, shared by all sub-runs
    current = {}

    def write_outputs(failure=None):
        """sub_runs.json and the root metrics.json; with `failure`, a partial pair (completed sub-runs only)."""
        provenance["n_gcs_retries_total"] = _gcs_total["retries"] - _gcs_run["base"]
        ctx.put_json("sub_runs.json", {"run_id": run_id, "repeat_kind": "same_container", "run_order": order,
                                       "complete": failure is None, "failure": failure,
                                       "server_startup_seconds": server_startup_seconds,
                                       "sub_runs": manifest, "provenance": provenance})
        root = {"runs": all_metrics, "systems": systems, "repeats": repeats, "repeat_kind": "same_container",
                "run_order": order, "complete": failure is None, "failure": failure,
                "server_startup_seconds": server_startup_seconds, "n_gcs_retries_total": provenance["n_gcs_retries_total"],
                "vllm_ready_at": (getattr(chat, "provenance", None) or {}).get("vllm_ready_at"), "provenance": provenance}
        ctx.put_json("metrics.json", root)
        return root

    try:
        for system in systems:
            for rep in range(repeats):
                prefix = f"{system}/r{rep}/" if multi else ""
                sub_id = f"{run_id}~{system}~r{rep}" if multi else ""
                current = {"sub_run_id": sub_id, "system": system, "repeat": rep}
                caching = (getattr(chat, "provenance", None) or {}).get("prefix_caching", True)
                if multi and hasattr(chat, "reset_prefix_cache") and not caching:
                    reset_res = None                             # nothing is cached: no reset, no probe (it would read 0)
                    ctx.log({"action": "prefix_cache_reset", "system": system, "repeat": rep, "sub_run_id": sub_id,
                             "skipped": "prefix caching disabled"})
                elif multi and hasattr(chat, "reset_prefix_cache"):
                    reset_res = reset_and_probe(chat, lambda ev: ctx.log({"system": system, "repeat": rep, "sub_run_id": sub_id, **ev}))
                    ctx.log({"action": "prefix_cache_reset", "system": system, "repeat": rep, "sub_run_id": sub_id, **reset_res})
                m = _run_one(system, rep, prefix, args, ctx, items, embedder, chat, opts, provenance, sub_id, skel_cache)
                if multi and reset_res is not None:              # evidence the reset worked: probe + the first request
                    ev = {"action": "prefix_cache_reset_evidence", "sub_run_id": sub_id,
                          "probe_cached_tokens_after": reset_res.get("probe_cached_tokens_after"),
                          "first_request_cached_tokens": m.get("first_request_cached_tokens"),
                          "first_request_prompt_tokens": m.get("first_request_prompt_tokens")}
                    if not evidence_logged:
                        evidence_logged = True
                        ev["server_log_lines"] = _vllm_log_reset_lines()
                    ctx.log(ev)
                all_metrics[f"{system}/r{rep}"] = m
                manifest.append({"sub_run_id": sub_id, "system": system, "repeat": rep, "dir": prefix.rstrip("/"),
                                 "records": prefix + "records.jsonl.gz", "metrics": prefix + "metrics.json",
                                 "n_records": m.get("n"), **{k: m.get(k) for k in COST_FIELDS}})
    except Exception as e:
        if multi:                                                # keep what finished, name the failing sub-run and why
            write_outputs({**current, "error": f"{type(e).__name__}: {str(e)[:2000]}"})
        raise
    if not multi:
        return {"metrics": all_metrics[f"{systems[0]}/r0"]}
    return {"metrics": write_outputs()}


COST_FIELDS = ("started_at", "finished_at", "wall_seconds", "tokens_in", "tokens_out", "n_llm_calls", "gpu_seconds_share",
               "cached_tokens", "uncached_prompt_tokens")


def reset_and_probe(chat, log=None) -> dict:
    """Reset the prefix cache and verify it with the fixed probe (chats without `probe_cache` get the reset only).
    The probe must be informative: the second probe before the reset has to hit the cache (cached_tokens > 0), else
    prefix caching or prompt-token details are not working and the post-reset check would test nothing; then a
    `prefix_cache_reset` event with probe_informative false is logged through `log` and the job raises."""
    probing = hasattr(chat, "probe_cache")
    out = {}
    if probing:
        chat.probe_cache()                                        # make sure the probe prefix is cached
        b = chat.probe_cache()
        out.update(probe_prompt_tokens=b["prompt_tokens"], probe_cached_tokens_before=b["cached_tokens"],
                   probe_informative=b["cached_tokens"] > 0)
        if not out["probe_informative"]:
            if log:
                log({"action": "prefix_cache_reset", **out})
            raise RuntimeError("cache probe is not informative: the repeated probe had 0 cached tokens before the reset "
                               "(prefix caching or --enable-prompt-tokens-details is not working)")
    res = chat.reset_prefix_cache() or {}
    out.update(success=res.get("success"), attempts=res.get("attempts"))
    if probing:
        a = chat.probe_cache()
        out["probe_cached_tokens_after"] = a["cached_tokens"]
        if res.get("success") is True and a["cached_tokens"] > 0:
            raise RuntimeError(f"prefix cache reset did not take: the probe still hit {a['cached_tokens']} cached tokens "
                               f"after a server-confirmed success")
    return out


def _vllm_log_reset_lines(n: int = 3) -> list[str]:
    """The last lines of the vLLM server log that mention a cache reset (evidence the reset request was served)."""
    try:
        with open("/tmp/vllm.log", encoding="utf-8", errors="replace") as f:
            return [ln.strip()[:300] for ln in f if "reset" in ln.lower() and "cache" in ln.lower()][-n:]
    except Exception:
        return []


def _run_one(system, rep, prefix, args, ctx, items, embedder, chat, opts, provenance, sub_run_id="", skel_cache=None) -> dict:
    needs_llm_sk = system in SYSTEMS and SYSTEMS[system][0] in ("llm", "hyb", "oracle")
    cfg = session_cfg(system, opts)
    lock = threading.Lock()
    records = []
    t0, started_at = time.time(), datetime.now(timezone.utc).isoformat()
    first_cached = []                                            # cached_tokens of the first completed llm_call

    def one(item):
        usage = Usage()
        sk = None
        gcs_stats: dict = {}
        if needs_llm_sk:
            uri = args.get("skeleton_llm_uri")
            uri = uri.replace("{bucket}", _bucket(ctx)) if uri else uri
            key = f"{uri.rstrip('/')}/{item['item_id']}.json" if uri else None
            got = skel_cache.get(key) if (skel_cache is not None and key) else None      # one read per item per run
            if got is None and key:
                got = load_json(key, gcs_stats)             # None only if the file does not exist; other errors raise
                if got:
                    if got.get("prompt_version") != SKELETON_PROMPT_VERSION:
                        raise RuntimeError(f"skeleton for {item['item_id']} has prompt_version {got.get('prompt_version')!r}, "
                                           f"expected {SKELETON_PROMPT_VERSION!r}: rebuild it with build_skeletons")
                    if got.get("status") == "error":
                        raise RuntimeError(f"skeleton for {item['item_id']} failed its quality check: {got.get('error')}")
                    if skel_cache is not None:
                        skel_cache[key] = got
            if got:
                sk = got["entities"]
            elif args.get("compute_llm_skeleton") and not sub_run_id:       # never recomputed inside a sub-run
                st: dict = {}
                sk = llm_skeleton(item, chat, usage, lambda ev, _id=item["item_id"]: ctx.log(
                    {"item_id": _id, "system": system, "repeat": rep, "model": chat.name,
                     **({"sub_run_id": sub_run_id} if sub_run_id else {}), **ev}), stats=st)
                if st["empty_origin_fraction"] > SKELETON_MAX_EMPTY_FRACTION:
                    raise RuntimeError(f"computed skeleton for {item['item_id']} is {st['empty_origin_fraction']:.0%} empty "
                                       f"(n_truncated={st['n_truncated']}, n_missing={st['n_missing']})")
            else:
                raise RuntimeError(f"no S-llm skeleton for {item['item_id']} (set skeleton_llm_uri or compute_llm_skeleton)")
        emb = embedder.for_item(item) if hasattr(embedder, "for_item") else embedder      # precomputed: bind to the item
        sess = Session(item, cfg, emb, skeleton_llm=sk)
        o = dict(opts, _q_emb=emb.embed_query([item["question"]])[0], _embedder=emb)
        tag = {"sub_run_id": sub_run_id} if sub_run_id else {}

        def log_fn(ev, _id=item["item_id"]):
            if ev.get("action") == "llm_call":
                with lock:
                    if not first_cached:
                        first_cached.append((int((ev.get("usage") or {}).get("cached_tokens") or 0),
                                             int((ev.get("usage") or {}).get("prompt_tokens") or 0)))
            ctx.log({"item_id": _id, "system": system, "repeat": rep, "model": chat.name, **tag, **ev})

        if hasattr(emb, "log"):
            emb.log = log_fn                                     # live-embedded texts are logged (sha256 of text, vector)
        c = Ctx(item=item, sess=sess, chat=chat, opts=o, log_fn=log_fn, usage=usage)
        rec = run_item(system, c)
        rec["repeat"] = rep
        rec["n_gcs_retries"] = gcs_stats.get("retries", 0)               # transient storage read errors that were retried
        rec["n_transport_retries"] = usage.n_transport_retries       # retried transport failures (identical policy for all systems)
        rec["cached_tokens"] = usage.cached_tokens              # vLLM prompt_tokens_details.cached_tokens (prefix-cache hits)
        if sub_run_id:
            rec["sub_run_id"] = sub_run_id
        rec["n_threads_detected"] = sess.n_threads
        rec.update(sess.geometry_counts())          # a2_exact_pairs, a2_exact_calls, a2_capped_pairs
        rec["edges"] = sess.edge_types() if cfg["graph"] else {}
        with lock:
            records.append(rec)
        return rec

    with ThreadPoolExecutor(int(args.get("workers", 8))) as ex:
        list(ex.map(one, items))
    records.sort(key=lambda r: r["item_id"])
    for r in records:
        r["run_id"] = getattr(ctx, "run_id", "")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in records:
            gz.write((json.dumps(r, default=str) + "\n").encode())
    ctx.put_bytes(prefix + "records.jsonl.gz", buf.getvalue(), "application/gzip")
    metrics = summarize(items, records, system)
    metrics["repeat"] = rep
    wall = time.time() - t0
    metrics.update(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=wall,
                   n_llm_calls=sum(r.get("n_llm_calls", 0) for r in records),
                   cached_tokens=sum(r.get("cached_tokens", 0) for r in records),
                   uncached_prompt_tokens=sum(r.get("tokens_in", 0) - r.get("cached_tokens", 0) for r in records),
                   gpu_seconds_share=wall)                     # the container serves one sub-run at a time
    metrics["first_request_cached_tokens"] = first_cached[0][0] if first_cached else None
    metrics["first_request_prompt_tokens"] = first_cached[0][1] if first_cached else None
    if sub_run_id:
        metrics.update(sub_run_id=sub_run_id, repeat_kind="same_container")
    # every storage-read retry of the job so far (items, embeddings, skeletons); n_gcs_retries above counts the per-item reads only
    metrics["n_gcs_retries_total"] = provenance["n_gcs_retries_total"] = _gcs_total["retries"] - _gcs_run["base"]
    metrics["provenance"] = provenance
    ctx.put_json(prefix + "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------- embedder check (PLAN_V3 Section 8, check 1)


def recall_at_k(sim_row, gold: list[int], k: int) -> float:
    """Fraction of the gold-fact origins among the k highest-scoring origins (ties broken towards lower id)."""
    import numpy as np
    top = set(int(i) for i in np.argsort(-np.asarray(sim_row), kind="stable")[:k])
    g = {int(x) for x in gold}
    return len(g & top) / max(1, len(g))


def embedder_recall(items: list[dict], emb, ks: list[int]) -> dict:
    """Per k: mean recall of the gold-fact origins and the share of items with every gold origin retrieved,
    scoring each origin by cosine with the item's question embedding (query prefix applied by `emb.embed_query`)."""
    import numpy as np
    rec = {k: [] for k in ks}
    for it in items:
        e = emb.for_item(it) if hasattr(emb, "for_item") else emb
        docs = e.embed([o["text"] for o in it["origins"]])
        q = e.embed_query([it["question"]])[0]
        sims = docs @ q
        gold = it["gold"]["fact_origins"]
        for k in ks:
            rec[k].append(recall_at_k(sims, gold, k))
    return {str(k): {"recall": float(np.mean(v)), "all_gold": float(np.mean([x >= 1.0 for x in v])), "n": len(v)}
            for k, v in rec.items()}


def embedder_check(args: dict, ctx) -> dict:
    """Recall@k of gold-fact origins per embedder, and the PLAN_V3 Section 8 check-1 decision.

    Args: items_uri, n_items, offset (as select_items), M (default 4), ks (default [1, 2, M, 8, 16]; M is always
    included), embedders: [{label, name, dtype?, batch?, max_length?, selfcheck?}] where name is "harrier" |
    "Qwen/Qwen3-Embedding-4B" (or any HF id) | "hash" (tests) | "precomputed:<npz>" (its stored vectors are used as is);
    baseline (label, default the first embedder), challenger (label, default the second).
    Queries get the embedder's own prefix: harrier "Instruct: <task>\\nQuery: <text>", Qwen3-Embedding
    "Instruct: <task>\\nQuery:<text>" (same task string); documents none; last-token pooling by attention mask.
    Live HF embedders run with allow_live_gpu=True and are self-checked against the reference implementation
    (cosine > 0.999 in fp32, > 0.99 for bf16/fp16) or the job fails.
    Dev split only: items must have split == "dev" (and the job split, if given, too) or the job raises.
    Self-check order: the reference vectors are computed first (CPU, then freed), then the embedder is loaded.
    Decision: keep the baseline unless the challenger's mean recall at k = M is >= 3 points (0.03) higher.
    Writes metrics.json and embedder_check.csv (embedder x k)."""
    items = select_items(args, ctx)
    bad = {it.get("split") for it in items} - {"dev"}
    job_split = (getattr(ctx, "spec", None) or {}).get("split", "dev")
    if bad or job_split != "dev":
        raise RuntimeError(f"embedder_check runs on the dev split only (items: {sorted(map(str, bad))}, job: {job_split!r})")
    M = int(args.get("M", 4))
    ks = sorted({int(k) for k in args.get("ks", [1, 2, M, 8, 16])} | {M})
    specs = args["embedders"]
    results, rows = {}, []
    for sp in specs:
        label = sp.get("label") or sp["name"]
        dtype = sp.get("dtype", "float32")
        live_hf = sp["name"] != "hash" and not sp["name"].startswith("precomputed:")
        ref = None
        if live_hf and sp.get("selfcheck", True):
            # reference vectors first (CPU, fp32, weights freed on return), so the GPU model is loaded afterwards and
            # no second fp32 copy of a 4B model is alive next to it
            rid = config.models()["embedder"] if sp["name"] in ("harrier", "default") else sp["name"]
            ref = reference_embeddings(rid, SELFCHECK_TEXTS, sp.get("max_length") or default_max_length(int(args.get("B", 512))),
                                       config.pinned_revision(rid))
        emb = make_embedder(sp["name"], max_length=sp.get("max_length"), B=int(args.get("B", 512)), allow_live_gpu=True,
                            bucket=_bucket(ctx), dtype=dtype, batch=sp.get("batch"))
        check = None
        if ref is not None and isinstance(emb, HFEmbedder):
            check = check_against_reference(emb.embed(SELFCHECK_TEXTS), ref, 0.999 if dtype == "float32" else 0.99)
            ctx.log({"action": "embedder_selfcheck", "label": label, "cosines": check, **emb.provenance()})
        res = embedder_recall(items, emb, ks)
        prov = emb.provenance() if hasattr(emb, "provenance") else {"model": getattr(emb, "name", "?")}
        results[label] = {"recall": res, "provenance": prov, "selfcheck_cosines": check}
        for k in ks:
            rows.append((label, k, res[str(k)]["recall"], res[str(k)]["all_gold"], res[str(k)]["n"]))
        del emb
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    labels = [(sp.get("label") or sp["name"]) for sp in specs]
    base = args.get("baseline", labels[0])
    chal = args.get("challenger", labels[1] if len(labels) > 1 else None)
    decision = {"rule": "keep baseline unless challenger recall@M is >= 0.03 higher", "M": M, "baseline": base,
                "challenger": chal}
    if chal is not None:
        d = results[chal]["recall"][str(M)]["recall"] - results[base]["recall"][str(M)]["recall"]
        decision.update(delta_recall_at_M=d, decision="switch_to_challenger" if d >= 0.03 - 1e-12 else "keep_baseline")
    else:
        decision.update(decision="keep_baseline", delta_recall_at_M=None)
    metrics = {"n_items": len(items), "ks": ks, "M": M, "embedders": results, "decision": decision}
    ctx.put_json("metrics.json", metrics)
    csv = "embedder,k,recall,all_gold_rate,n\n" + "".join(f"{l},{k},{r:.4f},{a:.4f},{n}\n" for l, k, r, a, n in rows)
    ctx.put_bytes("embedder_check.csv", csv.encode(), "text/csv")
    return {"metrics": metrics}


def summarize(items: list[dict], records: list[dict], system: str) -> dict:
    by_id = {it["item_id"]: it for it in items}
    try:
        from ... import scoring
        m = scoring.aggregate([by_id[r["item_id"]] for r in records], records)
        m = m.get(system, m) if isinstance(m, dict) else m
    except Exception as e:                                     # scorer unavailable: plain EM so the run still reports
        from .text import normalize_key
        em = [float(normalize_key(r.get("prediction", "")) in {normalize_key(a) for a in
                    [by_id[r["item_id"]]["answer"]] + by_id[r["item_id"]].get("answer_aliases", [])}) for r in records]
        m = {"em": sum(em) / max(1, len(em)), "scorer_error": repr(e)}
    m.update(system=system, oracle=system in ORACLE_SYSTEMS, n=len(records),
             tokens_in=sum(r.get("tokens_in", 0) for r in records), tokens_out=sum(r.get("tokens_out", 0) for r in records),
             errors=sum(r.get("status") == "error" for r in records))
    # answer parsing (not the scorer's F-codes): counts of how the answer was obtained when the first reply was not clean
    m["answer_parse"] = {k: sum(1 for r in records if r.get("answer_parse") == f"answer_{k}")
                         for k in ("salvaged", "salvaged_retry", "retried", "unparseable", "empty")}
    m["n_transport_retries"] = sum(r.get("n_transport_retries", 0) for r in records)
    m["n_gcs_retries"] = sum(r.get("n_gcs_retries", 0) for r in records)
    if any("gist_pages" in r for r in records):                  # S4: gist harness health (gold used here, after the run only)
        from .systems import gist_harness_failure, s4_harness_failure
        s4 = [r for r in records if "gist_pages" in r]
        m.update(config_version=s4[0].get("config_version"), tuning_iteration=s4[0].get("tuning_iteration"),
                 n_gist_calls=sum(r["n_gist_calls"] for r in s4), n_gist_truncated=sum(r["n_gist_truncated"] for r in s4),
                 n_gist_empty=sum(r["n_gist_empty"] for r in s4), n_gist_parse_errors=sum(r["n_gist_parse_errors"] for r in s4),
                 n_pages=sum(len(r["gist_pages"]) for r in s4), n_lookup_empty=sum(bool(r.get("lookup_empty")) for r in s4),
                 n_lookup_parse_errors=sum(bool(r.get("lookup_parse_error")) for r in s4),
                 gist_harness_failure_rate=sum(gist_harness_failure(by_id[r["item_id"]], r) for r in s4) / len(s4),
                 s4_harness_failure_rate=sum(s4_harness_failure(by_id[r["item_id"]], r) for r in s4) / len(s4))
    m["stop_reasons"] = dict(collections.Counter(r["stop_reason"] for r in records if r.get("stop_reason")))
    m["answer_salvaged_without_citations"] = sum(
        1 for r in records if r.get("answer_parse") in ("answer_salvaged", "answer_salvaged_retry") and not r.get("cited_pin_ids"))
    return m


class LocalCtx:
    """Stand-in for opa.jobs.JobContext for offline runs and tests: events and outputs go to a local folder."""

    def __init__(self, out_dir: str, run_id: str = "local", spec: dict | None = None):
        self.out, self.run_id, self.spec = Path(out_dir), run_id, spec or {"seed": 0}
        self.out.mkdir(parents=True, exist_ok=True)
        self.events, self._lock = [], threading.Lock()

    def log(self, ev: dict):
        with self._lock:
            self.events.append({"run_id": self.run_id, "seq": len(self.events) + 1, **ev})

    def put_json(self, rel, obj):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, default=str))
        return {"uri": str(p)}

    def put_bytes(self, rel, data, content_type="application/octet-stream"):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"uri": str(p)}

    def llm(self, *a, **k):
        raise RuntimeError("no API access in v3; use a fake or open model")
