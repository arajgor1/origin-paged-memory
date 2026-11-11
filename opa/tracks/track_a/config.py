"""v3 constants for Tracks A and A2 (PLAN_V3 Sections 3-5, config/project.yaml). One place for the model ids, the
vLLM server sizing for an L4 (24 GB) and the context-dependent budgets.

Defaults changed from v2 (all of them derive from the move from 128K to 32K items and from the API model to
Qwen3-4B-Instruct-2507 with a 40960-token window):
  token_budget   250_000 -> 100_000  per-question input-token budget. The proportional rescale (62_500) was too tight:
                                     one full fold pass over a ~33K archive measured ~80K input tokens (micro-pilot,
                                     design ruling 2026-11-07). Budget stops stay a reported outcome (stop_reason budget).
  context_limit  180_000 -> 36_000   S1 / D1 limit; a 32K item (32,768 Qwen tokens + origin tags) must fit the 40,960
                                     window with ~4.9K left for system prompt, question and answer
  embedder       bge-small -> harrier-oss-v1-0.6b (precomputed once per item)
  model          claude-sonnet-5-5 -> Qwen3-4B-Instruct-2507 on a local vLLM server (temperature 0)
Unchanged: M=4 slots, read window 3, P_pin 2048, B=512, tau_thread 0.7 (TODO recalibrate on harrier on dev in a paid
P0 job; the 0.7 was calibrated for bge-small and harrier cosines have a different spread).
"""
from __future__ import annotations

from pathlib import Path

MAX_MODEL_LEN = 40960
GPU_MEMORY_UTILIZATION = 0.90
CONTEXT_LIMIT = 36_000
TOKEN_BUDGET_V2, CONTEXT_V2 = 250_000, 128 * 1024
TOKEN_BUDGET = 100_000                                               # proportional rescale would be 62_500 (too tight)
D_CLAMP = None                                                       # A2: clamp only on numerical collapse (36.0 = old fixed clamp)
TAU_THREAD = 0.65                                                    # P0 recalibration on harrier, dev v2 MT 32K, max mean ARI 0.821 (bundle/tables/tau_thread_calibration_dev.csv)

_FALLBACK = {"subject": "Qwen/Qwen3-4B-Instruct-2507", "track_a_replication": "google/gemma-4-E4B-it",
             "embedder": "microsoft/harrier-oss-v1-0.6b"}


def models() -> dict:
    """Model ids from config/project.yaml (falls back to the v3 ids if the file or key is missing)."""
    out = dict(_FALLBACK)
    try:
        import yaml
        p = Path(__file__).resolve().parents[3] / "config" / "project.yaml"
        out.update({k: v for k, v in (yaml.safe_load(p.read_text(encoding="utf-8")).get("models") or {}).items()
                    if k in _FALLBACK})
    except Exception:
        pass
    return out


def pinned_revision(model_id: str) -> str | None:
    """Optional pin from config/project.yaml: `model_revisions: {<hf id>: <commit sha>}`."""
    try:
        import yaml
        p = Path(__file__).resolve().parents[3] / "config" / "project.yaml"
        return ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("model_revisions") or {}).get(model_id)
    except Exception:
        return None


def resolve_model(name: str | None) -> str:
    """'qwen' / 'subject' / None -> subject model; 'gemma' -> replication model; anything else is a literal HF id."""
    ms = models()
    key = (name or "qwen").strip()
    if key.lower() in ("qwen", "subject", "default", "qwen3"):
        return ms["subject"]
    if key.lower() in ("gemma", "replication", "track_a_replication"):
        return ms["track_a_replication"]
    return key


def needs_system_fold(model_id: str) -> bool:
    """Gemma chat templates have no system role: the system prompt is folded into the first user turn."""
    return "gemma" in model_id.lower()


def vllm_args(model: str, port: int = 8000, max_len: int = MAX_MODEL_LEN, extra: list[str] | None = None,
              revision: str | None = None, prefix_caching: bool = True) -> list[str]:
    rev = ["--revision", revision, "--tokenizer-revision", revision] if revision else []
    return ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", str(max_len), "--dtype", "bfloat16", "--seed", "0",
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--enable-prompt-tokens-details"] + (["--enable-prefix-caching"] if prefix_caching else ["--no-enable-prefix-caching"]) \
        + rev + list(extra or [])
