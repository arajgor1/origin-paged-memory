"""Dataset generators for the OPA experiments (work unit P0.5).

Modules
  io        gzip JSON Lines items + manifest
  schema    validate_item, rechunk
  names     invented entity names, alias variants, one-character twins
  filler    PG-19 filler blocks (emozilla/pg19), SyntheticFiller for tests, tokenizer adapters
  lbmhqa    LB-MHQA (protocol 3.2)
  mtlbmhqa  MT-LB-MHQA (protocol 3.3); decoys=True gives the thread-confusion variant (H12, cell suffix -conf)
  drift     50-turn specification sessions (protocol 4.1)
  thrash    joint-reasoning thrashing tasks (protocol 3.8)
  build     CLI that writes one cell to <local_data>/<dataset>/<split>/<version>/<cell>/

Item contract: opa/data/SCHEMA.md.
"""
from __future__ import annotations

import hashlib

import numpy as np

GENERATOR_VERSION = "v2"      # v1 = 128K protocol-v2 cells (Qwen2.5 tokenizer); v2 = PLAN_V3 cells (Qwen3-4B tokenizer)
DEV_SEED_MAX = 999
TEST_SEED_MIN = 100000
BASE_BLOCK_TOKENS = 512
LENGTH_TOLERANCE = 0.02          # item token length must be within 2% of the cell's target length


def data_config() -> dict:
    """Generator defaults from config/project.yaml: context_lengths, block_tokens (origin_tokens_B), tokenizer."""
    from opa.data.filler import project_cfg
    cfg = project_cfg()
    d = cfg.get("defaults", {})
    return {"context_lengths": [int(x) for x in d.get("context_lengths", [8192, 16384, 32768])],
            "block_tokens": int(d.get("origin_tokens_B", BASE_BLOCK_TOKENS)),
            "tokenizer": cfg["paths"]["tokenizer"]}


def default_block_tokens() -> int:
    return data_config()["block_tokens"]


def check_length(length_tokens: int, allowed: list[int] | None = None) -> None:
    """Refuse a context length that is not in config defaults.context_lengths (CLI may pass `allowed` or skip)."""
    allowed = data_config()["context_lengths"] if allowed is None else allowed
    if int(length_tokens) not in allowed:
        raise ValueError(f"length {length_tokens} is not one of the configured context lengths {allowed}")


class SplitRefused(RuntimeError):
    """Raised when test-split data is requested before the prereg-v2 tag exists."""


def check_split(split: str, seed: int) -> None:
    """Dev seeds are 0..999. Test seeds are >= 100000 and only after the prereg-v2 tag (CLAUDE.md Section 6)."""
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    seed = int(seed)
    if split == "test":
        from opa import launch  # lazy: launch shells out to git
        if not launch.prereg_tagged():
            raise SplitRefused("test-split data may only be generated after the prereg-v2 tag (CLAUDE.md Section 6)")
        if seed < TEST_SEED_MIN:
            raise ValueError(f"test-split seeds start at {TEST_SEED_MIN}, got {seed}")
    else:
        if not 0 <= seed <= DEV_SEED_MAX:
            raise ValueError(f"dev-split seeds are 0..{DEV_SEED_MAX}, got {seed}")


def item_rng(dataset: str, cell: str, seed: int) -> np.random.Generator:
    """Independent stream per (dataset, cell, seed): different cells never share items."""
    h = hashlib.sha256(f"opa-data|{GENERATOR_VERSION}|{dataset}|{cell}|{int(seed)}".encode()).digest()
    words = [int.from_bytes(h[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence(words))


def make_item_id(dataset: str, split: str, cell: str | None, seed: int) -> str:
    parts = [dataset, split] + ([cell] if cell else []) + [f"{int(seed):06d}"]
    return "-".join(parts)
