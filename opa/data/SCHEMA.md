# Item schemas (the contract between generators, the Track A harness and the scorer)

All datasets are stored as gzip JSON Lines (`items.jsonl.gz`), one item per line, plus a `manifest.json`
(generator version, git SHA, seeds, parameters, counts, SHA-256 of `items.jsonl.gz`).
Local copies live under `<local_data>/<dataset>/<split>/<version>/<cell>/` (`local_data` in config/project.yaml);
the same layout goes to `gs://<bucket>/opa/data/...`.

Splits: `dev` seeds 0–999; `test` seeds >= 100000, generated only after the `prereg-v2` tag (generators must refuse
`split="test"` unless `opa.launch.prereg_tagged()` is true).

Generator version `v2` (PLAN_V3): context lengths 8K/16K/32K (cell labels `8k`, `16k`, `32k`), B = 512, all read from
config/project.yaml (`defaults.context_lengths`, `defaults.origin_tokens_B`, `paths.tokenizer`; CLI overrides in
`opa.data.build`). `v1` (128K, Qwen2.5 tokenizer) stays on disk untouched.

Tokens: lengths and origin sizes are counted with the Qwen3-4B tokenizer (`Qwen/Qwen3-4B-Instruct-2507`); an item's total
origin tokens must be within 2% of `params.length_tokens` (checked at build time). Every origin is a **block** of about 512 tokens of filler plus any inserted sentences. Larger B (1024, 2048)
is obtained by merging 2 or 4 consecutive blocks (`rechunk(item, B)`), so thread runs are aligned to multiples of 4
blocks.

## Common fields (LB-MHQA and MT-LB-MHQA)

```json
{
  "item_id": "mtlbmhqa-dev-T8-h3-32k-cross-000017",
  "dataset": "lbmhqa | mtlbmhqa",
  "split": "dev",
  "seed": 17,
  "generator_version": "v2",
  "cell": "T8-h3-32k-cross",
  "params": {"hops": 3, "length_tokens": 32768, "block_tokens": 512, "placement": "late", "alpha": 0.0,
             "n_threads": 8, "cross_thread": true},
  "origins": [
    {"origin_id": 0, "text": "...", "thread_id": 0, "doc_title": "<PG-19 book title>", "n_tokens": 517,
     "inserted": [{"kind": "fact|distractor|confuser|alias|quote", "char_span": [120, 188]}]}
  ],
  "question": "...",
  "question_thread": 7,
  "answer": "83 °C",
  "answer_aliases": ["83 °C", "83 C", "83 degrees C", "83"],
  "gold": {
    "entities": ["Varrel Pumps", "Ostrand Ltd.", "Kelvane"],
    "aliases": {"Ostrand Ltd.": ["OSTRAND"]},
    "facts": [
      {"hop": 0, "origin_id": 211, "char_span": [40, 98], "text": "Varrel Pumps sources seals from Ostrand Ltd.",
       "subject": "Varrel Pumps", "reveals": "Ostrand Ltd.", "late_bound": false, "cross_thread": false}
    ],
    "fact_origins": [211, 140, 37],
    "h_late": 2,
    "thread_of": [0, 0, 1, 1],
    "cross_thread_hop": 2,
    "confusers": [{"entity": "Kelvanf", "value": "71 °C", "origin_id": 90, "thread_id": 3, "char_span": [10, 60]}],
    "distractors": [{"origin_id": 55, "char_span": [3, 50], "subject": "Dorrin Valves"}]
  }
}
```

- `origins` are in arrival order; `origin_id` equals the list index.
- LB-MHQA: one thread (`thread_id` 0 everywhere), `question_thread` 0, `cross_thread_hop` null, `confusers` [].
- Hop j's fact names `subject` (entity j) and `reveals` (entity j+1); the last hop's fact gives the answer attribute
  of entity `hops-1`. `entities` has `hops` names (entity 0 is named by the question).
- `late_bound` (Definition 6): hop j's fact lies in an origin before the origin where entity j became relevant
  (entity 0 is relevant from the start, i.e. from origin 0, because the question is known when reading starts).
  `h_late` = number of late-bound hops.
- MT-LB-MHQA thread layout (v2): runs of 4 blocks (2 blocks when 4-block runs leave fewer than T + hops runs, i.e.
  T = 16 at 32K; those cells rechunk to B = 512/1024 only). The question thread, which owns the last run, gets at least
  hops + 1 runs; `params.run_blocks` records the run length.
- **Thread-confusion variant** (H12; `dataset: "mtlbmhqa"`, cell suffix `-conf`, e.g. `T8-h3-32k-cross-conf`,
  `params.decoys: true`). Every thread other than the question's holds exactly one decoy of the final fact (same
  template, component and attribute type, different value). The decoys are stored in `gold.confusers` (so
  `opa.scoring.thread_confused` and failure code F8 apply unchanged) with extra fields:
  `kind` (`twin`: one-character twin of the final entity; `suffix`: same stem with another product-type suffix, only
  when the final entity already has a product-type suffix and never a legal form such as Ltd. or GmbH; items whose
  final entity has no such suffix get twin decoys only), `hop` (final hop), `value_aliases`, plus the usual `entity`,
  `value`, `thread_id`, `origin_id`, `char_span`, `text`. A decoy's entity is never an alias of the final entity
  (its stem, STEM or full name); `validate_item` enforces it. No cell uses an identical-name decoy: no arm sees thread
  labels, so it would make the item ambiguous. In all MT cells `gold.confusers` carry `value_aliases` (same alias set as
  `answer_aliases` for that value), and `opa.scoring.thread_confused` matches against them.
  `opa.data.confusion.classify_prediction(item, prediction)` returns `correct`, `wrong_thread` (+ the decoy's thread) or `other`.
- `char_span` is relative to that origin's `text`.
- `answer_aliases` are normalized variants accepted by the scorer (the scorer also normalizes).

## Drift sessions (`dataset: "drift"`, protocol 4.1)

```json
{
  "item_id": "drift-dev-000003", "dataset": "drift", "split": "dev", "seed": 3, "generator_version": "v2",
  "params": {"n_turns": 50, "n_constraints": 10, "n_updates": 3, "threads": ["design", "supplier", "testing"]},
  "turns": [{"turn": 1, "thread": "design", "speaker": "user", "text": "..."}],
  "constraints": [
    {"key": "max_coolant_temp", "label": "maximum coolant temperature", "unit": "°C",
     "initial": {"value": "80", "turn": 1, "thread": "design"},
     "updates": [{"value": "75", "turn": 22, "thread": "testing"}],
     "near_misses": [{"value": "70", "turn": 30, "thread": "supplier", "text": "the old spec said 70 °C"}]}
  ],
  "probes": [{"turn": 10, "gold": {"max_coolant_temp": "80", "...": "..."}},
             {"turn": 25, "gold": {}}, {"turn": 50, "gold": {}}]
}
```

`gold` at a probe is the current value of every constraint after all turns <= the probe turn.

## Thrashing tasks (`dataset: "thrash"`, protocol 3.8)

```json
{
  "item_id": "thrash-dev-r4-000011", "dataset": "thrash", "split": "dev", "seed": 11, "generator_version": "v2",
  "params": {"r": 4, "block_tokens": 512, "n_origins": 32},
  "origins": [{"origin_id": 0, "text": "...", "thread_id": 0, "n_tokens": 510, "inserted": []}],
  "question": "Which of these 4 suppliers quotes the lowest price for part QX-2291: A, B, C, D?",
  "answer": "Brennick Supply", "answer_aliases": ["Brennick Supply", "Brennick"],
  "gold": {"needed_origins": [3, 9, 17, 25], "quotes": [{"supplier": "...", "price": "41.20 USD", "origin_id": 3}]}
}
```

## Per-item result record (written by the Track A harness, read by opa.scoring)

One JSON object per (system, item) in `records.jsonl.gz` of a run:

```json
{
  "run_id": "...", "item_id": "...", "system": "S12", "arm": "A2", "lambda": 2.0, "model": "claude-sonnet-5-5",
  "prediction": "83 °C", "cited_pin_ids": ["p3", "p7"],
  "pins": [{"pin_id": "p3", "origin_id": 140, "char_span": [40, 98], "quote": "...", "evicted": false}],
  "paged_in": [0, 1, 2, 140, 3],
  "evicted": [{"origin_id": 2, "reason": "lru|explicit|budget"}],
  "faults": [{"origin_id": 140, "entity": "Ostrand Ltd.", "via": "skeleton|edge:ent|edge:ref|edge:x|edge:seq|descend|lookup"}],
  "descend_candidates": [140, 77, 102],
  "skeleton": {"<origin_id>": ["normalized entity keys in that origin's skeleton"]},
  "edges_from_resident": {"<gold fact origin_id>": true},
  "tokens_in": 51234, "tokens_out": 812, "api_usd": 0.12, "wall_ms": 41234, "n_llm_calls": 9,
  "status": "ok | error | budget",
  "error": null
}
```

- `skeleton` holds only the gold fact origins' skeletons (enough for gamma_s and failure code F4/F5).
- `edges_from_resident[o]` is true if gold fact origin o was one graph edge from an origin that was resident when the
  fact's entity became relevant (gamma_g); omitted for systems without a graph.
- `descend_candidates` is the union of candidates any descend call returned (empty if the system never descends).

## Python API (opa.data)

- `opa.data.io.write_items(items, path)`, `read_items(path_or_gs_uri) -> list[dict]`, `write_manifest(dir, manifest)`.
- `opa.data.schema.validate_item(item)` raises `ValueError` with a message if a field is missing or inconsistent.
- `opa.data.schema.rechunk(item, B)` returns a copy with origins merged to B tokens (B in {512, 1024, 2048}) and
  gold origin ids, spans and `thread_of` remapped.
- Generators: `opa.data.lbmhqa.make_item(seed, hops, length_tokens, placement, alpha, filler, tokenizer)`,
  `opa.data.mtlbmhqa.make_item(seed, n_threads, hops, length_tokens, cross_thread, alpha, filler, tokenizer)`,
  `opa.data.drift.make_session(seed)`, `opa.data.thrash.make_item(seed, r, filler, tokenizer)`.
- `python -m opa.data.build --dataset lbmhqa --split dev --cell late-h3-32k-a0 --n 100` writes a cell locally;
  `python -m opa.data.build --preset v3-dev --split dev` writes every PLAN_V3 dev cell to `<local_data>/<dataset>/dev/v2/<cell>/`.
