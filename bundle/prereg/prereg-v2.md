# Pre-registration — OPA experiments v2 (as rescoped by PLAN_V3 revisions 1–5)

Filled from `prereg/prereg-v2.template.md`. Committed together with `prereg/sample_sizes.yaml`, then `git tag
prereg-v2` and `prereg/PREREG_TAG.json`. Nothing here may change after the tag; changes go to `prereg/deviations/`
and `bundle/deviations.md`. Funded scope: the owner's approved package (PLAN_V3 revision 5, `budget/approvals.yaml`,
cap $148); what is not funded is listed in Section 2b of the detailed specification and is reported `inconclusive`
("not funded").

- **Date of registration (UTC):** 2026-11-07 (exact commit time in prereg/PREREG_TAG.json)
- **Git commit:** in prereg/PREREG_TAG.json (the commit the tag `prereg-v2` points to)
- **Protocol:** docs/protocol_v2.md (sha256: 338b51a099e08292054fb4bcd7b0ffb797e99ea81c4426c4d7384c8bb6d7cf71), as
  rescoped by PLAN_V3.md revisions 1–5 (where they disagree, PLAN_V3 wins).
- **Verdict rules:** prereg/rules.yaml version 2 with revision-2, revision-4 and revision-5 owner edits (sha256:
  dd79ff37bd2a94e6b6e75d1eedd16b42653caa1199ec321631b0773aefeb0dc5) — written by the human.
- **Sample sizes:** prereg/sample_sizes.yaml — from the P1 pilot (bundle/tables/pilot_dev/) and the owner's package.

## Models
- Subject model, Tracks A, B and C: `Qwen/Qwen3-4B-Instruct-2507` @ cdbee75f17c01a7cc42f958dc650907174af0554, served
  by vLLM 0.32.0 on an L4 (bf16, max_model_len 40,960, prefix caching on), temperature 0.
- Track A replication (P5): `google/gemma-4-E4B-it` @ 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c.
- Embedder: `microsoft/harrier-oss-v1-0.6b` @ f9b9dc8d367d443f2479d27aa5d8d2850c0774ee (kept by pilot check 1);
  robustness backup `Qwen/Qwen3-Embedding-4B` @ 5cf2132abc99cad020ac570b19d031efec650f2b.
- No API model is called by any experiment.

## Data
- Generators: LB-MHQA, MT-LB-MHQA (with the "-conf" thread-confusion variant), drift sessions, thrash tasks —
  `opa.data` GENERATOR_VERSION v2 (Qwen3-4B tokenizer, B = 512, 8K/16K/32K); dev split seeds 0–999; test split seeds
  from 100000, generated only after the tag. Track B synthetic archive (PG-19 filler, `opa/trackb/data.py`): pilot
  eval seeds 900–999, monitoring 2000–2063, G2 fresh-dev 3000+.
- Natural sets (protocol 3.4): not funded (detailed specification, Section 2b). H6 uses LB-MHQA and MT-LB-MHQA test
  items at 16K in place of LongBench (closure item 18).

## Primary contrast per hypothesis
| H | Primary key (rules.yaml) | Metric | Test | Minimum n |
|---|---|---|---|---|
| H1 | S11_vs_S2_lbmhqa_late_32k_primary (cell: LB-MHQA late, h = 2, 32K) | EM | exact McNemar + paired bootstrap CI | 607 (sized for δ 7; power 0.52 at the 5-point threshold, disclosed); S11_vs_S4 key: 391 |
| H2 | D4_vs_D2_retention_t50 | retention at turn 50 | Wilcoxon + bootstrap | 200 — NOT FUNDED (deferred): inconclusive |
| H3 | S11_vs_S8_late_bound; R2 leaf not testable at this scale (revision 3 step 5) | EM | McNemar | 391 (runs at the H1 n, 607); R² leaf not testable, so H3 cannot be supported: refuted if S11_vs_S8 is refuted, else inconclusive (frozen all_of rule) |
| H4 | V3_vs_B1_source_attribution | attribution accuracy | McNemar + bootstrap | 2,451 |
| H5 | A2_vs_A1_agent_mtlbmhqa_32k_primary_cross (cell: MT-LB-MHQA T4, h = 3, 32K, cross) | EM | McNemar + bootstrap; TOST ±3 for A2 vs A3 | 249 (δ 5); TOST 191; model level 940 (A2 arm conditional on E4 room, closure item 15); reported secondary `A2_vs_A1_pages_read_before_gold_hit` (revision 5) |
| H6 | R5_vs_R1_accuracy | EM | McNemar + bootstrap | 400 |
| H7 | measurements | step error vs depth | none (measurement) | — |
| H8 | descent_beam4_vs_flat_recall_at_M (H5 cell) | recall@M | non-inferiority, CI lower bound > −3 | 249 (= H5 n, CPU on the same items) |
| H9 | max_speedup_copy_over_prefill | latency ratio | none (measurement) | — |
| H10 | pinned_vs_unpinned_repeat_pagein_reduction | repeated page-ins per item | Wilcoxon + bootstrap | 860 |
| H11 | V3_paged_vs_B1_full_accuracy_lbmhqa_32k | EM | non-inferiority, CI lower bound > −3 (revision 4) | 2,614 (funded in E4, revision 5) |
| H12 | V3_vs_B1_thread_confusion_reduction_mtlbmhqa_32k | confusion rate | McNemar + bootstrap | 940 |
| H13 | copy_reuse_vs_full_recompute_accuracy_lbmhqa_32k | EM | non-inferiority, CI lower bound > −2 | 5,885 — NOT FUNDED: inconclusive |

## Analysis plan
Protocol Section 7, unchanged: paired bootstrap 10,000 resamples; Holm–Bonferroni over primary p-values (family:
primaries whose rule leaf requires Holm — H1, H3–H6, H10, H12; H2 not funded); non-inferiority keys (H8, H11, H13) read from the CI
lower bound with p_raw null; equivalence margin ±3 EM for A2 vs A3 (TOST). Intention-to-treat everywhere: an
unanswered item is scored as the worst value of its outcome (Section 11 of the detailed specification below).

## Exclusions decided in advance
- Runs that crash after 3 retries are reported as failed and excluded from that cell, with the count reported.
- Per-item failures inside a run are not excluded: intention-to-treat, scored as the worst value of the outcome.
- Items absent from one arm of a pair are excluded with a reason per item and counted.

## Detailed specification
The full specification, including every rule-key definition, gate definition (G0–G4 as revised), disclosure
(revisions 2, 3, 4 and 5 and the results already seen; the closure block at its top), the H3 and H5 cell decisions and every point where the
pre-registration goes beyond the protocol, is the text of `prereg/prereg-v2.DRAFT.md` at the tagged commit, which is
part of this pre-registration.
