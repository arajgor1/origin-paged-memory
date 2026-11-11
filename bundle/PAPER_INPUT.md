# PAPER_INPUT — Origin-Paged Attention experiments

> FINAL: every hypothesis H1 to H13 with the verdict the frozen rule gives (the verifier recomputes it); Holm-adjusted p-values over every primary that reports a p.

This file is the machine-checked summary of the results behind the paper. Every decimal number, percentage and dollar amount carries a reference tag that resolves to a bundle table (check: `python -m opa.analysis.paper_input_interim`). Integers (counts, steps, depths) are written plain. Numbers in backticks are quotes or code.

## 0. Run summary

- Data as of (UTC, last ledger line): `2026-11-08T07:59:55Z`. Split of the verdict table: `test`.
- Run window (UTC): `2026-11-07T04:46:15Z` to `2026-11-08T07:41:21Z` (first start and last finish among the cited runs, runs/index.csv).
- Models: subject model, all tracks `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`; embedder `microsoft/harrier-oss-v1-0.6b@f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`; Track A replication on Gemma 4 E4B and the embedder robustness check are not run (not funded).
- Pre-registration: tag `prereg-v2` at commit `c8a0a5e910ab`, committed `2026-11-07T18:59:43-05:00`; all 47 test-split runs (38 Track A + 9 Track B) started after it (checked by the UTC timestamp in each run id).
- Gates: G0 passed, G1 passed (owner override, disclosed below), G2 failed, G3 not reached, G4 passed.
- Deviations and disclosures: 16 (listed in full below).
- Holm family: H1, H3, H4, H5, H6, H10 report a p-value and are adjusted together (Holm-Bonferroni, `opa.stats.holm`, as the verifier recomputes it); H2 (inconclusive (not funded)), H12 (not_run_gate_failed (G2)) contribute none (registration Section 5).

## Results

Verdicts: the frozen rules (prereg/rules.yaml) applied by the verifier's own rule evaluator to the claim results. A leaf whose n is below the pre-registered minimum is inconclusive, whatever its point estimate. A hypothesis whose gate failed is `not_run_gate_failed` with the gate named; hypotheses without a funded test are inconclusive with "not funded" in their notes. Intention to treat applies throughout.

### H1 — Fold-left with skeleton and graph faults beats one-shot retrieval on late-binding multi-hop questions — VERDICT: supported

- **Primary result.** S11 vs S2 on LB-MHQA late h=2, 32K, n = 607 (pre-registered minimum 607): 14.2 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S2_lbmhqa_late_32k_primary&cols=value]] points (95% CI 9.2 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S2_lbmhqa_late_32k_primary&cols=ci_lo]] to 19.1 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S2_lbmhqa_late_32k_primary&cols=ci_hi]]), raw p = 0.000000 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S2_lbmhqa_late_32k_primary&cols=p_raw]] `(< 0.000001)`, Holm-adjusted p = 0.000000 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S2_lbmhqa_late_32k_primary&cols=p_holm]] `(< 0.000001)` (over every primary that reports a p).
- **Second leaf.** S11 vs S4 (first 391 H1 items): 39.6 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S4_lbmhqa_late_32k_primary&cols=value]] points (95% CI 33.2 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S4_lbmhqa_late_32k_primary&cols=ci_lo]] to 45.8 [[ref:tables/final_verdicts.csv#row=H1:S11_vs_S4_lbmhqa_late_32k_primary&cols=ci_hi]]); minimum n 391; leaf outcome supported.
- **Rule applied.** `all of (S11_vs_S2_lbmhqa_late_32k_primary >= 5.0 [95% CI lower bound > 0, Holm p < 0.05]; S11_vs_S4_lbmhqa_late_32k_primary >= 3.0)`. Leaf outcome S11 vs S2: supported. Basis: value meets the threshold; CI lower bound above zero; Holm p below the limit. The same estimates would give supported if the minimum n were met (informational only).
- **Arm EMs.** S11 0.77 [[ref:tables/p2_contrasts_test.csv#row=S11_vs_S2_lbmhqa_late_32k_primary&cols=em_a]], S2 0.62 [[ref:tables/p2_contrasts_test.csv#row=S11_vs_S2_lbmhqa_late_32k_primary&cols=em_b]], S4 0.37 [[ref:tables/p2_contrasts_test.csv#row=S11_vs_S4_lbmhqa_late_32k_primary&cols=em_b]] on the shared items.
- **Secondary results.** `S11_vs_S2_slope_per_h_late`, `S11_vs_S2_lbmhqa_late_16k`, `S11_vs_S2_lbmhqa_late_8k`: not run (only the 32K h=2 late cell is funded; prereg Section 2b).
- **Figure.** figures/fig_h1_depth_context_test_with_dev_pilot.png — the H1 contrasts on the single funded cell; the dev pilot is drawn as hollow diamonds for context, the test split as filled circles.
- **Failure analysis.** Intention-to-treat: pairs with an imputed (failed) record 0, pairs excluded for a missing record 0 (tables/p2_contrasts_test.csv).
- **Caveats and deviations.** DEV-G0-noise: S1 repeats differ by near-tie flips, per-item flip rate 20% [[ref:gates/G0.json#criteria.4.observed.per_item_flip_rate]] at n = 50 on G0; the H1 test is sized for 7 points (design power 0.52 [[ref:tables/interim_design.csv#row=h1_design_power_at_5_points&cols=value]] at the 5-point rule threshold, 0.80 [[ref:tables/interim_design.csv#row=h1_design_power_at_7_points&cols=value]] at 7 points; disclosed in the registration). S4 is the v2 baseline (gist cap 300, tuned within the protocol's tuning budget; G1 audit).

### H2 — Extractive session state keeps turn-1 constraints through 50 turns — VERDICT: inconclusive (not funded)

- Status: deferred, not in the funded package (prereg Section 2b): the frozen rule gives inconclusive (required result missing).

### H3 — Measured coverage predicts success as gamma_eff^h_late, and graph edges add recall beyond skeletons — VERDICT: refuted

- **Hypothesis verdict and leg outcome.** H3 cannot be supported because the R2 leaf has no result (closure item 9); the verdict is refuted because the S11 vs S8 leg is refuted (frozen all_of rule). Leg outcome on its own: refuted.
- **Primary result (the testable leg).** S11 vs S8 (graph edges add recall beyond skeletons) on LB-MHQA late h=2, 32K, n = 607 (minimum 391): -1.0 [[ref:tables/final_verdicts.csv#row=H3:S11_vs_S8_late_bound&cols=value]] points (95% CI -2.8 [[ref:tables/final_verdicts.csv#row=H3:S11_vs_S8_late_bound&cols=ci_lo]] to 0.8 [[ref:tables/final_verdicts.csv#row=H3:S11_vs_S8_late_bound&cols=ci_hi]]), raw p = 0.362 [[ref:tables/final_verdicts.csv#row=H3:S11_vs_S8_late_bound&cols=p_raw]], Holm-adjusted p = 1.000 [[ref:tables/final_verdicts.csv#row=H3:S11_vs_S8_late_bound&cols=p_holm]] (over every primary that reports a p).
- **Rule applied.** `all of (R2_predicted_vs_observed_success >= 0.5 [measurement]; S11_vs_S8_late_bound >= 2.0 [95% CI lower bound > 0, Holm p < 0.05])`. Reason: value misses the threshold; CI lower bound not above zero; Holm p not below the limit.
- **Secondary results.** none reported.
- **Figure.** none (the leg shares the cell of the H1 figure).
- **Failure analysis.** shares the H1 runs.
- **Caveats and deviations.** R2 leaf not testable at 32K: the coverage predictor equals its maximum for almost every item, so no theory can fit it (pilot identity-line R2 below zero; disclosed in the registration and in the G1 history).

### H4 — Fresh frames alone alias origins; the origin vector repairs it and adds provenance — VERDICT: refuted

- **Rule applied.** `all of (V0_own_origin_mass_ratio_to_uniform <= 1.5 [measurement]; V3_own_origin_mass_fraction >= 0.9 [measurement]; V3_vs_B1_source_attribution >= 0.0 [95% CI lower bound > 0, Holm p < 0.05])`. Hypothesis verdict refuted; leaves:
  - `V0_own_origin_mass_ratio_to_uniform`: 1.002 [[ref:tables/final_verdicts.csv#row=H4:V0_own_origin_mass_ratio_to_uniform&cols=value]], n = 16; leaf outcome supported; reason: value meets the limit.
  - `V3_own_origin_mass_fraction`: 0.323 [[ref:tables/final_verdicts.csv#row=H4:V3_own_origin_mass_fraction&cols=value]], n = 16; leaf outcome refuted; reason: value misses the threshold.
  - `V3_vs_B1_source_attribution`: -68.4 [[ref:tables/final_verdicts.csv#row=H4:V3_vs_B1_source_attribution&cols=value]] (95% CI -70.3 [[ref:tables/final_verdicts.csv#row=H4:V3_vs_B1_source_attribution&cols=ci_lo]] to -66.5 [[ref:tables/final_verdicts.csv#row=H4:V3_vs_B1_source_attribution&cols=ci_hi]]), n = 2451 (minimum 2451), raw p = 0.000000 [[ref:tables/final_verdicts.csv#row=H4:V3_vs_B1_source_attribution&cols=p_raw]] `(< 0.000001)`, Holm-adjusted p = 0.000000 [[ref:tables/final_verdicts.csv#row=H4:V3_vs_B1_source_attribution&cols=p_holm]] `(< 0.000001)` (over every primary that reports a p); leaf outcome refuted; reason: value misses the threshold; CI lower bound not above zero; Holm p below the limit.
- **Figure.** figures/fig_v3_learning_curve_dev.png — the V3 learning curve on the dev monitoring set (own-origin fraction, attribution, passkey, loss) with the stopping step and the G2 state in the title.
- **Disclosure.** B1's attribution reflects the text origin headers that both V3 and B1 carry (registration Section 12 item 22), so V3 minus B1 measures provenance beyond what the headers give.

### H5 — Hyperbolic distance beats Euclidean distance on the same tree — VERDICT: inconclusive

- **Hypothesis verdict and the agent arm are different things.** The rule is an any_of over the agent arm and the model arm; the model arm is not_run_gate_failed (G2), so a failed agent arm alone leaves the hypothesis inconclusive.
- **Primary result (agent arm).** A2 (S12) vs A1 (S14) on MT-LB-MHQA T4 h=3 cross, 32K, n = 249 (minimum 249): 0.4 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross&cols=value]] points (95% CI -2.0 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross&cols=ci_lo]] to 3.2 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross&cols=ci_hi]]), raw p = 1.000 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross&cols=p_raw]], Holm-adjusted p = 1.000 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross&cols=p_holm]] (over every primary that reports a p). Leaf outcome refuted.
- **Rule applied.** `any of (A2_vs_A1_agent_mtlbmhqa_32k_primary_cross >= 0.0 [95% CI lower bound > 0, Holm p < 0.05]; A2_vs_A1_model_thread_confusion_task >= 0.0 [95% CI lower bound > 0])`. Reason: value meets the threshold; CI lower bound not above zero; Holm p not below the limit.
- **Secondary: A2 vs A3 equivalence (TOST, margin 3 points).** 0.0 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A3_agent_equivalence&cols=value]] points (95% CI -2.6 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A3_agent_equivalence&cols=ci_lo]] to 2.6 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A3_agent_equivalence&cols=ci_hi]]), n = 191, TOST p = 0.030 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A3_agent_equivalence&cols=p_raw]]; reported, no verdict.
- **Secondary: pages read before the first gold page, A2 minus A1.** -0.15 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_pages_read_before_gold_hit&cols=value]] pages (95% CI -0.46 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_pages_read_before_gold_hit&cols=ci_lo]] to 0.06 [[ref:tables/final_verdicts.csv#row=H5:A2_vs_A1_pages_read_before_gold_hit&cols=ci_hi]]), n = 249; reported, no verdict (lower is better).
- **Figure.** figures/fig_h5_thread_depth_test_with_dev_pilot.png — left: the agent-level difference at T4 (the only funded thread depth); right: the model-free Euclidean-over-hyperbolic distortion ratio at T = 4, 8, 16 (dev only).
- **Failure analysis.** see tables/p2_contrasts_test.csv (ITT columns).
- **Caveats and deviations.** `A2_vs_A1_slope_per_thread_depth`: not run (one thread depth funded). The question thread is the largest thread at T8/T16 (structural confound, prereg Section 12 item 18). Track A lambda 2.0 [[ref:tables/interim_design.csv#row=track_a_lambda&cols=value]] was not tuned for S12/S14/S15.

- **Distortion check (dev, model-free).** Euclidean over hyperbolic tree distortion: T4 1.36 [[ref:tables/fig_h5_thread_depth_test_with_dev_pilot.csv#row=dev|T4|distortion_ratio&cols=value]]; T8 1.29 [[ref:tables/fig_h5_thread_depth_test_with_dev_pilot.csv#row=dev|T8|distortion_ratio&cols=value]]; T16 1.25 [[ref:tables/fig_h5_thread_depth_test_with_dev_pilot.csv#row=dev|T16|distortion_ratio&cols=value]].

### H6 — Red-dot prefix and graph-parent co-encoding close most of the contextual-dependence gap — VERDICT: refuted

- **Rule applied.** `all of (best_remedy_recovery_fraction >= 0.9 [measurement]; R5_vs_R1_accuracy >= 0.0 [95% CI lower bound > 0, Holm p < 0.05])`. Hypothesis verdict refuted; leaves:
  - `best_remedy_recovery_fraction`: 0.952 [[ref:tables/final_verdicts.csv#row=H6:best_remedy_recovery_fraction&cols=value]] (95% CI 0.909 [[ref:tables/final_verdicts.csv#row=H6:best_remedy_recovery_fraction&cols=ci_lo]] to 1.016 [[ref:tables/final_verdicts.csv#row=H6:best_remedy_recovery_fraction&cols=ci_hi]]), n = 400; leaf outcome supported; reason: value meets the threshold.
  - `R5_vs_R1_accuracy`: 0.5 [[ref:tables/final_verdicts.csv#row=H6:R5_vs_R1_accuracy&cols=value]] (95% CI -1.5 [[ref:tables/final_verdicts.csv#row=H6:R5_vs_R1_accuracy&cols=ci_lo]] to 2.5 [[ref:tables/final_verdicts.csv#row=H6:R5_vs_R1_accuracy&cols=ci_hi]]), n = 400 (minimum 400), raw p = 0.815 [[ref:tables/final_verdicts.csv#row=H6:R5_vs_R1_accuracy&cols=p_raw]], Holm-adjusted p = 1.000 [[ref:tables/final_verdicts.csv#row=H6:R5_vs_R1_accuracy&cols=p_holm]] (over every primary that reports a p); leaf outcome refuted; reason: value meets the threshold; CI lower bound not above zero; Holm p not below the limit.

### H7 — Global coordinates fail at the predicted depths; re-anchored offsets do not — VERDICT: supported

- **Primary result.** First failing depth of global coordinates: bf16 4, fp16 7, fp32 16 (predicted 4 / 7 / 16; rule bands 3-5, 6-8, 15-17). Re-anchored bf16 error growth, depth 40 over depth 1: 0.961 [[ref:tables/interim_h7.csv#row=reanchored_error_growth_ratio_depth40_vs_depth1_bf16&cols=value]] (limit 2.0 [[ref:tables/interim_design.csv#row=h7_reanchored_growth_limit&cols=value]]).
- **Rule applied.** `all of (reanchored_error_growth_ratio_depth40_vs_depth1_bf16 <= 2.0 [measurement]; 3 <= global_fail_depth_bf16 <= 5 [measurement]; 6 <= global_fail_depth_fp16 <= 8 [measurement]; 15 <= global_fail_depth_fp32 <= 17 [measurement])`. All four measurement leaves met -> supported.
- **Caveats.** A measurement on 512 synthetic chains (split n/a), run after the tag on dev-seeded chains; the re-anchored ratio composes two offsets in the LCA frame of one node, so it demonstrates same-branch precision and cannot show cross-branch error growth.

### H8 — Descent through shells matches flat routing with far fewer scores — VERDICT: supported

- **Primary result.** Descent (beam 4) minus flat scan, recall at M slots, n = 249 (minimum 249): -0.4 [[ref:tables/final_verdicts.csv#row=H8:descent_beam4_vs_flat_recall_at_M&cols=value]] points (95% CI -1.2 [[ref:tables/final_verdicts.csv#row=H8:descent_beam4_vs_flat_recall_at_M&cols=ci_lo]] to 0.0 [[ref:tables/final_verdicts.csv#row=H8:descent_beam4_vs_flat_recall_at_M&cols=ci_hi]]); non-inferiority decided from the CI lower bound, so no p-value.
- **Rule applied.** `descent_beam4_vs_flat_recall_at_M >= -3.0 [95% CI lower bound > -3.0]`. Reason: value meets the threshold; CI lower bound above the margin; both arms at floor (non-inferiority at floor).
- **Floor disclosure.** Arm rates: descent 0.000 [[ref:tables/p2_contrasts_test.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_a]], flat 0.004 [[ref:tables/p2_contrasts_test.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_b]]. Both arms are at floor (every gold origin within M = 4 slots for 0 and 1 of 249 items; dev pilot 0.000 [[ref:tables/p2_dev_check/p2_contrasts_dev.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_a]] vs 0.000 [[ref:tables/p2_dev_check/p2_contrasts_dev.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_b]]), so the supported verdict is non-inferiority at floor and says nothing about routing quality; the number of scores is not compared.
- **Caveats.** Computed on the H5 items with real skeletons and harrier embeddings; no model run (no intention-to-treat case).

### H9 — Page-in by copy beats re-encoding above a measurable chunk size — VERDICT: not_run_gate_failed (G2)

- Status: gate G2 failed (bundle/gates/G2.json): not run, by the pre-registered gate rule; the owner asked for an exploratory request with a cost estimate instead.

### H10 — Pinning prevents thrashing when a step needs more origins than slots — VERDICT: supported

- **Primary result.** Repeated page-ins per item, unpinned minus pinned, n = 860 (minimum 860): 2.54 [[ref:tables/final_verdicts.csv#row=H10:pinned_vs_unpinned_repeat_pagein_reduction&cols=value]] (95% CI 2.27 [[ref:tables/final_verdicts.csv#row=H10:pinned_vs_unpinned_repeat_pagein_reduction&cols=ci_lo]] to 2.83 [[ref:tables/final_verdicts.csv#row=H10:pinned_vs_unpinned_repeat_pagein_reduction&cols=ci_hi]]), raw p = 0.000000 [[ref:tables/final_verdicts.csv#row=H10:pinned_vs_unpinned_repeat_pagein_reduction&cols=p_raw]] `(< 0.000001)`, Holm-adjusted p = 0.000000 [[ref:tables/final_verdicts.csv#row=H10:pinned_vs_unpinned_repeat_pagein_reduction&cols=p_holm]] `(< 0.000001)` (over every primary that reports a p).
- **Rule applied.** `pinned_vs_unpinned_repeat_pagein_reduction >= 0.0 [95% CI lower bound > 0, Holm p < 0.05]`. Reason: value meets the threshold; CI lower bound above zero; Holm p below the limit.
- **Caveats.** Failures are imputed with the worst value of the (r, M) cell (intention to treat).

### H11 — Origin-paged attention keeps the standard model's accuracy while holding at most a quarter of the KV cache resident — VERDICT: not_run_gate_failed (G2)

- Status: gate G2 failed (bundle/gates/G2.json): not run, by the pre-registered gate rule.

### H12 — Origin-anchored attention makes fewer cross-thread mix-ups than standard attention — VERDICT: not_run_gate_failed (G2)

- Status: gate G2 failed (bundle/gates/G2.json): not run, by the pre-registered gate rule.

### H13 — Re-using a saved page by copy keeps full-recompute accuracy at a fraction of the reload time — VERDICT: not_run_gate_failed (G2)

- Status: gate G2 failed (bundle/gates/G2.json): not run; also not funded in the approved package (registration Section 2b).

## Measured numbers for the paper

Each row replaces a synthetic or assumed number in docs/paper_v2.md with a measured one, or states that it is not measured. Old values are in backticks (quotes, skipped by the verifier).

| Paper location | Synthetic or assumed value in v2 | Measured value | Note |
|---|---|---|---|
| §7.2 own-origin mass | `0.913` (positional head, simulation) | 0.323 [[ref:tables/final_verdicts.csv#row=H4:V3_own_origin_mass_fraction&cols=value]] | trained V3 at the stopping step, mean over the test probe windows, all layers; the dev monitoring set gave 0.320 [[ref:tables/trackb_status_dev.csv#row=V3_latest_own_origin_mass_fraction&cols=value]] at step 600 (dev, not a result); not at lambda 5 |
| §8.3 descent vs flat at matched M | simulation only | -0.4 [[ref:tables/final_verdicts.csv#row=H8:descent_beam4_vs_flat_recall_at_M&cols=value]] | real summaries, MT T4 h3 cross 32K, recall points, n = 249; arm recalls descent 0.000 [[ref:tables/p2_contrasts_test.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_a]], flat 0.004 [[ref:tables/p2_contrasts_test.csv#row=descent_beam4_vs_flat_recall_at_M&cols=em_b]]: both arms are at floor, so this is non-inferiority at floor, not evidence about routing quality |
| §11 page-in copy vs prefill | `2.7 ms` at `25 GB/s` (arithmetic only) | not measured | not_run_gate_failed (G2); owner request budget/requests/H9_exploratory.md |
| §(global-coordinate failure depth, bf16) | `4` (predicted) | 4 | table interim_h7 row global_fail_depth_bf16; fp16 7, fp32 16 |
| §(re-anchored bf16 error growth, depth 40 vs 1) | limit 2.0 [[ref:tables/interim_design.csv#row=h7_reanchored_growth_limit&cols=value]] | 0.961 [[ref:tables/interim_h7.csv#row=reanchored_error_growth_ratio_depth40_vs_depth1_bf16&cols=value]] | same-branch precision only |

### Units U1-U8

- U1: pass (DEV-U1); observed 0.000061 [[ref:tables/interim_units.csv#row=U1&cols=observed]] (`6.1e-05`), threshold 0.000010 [[ref:tables/interim_units.csv#row=U1&cols=threshold]] (`1e-05`); float64 max abs error 0.000000 [[ref:tables/interim_units.csv#row=U1&cols=aux_observed]] (`1.14e-13`), fp32 error in ulps of the largest key 2.0 [[ref:tables/interim_units.csv#row=U1&cols=ulps]]. passes the two-part definition of DEV-U1 (float64 error and fp32 error in ulps of the largest key, see the aux columns); the literal 1e-5 absolute criterion is reported, not met: the observed column is the fp32 error
- U2: pass; observed 0.000 [[ref:tables/interim_units.csv#row=U2&cols=observed]]. criterion: logits identical (max abs diff exactly 0)
- U3: pass; observed 0.000 [[ref:tables/interim_units.csv#row=U3&cols=observed]]. criterion: logits identical (max abs diff exactly 0)
- U4: pass; observed 0.000 [[ref:tables/interim_units.csv#row=U4&cols=observed]]. criterion: logits identical; live check changes logits
- U5: pass; observed 0.000000 [[ref:tables/interim_units.csv#row=U5&cols=observed]] (`4.77e-07`), threshold 0.000001 [[ref:tables/interim_units.csv#row=U5&cols=threshold]] (`1e-06`). primitives fp32 error (a); LCA distances float64 and fp32 port vs reference (b, c) in tests/U5.json
- U6: pass; observed 0.961 [[ref:tables/interim_units.csv#row=U6&cols=observed]], threshold 2.0 [[ref:tables/interim_units.csv#row=U6&cols=threshold]]. re-anchored bf16 error growth ratio depth 40 vs 1; global fail depths within one step of the predicted ones
- U7: pass; observed 0.000000 [[ref:tables/interim_units.csv#row=U7&cols=observed]] (`7.77e-16`), threshold 0.000010 [[ref:tables/interim_units.csv#row=U7&cols=threshold]] (`1e-05`); SDPA output max abs error 0.000007 [[ref:tables/interim_units.csv#row=U7&cols=aux_observed]] (`6.61e-06`). attention = router mass x local softmax; fp32 tolerance in the threshold column
- U8: pass (DEV-U8); observed 0.000000 [[ref:tables/interim_units.csv#row=U8&cols=observed]] (`3.33e-16`), threshold 0.000010 [[ref:tables/interim_units.csv#row=U8&cols=threshold]] (`1e-05`); auxiliary SDPA-vs-(a_paged @ v) max abs error 0.000021 [[ref:tables/interim_units.csv#row=U8&cols=aux_observed]] (`2.08e-05`). pre-registered paging identity met (observed column is within the threshold column); the harness flag is False because the auxiliary check (aux columns) exceeds the 1e-5 absolute tolerance (DEV-U8: reported literally, no re-run for a pass)

## Gates

- **G0** passed (decided `2026-11-07T08:25:34Z`, criteria met 3 of 6).
- **G1** passed (decided `2026-11-07T20:02:47Z`, criteria met 4 of 4).
- **G2** failed (decided `2026-11-08T07:14:02Z`, criteria met 1 of 3).
- **G3** not reached (decided `2026-11-08T08:29:56Z`, criteria met 0 of 0).
- **G4** passed (decided `2026-11-08T08:30:32Z`, criteria met 1 of 1).

G0 criteria, with the observed values ("met false" entries are attempts that were superseded by a pre-committed final attempt or a fallback, as the gate record explains; they are kept, not removed):

- S18 oracle EM, LB-MHQA dev late h=3 32K, 50 items (attempt at the default h): 0.62 [[ref:gates/G0.json#criteria.0.observed]]; met false (full criterion text in gates/G0.json).
- S18 oracle EM, LB-MHQA dev late h=2 32K, 50 items (PLAN_V3 s6 fallback re-check; stop only if < `0.60`): 0.62 [[ref:gates/G0.json#criteria.1.observed]]; met true (full criterion text in gates/G0.json).
- S18 oracle EM, LB-MHQA dev late h=3 32K, same 50 items, re-check on rebuilt S-llm skeletons (sllm-v2): 0.60 [[ref:gates/G0.json#criteria.2.observed]]; met false (full criterion text in gates/G0.json).
- S1 three repeats agree within 2 EM points (attempt 1): em [0.44 [[ref:gates/G0.json#criteria.3.observed.em.0]], 0.42 [[ref:gates/G0.json#criteria.3.observed.em.1]], 0.40 [[ref:gates/G0.json#criteria.3.observed.em.2]]]; spread_points 4; differing_items 6; harness_differences 2; met false (full criterion text in gates/G0.json).
- S1 three repeats agree within 2 EM points (attempt 2, after the answer-parsing fix; final by pre-commitment): em [0.38 [[ref:gates/G0.json#criteria.4.observed.em.0]], 0.40 [[ref:gates/G0.json#criteria.4.observed.em.1]], 0.44 [[ref:gates/G0.json#criteria.4.observed.em.2]]]; spread_points 6; differing_items 10; near_tie_flips 10; harness_differences 0; per_item_flip_rate 0.20 [[ref:gates/G0.json#criteria.4.observed.per_item_flip_rate]]; met true (full criterion text in gates/G0.json).
- Track B unit tests U1-U5 on Qwen3-4B (U1 under DEV-U1): U1 true; U2 true; U3 true; U4 true; U5 true; met true (full criterion text in gates/G0.json).

G2 criteria, with the observed values (G2 failed):

- V3_own_origin_mass_fraction (H4 part; trained V3 at the stopping-rule length L = 600, fresh dev seeds 3000+) (threshold >= `0.90`): value 0.314022 [[ref:gates/G2.json#criteria.0.observed.value]]; run `B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108050012-063b486-d41d`; resumed_from_sha256 `f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814`; opt_steps 600; met false.
- V3 - B1 source attribution (H4 part; same fresh dev items, both at L = 600) (threshold > 0): V3 0.34375 [[ref:gates/G2.json#criteria.1.observed.V3]]; B1 1.00 [[ref:gates/G2.json#criteria.1.observed.B1]]; difference_points -65.625 [[ref:gates/G2.json#criteria.1.observed.difference_points]]; runs [`B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108050012-063b486-d41d`, `B-B1-mtlbmhqa-archive-archive64-T8-dev-s0-20261108064551-b01a6df-fbfe`]; B1_resumed_from_sha256 `eeccb50e8f89f95dd82f008a376a6b6b245137c633a3017fd01b10b136ed8bff`; met false.
- best_remedy_recovery_fraction (H6 part; remedies pilot, non-oracle R3-R6) (threshold >= `0.90`): value 0.916667 [[ref:gates/G2.json#criteria.2.observed.value]]; acc_R0 0.50 [[ref:gates/G2.json#criteria.2.observed.acc_R0]]; run `B-R0-R7-mtlbmhqa-archive-T4-h2-8k-nocross-dev-s0-20261107124615-ac57b55-5c1e`; met true.

### G1 and the owner override (disclosure)

G1 was first decided under the protocol's definition at `2026-11-07T18:30:16Z` and failed on H3's R2 criterion alone; H1 passed. The owner then replaced the definition (PLAN_V3 revision 3, same day) and G1 was re-decided and passed. The original decision is kept unchanged in bundle/gates/history/.

- Original G1, the failed criterion: pilot R2 of the H3 predictor -7.38 [[ref:tables/interim_gate_criteria.csv#row=G1-original-2026-11-07T183016Z:2&cols=observed_value]] against a threshold of 0.5 [[ref:tables/interim_design.csv#row=h3_r2_threshold&cols=value]] (the predictor equals its maximum for almost every item at 32K).
- Revision 3 definition: S4 passes the baseline audit and H1 passes on the pilot's point estimates; H3 is not part of G1.
- Results that had been seen when the override was made are listed in the registration (Section 4).

## Deviations and disclosures

Every pre-registered deviation file and every disclosure (16 entries):

- **DEV-TRAIN-PIN** (deviation, `2026-11-08`): V3 and B1 checkpoints are pinned after the prereg-v2 tag, not before (source: prereg/deviations/DEV-TRAIN-PIN.md)
- **DEV-U8** (deviation, `2026-11-08`): U8's auxiliary kernel-consistency check exceeds 1e-5 on the L4 (reported, criterion unchanged) (source: prereg/deviations/DEV-U8.md)
- **DEV-U1** (deviation, `2026-11-07`): U1 two-part definition (float64 error <= 1e-10 and fp32 error <= 4 ulps of the largest key); the literal 1e-5 absolute criterion is below one fp32 ulp at real key magnitudes and is reported, not met (source: prereg draft Section 3 and 9; bundle/gates/G0.json; state/decisions.md 2026-11-07 (pre-tag))
- **DEV-G0-noise** (deviation, `2026-11-07`): G0 S1 repeat criterion passed under the owner's ruling recorded before the results: every difference was a near-tie flip, none a harness error; the measured flip rate and EM spread are in the G0 record and next to every primary contrast (source: prereg draft Section 3; bundle/gates/G0.json; state/decisions.md 2026-11-07 (pre-tag))
- **G1-OVERRIDE** (disclosure, `2026-11-07`): G1 first failed under the protocol's definition (H3 R2 alone), then the owner replaced the definition (PLAN_V3 revision 3); G1 re-decided and passed; the original record is kept in bundle/gates/history/ (source: PLAN_V3 revision 3 Section 6b; prereg draft Section 4; bundle/gates/history/)
- **H3-R2-NOT-TESTABLE** (disclosure, `2026-11-07`): H3's R2 leaf is not testable at 32K (the predicted coverage is at its maximum for almost every pilot item); no result is reported for it, so H3 cannot be supported (refuted if the S11 vs S8 leg is refuted, else inconclusive) (source: prereg draft Section 2b and closure item 9)
- **S12-BEYOND-PROTOCOL** (disclosure, `2026-11-07`): 33 items where the registration goes beyond the protocol (Holm family, TOST via Tango score, intention-to-treat scoring, k and lambda fixed, ...); each is listed in Section 12 of the registration (source: prereg draft Section 12)
- **DEV-G1-OVERRIDE** (deviation, `2026-11-07`): G1 was first decided under the protocol's definition at 18:30 UTC and failed on H3's R2 criterion alone (H1 passed); the owner then overrode the gate in PLAN_V3 revision 3 and G1 was re-decided under the new definition and passed. The original decision is kept in gates/history/. The pilot estimates for H1, H2, H3, the H5 agent arm and H10 had been seen when the override was made; no result yet existed for trained Track B models, Track C, H7, H8 or H9. (source: bundle/deviations.md)
- **DEV-PLAN-REV2** (deviation, `2026-11-07`): G2 is judged after V3 and B1 reach a pre-set stopping rule on a fresh dev set (not on the 30-minute pilot); a 400-step learning-curve check is added; the H5 trend key is renamed A2_vs_A1_slope_per_thread_depth. (source: bundle/deviations.md)
- **DEV-PLAN-REV3** (deviation, `2026-11-07`): G1 redefined (working S4 via a baseline audit, H1 on the pilot's point estimates; H3 removed from G1); H3 is tested on cells chosen by predictor variance alone. (source: bundle/deviations.md)
- **DEV-S4-V2** (deviation, `2026-11-07`): S4 v2 replaced the first S4 (gist cap 300 with salvage, one-origin pages, parallel lookup of up to 5 pages, harness-failure flags). (source: bundle/deviations.md)
- **DEV-PLAN-REV4** (deviation, `2026-11-07`): H11's non-inferiority margin widened from 2 to 3 EM points in prereg/rules.yaml. (source: bundle/deviations.md)
- **DEV-PLAN-REV5** (deviation, `2026-11-07`): The owner approved the 'fits `$126`' package: total cap `$148`, base envelopes above the plan's `$100` limit by owner exception, BOOST `$30` for more H1 test items (use b). (source: bundle/deviations.md)
- **DEV-H3-R2** (deviation, `2026-11-07`): H3's R2 leaf is reported as not testable at 32K: no result is entered for it. (source: bundle/deviations.md)
- **DEV-OWNER-INTERIM** (deviation, `2026-11-08`): Owner instruction (dated 2026-11-07, received 2026-11-08): a preprint on Track A, an interim Track A package, Track A first on contested GPU capacity, vLLM max_containers 8, reservations at `1.5` times measured cost for measured job types, B1 started in parallel with V3, H9 not run if G2 fails (an exploratory request instead). (source: bundle/deviations.md)
- **DEV-FAILED-RUNS** (deviation, `2026-11-08`): Failed and excluded runs: A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9 (p2-h1-lb-c07): replaced by retry 2 of 3 allowed (run A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108055403-10a14e2-6c9a); the retry's records are used; B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108022824-e36fdc0-201d (p4-v3-c01): failed (pre-fix code: the resume path was unresolved); excluded; V3 chunk 1 is the later run on the fixed deploy. (source: bundle/deviations.md)

## Failed and excluded runs

Runs that did not finish ok. None is deleted; each is excluded from the analysis for the reason given.

- `A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9` (job p2-h1-lb-c07, attempt 1 of 3): failed; upload of a run file to storage failed with HTTP 412. Handling: replaced by retry 2 of 3 allowed (run A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108055403-10a14e2-6c9a); the retry's records are used.
  - Sensitivity S11: EM of the first attempt minus the retry 1.2 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S11&cols=diff_points]] points on 86 shared items (first 0.721 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S11&cols=em_first]], retry 0.709 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S11&cols=em_retry]]).
  - Sensitivity S2: EM of the first attempt minus the retry 1.2 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S2&cols=diff_points]] points on 86 shared items (first 0.640 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S2&cols=em_first]], retry 0.628 [[ref:tables/interim_rerun_sensitivity.csv#row=A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9|S2&cols=em_retry]]).
  - S8: no records shared with the retry.
- `B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108022824-e36fdc0-201d` (job p4-v3-c01, attempt 1 of 3): failed; upload of a run file to storage failed with HTTP 412. Handling: failed (pre-fix code: the resume path was unresolved); excluded; V3 chunk 1 is the later run on the fixed deploy.

## Track B status

V3 learning curve on the dev monitoring set (arm V3):

- step 0: own-origin mass fraction 0.137 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step0&cols=own_origin_mass_fraction]], ratio to uniform 0.98 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step0&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.000 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step0&cols=attrib_em]], passkey EM 0.000 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step0&cols=passkey_em]], training loss n/a.
- step 50: own-origin mass fraction 0.183 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step50&cols=own_origin_mass_fraction]], ratio to uniform 1.31 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step50&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.188 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step50&cols=attrib_em]], passkey EM 0.781 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step50&cols=passkey_em]], training loss 3.09 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step50&cols=train_loss]].
- step 100: own-origin mass fraction 0.254 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step100&cols=own_origin_mass_fraction]], ratio to uniform 1.82 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step100&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.172 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step100&cols=attrib_em]], passkey EM 0.875 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step100&cols=passkey_em]], training loss 2.50 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step100&cols=train_loss]].
- step 200: own-origin mass fraction 0.249 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step200&cols=own_origin_mass_fraction]], ratio to uniform 1.78 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step200&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.141 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step200&cols=attrib_em]], passkey EM 0.938 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step200&cols=passkey_em]], training loss 2.47 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step200&cols=train_loss]].
- step 400: own-origin mass fraction 0.307 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step400&cols=own_origin_mass_fraction]], ratio to uniform 2.20 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step400&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.313 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step400&cols=attrib_em]], passkey EM 0.969 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step400&cols=passkey_em]], training loss 2.37 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step400&cols=train_loss]].
- step 600: own-origin mass fraction 0.320 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step600&cols=own_origin_mass_fraction]], ratio to uniform 2.30 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step600&cols=own_origin_mass_ratio_to_uniform]], attribution EM 0.375 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step600&cols=attrib_em]], passkey EM 1.000 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step600&cols=passkey_em]], training loss 2.28 [[ref:tables/v3_learning_curve_dev.csv#row=V3_step600&cols=train_loss]].
- Stopping rule: fired at step 600 (L = 600): the own-origin mass fraction rose by 0.0137 [[ref:tables/trackb_status_dev.csv#row=V3_rise_last_two_evaluations&cols=value]] between steps 400 and 600, below the threshold 0.02 [[ref:tables/trackb_status_dev.csv#row=V3_rise_last_two_evaluations&cols=threshold]].
- Pinned V3 checkpoint (DEV-TRAIN-PIN): final step 600, run `B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108031640-6c44911-9560`, SHA-256 `f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814`.
- Pinned B1 checkpoint (DEV-TRAIN-PIN): final step 600, run `B-B1-mtlbmhqa-archive-archive64-T8-dev-s0-20261108044732-a400cda-cae6`, SHA-256 `eeccb50e8f89f95dd82f008a376a6b6b245137c633a3017fd01b10b136ed8bff`.
- G2: failed (decided `2026-11-08T07:14:02Z`); the criteria and observed values are listed under Gates. The own-origin mass fraction threshold is 0.90 [[ref:tables/interim_design.csv#row=g2_own_origin_threshold&cols=value]], judged on fresh dev seeds.
- DEV-TRAIN-PIN: the V3 and B1 checkpoints are pinned after the tag (the training to the stopping rule ran after it); the configuration, the stopping rule and the G2 design were fixed before it.

## Track C status

- **TrackC**: not run (gate G2 failed) (Track C starts only if G2 passes; G2 failed, so no H9 run; owner request budget/requests/H9_exploratory.md).
- **H9**: not_run_gate_failed (G2); owner request budget/requests/H9_exploratory.md (page-in by copy vs re-encoding; a measurement; gate G2 failed (bundle/gates/G2.json): not run, by the pre-registered gate rule; the owner asked for an exploratory request with a cost estimate instead).
- **H13**: not_run_gate_failed (G2) (gate G2 failed (bundle/gates/G2.json): not run; also not funded in the approved package (registration Section 2b)).

## Framing implied by the results

- H5: the agent-level contrast is read together with the distortion ratio; only the T4 agent cell is funded, so no statement about thread depth beyond T4 follows.
- Evidence-backed: the Track A contrasts above, the paging identity and precision measurements and, where listed, the H4 / H6 test results. G2 failed, so the model-level arm of H5, H11, H12, H13 and H9 were not run and stay theory.
- Claims to soften: anything about hyperbolic distance at model level, memory at equal accuracy, and copy-vs-recompute, for which no run exists.

## Surprises

- H3's R2 leaf cannot be tested at 32K; the S11 vs S8 leg is the only testable part of H3.
- U8's auxiliary SDPA check exceeds its 1e-5 absolute tolerance while the pre-registered identity is met (DEV-U8).

## Limitations observed

- Model scale (4B), context 32K, synthetic data, one funded cell for each of H1 and H5, cost-driven reductions listed in the registration (Section 2b). Dev-pilot numbers are never test results; they appear only in files and figure markers labelled dev.

## Reproduction

- Final bundle: `python -m opa.analysis.final_bundle --runs jobs/interim/series_test.yaml --trackb jobs/interim/trackb_test_runs.yaml --src bundle --out <new folder>` (reads the interim tables, writes claims, tests, deviations, PAPER_INPUT and the manifest, then runs the verifier on a scratch copy); upload with `python -m opa.bundle upload --final`.
- Interim tables behind it: `python -m opa.analysis.interim --split test --runs <series yaml> --out bundle`.
- Re-run a primary result: `python -m opa.launch run jobs/<job>.yaml` with the recorded seed and git SHA.
- Data and run logs: the bucket in config/project.yaml under opa/ (append-only).
