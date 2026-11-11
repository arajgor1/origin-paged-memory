### DEV-U1
- Date: 2026-11-07
- Change: U1 is judged by a two-part definition: float64 error at most 1e-10 and fp32 error at most 4 ulps of the largest key.
- Reason: The protocol's absolute 1e-5 tolerance is below one fp32 ulp at real key magnitudes (the largest key is about 298).
- Impact: U1 passes; the literal criterion is reported and not met. Anticipated and recorded before the tag.

### DEV-G0-noise
- Date: 2026-11-07
- Change: G0's 'three S1 repeats agree within 2 EM points' counts as passed if every difference is a near-tie flip.
- Reason: Owner ruling recorded before the attempt-2 results: attempt 2 had ten near-tie flips and no harness error (EM spread 6 points at n = 50).
- Impact: G0 passed; the per-item flip rate is reported next to every primary contrast. Pre-tag.

### DEV-G1-OVERRIDE
- Date: 2026-11-07
- Change: G1 was first decided under the protocol's definition at 18:30 UTC and failed on H3's R2 criterion alone (H1 passed); the owner then overrode the gate in PLAN_V3 revision 3 and G1 was re-decided under the new definition and passed. The original decision is kept in gates/history/. The pilot estimates for H1, H2, H3, the H5 agent arm and H10 had been seen when the override was made; no result yet existed for trained Track B models, Track C, H7, H8 or H9.
- Reason: At 32K the H3 coverage predictor was 1.0 for 149 of 150 pilot items (verified), so its R2 test could not succeed for any theory.
- Impact: Without the override G1 would have failed and H4 to H9 and H11 to H13 would not have run. The new G1 needs a working S4 and H1 on the pilot.

### DEV-PLAN-REV2
- Date: 2026-11-07
- Change: G2 is judged after V3 and B1 reach a pre-set stopping rule on a fresh dev set (not on the 30-minute pilot); a 400-step learning-curve check is added; the H5 trend key is renamed A2_vs_A1_slope_per_thread_depth.
- Reason: Corrects a plan error (the pilot allowed 39 optimizer steps); the pilot results had been seen; thresholds unchanged. Before the tag.
- Impact: G2 judged on fresh dev seeds after the stopping rule fired at step 600.

### DEV-PLAN-REV3
- Date: 2026-11-07
- Change: G1 redefined (working S4 via a baseline audit, H1 on the pilot's point estimates; H3 removed from G1); H3 is tested on cells chosen by predictor variance alone.
- Reason: At 32K the H3 predictor is constant, so the original G1 could not pass for any theory.
- Impact: See DEV-G1-OVERRIDE. H3 keeps its frozen rule.

### DEV-S4-V2
- Date: 2026-11-07
- Change: S4 v2 replaced the first S4 (gist cap 300 with salvage, one-origin pages, parallel lookup of up to 5 pages, harness-failure flags).
- Reason: The S4 baseline audit failed: a 160-token gist cap emptied 23.5% of the gists, invisible to the failure codes.
- Impact: Dev tuning iteration 1 inside the protocol's tuning budget; the audit then passed; S4 v2 is the baseline on the test split.

### DEV-PLAN-REV4
- Date: 2026-11-07
- Change: H11's non-inferiority margin widened from 2 to 3 EM points in prereg/rules.yaml.
- Reason: A 2-point margin needs about 5,885 questions, which the budget cannot fund. Set before the tag and before any H11 accuracy was measured.
- Impact: H11 was not run (G2 failed); H13 keeps its 2-point margin.

### DEV-PLAN-REV5
- Date: 2026-11-07
- Change: The owner approved the 'fits $126' package: total cap $148, base envelopes above the plan's $100 limit by owner exception, BOOST $30 for more H1 test items (use b).
- Reason: Budget decision before the tag.
- Impact: H2 (deferred) and H13's accuracy test were not funded and are inconclusive; the robustness extras and the 64K check were not run.

### DEV-H3-R2
- Date: 2026-11-07
- Change: H3's R2 leaf is reported as not testable at 32K: no result is entered for it.
- Reason: Both candidate cells have a constant predicted coverage of 1.0, so the identity-line R2 cannot be estimated (registration Section 2b, closure item 9).
- Impact: H3 cannot be supported; by the frozen all_of rule it is refuted if the S11 vs S8 leg is refuted, otherwise inconclusive.

### DEV-TRAIN-PIN
- Date: 2026-11-08
- Change: The V3 and B1 checkpoints are pinned after the prereg-v2 tag, not before.
- Reason: The tag was placed before the training to the stopping rule had run. The configuration, the stopping rule and the G2 design were fixed before the tag.
- Impact: V3 and B1 are pinned by SHA-256 in prereg/deviations/DEV-TRAIN-PIN.md (step 600); no checkpoint was chosen by test results.

### DEV-U8
- Date: 2026-11-08
- Change: U8 is reported by its pre-registered criterion (TV distance equals the evicted router mass, tolerance 1e-5): met. The harness's own flag is false because an auxiliary SDPA-vs-(a_paged @ v) check exceeds 1e-5 absolute.
- Reason: The auxiliary check was added by the agent before the tag to tie U8 to the real kernel; it compares two fp32 implementations.
- Impact: tests/U8.json records passed: true by the pre-registered criterion with the harness flag and the auxiliary value stated literally; no re-run for a pass.

### DEV-OWNER-INTERIM
- Date: 2026-11-08
- Change: Owner instruction (dated 2026-11-07, received 2026-11-08): a preprint on Track A, an interim Track A package, Track A first on contested GPU capacity, vLLM max_containers 8, reservations at 1.5 times measured cost for measured job types, B1 started in parallel with V3, H9 not run if G2 fails (an exploratory request instead).
- Reason: Time to the preprint; execution stays as pre-registered.
- Impact: No rule, gate, threshold or approval changed; no money outside the approved envelopes.

### DEV-FAILED-RUNS
- Date: 2026-11-08
- Change: Failed and excluded runs: A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108043839-48e13a3-87a9 (p2-h1-lb-c07): replaced by retry 2 of 3 allowed (run A-S11+S2+S8-lbmhqa-late-h2-32k-a0-test-s100206-20261108055403-10a14e2-6c9a); the retry's records are used; B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108022824-e36fdc0-201d (p4-v3-c01): failed (pre-fix code: the resume path was unresolved); excluded; V3 chunk 1 is the later run on the fixed deploy.
- Reason: A storage precondition failure (HTTP 412) ended one H1 container's first attempt; a V3 chunk was launched on code from before a fix.
- Impact: The retry is used in place of the first H1 attempt (sensitivity in tables/interim_rerun_sensitivity.csv); the pre-fix V3 chunk is excluded; nothing was deleted; both stay in runs/index.csv history and the ledger.
