# DEV-TRAIN-PIN — V3 and B1 checkpoints are pinned after the prereg-v2 tag, not before

- **Date:** 2026-11-08.
- **Plan text:** PLAN_V3 Section 6a, "Reuse": "The trained V3 and B1 checkpoints are the P4 V3 and B1 setups. Pin them by
  hash in `prereg/prereg-v2.md` before the tag; they are not trained again."
- **What happened:** the tag (c8a0a5e, 2026-11-07T23:59:43Z) was placed before Step 2 (training to the stopping rule)
  had run: only the 400-step learning-curve check (Step 1) existed. So no trained V3/B1 checkpoint could be pinned in
  the registration.
- **What is fixed instead, and was fixed before the tag:** the training configuration (the learning-curve run's
  config, unchanged), the stopping rule (Section 6a Step 2, verbatim: evaluate every 200 optimizer steps on the
  monitoring set, dev seeds 2000–2063; stop when V3's own-origin mass fraction rises by less than 0.02 between two
  consecutive evaluations 200 steps apart, or at 2,000 steps; never before 400; final checkpoint, never the best),
  B1 at exactly V3's step count, and G2 on fresh dev seeds 3000+. None of these depends on test data.
- **Handling:** V3 continues from the step-400 checkpoint of
  B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261107223205-86f471f-64ea (configuration unchanged, rule 2). The
  SHA-256 of each final checkpoint (V3, B1, and A1/A2/A3 if trained) is recorded in this file and in `bundle/` when it
  is produced, before any test-split evaluation uses it. No checkpoint is selected by test results; the rule decides.
- **Learning curve (Step 1, dev monitoring set) for the record:** own-origin mass fraction 0.137 / 0.183 / 0.254 /
  0.249 / 0.307 at steps 0 / 50 / 100 / 200 / 400 (ratio to uniform 0.98 → 2.20); attribution 0.00 → 0.31 (chance
  0.167); passkey 0.00 → 0.97; loss 3.09 → 2.37; 22.75 s per optimizer step on an L4. Rise 200→400 = +0.058 (> 0.02),
  so the stopping rule continues training.

## Pinned checkpoints (filled in as they are produced)
| Arm | Final step | Run id | SHA-256 of the checkpoint |
|---|---|---|---|
| V3 | 600 (stopping rule: rise 400→600 = 0.0137 < 0.02; own-origin fraction 0.3204 at 600) | B-V3-mtlbmhqa-archive-archive64-T8-dev-s0-20261108031640-6c44911-9560 (resumed from the step-400 checkpoint sha256 e17baa47a2c6b41bd402eecdd69d5d744695c05542c758f544dc3dffa1c05682) | f80aa344cbce0e461e3958f2ed16bb373ffd8900792ca3168c6f87214ddc3814 |
| B1 | 600 (= V3's L; checkpoint saved mid-run by the parallel B1 run, owner instruction A5; checker a113189 verified a mid-run step-L checkpoint equals training to exactly L) | B-B1-mtlbmhqa-archive-archive64-T8-dev-s0-20261108044732-a400cda-cae6 (step600/checkpoint.pt, 141,885,527 bytes) | eeccb50e8f89f95dd82f008a376a6b6b245137c633a3017fd01b10b136ed8bff |
