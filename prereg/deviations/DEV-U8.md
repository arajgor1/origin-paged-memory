# DEV-U8 — U8's auxiliary kernel-consistency check exceeds 1e-5 on the L4 (reported, criterion unchanged)

- **Date:** 2026-11-08 (after the prereg-v2 tag; dev-split unit test, no hypothesis verdict depends on it).
- **Run:** B-units-none-u6-u8-h7-dev-s0-20261108002243-31bbd86-36da (Qwen3-4B-Instruct-2507, fp32, NVIDIA L4,
  torch 2.15.0+cu132, tf32 matmul off), `tests/U8.json` in its GCS run folder.
- **Pre-registered U8 (protocol 5.1 table, as adopted by the pre-registration Section 9):** "Paging identity:
  total-variation distance between full and paged attention equals the evicted router mass", tolerance ≤ 1e-5 in fp32.
  **Measured:** max |TV(a_full, a_paged) − evicted router mass| = 3.33e-16 over 81 cases (9 real paged kernel passes,
  layers 0/18/35, 25/50/75% of the window origins evicted, evicted masses 5e-16 to 1.0); q, k, v identical between the
  passes (difference 0); weight on evicted keys exactly 0. **The pre-registered U8 criterion is met.**
- **Auxiliary check (added by the agent before the tag to tie U8 to the real kernel; not in the protocol or the
  pre-registration):** the paged pass's fused SDPA output against the explicit product a_paged @ v. Measured max
  absolute difference 2.08e-5, above the 1e-5 absolute tolerance the code applied to it, so the harness printed
  `U8_passed: false`. This compares two fp32 implementations of the same attention (a fused GPU kernel vs an explicit
  matmul), not the paging identity; it is the same kind of absolute-tolerance-below-rounding issue as DEV-U1 (fp32
  ulp at real activation magnitudes). Its relative error was not recorded by this run.
- **Handling:** the bundle reports U8 by its pre-registered criterion (met, 3.33e-16) and reports the auxiliary
  check at its literal value (2.08e-5 absolute, above the code's 1e-5) next to it; the auxiliary tolerance is not
  changed after the fact and the U8 record is not re-run to obtain a pass. G0 was decided earlier on U1–U5 and is not
  revisited. No verdict rule is affected.
