# Origin-Paged Memory: code, pre-registration and verified results

This repository holds the code, job specifications, pre-registration, frozen verdict rules and the verified final result bundle behind the preprint *Origin-Paged Memory: Treating Language-Model Context as Virtual Memory* (Aaditya Rajgor, 2026).

**Preprint:** link to follow.

## What OPA, OPM and OAA are

**Origin-Paged Attention (OPA)** is the research program as a whole. It treats a language model's context like virtual memory in an operating system: the archive is split into fixed-size chunks called *origins* (512 tokens here), and the last few tokens of each origin, its *red dot*, mark where the next origin begins and carry context across the boundary. The program has two layers, a memory layer (OPM) and an attention layer (OAA). The earlier design and theory draft is in [`docs/paper_v2.md`](docs/paper_v2.md).

**Origin-Paged Memory (OPM)** is the memory layer, and it works with an unchanged model. It reads origins in order with a few working slots (M = 4), copies exact evidence into a small pinned store (2,048 tokens), and keeps a page table that lists the entities each origin mentions (its *skeleton*). When an entity becomes relevant only after its origin has been evicted, OPM looks it up in the page table and pages that origin back in: a *page fault*. This is the "fold-left with skeleton and graph faults" system, labelled `S11` in the code and results (`S8` is the same without links between origins).

**Origin-Anchored Attention (OAA)** is the attention layer, which would move paging inside the model. Each origin restarts its rotary positions at zero (a fresh positional frame), and an address vector for each origin, placed in a hyperbolic (Poincaré-ball) space, enters the attention logit through a content-router term and a structural distance penalty. The paper proves that this attention factors exactly into a choice of origin times ordinary attention within it, so the error caused by paging equals the attention mass that was paged out. The Track B code in [`opa/trackb/`](opa/trackb/) implements it on Qwen3-4B (arm `V3`, against standard attention `B1`).

## Headline results

13 hypotheses (H1-H13) were pre-registered (tag `prereg-v2`) with verdict rules fixed before any test data existed, then tested on `Qwen/Qwen3-4B-Instruct-2507`. An independent verifier recomputes every verdict from the frozen rules.

- **OPM beats one-shot retrieval.** On synthetic late-binding multi-hop questions (LB-MHQA, late h = 2, 32K tokens, n = 607), OPM (`S11`) answered 76.6% exactly, against 62.4% for one-shot retrieval (`S2`): **+14.2 EM points** (95% CI 9.2 to 19.1, Holm-adjusted p < 0.000001). It also beat a gist-memory agent (`S4`, 37.3%) by 39.6 points (95% CI 33.2 to 45.8, n = 391). H1: supported.
- **Pinning prevents thrashing** (H10: supported; 2.54 fewer repeated page-ins per item, 95% CI 2.27 to 2.83).
- **Several distinctive ingredients added nothing measurable:** the graph of links between origins, hyperbolic rather than Euclidean distances, and the red-dot prefix.
- **The attention layer did not learn in this first test.** A short fine-tuning run (600 steps, stopped by the pre-registered stopping rule) did not teach OAA to attribute facts to their origin: source attribution 31.2% for OAA against 99.6% for standard attention, and an own-origin attention mass of about 0.32 against a 0.90 gate threshold. Gate G2 therefore failed, and the tests that depend on a working attention layer (H9, H11, H12, H13) were not run, as the pre-registration requires.

In short: the memory layer is supported; moving it inside attention remains open. Every verdict, with its rule, estimate, CI and deviations, is in [`bundle/PAPER_INPUT.md`](bundle/PAPER_INPUT.md) and [`bundle/tables/final_verdicts.csv`](bundle/tables/final_verdicts.csv).

## How to reproduce

The experiments ran on [Modal](https://modal.com) GPUs (L4 by default, T4 for small jobs), with artifacts written write-once to a Google Cloud Storage bucket.

1. **Install** Python 3 and the dependencies: `pip install -r requirements.txt`.
2. **Configure** [`config/project.yaml`](config/project.yaml): replace `YOUR_GCP_PROJECT` and `gs://YOUR_BUCKET` with your own GCP project and bucket. Everything is written under `opa/` in that bucket.
3. **Set up Modal**: `modal setup`, then create a Modal secret named `opa-gcp` holding a service-account key that can create and read (not delete) objects in your bucket. An `opa-hf` Hugging Face secret is optional; all models used are ungated.
4. **Run the tests** (no GPU, no spend): `python -m pytest -q tests` and `python -m pytest -q tools/tests`.
5. **Re-run a job**: `python -m opa.launch run jobs/<job>.yaml`. The test-split jobs behind the verdicts are listed in [`jobs/interim/`](jobs/interim/); each run's seed and git SHA are recorded in its run id under [`bundle/runs/`](bundle/runs/). Each job spec carries a cost cap (`max_usd`), and `opa.launch` checks spending against a `budget/approvals.yaml` and `budget/ledger.csv` that you create for your own account (see [`opa/budget.py`](opa/budget.py)).
6. **Verify the bundle**: `python tools/verify_bundle.py bundle/ --no-git`. It recomputes every verdict and Holm p-value from [`prereg/rules.yaml`](prereg/rules.yaml), checks the claim, gate and test records against the schemas in [`bundle_spec/`](bundle_spec/), and checks every file hash in `bundle/manifest.json`. `--no-git` skips the checks against the original repository's `prereg-v2` tag, which is not part of this repository's history.

### What differs from the internal bundle

This is a public copy of the result bundle, with internal and account-specific material taken out:

- The budget records (approvals, ledger, prices, provider reconciliations) and the cost and planning tables are not included, so the verifier skips its budget check (D9) when `bundle/budget/` is absent. Rebuilding the bundle with `opa.analysis.final_bundle` needs those records and is not possible from this copy.
- The original bucket name is replaced with `YOUR_BUCKET` in the run records, two tables and two tests; a local file path is removed from the G4 gate record; the cost section is removed from `PAPER_INPUT.md`. The SHA-256 entries in `bundle/manifest.json` were updated for exactly these files.
- The `gs://` URIs in the run records point to run artifacts and datasets that are not public.
- Some records and code comments refer to the internal run plan (`PLAN_V3`) and run log (`state/...`), which are not released. The public design is the pre-registration in [`prereg/`](prereg/); every deviation from it is listed in [`bundle/deviations.md`](bundle/deviations.md).

## Folder map

| Path | What it holds |
|---|---|
| [`opa/`](opa/) | Harness code: data generators (`data/`), Track A agent-level memory systems (`tracks/`), Track B attention-level OAA arms and training (`trackb/`), analysis and bundle building (`analysis/`), statistics, scoring, budget and Modal launcher |
| [`jobs/`](jobs/) | Job specifications (YAML) for every run and the scripts that generate them |
| [`prereg/`](prereg/) | Pre-registration `prereg-v2.md`, frozen verdict rules `rules.yaml`, sample sizes, and registered deviations |
| [`bundle/`](bundle/) | Verified final result bundle: `PAPER_INPUT.md` (results summary), claims H1-H13, gates G0-G4, unit tests U1-U8, tables, figures, run records, deviations and `manifest.json` (file hashes) |
| [`bundle_spec/`](bundle_spec/) | JSON schemas for the claim, gate and unit-test records, used by the verifier |
| [`tools/`](tools/) | `verify_bundle.py`, the machine-checkable definition of done, and its tests |
| [`tests/`](tests/) | Tests for the harness |
| [`config/`](config/) | `project.yaml`: GCP placeholders, Modal settings, models and pinned revisions |
| [`docs/`](docs/) | Design paper draft v2, test protocol v2 and their figures |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata |

## Citation

See [`CITATION.cff`](CITATION.cff). A DOI will be added when the release is archived.

## License

MIT. See [`LICENSE`](LICENSE).
