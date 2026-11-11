# OPA Experiment Protocol v2 — Agentic AI Team

Nov 6, 2026 · @Aaditya

This protocol tests the ten hypotheses of the Origin-Paged Attention draft v2 in three tracks. Track A is agent-level, needs only API access, and decides by week 3 whether the idea deserves GPU time. Track B is model-level on a 1.5B open model and tests the attention geometry. Track C measures systems costs. Every experiment says **why** it exists (which claim it can kill), **what** to build, **where** it runs, and **how** to read the result.

**What changed from protocol v1.**

| Area | v1 | v2 |
| --- | --- | --- |
| Hypotheses | H1–H7 | H1–H10; new H5 (hyperbolic vs Euclidean vs exact tree distance), H7 (precision), H8 (descent routing) |
| Agent harness | directory + skeleton + page faults | adds an address book, a shell index with descent, and a graph page table with five edge types |
| Benchmarks | LB-MHQA, natural multi-hop sets, BABILong | adds MT-LB-MHQA: several threads, cross-thread late binding and near-duplicate confusers |
| Model-level variants | position maps M1–M4 | fresh frames (M5) plus origin router, hyperbolic penalty, value term and learned red-dot chain; baselines APE-style, landmark tokens, Euclidean and exact-tree penalties, Graph-KV-style masks |
| Contextual-dependence remedies | R0–R4 | R0–R7, adding red-dot prefix and co-encoding with graph parents |
| Systems | copy vs prefill | adds routing cost, hyperbolic overhead in the kernel, per-tier latency and graph-prefetch hit rate |
| Timeline | 8 weeks | 10 weeks, with a dedicated gate for the decisive ablation |

**How to read it.** Section 1 is the contract: gates and kill criteria, fixed before any data. Sections 3–6 are the experiments, each with the hypothesis it serves. Section 7 says how many items are needed and how results are tested. Section 9 points to tested code: protocol\_snippets.py (v1) and protocol\_snippets\_v2.py (v2), both in the opa\_checks folder, both with self-tests that pass.

## 1 Goal, decision gates and kill criteria

The goal is to find out, as cheaply as possible, which OPA claims survive contact with real models. Run Track A first: it costs only API calls and decides whether Tracks B and C are worth their GPU time. The decisive ablation (H5) gets its own gate because it decides whether the paper may call itself hyperbolic.

| Gate | When | Pass condition | If pass | If fail |
| --- | --- | --- | --- | --- |
| G0 Harness valid | end of week 1 | oracle-skeleton, oracle-graph OPA (γ\_eff = 1) reaches ≥ 90% EM on the LB-MHQA pilot; three repeats of the full-context baseline agree within 2 EM points; unit tests U1–U8 pass | start Track A full runs | fix the harness; no science yet |
| G1 Agent-level signal | end of week 3 | H1 and H3 pass on the pilot (50 items per cell) | full Track A; start Track B | write up a negative result; stop B and C |
| G2 Geometry signal | end of week 6 | H4: fresh frames alias and the origin vector repairs it (own-origin mass ≥ 90% on positional probes, provenance gain over M4); H6: some remedy recovers ≥ 90% of full-recompute accuracy | H5 at model level; Track C | publish the agent-level paper only |
| G3 Decisive ablation | end of week 8 | H5 at both levels with confidence intervals | choose the paper's framing by the H5 outcome (Section 5.5) | (no fail: every outcome decides the framing) |
| G4 Paper-ready | end of week 10 | all primary results on frozen test splits with CIs; reviewer audit done | submit | extend or narrow claims |

### Hypotheses and kill criteria (pre-registered)

| ID | Hypothesis | Track | Kill criterion |
| --- | --- | --- | --- |
| H1 | Fold-left with skeleton and graph faults beats one-shot retrieval on late-binding multi-hop questions | A | gain < 5 EM points over top-k RAG, or < 3 over the ReadAgent-style baseline, at equal token budget |
| H2 | Extractive session state keeps turn-1 constraints through 50 turns | A2 | extractive retention not ≥ 10 points above the rolling summary |
| H3 | Measured coverage predicts success as about γ\_eff^h\_late, and graph edges add recall beyond skeletons | A | R² < 0.5, or adding the graph gains < 2 EM points on late-bound chains |
| H4 | Fresh frames alone alias origins; the origin vector repairs it and adds provenance | B | M5 alone shows no aliasing, or M5 + origin vector shows no provenance gain over M4 |
| H5 | Hyperbolic distance beats Euclidean distance on the same tree | A and B | arm A2 not better than A1 beyond the 95% interval at both levels |
| H6 | Red-dot prefix and graph-parent co-encoding close most of the contextual-dependence gap | B | no remedy recovers ≥ 90% of full-recompute accuracy, or the red-dot prefix is no better than none |
| H7 | Global coordinates fail at the predicted depths; re-anchored offsets do not | B (kernel) | re-anchored distance error grows with depth |
| H8 | Descent through shells matches flat routing with far fewer scores | A, B and C | descent loses > 3 points of recall at equal M |
| H9 | Page-in by copy beats re-encoding above a measurable chunk size, at each tier | C | copy never beats prefill on the test hardware |
| H10 | Pinning prevents thrashing when a step needs more origins than slots | A | no drop in repeated page-ins |

A hypothesis that hits its kill criterion is reported as refuted, in the paper, with the data. Negative results are part of the deliverable.

## 2 Team, environment and logging

| Role | Owns | Key output |
| --- | --- | --- |
| Lead (author) | hypotheses, gates, paper | pre-registration file, final claims, the H5 framing decision |
| Agent engineer | Track A harness, controllers, shell index and graph page table at the agent level | the OPA agent, all baselines S1–S18 |
| Data/eval engineer | LB-MHQA and MT-LB-MHQA generators, dataset adapters, scoring | frozen dev/test splits, scorer, gold annotations (fact origins, h\_late, threads) |
| ML engineer | Track B: fresh frames, origin-anchored attention, LoRA runs, KV-reuse remedies | patched attention, measurement scripts |
| Geometry/kernel engineer (can be the ML engineer) | Poincaré functions in torch, red-dot frames, precision tests, the per-block bias in attention | U5–U8 unit tests, H7 tables |
| Systems engineer (can be the ML engineer) | Track C timing, tiered storage | latency, routing and memory tables |
| Independent reviewer | blind grading audit, code review of the scorer and of skeleton/graph leakage | audit log |

**Environment.** Python 3.11; PyTorch and Hugging Face transformers for Track B; any API client for Track A. One 80 GB GPU (A100 or H100) is enough for Track B at 1.5B parameters. Pin every library version in a lock file and record the GPU, driver and CUDA version with each run. No hyperbolic-geometry library is required: port protocol\_snippets\_v2.py to torch (about 60 lines) and test it against the numpy reference.

**Where the bias goes.** In Track B, add the origin bias b\_u(c) as a float attention mask to PyTorch's scaled\_dot\_product\_attention (the mask is added to the scores before the softmax). This disables the FlashAttention backend for those calls; at 1.5B parameters and a working window of a few thousand tokens the memory-efficient or math backend is fast enough. A fused per-tile bias kernel is a Track C stretch goal.

**Models.** Track A: one frontier API model (record the exact model string and date) plus [Qwen/Qwen2.5-7B-Instruct](https://hf.co/Qwen/Qwen2.5-7B-Instruct) to show the controller is model-agnostic. Track B: [Qwen/Qwen2.5-1.5B-Instruct](https://hf.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0) as the main model, [meta-llama/Llama-3.2-1B-Instruct](https://hf.co/meta-llama/Llama-3.2-1B-Instruct) (gated; request access early) as replication, and 7B only for a final scale check. Temperature 0 everywhere.

**Pre-registration.** Before any test-split run, commit a file listing H1–H10, the kill criteria, the primary metric and primary contrast of each hypothesis, sample sizes, the H5 arms and the equivalence margin, and the analysis plan; tag the commit prereg-v2. Changes after the tag are allowed but must be logged as deviations with a reason.

**Splits.** Every dataset gets a dev split (for prompt and hyperparameter work, including λ) and a test split that nobody looks at until the gate review. Each system gets the same tuning budget: at most 10 dev iterations. For H5, every arm gets the same λ grid.

**Logging.** Every action of every controller is one JSON line (schema v2 in Section 9). Store raw model inputs and outputs, router masses for resident origins, and the address-book state at query time, so any number in the paper can be traced to a run. Name runs {track}-{system}-{dataset}-{length}-{seed}-{git\_sha}.

## 3 Track A — agent-level OPA with shells and graph (H1, H3, H5, H8, H10)

**Why.** Track A tests the memory-management claims without touching model weights: does fold-left with skeleton and graph faults recover late-bound facts, does the graph add recall, does the shell geometry help rank origins, and does descent route as well as a flat scan. It is the fastest way to kill or keep the idea.

**Where.** Any API model plus Qwen2.5-7B-Instruct, a SQLite store, and a small Python controller.

### 3.1 Build the harness (week 1)

1. **Chunker.** Split each document stream into origins of B tokens with the model's own tokenizer, B ∈ {512, 1024, 2048}. Store origin\_id, text, source title, token span and arrival order.
2. **Origin store.** SQLite tables: origins(origin\_id, doc\_title, start\_tok, end\_tok, text, thread\_id, parent\_id), addresses(node\_id, parent\_id, offset\_vector, depth), edges(src, dst, type, weight, created\_at), skeleton(entity\_key, origin\_id, surface\_form).
3. **Skeleton builder,** run at ingest and never shown the question:
   - S-lex: regular expressions for identifiers, part numbers, numbers with units, dates, file paths and capitalized multi-word names.
   - S-llm: the model lists the entities in each origin with a fixed prompt.
   - S-hyb: the union of both. Store entity → origin ids with normalized keys (lower case, punctuation stripped) and an alias table.
4. **Graph builder,** also at ingest (paper Section 9): seq edges to the parent origin; tree edges from the shell index; ent edges between origins sharing a skeleton key; ref edges from explicit structure (reply-to ids, file paths, citations, tool-call ids); x edges between origins in different threads whose dense-embedding cosine exceeds a threshold tuned on dev. Cap each node at k\_max = 32 edges per type.
5. **Address book and shell index** (paper Sections 7.5 and 8). Thread assignment: a new origin joins the thread whose head is most similar (dense embedding cosine) if above τ\_thread, else it starts a new thread under the closest inner node. Layout: the training-free one (one step δ = 1 outward per origin along the thread's direction, directions well spread), stored as offsets in the parent's frame, exactly as AddressBook in protocol\_snippets\_v2.py. Inner nodes: one per b = 8 children, holding a routing key (mean child embedding plus a BM25 index over the merged skeleton) and child pointers; no abstractive summary is used as evidence.
6. **Tools exposed to the model:** list\_directory() (top shell only); descend(query, beam) returns candidate origin ids with scores; page\_in(origin\_id) returns text; evict(slot); pin(origin\_id, quote, note); lookup(entity) returns origin ids; neighbors(origin\_id, kinds) returns linked origin ids with edge types; answer(text, cited\_pin\_ids).
7. **Ranking with the structural penalty.** For arms that use geometry, candidates are re-ranked by relevance(c) − λ·d(q, c), where q is the origin the question is attached to (the current thread head) and d is the arm's distance (Section 3.7).
8. **Exact-pin rule.** The harness rejects any pin whose quote is not an exact substring of the named origin: the agent-level guarantee of copy by reference.
9. **Budgets.** M working slots (a page\_in beyond M evicts the least recently used origin and logs it) and a pin budget of P\_pin tokens.
10. **Session registers** (used in Track A2): JSON with goal, constraints \[{key, value, unit, pin\_id, origin\_id, superseded\_by}\] and decisions.

### 3.2 Primary benchmark 1: LB-MHQA (single thread, late binding)

Synthetic data is primary because the entities are invented, so pretraining cannot leak answers.

- **Filler:** passages from [emozilla/pg19](https://hf.co/datasets/emozilla/pg19), split into origins.
- **Facts:** invented entities in h-hop relation chains, for example "Varrel Pumps sources seals from Ostrand Ltd." → "Ostrand Ltd. makes its seals from Kelvane." → "Kelvane seals degrade above 80 °C in ester coolant." The question asks for the final attribute.
- **Placement:** forward, late (reversed, so every hop after the first is late-bound) and mixed.
- **Distractors:** same-template facts about other entities, plus alias variants ("Ostrand Ltd." / "OSTRAND") at rate α ∈ {0, 0.3}.
- **Grid:** h ∈ {2, 3, 4} × length ∈ {32K, 128K, 512K} × placement × α. Pilot: h = 3, 128K, all placements, α = 0, 50 items per cell.
- **Gold:** the origin of every fact and h\_late.

### 3.3 Primary benchmark 2: MT-LB-MHQA (multi-thread, cross-thread late binding) — new

**Why.** The v2 claims are about threads and structure. LB-MHQA has one thread, so geometry cannot help or hurt there. MT-LB-MHQA builds sessions that look like real agent work: several interleaved topics, facts that cross topics, and near-duplicate facts that punish attending to the wrong thread.

- **Threads:** T ∈ {4, 8, 16} topics, interleaved in arrival order the way real sessions switch (for example A A B A C B C C A …), each with its own filler source.
- **Chain:** the question belongs to the most recent thread. Its chain's facts are placed in reverse order inside that thread (late-bound). With probability 0.5 per item, the last hop's fact is stated earlier in a different thread (a cross-thread hop).
- **Confusers:** every other thread contains a near-duplicate of the final fact, about an entity whose name differs by one character and with a different value. A system that attends to the wrong thread answers with the confuser's value.
- **Gold:** thread labels, fact origins, h\_late, which hop is cross-thread, and the confuser values.
- **Grid:** T × h ∈ {2, 3} × length ∈ {32K, 128K} × cross-thread ∈ {no, yes}. Pilot: T = 8, h = 3, 128K, both cross-thread settings, 50 items per cell.
- **Generator:** make\_mt\_item in protocol\_snippets\_v2.py builds contiguous threads; extend it to interleaved arrival before the pilot.

### 3.4 Secondary natural benchmarks

- [dgslibisey/MuSiQue](https://hf.co/datasets/dgslibisey/MuSiQue) (answerable, 3–4 hop), [xanhho/2WikiMultihopQA](https://hf.co/datasets/xanhho/2WikiMultihopQA) and [hotpotqa/hotpot\_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) (distractor setting). Pad each question's paragraphs with other questions' paragraphs to the target length, shuffle, chunk, and record where the supporting paragraphs land to compute h\_late. Treat each source question's paragraphs as one thread to get a natural multi-thread variant.
- [RMT-team/babilong](https://hf.co/datasets/RMT-team/babilong) tasks qa2 and qa3 (two and three supporting facts) at 32K, 64K and 128K.

These are public and may be contaminated, so they support but do not carry the claim.

### 3.5 Systems compared

All systems use the same model, the same prompt-tuning budget and the same total token budget per question.

| ID | System | Purpose |
| --- | --- | --- |
| S1 | Full context (where it fits) | ceiling for short lengths |
| S2 | Top-k RAG, BM25 + dense hybrid, k tuned on dev | main baseline for H1 |
| S3 | Iterative retrieval in the style of IRCoT | retrieval during reasoning |
| S4 | ReadAgent-style gist memory with page lookup | closest prior agent design |
| S5 | Chain-of-Agents-style sequential workers + manager | closest fold-left design |
| S6 | OPA fold-left without skeleton | is the skeleton needed? |
| S7 | OPA + lexical skeleton (S-lex) | cheap skeleton |
| S8 | OPA + hybrid skeleton (S-hyb) | v1 proposed system |
| S9 | OPA + oracle skeleton | upper bound for the skeleton, labelled oracle |
| S10 | OPA with abstractive notes instead of exact pins | is extractiveness needed? |
| S11 | S8 + graph faults (all five edge types) | does the graph add recall? (H3) |
| S12 | S11 + shell descent + hyperbolic penalty (full v2 agent; arm A2) | proposed system |
| S13 | RAPTOR-style tree of abstractive summaries with Euclidean embeddings | closest prior tree retrieval |
| S14 | S12 with Euclidean distances between flat embeddings of the same tree (arm A1) | geometry vs structure (H5) |
| S15 | S12 with exact tree distance (arm A3) | is the geometry more than a tree? (H5) |
| S16 | S12 with λ = 0 (arm A0) | does any structural penalty help? |
| S17 | HNSW-style flat graph index over origins, no shells | descent vs a strong flat index (H8) |
| S18 | S12 with oracle skeleton and oracle graph | upper bound, labelled oracle |

Edge ablations of S11: drop seq, ent, ref or x edges one at a time.

### 3.6 Metrics and what to observe

**Primary:** exact match after standard answer normalization. **Secondary:** F1; evidence precision and recall (pins vs gold facts); total tokens; page-ins; necessary vs spurious faults; routing recall (was every gold-fact origin among the candidates descend returned?) and scores per query; thread-confusion rate (answer equals a confuser's value); wall time; cost.

Produce these plots and read them in this order:

1. EM vs context length, one line per system. Expect S2 to fall on late placement while S8–S12 stay flatter.
2. EM vs h\_late. Expect S6 to collapse as h\_late grows and S8/S11 to decay roughly as γ\_eff^h\_late.
3. Observed vs predicted success. Measure per item γ\_s (was each late-bound entity in its fact origin's skeleton?) and γ\_g (was the fact origin one edge from a resident origin when the entity became relevant?); plot binned success against γ\_eff^h\_late and report R². Report the correlation (phi coefficient) between skeleton misses and graph misses: Corollary 14.1 assumes zero.
4. MT-LB-MHQA: EM and thread-confusion rate vs number of threads T, for S11, S12, S14, S15, S16. This is where geometry should show.
5. Routing recall vs scores per query for descent (beams 1, 2, 4), S17 and a flat scan.
6. Fault histogram per item and spurious-fault rate per skeleton type and per edge type.
7. Cost vs EM frontier: which systems are Pareto-optimal.

Tag every wrong answer with one failure code: F1 fact never paged in; F2 paged but not pinned; F3 pinned but misread; F4 skeleton miss; F5 alias mismatch; F6 evicted by budget; F7 reasoning error with all facts pinned; F8 wrong thread (confuser value); F9 descent took a wrong turn (needed origin never reached); F10 needed edge missing. The distribution says what to fix next.

### 3.7 The decisive ablation at agent level (H5)

| Arm | System | Distance used for re-ranking |
| --- | --- | --- |
| A0 | S16 | none |
| A1 | S14 | Euclidean distance between flat embeddings of the same tree (same layout rule, Euclidean steps) |
| A2 | S12 | hyperbolic distance in the lowest-common-ancestor frame |
| A3 | S15 | exact tree path length through the lowest common ancestor |
| A4 | S12 with fitted addresses | hyperbolic distance after fitting Poincaré embeddings to the session's tree plus graph edges, the agent-level analogue of learned addresses |

Same tree, same candidates, same token budget; λ ∈ {0.5, 1, 2, 3, 5} tuned per arm on dev. **Primary contrast:** A2 vs A1 on MT-LB-MHQA, T = 8, h = 3, 128K, cross-thread items. **Secondary:** A2 vs A3 as an equivalence test with a ±3 EM margin (Section 7). Read the outcome with the table in Section 5.5.

### 3.8 Thrashing (H10)

Build tasks that need joint reasoning over r ∈ {2, 3, 4, 6} origins at once (for example, "which of these r suppliers quotes the lowest price for part X?"), with M ∈ {2, 3} slots, with and without pinning. Count repeated page-ins of the same origin per item and EM. Expect repeated page-ins without pinning to rise sharply once r > M, and to stay near zero with pinning.

## 4 Track A2 — session drift (H2)

**Why.** The paper claims that extractive state does not drift and that abstractive state loses facts at about (1 − ε)^T. v2 adds a second place where drift could creep in: inner-shell digests. This track measures both.

### 4.1 Synthetic 50-turn specification sessions

- Turn 1 states 10 constraints with exact values: temperatures, tolerances such as ±0.05 mm, part numbers, supplier names, dates.
- Turns 2–50 mix distractor discussion, explicit updates to 3 of the 10 constraints ("change the max coolant temperature to 75 °C"), and near-miss values that must not overwrite anything ("the old spec said 70 °C").
- Spread the session over 3 threads (design, supplier, testing), so updates arrive in a different thread from the original statement.
- Probes at turns 10, 25 and 50 ask for the current value of every constraint.
- Generate 200 sessions from templates with invented part numbers and suppliers.

### 4.2 Conditions

| ID | Session state | Notes |
| --- | --- | --- |
| D1 | Full history in context | only while it fits |
| D2 | Rolling abstractive summary, rewritten every turn, 1K-token budget | the drift baseline |
| D3 | Abstractive memory notes, rewritten when the model chooses (MemGPT-style) | common agent practice |
| D4 | OPA extractive registers: exact pins with origin pointers; an update pins the new span and marks the old one superseded | proposed |
| D5 | D4 without supersession links | ablation |
| D6 | D4, but inner-shell nodes carry abstractive digests that the model may cite as evidence | tests the "inner shells stay extractive" rule |

### 4.3 What to measure

- Current-value accuracy at each probe (primary at turn 50).
- Stale-value rate: the answer gives a superseded value.
- Fabricated-value rate: the answer gives a value that never appeared.
- Per-rewrite loss ε for D2: after every rewrite, check which of the 10 constraints are still present and exact; ε is the fraction lost or mutated per rewrite. Plot observed survival against (1 − ε)^T. A curve below the prediction means correlated losses, which the paper must report.
- For D6, the share of wrong answers that cite a digest rather than an origin.
- Natural check: run D2 and D4 on [xiaowu0162/longmemeval-cleaned](https://hf.co/datasets/xiaowu0162/longmemeval-cleaned) (the original LongMemEval release is marked deprecated in favour of this one).

Expect D4 ≈ D1 while D1 fits, D4 well above D2 at turn 50, D5 to show more stale values than D4, and D6 to fall between D4 and D2.

## 5 Track B — model level: fresh frames, origin-anchored attention and the hyperbolic penalty (H4, H5, H6, H7)

**Why.** Track B tests the claims that only a model can test: that fresh frames alias origins (Proposition 5), that the origin vector repairs it (Theorem 6, Corollary 6.1), that the value term carries provenance (Proposition 7), that re-anchoring is needed for precision (Proposition 10), and whether the red-dot prefix closes the contextual-dependence gap (Proposition 2).

**Where.** Qwen2.5-1.5B-Instruct with a patched attention and a custom KV cache, one 80 GB GPU; Llama-3.2-1B-Instruct for replication.

**Implementation notes.**

- In the Hugging Face Llama and Qwen2 code, RoPE pairs dimension j with j + d/2 (the rotate\_half layout). "Plane i" is dimensions (i, i + d/2) with frequency inv\_freq\[i\]; i = 0 is the highest frequency. Check num\_key\_value\_heads: these models use grouped KV heads, which changes cache sizes.
- **Fresh frame (M5):** give every origin position ids 0 … B − 1. With a red-dot prefix of r tokens, use 0 … r − 1 for the red dot and r … r + B − 1 for the origin, which keeps relative positions identical to offsets −r … −1 and stays within non-negative position ids.
- **Origin bias:** compute b\_u(c) = r\_uᵀA·o(c) − λ·d\_H(o(c\_u), o(c)) per (query, origin) and expand it to a float attention mask over that origin's keys; pass it to scaled\_dot\_product\_attention. Distances in float32 in the lowest-common-ancestor frame (AddressBook.pair\_distance in protocol\_snippets\_v2.py).
- **Value term:** add g·U·o(c) to every value vector of origin c after the value projection; g is a scalar per layer initialized at 0.

### 5.1 Unit tests before any experiment

| ID | Test | Pass condition |
| --- | --- | --- |
| U1 | Re-indexing: keys cached at m and rotated by Δ match keys computed at m + Δ | fp32 max error ≤ 10⁻⁵ |
| U2 | M2 with origins in archival order in contiguous slots reproduces the unpatched model | logits identical |
| U3 | M5 with a single origin reproduces the unpatched model | logits identical |
| U4 | OAA with A = 0, λ = 0, g = 0 reproduces M5 | logits identical |
| U5 | Torch port of the Poincaré functions and LCA-frame distances matches protocol\_snippets\_v2.py | ≤ 10⁻⁶ in fp32 |
| U6 | Precision table: global coordinates fail at depths 4 / 7 / 16 (bf16 / fp16 / fp32), re-anchored error flat to depth 40 (this is H7) | depths match within one step; re-anchored error does not grow |
| U7 | Factorization on real logits: per-origin router masses π computed from the model's own logits satisfy a = π·α | ≤ 10⁻⁵ in fp32 |
| U8 | Paging identity: total-variation distance between full and paged attention equals the evicted router mass | ≤ 10⁻⁵ in fp32 |

### 5.2 Contextual dependence and remedies (H6)

Data: 200 multi-document questions from [zai-org/LongBench](https://hf.co/datasets/zai-org/LongBench) (its multi-document QA subsets) or padded HotpotQA, plus 200 MT-LB-MHQA items at 16K tokens.

| ID | How the documents' KV is built | Role |
| --- | --- | --- |
| R0 | All documents encoded together in order | reference |
| R1 | Each document encoded alone in a fresh frame, then co-loaded | naive reuse (what M5 does by default) |
| R2 | R1 + recompute a fraction r ∈ {5, 15, 30}% of tokens chosen by KV deviation, following CacheBlend | selective recompute |
| R3 | Each document encoded after the same short shared prefix | APE-style |
| R4 | R1 with each document's first-token attention sink neutralized | EPIC-style |
| R5 | Each origin encoded with its parent's red-dot KV as prefix, r ∈ {4, 8, 16} | red-dot prefix (new) |
| R6 | Each origin encoded attending to its top-k graph parents' KV, k ∈ {1, 2} | co-encoding, Graph-KV mechanism (new) |
| R7 | R5 (r = 8) + R2 (5%) | combined |

Measure per layer the cosine of K and V against R0 (the toy predicted 1.00 at layer 0, falling with depth), the KL divergence of next-token distributions over answer tokens against R0, task EM/F1, and time to first token. Read: if R5 closes most of R1's gap at r = 8, the red dot is doing its job; if only R2 or R6 does, the red dot is decoration and the paper says so.

### 5.3 Aliasing and its repair (H4)

**Archive:** 64 origins of 512 tokens in 8 threads (PG-19 filler, one source book per thread, with inserted needles). **Working window:** 4 or 8 co-loaded origins, loaded in random order.

| ID | Variant | Training |
| --- | --- | --- |
| V0 | M5: fresh frames only | none |
| V1 | M5 + fixed-λ hyperbolic penalty, training-free layout (paper Section 7.5) | none |
| V2 | M5 + router term in keys | LoRA |
| V3 | V2 + value term (provenance) | LoRA |
| V4 | V3 + hyperbolic penalty with learned λ | LoRA |
| V5 | V4 + learned red-dot chain (addresses from content) | LoRA |
| B1 | M4 from v1 (slot positions + origin planes) | LoRA |
| B2 | APE-style: M5 + shared prefix, no origin signal | none |
| B3 | Landmark-style: one learned landmark token per origin | LoRA |
| B4 | Graph-KV-style structured mask: each origin attends only to its graph parents | none |
| B5 | HoPE positional encoding, only if its code is public by week 4 (its abstract says code will be released) | per its paper |

**Tasks.**

1. *Non-adjacent passkey.* A needle sits in one origin co-loaded with distant ones; ask for it. Content-addressed, so V0 may pass.
2. *Archival order.* Two co-loaded origins state conflicting values for one key; ask which came first. Needs order: V0 should be at chance.
3. *Source attribution.* "Which origin (from this list of ids) said X?" Tests provenance (G6); V3 should beat V2.
4. *Thread confusion.* Same-offset facts with near-duplicate names in different threads; ask about one thread's version. Fresh frames make the twins positionally identical; V1 and V4 should separate them.
5. *Perplexity* of a PG-19 continuation conditioned on paged-in context, and held-out perplexity at standard positions to catch damage.

**Measure per layer and head:** own-origin mass and own-thread mass on positional probes; ghost ratio (attention on same-offset tokens in other origins divided by what uniform attention would give); router entropy H(π); and a linear probe that predicts the source origin id from head outputs. Predicted: V0 near 1/C own-origin mass and chance on tasks 2–4; V1 better on task 4 without training; V2–V5 at or above B1; V3 above V2 on task 3.

### 5.4 Training recipe

- **Trainable:** LoRA rank 16 on q, k, v, o projections, plus the new parameters W\_r, A, U, W\_Δ, λ and g (about 3 million parameters at m = 32 across the 28 layers of the 1.5B model, comparable to the LoRA adapters themselves).
- **Data:** 50% PG-19 with random paging and random thread assignment, 50% synthetic MT sessions. Each training sequence holds 4–8 co-loaded origins plus a few pinned spans.
- **Settings:** about 2,000 steps per configuration to start; learning rate 10⁻⁴ for LoRA and 10⁻³ for new parameters; g = 0 and A small at initialization; λ initialized at 2; chain step clipped at δ\_max = 1; distances in float32.
- **Monitors, logged every 50 steps:** router entropy (collapse to 0 or to ln C is a failure), own-origin mass on a fixed positional probe, ghost ratio, held-out perplexity at standard positions, NaN count, and the largest origin radius (should stay bounded because of re-anchoring and clipping).

### 5.5 The decisive ablation at model level (H5)

Same LoRA budget and data for every arm; only the distance in the bias changes.

| Arm | Bias | Built from |
| --- | --- | --- |
| A0 | router only, λ = 0 | V3 |
| A1 | −λ·Euclidean distance between flat embeddings of the same tree | V3 + Euclidean penalty |
| A2 | −λ·hyperbolic distance | V4 |
| A3 | −λ·exact tree path length | V3 + tree penalty |
| A4 | −λ·hyperbolic distance with learned addresses | V5 |

**Primary contrast:** A2 vs A1 on task 4 and on MT-LB-MHQA at 16K. **Reading the outcome (decided at gate G3):**

| Outcome | Conclusion for the paper |
| --- | --- |
| A2 > A1 and A2 ≈ A3 (within the equivalence margin) | hyperbolic geometry is a faithful, differentiable way to compute tree distance inside attention; frame the claim that way |
| A2 > A1 and A2 > A3 | the continuous geometry adds something beyond the exact tree; strongest framing |
| A2 ≤ A1 | drop the hyperbolic claim; keep fresh frames, origin vector, graph and memory manager |
| A4 > A2 | learned red-dot addresses earn their training cost; report as a separate claim |

**Compute estimate** (confirm in the pilot): unit tests, 5.2 and the training-free parts of 5.3 take a few GPU-hours on one 80 GB GPU. Each LoRA configuration takes roughly a GPU-day at 1.5B parameters; V2–V5, B1, B3 and A1, A3 are about 8 configurations, so about 8 GPU-days, plus replication of the best two on Llama-3.2-1B.

## 6 Track C — systems: latency, routing and tiers (H8, H9)

**Why.** OPA's efficiency claims are arithmetic until measured: that copying an origin's KV beats re-encoding it, that descent through shells is cheaper than a flat scan, that the hyperbolic bias is a rounding error next to attention, and that the graph makes prefetching worth doing.

**Where.** The Track B machine, plus pinned host memory and a local NVMe drive for the outer shell.

1. **Copy vs prefill (H9).** Keep origin KV in pinned host memory and on NVMe. Time the copy (async, on its own CUDA stream) from each tier, plus any recompute from Track B's best remedy. Under fresh frames no re-index is needed; record it as zero, and separately time Lemma 1's re-index for the M4 baseline. Prefill path: time a prefill of the same B tokens with L\_ctx tokens already in context. Sweep B ∈ {128, 256, 512, 1024, 2048, 4096}; L\_ctx ∈ {0, 8K, 32K}; batch ∈ {1, 8}; 16-bit KV.
2. **Routing cost (H8).** For archives of C ∈ {2K, 16K, 128K} origins, time a flat scan of routing keys against descent with beams 1, 2 and 4 (b = 8), on GPU and on CPU. Report routing recall from Track A beside the time.
3. **Hyperbolic overhead.** Time the bias computation (router product, LCA-frame distance, mask expansion) against the attention call it feeds, for M ∈ {4, 8, 16} co-loaded origins. The paper predicts about 0.2% of the attention work; the mask expansion may cost more than the arithmetic, and that is worth knowing.
4. **Per-tier latency.** GPU-resident, CPU pinned, NVMe: median and p95 time to make one origin resident.
5. **Graph prefetch.** Replay Track A traces: when an origin is paged in, prefetch its top-k neighbours one tier closer. Report the hit rate (a later page-in found its origin already promoted) and the bytes wasted on unused prefetches, for k ∈ {1, 2, 4}.

Measure with CUDA events, 20 warm-up runs and 100 timed runs per cell; report median and p95. Record peak GPU memory with the framework's peak-memory counter.

**Check the arithmetic.** Bytes moved per origin must equal 2 × layers × KV heads × head dim × B × 2 bytes; a mismatch means the cache layout is not what you think. For Qwen2.5-1.5B read the layer and KV-head counts from its config rather than assuming the 8B shape used in the paper.

**Outputs:** a latency-vs-B plot per tier with the crossover B\* marked; a routing time-vs-recall table; the overhead table; prefetch hit rates; and a table of peak GPU memory for full context vs OPA at 32K, 128K and 512K effective context. Overlapping copy with compute is a stretch goal; LMCache already implements pipelined transfers and is the reference to compare against.

## 7 Statistics, power and threats to validity

Power only the primary contrast of each hypothesis and treat the rest of the grid as exploratory. The primary contrasts are: H1, S8/S11 vs S2 on LB-MHQA (late, h = 3, 128K); H5, A2 vs A1 on MT-LB-MHQA (T = 8, h = 3, 128K, cross-thread); H3, the R² of predicted vs observed success. Measure the discordant-pair rate in the pilot, then read the items needed per cell from these tables.

**Superiority (paired McNemar test, two-sided α = 0.05, power 0.8).**

| Discordant pairs | EM gain 3 points | EM gain 5 points | EM gain 10 points |
| --- | --- | --- | --- |
| 10% | 870 | 312 | 77 |
| 15% | 1,306 | 469 | 116 |
| 20% | 1,742 | 626 | 155 |
| 30% | 2,614 | 940 | 234 |

**Equivalence for H5's A2 vs A3 (two one-sided tests, α = 0.05 each, power 0.8, true difference 0).**

| Discordant pairs | margin ±2 EM | margin ±3 EM | margin ±5 EM |
| --- | --- | --- | --- |
| 10% | 2,141 | 952 | 343 |
| 15% | 3,212 | 1,428 | 514 |
| 20% | 4,282 | 1,904 | 686 |
| 30% | 6,423 | 2,855 | 1,028 |

The pre-registered margin is ±3 EM. Synthetic items are cheap to generate, so the binding cost is API calls: budget them in the pilot. (Formulas: Connor's approximation for McNemar; n = (z₀.₉₅ + z₀.₉)²·p\_d / margin² for the equivalence test. The superiority table matches the v1 protocol up to rounding up.)

### Analysis plan

- **Differences in EM:** paired bootstrap with 10,000 resamples for 95% confidence intervals, plus an exact McNemar test.
- **Multiple hypotheses:** Holm–Bonferroni across the ten primary tests.
- **H3:** bin items by predicted γ\_eff^h\_late, compare binned observed success with prediction, report R²; also fit a logistic regression of success on h\_late, measured γ\_s and measured γ\_g, and report the skeleton–graph miss correlation.
- **H5:** superiority test A2 vs A1; equivalence test A2 vs A3; both at agent and model level. Report all five arms in one table regardless of outcome.
- **H7 and H8:** these are measurements, not significance tests: report error vs depth and recall vs scores with bootstrap bands.
- **Run-to-run variance:** API models are not fully deterministic at temperature 0. Repeat the pilot 3 times and report the spread.
- **Grading:** exact match is automatic. Where an LLM judge is needed (free-form answers), the judge is blind to the system, and a human audits a stratified 10% sample.

### Threats and controls

| Threat | Control |
| --- | --- |
| Pretraining contamination | synthetic LB-MHQA and MT-LB-MHQA with invented entities are primary; public sets are secondary |
| Unequal effort across systems | same token budget, same 10-iteration tuning budget, same model, same λ grid for every H5 arm |
| API model drift | pin the model string; run all systems in the same week, interleaved |
| Skeleton or graph leakage | skeletons and edges built at ingest, never shown the question; audited in code review |
| Geometry credited for structure | H5 holds the tree fixed and changes only the distance (A1, A3) |
| Oracle results read as real | oracle rows (S9, S18) labelled in every table and plot |
| Synthetic-to-real gap | report natural benchmarks beside synthetic ones; MT variant of the natural sets |
| Confusers too easy or too hard | calibrate the one-character-twin rate on dev so S2's confusion rate is between 20% and 60% |
| Cherry-picking | every pre-registered cell is reported, including failures |

## 8 Ten-week plan and deliverables

| Week | Track A (agent) | Track B (model) | Track C (systems) | Gate |
| --- | --- | --- | --- | --- |
| 1 | harness, skeleton and graph builders, address book, LB-MHQA and MT-LB-MHQA generators, scorer; prereg-v2 tag | unit tests U1–U5 | — | G0 |
| 2–3 | pilot (50 items per cell, 3 repeats) on both benchmarks; failure codes; power from discordant rates; H5 agent arms on the pilot | U6–U8; 5.2 remedies R0–R5 | — | G1 |
| 4–5 | full runs on frozen test splits (H1, H3, H10); Track A2 drift (H2) | 5.2 R6–R7; 5.3 training-free variants V0, V1, B2, B4 | copy vs prefill (H9) | — |
| 6 | H5 agent arms at full size | 5.3 LoRA variants V2–V5, B1, B3 | routing cost, hyperbolic overhead (H8) | G2 |
| 7–8 | H8 routing recall at full size | 5.5 decisive ablation A0–A4 (LoRA) | per-tier latency, graph prefetch | G3 |
| 9 | natural benchmarks; robustness (B ∈ {512, 1024, 2048}) | replication on Llama-3.2-1B for the two best configurations | peak-memory tables | — |
| 10 | independent reviewer audit; all plots regenerated from logs | 7B scale check of the best configuration | — | G4 |

### Deliverables

- [ ] prereg-v2 tag with hypotheses, kill criteria, primary contrasts, H5 arms, equivalence margin and analysis plan
- [ ] LB-MHQA and MT-LB-MHQA generators with frozen dev/test splits, scorer and README
- [ ] OPA agent harness with skeleton, graph page table, address book and shell descent; baselines S1–S18
- [ ] Session-drift generator and conditions D1–D6
- [ ] Patched attention: maps M1–M5, origin router, value term, hyperbolic penalty with LCA-frame distances; unit tests U1–U8
- [ ] KV-reuse measurements R0–R7
- [ ] Aliasing and provenance results V0–V5, B1–B5
- [ ] Decisive ablation tables at agent and model level (A0–A4)
- [ ] Latency, routing, overhead, per-tier and prefetch tables (Track C)
- [ ] One results table per hypothesis (template below), every number traceable to a run id
- [ ] Reviewer audit log

### Results template (one per hypothesis)

| System / arm | Dataset / cell | n | EM (95% CI) | Δ vs baseline (95% CI) | p (Holm) | Tokens per item | Run ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S12 / A2 full v2 agent | MT-LB-MHQA, T = 8, h = 3, 128K, cross-thread |  |  |  |  |  |  |
| S14 / A1 Euclidean on the same tree | same cell |  |  |  |  |  |  |
| S15 / A3 exact tree distance | same cell |  |  |  |  |  |  |
| S2 top-k RAG | same cell |  |  |  |  |  |  |

## 9 Code starters

Two tested numpy files sit in the opa\_checks folder. Port them to torch by swapping np for torch; Track B's unit tests compare against them.

- **protocol\_snippets.py (v1):** position maps M1–M4, rotate\_half RoPE, re-indexing, and the LB-MHQA generator make\_item. Five self-tests pass (M2 = M1 when contiguous, re-indexing error 1.3×10⁻¹², M4 deviation capped at 0.511 rad, M3 ghost alignment, generator h\_late).
- **protocol\_snippets\_v2.py (v2):** Poincaré functions, red-dot frames with lowest-common-ancestor distances, map M5, origin-anchored attention for one query, beam descent, and the MT-LB-MHQA generator make\_mt\_item. Seven self-tests pass: Möbius isometry (7×10⁻¹⁵), LCA-frame distance equal to the root-frame distance (8×10⁻¹⁵), neighbours at depth 40 still exactly 1.0 apart in float32 while float32 global coordinates have collapsed, M5 positions, the router × local factorization, total-variation paging error equal to evicted router mass, descent finding its target with b + (L − 1)·w·b scores, and a cross-thread late-bound generator item.

### Poincaré ball and red-dot frames

```python
import numpy as np

def mobius_add(x, y):
    xy, x2, y2 = np.dot(x, y), np.dot(x, x), np.dot(y, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / (1 + 2 * xy + x2 * y2)

def dist(x, y):
    diff = np.dot(x - y, x - y)
    den = (1 - np.dot(x, x)) * (1 - np.dot(y, y))
    return float(np.arccosh(1 + 2 * diff / den))

def expmap0(v):
    n = np.linalg.norm(v)
    return np.tanh(n) * v / n if n > 0 else v

class AddressBook:
    """Origin 0 is the root. Origin c stores only offset[c] = exp0(delta_c),
    its position in the frame of its parent's red dot."""
    def __init__(self, m, dtype=np.float64):
        self.m, self.dtype = m, dtype
        self.parent, self.offset = {0: None}, {0: np.zeros(m, dtype=dtype)}

    def add(self, c, par, delta, d_max=None):
        delta = np.asarray(delta, dtype=np.float64)
        if d_max is not None and np.linalg.norm(delta) > d_max:
            delta = delta * (d_max / np.linalg.norm(delta))     # clip the chain step
        self.parent[c], self.offset[c] = par, expmap0(delta).astype(self.dtype)

    def path_to_root(self, c):
        out = []
        while c is not None:
            out.append(c); c = self.parent[c]
        return out

    def lca(self, a, b):
        anc = set(self.path_to_root(a))
        return next(n for n in self.path_to_root(b) if n in anc)

    def coords_in_frame(self, node, ancestor, work_dtype=np.float32):
        """Right-nested u1 (+) (u2 (+) (... (+) u_node)) over offsets below `ancestor`."""
        path, n = [], node
        while n != ancestor:
            path.append(n); n = self.parent[n]
        y = np.zeros(self.m, dtype=work_dtype)
        for n in path:                                           # deepest first
            y = mobius_add(self.offset[n].astype(work_dtype), y).astype(work_dtype)
        return y

    def pair_distance(self, q, c, work_dtype=np.float32, d_clamp=None):
        a = self.lca(q, c)
        yq, yc = self.coords_in_frame(q, a, work_dtype), self.coords_in_frame(c, a, work_dtype)
        d = dist(yq.astype(np.float64), yc.astype(np.float64)) if q != c else 0.0
        return min(d, d_clamp) if d_clamp is not None else d
```

### Origin-anchored attention for one query

```python
def oaa_attention(s_loc, router_bias, d_struct, lam):
    """s_loc [C, B]: local RoPE logits of one query against C co-loaded origins (fresh frames).
    router_bias [C]: r_u^T A o(c).  d_struct [C]: hyperbolic distances from the query's origin."""
    b = router_bias - lam * d_struct                               # per-origin bias b_u(c)
    full = s_loc + b[:, None]
    a = np.exp(full - full.max()); a /= a.sum()                     # ordinary softmax
    lse = np.log(np.exp(s_loc - s_loc.max(1, keepdims=True)).sum(1)) + s_loc.max(1)
    z = b + lse
    pi = np.exp(z - z.max()); pi /= pi.sum()                        # router masses (Theorem 6)
    alpha = np.exp(s_loc - lse[:, None])                            # local attention within each origin
    return a, pi, alpha                                             # a == pi[:, None] * alpha

def paging_tv(a, pi, resident):
    mask = np.zeros(a.shape[0], bool); mask[list(resident)] = True
    a_s = np.where(mask[:, None], a, 0.0); a_s /= a_s.sum()
    return 0.5 * np.abs(a - a_s).sum(), 1.0 - pi[mask].sum()      # equal (Corollary 6.2)
```

### Beam descent through shells

```python
def descend(children, score, root=0, beam=2):
    """children: node -> list of child ids; score: node -> float. Returns (leaves, n_scores)."""
    frontier, n_scores, leaves = [root], 0, []
    while frontier:
        cands = [ch for n in frontier for ch in children.get(n, [])]
        if not cands:
            leaves = frontier; break
        s = np.array([score(ch) for ch in cands]); n_scores += len(cands)
        frontier = [cands[i] for i in np.argsort(-s)[:beam]]
        if all(not children.get(n) for n in frontier):
            leaves = frontier; break
    return leaves, n_scores
```

The fresh-frame map is one line in plane\_positions: for M5 every plane of token (c, ℓ) gets position ℓ. In Hugging Face models this means position\_ids restart at 0 for every origin (or at r after an r-token red-dot prefix).

### Event log v2: one JSON line per controller action

```json
{"run_id": "A-S12-mtlbmhqa-128k-T8-s0-3f2a9c1", "item_id": "mt-000417", "step": 7, "t_ms": 1834,
 "action": "fault", "origin_id": 41, "thread_id": 3, "slot": 1, "reason": "late_bound:EntQWRTZ",
 "via": "edge:ent", "fault_kind": "necessary", "edge_types_followed": ["ent"],
 "descend": {"beam": 2, "scores": 56, "candidates": [41, 77, 102]},
 "router_mass": {"41": 0.62, "77": 0.21, "102": 0.09},
 "distance": {"arm": "A2", "lambda": 2.0, "d": {"41": 1.84, "77": 3.10, "102": 4.02}},
 "tokens_in": 1210, "tokens_out": 96,
 "pins": [{"pin_id": "p12", "origin_id": 41, "char_span": [120, 188]}],
 "relevant_entities": ["EntKLMNP", "EntQWRTZ"], "model": "<exact model string>", "git_sha": "3f2a9c1"}
```

action is one of page\_in, evict, pin, lookup, neighbors, descend, prefetch, fault, answer. via says how a fault found its origin (skeleton, edge:ent, edge:ref, edge:x, edge:seq). fault\_kind is filled by the scorer after the run (necessary if the origin holds a gold fact, else spurious). router\_mass is logged only at the model level; at the agent level it holds the re-ranking scores.

### MT-LB-MHQA item generator

make\_mt\_item(filler\_chunks, n\_threads, origins\_per\_thread, hops, rng, cross\_thread=True, confusers=True) builds one item: an invented entity chain placed late-bound in the most recent thread, an optional cross-thread last hop stated earlier in another thread, one-character-twin confusers in every other thread, and gold fact origins, thread labels and h\_late. Before the pilot, extend it to interleaved thread arrival and to the alias variants of Section 3.2.
