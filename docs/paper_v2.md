> Exported from the paper draft on 6 Nov 2026. Figures 1–6 are in docs/figures/ (the export replaces images with placeholders).

# Origin-Paged Attention: Hyperbolic Shell Memory for Bounded, Addressable Transformer Context

Nov 6, 2026 · @Aaditya

**Status.** Working draft v2, 6 November 2026. This revision follows the revised design: every origin restarts all 64 rotary planes (a fresh positional frame) and carries an index; consecutive origins are chained at a "red dot"; origins live as points in nested Poincaré balls; and a graph at the KV level links new context to old. No language-model experiments have been run yet. Every number in this draft comes from two seeded scripts, checks.py (v1) and checks\_v2.py (v2), described in Appendix B. Each theorem states what it assumes.

**What changed since v1.**

| Topic | v1 | v2 |
| --- | --- | --- |
| Position inside an origin | hybrid map M4: slot position on local planes, origin tag on a few planes | fresh frame M5: all P planes restart at 0 in every origin |
| How attention tells origins apart | origin planes inside RoPE | an origin vector o(c) outside RoPE: a content router term plus a structural distance penalty |
| Geometry of origins | a line (origin index) | points in a Poincaré ball; radius = level of detail and memory tier, direction = thread |
| Links between origins | skeleton (entity → origin) table | typed graph page table: sequence, tree, entity, reference and cross-thread edges |
| Routing | flat scan of descriptors | coarse-to-fine descent through shells, plus graph prefetch |
| Contextual dependence remedy | pick one of four from the literature | red-dot prefix plus co-encoding with graph parents, measured against the four |
| Kept unchanged | Lemma 1, Propositions 2–3, the three-zone MMU, fold-left with page faults, extractive state, thrashing | same, generalized where the graph adds a recovery path |

## Abstract

Long-running agents and long documents outgrow any fixed context window, and the usual remedies either forget (eviction, summaries) or re-read from scratch (retrieval with re-encoding). We propose Origin-Paged Attention (OPA), which treats context as a paged, addressable memory. The archive is split into origins of B tokens. Each origin is encoded in a fresh positional frame, with all P rotary planes restarted at zero, and is given an origin vector o(c), a point in a Poincaré ball.

Attention uses a three-part logit: the usual within-frame RoPE score, a content-to-origin router term, and a structural penalty −λ·d\_H between the query's origin and the key's origin. We prove that this logit factors exactly into a soft chunk router times ordinary local attention (checked to 6×10⁻¹⁷), so the same quantity decides both what attention reads and what the memory system should page in. We show that fresh frames alone alias completely: tokens at the same offset in different origins have positional kernel exactly 1.0, so a positional head spreads its mass evenly over the C co-loaded chunks (1/C = 6.25% at C = 16). An origin margin β = ln(ρ(C−1)/(1−ρ)) restores a share ρ to the own chunk; 90% at C = 16 needs 4.9 logits, and the requirement grows only as ln C.

Hyperbolic geometry supplies the structure. Nested shells index level of detail and memory tier; directions index threads. The circumference at hyperbolic radius R is 2π·sinh R (1,267 at R = 6, against 37.7 in the plane), so exponentially many chunks fit at each level. Two points on the same shell are close only through their common ancestor: the distance behaves like 2(R − R\_a) with R\_a = ln(1/sin(θ/2)), so a hyperbolic distance acts as a tree distance whose branching depth is set by angle. Möbius translation makes every red dot the new coordinate origin exactly (checked to 4×10⁻¹⁶). This matters for precision: in global coordinates a unit step has more than 10% error after 4 steps in bfloat16, 7 in float16 and 16 in float32, while offsets stored relative to each red dot keep a constant 0.2% error in bfloat16 at every depth.

A graph page table adds explicit cross-links for prefetch, page faults and co-encoding, because geometry alone cannot make two distant branches close. On top sits the v1 three-zone memory manager: fold-left with skeleton page faults, whose success is at least γ\_eff^h\_late, where h\_late is the number of late-bound hops and γ\_eff = 1 − (1 − γ\_s)(1 − γ\_g) combines skeleton and graph coverage; and an extractive session state that never rewrites what it keeps.

In synthetic checks, a structural penalty of λ = 5 puts 91% of a positional head's mass on its own chunk and under 0.01% on same-offset tokens in other threads. Coarse-to-fine descent through shells finds the right chunk among 4,096 with 104 scores instead of 4,096, and stays more robust to noisy summaries (95% against 38% at noise σ = 0.3). The decisive open question is whether hyperbolic distances beat Euclidean distances and exact tree distance on the same tree. Section 13 makes that the first experiment.

## 1 Introduction

### 1.1 Why: a flat tape is the wrong shape for long-lived context

A transformer's context is a flat tape: token positions 0, 1, 2, … and a key–value (KV) cache that grows by one entry per token. Three things go wrong when an agent works for hours or reads a document far longer than its window.

- **Cost and capacity.** Attention per new token is linear in context length and the KV cache is linear in memory. A 1-million-token session for a Llama-3-8B-shaped model holds about 122 GiB of KV state (128 KiB per token, Section 11).
- **Forgetting.** Eviction policies (StreamingLLM \[G. Xiao et al., 2024\], H2O \[Zhang et al., 2023\], SnapKV \[Li et al., 2024\]) drop tokens for good, and rolling summaries rewrite what they keep, so facts degrade with every rewrite (Proposition 16).
- **Lost structure.** A session is not a line. It has threads (the bug being fixed, the design being discussed, the user's constraints), levels of detail (a turn, a topic, the whole session), and cross-references (a turn that answers an earlier one). The tape flattens all of this into one distance, |i − j|, and models use the middle of a long tape poorly \[N. F. Liu et al., 2024\].

Operating systems solved an analogous problem decades ago with virtual memory: an address space far larger than physical memory, pages moved in and out on demand, and a page table that says where everything is \[Denning, 1970\]. The question this paper asks is what the attention-native version of that looks like.

### 1.2 What: origins, red dots, shells and a graph

OPA divides the archive into **origins**: chunks of B tokens (B = 512 in our examples). The design has four parts, each answering one weakness of the tape.

1. **Fresh frames.** Every origin restarts all P = d/2 rotary planes at position 0. No relative distance on the rotary planes ever exceeds B, so positions are always in the training distribution, however long the archive grows.
2. **Origin vectors and red dots.** Each origin c carries an index and an origin vector o(c). Origins are chained: the last few tokens of origin c (the "red dot") condition the encoding of origin c + 1 and define the step from o(c) to o(c + 1). Attention sees the origin vector directly through a router term and a distance penalty.
3. **Hyperbolic shells.** Origin vectors live in a Poincaré ball. The radius of a point is its level of detail, from the session root at the centre to full-resolution chunks on the outer shell. Its direction is its thread. Shells also map to memory tiers: the small inner shells stay on the GPU, the huge outer shell lives in CPU memory or on disk.
4. **A graph page table.** Typed edges (sequence, parent/child, shared entity, explicit reference, cross-thread) link new origins to old ones. The graph drives prefetching, page faults and co-encoding, and it covers the relations the geometry cannot (Section 8.2).

Underneath sits the v1 memory manager: a pointer zone of descriptors and skeletons, a working zone of M paged-in origins, and a session zone of pinned, exact evidence.

### 1.3 Where: two levels of implementation

| Level | What changes | What it needs | Tested first? |
| --- | --- | --- | --- |
| Agent level | Origins are text chunks; shells and graph are an index the agent queries with tool calls; pinned evidence is quoted text | any API model, no weight access | yes: cheapest test of the memory-management claims |
| KV level | Origins are cached KV blocks; the origin vector enters the attention logit; page-in is a memory copy | open weights, a custom attention kernel, LoRA fine-tuning for the learned variants | second: tests the attention-geometry claims |

### 1.4 How: the attention logit in one line

For a query token u in origin c\_u and a key token v in origin c\_v at offsets ℓ\_u and ℓ\_v:

```latex
s(u,v) \;=\; \underbrace{\tfrac{1}{\sqrt d}\,\big(R(\ell_u\omega)\,q_u\big)^{\!\top} R(\ell_v\omega)\,k_v}_{\text{RoPE inside a fresh frame}} \;+\; \underbrace{r_u^{\top} A\, o(c_v)}_{\text{content router}} \;-\; \underbrace{\lambda\, d_{\mathbb H}\big(o(c_u),\, o(c_v)\big)}_{\text{structural penalty}}
```

The last two terms are constant within a chunk, so the softmax splits exactly into "which chunk" times "which token in that chunk" (Theorem 6). That one fact connects the geometry to paging: the router mass of a chunk is both its share of attention and its claim on GPU memory.

### 1.5 Contributions

- **C1. Two negative results** that rule out the simple versions: split-plane origin coordinates alias same-offset tokens (kernel ≥ (k − p)/(k + p)), and fresh frames alias them completely (kernel = 1). Both say an origin signal must come from outside the shared rotary planes (Section 5).
- **C2. Origin-Anchored Attention (OAA)**, a KV equation with an origin vector in keys and values, an exact router × local factorization, a closed-form margin requirement, and a provenance term that tells the output where it read from (Section 6).
- **C3. A hyperbolic address space** for origins: a structural penalty that acts like ALiBi over a tree, exact Möbius re-anchoring at each red dot, and a precision analysis showing why re-anchoring is required and where precision is actually needed (Section 7).
- **C4. Nested shells** that tie level of detail, memory tier and routing together, with the arc-versus-radius distance law and a coarse-to-fine descent that cuts routing from O(C) to O(w·b·L) scores (Section 8).
- **C5. A graph page table** with prefetch, page faults and co-encoding with graph parents, and a recall bound that combines skeleton and graph coverage (Section 9).
- **C6. The complete algorithms** for ingest and query, carrying over fold-left with page faults, extractive state and the thrashing condition from v1 (Section 10).
- **C7. A falsifiable plan**: ten hypotheses with kill criteria, led by the decisive ablation of hyperbolic versus Euclidean versus exact tree distance on the same tree (Section 13).

### 1.6 What is not claimed

- No claim that OPA improves any benchmark. Nothing here has been run on a language model.
- No claim that each component is new. Fresh frames appear in Parallel Context Windows \[Ratner et al., 2023\] and APE \[X. Yang et al., 2025\]; block routing in Landmark Attention \[Mohtashami & Jaggi, 2023\] and MoBA \[Lu et al., 2025\]; recurrent block state in Block-Recurrent Transformers \[Hutchins et al., 2022\] and RMT \[Bulatov et al., 2022\]; structured KV composition in Graph-KV \[Wang et al., 2025\]; hyperbolic language models in HELM \[He et al., 2025\]; coarse-to-fine search in HNSW \[Malkov & Yashunin, 2020\] and RAPTOR \[Sarthi et al., 2024\]. The claim is the combination: a hyperbolic address space used as the page table of a KV memory, and the analysis that makes the combination work.
- No claim that hyperbolic geometry is necessary. It may lose to an exact tree distance; Section 13 is designed to find out.

## 2 Background and notation

### 2.1 Attention and the KV cache

In each layer and head, token t has query, key and value vectors q\_t, k\_t, v\_t in R^d (typically d = 128), computed from the layer input h\_t \[Vaswani et al., 2017\]. Causal attention scores a query against all earlier keys and mixes their values. At inference the keys and values of past tokens are cached, so each new token costs O(N·d) per head instead of recomputing the prefix. Most engines cache keys after rotation. PagedAttention stores this cache in fixed-size physical blocks so it can be allocated like operating-system pages \[Kwon et al., 2023\]; OPA adds the missing layer above it, a logical address space with a page table.

### 2.2 RoPE is 64 independent planes

Rotary Position Embedding \[Su et al., 2024\] splits a d-dimensional vector into P = d/2 two-dimensional planes and rotates plane i by an angle proportional to the position m:

```latex
R_{\Theta,m}\,x \;=\; \bigoplus_{i=1}^{P} \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}\begin{pmatrix}x_{2i-1}\\ x_{2i}\end{pmatrix},\qquad \theta_i = b^{-2(i-1)/d},\; b = 10000
```

Because rotations compose, the score depends only on relative position:

```latex
(R_{\Theta,m}\,q)^{\top}(R_{\Theta,n}\,k) \;=\; q^{\top} R_{\Theta,\,n-m}\, k \;=\; \sum_{i=1}^{P} \big\langle q_{[i]},\, R\big((n-m)\theta_i\big)\,k_{[i]} \big\rangle
```

A 128-dimensional head is therefore 64 orthogonal planes, each with its own frequency. High-frequency planes rotate about 1 radian per token; the lowest rotate about 10⁻⁴ radians per token. Trained models use the highest frequencies for positional heads and prefer the lowest for semantic content \[Barbero et al., 2024\]. Implementations differ in which coordinates form a plane: the original pairs (2i−1, 2i), while the Hugging Face rotate\_half layout pairs dimension j with j + d/2. Every result here holds for either pairing.

"Restarting all 64 planes" (the fresh frame of Section 5) means that inside each origin the position m runs from 0 to B − 1 again. Nothing about the planes themselves changes; what changes is which tokens share a coordinate system.

### 2.3 A primer on the Poincaré ball

**Why hyperbolic space at all.** In a tree with branching factor b, the number of nodes at depth k grows as b^k. In flat space the room available at distance r grows only polynomially (as r^(n−1) in n dimensions), so a deep tree must be squeezed and distances distorted. In hyperbolic space the room grows exponentially with r, like the tree. That is why any finite tree embeds in the hyperbolic plane with distortion arbitrarily close to 1 \[Sarkar, 2012\], and why hyperbolic embeddings represent hierarchies in very few dimensions \[Nickel & Kiela, 2017; Sala et al., 2018\].

**The model.** We use the Poincaré ball of curvature −1: the open unit ball B^m = {x ∈ R^m : ‖x‖ < 1}. The distance between two points is

```latex
d_{\mathbb H}(x,y) \;=\; \operatorname{arcosh}\!\left(1 + \frac{2\,\lVert x-y\rVert^2}{\big(1-\lVert x\rVert^2\big)\big(1-\lVert y\rVert^2\big)}\right), \qquad d_{\mathbb H}(0,x) = 2\,\operatorname{artanh}\lVert x\rVert
```

A point at Euclidean radius r = tanh(R/2) is at hyperbolic distance R from the centre. As r approaches 1 the hyperbolic distance goes to infinity: the boundary is infinitely far away, and almost all of the space's volume sits close to it.

**Möbius addition and translation.** The ball's analogue of vector addition is

```latex
x \oplus y \;=\; \frac{\big(1 + 2\langle x,y\rangle + \lVert y\rVert^2\big)\,x \;+\; \big(1-\lVert x\rVert^2\big)\,y}{1 + 2\langle x,y\rangle + \lVert x\rVert^2\lVert y\rVert^2}
```

It is not commutative or associative, but the map x ↦ (−a) ⊕ x is an isometry of the ball that sends a to the centre \[Ganea et al., 2018\]. So any point can be made the new origin without changing a single distance. Section 7 uses this to make each red dot the origin of its own coordinate system.

**Exponential map at the centre.** A tangent vector v at 0 is mapped into the ball by

```latex
\exp_0(v) \;=\; \tanh\!\big(\lVert v\rVert\big)\,\frac{v}{\lVert v\rVert}, \qquad d_{\mathbb H}\big(0,\exp_0(v)\big) = 2\lVert v\rVert
```

This is how an ordinary network output (a vector in R^m) becomes a point, or a step, in the ball.

**Circles grow exponentially.** In the hyperbolic plane a circle of radius R has circumference 2π·sinh R and the disk has area 2π(cosh R − 1). In B^m the sphere of radius R has area proportional to sinh^(m−1) R. Section 8 turns this into a capacity statement for shells.

### 2.4 Notation

| Symbol | Meaning |
| --- | --- |
| N | archive length in tokens |
| B | origin (chunk) size in tokens, e.g. 512 |
| C = N/B | number of origins in the archive |
| c, ℓ | origin index and offset of a token inside its origin, 0 ≤ ℓ < B |
| M | number of working slots (origins resident on the GPU) |
| P = d/2 | rotary planes per head (64 for d = 128) |
| ω (or θ\_i) | rotary frequencies |
| L\_train | longest context seen in training |
| o(c) | origin vector of origin c, a point in the Poincaré ball B^m |
| m | dimension of the ball (we suggest 16 to 32) |
| d\_H | hyperbolic distance |
| ⊕, exp\_0 | Möbius addition, exponential map at the centre |
| λ | slope of the structural penalty, in logits per unit of hyperbolic distance |
| r\_u, A | query-side router vector and the origin projection in the router term |
| π\_u(c) | router mass of origin c for query u (Theorem 6) |
| β | logit margin of the own origin over the others |
| R\_k, θ | hyperbolic radius of shell k; angle between two directions |
| b, L, w | branching factor, number of shell levels, beam width |
| Σ\_c | skeleton of origin c: the entities and identifiers it mentions, without values |
| W\_t | pinned session accumulator after t steps |
| γ\_s, γ\_g, γ\_eff | skeleton coverage, graph coverage, combined coverage (Corollary 14.1) |
| h\_late | number of late-bound hops in a query (Definition 6) |
| ε | per-rewrite fact-loss probability (Proposition 16) |

## 3 Problem formulation: context as an addressable memory

### 3.1 Archive, origins, threads and red dots

**Definition 1 (Archive).** A session is an archive A of N tokens split into C = N/B origins. Token (c, ℓ) is offset ℓ of origin c. Origins are immutable once written; new conversation turns, tool outputs and documents append new origins. Each origin has:

- an **index** c (its archival order), which never changes;
- a **red dot** ρ(c): the last r tokens of the origin (we suggest r = 4 to 16), whose hidden states summarise what the origin ended on and condition the encoding of its successor;
- a **parent** par(c): the origin it continues (usually c − 1 on the same thread, but a topic switch can attach it elsewhere);
- a **thread** τ(c) and a **level** k(c), defined through its origin vector o(c) in Section 7.

The parent relation makes the archive a forest. Cross-references (a turn that answers a much earlier one, a file included twice) make it a directed acyclic graph rather than a tree; Section 9 handles those with explicit edges.

This is the formal version of the staircase picture: each red dot is where a new origin begins, with its own full set of 64 planes and its own coordinate system, and the chain of red dots is what ties the steps together.

### 3.2 Three zones

**Definition 2 (Three-zone context state).** At step t the model's active context is the union of three zones, which in v2 are also three regions of the shell geometry:

| Zone | Contents | Shells (Section 8) | Typical tier |
| --- | --- | --- | --- |
| Session zone Z\_sess | the pinned accumulator W\_t of exact evidence (text spans or exact KV slices, each tagged with its origin), plus a register file of goals, constraints and decisions, each with a back-pointer to its supporting origin | the centre (shell 0) | GPU memory, always resident |
| Pointer zone Z\_ptr | per origin: index, origin vector, routing key (a landmark token \[Mohtashami & Jaggi, 2023\], min/max key bounds \[Tang et al., 2024\] or representative tokens \[C. Xiao et al., 2024\]), red-dot KV, skeleton Σ\_c, and the origin's graph edges; per inner node: an extractive digest of its subtree | inner shells 1 … L − 1 | GPU for the top levels, CPU memory below |
| Working zone Z\_work | at most M origins whose full KV is paged in | outer shell L | GPU slots 0 … M − 1; the rest in CPU memory or on disk |

### 3.3 Operations

| Operation | What it does | Agent level | KV level |
| --- | --- | --- | --- |
| page\_in(c) | make origin c resident in a working slot | tool call returning the origin's text | copy the cached KV block into GPU memory |
| evict(s) | free working slot s | drop the text from the prompt | free the KV block (it stays in CPU memory or on disk) |
| pin(c, span) | copy exact evidence into W | quote the span verbatim with its origin id | copy the KV slice with its origin tag |
| fault(e) | find the origin(s) that mention entity e and page one in | index lookup | index lookup, then page\_in |
| prefetch(c) | start copying origin c towards the GPU before it is needed | warm a cache of texts | asynchronous copy from disk to CPU or CPU to GPU |
| link(c, c′, type) | add a typed edge to the graph page table | index write | index write |
| descend(q) | route a query from the root down the shells to candidate origins | ranked index search | scoring of routing keys level by level |

### 3.4 Goals

A design is judged on six properties. G6 is new in v2: the revised design is meant to let the model know where each fact came from, not just retrieve it.

| Goal | Statement | Measured by |
| --- | --- | --- |
| G1 Bounded memory | active state is O(P\_pin + M·B + descriptors) tokens, not O(N) | peak GPU memory, tokens in context |
| G2 Recall | every fact a query needs is reachable, including facts whose relevance appears late | exact match on multi-hop tasks with late binding |
| G3 Stability | retained facts do not change across T updates | retention of turn-1 constraints at turn T |
| G4 Cheap page-in | loading an origin costs far less than re-encoding it | latency per page-in vs prefill |
| G5 Positional validity | every relative distance the model sees is inside its training range | perplexity and retrieval accuracy vs distance |
| G6 Provenance | the model can tell which origin a fact came from, and in what order origins occurred | source-attribution accuracy, archival-order accuracy |

The original proposal targeted G1 to G3 and expected G4 and G5 to follow from a new positional scheme. Sections 4 and 5 show what positions can and cannot deliver; Sections 6 to 9 supply the rest.

## 4 Three facts that constrain any design

Any scheme that pages chunks of a KV cache in and out must respect three facts. The first removes a problem the original proposal set out to solve; the second is the main obstacle to reusing cached chunks; the third is the problem that fresh frames remove.

### 4.1 Re-indexing cached keys is exact and cheap

**Lemma 1 (Exact re-indexing).** Let k = W\_K h be a pre-rotation key and k\_m = R\_{Θ,m} k its cached rotated form. For any shift Δ, R\_{Θ,Δ} k\_m = R\_{Θ,m+Δ} k. Re-indexing an origin of B tokens to a new start position costs one elementwise rotation per cached key: O(B·d) per layer and head, with no matrix multiplies and no model weights.

*Proof.* Each plane's rotations form a commuting group, R(a)R(b) = R(a + b). Apply plane by plane. ∎

The identity holds to 2.2×10⁻¹³ in float64 (Appendix B). Engines can also store pre-rotation keys and rotate at attention time; ShadowKV keeps a low-rank pre-RoPE key cache for this reason \[Sun et al., 2024\]. Prompt Cache reserves position ranges for reusable modules \[Gim et al., 2024\], and Block-Attention \[Ma et al., 2025\] and KVLink \[J. Yang et al., 2025\] re-encode positions of independently cached documents. Under fresh frames (Section 5) no re-indexing is needed at all, because every origin is cached at positions 0 … B − 1 and attended at those same positions.

### 4.2 Deep-layer KV states depend on their original context

**Proposition 2 (Contextual dependence).** For layers l ≥ 1, the key and value of token (c, ℓ) are functions of every token that preceded it when origin c was encoded. If origin c was encoded after context X, its cached KV states generally differ from those it would have after a different context Y. No position map can remove this difference, because it lives in the content of the layer input, not in the rotation.

*Proof sketch.* The layer-1 state of token t is h¹(t) = h⁰(t) + Attn(h⁰ of tokens ≤ t) + MLP(·), which depends on every earlier token, and the layer-2 key is W\_K² LN(h¹(t)). Choose prefixes X and Y that change the attention output at t. Full proof in A.2. ∎

In a toy 4-layer network with random weights, keys of a chunk encoded alone versus after a 64-token prefix agree exactly at layer 0 after re-indexing (cosine 1.00) and drift to 0.96, 0.93 and 0.92 in layers 1 to 3 (Appendix B). That toy only illustrates the mechanism; real models must be measured. CacheBlend states the same issue: precomputed caches of non-prefix chunks miss their cross-attention with the preceding text \[Yao et al., 2025\].

This is the price of the revised design's fresh frames. An origin encoded in isolation is a slightly different object from the same text encoded in sequence. The literature offers four remedies; v2 adds two that fit the red-dot and graph structure, and the protocol measures all six against full recomputation.

| Remedy | Representative work | Cost | Quality risk |
| --- | --- | --- | --- |
| Accept the approximation (parallel encoding) | Parallel Context Windows \[Ratner et al., 2023\]; APE \[X. Yang et al., 2025\] | none | distribution shift; APE recovers 93–98% of sequential performance |
| Selectively recompute a few tokens | CacheBlend \[Yao et al., 2025\] | a fraction of a full prefill | low when the subset is chosen well |
| Train for block independence | Block-Attention \[Ma et al., 2025\]; KVLink link tokens \[J. Yang et al., 2025\] | fine-tuning | low after training |
| Fix per-chunk attention sinks | EPIC \[Hu et al., 2024\] | minimal | partial fix only |
| Red-dot prefix (new here) | encode origin c with its parent's red-dot KV as a prefix; related to KVLink link tokens, landmark tokens and Block-Recurrent state \[Hutchins et al., 2022\] | r extra tokens per origin at encode time | the red dot must carry what the origin needs from its past; measured |
| Co-encode with graph parents (new here) | encode origin c attending to the KV of its top-k graph parents, the mechanism of Graph-KV \[Wang et al., 2025\] | k parent blocks at encode time | parents chosen at ingest may differ from the context it is later read in |

### 4.3 Co-loading distant origins creates out-of-distribution distances

**Proposition 3 (Out-of-range relative distance).** If origins keep their archival positions, co-loading origins c₁ and c₂ exposes relative distances up to |c₁ − c₂|·B + B. With B = 512, origins 0 and 2,000 are about 1.02 million positions apart. When that exceeds L\_train, the low-frequency planes see angles never seen in training.

Known responses are interpolation \[S. Chen et al., 2023\], frequency rescaling \[Peng et al., 2023\], grouped positions \[H. Jin et al., 2024\], capped distances \[Han et al., 2024\] and compact virtual positions in the working window (map M2 or M4, Section 5).

The revised design removes this problem by construction. Under fresh frames every relative distance on the rotary planes lies in (−B, B), whatever the archive length, so G5 holds for any B ≤ L\_train. What fresh frames lose is the ability to tell origins apart, which is the subject of the next section.

## 5 Two negative results: positions alone cannot name an origin

The original idea was to make the origin visible through positions: give it its own rotary coordinate (v1's split planes) or give it a fresh coordinate system (v2's restart of all 64 planes). This section shows that neither, on its own, lets attention tell origins apart. That is the reason Section 6 changes the KV equation itself.

### 5.1 Position maps

**Definition 3 (Position map).** A RoPE-compatible position map assigns each cached token v a coordinate p\_i(v) for every plane i. Plane i of its key is rotated by p\_i(v)·ω\_i, and the logit between query u and key v is Σ\_i ⟨q\_\[i\], R((p\_i(v) − p\_i(u))ω\_i) k\_\[i\]⟩. Standard RoPE is p\_i = m for every plane.

| Map | Local planes | Origin planes | Distances on rotary planes | Origin visible to attention? |
| --- | --- | --- | --- | --- |
| M1 Archival | c·B + ℓ | (no split) | up to the archive length | only through huge distances (Proposition 3) |
| M2 Virtual slot | s(c)·B + ℓ | (no split) | up to M·B | no: archival identity and order are lost |
| M3 Split plane (v1 proposal, like HiRoPE) | ℓ on k planes | c on p planes | < B on local planes | weakly (Proposition 4) |
| M4 Hybrid (v1 recommendation) | s(c)·B + ℓ | g(c) on p planes | up to M·B | yes, partly |
| M5 Fresh frame (v2 revised design) | ℓ on all P planes | none | < B always | no (Proposition 5) |

Standard RoPE already factorizes origin and offset: for m = c·B + ℓ, every plane satisfies R(mω\_i) = R(c·B·ω\_i)·R(ℓω\_i) (checked to 3.6×10⁻¹³). M3 differs only in giving the two coordinates disjoint planes instead of composing them. That is the construction of HiRoPE, which uses some dimensions for token-level and others for function- or statement-level position in code \[Zhang et al., 2024\]. It is also how multi-axis rotary embeddings work: Qwen2-VL's M-RoPE splits rotary dimensions into temporal, height and width parts \[Wang et al., 2024\], RoPE for vision transformers does the same in 2-D \[Heo et al., 2024\], and BiPE mixes an absolute intra-segment encoding with a relative inter-segment one \[He et al., 2024\].

### 5.2 Split planes alias same-offset tokens

We use the alignment kernel K(u, v) = (1/P) Σ\_i cos((p\_i(v) − p\_i(u))ω\_i), the normalized logit when q and k have identical unit content in every plane. It is the envelope behind RoFormer's long-term-decay argument and isolates what positions alone contribute.

**Proposition 4 (Split-plane aliasing).** Under M3 with k local planes and p origin planes, any two tokens with the same offset ℓ in different origins satisfy

```latex
K(u,v) \;\ge\; \frac{k-p}{k+p}
```

for every choice of origin frequencies. *Proof.* The k local terms equal cos 0 = 1; each origin term is at least −1. ∎

With the proposed split (k = 48, p = 16) the bound is 0.5. Measured over 16 origins of 512 tokens (Figure 1), the mean kernel at same-offset tokens in other origins is 0.944 under M3, against 0.096 under M1 and 0.276 under M4. A purely positional head (q = k, temperature 12) puts 10.7% of its mass on these "ghost" tokens and only 11.1% on its own origin under M3. M1 puts 91.3% on its own origin and 0.01% on ghosts; M4 puts 74.0% and 0.03%.

**Corollary 4.1 (Weak separability).** Under M3 the position-only margin between same-offset keys in origins c and c + Δc is (1/P)·Σ\_{i∈O}(1 − cos(Δc·φ\_i)). With p = 16 planes, base 10⁴ and Δc = 2 it is 0.035, that is, 3.5% of the maximum logit.

### 5.3 Fresh frames alias completely

The revised design restarts all 64 planes in every origin. That fixes Proposition 3, but it removes the last positional trace of which origin a token belongs to.

**Proposition 5 (Fresh-frame aliasing).** Under M5, for any two tokens with the same offset in different origins, K(u, v) = 1 exactly. More generally, for any query u and any offset ℓ, the position-only logit of key (c, ℓ) does not depend on c. A head whose logits depend only on positions therefore gives every co-loaded origin the same total mass: 1/C each, for C co-loaded origins.

*Proof.* Under M5, p\_i(c, ℓ) = ℓ for all planes, so the angle difference between u = (c\_u, ℓ\_u) and v = (c, ℓ) is (ℓ − ℓ\_u)ω\_i, independent of c. Every origin contributes the same set of logits, so softmax mass is split equally. ∎

The check confirms it: with 16 co-loaded origins in 4 threads and no other signal, the own origin receives 6.2% of the mass (exactly 1/16), the own thread 25% (4/16), and same-offset ghost tokens in other threads 16.5% (Figure 4, λ = 0).

**What the two results say together.** Positions answer "where inside a chunk". They cannot answer "which chunk" without either breaking positional validity (M1), losing identity (M2), aliasing (M3, M5), or a compromise (M4). The design consequence is to give all 64 planes to local position, as the revised design wants, and to carry origin identity in a separate signal that does not touch the rotary planes. Section 6 defines that signal and computes exactly how strong it must be.

## 6 Origin-Anchored Attention: a new KV equation

The revised design asked for "a new KV methodology": if a new origin's KV is invisible as an origin, add a variable that makes it visible. This section is that variable, written down so it can be implemented and tested.

### 6.1 Definition

**Definition 4 (Origin-Anchored Attention, OAA).** Let token v sit at offset ℓ\_v of origin c\_v, with layer input h\_v. Keys, queries and values are

```latex
k_v = \Big[\, R(\ell_v\omega)\,W_K h_v \;;\; A\,o(c_v) \,\Big], \qquad
q_u = \Big[\, R(\ell_u\omega)\,W_Q h_u \;;\; r_u \,\Big], \qquad
v_v = W_V h_v \;+\; g\,U\,o(c_v)
```

with r\_u = W\_r h\_u, and the attention logit is

```latex
s(u,v) \;=\; \frac{\big(R(\ell_u\omega)W_Q h_u\big)^{\top} R(\ell_v\omega)\,W_K h_v}{\sqrt d} \;+\; \underbrace{r_u^{\top} A\,o(c_v) \;-\; \lambda\, d_{\mathbb H}\big(o(c_u),\,o(c_v)\big)}_{b_u(c_v):\ \text{constant within an origin}}
```

The origin vectors form a chain anchored at the red dots:

```latex
\Phi_{\mathrm{root}} = \mathrm{id}, \qquad \Phi_c(x) = \Phi_{\mathrm{par}(c)}\big(u_c \oplus x\big), \qquad u_c = \exp_0\!\big(\Delta(c)\big), \qquad o(c) = \Phi_c(0) = \Phi_{\mathrm{par}(c)}(u_c), \qquad \Delta(c) = \mathrm{clip}_{\delta_{\max}}\!\Big(W_\Delta\, \mathrm{pool}\big(h_{\rho(\mathrm{par}(c))}\big)\Big)
```

Each origin is stored only as its offset u\_c in the frame of its parent's red dot. Frames are chained by Möbius translations, so along the path from the root o(c) = u₁ ⊕ (u₂ ⊕ (⋯ ⊕ u\_c)), composed right-nested. Every Φ is an isometry, and the coordinates of any descendant in an ancestor's frame need only the offsets below that ancestor; Section 7.3 and Algorithm 2 rely on this. (Composing left-nested instead, ((u₁ ⊕ u₂) ⊕ u₃), would hide rotation terms that depend on global coordinates.)

Why each piece is there:

| Term | What it is | Why it is needed | Where it lives |
| --- | --- | --- | --- |
| R(ℓω) on all P planes | ordinary RoPE inside the origin's fresh frame | keeps every rotary distance below B (G5) | the existing head dimensions |
| r\_uᵀ A o(c\_v) | a content router: the query asks which origin it needs, the key answers with its origin's address | breaks the aliasing of Proposition 5 using content | m extra key dimensions, or equivalently a per-block bias |
| −λ d\_H(o(c\_u), o(c\_v)) | a structural penalty, ALiBi-like but over the origin tree | breaks aliasing using structure, even when content is ambiguous | a per-block bias computed from two m-dimensional points |
| g U o(c\_v) | an origin term in the value | lets the output carry where it read from (G6) | the value vectors; g starts at 0 |
| u\_c = exp\_0(Δ(c)) in the parent's frame | the red-dot chain | each red dot moves the origin by a content-dependent step | the pointer zone, m floats per origin |

The clip bounds each step's hyperbolic length by 2δ\_max, which keeps the chain from running to the boundary of the ball (Section 7.3). A training-free variant fixes o(c) by a deterministic tree layout and fixes λ (Section 7.5).

### 6.2 The logit factors into a router and local attention

**Theorem 6 (Router × local factorization).** Let s(u, v) = s\_loc(u, v) + b\_u(c\_v), where b\_u depends on v only through its origin. Write Λ\_u(c) = log Σ\_{v ∈ c} exp s\_loc(u, v) for the local log-partition of origin c. Then the attention weight factors exactly as

```latex
a(u,v) \;=\; \pi_u(c_v)\cdot \alpha_u(v \mid c_v), \qquad \pi_u(c) = \frac{\exp\big(b_u(c) + \Lambda_u(c)\big)}{\sum_{c'} \exp\big(b_u(c') + \Lambda_u(c')\big)}, \qquad \alpha_u(v\mid c) = \frac{\exp s_{\mathrm{loc}}(u,v)}{\exp \Lambda_u(c)}
```

*Proof.* Multiply and divide exp s(u, v) by exp Λ\_u(c\_v) and sum over origins. Full proof in A.5. ∎

The check reproduces the identity to 5.6×10⁻¹⁷ (Appendix B). In words: OAA is a soft mixture over origins, where origin c gets weight π\_u(c), and inside each origin ordinary attention runs untouched. Hard block routers such as MoBA \[Lu et al., 2025\] and NSA \[Yuan et al., 2025\] choose top-k blocks by a gate; Theorem 6 says OAA's gate is exact and differentiable, and that it is the same number that decides attention.

**Corollary 6.1 (Margin needed for focus).** If the local log-partitions are equal across co-loaded origins (true for positional heads under M5, Proposition 5) and the own origin's bias exceeds every other by β, the own origin's share of attention is e^β / (e^β + C − 1). To reach share ρ the margin must be

```latex
\beta \;=\; \ln\frac{\rho\,(C-1)}{1-\rho}
```

| Co-loaded origins C | 4 | 16 | 64 | 256 | 1,024 | 4,096 |
| --- | --- | --- | --- | --- | --- | --- |
| β for 90% own-origin share (logits) | 3.30 | 4.91 | 6.34 | 7.74 | 9.13 | 10.51 |

The requirement grows only as ln C, and C here is the number of co-loaded origins, M, not the archive size: paging keeps it small.

**Corollary 6.2 (Paging error equals evicted router mass).** If only a set S of origins is resident and attention is renormalized over S, the total-variation distance between the full and the paged attention distributions is exactly 1 − Σ\_{c ∈ S} π\_u(c), and the output changes by at most 2(1 − Σ\_{c∈S} π\_u(c))·max\_v ‖v\_v‖. *Proof in A.6.* ∎

This is the formal link between attention and memory management: a page is worth keeping exactly as much router mass as it carries. The bound uses the true router masses; at inference, masses of non-resident origins are estimated from routing keys, and that estimation error is measured separately (H8).

### 6.3 Provenance through the value term

**Proposition 7 (The output carries its source).** Under OAA the head output is

```latex
y_u \;=\; \sum_{c} \pi_u(c)\, \bar y_u(c) \;+\; g\,U\,\bar o_u, \qquad \bar o_u = \sum_c \pi_u(c)\, o(c)
```

where ȳ\_u(c) is the ordinary local attention output inside origin c. If U has full column rank and its range is orthogonal to the span of the content values, a fixed linear map recovers ō\_u, the router-weighted address of what the head read, from y\_u. *Proof.* Expand the softmax with Theorem 6 and project onto the range of U. ∎

The orthogonality condition will not hold exactly in a trained model; the point is that the information is present by construction and can be probed. The gate g starts at 0, so the modified model starts identical to the base model, in the spirit of zero-initialized adapters. Source attribution (G6) is measured with a linear probe on y\_u and with the model's own answers.

### 6.4 What RoPE's origin signal looks like, for comparison

**Proposition 8 (RoPE carries the origin as a fixed rotation chain).** Under the archival map M1, the logit between (c\_u, ℓ\_u) and (c\_v, ℓ\_v) is Σ\_i ⟨q\_\[i\], R((c\_v − c\_u)·B·ω\_i)·R((ℓ\_v − ℓ\_u)ω\_i)·k\_\[i\]⟩. The origin enters multiplicatively, through a chain whose every step is the same rotation R(Bω), independent of content. *Proof.* Apply the factorization R(mω) = R(cBω)R(ℓω) of Section 5.1 to query and key. ∎

|  | RoPE (M1) | OAA |
| --- | --- | --- |
| How the origin enters | multiplies the content score by a rotation | adds a bias to the logit |
| Chain step | fixed rotation R(Bω) | content-dependent Möbius step exp\_0(Δ(c)) at each red dot |
| Range | grows with archive length, out of distribution beyond L\_train | bounded per step; distances computed in the query's frame |
| Structure | a line | a tree (or DAG) of origins |
| Interferes with local position? | yes, shares the planes | no, separate dimensions or a bias |

### 6.5 Red-dot prefix conditioning

The staircase picture has the new origin forming at the red dot. OAA makes that literal. When origin c is encoded, the r red-dot tokens of its parent are placed just before position 0 of the new frame, at offsets −r … −1, and the new origin's tokens attend to their KV. Three things follow:

- The encoding of origin c is conditioned on what its parent ended on, which shrinks the contextual-dependence gap of Proposition 2 at a cost of r tokens. Whether r = 4 to 16 tokens is enough is measured (H6).
- The pooled red-dot states define the chain step Δ(c), so the origin vector moves according to content, not a fixed rule.
- Red-dot KV is small (r × 128 KiB per token for the 8B shape, about 1 MiB for r = 8) and is kept in the pointer zone, so any origin can be re-entered with its prefix without loading its parent.

Close relatives are KVLink's link tokens \[J. Yang et al., 2025\], landmark tokens \[Mohtashami & Jaggi, 2023\], the recurrent state passed between blocks in Block-Recurrent Transformers \[Hutchins et al., 2022\] and RMT's memory tokens \[Bulatov et al., 2022\]. The difference is that the red dot here also positions the new origin in the address space.

### 6.6 Implementation notes

- **Kernel.** The bias b\_u(c) is constant across the B keys of an origin, so it can be added once per key tile in a FlashAttention-style kernel when the tile size divides B. Cost: one m-dimensional dot product and one distance per (query, co-loaded origin), against B·d multiply-adds for the local scores.
- **Training.** Learned pieces are W\_r, A, U, W\_Δ, λ and g. LoRA on these plus the attention projections is the intended budget. Initialize g = 0 and A small so training starts from the M5 model.
- **Failure modes to monitor.** Router collapse (all mass on one origin, or uniform mass), measured by router entropy; ghost mass on same-offset tokens in other origins; own-origin mass on positional probes; and drift of origin vectors towards the boundary.

## 7 A hyperbolic address space for origins

The revised design bounds the origin chains by a "multi-dimensional circle". We read that circle as the boundary of a Poincaré ball: a bounded region (every coordinate has norm below 1) that nonetheless holds an unbounded, tree-shaped space. This section explains why that reading works, and what has to be done to make it numerically safe.

### 7.1 Addresses

**Definition 5 (Hyperbolic origin address).** Every origin c has an address o(c) in the Poincaré ball B^m. Its **thread** is the direction o(c)/‖o(c)‖; its **level** is its hyperbolic radius R(c) = d\_H(0, o(c)), quantized into shells R\_k = k·δ for k = 0 … L. The session root sits at the centre. Origins on one thread lie along a ray; a topic switch starts a new direction; inner shells hold coarser nodes that summarise the origins beneath them (Section 8).

In the four-quadrant picture, each quadrant's chain of origins is a thread (a ray from the centre), the master plot at the centre is the root, and the red circle is the ball's boundary. Nothing ever reaches the boundary: it is infinitely far away in hyperbolic distance, which is why a bounded ball can hold an unbounded session.

### 7.2 The structural penalty

The penalty −λ·d\_H(o(c\_u), o(c\_v)) is the tree analogue of ALiBi's −λ·|i − j| on a line \[Press et al., 2022\]. Its job is to make attention prefer origins that are structurally close to the query's origin: the same thread first, then its ancestors, then sibling threads.

Why hyperbolic distance can stand in for tree distance: any finite tree embeds in the hyperbolic plane with distortion arbitrarily close to 1 \[Sarkar, 2012\], and the precision and dimension needed for that are characterized by Sala et al. \[2018\]. Learned Poincaré and Lorentz embeddings recover hierarchies in a handful of dimensions \[Nickel & Kiela, 2017; 2018\].

The check builds 4 threads of 4 origins each with fresh frames, a query in thread 0, and a positional head with logits 12·K − λ·d\_H. The nearest origin on the same thread is at hyperbolic distance 0.61 and the nearest origin on another thread at 2.02.

| λ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mass on own origin | 0.062 | 0.327 | 0.580 | 0.743 | 0.848 | 0.913 | 0.951 |
| mass on own thread | 0.250 | 0.774 | 0.967 | 0.996 | 0.999 | 1.000 | 1.000 |
| ghost mass in other threads | 0.165 | 0.050 | 0.0073 | 0.0009 | 0.0001 | < 0.0001 | < 0.0001 |

At λ = 0 this is Proposition 5's uniform split. At λ = 2 the head already stays inside its own thread (97%); at λ = 5 it puts 91% on its own origin. A thread is easier to isolate than a single origin because threads are farther apart than neighbours on one thread. This is a positional head only; content heads will use the router term too.

### 7.3 Re-anchoring at each red dot, and why it is required

Hyperbolic embeddings have a well-known numerical weakness: points far from the centre crowd towards the boundary, where floating-point numbers cannot tell them apart \[Sala et al., 2018; Yu & De Sa, 2019; Mishne et al., 2022\]. A session that grows for hours pushes origins exactly there.

**Proposition 9 (Exact re-anchoring).** For any point a in the ball, T\_a(x) = (−a) ⊕ x is an isometry with T\_a(a) = 0. Therefore every distance can be computed in a frame centred at any red dot, or at the query's own origin, with no change in value. *Proof.* Möbius translations are isometries of the Poincaré ball \[Ganea et al., 2018\]; see A.7. ∎ (Checked to 4.4×10⁻¹⁶.)

**Proposition 10 (Where precision is lost, and where it is needed).** Take a chain of unit steps (δ = 1 in hyperbolic length).

1. *Global coordinates fail at shallow depth.* If every origin is stored in the root's frame, the computed length of one step has more than 10% relative error from depth 16 in float32, depth 7 in float16 and depth 4 in bfloat16.
2. *Local offsets do not.* If each origin is stored as its offset from its parent's red dot, every stored point has radius tanh(δ/2) and the relative error of a step is 2.1×10⁻⁸ (float32), 1.0×10⁻⁴ (float16) and 2.0×10⁻³ (bfloat16) at every depth. Composing bfloat16-stored offsets in higher precision keeps the error of the distance to the root at most 1.2×10⁻³ through depth 12.
3. *Precision is only needed inside the attention horizon.* In the query's frame, a distance d is represented with under 10% error up to d = 18.25 (float32), 9.25 (float16) and 7.0 (bfloat16). Other terms being equal, an origin at distance d receives at most e^(−λd) of the weight of one at distance 0, so origins beyond d\_att = ln(1/τ)/λ get less than a fraction τ. For τ = 10⁻⁴, d\_att is 9.21 at λ = 1, 4.61 at λ = 2 and 1.84 at λ = 5.

*Why.* A point at depth n in global coordinates has Euclidean radius tanh(nδ/2), so its distance to the boundary is about 2e^(−nδ). Once that falls to the spacing of the number format near 1 (2⁻⁸ for bfloat16, 2⁻¹¹ for float16, 2⁻²⁴ for float32), neighbouring points become indistinguishable, and well before that a short step between two far points is computed from a tiny difference of large numbers. Storing offsets keeps every stored radius small; computing in the query's frame puts full precision where attention puts its mass. *Proof and the check's construction in A.8 and Appendix B.* ∎

The practical rule: store origin vectors as offsets from their parent's red dot (bfloat16 is fine), compose them in float32 in the frame of the lowest common ancestor of the two origins being compared, and clamp distances beyond the precision horizon to that horizon. With λ ≥ 2 the clamp changes nothing that attention can see. This is a simple instance of the idea behind tiling-based models, which represent a hyperbolic point as a discrete reference plus a bounded-error offset \[Yu & De Sa, 2019\].

### 7.4 Capacity

**Proposition 11 (Exponential room per shell).** In the hyperbolic plane the circle of radius R has circumference 2π·sinh R; in B^m the sphere of radius R has area proportional to sinh^(m−1) R. With a fixed minimum separation between origins, the number that fit on shell R grows as e^((m−1)R). *Proof.* Standard; see A.9. ∎

| Radius R | 1 | 3 | 6 | 10 |
| --- | --- | --- | --- | --- |
| hyperbolic circumference 2π·sinh R | 7.4 | 62.9 | 1,267.4 | 69,198.2 |
| flat circumference 2πR | 6.3 | 18.8 | 37.7 | 62.8 |

Threads are directions, and directions are plentiful. For 4,096 random thread directions:

| Ball dimension m | 16 | 32 | 128 |
| --- | --- | --- | --- |
| mean \|cos\| between two threads | 0.203 | 0.142 | 0.071 |
| worst pair \|cos\| | 0.957 | 0.790 | 0.465 |
| share of pairs closer than 30° | 8×10⁻⁶ | 0 | 0 |

At m = 32, 4,096 random threads never come within 30° of each other. Deliberate placement (spreading directions rather than drawing them at random) does better still.

### 7.5 A training-free variant to test first

The learned chain of Definition 4 needs fine-tuning. A training-free version needs none and isolates the geometry:

- **Layout.** Place the root at 0. Give each new thread a direction from a well-spread set. Place each new origin one step δ further out along its thread's ray, and each inner node on the shell above its children, at their mean direction.
- **Penalty only.** Use λ·d\_H as a per-block bias with a fixed λ chosen on a development set; no router term, no value term.
- **What it tests.** Whether a pretrained RoPE model, unmodified, benefits from a structural block bias when origins are encoded in fresh frames. Attention masks are the hard version of this bias, which gives some reason to expect tolerance, but it is untested.

### 7.6 Why hyperbolic geometry and not just a tree distance?

An honest reviewer will ask this, so we ask it first. Exact tree distance between two origins can be computed from their lowest common ancestor in O(depth) and used as the penalty directly. What the geometry adds:

- **Learnability.** Addresses are continuous and differentiable, so the red-dot step Δ(c) can be learned from content; a discrete tree cannot be trained by gradient descent.
- **Graceful approximation.** A session is a DAG, and topics overlap. A point can sit between two branches; a tree node cannot.
- **A closed form in the kernel.** One m-dimensional point per origin and one formula give every pairwise distance, with no tree walk inside the attention kernel.
- **Incremental placement.** A new origin is placed by one Möbius step from its parent, without re-embedding anything else.

None of these guarantees better accuracy. Hypothesis H5 compares hyperbolic distance, Euclidean distance on the same tree, and exact tree distance. If exact tree distance matches the hyperbolic penalty, the honest conclusion is that the geometry is an implementation device, not a modelling gain.

## 8 Nested shells: level of detail, memory tier and routing

The nested-balls picture uses circumference and radius to connect chunks. In the model this becomes: radius is level of detail and memory tier, direction is thread, and the two together give a routing procedure. The picture also hides one trap, which Proposition 12 makes precise.

### 8.1 Shells as levels and as tiers

Shell 0 is the root. Shell L, the outermost, holds the origins themselves with their full KV. Shells 1 … L − 1 hold inner nodes, each summarising (extractively, Section 8.4) the b nodes beneath it. A complete b-ary hierarchy over C origins has about (C − 1)/(b − 1) inner nodes, so the inner shells are small and the outer shell is huge, which is exactly the shape of a memory hierarchy.

| Shell | Holds | Count for N = 10⁶, B = 512, b = 8 | Size at one landmark token per node (8B shape) | Tier |
| --- | --- | --- | --- | --- |
| 0 | root: session zone, registers, pinned evidence | 1 | bounded by the pin budget | GPU, always |
| 1 … L − 1 | inner nodes: routing key, merged skeleton, pinned pointers, child list | about 279 (L = 4) | about 35 MiB | GPU for the top levels, CPU memory below |
| L | origins: full KV, red dot, skeleton, edges | 1,953 | about 122 GiB (64 MiB each) | CPU memory or disk; M of them in GPU slots |

Promotion and demotion follow the shells: descending a query pulls the visited inner nodes towards the GPU, and paging in an origin moves one outer-shell block. LMCache already provides the GPU/CPU/disk transfer layer this needs \[Y. Liu et al., 2025\].

### 8.2 Arc versus radius: why the circumference is not a shortcut

**Proposition 12 (Same-shell distance goes through an ancestor).** For two points on the shell of radius R separated by angle θ,

```latex
d_{\mathbb H} \;=\; \operatorname{arcosh}\!\big(\cosh^2 R - \sinh^2 R\,\cos\theta\big) \;=\; \operatorname{arcosh}\!\big(1 + 2\sinh^2 R\,\sin^2(\theta/2)\big) \;\xrightarrow{\;R\to\infty\;}\; 2\big(R - R_a\big), \qquad R_a = \ln\frac{1}{\sin(\theta/2)}
```

The geodesic between them dips towards the centre and comes closest to it at radius arcosh(e^R\_a), which is R\_a + ln 2 for large R\_a. Two leaves at depth R whose common ancestor is at depth R\_a are at tree distance 2(R − R\_a), so hyperbolic distance acts as a tree distance in which the angle between two points sets the depth of their common ancestor. *Proof in A.10.* ∎

| Angle θ | 1° | 10° | 45° | 90° | 180° |
| --- | --- | --- | --- | --- | --- |
| R = 1 | 0.02 | 0.20 | 0.87 | 1.51 | 2.00 |
| R = 3 | 0.17 | 1.58 | 4.11 | 5.31 | 6.00 |
| R = 6 | 2.66 | 7.12 | 10.08 | 11.31 | 12.00 |

At R = 6 two points only 10° apart are already 7.12 apart, more than half of the 12.0 it takes to go through the centre; the asymptotic formula gives 7.12 as well (R\_a = 2.44). Three consequences shape the design:

- **Radius connects levels.** A parent and its child are one radial step apart, the cheapest connection in the space.
- **Angle groups siblings, but only if they share a deep ancestor.** Children of a node at radius R\_a must sit within an angle of order e^(−R\_a) of each other. In the root's frame that angular budget shrinks exponentially, which is the precision problem of Proposition 10 again; in the parent's frame (Proposition 9) the children spread over a normal range of angles.
- **Travelling along the circumference is never short.** A relation that cuts across the hierarchy (a turn in one thread that answers a turn in another, a file used by two tasks) cannot be expressed by putting the two origins near each other without distorting everything around them. Such relations need explicit edges, which is the job of the graph page table (Section 9).

### 8.3 Coarse-to-fine descent

**Proposition 13 (Descent cost).** Route a query from the root through L levels of a b-ary hierarchy, keeping the w best nodes per level. The number of routing scores is b + (L − 1)·w·b instead of b^L for a flat scan of all origins. *Proof.* The first level scores b children of the root; every later level scores the b children of each of w kept nodes. ∎

For b = 8 and L = 4 (4,096 origins) descent needs 32, 56 and 104 scores for w = 1, 2 and 4, against 4,096 for the flat scan. Accuracy under noise, from a simulation in which every routing score is the true relevance (1 for the right branch, 0 otherwise) plus Gaussian noise of standard deviation σ:

| Noise σ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
| --- | --- | --- | --- | --- | --- | --- |
| flat scan of 4,096 | 1.000 | 0.901 | 0.384 | 0.149 | 0.064 | 0.025 |
| descent, beam 1 | 1.000 | 0.996 | 0.837 | 0.479 | 0.249 | 0.131 |
| descent, beam 2 | 1.000 | 0.999 | 0.903 | 0.634 | 0.364 | 0.231 |
| descent, beam 4 | 1.000 | 1.000 | 0.951 | 0.714 | 0.493 | 0.300 |

*Why descent wins here.* A flat scan must beat the maximum of 4,095 noisy competitors; descent only beats a few at each level. The simulation assumes the same noise at every level. Real coarse nodes may be noisier (they summarise more) or cleaner (they aggregate evidence), and an early wrong turn cannot be undone without backtracking. The beam, graph prefetch (Section 9) and a flat-scan fallback when the top scores are close are the mitigations; H8 measures the real trade-off.

Related ideas: hierarchical navigable small-world graphs search coarse layers first \[Malkov & Yashunin, 2020\]; RAPTOR retrieves over a tree of recursive summaries \[Sarthi et al., 2024\]; MemWalker navigates a summary tree with an agent \[H. Chen et al., 2023\]. Hyperbolic space is known to support greedy routing: every connected finite graph has an embedding in the hyperbolic plane in which greedy forwarding always reaches its target \[Kleinberg, 2007\], and hyperbolic geometry explains efficient navigation in complex networks \[Krioukov et al., 2010\].

### 8.4 Inner shells stay extractive

An inner node could hold an abstractive summary of its subtree, as RAPTOR does. We do not let summaries serve as evidence, because rewritten text drifts (Proposition 16). An inner node holds:

- a **routing key**: a landmark token or the mean and bounds of its children's keys;
- a **merged skeleton**: the union of its children's entity sets, each entry pointing to the child that contains it;
- **pinned pointers**: references to exact spans in descendant origins, never copies of rewritten text;
- its **child list** and its own address o(node).

At the agent level an inner node may also carry a short gist for routing, as ReadAgent does \[K.-H. Lee et al., 2024\], but answers are always grounded in an origin's exact text or KV.

## 9 The graph page table

The revised design asks for something like a code-graph tool, but at the KV level: a graph that connects new context to old "so the model knows what is where and what is what", so that paging chunks in and out never loses the one fact that matters. Section 8.2 showed why geometry alone cannot do this. This section adds the graph.

### 9.1 Why: late-bound facts

**Definition 6 (Late-bound hop).** Hop j of a multi-hop query is late-bound if the fact it needs lies in an origin processed before the entity e\_j became relevant. Let h\_late be the number of late-bound hops.

Late binding is what defeats one-shot retrieval and simple forward reading. A footnote on page 3 defines a code name; page 40 reveals that the code name is the answer. By the time the reader knows the footnote matters, it has been evicted. The v1 skeleton (an entity → origin index) handles the case where the entity is named exactly; the graph adds every other kind of link.

### 9.2 What: typed edges

**Definition 7 (Graph page table).** A directed multigraph G = (V, E) over origins and inner nodes, with typed, weighted edges stored in the pointer zone as adjacency lists of at most k\_max entries per node.

| Edge type | Connects | Created when | Used for |
| --- | --- | --- | --- |
| E\_seq (red-dot chain) | origin c to the origin it continues | at ingest, from par(c) | red-dot prefix, archival order, "what came just before" |
| E\_tree (shell) | inner node to its children | at ingest, when a node is attached under a parent | descent (Section 8.3), extractive digests |
| E\_ent (shared entity) | two origins that mention the same entity or identifier | at ingest, by joining skeletons | page faults on entities, prefetch |
| E\_ref (explicit reference) | a turn and the turn it answers, a file and where it is included, a citation and its target | at ingest, from structure the session already has (reply-to ids, file paths, tool-call ids) | co-encoding, prefetch, provenance |
| E\_x (cross-thread) | origins in different threads judged related | at ingest by router similarity or entity overlap, or later from co-access statistics | breaking the geometry's tree assumption where the session is not a tree |

The v1 skeleton is a degenerate case: a bipartite graph between entities and origins with only E\_ent edges. GraphRAG \[Edge et al., 2024\] and HippoRAG \[Gutiérrez et al., 2024\] build similar graphs over text for retrieval; Graph-KV composes KV caches of text segments along a source → target graph \[Wang et al., 2025\]. Here the graph is the page table of a KV memory.

### 9.3 How: three operations driven by the graph

- **Prefetch.** When origin c is paged in, start copying its highest-weight neighbours one tier closer to the GPU. A page fault that the graph predicted costs a CPU-to-GPU copy instead of a disk read.
- **Fault.** When an entity becomes relevant and is not covered by pinned evidence, look it up in the skeletons and follow E\_ent and E\_ref edges from the resident origins; page in the best candidate.
- **Co-encode.** When origin c is first encoded, let it attend to the red-dot KV of its sequential parent and to the KV of its top-k graph parents (E\_ref and E\_ent). This is Graph-KV's mechanism, in which target segments attend only to their designated source segments' caches, used here as a remedy for contextual dependence (Proposition 2): the origin is encoded in the context it is most likely to be read with.

### 9.4 Recall with skeleton and graph

**Definition 8 (Coverage).** γ\_s is the probability that, when entity e is mentioned in origin r and later needed, e ∈ Σ\_r under the matching rule fault() uses (exact identifier, normalized name or embedding match). γ\_g is the probability that the origin holding that fact is reachable by one graph edge from an origin that is resident when the entity becomes relevant.

**Proposition 14 (Recall and page-fault bound).** Assume (A1) every fact an h-hop query needs is stated in one origin together with an entity that later becomes relevant; (A2) every late-bound fact is recoverable, by skeleton or by edge; (A3) relevance is monotone and newly\_relevant flags an entity whenever the origin that reveals it is resident; (A4) select pins the evidence for every relevant entity in a resident origin; (A5) the pin budget holds all h facts; (A6) M ≥ 2. Then the fold-left procedure of Section 10.3 pins all h facts using at most h\_late necessary page faults, with peak working memory P\_pin + 2B tokens. *Proof in A.11.* ∎

**Corollary 14.1 (Partial coverage).** If each late-bound fact is recoverable by skeleton with probability γ\_s and by graph with probability γ\_g, independently, success is at least γ\_eff^h\_late, with

```latex
\gamma_{\mathrm{eff}} \;=\; 1 - (1-\gamma_s)(1-\gamma_g)
```

The check confirms the formula by Monte Carlo (0.8999 simulated against 0.9 predicted for γ\_s = 0.8, γ\_g = 0.5). For two late-bound hops that lifts the bound from 0.64 with the skeleton alone to 0.81 with the graph. The v1 logic simulation of 3-hop chains with two late-bound hops already showed success tracking γ² (0.803 at γ = 0.9 against 0.81 predicted; 0.236 at γ = 0.5 against 0.25), while one-shot retrieval and a fold without skeleton both scored 0 (Figure 2).

The independence assumption is the weak point. The fact a skeleton misses (an unusual name in a footnote) is often the fact an entity edge misses too, so real gains will be smaller than the formula. H3 measures γ\_s, γ\_g and their correlation directly.

### 9.5 Keeping the graph bounded

An unbounded graph would recreate the O(N) problem in the pointer zone. Each node keeps at most k\_max edges per type, ranked by weight, with ties broken towards recency. Edges are cheap (a few bytes each), so k\_max = 16 to 64 is affordable: for 1,953 origins and k\_max = 32 across five types, about 312,000 edges. Edges whose endpoints are never co-accessed decay; edges that a fault actually used are reinforced. This is the working-set idea of Denning \[1968\] applied to links rather than pages.

## 10 The complete system

### 10.1 The memory manager at two levels

| Component | Agent level (any API model) | KV level (open weights) |
| --- | --- | --- |
| Origin | a text chunk of about B tokens with an id | a cached KV block encoded in a fresh frame |
| Origin address | a point o(c) stored in the index, used to rank and to penalize | a point o(c) used inside the attention logit (Definition 4) |
| Shells | an index tree with per-level digests | routing keys per inner node, tiered across GPU, CPU and disk |
| Graph page table | adjacency lists in the index | the same, plus co-encoding at ingest |
| Page-in | a tool call returning the origin's text | a memory copy of the KV block; no re-indexing under fresh frames |
| Session zone | quoted spans and JSON registers with origin back-pointers | pinned KV slices with origin tags |
| What is tested | memory management, recall, drift (H1–H3, H10) | attention geometry, precision, cost (H4–H9) |

### 10.2 Algorithm 1: ingest

```python
def ingest(tokens, S):                         # S: session state (index, graph, stores)
    c   = S.new_origin_id()
    par = S.choose_parent(tokens)              # same thread unless the router sees a topic switch
    pre = S.red_dot_kv(par)                    # r tokens at offsets -r..-1 of the new frame
    gps = S.graph_parents(tokens, k=2)         # E_ref / E_ent candidates found by skeleton match
    kv, hid = encode(tokens,
                     positions=range(len(tokens)),   # fresh frame: all P planes restart at 0
                     prefix=[pre] + [S.kv(p) for p in gps])
    delta = clip(S.W_delta @ pool(hid[-S.r:]), S.d_max)
    S.offset[c] = expmap0(delta)               # stored relative to par's red dot (Prop. 9-10)
    S.parent[c] = par
    S.red_dot[c] = kv[-S.r:]
    S.skeleton[c] = lexical_entities(tokens) | model_entities(tokens)
    S.graph.link(c, par, "seq")
    for p in gps:
        S.graph.link(c, p, "ref")
    for e in S.skeleton[c]:
        for c2 in S.entity_index[e]:
            S.graph.link(c, c2, "ent")
        S.entity_index[e].add(c)
    S.attach_to_shell(c)                       # update routing keys and merged skeletons upward
    S.store(c, kv, tier="cpu")                 # the outer shell lives off the GPU
    return c
```

choose\_parent compares the new chunk's pooled state with the heads of recent threads using the router; below a similarity threshold it starts a new thread under the closest inner node. lexical\_entities is a deterministic pass (identifiers, numbers with units, capitalized names, file paths) and is the safest base layer, because a skeleton built only by a model can miss exactly the footnote that later matters.

### 10.3 Algorithm 2: query

```python
def answer(Q, S, M, w):
    frontier = descend(Q, S.root, beam=w)                      # Section 8.3
    cand  = rank_by_router_estimate(Q, frontier)
    cand |= S.graph.neighbors(cand, kinds=("ref", "ent", "x"), top=8)
    resident = page_in(top(cand, M))                           # router mass decides (Cor. 6.2)
    prefetch(next_best(cand))
    frame = query_frame(S, origin_of(Q))                       # re-anchor at the query's origin, float32
    bias = {c: router(Q, frame[c]) - S.lam * min(dist0(frame[c]), S.d_prec) for c in resident}
    W = fold_left(Q, resident, S)                              # Algorithm 3
    return generate(Q, W, resident, bias)
```

query\_frame computes each distance in the frame of the two origins' lowest common ancestor. The offsets below that ancestor are composed right-nested, u₁ ⊕ (u₂ ⊕ …), in float32, so precision depends on depth below the ancestor, not on depth in the session (Proposition 10). In protocol\_snippets\_v2.py two neighbouring origins at depth 40 keep a float32 distance error below 10⁻⁵ this way, while their float32 global coordinates have already collapsed onto the boundary. d\_prec is the precision horizon of the format, and the clamp is invisible to attention whenever λ·d\_prec exceeds ln(1/τ).

### 10.4 Algorithm 3: fold-left with skeleton and graph faults

The v1 fold-left reads origins in order, pins evidence, and backtracks with page faults when an entity becomes relevant after its fact was evicted. v2 adds the graph as a second lookup route.

```python
W = PinnedSet(budget=P_pin)              # exact spans or KV slices, each with its origin
R = relevant_entities(Q)                 # entities the query names
tried = set()                            # (entity, origin) pairs already faulted

def sources(e, c):                       # where entity e's fact may live, before origin c
    return S.skeleton_lookup(e, before=c) | S.graph.neighbors(c, kinds=("ent", "ref", "x"), mentioning=e)

for c in origins_in_order(resident_first=True):
    page_in(c)
    R |= newly_relevant(c, W, Q)
    for e in R - covered(W):
        W.pin(select(c, e))              # evidence present in the current origin
    pending = [(e, r) for e in R - covered(W) for r in sources(e, c) if (e, r) not in tried]
    while pending:                       # fixpoint: a fault can reveal new entities
        e, r = pending.pop()
        tried.add((e, r))
        page_in(r)                       # page fault into the second slot
        W.pin(select(r, e))
        new = newly_relevant(r, W, Q) - R
        R |= new
        evict(r)
        for e2 in new:
            W.pin(select(c, e2))         # the current origin is still resident
        pending += [(e2, r2) for e2 in new - covered(W) for r2 in sources(e2, c) if (e2, r2) not in tried]
    W.enforce_budget()                   # evict lowest-utility pins, keep pointers
    evict(c)
return generate(Q, W, pointer_zone())
```

Spurious matches add faults but never lose recall; a deployment should report necessary and spurious faults separately.

### 10.5 Thrashing

**Proposition 15 (Thrashing condition).** Following Denning's working-set model \[Denning, 1968\], let a reasoning step need facts from r origins at once. Without pinning, the step thrashes when r > M: each page-in evicts an origin that is still needed. With extraction into W, the step's working set is one origin plus W, so it does not thrash whenever the pinned evidence fits in the pin budget. *Proof.* Direct from the definition of the working set and the slot count. ∎

This is the formal reason for the session zone: pinned facts are registers that let the working set shrink to one page. Belady's analysis of replacement algorithms \[Belady, 1966\] is the baseline for how well any eviction rule can do.

### 10.6 Extractive session state and drift

**Definition 9 (Abstractive and extractive state).** An abstractive state is rewritten by the model each step, S\_t = Dec(S\_{t−1}, C\_t). An extractive state W\_t is a set of exact spans (agent level) or exact KV slices (KV level), each with its origin pointer, copied by reference and never regenerated.

**Proposition 16 (Rewrite-loss model).** If each abstractive rewrite independently keeps a given fact with probability 1 − ε, the fact survives T rewrites with probability (1 − ε)^T: 36.4% at ε = 0.02 and T = 50, and 77.8% at ε = 0.005. Under extractive pinning a retained item is identical at every later step; its survival equals the selection recall at pin time and falls only through explicit eviction. *Proof in A.12.* ∎

"Zero drift" is therefore true of the representation of retained items, and only of that. The error budget moves to selection and eviction, and page faults make those errors recoverable when the fact's key is indexed. Three caveats apply. Losses are rarely independent: a model that habitually rounds numbers makes abstractive state worse than the formula predicts. The v1 draft's linear O(Tδ) drift claim had no support and stays dropped. KV-level pinning means attending a sparse subset of tokens away from their neighbours, the regime of H2O \[Zhang et al., 2023\] and SnapKV \[Li et al., 2024\], so its quality must be measured.

Persistent state is also an attack surface. Instructions injected into an archived document can be pinned and persist across a session. Pinned items carry provenance and a trust level, and pinned text is never treated as instructions.

## 11 Complexity, memory and bandwidth

OPA's active memory is bounded by the pin budget, the working slots and the pointer zone; its page-in cost is a memory transfer instead of a prefill; and the hyperbolic machinery adds a cost that is small next to attention itself.

| Method | Active KV on GPU (tokens) | Decode attention per token | Cost to bring in B tokens | Recall of evicted text |
| --- | --- | --- | --- | --- |
| Dense full context | N | O(N·d) | none (resident) | exact, but mid-context use degrades \[N. F. Liu et al., 2024\] |
| Token-level RAG | M·B + prompt | O(M·B·d) | full prefill: O(B·(L\_ctx + B)·d) attention + about 2·Φ·B weight FLOPs | only by re-retrieval |
| Eviction (StreamingLLM, H2O, SnapKV) | fixed budget | O(budget·d) | not applicable | lost |
| Block retrieval (Landmark, Quest, InfLLM) | Quest keeps all N on GPU; InfLLM offloads and loads top-K blocks | O(K·B·d) + block scoring O((N/B)·d) | none (Quest) or a host-to-GPU copy (InfLLM) | per-query top-K, no accumulator |
| KV reuse (CacheBlend, Block-Attention, KVLink) | M·B | O(M·B·d) | copy + recompute of a fraction of tokens, or none after fine-tuning | not addressed |
| OPA v2, KV level | P\_pin + M·B + inner-shell keys | O((P\_pin + M·B)·d) + M bias terms of O(m) | copy only (fresh frames need no re-indexing) | page faults through skeleton and graph |
| OPA v2, agent level | tokens of W + M·B + index results | as RAG | prefill of B tokens through the API | page faults through skeleton and graph |

Φ is the parameter count and L\_ctx the tokens already in context.

**Bytes per origin.** A KV page-in moves

```latex
\text{bytes} \;=\; 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per value} \times B
```

For a Llama-3-8B-shaped model (32 layers, 8 grouped KV heads, d = 128, 16-bit values) that is 128 KiB per token and 64 MiB per 512-token origin, about 2.7 ms at an assumed 25 GB/s effective host-to-GPU rate. Re-encoding the same origin costs about 2·Φ·B = 8.2×10¹² FLOPs for Φ = 8×10⁹. Whether the copy beats the prefill depends on hardware, batch size and overlap with compute, so it is measured (H9).

**What v2 adds per origin.** The address: m values (64 bytes at m = 32 in 16-bit). The red dot: r tokens of KV (1 MiB for r = 8). The skeleton and edges: a few kilobytes. The red dot is the only sizable addition, about 1.6% of the origin's KV at r = 8.

**Hyperbolic overhead.** For each query and each co-loaded origin, the bias needs one m-dimensional router product and one distance (about 200 FLOPs at m = 32 plus one arcosh). Attention over that origin's B keys needs B·d multiply-adds, 65,536 for B = 512 and d = 128. The bias is about 0.2% of the attention work for that origin, and it is computed once per query, not per key.

**Routing.** A flat scan of descriptors costs C scores per query (1,953 for N = 10⁶); descent costs b + (L − 1)·w·b (32 to 104 for b = 8, L = 4). At the agent level each score is an index lookup; at the KV level it is a dot product with a routing key.

**Pointer-zone size.** About 279 inner nodes and 1,953 origins for N = 10⁶, B = 512, b = 8. With one landmark token per inner node the inner shells take about 35 MiB on the GPU; the origins' routing keys can stay in CPU memory and be promoted as descent reaches them.

## 12 Related work and positioning

Every OPA component has a close relative. The table maps each part of the revised design to its nearest prior work and says what differs.

| OPA component | Closest prior work | What the prior work does | What differs here |
| --- | --- | --- | --- |
| Fresh frame per origin (M5) | Parallel Context Windows \[Ratner et al., 2023\]; APE \[X. Yang et al., 2025\]; CEPE \[Yen et al., 2024\]; Star Attention \[Acharya et al., 2025\] | encode chunks in parallel with shared or restarted positions | used as the frame of a paged memory, with an explicit fix for the aliasing it causes |
| Origin index on separate planes (v1) | HiRoPE \[Zhang et al., 2024\]; M-RoPE \[Wang et al., 2024\]; BiPE \[He et al., 2024\] | hierarchical or multi-axis positions inside RoPE | v2 moves the origin out of the rotary planes (Proposition 4 shows why) |
| Router term and factorization | Landmark Attention \[Mohtashami & Jaggi, 2023\]; MoBA \[Lu et al., 2025\]; NSA \[Yuan et al., 2025\]; InfLLM \[C. Xiao et al., 2024\] | a block-level gate or landmark selects blocks | exact soft factorization (Theorem 6) whose router mass is also the paging criterion (Corollary 6.2) |
| Content-dependent origin increments | CoPE \[Golovneva et al., 2024\] | positions that count content-defined units | the increment moves an address in a ball, at chunk granularity |
| Red dot | KVLink link tokens \[J. Yang et al., 2025\]; Block-Recurrent Transformers \[Hutchins et al., 2022\]; RMT \[Bulatov et al., 2022\] | carry state across segment boundaries | the carried state also positions the next origin |
| Hyperbolic addresses | Poincaré and Lorentz embeddings \[Nickel & Kiela, 2017; 2018\]; hyperbolic networks \[Ganea et al., 2018; Shimizu et al., 2021; W. Chen et al., 2022\]; hyperbolic attention \[Gulcehre et al., 2018\]; HGCN \[Chami et al., 2019\]; Hypformer \[M. Yang et al., 2024\]; HELM \[He et al., 2025\] | represent tokens, nodes or whole models in hyperbolic space | only the address of a chunk is hyperbolic; token computation stays Euclidean |
| Hyperbolic positional encodings | HoPE \[Dai et al., 2025\]; HyPE \[Angelotti, 2023\] | HoPE uses Lorentz-style hyperbolic rotations for monotone decay with distance; HyPE uses hyperbolic functions to approximate ALiBi's bias and leaves experiments to future work | these act on token positions on a line; OPA's penalty acts on chunk addresses on a tree |
| Structural penalty | ALiBi \[Press et al., 2022\] | linear distance bias on a line | distance bias over a tree, computed in the query's frame |
| Precision handling | Sala et al. \[2018\]; tiling-based models \[Yu & De Sa, 2019\]; numerical-stability analysis \[Mishne et al., 2022\] | characterize and bound float error of hyperbolic embeddings | re-anchoring at red dots plus an attention-horizon argument for where precision matters |
| Nested shells and descent | HNSW \[Malkov & Yashunin, 2020\]; RAPTOR \[Sarthi et al., 2024\]; MemWalker \[H. Chen et al., 2023\]; greedy hyperbolic routing \[Kleinberg, 2007; Krioukov et al., 2010\] | coarse-to-fine search and summary trees | shells double as memory tiers; inner shells are extractive |
| Graph page table | Graph-KV \[Wang et al., 2025\]; GraphRAG \[Edge et al., 2024\]; HippoRAG \[Gutiérrez et al., 2024\] | structure-aware KV composition, graph-structured retrieval | the graph is the page table: prefetch, faults and co-encoding |
| Paging and tiers | PagedAttention \[Kwon et al., 2023\]; LMCache \[Y. Liu et al., 2025\]; InfiniGen \[W. Lee et al., 2024b\]; ShadowKV \[Sun et al., 2024\] | physical KV blocks, offloading, prefetching | a logical address space above these layers |
| Agent memory | MemGPT \[Packer et al., 2023\]; ReadAgent \[K.-H. Lee et al., 2024\]; Chain of Agents \[Y. Zhang et al., 2024\]; Mem0 \[Chhikara et al., 2025\]; A-MEM \[Xu et al., 2025\] | paging, gists, message passing, long-term stores | extractive accumulator, skeleton and graph faults, provenance |

**Capability comparison.**

| System | Fresh frame per chunk | Chunk identity in attention | Structured distance between chunks | Tiered paging | Backtrack to evicted chunks | Extractive state |
| --- | --- | --- | --- | --- | --- | --- |
| PCW / APE | yes | no | no | no | no | no |
| Landmark / InfLLM | no | landmark or representative tokens | no | InfLLM offloads | per query | no |
| MoBA / NSA | no | block gate | no | no | per query | no |
| HiRoPE / BiPE | no | positions in split planes or segments | a line or two levels | no | no | no |
| Graph-KV | no (allocated positions) | via the block mask | graph edges | no | no | no |
| HELM / Hypformer | no | no | hyperbolic token geometry | no | no | no |
| RAPTOR / MemWalker | not applicable (text) | no | summary tree | no | via the tree | no (abstractive) |
| MemGPT / ReadAgent | not applicable (text) | text identifiers | no | yes (agent level) | via search or page lookup | partly |
| OPA v2 | yes | origin vector: router + penalty + value term | hyperbolic tree + graph edges | yes, by shell | skeleton and graph faults | yes |

**Was this tried and failed?** No prior work we found combines these pieces, so there is no direct failure to point to. Two known weaknesses apply. Parallel encoding loses some quality against sequential encoding (APE reports recovering 93–98% of it), which is why OPA measures six contextual-dependence remedies. Hyperbolic models have a documented history of numerical trouble \[Sala et al., 2018; Yu & De Sa, 2019; Mishne et al., 2022\], which is why Section 7.3 exists. The pre-submission checklist in the companion dossier lists the searches to repeat before submission, because this area moves monthly.

**Further context.** Retrieval-augmented generation \[Lewis et al., 2020\] and retrieval interleaved with reasoning (IRCoT \[Trivedi et al., 2023\], FLARE \[Z. Jiang et al., 2023\]) are the agent-level baselines. Unlimiformer \[Bertsch et al., 2023\], Memorizing Transformers \[Wu et al., 2022\], RetrievalAttention \[D. Liu et al., 2024\] and DuoAttention \[G. Xiao et al., 2025\] retrieve or specialize attention over long KV stores. Transformer-XL \[Dai et al., 2019\], Compressive Transformers \[Rae et al., 2020\], Infini-attention \[Munkhdalai et al., 2024\], AutoCompressors \[Chevalier et al., 2023\], gist tokens \[Mu et al., 2023\], Cartridges \[Eyuboglu et al., 2025\] and LLMLingua \[H. Jiang et al., 2023b\] carry or compress state in learned, lossy form, which is the contrast with OPA's extractive accumulator. RAGCache \[C. Jin et al., 2024\], TurboRAG \[S. Lu et al., 2024\] and RadixAttention in SGLang \[Zheng et al., 2024\] reuse cached KV across requests. On positions, NoPE analyses \[Kazemnejad et al., 2023\] and frequency studies \[X. Liu et al., 2024; Men et al., 2024\] bear on which planes are safe to repurpose, the question v2 sidesteps by keeping all planes local.

## 13 Evaluation hypotheses and the decisive ablation

Each claim becomes a hypothesis with a kill criterion stated before any data is seen. Agent-level tests need only API access; model-level tests need a 1–8B open-weights model, LoRA and a custom attention kernel. Datasets, baselines, logging and statistics are in the companion Experiment Protocol doc.

### 13.1 The decisive ablation (H5)

The revised design's most distinctive claim is that a hyperbolic address space helps. That claim is easy to fool yourself about, because any structure at all (threads, a tree) may help, and the geometry may get credit for the structure. The ablation therefore holds the structure fixed and changes only the distance:

| Arm | Penalty between origins | What it isolates |
| --- | --- | --- |
| A0 | none (λ = 0) | fresh frames plus router only |
| A1 | Euclidean distance between flat embeddings of the same tree | structure without hyperbolic geometry |
| A2 | hyperbolic distance d\_H between the origins' addresses | the proposed method |
| A3 | exact tree distance (path length through the lowest common ancestor) | the structure itself, computed exactly |
| A4 | A2 with the learned red-dot chain instead of a fixed layout | learning the addresses from content |

Every arm gets the same tree, the same λ search on a development split, and the same token budget. Run it twice: at the agent level, where the distance re-ranks candidate origins, and at the model level, where it is the attention bias. Reading the outcomes:

- **A2 > A1 and A2 ≈ A3.** Hyperbolic geometry is a faithful, cheap way to compute tree distance inside attention. A useful but modest claim.
- **A2 > A3.** The continuous geometry does something an exact tree cannot (soft placement between branches, learned addresses). The strongest outcome.
- **A2 ≤ A1.** The geometry adds nothing; drop the hyperbolic claim and keep the rest.
- **A4 > A2.** Content-learned addresses beat a fixed layout; the red-dot chain earns its training cost.

### 13.2 All hypotheses

| ID | Hypothesis | Level | Primary metric | Kill criterion |
| --- | --- | --- | --- | --- |
| H1 | Fold-left with skeleton and graph faults beats one-shot retrieval on late-binding multi-hop questions | agent | exact match, tokens used | gain < 5 points over top-k RAG, or < 3 over a ReadAgent-style baseline, at equal token budget |
| H2 | Extractive session state keeps turn-1 constraints through 50 turns | agent | exact retention at turn 50 | not ≥ 10 points above a rolling summary |
| H3 | Measured coverage predicts success as about γ\_eff^h\_late, and graph edges add recall beyond skeletons | agent | R² of predicted vs observed success; gain from adding the graph | R² < 0.5, or the graph adds < 2 points on late-bound chains |
| H4 | Fresh frames alone alias origins; the origin vector repairs it and adds provenance | model | own-origin mass on positional probes, same-offset confusion, source-attribution accuracy | M5 alone shows no aliasing, or M5 + origin vector shows no provenance gain over M4 |
| H5 | Hyperbolic distance beats Euclidean distance on the same tree (Section 13.1) | both | thread-confusion error, multi-thread late-binding accuracy | A2 not better than A1 beyond the 95% interval |
| H6 | Red-dot prefix and graph-parent co-encoding close most of the contextual-dependence gap | model | KL to full recomputation, task accuracy | no remedy recovers ≥ 90% of full-recomputation accuracy, or the red-dot prefix is no better than none |
| H7 | Global coordinates fail at the predicted depths; re-anchored offsets do not | model (kernel) | distance error vs depth in bfloat16, float16, float32 | re-anchored error grows with depth |
| H8 | Descent through shells matches flat routing with far fewer scores | both | recall of the needed origin in the top M; scores per query; error of estimated router mass | descent loses > 3 points of recall at equal M |
| H9 | Page-in by copy is faster than re-encoding above a measurable chunk size, at each tier | systems | latency per origin by tier, peak GPU memory | copy never beats prefill on the test hardware |
| H10 | Pinning prevents thrashing when a step needs more origins than slots | agent | repeated page-ins per step, accuracy | no reduction in repeated page-ins |

Positive results on H1–H3 and H10 alone would support an agent-level paper on memory management. H4–H9 are needed for any claim about the attention geometry. H5 decides whether the word "hyperbolic" belongs in the title.

## 14 Limitations, disclosure and conclusion

### 14.1 Limitations

- **No language-model evidence yet.** Every number here comes from algebraic checks, a positional-kernel analysis, a random-weight toy network, a logic simulation of page faults, and synthetic routing and geometry simulations.
- **Assumptions carry the guarantees.** Proposition 14 rests on (A1)–(A6); Corollary 14.1 and Proposition 16 assume independence. Real skeletons, real graphs and real summaries fail in correlated ways.
- **The alignment kernel and the positional head are envelopes.** They show what positions and the penalty contribute, not how a trained model distributes attention.
- **Contextual dependence remains the largest quality risk.** Fresh frames make every origin an isolated encoding. The red-dot prefix and co-encoding are plausible remedies, not proven ones.
- **The tree assumption is approximate.** Sessions are DAGs with overlapping topics. The graph page table is the escape hatch, and its bounded edge budget may drop the very link that matters.
- **Hyperbolic training is fragile.** NaN failures near the boundary are documented \[Mishne et al., 2022\]. Re-anchoring, clipping the chain step and computing distances in float32 are the mitigations; router collapse is a separate risk to monitor.
- **The geometry may not beat exact tree distance.** If H5 finds A2 ≈ A3, the honest framing is "a differentiable tree distance", not "hyperbolic memory".
- **Skeleton and edge quality is the crux of recall.** Recall is bounded by γ\_eff, and building skeletons with a model reintroduces the lookahead problem one level down.
- **KV-level OPA needs open weights, a custom kernel and fine-tuning** for every learned piece. The training-free variant of Section 7.5 is the cheapest first test.
- **Persistent state is an attack surface.** Injected instructions can be pinned and persist; pinned items carry provenance and trust levels and are never executed as instructions.

### 14.2 Disclosure of AI assistance

This manuscript was drafted with help from AI assistants (Google Gemini and Anthropic Claude). The author is responsible for all content. Every reference was checked against its arXiv, DOI or publisher record on 6 November 2026, and every number in Sections 4–11 and Appendix B was produced by the code described in Appendix B.

### 14.3 Conclusion

Treating context as virtual memory works once the right constraints are named. Re-indexing cached positions is cheap, and fresh frames make positions always valid. But positions alone cannot name an origin: split planes alias same-offset tokens and fresh frames alias them completely. Origin-Anchored Attention fixes that outside the rotary planes, with an exact router × local factorization whose router mass is also the paging criterion. A Poincaré ball gives origins a bounded address space with exponential room, in which radius is level and tier, direction is thread, and same-shell distance runs through a common ancestor; Möbius re-anchoring at every red dot keeps it numerically safe. A graph page table covers what geometry cannot, and the v1 memory manager turns all of this into recall that degrades gracefully with coverage and state that does not drift. Section 13 says how each of these claims can be shown wrong.

## Appendix A — Proofs

**A.1 Lemma 1.** Planar rotations about the origin form the abelian group SO(2), so R(a)R(b) = R(a + b) for all angles. A cached key k\_m holds plane i rotated by mω\_i; applying R(Δω\_i) to that plane yields rotation (m + Δ)ω\_i, exactly what a fresh encoding at position m + Δ would cache at the rotation step. The cost is 4 multiply-adds per plane per token, so O(B·d) per origin per layer and head. ∎

**A.2 Proposition 2.** Take one attention layer before the layer of interest. Choose prefixes X and Y and a token t of origin c whose layer-1 attention places non-zero weight on a prefix token whose value vector differs between X and Y. Then the layer-1 state of t differs, and for any layer-2 key projection outside a measure-zero set the layer-2 key differs. Position maps act only by rotating already-formed queries and keys, so they cannot undo a difference in the layer input. ∎

**A.3 Proposition 4 and Corollary 4.1.** Under M3, plane i's angle difference is (ℓ\_v − ℓ\_u)θ\_i for local planes and (c\_v − c\_u)φ\_i for origin planes. With ℓ\_u = ℓ\_v the k local terms are cos 0 = 1, so K = (k + Σ\_{i∈O} cos(Δc·φ\_i))/P ≥ (k − p)/(k + p). The margin to the self-kernel K = 1 is (1/P)·Σ\_{i∈O}(1 − cos(Δc·φ\_i)); with p = 16, φ\_i = b^(−2i/32) for i = 0 … 15, b = 10⁴ and Δc = 2 it is 0.0355. ∎

**A.4 Proposition 5.** Under M5 every plane of every token in origin c is driven by ℓ alone. The angle difference between query (c\_u, ℓ\_u) and key (c, ℓ) is (ℓ − ℓ\_u)ω\_i, independent of c, so the position-only logit z(ℓ − ℓ\_u) is the same function for every origin. Origin c's softmax mass is Σ\_ℓ e^z(ℓ−ℓ\_u) / (C·Σ\_ℓ e^z(ℓ−ℓ\_u)) = 1/C. For ℓ = ℓ\_u every plane contributes cos 0 = 1, so K = 1. ∎

**A.5 Theorem 6.** For a key v in origin c,

```latex
a(u,v) = \frac{e^{s_{\mathrm{loc}}(u,v) + b_u(c)}}{\sum_{c'}\sum_{v' \in c'} e^{s_{\mathrm{loc}}(u,v') + b_u(c')}}
= \frac{e^{b_u(c)+\Lambda_u(c)}}{\sum_{c'} e^{b_u(c')+\Lambda_u(c')}} \cdot \frac{e^{s_{\mathrm{loc}}(u,v)}}{e^{\Lambda_u(c)}} = \pi_u(c)\,\alpha_u(v\mid c)
```

using Σ\_{v'∈c'} e^s\_loc(u,v') = e^Λ\_u(c'). Both factors are probability distributions (over origins, and over keys within an origin). ∎

**A.6 Corollaries 6.1, 6.2 and Proposition 7.** (6.1) With equal Λ\_u and bias b + β on the own origin and b on the C − 1 others, π\_u(own) = e^β/(e^β + C − 1). Setting this to ρ and solving gives β = ln(ρ(C − 1)/(1 − ρ)). (6.2) Let Z\_S = Σ\_{c∈S} π\_u(c). The paged distribution is a\_S(v) = a(v)/Z\_S on S and 0 elsewhere. Then ½Σ\_v|a(v) − a\_S(v)| = ½\[(1/Z\_S − 1)·Z\_S + (1 − Z\_S)\] = 1 − Z\_S, and ‖Σ\_v (a(v) − a\_S(v)) v\_v‖ ≤ Σ\_v|a(v) − a\_S(v)|·max‖v\_v‖ = 2(1 − Z\_S)·max‖v\_v‖. (Prop. 7) y\_u = Σ\_v a(u,v)(W\_V h\_v + g U o(c\_v)) = Σ\_c π\_u(c) ȳ\_u(c) + g U Σ\_c π\_u(c) o(c) by Theorem 6. If range(U) is orthogonal to the span of the content values and U has full column rank, the projection onto range(U) followed by U⁺ returns g·ō\_u. ∎

**A.7 Proposition 9.** In the Poincaré ball the distance can be written d\_H(x, y) = 2·artanh‖(−x) ⊕ y‖ \[Ganea et al., 2018\]. Write T\_a(x) = (−a) ⊕ x. The gyrotranslation identity of Möbius addition gives (−T\_a x) ⊕ T\_a y = gyr\[−a, x\]((−x) ⊕ y), where gyr\[·,·\] is an orthogonal map, so ‖(−T\_a x) ⊕ T\_a y‖ = ‖(−x) ⊕ y‖ and d\_H(T\_a x, T\_a y) = d\_H(x, y). T\_a(a) = (−a) ⊕ a = 0 by left cancellation. The check evaluates both sides on random points and finds a difference of 4.4×10⁻¹⁶. ∎

**A.8 Proposition 10.** *Construction.* Points of a chain lie on one ray at hyperbolic radii nδ, that is, Euclidean radii r\_n = tanh(nδ/2). Global storage rounds r\_n and r\_{n+1} to the format and computes the step length with the distance formula. Re-anchored storage represents the next point in the previous point's frame, at radius tanh(δ/2), so every step is the n = 0 case. The query-frame horizon is the smallest d for which 2·artanh(round(tanh(d/2))) is off by more than 10% (or the rounded radius reaches 1). The chain-composition check stores each offset in the low-precision format and composes in float64. *Estimate.* Near the boundary 1 − r\_n ≈ 2e^(−nδ) and d(0, r) ≈ ln(2/(1 − r)). A rounding error of at most u in r (u = 2⁻⁹, 2⁻¹² and 2⁻²⁵ just below 1 for bfloat16, float16 and float32) perturbs ln(1 − r\_n) by about u·e^(nδ)/2, so the relative error of a unit step is of order u·e^(nδ) and exceeds 10% near n\* ≈ ln(0.1/u)/δ: 3.9, 6.0 and 15.0, against observed failure depths 4, 7 and 16. Re-anchored storage keeps r = tanh(δ/2) ≈ 0.46, far from 1, at every depth. *Attention horizon.* With other terms equal, π\_u(c)/π\_u(c₀) = e^(−λ(d − d₀)); with d₀ = 0 this is below τ once d > ln(1/τ)/λ. ∎

**A.9 Proposition 11.** In H² the circle of radius R has circumference 2π·sinh R; in H^m the geodesic sphere has area ω\_(m−1)·sinh^(m−1) R, where ω\_(m−1) is the area of the unit Euclidean sphere. A packing with minimum separation s fits at least that area divided by the area of a ball of radius s on the sphere, which is bounded independently of R for R ≥ s, so the count grows as e^((m−1)R). ∎

**A.10 Proposition 12.** The hyperbolic law of cosines for a triangle with two sides R, R from the centre and angle θ between them gives cosh d = cosh²R − sinh²R·cos θ = 1 + sinh²R(1 − cos θ) = 1 + 2 sinh²R·sin²(θ/2). For large R, cosh d ≈ ½ e^(2R) sin²(θ/2) and arcosh x = ln(2x) + O(x⁻²), so d ≈ 2R + 2 ln sin(θ/2) = 2(R − R\_a). The geodesic's closest approach to the centre is the altitude h of the isosceles triangle, with cosh R = cosh h·cosh(d/2); as R grows, cosh h → e^(R − d/2) = e^R\_a, so h → arcosh(e^R\_a) = R\_a + ln 2 − o(1). For R = 6 and θ = 10°: d = 7.12, R\_a = 2.44 and h = 3.13. ∎

**A.11 Proposition 14 and Corollary 14.1.** Let e₁, …, e\_h be the hop entities in the order they become relevant, at origins a₁ ≤ … ≤ a\_h, with facts in origins f₁, …, f\_h. Induction on j. If f\_j ≥ a\_j, origin f\_j is resident at or after the step where e\_j becomes relevant, and select pins the fact (A4) with no fault. If f\_j < a\_j (hop j is late-bound), (A1) and (A2) put f\_j in sources(e\_j, a\_j), either through e\_j ∈ Σ\_(f\_j) or through an edge from a resident origin; the fault loads f\_j into the second slot (A6) and select pins the fact (A4). Facts revealed during a fault enter the pending list, so cascades are handled; the tried set bounds faults per (entity, origin) pair at one, and (A3) means each entity is processed once, so necessary faults number at most h\_late. Peak memory is the pin budget (A5) plus the current origin and one faulted origin. For the corollary: success requires every late-bound fact to be recovered; under independence each is recovered with probability P(skeleton ∪ graph) = 1 − (1 − γ\_s)(1 − γ\_g) = γ\_eff, giving γ\_eff^h\_late, and other recovery routes can only raise it. ∎

**A.12 Proposition 16.** Survival after T independent rewrites is a product of T Bernoulli(1 − ε) survivals, (1 − ε)^T. An extractive pin is never regenerated, so its content is unchanged at every step; it is lost only by an explicit eviction. ∎

## Appendix B — Numerical checks

All checks are synthetic and seeded (numpy, seed 0; the v2 additions 7a–7f use a separate seed 1 so that earlier results do not move). The scripts are checks.py (v1), checks\_v2.py (v2) and protocol\_snippets.py (a tested reference implementation) in the opa\_checks folder that accompanies this draft; summary.json and summary\_v2.json hold every number quoted. They test algebra, geometry and routing logic, not language-model quality.

### B.1 v1 checks (positions, contextual dependence, page faults)

**Setup.** Head dimension d = 128 (P = 64 planes), base 10⁴, B = 512, 16 origins, query at origin 8, offset 100. M3 uses k = 48 local planes and p = 16 origin planes, as in the original proposal. M4 drives the local planes with slot positions and the origin planes with the origin index. The positional head uses logits 12·K. The toy network has 4 pre-LayerNorm layers, width 64, one causal head, random Gaussian weights and a ReLU MLP; a 64-token chunk is encoded alone and after a 64-token prefix, and isolated keys are re-indexed by +64 before comparison. The fold simulation uses 24 origins, 3-hop chains and 4,000 trials with two late-bound hops.

| Check | Result |
| --- | --- |
| Re-indexing identity R(Δ)R(m)k = R(m + Δ)k | max abs error 2.2×10⁻¹³ |
| Factorization R(cB + ℓ) = R(cB)R(ℓ) | max abs error 3.6×10⁻¹³ |
| Mean kernel at same-offset tokens in other origins (M1 / M3 / M4) | 0.096 / 0.944 / 0.276 |
| Positional-head mass on ghosts and on own origin, M1 | 0.01%, 91.3% |
| Same, M3 | 10.7%, 11.1% |
| Same, M4 | 0.03%, 74.0% |
| M3 position-only margin at Δc = 2 | 0.035 |
| Toy key cosine, isolated vs in-context, layers 0–3 | 1.00, 0.96, 0.93, 0.92 |
| Rewrite survival (1 − ε)^50 at ε = 0.02 / 0.005 | 0.364 / 0.778 |
| 3-hop late-bound success: top-k / fold / fold + skeleton (γ = 1) | 0 / 0 / 1.00, with 2.0 faults |
| M4 origin-plane deviation from M1 with g(c) = c·B | 0.06 to 0.51 rad |
| KV bytes per 512-token origin, Llama-3-8B shape | 64 MiB |

&#91;image: Figure 1: alignment kernel vs key position under maps M1, M3 and M4; red dots are same-offset tokens in other origins\]

Figure 1. Under the split-plane map M3 every origin shows a ghost peak at the query's offset (mean 0.94); M1 and the hybrid M4 do not. Under the fresh-frame map M5 the ghost peaks would reach exactly 1.0 (Proposition 5).

&#91;image: Figure 2: simulated success vs skeleton coverage gamma for 3-hop late-bound chains, with gamma squared prediction\]

Figure 2. Fold-left with skeleton faults on 3-hop chains with two late-bound hops: success follows γ² (Corollary 14.1 with γ\_g = 0).

### B.2 v2 checks (origin vectors, hyperbolic geometry, routing)

**Setup.** Poincaré ball of curvature −1 with the Möbius addition, distance and exponential map of Section 2.3. The λ sweep places 4 threads of 4 origins in the plane, thread t at angle π/4 + t·π/2, origin j of a thread at hyperbolic radius 1.0 + 0.6j with a small angular jitter; every origin uses a fresh frame; the query sits at offset 100 of the second origin of thread 0; logits are 12·K − λ·d\_H. The precision checks use unit hyperbolic steps (δ = 1) and emulate bfloat16 by round-to-nearest-even on the upper 16 bits of float32. The routing simulation uses b = 8, L = 4 (4,096 origins), 1,500 trials per point for descent and 3,000 for the flat scan; each candidate's score is 1 for the correct branch and 0 otherwise, plus N(0, σ²) noise.

| Check | Result |
| --- | --- |
| Möbius isometry, \|d(T\_a x, T\_a y) − d(x, y)\| | 4.4×10⁻¹⁶ |
| Router × local factorization (Theorem 6), max abs error | 5.6×10⁻¹⁷ |
| Fresh-frame same-offset kernel; own-origin share with no origin signal, C = 16 | 1.0; 0.0625 |
| Margin for 90% own-origin share at C = 4 / 16 / 64 / 256 / 1,024 / 4,096 | 3.30 / 4.91 / 6.34 / 7.74 / 9.13 / 10.51 |
| λ sweep, own origin / own thread / ghost mass at λ = 0, 2, 5 | 0.062 / 0.250 / 0.165; 0.580 / 0.967 / 0.0073; 0.913 / 1.000 / < 0.0001 |
| Distance to same-thread neighbour; to nearest other-thread origin | 0.61; 2.02 |
| Same-shell distance at R = 6 for 1° / 10° / 45° / 90° / 180° | 2.66 / 7.12 / 10.08 / 11.31 / 12.00 |
| Circumference at R = 6, hyperbolic vs flat | 1,267.4 vs 37.7 |
| 4,096 random thread directions, mean / worst \|cos\| at m = 16 / 32 / 128 | 0.203 / 0.957; 0.142 / 0.790; 0.071 / 0.465 |
| First depth where a unit step has > 10% error in global coordinates (float32 / float16 / bfloat16) | 16 / 7 / 4 |
| Re-anchored step error (float32 / float16 / bfloat16), any depth | 2.1×10⁻⁸ / 1.0×10⁻⁴ / 2.0×10⁻³ |
| Offsets stored in low precision, composed in float64, max error through depth 12 | 2.4×10⁻⁸ / 2.2×10⁻⁵ / 1.2×10⁻³ |
| Query-frame precision horizon at 10% error (float32 / float16 / bfloat16) | 18.25 / 9.25 / 7.0 |
| Attention horizon ln(10⁴)/λ at λ = 1 / 2 / 5 | 9.21 / 4.61 / 1.84 |
| Routing success at σ = 0.3: flat / beam 1 / beam 2 / beam 4 | 0.384 / 0.837 / 0.903 / 0.951 |
| Routing scores per query: flat / beam 1 / beam 2 / beam 4 | 4,096 / 32 / 56 / 104 |
| Coverage with skeleton or graph, γ\_s = 0.8, γ\_g = 0.5: predicted vs simulated (200,000 trials) | 0.9 vs 0.8999 |

&#91;image: Figure 3: Poincare disk with shells as levels, threads as directions, origins on the outer shell and a dashed cross-thread graph edge\]

Figure 3. The address space in two dimensions: dashed circles are shells (levels of detail and memory tiers), grey lines are parent–child edges along threads, orange points are origins on the outer shell, and the dashed purple arc is a graph edge between two threads that geometry alone would keep far apart.

&#91;image: Figure 4: attention mass on own thread, own origin and ghost tokens versus penalty slope lambda\]

Figure 4. Fresh frames plus the hyperbolic penalty, 4 threads × 4 origins, positional head: mass on the own thread, the own origin and same-offset ghosts in other threads as λ grows (Section 7.2).

&#91;image: Figure 5: relative error of one unit step versus depth, global coordinates versus re-anchored offsets, in float32, float16 and bfloat16\]

Figure 5. Precision: relative error of one unit step against depth. Global coordinates (solid) fail at depth 4, 7 and 16 for bfloat16, float16 and float32; re-anchored offsets (dotted) keep the same error at every depth (Proposition 10).

&#91;image: Figure 6: probability of finding the right origin among 4096 versus summary noise, flat scan versus shell descent with beams 1, 2 and 4\]

Figure 6. Routing: probability of finding the right origin among 4,096 as summary noise grows, for a flat scan and for shell descent with beams 1, 2 and 4 (Proposition 13).

## References

Every entry below was opened and matched (title, authors, venue) against its arXiv, DOI or publisher record on 6 November 2026. Venues are given only where the record confirms them; otherwise the arXiv preprint is cited.

- Acharya, S., Jia, F., Ginsburg, B. (2025). [Star Attention: Efficient LLM Inference over Long Sequences](https://arxiv.org/abs/2411.17116). ICML 2025.
- Angelotti, G. (2023). [HyPE: Attention with Hyperbolic Biases for Relative Positional Encoding](https://arxiv.org/abs/2310.19676). arXiv:2310.19676.
- Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., Veličković, P. (2024). [Round and Round We Go! What makes Rotary Positional Encodings useful?](https://arxiv.org/abs/2410.06205) arXiv:2410.06205.
- Belady, L. A. (1966). [A study of replacement algorithms for a virtual-storage computer](https://doi.org/10.1147/sj.52.0078). IBM Systems Journal 5(2).
- Bertsch, A., Alon, U., Neubig, G., Gormley, M. (2023). [Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://arxiv.org/abs/2305.01625). NeurIPS 2023.
- Bulatov, A., Kuratov, Y., Burtsev, M. (2022). [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881). NeurIPS 2022.
- Chami, I., Ying, R., Ré, C., Leskovec, J. (2019). [Hyperbolic Graph Convolutional Neural Networks](https://arxiv.org/abs/1910.12933). NeurIPS 2019.
- Chen, H., Pasunuru, R., Weston, J., Celikyilmaz, A. (2023). [Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading](https://arxiv.org/abs/2310.05029). arXiv:2310.05029.
- Chen, S., Wong, S., Chen, L., Tian, Y. (2023). [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595). arXiv:2306.15595.
- Chen, W., Han, X., Lin, Y., Zhao, H., Liu, Z., Li, P., Sun, M., Zhou, J. (2022). [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686). ACL 2022.
- Chevalier, A., Wettig, A., Ajith, A., Chen, D. (2023). [Adapting Language Models to Compress Contexts](https://arxiv.org/abs/2305.14788). EMNLP 2023.
- Chhikara, P. et al. (2025). [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413). arXiv:2504.19413.
- Dai, C., Shan, H., Song, M., Liang, D. (2025). [HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models](https://arxiv.org/abs/2509.05218). arXiv:2509.05218.
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., Salakhutdinov, R. (2019). [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://doi.org/10.18653/v1/P19-1285). ACL 2019.
- Denning, P. J. (1968). [The working set model for program behavior](https://doi.org/10.1145/363095.363141). Communications of the ACM 11(5).
- Denning, P. J. (1970). [Virtual Memory](https://doi.org/10.1145/356571.356573). ACM Computing Surveys 2(3).
- Edge, D. et al. (2024). [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). arXiv:2404.16130.
- Eyuboglu, S. et al. (2025). [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266). arXiv:2506.06266.
- Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112). arXiv:1805.09112.
- Gim, I. et al. (2024). [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934). MLSys 2024.
- Golovneva, O., Wang, T., Weston, J., Sukhbaatar, S. (2024). [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/abs/2405.18719). arXiv:2405.18719.
- Gulcehre, C. et al. (2018). [Hyperbolic Attention Networks](https://arxiv.org/abs/1805.09786). arXiv:1805.09786.
- Gutiérrez, B. J. et al. (2024). [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831). NeurIPS 2024.
- Han, C. et al. (2024). [LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models](https://arxiv.org/abs/2308.16137). NAACL 2024.
- He, N., Anand, R., Madhu, H., Maatouk, A., Krishnaswamy, S., Tassiulas, L., Yang, M., Ying, R. (2025). [HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts](https://arxiv.org/abs/2505.24722). arXiv:2505.24722.
- He, Z. et al. (2024). [Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation](https://arxiv.org/abs/2401.16421). ICML 2024.
- Heo, B., Park, S., Han, D., Yun, S. (2024). [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298). ECCV 2024.
- Hu, J. et al. (2024). [EPIC: Efficient Position-Independent Caching for Serving Large Language Models](https://arxiv.org/abs/2410.15332). arXiv:2410.15332.
- Hutchins, D., Schlag, I., Wu, Y., Dyer, E., Neyshabur, B. (2022). [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852). NeurIPS 2022.
- Jiang, H. et al. (2023b). [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736). EMNLP 2023.
- Jiang, Z. et al. (2023). [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983). EMNLP 2023.
- Jin, C. et al. (2024). [RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation](https://arxiv.org/abs/2404.12457). arXiv:2404.12457.
- Jin, H. et al. (2024). [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325). ICML 2024.
- Kazemnejad, A. et al. (2023). [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466). NeurIPS 2023.
- Kleinberg, R. (2007). [Geographic Routing Using Hyperbolic Space](https://doi.org/10.1109/INFCOM.2007.221). IEEE INFOCOM 2007.
- Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., Boguñá, M. (2010). [Hyperbolic Geometry of Complex Networks](https://doi.org/10.1103/PhysRevE.82.036106). Physical Review E 82, 036106. Open copy: [arXiv:1006.5169](https://arxiv.org/abs/1006.5169).
- Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165). SOSP 2023. Open copy: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- Lee, K.-H. et al. (2024). [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727). arXiv:2402.09727.
- Lee, W., Lee, J., Seo, J., Sim, J. (2024b). [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/abs/2406.19707). OSDI 2024.
- Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.
- Li, Y. et al. (2024). [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469). arXiv:2404.14469.
- Liu, D. et al. (2024). [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516). arXiv:2409.10516.
- Liu, N. F. et al. (2024). [Lost in the Middle: How Language Models Use Long Contexts](https://doi.org/10.1162/tacl_a_00638). TACL 12. Open copy: [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
- Liu, X. et al. (2024). [Scaling Laws of RoPE-based Extrapolation](https://arxiv.org/abs/2310.05209). ICLR 2024.
- Liu, Y. et al. (2025). [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665). arXiv:2510.09665.
- Lu, E. et al. (2025). [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189). arXiv:2502.13189.
- Lu, S. et al. (2024). [TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text](https://arxiv.org/abs/2410.07590). arXiv:2410.07590.
- Ma, D. et al. (2025). [Block-Attention for Efficient Prefilling](https://arxiv.org/abs/2409.15355). ICLR 2025.
- Malkov, Y. A., Yashunin, D. A. (2020). [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473). IEEE TPAMI. Open copy: [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).
- Men, X. et al. (2024). [Base of RoPE Bounds Context Length](https://arxiv.org/abs/2405.14591). arXiv:2405.14591.
- Mishne, G., Wan, Z., Wang, Y., Yang, S. (2022). [The Numerical Stability of Hyperbolic Representation Learning](https://arxiv.org/abs/2211.00181). arXiv:2211.00181.
- Mohtashami, A., Jaggi, M. (2023). [Landmark Attention: Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300). NeurIPS 2023.
- Mu, J., Li, X. L., Goodman, N. (2023). [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467). NeurIPS 2023.
- Munkhdalai, T., Faruqui, M., Gopal, S. (2024). [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143). arXiv:2404.07143.
- Nickel, M., Kiela, D. (2017). [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039). arXiv:1705.08039.
- Nickel, M., Kiela, D. (2018). [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/abs/1806.03417). ICML 2018.
- Packer, C. et al. (2023). [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560). arXiv:2310.08560.
- Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023). [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071). arXiv:2309.00071.
- Press, O., Smith, N. A., Lewis, M. (2022). [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409). ICLR 2022.
- Rae, J. W. et al. (2020). [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507). ICLR 2020.
- Ratner, N. et al. (2023). [Parallel Context Windows for Large Language Models](https://doi.org/10.18653/v1/2023.acl-long.352). ACL 2023.
- Sala, F., De Sa, C., Gu, A., Ré, C. (2018). [Representation Tradeoffs for Hyperbolic Embeddings](https://proceedings.mlr.press/v80/sala18a.html). ICML 2018, PMLR 80. Open copy: [arXiv:1804.03329](https://arxiv.org/abs/1804.03329).
- Sarkar, R. (2012). [Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane](https://doi.org/10.1007/978-3-642-25878-7_34). Graph Drawing 2011, Lecture Notes in Computer Science.
- Sarthi, P. et al. (2024). [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059). arXiv:2401.18059.
- Shimizu, R., Mukuta, Y., Harada, T. (2021). [Hyperbolic Neural Networks++](https://arxiv.org/abs/2006.08210). ICLR 2021.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y. (2024). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://doi.org/10.1016/j.neucom.2023.127063). Neurocomputing 568. Open copy: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- Sun, H. et al. (2024). [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://arxiv.org/abs/2410.21465). arXiv:2410.21465.
- Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., Han, S. (2024). [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774). ICML 2024.
- Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. (2023). [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://doi.org/10.18653/v1/2023.acl-long.557). ACL 2023.
- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Wang, H., Wang, P., Li, M., Liu, S., Miao, S., Wang, Z., Li, P. (2025). [Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models](https://arxiv.org/abs/2506.07334). arXiv:2506.07334.
- Wang, P. et al. (2024). [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191). arXiv:2409.12191.
- Wu, Y., Rabe, M. N., Hutchins, D., Szegedy, C. (2022). [Memorizing Transformers](https://arxiv.org/abs/2203.08913). ICLR 2022.
- Xiao, C. et al. (2024). [InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory](https://arxiv.org/abs/2402.04617). arXiv:2402.04617.
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). ICLR 2024.
- Xiao, G. et al. (2025). [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819). ICLR 2025.
- Xu, W. et al. (2025). [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110). NeurIPS 2025.
- Yang, J. et al. (2025). [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](https://arxiv.org/abs/2502.16002). arXiv:2502.16002.
- Yang, M., Verma, H., Zhang, D. C., Liu, J., King, I., Ying, R. (2024). [Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290). KDD 2024.
- Yang, X., Chen, T., Chen, B. (2025). [APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](https://arxiv.org/abs/2502.05431). ICLR 2025.
- Yao, J. et al. (2025). [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://doi.org/10.1145/3689031.3696098). EuroSys 2025. Open copy: [arXiv:2405.16444](https://arxiv.org/abs/2405.16444).
- Yen, H., Gao, T., Chen, D. (2024). [Long-Context Language Modeling with Parallel Context Encoding](https://doi.org/10.18653/v1/2024.acl-long.142). ACL 2024.
- Yu, T., De Sa, C. (2019). [Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models](https://proceedings.neurips.cc/paper/2019/hash/82c2559140b95ccda9c6ca4a8b981f1e-Abstract.html). NeurIPS 2019.
- Yuan, J. et al. (2025). [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089). arXiv:2502.11089.
- Zhang, K., Li, G., Zhang, H., Jin, Z. (2024). [HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position](https://arxiv.org/abs/2403.19115). ACL 2024.
- Zhang, Y. et al. (2024). [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818). arXiv:2406.02818.
- Zhang, Z. et al. (2023). [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048). NeurIPS 2023.
- Zheng, L. et al. (2024). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). arXiv:2312.07104.
