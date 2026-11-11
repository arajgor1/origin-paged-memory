"""Ingest for the Track A harness (protocol 3.1 items 1-5, paper Algorithm 1 at the agent level).

A Session holds one item's archive: the SQLite origin store (protocol 3.1 item 2), skeletons (S-lex, S-llm, S-hyb,
oracle), dense embeddings, the graph page table (seq, tree, ent, ref, x edges, k_max per type), thread assignment, the
address-book tree (opa.geometry.TreeLayout) and the shell index used by descent. Nothing here sees the question.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict

import numpy as np

from ... import geometry
from .config import D_CLAMP, TAU_THREAD
from .embed import HFEmbedder, PrecomputedEmbedder  # noqa: F401  (re-exported)
from .text import BM25, alias_keys, lexical_entities, mentions, normalize_key, tokenize

EDGE_TYPES = ("seq", "tree", "ent", "ref", "x")


# ---------------------------------------------------------------- embeddings


class HashEmbedder:
    """Deterministic hashed bag-of-words embedding (unit norm). Used in tests and as a no-model fallback."""

    name = "hash-bow-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 0, n, 1.0)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)




# ---------------------------------------------------------------- session


class Session:
    def __init__(self, item: dict, cfg: dict, embedder, skeleton_llm: dict | None = None):
        """cfg keys: skeleton (none|lex|llm|hyb|oracle), graph (bool), oracle_graph (bool), k_max, tau_thread, tau_x,
        b (shell fan-out), m (ball dimension), delta, use_gold_threads (bool, default False)."""
        self.item, self.cfg = item, cfg
        self.origins = item["origins"]
        self.N = len(self.origins)
        self.texts = [o["text"] for o in self.origins]
        self.k_max = int(cfg.get("k_max", 32))
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self._create_tables()
        self.emb = embedder.embed(self.texts)
        self.bm25 = BM25(self.texts)
        self._build_skeleton(cfg.get("skeleton", "hyb"), skeleton_llm or {})
        self._assign_threads()
        self.edges: dict[str, dict[int, list[tuple[int, float]]]] = {t: defaultdict(list) for t in EDGE_TYPES}
        if cfg.get("graph", False):
            self._build_graph(oracle=cfg.get("oracle_graph", False))
        self._build_shells(int(cfg.get("b", 8)))
        self._store_rows()

    # ---- SQLite store (protocol 3.1 item 2)
    def _create_tables(self):
        c = self.db.cursor()
        c.execute("CREATE TABLE origins(origin_id INTEGER PRIMARY KEY, doc_title TEXT, start_tok INTEGER, end_tok INTEGER,"
                  " text TEXT, thread_id INTEGER, parent_id INTEGER)")
        c.execute("CREATE TABLE addresses(node_id INTEGER PRIMARY KEY, parent_id INTEGER, offset_vector BLOB, depth INTEGER)")
        c.execute("CREATE TABLE edges(src INTEGER, dst INTEGER, type TEXT, weight REAL, created_at INTEGER)")
        c.execute("CREATE TABLE skeleton(entity_key TEXT, origin_id INTEGER, surface_form TEXT)")
        c.execute("CREATE INDEX sk ON skeleton(entity_key)")
        self.db.commit()

    def _store_rows(self):
        c = self.db.cursor()
        tok = 0
        for o in self.origins:
            n = int(o.get("n_tokens", 0))
            par = self.parent_origin.get(o["origin_id"])
            c.execute("INSERT INTO origins VALUES (?,?,?,?,?,?,?)", (o["origin_id"], o.get("doc_title", ""), tok, tok + n,
                                                                      o["text"], self.thread_of[o["origin_id"]], par))
            tok += n
        for node, par in self.layout.book.parent.items():
            c.execute("INSERT INTO addresses VALUES (?,?,?,?)", (node, par, self.layout.book.offset[node].astype(np.float32).tobytes(),
                                                                  self.layout.book.depth[node]))
        for t, adj in self.edges.items():
            for s, lst in adj.items():
                for d, w in lst:
                    c.execute("INSERT INTO edges VALUES (?,?,?,?,?)", (s, d, t, w, max(s, d)))
        for o, keys in self.skeleton.items():
            for k in keys:
                c.execute("INSERT INTO skeleton VALUES (?,?,?)", (k, o, self.surface.get((o, k), k)))
        self.db.commit()

    # ---- skeletons (protocol 3.1 item 3)
    def _build_skeleton(self, kind: str, llm_sk: dict):
        self.skeleton_kind = kind
        self.skeleton: dict[int, set[str]] = {}
        self.surface: dict[tuple[int, str], str] = {}
        for o in range(self.N):
            keys: set[str] = set()
            if kind in ("lex", "hyb", "oracle"):
                keys |= lexical_entities(self.texts[o])
            if kind in ("llm", "hyb", "oracle"):
                keys |= {normalize_key(e) for e in llm_sk.get(str(o), llm_sk.get(o, []))}
            if kind == "oracle":
                keys |= self._oracle_keys(o)
            if kind != "none":
                for k in list(keys):
                    keys |= alias_keys(k)
            self.skeleton[o] = {k for k in keys if k}
        self.entity_index: dict[str, list[int]] = defaultdict(list)
        for o in range(self.N):
            for k in self.skeleton[o]:
                self.entity_index[k].append(o)

    def _oracle_keys(self, o: int) -> set[str]:
        g = self.item.get("gold", {})
        out = set()
        for f in g.get("facts", []):
            if f["origin_id"] == o:
                for e in (f.get("subject"), f.get("reveals")):
                    if e:
                        out.add(normalize_key(e))
                        out |= {normalize_key(a) for a in g.get("aliases", {}).get(e, [])}
        return out

    def lookup(self, entity: str, before: int | None = None, exclude=()) -> list[int]:
        """Skeleton lookup (tool `lookup`): origins whose skeleton holds the entity key or one of its alias keys."""
        if self.skeleton_kind == "none":
            return []
        keys = {normalize_key(entity)} | alias_keys(normalize_key(entity))
        q = "SELECT DISTINCT origin_id FROM skeleton WHERE entity_key IN (%s)" % ",".join("?" * len(keys))
        rows = [r[0] for r in self.db.execute(q, tuple(keys))]
        if before is not None:
            rows = [r for r in rows if r < before]
        return sorted(r for r in rows if r not in exclude)

    # ---- threads and the address book (protocol 3.1 item 5, paper 7.5)
    def _assign_threads(self):
        tau = float(self.cfg.get("tau_thread", TAU_THREAD))      # TODO(P0): recalibrate on harrier
        m, delta = int(self.cfg.get("m", 32)), float(self.cfg.get("delta", 1.0))
        self.layout = geometry.TreeLayout(m=m, delta=delta, d_clamp=self.cfg.get("d_clamp", D_CLAMP))
        dirs = geometry.spread_directions(max(64, 2 * m), m, seed=0)
        self.thread_of: dict[int, int] = {}
        self.parent_origin: dict[int, int | None] = {}
        heads: dict[int, int] = {}                       # thread -> latest origin
        gold_threads = self.cfg.get("use_gold_threads", False)
        node = lambda o: o + 1                           # address-book node id (0 is the root)
        for o in range(self.N):
            if gold_threads:
                t = int(self.origins[o].get("thread_id", 0))
            elif not heads:
                t = 0
            else:
                sims = {t_: float(self.emb[o] @ self.emb[h]) for t_, h in heads.items()}
                t_best = max(sims, key=sims.get)
                t = t_best if sims[t_best] >= tau else len(heads)
            par = heads.get(t)
            self.thread_of[o] = t
            self.parent_origin[o] = par
            self.layout.add_node(node(o), node(par) if par is not None else 0, dirs[t % len(dirs)])
            heads[t] = o
        self.thread_heads = heads
        self.n_threads = len(heads)

    def distance(self, q: int, c: int, arm: str, fitted: np.ndarray | None = None) -> float:
        """Structural distance between origins q and c under an H5 arm (A0..A4)."""
        if arm == "A0":
            return 0.0
        if arm == "A4":
            if fitted is None:
                raise ValueError("A4 needs fitted addresses")
            return geometry.dist(fitted[q + 1], fitted[c + 1]) if q != c else 0.0
        return self.layout.distance(q + 1, c + 1, arm)

    def geometry_counts(self) -> dict:
        """Cumulative for this session: unique origin pairs answered by the arbitrary-precision path (a2_exact_pairs),
        the number of such distance calls (a2_exact_calls, repeats included), and unique pairs that fell to the
        last-resort cap (a2_capped_pairs)."""
        b = self.layout.book
        return {"a2_exact_pairs": len(b.exact_pairs), "a2_exact_calls": b.n_exact_fallback,
                "a2_capped_pairs": len(b.capped_pairs)}

    def fit_addresses(self, dim: int = 10, epochs: int = 100, seed: int = 0) -> np.ndarray:
        """A4: Poincare embeddings fitted to the session's tree plus graph edges (nodes: root 0, origin o -> o+1)."""
        edges = [(self.layout.parent(n), n) for n in range(1, self.N + 1)]
        for t in ("ent", "ref", "x"):
            for s, lst in self.edges[t].items():
                edges += [(s + 1, d + 1) for d, _ in lst if d > s]
        return geometry.fit_poincare(self.N + 1, edges, dim=dim, epochs=epochs, seed=seed)

    # ---- graph page table (protocol 3.1 item 4, paper 9.2)
    def _link(self, t: str, a: int, b: int, w: float):
        for s, d in ((a, b), (b, a)):
            self.edges[t][s].append((d, w))

    def _build_graph(self, oracle: bool = False):
        for o, par in self.parent_origin.items():
            if par is not None:
                self._link("seq", o, par, 1.0)
        # ent: origins sharing a skeleton key, weighted by the key's rarity
        n = max(self.N, 1)
        for k, os_ in self.entity_index.items():
            if len(os_) < 2 or len(os_) > max(8, n // 8):          # very common keys carry no information
                continue
            w = float(np.log(n / len(os_)))
            for i in range(len(os_)):
                for j in range(i + 1, len(os_)):
                    self._link("ent", os_[i], os_[j], w)
        # ref: explicit structure (paths, identifiers and reply/tool ids shared between origins)
        refs: dict[str, list[int]] = defaultdict(list)
        import re
        for o, t in enumerate(self.texts):
            for m in re.finditer(r"(?:\b[\w.\-]+/)+[\w.\-]+\.\w+|\b(?:reply-to|re|tool-call|call|ref)[:#]\s?[\w\-]+", t, re.I):
                refs[m.group(0).lower()].append(o)
        for k, os_ in refs.items():
            for i in range(len(os_)):
                for j in range(i + 1, len(os_)):
                    self._link("ref", os_[i], os_[j], 1.0)
        # x: cross-thread origins whose dense cosine exceeds tau_x
        tau_x = float(self.cfg.get("tau_x", 0.8))
        sims = self.emb @ self.emb.T
        th = np.array([self.thread_of[o] for o in range(self.N)])
        for o in range(self.N):
            cand = np.where((th != th[o]) & (sims[o] >= tau_x))[0]
            for c in cand[np.argsort(-sims[o, cand])][: self.k_max]:
                if c > o:
                    self._link("x", o, int(c), float(sims[o, c]))
        if oracle:                                           # S18: link every pair of gold fact origins
            fo = sorted(set(self.item.get("gold", {}).get("fact_origins", [])))
            for i in range(len(fo)):
                for j in range(i + 1, len(fo)):
                    self._link("ent", fo[i], fo[j], 10.0)
        for t in ("seq", "ent", "ref", "x"):                  # cap k_max per type, ties broken towards recency
            for s in list(self.edges[t]):
                best = {}
                for d, w in self.edges[t][s]:
                    best[d] = max(w, best.get(d, -1e9))
                self.edges[t][s] = sorted(best.items(), key=lambda dw: (-dw[1], -dw[0]))[: self.k_max]

    def neighbors(self, o: int, kinds=("ent", "ref", "x"), mentioning: str | None = None) -> list[int]:
        out = []
        for t in kinds:
            for d, _ in self.edges[t].get(o, []):
                if mentioning is None or mentions(self.texts[d], normalize_key(mentioning)) or \
                        normalize_key(mentioning) in self.skeleton.get(d, ()):
                    out.append(d)
        return list(dict.fromkeys(out))

    def edge_types(self) -> dict[str, int]:
        return {t: sum(len(v) for v in adj.values()) // (1 if t == "tree" else 2) for t, adj in self.edges.items()}

    # ---- shell index (paper 8.1-8.4): inner nodes over each thread's origins, b children each
    def _build_shells(self, b: int):
        self.children: dict[str, list[str]] = {}
        self.node_emb: dict[str, np.ndarray] = {}
        self.node_skel: dict[str, set[str]] = {}
        self.node_origins: dict[str, list[int]] = {}
        by_thread: dict[int, list[int]] = defaultdict(list)
        for o in range(self.N):
            by_thread[self.thread_of[o]].append(o)
        tops = []
        for t, os_ in sorted(by_thread.items()):
            level = [f"o{o}" for o in os_]
            for o in os_:
                self.node_emb[f"o{o}"] = self.emb[o]
                self.node_skel[f"o{o}"] = self.skeleton[o]
                self.node_origins[f"o{o}"] = [o]
            depth = 0
            while len(level) > 1:
                depth += 1
                nxt = []
                for i in range(0, len(level), b):
                    nid = f"t{t}d{depth}n{i // b}"
                    kids = level[i:i + b]
                    self._make_inner(nid, kids)
                    nxt.append(nid)
                level = nxt
            tops.append(level[0])
        level, depth = tops, 0
        while len(level) > b:
            depth += 1
            nxt = []
            for i in range(0, len(level), b):
                nid = f"rootd{depth}n{i // b}"
                self._make_inner(nid, level[i:i + b])
                nxt.append(nid)
            level = nxt
        self._make_inner("root", level)
        self.inner_bm25_docs = None
        for nid, kids in self.children.items():
            for k in kids:
                self.edges["tree"].setdefault(nid, []).append((k, 1.0))    # E_tree: inner node -> child

    def _make_inner(self, nid: str, kids: list[str]):
        self.children[nid] = kids
        e = np.mean([self.node_emb[k] for k in kids], axis=0)
        self.node_emb[nid] = e / (np.linalg.norm(e) or 1.0)
        self.node_skel[nid] = set().union(*[self.node_skel[k] for k in kids])
        self.node_origins[nid] = sum((self.node_origins[k] for k in kids), [])

    def route_score(self, query: str, q_emb: np.ndarray, node: str) -> float:
        """Routing key score: dense cosine with the node's mean child embedding plus entity overlap with the merged
        skeleton (a BM25-style term: matched keys weighted by inverse document frequency)."""
        cos = float(q_emb @ self.node_emb[node])
        qk = lexical_entities(query) | {normalize_key(w) for w in tokenize(query) if len(w) > 3}
        hits = qk & self.node_skel[node]
        ent = sum(np.log(1 + self.N / (1 + len(self.entity_index.get(k, ())))) for k in hits)
        return cos + 0.25 * ent

    def descend(self, query: str, q_emb: np.ndarray, beam: int = 2, top: int | None = None) -> tuple[list[int], int]:
        """Coarse-to-fine beam descent (paper 8.3). Returns (candidate origin ids, number of routing scores)."""
        children = {k: v for k, v in self.children.items()}
        leaves, n_scores = geometry.descend(children, lambda n: self.route_score(query, q_emb, n), root="root", beam=beam)
        cands = [int(n[1:]) for n in leaves if n.startswith("o")]
        for n in leaves:
            if not n.startswith("o"):
                cands += self.node_origins[n]
        cands = list(dict.fromkeys(cands))
        return (cands[:top] if top else cands), n_scores

    def flat_scan(self, query: str, q_emb: np.ndarray, top: int) -> tuple[list[int], int]:
        s = [self.route_score(query, q_emb, f"o{o}") for o in range(self.N)]
        return [int(i) for i in np.argsort(-np.array(s), kind="stable")[:top]], self.N

    def hnsw_search(self, query: str, q_emb: np.ndarray, top: int, k: int = 8, ef: int = 16) -> tuple[list[int], int]:
        """S17: greedy best-first search on a flat k-NN graph over origin embeddings (HNSW's bottom layer, no shells).
        Entry point: the most recent origin. Returns (top candidates, number of scores computed)."""
        if not hasattr(self, "_knn"):
            sims = self.emb @ self.emb.T
            np.fill_diagonal(sims, -np.inf)
            self._knn = np.argsort(-sims, axis=1)[:, :k]
        score = lambda o: self.route_score(query, q_emb, f"o{o}")
        entry = self.N - 1
        seen = {entry: score(entry)}
        frontier = [entry]
        while frontier:
            frontier.sort(key=lambda o: -seen[o])
            cur = frontier.pop(0)
            worst = sorted(seen.values(), reverse=True)[: ef][-1]
            if seen[cur] < worst and len(seen) >= ef:
                break
            for nb in self._knn[cur]:
                nb = int(nb)
                if nb not in seen:
                    seen[nb] = score(nb)
                    frontier.append(nb)
        ranked = sorted(seen, key=lambda o: -seen[o])
        return ranked[:top], len(seen)

    # ---- relevance used by every re-ranking arm
    def relevance(self, query: str, o: int) -> float:
        return self.bm25.score(query, o)
