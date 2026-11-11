"""Monitors and task metrics for the Track B mini-LoRA checks (protocol 5.3 "Measure per layer and head", 5.4 "Monitors").

AttentionProbe  read-only hook (opa.trackb.attention.PROBE) called inside the "opa_oaa" attention function with the final
                float mask (causal + OAA bias). For every query token in window origin >= 1 (origin 0 sees only itself) and
                every layer and head it accumulates, from the softmax over the keys,
                  own_origin_mass      attention mass on keys of the query's own origin
                  uniform_own          the same mass under uniform causal attention (own visible keys / all visible keys)
                  own_thread_mass      mass on keys of origins of the query's thread (own origin included)
                  ghost_mass           mass on the same-offset tokens of OTHER origins (the aliasing target of fresh frames)
                  ghost_uniform        the uniform value of ghost_mass
                  router_entropy       entropy of the per-origin mass distribution (nats; ln C is the maximum)
                Summary (mean over layers and heads of the per-head ratio-of-sums):
                  own_origin_mass_fraction       mean own_origin_mass                        (H4: V3 >= 0.90)
                  own_origin_mass_ratio_to_uniform  own_origin_mass / uniform_own            (H4: V0 <= 1.5)
                  ghost_ratio = ghost_mass / ghost_uniform;  own_thread_mass_fraction;  router_entropy
                The probe queries are all tokens of origins 1..C-1 of a window of C = 8 origins of the full archive
                geometry (random thread assignment, random order); the prereg draft aggregates over all layers and heads.
Task metrics    attrib (task 3): greedy EM of the one label token, and rank accuracy among the window's option labels (both
                from the same logits); passkey (task 1): greedy EM of the 5-digit code, computed teacher-forced (identical to
                greedy decoding: the decoded string equals the gold string iff each step's argmax equals the gold token given
                the gold prefix); lm (task 5): perplexity on the 48 continuation tokens; std_ppl: perplexity of the same kind
                of continuation after a single origin with standard contiguous positions and OAA off (damage monitor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from opa.trackb import attention as att
from opa.trackb import data as D
from opa.trackb.arms import Arm, forward_example


class AttentionProbe:
    def __init__(self, ex: D.Example, archive: D.Archive, chunk: int = 512):
        T, C = ex.T, len(ex.window)
        self.T, self.C, self.chunk = T, C, chunk
        win = torch.full((T,), -1, dtype=torch.long)
        off = torch.zeros(T, dtype=torch.long)
        start = []
        i, k = 0, 0                                   # k = ordinal among the anchored segments (an unanchored prefix / tail does not count)
        for s in ex.segments:
            if s.origin == att.NULL_ORIGIN:
                i += s.length
                continue
            win[i:i + s.length] = k
            off[i:i + s.length] = torch.arange(s.length)
            start.append(i)
            i += s.length
            k += 1
        self.win, self.off, self.start = win, off, torch.tensor(start)
        thr = torch.full((T,), -1, dtype=torch.long)
        for k, c in enumerate(ex.window):
            thr[win == k] = archive.thread(c)
        self.thr = thr
        self.onehot = torch.zeros(T, C)
        m = win >= 0
        self.onehot[m.nonzero().squeeze(-1), win[m]] = 1.0
        self.q_first = int(self.start[1]) if C > 1 else T
        self.acc: dict[int, dict] = {}
        self.n_q = int((win[self.q_first:] >= 0).sum())

    def __call__(self, module, query, k_rep, mask, scaling):
        li = module.layer_idx
        b, H, Tq, _ = query.shape
        dev = query.device
        if Tq != self.T:
            raise ValueError("AttentionProbe expects a full-window forward pass (no KV cache)")
        win, off, thr = self.win.to(dev), self.off.to(dev), self.thr.to(dev)
        onehot, start = self.onehot.to(dev), self.start.to(dev)
        idx_all = torch.arange(self.T, device=dev)
        acc = self.acc.setdefault(li, {k: torch.zeros(H, dtype=torch.float64) for k in
                                       ("own", "uni", "thr", "thr_uni", "ghost", "ghost_uni", "ent", "prefix", "own_ren")})
        kT = k_rep[0].float().transpose(-1, -2)                                   # [H, d, Tk]
        for q0 in range(self.q_first, self.T, self.chunk):
            q1 = min(q0 + self.chunk, self.T)
            qi = idx_all[q0:q1]
            keep = win[qi] >= 0
            if not bool(keep.any()):
                continue
            s = torch.matmul(query[0, :, q0:q1].float(), kT) * (scaling if scaling is not None else query.shape[-1] ** -0.5)
            s = s + mask[0, :, q0:q1].float()
            p = torch.softmax(s, dim=-1) * keep[None, :, None].float()             # [H, qc, Tk]; non-window queries zeroed
            m_org = torch.matmul(p, onehot)                                        # [H, qc, C]
            qw = win[qi].clamp(min=0)
            own = m_org.gather(-1, qw[None, :, None].expand(H, -1, 1)).squeeze(-1)  # [H, qc]
            # the keys a query can actually see (causal AND the variant's mask, e.g. B4's graph mask): the uniform baselines use these
            vis = mask[0, 0, q0:q1] > torch.finfo(mask.dtype).min / 2              # [qc, Tk]
            n_vis = vis.sum(-1).clamp_min(1).float()
            kw_ = win[None, :]
            own_vis = (vis & (kw_ == win[qi][:, None])).sum(-1).float()
            uni = (own_vis / n_vis) * keep
            row = m_org.sum(-1)                                                    # mass on origin keys; the rest is on prefix / tail keys
            ent = -((m_org / row.clamp_min(1e-30)[..., None]) * torch.log((m_org / row.clamp_min(1e-30)[..., None]).clamp_min(1e-30))).sum(-1)
            ghost_m = (off[None, :] == off[qi][:, None]) & (kw_ != win[qi][:, None]) & (kw_ >= 0) & vis
            thr_m = (thr[None, :] == thr[qi][:, None]) & (kw_ >= 0) & vis
            kf = keep.float()
            acc["prefix"] += ((1.0 - row) * kf).sum(-1).double().cpu()
            acc["own_ren"] += ((own / row.clamp_min(1e-30)) * kf).sum(-1).double().cpu()
            acc["own"] += (own * kf).sum(-1).double().cpu()
            acc["uni"] += uni.sum().double().cpu().expand(H)
            acc["ent"] += (ent * kf).sum(-1).double().cpu()
            acc["ghost"] += (p * ghost_m[None].float()).sum(-1).sum(-1).double().cpu()
            acc["ghost_uni"] += ((ghost_m.sum(-1).float() / n_vis) * kf).sum().double().cpu().expand(H)
            acc["thr"] += (p * thr_m[None].float()).sum(-1).sum(-1).double().cpu()
            acc["thr_uni"] += ((thr_m.sum(-1).float() / n_vis) * kf).sum().double().cpu().expand(H)

    def layer_head(self) -> dict:
        """Per-layer, per-head arrays [L][H] of own-origin mass, own-thread mass, ghost ratio and router entropy (protocol 5.3)."""
        n = max(self.n_q, 1)
        layers = sorted(self.acc)

        def arr(k):
            return torch.stack([self.acc[l][k] for l in layers]) / n

        own, thr, ghost, ghost_u, ent = arr("own"), arr("thr"), arr("ghost"), arr("ghost_uni"), arr("ent")
        return {"layers": layers, "own_origin_mass": own.tolist(), "own_thread_mass": thr.tolist(),
                "ghost_ratio": (ghost / ghost_u.clamp_min(1e-12)).tolist(), "router_entropy": ent.tolist()}

    def summary(self) -> dict:
        n = max(self.n_q, 1)
        layers = sorted(self.acc)
        per = {k: torch.stack([self.acc[l][k] for l in layers]) / n for k in ("own", "uni", "thr", "thr_uni", "ghost",
                                                                              "ghost_uni", "ent", "prefix", "own_ren")}
        frac = float(per["own"].mean())
        return {"own_origin_mass_fraction": frac,
                "own_origin_mass_fraction_excl_prefix": float(per["own_ren"].mean()),
                "prefix_mass_fraction": float(per["prefix"].mean()),
                "own_origin_mass_ratio_to_uniform": frac / max(float(per["uni"].mean()), 1e-12),
                "own_thread_mass_fraction": float(per["thr"].mean()),
                "own_thread_mass_ratio_to_uniform": float(per["thr"].mean()) / max(float(per["thr_uni"].mean()), 1e-12),
                "ghost_ratio": float(per["ghost"].mean()) / max(float(per["ghost_uni"].mean()), 1e-12),
                "router_entropy": float(per["ent"].mean()), "router_entropy_max": math.log(self.C),
                "own_origin_mass_fraction_by_layer": [float(x) for x in per["own"].mean(1)],
                "n_queries": self.n_q, "layers": len(layers), "heads": int(per["own"].shape[1]), "C": self.C}


@torch.no_grad()
def probe_window(model, ex: D.Example, arm: Arm, archive: D.Archive, tables) -> dict:
    probe = AttentionProbe(ex, archive)
    att.PROBE = probe
    try:
        forward_example(model, ex, arm, tables, probe_kernel=True)
    finally:
        att.PROBE = None
    return probe.summary()


def mean_probe(summaries: list[dict]) -> dict:
    keys = [k for k, v in summaries[0].items() if isinstance(v, float)]
    out = {k: float(np.mean([s[k] for s in summaries])) for k in keys}
    out["own_origin_mass_fraction_by_layer"] = [float(x) for x in np.mean(
        [s["own_origin_mass_fraction_by_layer"] for s in summaries], axis=0)]
    out.update({"n_windows": len(summaries), "n_queries": int(sum(s["n_queries"] for s in summaries)),
                "C": summaries[0]["C"], "layers": summaries[0]["layers"], "heads": summaries[0]["heads"]})
    return out


# ============================================================================ task metrics


@dataclass
class EvalSizes:
    attrib: int = 32
    passkey: int = 16
    lm: int = 16
    std: int = 16
    probe: int = 3
    C: int = 6                      # origins per task window (train range 4-6; the probe uses probe_C)
    probe_C: int = 8
    seed0: int = D.DEV_EVAL_SEED0     # first evaluation seed (pilot 900, monitoring set 2000, G2 set 3000)
    role: str = "eval"                # book pool of the evaluation windows: "eval" (dev) or "test" (PG-19 test books, prereg tag required)


@torch.no_grad()
def eval_tasks(model, arm: Arm, archive: D.Archive, sampler: D.Sampler, sizes: EvalSizes, tables=None, probe: bool = True) -> dict:
    """All monitors for one arm at one checkpoint. Items use the dev evaluation seeds and the eval book pool, so every
    arm and every checkpoint sees the same items (paired comparison)."""
    tables = tables if tables is not None else archive.tables
    model.eval()
    res: dict = {"per_item": {}, "per_item_meta": {}}
    em, rank = [], []
    meta_attrib, meta_pk, meta_lm = [], [], []
    for i in range(sizes.attrib):
        ex = sampler.make("attrib", sizes.seed0 + i, sizes.C, role=sizes.role)
        _, logits = forward_example(model, ex, arm, tables)
        gold = int(ex.targets[0])
        em.append(int(int(logits[0].argmax()) == gold))
        opt = torch.tensor(ex.options, device=logits.device)
        rank.append(int(int(opt[int(logits[0, opt].argmax())]) == gold))
        meta_attrib.append({"seed": sizes.seed0 + i, "window": ex.window, "target_origin": ex.meta["target_origin"],
                            "options_origin_ids": ex.meta["options_origin_ids"], "gold_option": ex.gold_option, "T": ex.T})
    if sizes.attrib:
        res["attrib_em"], res["attrib_rank_acc"] = float(np.mean(em)), float(np.mean(rank))
        res["attrib_chance"] = 1.0 / sizes.C
        res["per_item"]["attrib_em"], res["per_item"]["attrib_rank"] = em, rank
        res["per_item_meta"]["attrib"] = meta_attrib
    pk = []
    for i in range(sizes.passkey):
        ex = sampler.make("passkey", sizes.seed0 + i, sizes.C, role=sizes.role)
        _, logits = forward_example(model, ex, arm, tables)
        pk.append(int(bool((logits.argmax(-1).cpu() == ex.targets).all())))
        meta_pk.append({"seed": sizes.seed0 + i, "window": ex.window, "target_origin": ex.meta["target_origin"], "T": ex.T})
    if sizes.passkey:
        res["passkey_em"] = float(np.mean(pk))
        res["per_item"]["passkey_em"] = pk
        res["per_item_meta"]["passkey"] = meta_pk
    for key, C, role_kind in (("lm", sizes.C, "lm"), ("std", 1, "lm")):
        n = getattr(sizes, key)
        tot, cnt = 0.0, 0
        metas = []
        for i in range(n):
            ex = sampler.make("lm", sizes.seed0 + i, C, role=sizes.role)
            if key == "std":
                loss, _ = forward_example(model, ex, arm, tables, oaa_off=True, mapping="contig")
            else:
                loss, _ = forward_example(model, ex, arm, tables)
            tot += float(loss) * ex.ans_len
            cnt += ex.ans_len
            metas.append({"seed": sizes.seed0 + i, "window": ex.window, "loss": float(loss), "n_tokens": ex.ans_len})
        if n:
            res[f"{key}_ppl"] = math.exp(tot / cnt)
            res["per_item_meta"][key] = metas
    if probe and sizes.probe:
        summ = []
        for i in range(sizes.probe):
            ex = sampler.make("probe", sizes.seed0 + i, sizes.probe_C, role=sizes.role)
            summ.append(probe_window(model, ex, arm, archive, tables))
            res["per_item_meta"].setdefault("probe", []).append({"seed": sizes.seed0 + i, "window": ex.window, "T": ex.T})
        res["probe"] = mean_probe(summ)
    return res


def learn_state(model) -> dict:
    """Cheap training monitors of the OAA parameters: largest |A|, g per layer, largest address norm."""
    out = {}
    ps = list(att.oaa_parameters(model))
    if ps:
        out["max_abs_A"] = float(max(p.A.detach().abs().max() for _, p in ps))
        out["g_mean"] = float(np.mean([float(p.g.detach()) for _, p in ps]))
        out["g_max_abs"] = float(max(abs(float(p.g.detach())) for _, p in ps))
        out["lam_mean"] = float(np.mean([float(p.lam.mean()) for _, p in ps]))
    return out


# ============================================================================ linear probe of the source origin id from head outputs


class HeadOutputCollector:
    """Read-only attention-output hook (attention.PROBE_OUT): keeps the output of every head, for the chosen layers and token indices,
    on the CPU. One collector per window; the labels are the global origin ids of those tokens."""

    def __init__(self, layers, token_idx):
        self.layers, self.idx = set(layers), torch.as_tensor(token_idx, dtype=torch.long)
        self.buf: dict = {}

    def __call__(self, module, out):
        if module.layer_idx in self.layers:
            self.buf[module.layer_idx] = out[0][:, self.idx.to(out.device), :].float().cpu()      # [H, n, d]


def probe_token_indices(ex, stride: int = 16) -> tuple:
    """Every stride-th token of the window origins (not of prefix / tail) and its origin id."""
    idx = [i for i in range(0, ex.T, stride) if int(ex.origin_tok[i]) != att.NULL_ORIGIN]
    return idx, [int(ex.origin_tok[i]) for i in idx]


def linear_probe_accuracy(train: dict, y_train, test: dict, y_test, n_classes: int, ridge: float = 1.0) -> dict:
    """Multinomial linear probe by ridge regression on one-hot labels, one per (layer, head). train/test: {layer: [H, n, d]}.
    Returns {"accuracy": {layer: [H]}, "chance": 1 / n_classes, "n_train": ..., "n_test": ..., "n_classes_seen": ...}."""
    yt, ye = torch.as_tensor(y_train), torch.as_tensor(y_test)
    Y = torch.zeros(len(yt), n_classes, dtype=torch.float64)
    Y[torch.arange(len(yt)), yt] = 1.0
    acc = {}
    for l in sorted(train):
        H = train[l].shape[0]
        accs = []
        for h in range(H):
            X, Xe = train[l][h].double(), test[l][h].double()
            mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
            X, Xe = (X - mu) / sd, (Xe - mu) / sd
            X1, Xe1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1), torch.cat([Xe, torch.ones(len(Xe), 1, dtype=Xe.dtype)], 1)
            W = torch.linalg.solve(X1.T @ X1 + ridge * torch.eye(X1.shape[1], dtype=X1.dtype), X1.T @ Y)
            accs.append(float(((Xe1 @ W).argmax(1) == ye).double().mean()))
        acc[l] = accs
    seen = set(yt.tolist())
    majority = int(torch.bincount(yt, minlength=n_classes).argmax())
    return {"accuracy": acc, "chance": 1.0 / n_classes, "n_train": int(len(yt)), "n_test": int(len(ye)),
            "n_classes_seen": int(len(seen)), "n_classes": int(n_classes),
            "majority_baseline": float((ye == majority).double().mean()),            # always predict the most frequent training label
            "test_labels_unseen_in_training": float(sum(1 for y in ye.tolist() if y not in seen) / max(len(ye), 1))}
