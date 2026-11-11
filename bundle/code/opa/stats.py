"""Statistics from protocol Section 7. Pure numpy/scipy so the verifier can reuse holm()."""
from __future__ import annotations

import numpy as np


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0, scale: float = 100.0):
    """Mean difference (a - b) in points (scale=100 for EM in 0/1) with a 95% percentile CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    d = (a - b) * scale
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float, n_boot: int = 10_000, seed: int = 0, scale: float = 100.0):
    """Equivalence of paired outcomes within +-margin (points): equivalent if the 90% bootstrap CI lies inside."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (a - b) * scale
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    return {"diff": float(d.mean()), "ci90": (lo, hi), "equivalent": bool(lo > -margin and hi < margin), "margin": margin}


def discordant_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return int((a & ~b).sum()), int((~a & b).sum())


def n_mcnemar(p_discordant: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per cell for a paired McNemar test (Connor's approximation); delta as a fraction (0.05 = 5 points)."""
    from math import ceil, sqrt
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return int(ceil((z(1 - alpha / 2) * sqrt(p_discordant) + z(power) * sqrt(p_discordant - delta ** 2)) ** 2 / delta ** 2))


def n_tost(p_discordant: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> int:
    from math import ceil
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return int(ceil((z(1 - alpha) + z(1 - (1 - power) / 2)) ** 2 * p_discordant / margin ** 2))


# ====================================================================== additions (P0.12)
# Everything below is new; the functions above are unchanged (the bundle verifier and the repo tests rely on them).

def _bootstrap_means(d: np.ndarray, n_boot: int = 10_000, seed: int = 0, chunk_rows: int | None = None) -> np.ndarray:
    """Bootstrap means of ``d`` drawn exactly as ``paired_bootstrap`` draws them (same generator, same seed, same
    stream: ``rng.integers(0, n, size=(n_boot, n))`` row by row), but in row chunks so that memory stays bounded for
    large n. With numpy's Generator, drawing the index matrix in consecutive row blocks yields the same numbers as one
    call; tests/test_stats.py checks that this reproduces paired_bootstrap exactly."""
    d = np.asarray(d, float)
    n = len(d)
    if chunk_rows is None:
        chunk_rows = max(1, int(2_000_000 // max(n, 1)))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, float)
    done = 0
    while done < n_boot:
        k = min(chunk_rows, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        out[done:done + k] = d[idx].mean(axis=1)
        done += k
    return out


def tango_score_z(n10: int, n01: int, n: int, delta0: float) -> float:
    """Tango's (1998, Stat Med 17:891) score statistic for H0: p1 - p2 = delta0 with paired binary data.

    n10 = pairs where arm 1 is right and arm 2 wrong, n01 = the reverse, n = all pairs; delta0 as a fraction.
    The variance uses the restricted MLE of q01 under H0, the positive root of
    2n q^2 + W q - n01 delta0 (1 - delta0) = 0 with W = -(n10 + n01) + delta0 (2n - n10 + n01), so the statistic is
    finite even with zero discordant pairs. At delta0 = 0 it equals the McNemar z = (n10 - n01) / sqrt(n10 + n01)."""
    if n <= 0:
        return float("nan")
    w = -(n10 + n01) + delta0 * (2 * n - n10 + n01)
    q01 = (np.sqrt(w * w + 8.0 * n * n01 * delta0 * (1.0 - delta0)) - w) / (4.0 * n)
    var = n * (2.0 * q01 + delta0 * (1.0 - delta0))
    num = n10 - n01 - n * delta0
    if var <= 0:
        return 0.0 if num == 0 else float(np.sign(num) * np.inf)
    return float(num / np.sqrt(var))


def tost_paired_p(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05, method: str = "score",
                  scale: float = 100.0) -> dict:
    """Two one-sided tests (TOST) for equivalence of paired binary outcomes a and b within +-margin points.

    H0: |p_a - p_b| >= margin  vs  H1: |p_a - p_b| < margin, each one-sided test at level ``alpha`` (0.05 each side,
    protocol 7). Lower test H0: d <= -margin, p_lo = 1 - Phi(Z(-margin)); upper test H0: d >= +margin,
    p_hi = Phi(Z(+margin)); p = max(p_lo, p_hi); equivalent iff p < alpha.

    method="score" (default): Tango's restricted-MLE score statistic (``tango_score_z``), which behaves well with few
    discordant pairs. method="wald": Z(delta0) = (d_hat - delta0) / se, se^2 = ((n10 + n01)/n - d_hat^2) / n
    (unrestricted; degenerate when n10 = n01 = 0, in which case p = 1 is returned).
    The sample-size formula n_tost() is the large-sample version of either test at true difference 0.

    Returns {"diff" (points), "p", "p_lower", "p_upper", "z_lower", "z_upper", "equivalent", "margin", "n",
    "n10", "n01", "method"}. Complements the bootstrap-interval tost_paired() above."""
    from statistics import NormalDist
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
        raise ValueError("tost_paired_p needs binary (0/1) outcomes")
    n = int(a.size)
    if n == 0:
        raise ValueError("tost_paired_p needs at least one pair")
    n10, n01 = discordant_counts(a, b)
    m = margin / scale
    d_hat = (n10 - n01) / n
    Phi = NormalDist().cdf
    if method == "score":
        z_lo, z_hi = tango_score_z(n10, n01, n, -m), tango_score_z(n10, n01, n, m)
    elif method == "wald":
        var = ((n10 + n01) / n - d_hat ** 2) / n
        if var <= 0:
            z_lo = z_hi = float("nan")
        else:
            se = float(np.sqrt(var))
            z_lo, z_hi = (d_hat + m) / se, (d_hat - m) / se
    else:
        raise ValueError(f"unknown method {method!r}")
    if np.isnan(z_lo) or np.isnan(z_hi):
        p_lo = p_hi = 1.0
    else:
        p_lo = 1.0 - Phi(z_lo) if np.isfinite(z_lo) else float(z_lo < 0)
        p_hi = Phi(z_hi) if np.isfinite(z_hi) else float(z_hi > 0)
    p = max(p_lo, p_hi)
    return {"diff": d_hat * scale, "p": float(p), "p_lower": float(p_lo), "p_upper": float(p_hi),
            "z_lower": float(z_lo), "z_upper": float(z_hi), "equivalent": bool(p < alpha), "margin": margin,
            "n": n, "n10": n10, "n01": n01, "method": method}


def r2_binned(predicted, observed, bins: int = 10, method: str = "quantile", ids=None) -> dict:
    """H3: R^2 of binned observed success against binned predicted gamma_eff^h_late.

    Binning: method="quantile" (default) sorts items by predicted value and splits them into ``bins`` groups of equal count (sizes differ by at most one; np.array_split). Equal-count bins
    give every bin the same weight and the same sampling noise. Tied predictions are common (every item with
    h_late = 0 predicts 1.0), and a tie that straddles a bin edge makes the bin means depend on the order the items
    arrive in. With ``ids`` (one item id per item) the sort key is (predicted, id), so the bins are the same for any
    input order; without ids ties keep their input order (stable sort), as before. method="uniform" uses ``bins`` equal-width bins on
    [0, 1] (right edge closed on the last bin) and drops empty bins. Items with NaN in either input are dropped.

    The primary statistic ``r2`` is the coefficient of determination of the prediction itself, measured against the
    identity line (the theory predicts the success rate, not a fitted line):
        r2 = 1 - sum_k (obs_k - pred_k)^2 / sum_k (obs_k - mean(obs))^2
    over bins k (unweighted; NaN if fewer than 2 bins or zero variance of the binned observed means). It can be
    negative when the prediction is worse than the mean. ``r2_linear`` (squared Pearson correlation of the bin means,
    i.e. R^2 of a fitted least-squares line) is reported alongside for reference.

    Returns {"r2", "r2_linear", "n", "n_bins", "method", "bins": [{"lo", "hi", "n", "mean_predicted",
    "mean_observed"}]}."""
    p = np.asarray(predicted, float)
    o = np.asarray(observed, float)
    if p.shape != o.shape or p.ndim != 1:
        raise ValueError("predicted and observed must be 1-D arrays of equal length")
    keep = ~(np.isnan(p) | np.isnan(o))
    p, o = p[keep], o[keep]
    if ids is not None:
        id_list = [str(x) for x in ids]
        if len(id_list) != len(keep):
            raise ValueError("ids must have one entry per item")
        id_list = [x for x, k in zip(id_list, keep) if k]
    if method == "quantile":
        if ids is not None:
            order = np.array(sorted(range(len(p)), key=lambda i: (p[i], id_list[i])), dtype=int)
        else:
            order = np.argsort(p, kind="stable")
        groups = [g for g in np.array_split(order, min(bins, len(p))) if len(g)] if len(p) else []
    elif method == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
        groups = [g for g in (np.flatnonzero(which == k) for k in range(bins)) if len(g)]
    else:
        raise ValueError(f"unknown method {method!r}")
    rows = [{"lo": float(p[g].min()), "hi": float(p[g].max()), "n": int(len(g)),
             "mean_predicted": float(p[g].mean()), "mean_observed": float(o[g].mean())} for g in groups]
    bp = np.array([r["mean_predicted"] for r in rows])
    bo = np.array([r["mean_observed"] for r in rows])
    r2 = r2_lin = float("nan")
    if len(rows) >= 2:
        ss_tot = float(((bo - bo.mean()) ** 2).sum())
        if ss_tot > 0:
            r2 = 1.0 - float(((bo - bp) ** 2).sum()) / ss_tot
            if bp.std() > 0:
                r2_lin = float(np.corrcoef(bp, bo)[0, 1] ** 2)
    return {"r2": r2, "r2_linear": r2_lin, "n": int(len(p)), "n_bins": len(rows), "method": method, "bins": rows}


def logistic_regression(X, y, names: list[str] | None = None, add_intercept: bool = True, max_iter: int = 100,
                        tol: float = 1e-10) -> dict:
    """Maximum-likelihood logistic regression by Newton-Raphson (IRLS), numpy only.

    X: (n, k) predictors (e.g. columns h_late, gamma_s, gamma_g for H3); y: 0/1 outcomes. An intercept column is
    prepended unless add_intercept=False. Standard errors from the inverse Fisher information at the optimum; Wald
    z = coef / se and two-sided normal p-values. ``converged`` is False if the step did not fall below ``tol`` within
    ``max_iter`` iterations or the information matrix became singular (e.g. perfect separation: coefficients then
    diverge and must not be reported as estimates).

    Returns {"names", "coef", "se", "z", "p", "converged", "n_iter", "loglik", "n"} with list-valued entries in the
    order of ``names`` ("intercept" first when added)."""
    from scipy.stats import norm
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1")
    names = list(names) if names is not None else [f"x{i}" for i in range(X.shape[1])]
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["intercept"] + names
    beta = np.zeros(X.shape[1])
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        info = X.T @ (X * (mu * (1.0 - mu))[:, None])
        try:
            step = np.linalg.solve(info, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            break
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    info = X.T @ (X * (mu * (1.0 - mu))[:, None])
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(info)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)
        converged = False
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    p = 2.0 * norm.sf(np.abs(z))
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-300, None)) + (1 - y) * np.log(np.clip(1 - mu, 1e-300, None))))
    return {"names": names, "coef": beta.tolist(), "se": se.tolist(), "z": z.tolist(), "p": p.tolist(),
            "converged": bool(converged), "n_iter": it, "loglik": ll, "n": int(len(y))}


def phi_coefficient(a, b) -> float:
    """Phi coefficient of two binary vectors (e.g. per late-bound hop: skeleton miss, graph miss; Corollary 14.1
    assumes 0). phi = (n11 n00 - n10 n01) / sqrt(n1. n0. n.1 n.0); equals the Pearson correlation of the 0/1
    vectors. NaN if any margin is zero."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
        raise ValueError("phi_coefficient needs binary (0/1 or bool) vectors")
    a, b = a.astype(bool), b.astype(bool)
    n11, n00 = int((a & b).sum()), int((~a & ~b).sum())
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    den = float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if den == 0:
        return float("nan")
    return float((n11 * n00 - n10 * n01) / np.sqrt(den))


def contrast(a, b, key: str | None = None, *, test: str = "mcnemar", n_boot: int = 10_000, seed: int = 0,
             scale: float = 100.0, run_ids: list[str] | None = None, split: str | None = None) -> dict:
    """One contrast row for tables and claims: value = mean(a - b) * scale (points for 0/1 outcomes), 95% paired
    bootstrap percentile CI (10,000 resamples, fixed seed; the same resamples as paired_bootstrap), n pairs,
    discordant counts and the raw two-sided p-value.

    test="mcnemar" (default, binary outcomes): exact McNemar (mcnemar_exact). test="wilcoxon" (paired continuous
    or count outcomes, e.g. repeated page-ins; use scale=1): scipy.stats.wilcoxon, two-sided, zero differences
    dropped (zero_method="wilcox"); p = 1 if all differences are zero. test="sign": exact sign test on the non-zero
    differences. n10 / n01 count pairs with a > b and a < b. p_holm is left None: Holm is applied across the
    primary family by the caller (stats.holm).

    Returns {"key", "value", "ci_lo", "ci_hi", "ci95", "n", "n10", "n01", "discordant_rate", "p_raw", "p_holm",
    "test", "run_ids", "split"}."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    n = int(a.size)
    if n == 0:
        raise ValueError("contrast needs at least one pair")
    d = (a - b) * scale
    boots = _bootstrap_means(d, n_boot=n_boot, seed=seed)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    n10, n01 = int((a > b).sum()), int((a < b).sum())
    if test == "mcnemar":
        if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
            raise ValueError("test='mcnemar' needs binary outcomes; use 'wilcoxon' or 'sign'")
        p, label = mcnemar_exact(n10, n01), "paired bootstrap 95% CI (10,000 resamples) + exact McNemar"
    elif test == "wilcoxon":
        from scipy.stats import wilcoxon
        p = 1.0 if n10 + n01 == 0 else float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        label = "paired bootstrap 95% CI (10,000 resamples) + Wilcoxon signed-rank"
    elif test == "sign":
        p, label = mcnemar_exact(n10, n01), "paired bootstrap 95% CI (10,000 resamples) + exact sign test"
    else:
        raise ValueError(f"unknown test {test!r}")
    if n_boot != 10_000:
        label = label.replace("10,000", f"{n_boot:,}")
    return {"key": key, "value": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "ci95": [lo, hi], "n": n,
            "n10": n10, "n01": n01, "discordant_rate": (n10 + n01) / n, "p_raw": float(p), "p_holm": None,
            "test": label, "run_ids": list(run_ids or []), "split": split}
