"""P0.12: opa.stats against known values (scipy where it has an equivalent, textbook or hand-computed otherwise)."""
import math
from statistics import NormalDist

import numpy as np
import pytest
from scipy import optimize, stats as sps

from opa import stats as st

# ------------------------------------------------------------------ protocol Section 7 power tables (all 24 cells)
SUPERIORITY = {0.10: (870, 312, 77), 0.15: (1306, 469, 116), 0.20: (1742, 626, 155), 0.30: (2614, 940, 234)}
EQUIVALENCE = {0.10: (2141, 952, 343), 0.15: (3212, 1428, 514), 0.20: (4282, 1904, 686), 0.30: (6423, 2855, 1028)}


@pytest.mark.parametrize("pd_", sorted(SUPERIORITY))
def test_n_mcnemar_reproduces_protocol_table(pd_):
    for gain, expected in zip((0.03, 0.05, 0.10), SUPERIORITY[pd_]):
        assert st.n_mcnemar(pd_, gain) == expected, (pd_, gain)


@pytest.mark.parametrize("pd_", sorted(EQUIVALENCE))
def test_n_tost_reproduces_protocol_table(pd_):
    for margin, expected in zip((0.02, 0.03, 0.05), EQUIVALENCE[pd_]):
        assert st.n_tost(pd_, margin) == expected, (pd_, margin)


def test_n_mcnemar_hand_value():
    # Connor: ((1.959964 * sqrt(.1) + 0.841621 * sqrt(.1 - .0025))^2) / .0025 = 311.6 -> 312
    z = NormalDist().inv_cdf
    raw = (z(0.975) * math.sqrt(0.1) + z(0.8) * math.sqrt(0.0975)) ** 2 / 0.0025
    assert 311 < raw < 312 and st.n_mcnemar(0.10, 0.05) == 312


# ------------------------------------------------------------------ existing functions (unchanged)
def test_existing_functions_known_values():
    adj = st.holm({"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.3})
    assert [round(adj[k], 6) for k in "abcd"] == [0.004, 0.03, 0.08, 0.3]
    assert st.mcnemar_exact(10, 0) == pytest.approx(0.001953125, abs=1e-15)
    assert st.mcnemar_exact(6, 1) == pytest.approx(2 * 8 / 128, abs=1e-15)        # 2 * P(X <= 1 | n=7, 1/2)
    assert st.mcnemar_exact(0, 0) == 1.0
    assert st.mcnemar_exact(30, 18) == pytest.approx(sps.binomtest(18, 48, 0.5).pvalue, abs=1e-15)
    assert st.discordant_counts([1, 1, 0, 0, 1], [0, 1, 1, 0, 0]) == (2, 1)


# ------------------------------------------------------------------ bootstrap and contrast
def test_chunked_bootstrap_reproduces_paired_bootstrap():
    rng = np.random.default_rng(5)
    for n in (7, 50, 333):
        a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
        d = (a - b) * 100.0
        for chunk in (1, 3, 997, None):
            boots = st._bootstrap_means(d, n_boot=2000, seed=3, chunk_rows=chunk)
            ref_idx = np.random.default_rng(3).integers(0, n, size=(2000, n))
            assert np.array_equal(boots, d[ref_idx].mean(axis=1))
        diff, (lo, hi) = st.paired_bootstrap(a, b, n_boot=2000, seed=3)
        c = st.contrast(a, b, "k", n_boot=2000, seed=3)
        assert c["value"] == diff and c["ci_lo"] == lo and c["ci_hi"] == hi


def test_contrast_row():
    a = np.array([1] * 60 + [0] * 40)
    b = np.array([1] * 50 + [0] * 50)                        # 10 pairs a-right/b-wrong, 0 the other way
    c = st.contrast(a, b, "S11_vs_S2", run_ids=["r1", "r2"], split="dev")
    assert c["key"] == "S11_vs_S2" and c["value"] == 10.0 and c["n"] == 100
    assert (c["n10"], c["n01"]) == (10, 0) and c["discordant_rate"] == 0.1
    assert c["p_raw"] == pytest.approx(0.001953125)
    assert c["ci_lo"] <= 10.0 <= c["ci_hi"] and c["ci95"] == [c["ci_lo"], c["ci_hi"]]
    assert c["p_holm"] is None and c["run_ids"] == ["r1", "r2"] and "McNemar" in c["test"]
    with pytest.raises(ValueError):
        st.contrast([0.5, 1], [0, 1])                         # McNemar needs binary outcomes
    w = st.contrast([3, 5, 2, 8, 6, 4, 9], [1, 2, 2, 3, 1, 0, 2], test="wilcoxon", scale=1.0, n_boot=500)
    ref = sps.wilcoxon([3, 5, 2, 8, 6, 4, 9], [1, 2, 2, 3, 1, 0, 2], zero_method="wilcox").pvalue
    assert w["p_raw"] == pytest.approx(ref) and w["value"] == pytest.approx(26 / 7)
    s = st.contrast([3, 5, 2, 8], [1, 2, 2, 3], test="sign", scale=1.0, n_boot=500)
    assert s["p_raw"] == pytest.approx(sps.binomtest(0, 3, 0.5).pvalue)


# ------------------------------------------------------------------ TOST
def _brute_force_score_z(n10, n01, n, d0):
    """Restricted MLE of (q10, q01) under q10 - q01 = d0 by numerical optimization, then the score statistic."""
    n00 = n - n10 - n01
    lo, hi = max(0.0, -d0) + 1e-12, (1 - d0) / 2 - 1e-12

    def nll(q01):
        q10 = q01 + d0
        rest = 1 - q10 - q01
        return -(n10 * np.log(q10) + n01 * np.log(q01) + n00 * np.log(rest))
    q01 = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13}).x
    var = n * (2 * q01 + d0 - d0 ** 2)
    return (n10 - n01 - n * d0) / math.sqrt(var)


@pytest.mark.parametrize("n10,n01,n,d0", [(20, 14, 200, 0.03), (20, 14, 200, -0.03), (3, 9, 120, 0.05),
                                          (40, 41, 500, -0.02), (1, 1, 60, 0.03)])
def test_tango_score_matches_numerical_rmle(n10, n01, n, d0):
    assert st.tango_score_z(n10, n01, n, d0) == pytest.approx(_brute_force_score_z(n10, n01, n, d0), rel=1e-6)


def test_tango_reduces_to_mcnemar_z():
    assert st.tango_score_z(10, 4, 100, 0.0) == pytest.approx(6 / math.sqrt(14), rel=1e-12)


def test_tost_wald_hand_computed():
    a = np.array([1] * 20 + [0] * 14 + [1] * 100 + [0] * 66)
    b = np.array([0] * 20 + [1] * 14 + [1] * 100 + [0] * 66)       # n = 200, n10 = 20, n01 = 14, d = 0.03
    r = st.tost_paired_p(a, b, margin=3.0, method="wald")
    se = math.sqrt((34 / 200 - 0.03 ** 2) / 200)                       # 0.0290775
    assert r["z_lower"] == pytest.approx(0.06 / se) and r["z_upper"] == pytest.approx(0.0, abs=1e-12)
    assert r["p_lower"] == pytest.approx(sps.norm.sf(0.06 / se)) and r["p"] == pytest.approx(0.5)
    assert r["diff"] == pytest.approx(3.0) and not r["equivalent"]


def test_tost_equivalence_decisions():
    same = np.r_[np.ones(600), np.zeros(400)]
    r = st.tost_paired_p(same, same, margin=3.0)
    assert r["equivalent"] and r["p"] < 1e-6 and r["n10"] == r["n01"] == 0
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 40)
    b = a.copy()
    b[:6] = 1 - b[:6]
    assert not st.tost_paired_p(a, b, margin=3.0)["equivalent"]       # small n cannot show equivalence
    # symmetry: swapping the arms swaps the two one-sided p-values
    x, y = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    r1, r2 = st.tost_paired_p(x, y, 5.0), st.tost_paired_p(y, x, 5.0)
    assert r1["p_lower"] == pytest.approx(r2["p_upper"]) and r1["p"] == pytest.approx(r2["p"])


def test_tost_size_at_margin_boundary():
    """At a true difference equal to the margin, the score TOST rejects at about alpha (<= 0.05 + MC error)."""
    rng = np.random.default_rng(42)
    n, p10, p01 = 1000, 0.08, 0.05                                   # true difference = 3 points = margin
    counts = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=3000)
    rej = 0
    for n10, n01, _ in counts:
        z_hi = st.tango_score_z(int(n10), int(n01), n, 0.03)
        z_lo = st.tango_score_z(int(n10), int(n01), n, -0.03)
        p = max(1 - NormalDist().cdf(z_lo), NormalDist().cdf(z_hi))
        rej += p < 0.05
    assert 0.03 < rej / 3000 < 0.065


def test_tost_agrees_with_bootstrap_version_on_clear_cases():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 3000)
    b = a.copy()
    flip = rng.choice(3000, 150, replace=False)
    b[flip] = 1 - b[flip]
    assert st.tost_paired_p(a, b, 3.0)["equivalent"] == st.tost_paired(a, b, 3.0, n_boot=2000)["equivalent"] is True


# ------------------------------------------------------------------ R^2 binned
def test_r2_binned_hand_example():
    pred = np.array([0.2] * 4 + [0.5] * 4 + [0.8] * 4)
    obs = np.array([1, 0, 0, 0] + [1, 1, 0, 0] + [1, 1, 1, 0], float)   # bin means 0.25, 0.5, 0.75
    r = st.r2_binned(pred, obs, bins=3)
    assert [b["n"] for b in r["bins"]] == [4, 4, 4]
    assert r["r2"] == pytest.approx(1 - (0.0025 + 0 + 0.0025) / (0.0625 + 0 + 0.0625))   # 0.96
    assert r["r2_linear"] == pytest.approx(1.0)


def test_r2_binned_equal_counts_and_uniform():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=103)
    o = (rng.uniform(size=103) < p).astype(float)
    r = st.r2_binned(p, o, bins=10)
    assert sorted({b["n"] for b in r["bins"]}) == [10, 11] and sum(b["n"] for b in r["bins"]) == 103
    assert all(r["bins"][i]["hi"] <= r["bins"][i + 1]["lo"] for i in range(9))
    u = st.r2_binned(p, o, bins=4, method="uniform")
    assert u["n_bins"] == 4 and all(b["hi"] - b["lo"] <= 0.25 for b in u["bins"])
    # calibrated prediction on a large sample: R^2 close to 1
    p = rng.uniform(size=20000)
    o = (rng.uniform(size=20000) < p).astype(float)
    assert st.r2_binned(p, o, bins=10)["r2"] > 0.98
    # useless prediction: R^2 well below 0.5
    assert st.r2_binned(p, rng.integers(0, 2, 20000), bins=10)["r2"] < 0.5
    assert math.isnan(st.r2_binned([0.5, 0.5], [1, 1], bins=2)["r2"])


# ------------------------------------------------------------------ logistic regression
CHALLENGER_T = [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81]
CHALLENGER_Y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_logistic_regression_challenger_textbook_values():
    # Dalal, Fowlkes & Hoadley (1989) / Agresti: logit = 15.0429 - 0.2322 temp, SE 7.3786 and 0.1082
    r = st.logistic_regression(CHALLENGER_T, CHALLENGER_Y, names=["temp"])
    assert r["converged"] and r["names"] == ["intercept", "temp"]
    assert r["coef"][0] == pytest.approx(15.0429, abs=1e-4) and r["coef"][1] == pytest.approx(-0.23216, abs=1e-5)
    assert r["se"][0] == pytest.approx(7.3786, abs=1e-4) and r["se"][1] == pytest.approx(0.10824, abs=1e-5)
    assert r["p"][1] == pytest.approx(2 * sps.norm.sf(abs(r["coef"][1] / r["se"][1])))


def test_logistic_regression_matches_scipy_optimizer():
    rng = np.random.default_rng(3)
    n = 800
    X = np.column_stack([rng.integers(0, 5, n), rng.uniform(size=n), rng.uniform(size=n)])   # h_late, g_s, g_g
    eta = 0.5 - 0.6 * X[:, 0] + 1.5 * X[:, 1] + 1.0 * X[:, 2]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    r = st.logistic_regression(X, y, names=["h_late", "gamma_s", "gamma_g"])
    Xi = np.column_stack([np.ones(n), X])

    def nll(b):
        e = Xi @ b
        return float(np.sum(np.logaddexp(0, e) - y * e))
    ref = optimize.minimize(nll, np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    assert np.allclose(r["coef"], ref, atol=1e-4)
    assert r["loglik"] == pytest.approx(-nll(np.array(r["coef"])), rel=1e-10)


def test_logistic_regression_separation_not_converged():
    r = st.logistic_regression([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1], max_iter=50)
    assert not r["converged"]


# ------------------------------------------------------------------ phi
def test_phi_coefficient():
    # n11 = 1, n10 = 1, n01 = 0, n00 = 3: (1*3 - 1*0) / sqrt(2 * 3 * 1 * 4)
    assert st.phi_coefficient([1, 1, 0, 0, 0], [1, 0, 0, 0, 0]) == pytest.approx(3 / math.sqrt(24))
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 500)
    b = (rng.uniform(size=500) < np.where(a == 1, 0.7, 0.3)).astype(int)
    phi = st.phi_coefficient(a, b)
    assert phi == pytest.approx(np.corrcoef(a, b)[0, 1], rel=1e-12)
    table = np.array([[((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()],
                      [((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()]])
    from scipy.stats.contingency import association
    assert abs(phi) == pytest.approx(association(table, method="cramer"), rel=1e-12)
    assert st.phi_coefficient([True, False, True], [True, False, True]) == pytest.approx(1.0)
    assert math.isnan(st.phi_coefficient([1, 1, 1], [0, 1, 0]))
