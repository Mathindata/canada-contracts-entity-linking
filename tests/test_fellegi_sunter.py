import numpy as np
import pandas as pd

from cel.fellegi_sunter import FellegiSunter

L = {"f1": 3, "f2": 3, "f3": 2}
M = {"f1": [0.05, 0.15, 0.80], "f2": [0.10, 0.20, 0.70], "f3": [0.3, 0.7]}
U = {"f1": [0.85, 0.10, 0.05], "f2": [0.80, 0.15, 0.05], "f3": [0.9, 0.1]}


def _sample(n, lam, rng):
    z = rng.random(n) < lam
    X = {f: np.where(z, rng.choice(L[f], n, p=M[f]), rng.choice(L[f], n, p=U[f])) for f in L}
    return pd.DataFrame(X), z


def test_em_session_recovers_m():
    rng = np.random.default_rng(0)
    X_rand, _ = _sample(200_000, 0.0, rng)        # random pairs: non-matches
    X_block, z = _sample(40_000, 0.3, rng)
    fs = FellegiSunter(L).fit_u(X_rand)
    info = fs.fit_m_session(X_block, ["f1", "f2", "f3"])
    for f in L:
        assert np.allclose(fs.m[f], M[f], atol=0.03), (f, fs.m[f])
    assert abs(info["block_prior"] - 0.3) < 0.03
    assert info["suspect_swapped"] == []


def test_missing_level_contributes_nothing():
    rng = np.random.default_rng(1)
    X_rand, _ = _sample(50_000, 0.0, rng)
    fs = FellegiSunter(L).fit_u(X_rand)
    fs.m = {f: np.array(M[f]) for f in L}
    a = pd.DataFrame({"f1": [2], "f2": [2], "f3": [-1]})
    b = pd.DataFrame({"f1": [2], "f2": [2], "f3": [0]})
    c = pd.DataFrame({"f1": [2], "f2": [2], "f3": [1]})
    assert fs.log2_bf(b)[0] < fs.log2_bf(a)[0] < fs.log2_bf(c)[0]


def test_global_lambda_fixed_point():
    rng = np.random.default_rng(2)
    X_rand, _ = _sample(100_000, 0.0, rng)
    X_cand, z = _sample(5_000, 0.6, rng)
    fs = FellegiSunter(L).fit_u(X_rand)
    fs.m = {f: np.array(M[f]) for f in L}
    lam = fs.fit_lambda(X_cand, n_all_pairs=1_000_000)
    expected = fs.predict_proba(X_cand).sum()
    assert np.isclose(lam * 1_000_000, expected, rtol=1e-3)
