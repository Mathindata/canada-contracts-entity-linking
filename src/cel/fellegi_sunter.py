"""Fellegi-Sunter record linkage, written out from first principles.

For a pair with comparison levels x = (x_1, ..., x_K):
  m_k(l) = P(x_k = l | same vendor)      u_k(l) = P(x_k = l | different vendors)
Assuming the fields are conditionally independent given the match state, the
log-likelihood ratio is a sum of per-field weights:
  log2 BF(x) = sum_k log2( m_k(x_k) / u_k(x_k) )
and with a prior lambda = P(match) among candidate pairs,
  P(match | x) = 1 / (1 + (1 - lambda) / (lambda * BF(x))).
This is a naive-Bayes likelihood-ratio detector. A missing field (level -1)
contributes nothing.

Estimation (no labels), the standard recipe used by Splink:
  1. u from random pairs, then held fixed. Almost all random pairs are non-matches,
     so u(x) is the distribution of x over the whole space of non-matching pairs.
  2. m by EM in "training sessions". Each session takes the pairs produced by one
     blocking rule and estimates m only for the fields that rule does NOT use.
     Why: inside a block, the blocked-on field takes the same value for every
     pair, so it carries no information about match status; and pooling pairs
     from different rules lets EM separate the pairs by rule instead of by match
     status (seen in testing: it learned "same postal code" as evidence AGAINST a match).
  3. A global prior lambda = P(match) over ALL n(n-1)/2 pairs, by the fixed point
        lambda = (1/N_all) * sum over candidate pairs of P(match | x; lambda),
     treating pairs outside the blocks as non-matches (their posterior is ~0).
  With u and lambda defined over the whole pair space, P(match | x) is valid for
  any pair, and selecting pairs by x (blocking) does not change it.
  4. Each session can converge with its two classes swapped; the class whose m puts
     more weight on the top agreement level is taken as "match".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FellegiSunter:
    def __init__(self, n_levels: dict[str, int], alpha: float = 1.0):
        self.fields = list(n_levels)
        self.L = n_levels
        self.alpha = alpha            # additive smoothing
        self.m: dict[str, np.ndarray] = {}
        self.u: dict[str, np.ndarray] = {}
        self.lam: float = 0.1
        self.sessions: list[dict] = []

    # ---- estimation ----
    def _freq(self, x: np.ndarray, L: int, w: np.ndarray | None = None) -> np.ndarray:
        ok = x >= 0
        w = np.ones_like(x, dtype=float) if w is None else w
        counts = np.bincount(x[ok], weights=w[ok], minlength=L)
        return (counts + self.alpha) / (counts.sum() + self.alpha * L)

    def fit_u(self, X_random: pd.DataFrame) -> "FellegiSunter":
        for f in self.fields:
            self.u[f] = self._freq(X_random[f].to_numpy(), self.L[f])
        return self

    def _init_m(self) -> None:
        for f in self.fields:
            p = (np.arange(self.L[f]) + 1.0) ** 2   # tilt toward agreement levels
            self.m[f] = p / p.sum()

    def fit_m_session(self, X: pd.DataFrame, fields: list[str], use_fields: list[str] | None = None,
                      n_iter: int = 500, tol: float = 1e-9) -> dict:
        """EM inside one training block: u fixed, block prior free.

        `fields` are updated; `use_fields` (default: `fields`) enter the likelihood.
        Using already-trained fields in the likelihood anchors the two classes, which
        makes a single new field identifiable.
        """
        if not self.u:
            raise RuntimeError("call fit_u first")
        if not self.m:
            self._init_m()
        use = use_fields or fields
        lam_b, prev, it = 0.5, -np.inf, 0
        for it in range(n_iter):
            g, ll = self._posterior(X, fields=use, lam=lam_b, return_ll=True)
            lam_b = float(np.clip(g.mean(), 1e-6, 1 - 1e-6))
            for f in fields:
                self.m[f] = self._freq(X[f].to_numpy(), self.L[f], g)
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        swapped = []
        for f in fields:   # class-swap guard, field by field
            top = self.L[f] - 1
            if self.m[f][top] < self.u[f][top]:
                swapped.append(f)
        info = {"fields": fields, "n_pairs": len(X), "block_prior": lam_b,
                "iterations": it + 1, "suspect_swapped": swapped}
        self.sessions.append(info)
        return info

    def fit_lambda(self, X_cand: pd.DataFrame, n_all_pairs: float, n_iter: int = 200) -> float:
        lam = max(1.0 / n_all_pairs, 1e-12)
        for _ in range(n_iter):
            g = self._posterior(X_cand, lam=lam)
            new = float(np.clip(g.sum() / n_all_pairs, 1e-12, 1 - 1e-12))
            if abs(new - lam) < 1e-12 + 1e-9 * lam:
                lam = new
                break
            lam = new
        self.lam = lam
        return lam

    # ---- prediction ----
    def log2_bf(self, X: pd.DataFrame) -> np.ndarray:
        s = np.zeros(len(X))
        for f in self.fields:
            x = X[f].to_numpy()
            ok = x >= 0
            w = np.log2(self.m[f] / self.u[f])
            s[ok] += w[x[ok]]
        return s

    def _posterior(self, X: pd.DataFrame, fields: list[str] | None = None,
                   lam: float | None = None, return_ll: bool = False):
        lam = self.lam if lam is None else lam
        lm = np.zeros(len(X))
        lu = np.zeros(len(X))
        for f in (fields or self.fields):
            x = X[f].to_numpy()
            ok = x >= 0
            lm[ok] += np.log(self.m[f][x[ok]])
            lu[ok] += np.log(self.u[f][x[ok]])
        a = np.log(lam) + lm
        b = np.log(1 - lam) + lu
        mx = np.maximum(a, b)
        logz = mx + np.log(np.exp(a - mx) + np.exp(b - mx))
        g = np.exp(a - logz)
        return (g, float(logz.sum())) if return_ll else g

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._posterior(X)

    def weights_table(self, labels: dict[str, list[str]] | None = None) -> pd.DataFrame:
        rows = []
        for f in self.fields:
            for lvl in range(self.L[f]):
                rows.append({
                    "field": f, "level": lvl,
                    "meaning": labels[f][lvl] if labels else "",
                    "m": self.m[f][lvl], "u": self.u[f][lvl],
                    "log2_m_over_u": float(np.log2(self.m[f][lvl] / self.u[f][lvl])),
                })
        return pd.DataFrame(rows)
