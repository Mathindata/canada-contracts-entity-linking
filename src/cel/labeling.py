"""Stratified labeling sheet and inverse-probability weights.

Random pairs are almost all easy non-matches, so a random sample of labels says
little. Instead, candidate pairs are split into equal-size strata by the model's
evidence score (log2 Bayes factor) and each stratum is sampled. Strata are built
on the score's rank, not on P(match), because P(match) piles up near 0 and 1 and
would leave most probability bins empty. A labeled pair from stratum s stands for N_s / n_s pairs,
which keeps precision and recall estimates unbiased for the whole set.

One extra stratum, NEAR, holds similar-looking pairs that blocking never
produced. True matches found there measure what blocking costs in recall.

The sheet hides model scores so they cannot bias the person labeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YES = {"1", "y", "yes", "same", "s", "true", "t"}
NO = {"0", "n", "no", "diff", "different", "d", "false", "f"}


def score_strata(score: pd.Series, n_strata: int) -> np.ndarray:
    k = min(n_strata, len(score))
    q = pd.qcut(score.rank(method="first"), q=k, labels=False)
    return np.array([f"S{i:02d}" for i in q])


def make_sheet(cand: pd.DataFrame, near: pd.DataFrame, v: pd.DataFrame, per_stratum: int,
               near_n: int, seed: int, n_strata: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    c = cand[["a", "b", "prob"]].copy()
    c["stratum"] = score_strata(cand["log2_bf"], n_strata)
    c["in_candidates"] = True
    n = near[["a", "b", "prob"]].copy()
    n["stratum"] = "NEAR"
    n["in_candidates"] = False
    allp = pd.concat([c, n], ignore_index=True)

    picks = []
    for s, grp in allp.groupby("stratum"):
        k = min(len(grp), near_n if s == "NEAR" else per_stratum)
        take = grp.iloc[rng.choice(len(grp), size=k, replace=False)].copy()
        take["stratum_size"] = len(grp)
        take["n_sampled"] = k
        take["weight"] = len(grp) / k
        picks.append(take)
    key = pd.concat(picks, ignore_index=True)
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key.insert(0, "pair_id", [f"L{i:04d}" for i in range(len(key))])

    A, B = v.loc[key["a"]].reset_index(drop=True), v.loc[key["b"]].reset_index(drop=True)
    sheet = pd.DataFrame({
        "pair_id": key["pair_id"],
        "name_a": A["vendor_name"], "postal_a": A["fsa"], "departments_a": A["departments"],
        "example_work_a": A["example_description"],
        "name_b": B["vendor_name"], "postal_b": B["fsa"], "departments_b": B["departments"],
        "example_work_b": B["example_description"],
        "label": "", "note": "",
    })
    return sheet, key


def parse_label(x) -> float:
    s = str(x).strip().lower()
    if s in YES:
        return 1.0
    if s in NO:
        return 0.0
    return np.nan  # blank, "?", "unsure"


def read_labels(sheet_path: str, key_path: str) -> pd.DataFrame:
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    key = pd.read_csv(key_path)
    lab = sheet[["pair_id", "label", "note"]].copy()
    lab["y"] = lab["label"].map(parse_label)
    return key.merge(lab, on="pair_id", how="left")


def oracle_labels(key: pd.DataFrame, truth: dict[str, str]) -> pd.DataFrame:
    """Synthetic data only: fill labels from the known ground truth."""
    out = key.copy()
    out["y"] = [float(truth[a] == truth[b]) for a, b in zip(out["a"], out["b"])]
    out["label"] = out["y"].astype(int).astype(str)
    out["note"] = "oracle (synthetic)"
    return out
