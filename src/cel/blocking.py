"""Blocking: decide which pairs are worth comparing.

Comparing every pair of n vendor strings costs n(n-1)/2. Blocking keeps only
pairs that share a cheap key. Any true match that shares no key is lost for good,
so the evaluation also samples "near-miss" pairs outside the blocks to estimate
how much recall blocking costs.
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

log = logging.getLogger(__name__)


def _pairs_from_groups(keys: pd.Series, max_block: int, rule: str, skipped: list) -> set:
    out = set()
    for k, idx in keys.groupby(keys).groups.items():
        if k == "" or len(idx) < 2:
            continue
        if len(idx) > max_block:
            skipped.append((rule, k, len(idx)))
            continue
        for a, b in combinations(sorted(idx), 2):
            out.add((a, b))
    return out


def candidate_pairs(v: pd.DataFrame, max_block: int = 3000) -> tuple[pd.DataFrame, list]:
    skipped: list = []
    rules = {}
    # R1: same postal prefix and same first word of the cleaned name
    k1 = np.where(v["fsa"] != "", v["fsa"] + "|" + v["tok0"], "")
    rules["same_fsa_first_word"] = _pairs_from_groups(pd.Series(k1, index=v.index),
                                                      max_block, "R1", skipped)
    # R2: identical cleaned name, any postal code (catches moves and missing postal codes)
    rules["same_clean_name"] = _pairs_from_groups(v["clean"], max_block, "R2", skipped)
    # R3: one name is the initials of the other (e.g. "IBM" vs "INTERNATIONAL BUSINESS MACHINES")
    r3 = set()
    by_nospace = v[v["clean_nospace"].str.len() >= 3].groupby("clean_nospace").groups
    for vk, ini in v.loc[v["initials"].str.len() >= 3, "initials"].items():
        for other in by_nospace.get(ini, []):
            if other != vk:
                r3.add(tuple(sorted((vk, other))))
    rules["acronym"] = r3

    allp = set().union(*rules.values())
    df = pd.DataFrame(sorted(allp), columns=["a", "b"])
    for name, s in rules.items():
        df[f"rule_{name}"] = [p in s for p in zip(df["a"], df["b"])]
    if skipped:
        log.warning("skipped %d oversized blocks: %s", len(skipped), skipped[:5])
    return df, skipped


def near_miss_pairs(v: pd.DataFrame, blocked: pd.DataFrame, window: int, jw_min: float
                    ) -> pd.DataFrame:
    """Pairs outside the blocks that look similar: sorted-neighbourhood on the clean name.

    Sort names alphabetically and look `window` places ahead; keep pairs with
    Jaro-Winkler similarity >= jw_min that blocking did not produce.
    """
    order = v.sort_values("clean")
    keys, names = order.index.to_list(), order["clean"].to_list()
    have = set(zip(blocked["a"], blocked["b"]))
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, min(i + 1 + window, len(keys))):
            a, b = sorted((keys[i], keys[j]))
            if (a, b) in have:
                continue
            if JaroWinkler.normalized_similarity(names[i], names[j]) >= jw_min:
                out.append((a, b))
    return pd.DataFrame(sorted(set(out)), columns=["a", "b"])


def training_pairs(v: pd.DataFrame, key: str, max_pairs: int, seed: int) -> pd.DataFrame:
    """Pairs that share `key` (e.g. 'fsa' or 'tok0'), sampled down to max_pairs.

    Used only to train m. Sampling is proportional to each group's pair count, so
    the sample looks like the full set of within-key pairs.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for k, idx in v.groupby(key).groups.items()
              if k != "" and len(idx) > 1]
    sizes = np.array([len(g) for g in groups], dtype=float)
    npairs = sizes * (sizes - 1) / 2
    total = npairs.sum()
    out = []
    for g, n in zip(groups, npairs):
        if total <= max_pairs:
            out += [(a, b) for a, b in combinations(sorted(g), 2)]
            continue
        k = int(round(max_pairs * n / total))
        if k == 0:
            continue
        i = rng.integers(0, len(g), size=2 * k)
        j = rng.integers(0, len(g), size=2 * k)
        keep = i != j
        out += [tuple(sorted((g[x], g[y]))) for x, y in zip(i[keep][:k], j[keep][:k])]
    return pd.DataFrame(sorted(set(out)), columns=["a", "b"])


def reduction_ratio(n_strings: int, n_pairs: int) -> float:
    total = n_strings * (n_strings - 1) / 2
    return 1.0 - n_pairs / total if total else 0.0
