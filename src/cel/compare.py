"""Turn each pair into discrete comparison levels (the 'observations' of the detector).

Level -1 means 'missing': that field gives no evidence either way.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

FIELDS = ["name", "tokens", "postal", "acronym"]
N_LEVELS = {"name": 4, "tokens": 3, "postal": 3, "acronym": 2}
LEVEL_LABELS = {
    "name": ["Jaro-Winkler < 0.88", "Jaro-Winkler 0.88 to 0.95", "Jaro-Winkler >= 0.95",
             "identical cleaned name"],
    "tokens": ["weighted token overlap < 0.5", "overlap 0.5 to 0.8", "overlap >= 0.8"],
    "postal": ["different region", "same first letter (region)", "same 3-character prefix"],
    "acronym": ["no acronym relation", "one name is the other's initials"],
}


def token_idf(clean_names: pd.Series) -> dict[str, float]:
    df = Counter()
    for s in clean_names:
        df.update(set(s.split()))
    n = len(clean_names)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _weighted_jaccard(a: str, b: str, idf: dict, default: float) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    w = lambda s: sum(idf.get(t, default) for t in s)  # noqa: E731
    return w(sa & sb) / w(sa | sb)


def compare(pairs: pd.DataFrame, v: pd.DataFrame, idf: dict) -> pd.DataFrame:
    A, B = v.loc[pairs["a"]], v.loc[pairs["b"]]
    ca, cb = A["clean"].to_numpy(), B["clean"].to_numpy()
    jw = cpdist(list(ca), list(cb), scorer=JaroWinkler.normalized_similarity, workers=-1) \
        if len(pairs) else np.array([])
    name = np.select([ca == cb, jw >= 0.95, jw >= 0.88], [3, 2, 1], 0)

    default = max(idf.values()) if idf else 1.0
    wj = np.array([_weighted_jaccard(x, y, idf, default) for x, y in zip(ca, cb)])
    tokens = np.select([wj >= 0.8, wj >= 0.5], [2, 1], 0)

    fa, fb = A["fsa"].to_numpy(), B["fsa"].to_numpy()
    missing = (fa == "") | (fb == "")
    same = fa == fb
    region = np.array([x[:1] == y[:1] for x, y in zip(fa, fb)])
    postal = np.where(missing, -1, np.select([same, region], [2, 1], 0))

    ia, ib = A["initials"].to_numpy(), B["initials"].to_numpy()
    na, nb = A["clean_nospace"].to_numpy(), B["clean_nospace"].to_numpy()
    acr = (((ia == nb) & (np.char.str_len(ia.astype(str)) >= 3)) |
           ((ib == na) & (np.char.str_len(ib.astype(str)) >= 3))).astype(int)

    out = pairs[["a", "b"]].copy()
    out["name"], out["tokens"], out["postal"], out["acronym"] = name, tokens, postal, acr
    out["jw"] = jw
    out["wjaccard"] = wj
    out["exact_clean"] = (ca == cb).astype(int)
    return out


def random_pairs(v: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = v.index.to_numpy()
    a = rng.choice(idx, size=n)
    b = rng.choice(idx, size=n)
    keep = a != b
    df = pd.DataFrame({"a": a[keep], "b": b[keep]})
    s = np.sort(df[["a", "b"]].to_numpy(), axis=1)
    return pd.DataFrame(s, columns=["a", "b"]).drop_duplicates().reset_index(drop=True)
