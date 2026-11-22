"""From pairwise links to vendor entities, plus an audit of the merges.

Accepted links (P >= threshold) are edges; connected components are entities.
Components can chain: A~B and B~C may merge A with C even when A and C look
nothing alike. The audit scores every pair inside each entity and flags pairs
the model itself would reject.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .compare import compare


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_entities(v: pd.DataFrame, scored: pd.DataFrame, threshold: float):
    acc = scored[scored["prob"] >= threshold]
    uf = UnionFind(v.index)
    for a, b in zip(acc["a"], acc["b"]):
        uf.union(a, b)
    root = pd.Series({k: uf.find(k) for k in v.index}, name="root")

    best = pd.concat([acc[["a", "prob"]].rename(columns={"a": "vkey"}),
                      acc[["b", "prob"]].rename(columns={"b": "vkey"})])
    best = best.groupby("vkey")["prob"].max()

    ent = v[["vkey", "vendor_name", "n_rows"]].copy()
    ent["root"] = root
    sizes = ent.groupby("root")["vkey"].transform("size")
    ent["membership_prob"] = np.where(sizes > 1, ent["vkey"].map(best).fillna(1.0), 1.0)
    # canonical name: the raw name used on the most rows
    canon = ent.sort_values(["n_rows", "vendor_name"], ascending=[False, True]) \
               .groupby("root")["vendor_name"].first()
    roots = sorted(ent["root"].unique())
    rid = {r: f"E{i:06d}" for i, r in enumerate(roots)}
    ent["entity_id"] = ent["root"].map(rid)
    ent["canonical_name"] = ent["root"].map(canon)
    ent["entity_size"] = sizes
    return ent.drop(columns="root"), acc


def audit_entities(ent: pd.DataFrame, v: pd.DataFrame, fs, idf, weak: float = 0.05,
                   max_size: int = 200) -> pd.DataFrame:
    pairs = []
    for eid, grp in ent[ent["entity_size"] >= 3].groupby("entity_id"):
        keys = sorted(grp["vkey"])
        if len(keys) > max_size:
            continue
        pairs += [(eid, a, b) for a, b in combinations(keys, 2)]
    if not pairs:
        return pd.DataFrame(columns=["entity_id", "a", "b", "name_a", "name_b", "prob"])
    p = pd.DataFrame(pairs, columns=["entity_id", "a", "b"])
    lv = compare(p, v, idf)
    p["prob"] = fs.predict_proba(lv)
    p["name_a"] = v.loc[p["a"], "vendor_name"].to_numpy()
    p["name_b"] = v.loc[p["b"], "vendor_name"].to_numpy()
    return p[p["prob"] < weak].sort_values("prob").reset_index(drop=True)
