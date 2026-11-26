"""Linguistic summaries of the contracts graph.

A linguistic summary is a short sentence about a whole table, with a degree of
truth in [0, 1]. Two forms (Yager 1982; Kacprzyk and Yager 2001):

  Type I    "Q contracts are S."           e.g. "Most contracts are low value."
  Type II   "Q R contracts are S."          e.g. "Most contracts with frequent vendors
                                                  were never amended."

  Q  a fuzzy quantifier ("most", "about half", ...), a membership function on the
     proportion r in [0, 1]
  R  the qualifier, a fuzzy set on the objects (which contracts we talk about)
  S  the summarizer, a fuzzy set on the objects (what we say about them)

Truth (Zadeh's calculus of quantified propositions):

  Type I   r = sum_i mu_S(y_i) / n
  Type II  r = sum_i min(mu_R(y_i), mu_S(y_i)) / sum_i mu_R(y_i)
  T = mu_Q(r)

The sums are sigma-counts: a contract that is "low value" to degree 0.4 counts
as 0.4 of a contract. min is the fuzzy AND.

What this module adds to the textbook version:
  - vendor attributes (how many contracts a vendor holds, how many departments it
    works with) depend on WHICH records count as the same vendor, so the same
    summaries are computed under several vendor definitions: raw strings,
    cleaned names, linked entities and (synthetic data only) the answer key.
    The gap between them shows how much entity linking changes what the data says.
  - a contrast score, r - r_S, which compares the group R with the whole table.
    A true sentence with zero contrast just repeats the base rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------- fuzzy sets and quantifiers ----------

@dataclass(frozen=True)
class FuzzySet:
    """Piecewise-linear membership function: straight lines between the points
    (xs[k], ys[k]), flat outside them. Shoulders and crisp steps are special cases."""
    xs: tuple
    ys: tuple

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.interp(x, self.xs, self.ys)
        return np.where(np.isnan(x), 0.0, mu)     # missing value: no membership


def falling(a: float, b: float) -> FuzzySet:
    """1 up to a, then down to 0 at b."""
    return FuzzySet((a, b), (1.0, 0.0))


def rising(a: float, b: float) -> FuzzySet:
    """0 up to a, then up to 1 at b."""
    return FuzzySet((a, b), (0.0, 1.0))


def trapezoid(a: float, b: float, c: float, d: float) -> FuzzySet:
    return FuzzySet((a, b, c, d), (0.0, 1.0, 1.0, 0.0))


@dataclass(frozen=True)
class Quantifier:
    word: str
    mf: FuzzySet

    def __call__(self, r) -> np.ndarray:
        return self.mf(np.clip(r, 0.0, 1.0))

    @property
    def width(self) -> float:
        """Area under mu_Q on [0, 1]. Smaller = more specific word."""
        return float(self(np.linspace(0, 1, 1001)).mean())


# The words cover [0, 1] with overlaps, so every proportion has a word that fits
# well. "most" keeps its usual definition in this literature (0 below 0.3, 1 above
# 0.8); between 0.55 and 0.8 the sharper "about two thirds" usually wins.
QUANTIFIERS = [
    Quantifier("almost none of the", falling(0.05, 0.15)),
    Quantifier("few", trapezoid(0.05, 0.12, 0.2, 0.3)),
    Quantifier("about a third of the", trapezoid(0.2, 0.28, 0.38, 0.46)),
    Quantifier("about half of the", trapezoid(0.38, 0.45, 0.55, 0.62)),
    Quantifier("about two thirds of the", trapezoid(0.55, 0.62, 0.72, 0.8)),
    Quantifier("most", rising(0.3, 0.8)),
    Quantifier("almost all", rising(0.85, 0.95)),
]


# ---------- linguistic variables ----------

@dataclass
class Label:
    """One fuzzy value of an attribute, with the words used in a sentence."""
    name: str           # short id, e.g. "value=low"
    mf: FuzzySet
    as_group: str       # qualifier wording: "low-value contracts"
    as_claim: str       # summarizer wording: "are low value"


@dataclass
class Variable:
    attribute: str      # column in the object table
    labels: list[Label]
    qualifier_only: bool = False
    uses_vendor: bool = False   # depends on the vendor definition


def _log_quantiles(s: pd.Series, qs) -> list[float]:
    v = np.log10(s[s > 0].dropna())
    return [float(10 ** v.quantile(q)) for q in qs]


def contract_variables(ref: pd.DataFrame) -> list[Variable]:
    """Fuzzy sets for contract-level attributes.

    Value uses quantiles of the reference table (on a log scale, since contract
    values span four orders of magnitude), so the words adapt to the slice. All
    other sets use fixed, readable break points. Edit them here; every run writes
    the break points it used to the output, so results are reproducible.
    """
    q20, q30, q40, q60, q70, q80 = _log_quantiles(ref["original_value"],
                                                  [.2, .3, .4, .6, .7, .8])
    return [
        Variable("original_value", [
            Label("value=low", falling(q20, q40), "low-value contracts", "are low value"),
            Label("value=medium", trapezoid(q20, q40, q60, q80), "medium-value contracts",
                  "are of medium value"),
            Label("value=high", rising(q60, q80), "high-value contracts", "are high value"),
        ]),
        Variable("growth", [
            Label("growth=none", falling(1.0, 1.1), "contracts that kept their original value",
                  "kept their original value"),
            Label("growth=large", rising(1.3, 1.8),
                  "contracts that grew a lot through amendments",
                  "grew a lot through amendments"),
        ]),
        Variable("n_amendments", [
            Label("amended=never", falling(0, 1), "never-amended contracts",
                  "were never amended"),
            Label("amended=often", rising(1, 3), "contracts amended several times",
                  "were amended several times"),
        ]),
        Variable("duration_months", [
            Label("duration=short", falling(6, 12), "short contracts (under about 6 months)",
                  "are short (under about 6 months)"),
            Label("duration=long", rising(18, 30), "long contracts (over about 2 years)",
                  "are long (over about 2 years)"),
        ]),
        Variable("vendor_n_contracts", [
            Label("vendor=one-off", falling(1, 2), "contracts with one-off vendors",
                  "are with one-off vendors"),
            Label("vendor=frequent", rising(2, 6), "contracts with frequent vendors",
                  "are with frequent vendors"),
        ], uses_vendor=True),
    ]


def department_variable(departments) -> Variable:
    return Variable("department", [
        Label(f"dept={d}", FuzzySet((0.0, 1.0), (0.0, 1.0)), f"contracts of {d}", "")
        for d in sorted(departments)
    ], qualifier_only=True)


def vendor_variables(ref_contracts: pd.DataFrame) -> list[Variable]:
    """Fuzzy sets for vendor-level attributes. Total value uses contract-value
    quantiles from the reference table so every vendor definition is judged by the
    same yardstick."""
    q50, q90 = _log_quantiles(ref_contracts["original_value"], [.5, .9])
    return [
        Variable("n_contracts", [
            Label("size=one-off", falling(1, 2), "one-off vendors", "hold a single contract"),
            Label("size=frequent", rising(2, 6), "frequent vendors",
                  "hold many contracts (about 6 or more)"),
        ]),
        Variable("n_departments", [
            Label("breadth=one", falling(1, 2), "vendors working with one department",
                  "work with only one department"),
            Label("breadth=several", rising(1, 2), "vendors working with several departments",
                  "work with several departments"),
        ]),
        Variable("share_amended", [
            Label("amend=rarely", falling(0.2, 0.5), "vendors whose contracts are rarely amended",
                  "rarely have contracts amended"),
            Label("amend=often", rising(0.3, 0.6), "vendors whose contracts are often amended",
                  "often have contracts amended"),
        ]),
        Variable("total_value", [
            Label("total=small", falling(q50, 2 * q50), "vendors with a small total value",
                  "have a small total value"),
            Label("total=large", rising(q90, 3 * q90), "vendors with a large total value",
                  "have a large total value"),
        ]),
    ]


# ---------- object tables ----------

def contract_table(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per contract, from the graph tables."""
    c = t["contracts"].copy()
    dur = (pd.to_datetime(c["end_date"], errors="coerce")
           - pd.to_datetime(c["first_date"], errors="coerce")).dt.days / 30.44
    c["duration_months"] = dur.clip(lower=0)
    return c


def contract_vendor_keys(t: dict[str, pd.DataFrame], truth: pd.DataFrame | None = None
                         ) -> pd.DataFrame:
    """Which vendor each contract belongs to, under each vendor definition.

    A contract's vendor is taken from its original record (instrument C), or its
    first record if there is none, since amendments sometimes spell the name
    differently.

      raw       the vendor_name text exactly as published
      cleaned   the cleaned name (case, punctuation, legal forms removed)
      entities  the linked vendor entity
      truth     the answer key (synthetic data only)
    """
    r = t["records"].copy()
    r["is_c"] = (r["instrument_type"] != "A").astype(int)
    r = r.sort_values(["contract_key", "is_c", "reference_number"],
                      ascending=[True, False, True])
    first = r.drop_duplicates("contract_key")[["contract_key", "reference_number", "vkey"]]
    vn = t["vendor_names"].set_index("vkey")
    out = pd.DataFrame({"contract_key": first["contract_key"].to_numpy()})
    out["raw"] = vn.loc[first["vkey"], "vendor_name"].to_numpy()
    out["cleaned"] = vn.loc[first["vkey"], "clean"].fillna("").to_numpy()
    ent = t["resolves_to"].set_index("vkey")["entity_id"]
    out["entities"] = ent.reindex(first["vkey"]).to_numpy()
    if truth is not None:
        tk = truth.set_index("reference_number")["company_id"]
        out["truth"] = tk.reindex(first["reference_number"]).to_numpy()
    return out


def vendor_table(contracts: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """One row per vendor under one definition (keys: contract -> vendor key)."""
    c = contracts.assign(vendor=keys.to_numpy())
    g = c.groupby("vendor")
    return pd.DataFrame({
        "n_contracts": g.size(),
        "n_departments": g["department"].nunique(),
        "share_amended": g["n_amendments"].agg(lambda s: float((s > 0).mean())),
        "total_value": g["total_value"].sum(),
    }).reset_index()


def with_vendor_size(contracts: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    c = contracts.copy()
    k = pd.Series(keys.to_numpy(), index=c.index)
    c["vendor_n_contracts"] = k.map(k.value_counts()).astype(float)
    return c


# ---------- summaries ----------

# Attribute pairs that are linked by how they are computed, not by anything the
# data tells us: growth only comes from amendments; a one-contract vendor can only
# have one department; a vendor's total value is a sum over its contracts.
# Sentences pairing them are kept in the full output, flagged, but left out of
# the shortlist.
RELATED = [{"growth", "n_amendments"}, {"n_contracts", "n_departments"},
           {"n_contracts", "total_value"}, {"n_contracts", "share_amended"}]


def _related(a: str, b: str) -> bool:
    return any({a, b} == pair for pair in RELATED)


def proportion(mu_r: np.ndarray | None, mu_s: np.ndarray) -> float:
    """r for Type I (mu_r None) or Type II, as sigma-counts."""
    if mu_r is None:
        return float(mu_s.mean()) if len(mu_s) else float("nan")
    den = float(mu_r.sum())
    return float(np.minimum(mu_r, mu_s).sum() / den) if den > 0 else float("nan")


def truth(q: Quantifier, mu_r, mu_s) -> float:
    return float(q(proportion(mu_r, mu_s)))


def best_quantifier(r: float, quantifiers=QUANTIFIERS) -> tuple[Quantifier, float]:
    """The word that fits r best; ties go to the more specific word."""
    scored = [(float(q(r)), -q.width, q) for q in quantifiers]
    t, _, q = max(scored, key=lambda x: (x[0], x[1]))
    return q, t


def _memberships(table: pd.DataFrame, variables: list[Variable]) -> dict:
    mu = {}
    for var in variables:
        for lab in var.labels:
            if var.attribute == "department":
                d = lab.name.split("=", 1)[1]
                mu[lab.name] = (table["department"] == d).to_numpy(dtype=float)
            else:
                mu[lab.name] = lab.mf(table[var.attribute].to_numpy(dtype=float))
    return mu


def summarize(table: pd.DataFrame, variables: list[Variable], subject: str,
              min_group: float = 10.0, min_support: float = 0.03) -> pd.DataFrame:
    """All Type I and Type II summaries over one object table.

    subject: plural noun for Type I sentences ("contracts", "vendors").
    min_group / min_support: skip qualifiers whose sigma-count is under min_group
    objects or under min_support of the table, so a sentence is never about a
    handful of rows.
    """
    n = len(table)
    mu = _memberships(table, variables)
    attr_of = {lab.name: var for var in variables for lab in var.labels}
    claims = [lab for var in variables if not var.qualifier_only for lab in var.labels]
    groups = [None] + [lab for var in variables for lab in var.labels]

    rows = []
    for R in groups:
        mu_r = None if R is None else mu[R.name]
        if R is not None and (mu_r.sum() < min_group or mu_r.sum() / n < min_support):
            continue
        for S in claims:
            if R is not None and attr_of[R.name].attribute == attr_of[S.name].attribute:
                continue
            mu_s = mu[S.name]
            r = proportion(mu_r, mu_s)
            if np.isnan(r):
                continue
            q, t = best_quantifier(r)
            base = float(mu_s.mean())
            who = subject if R is None else R.as_group
            rows.append({
                "sentence": f"{q.word[0].upper() + q.word[1:]} {who} {S.as_claim}.",
                "type": "I" if R is None else "II",
                "quantifier": q.word, "qualifier": "" if R is None else R.name,
                "summarizer": S.name, "truth": round(t, 4), "r": round(r, 4),
                "base_rate": round(base, 4), "contrast": round(r - base, 4),
                "group_size": round(n if R is None else float(mu_r.sum()), 2),
                "group_share": round(1.0 if R is None else float(mu_r.sum()) / n, 4),
                # Kacprzyk-Yager degree of imprecision, single summarizer:
                # 1 - share of objects the summarizer touches at all. Higher = sharper claim.
                "specificity": round(1.0 - float((mu_s > 0).mean()), 4),
                "related_by_definition": R is not None and _related(
                    attr_of[R.name].attribute, attr_of[S.name].attribute),
            })
    return pd.DataFrame(rows)


def top_summaries(s: pd.DataFrame, min_truth: float = 0.8, k: int = 15) -> pd.DataFrame:
    """Readable shortlist: true sentences about a group that differs from the whole."""
    x = s[(s["type"] == "II") & (s["truth"] >= min_truth)
          & ~s["related_by_definition"]].copy()
    x["abs_contrast"] = x["contrast"].abs()
    return x.sort_values(["abs_contrast", "truth"], ascending=False).head(k) \
            .drop(columns="abs_contrast")


# ---------- does entity linking change what the data says? ----------

@dataclass
class DefinitionRun:
    name: str
    contract_summaries: pd.DataFrame
    vendor_summaries: pd.DataFrame
    n_vendors: int
    extra: dict = field(default_factory=dict)


def run_definitions(t: dict[str, pd.DataFrame], truth: pd.DataFrame | None = None,
                    reference: str = "entities", **kw) -> tuple[list[DefinitionRun], dict]:
    """Compute every summary under each vendor definition.

    The fuzzy sets are fitted once on the contract table, which does not depend on
    the vendor definition, so every definition is judged with the same words.
    """
    contracts = contract_table(t)
    keys = contract_vendor_keys(t, truth)
    cvars = contract_variables(contracts) + [department_variable(contracts["department"].unique())]
    vvars = vendor_variables(contracts)
    defs = ["raw", "cleaned", "entities"] + (["truth"] if truth is not None else [])
    runs = []
    for d in defs:
        k = keys[d].fillna("?" + keys["contract_key"])  # unknown -> its own vendor
        c = with_vendor_size(contracts, k)
        v = vendor_table(contracts, k)
        runs.append(DefinitionRun(
            d, summarize(c, cvars, "contracts", **kw), summarize(v, vvars, "vendors", **kw),
            n_vendors=len(v)))
    params = {
        "quantifiers": {q.word: {"xs": q.mf.xs, "ys": q.mf.ys} for q in QUANTIFIERS},
        "contract_sets": {lab.name: {"attribute": var.attribute, "xs": lab.mf.xs,
                                     "ys": lab.mf.ys}
                          for var in cvars if var.attribute != "department"
                          for lab in var.labels},
        "vendor_sets": {lab.name: {"attribute": var.attribute, "xs": lab.mf.xs, "ys": lab.mf.ys}
                        for var in vvars for lab in var.labels},
    }
    return runs, params


def compare_definitions(runs: list[DefinitionRun], target: str) -> pd.DataFrame:
    """How far each definition's summaries are from the target definition's.

    Only summaries that involve vendors can differ: contract summaries with a
    vendor qualifier or summarizer, and all vendor-level summaries. For each such
    summary (same qualifier and summarizer) we compare the proportion r and ask
    whether the sentence uses a different quantifier word.
    """
    by = {x.name: x for x in runs}
    tgt = by[target]
    out = []
    for x in runs:
        if x.name == target:
            continue
        for level, a, b in [("contract", x.contract_summaries, tgt.contract_summaries),
                            ("vendor", x.vendor_summaries, tgt.vendor_summaries)]:
            key = ["type", "qualifier", "summarizer"]
            m = a.merge(b, on=key, suffixes=("", "_t"))
            if level == "contract":
                m = m[m["qualifier"].str.startswith("vendor=")
                      | m["summarizer"].str.startswith("vendor=")]
            if m.empty:
                continue
            out.append({
                "definition": x.name, "compared_to": target, "level": level,
                "n_summaries": len(m),
                "mean_abs_diff_r": round(float((m["r"] - m["r_t"]).abs().mean()), 4),
                "max_abs_diff_r": round(float((m["r"] - m["r_t"]).abs().max()), 4),
                "share_different_word": round(float((m["quantifier"] != m["quantifier_t"])
                                                    .mean()), 4),
                "n_vendors": x.n_vendors, "n_vendors_target": tgt.n_vendors,
            })
    return pd.DataFrame(out)


def biggest_changes(runs: list[DefinitionRun], a: str, b: str, k: int = 8) -> pd.DataFrame:
    """Vendor-level sentences whose wording changes most between definitions a and b."""
    by = {x.name: x for x in runs}
    m = by[a].vendor_summaries.merge(by[b].vendor_summaries,
                                     on=["type", "qualifier", "summarizer"],
                                     suffixes=(f"_{a}", f"_{b}"))
    m["diff_r"] = (m[f"r_{a}"] - m[f"r_{b}"]).abs()
    m["word_changed"] = m[f"quantifier_{a}"] != m[f"quantifier_{b}"]
    cols = [f"sentence_{a}", f"r_{a}", f"sentence_{b}", f"r_{b}", "diff_r"]
    return m.sort_values(["word_changed", "diff_r"], ascending=False).head(k)[cols] \
            .reset_index(drop=True)
