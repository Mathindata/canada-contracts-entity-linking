"""Synthetic contracts with the real 41-column schema and a known answer key.

For tests and for checking the pipeline end to end. It is NOT evidence about the
real data: every result in the README comes from the real CSV and hand labels.

What makes it hard (on purpose):
  - legal-suffix, punctuation, case, accent and "&"/"AND" variants of one name
  - typos and abbreviations (SERVICES -> SVCS)
  - acronyms (NORTHERN DATA SYSTEMS -> NDS INC)
  - look-alike different companies: "SUMMIT DATA SOLUTIONS" vs "SUMMIT DATA SYSTEMS"
  - the same name used by two different companies in different regions
  - missing or moved postal codes
  - amendment rows that repeat a procurement_id
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "reference_number", "procurement_id", "vendor_name", "vendor_postal_code", "buyer_name",
    "contract_date", "economic_object_code", "description_en", "description_fr",
    "contract_period_start", "delivery_date", "contract_value", "original_value",
    "amendment_value", "comments_en", "comments_fr", "additional_comments_en",
    "additional_comments_fr", "agreement_type_code", "trade_agreement", "land_claims",
    "commodity_type", "commodity_code", "country_of_vendor", "solicitation_procedure",
    "limited_tendering_reason", "trade_agreement_exceptions", "indigenous_business",
    "indigenous_business_excluding_psib", "intellectual_property",
    "potential_commercial_exploitation", "former_public_servant", "contracting_entity",
    "standing_offer_number", "instrument_type", "ministers_office", "number_of_bids",
    "article_6_exceptions", "award_criteria", "socioeconomic_indicator", "reporting_period",
    "owner_org", "owner_org_title",
]

DEPTS = [
    ("hc-sc", "Health Canada | Santé Canada"),
    ("pspc-spac", "Public Services and Procurement Canada | Services publics et "
                  "Approvisionnement Canada"),
    ("eccc", "Environment and Climate Change Canada | Environnement et Changement "
             "climatique Canada"),
    ("nrc-cnrc", "National Research Council Canada | Conseil national de recherches Canada"),
]
FSAS = ["K1A", "K1P", "K2P", "M5V", "M4W", "H3B", "H2X", "T5J", "T6E", "T2P", "V6B", "V5K",
        "R3C", "S4P", "B3J", "E1C", "A1C", "G1R"]
PRE = ["Northern", "Prairie", "Maple", "Aurora", "Summit", "Pinnacle", "Boreal", "Coastal",
       "Granite", "Cedar", "Polar", "Atlas", "Vertex", "Keystone", "Meridian", "Lakeshore",
       "Riverbend", "Frontier", "Evergreen", "Harbour", "Chinook", "Laurentian", "Tundra",
       "Fjord", "Ridgeline", "Bluewater", "Ironwood", "Silverline", "Clearwater", "Redstone"]
CORE = ["Data", "Tech", "Systems", "Consulting", "Analytics", "Engineering", "Logistics",
        "Environmental", "Health", "Software", "Security", "Research", "Translation",
        "Facilities", "Networks", "Geomatics", "Marine", "Energy"]
TAIL = ["Group", "Solutions", "Services", "Partners", "Associates", "Consultants",
        "Technologies", "Systems"]
LEGAL = ["Inc.", "Ltd.", "Corporation", "Ltée", "LLP", "Inc."]
ABBR = {"SERVICES": "SVCS", "TECHNOLOGIES": "TECH", "CONSULTING": "CONSLT",
        "ASSOCIATES": "ASSOC", "ENGINEERING": "ENG", "SOLUTIONS": "SOLNS"}
WORK = ["IT consulting services", "Environmental sampling", "Translation services",
        "Software licences", "Engineering services", "Facility maintenance",
        "Laboratory equipment", "Data analysis services", "Security services",
        "Marine survey"]


def _companies(rng, n):
    comps, seen = [], set()
    while len(comps) < n:
        name = f"{rng.choice(PRE)} {rng.choice(CORE)} {rng.choice(TAIL)}"
        if rng.random() < 0.08:
            name = name.replace(" ", " & ", 1)
        if rng.random() < 0.06:
            name = name.replace("Boreal", "Boréal").replace("Laurentian", "Laurentienne")
        fsa = str(rng.choice(FSAS))
        if (name, fsa[0]) in seen:
            continue
        seen.add((name, fsa[0]))
        comps.append({"name": name, "legal": str(rng.choice(LEGAL)), "fsa": fsa,
                      "acronym_ok": rng.random() < 0.08})
    # look-alikes: same first two words, different last word, often same city
    for _ in range(n // 6):
        base = comps[rng.integers(len(comps))]
        w = base["name"].split()
        alt = " ".join(w[:-1] + [str(rng.choice([t for t in TAIL if t != w[-1]]))])
        comps.append({"name": alt, "legal": str(rng.choice(LEGAL)),
                      "fsa": base["fsa"] if rng.random() < 0.6 else str(rng.choice(FSAS)),
                      "acronym_ok": False})
    # same name, different company, different region
    for _ in range(n // 12):
        base = comps[rng.integers(len(comps))]
        other = [f for f in FSAS if f[0] != base["fsa"][0]]
        comps.append({"name": base["name"], "legal": str(rng.choice(LEGAL)),
                      "fsa": str(rng.choice(other)), "acronym_ok": False})
    return comps


def _typo(rng, s):
    if len(s) < 6:
        return s
    i = int(rng.integers(1, len(s) - 2))
    op = rng.integers(3)
    if op == 0:
        return s[:i] + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def _variant(rng, c):
    words = c["name"].upper().split()
    if c["acronym_ok"] and rng.random() < 0.3:
        core = "".join(w[0] for w in words if w not in {"&", "AND"})
    else:
        if rng.random() < 0.15:
            words = [ABBR.get(w, w) for w in words]
        core = " ".join(words)
        if rng.random() < 0.15:
            core = core.replace(" & ", " AND ") if "&" in core else core.replace(" AND ", " & ")
        if rng.random() < 0.06:
            core = _typo(rng, core)
        if rng.random() < 0.05:
            core = "THE " + core
        if rng.random() < 0.1:
            core = core.replace("É", "E")
    legal = c["legal"].upper()
    r = rng.random()
    if r < 0.15:
        legal = ""
    elif r < 0.3:
        legal = {"INC.": "INC", "LTD.": "LIMITED", "CORPORATION": "CORP.", "LTÉE": "LTEE",
                 "LLP": "L.L.P."}.get(legal, legal)
    sep = ", " if rng.random() < 0.3 else " "
    out = (core + (sep + legal if legal else "")).strip()
    return out.title() if rng.random() < 0.2 else out


def _postal(rng, c):
    r = rng.random()
    if r < 0.82:
        return c["fsa"] + " " + "".join(map(str, rng.integers(0, 9, 1))) + "X" + "1"
    if r < 0.89:
        same = [f for f in FSAS if f[0] == c["fsa"][0] and f != c["fsa"]]
        return (str(rng.choice(same)) if same else c["fsa"]) + " 1A1"
    if r < 0.95:
        return "NA" if rng.random() < 0.5 else ""
    return str(rng.choice(FSAS)) + " 2B2"


def generate(n_companies: int = 300, n_contracts: int = 2500, seed: int = 7
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    comps = _companies(rng, n_companies)
    # each company sticks to 1-3 preferred spellings, like a department's finance system
    prefs = [[_variant(rng, c) for _ in range(rng.integers(1, 4))] for c in comps]
    depts_of = [rng.choice(len(DEPTS) - 1, size=rng.integers(1, 4), replace=False)
                for _ in comps]
    pop = 1.0 / np.arange(1, len(comps) + 1) ** 0.9
    pop = pop[rng.permutation(len(comps))]
    pop /= pop.sum()

    rows, truth, seq = [], [], 0
    for k in range(n_contracts):
        ci = int(rng.choice(len(comps), p=pop))
        c = comps[ci]
        d = int(rng.choice(depts_of[ci])) if rng.random() < 0.97 else 3
        year = int(rng.choice([2021, 2022, 2023, 2024], p=[0.08, 0.3, 0.32, 0.3]))
        start = pd.Timestamp(year, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        end = start + pd.DateOffset(months=int(rng.integers(3, 37)))
        orig = float(np.round(np.exp(rng.normal(11.5, 1.2)) + 10_000, 2))
        pid = f"{DEPTS[d][0].upper()}-{year}-{k:05d}"
        work = str(rng.choice(WORK))
        n_am = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.6, 0.2, 0.08, 0.06, 0.04, 0.02]))
        total = orig
        for j in range(n_am + 1):
            seq += 1
            is_am = j > 0
            amend = float(np.round(orig * rng.uniform(0.05, 0.6), 2)) if is_am else np.nan
            if is_am:
                total += amend
            date = start + pd.DateOffset(months=int(4 * j))
            q = (date.month - 1) // 3 + 1
            fy = date.year if date.month >= 4 else date.year - 1
            name = str(rng.choice(prefs[ci])) if rng.random() < 0.93 else _variant(rng, c)
            row = {col: "" for col in COLUMNS}
            row.update({
                "reference_number": f"C-{fy}-{fy + 1}-Q{q}-{seq:06d}",
                "procurement_id": pid, "vendor_name": name, "vendor_postal_code": _postal(rng, c),
                "buyer_name": "", "contract_date": date.strftime("%Y-%m-%d"),
                "description_en": work, "contract_period_start": start.strftime("%Y-%m-%d"),
                "delivery_date": end.strftime("%Y-%m-%d"), "contract_value": f"{total:.2f}",
                "original_value": f"{orig:.2f}",
                "amendment_value": "" if not is_am else f"{amend:.2f}",
                "commodity_type": "S", "country_of_vendor": "CA",
                "instrument_type": "A" if is_am else "C",
                "reporting_period": f"{fy}-{fy + 1}-Q{q}",
                "owner_org": DEPTS[d][0], "owner_org_title": DEPTS[d][1],
            })
            rows.append(row)
            truth.append({"reference_number": row["reference_number"], "company_id": f"CO{ci:04d}"})
    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(truth)


def vendor_string_truth(rows: pd.DataFrame, truth: pd.DataFrame) -> dict[str, str]:
    """Map each vendor string key to its company (majority vote if a string is shared)."""
    m = rows[["reference_number", "vkey"]].merge(truth, on="reference_number")
    return m.groupby("vkey")["company_id"].agg(lambda s: s.value_counts().index[0]).to_dict()
