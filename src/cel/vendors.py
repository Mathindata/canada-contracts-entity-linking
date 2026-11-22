"""Collapse rows into unique vendor strings.

Many rows repeat the exact same (vendor_name, postal prefix). Linking is done on
these unique strings, then every row inherits the result. This keeps the number
of comparisons small and keeps a clean trail from each decision to its rows.
"""
from __future__ import annotations

import pandas as pd

from .normalize import clean_name, clean_postal, initials


def build_vendor_strings(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["fsa"] = rows["vendor_postal_code"].map(clean_postal)
    rows["vendor_string"] = rows["vendor_name"] + "||" + rows["fsa"]
    g = rows.groupby("vendor_string", sort=True)
    v = pd.DataFrame({
        "vendor_name": g["vendor_name"].first(),
        "fsa": g["fsa"].first(),
        "n_rows": g.size(),
        "departments": g["department"].agg(lambda s: "; ".join(sorted(set(s)))),
        "example_description": g["description_en"].first() if "description_en" in rows
        else "",
    }).reset_index()
    v["vkey"] = [f"N{i:06d}" for i in range(len(v))]
    v["clean"] = v["vendor_name"].map(clean_name)
    v["initials"] = v["clean"].map(initials)
    v["tok0"] = v["clean"].str.split(" ").str[0].fillna("")
    v["clean_nospace"] = v["clean"].str.replace(" ", "", regex=False)
    rows["vkey"] = rows["vendor_string"].map(dict(zip(v["vendor_string"], v["vkey"])))
    return rows, v.set_index("vkey", drop=False)
