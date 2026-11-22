"""Read a slice of the federal contracts CSV.

Field names follow the open.canada.ca data dictionary for
"Proactive Publication - Contracts over $10,000". The file is large (~655 MB),
so it is read in chunks with only the needed columns.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

KEEP = [
    "reference_number", "procurement_id", "vendor_name", "vendor_postal_code",
    "buyer_name", "contract_date", "contract_period_start", "delivery_date",
    "contract_value", "original_value", "amendment_value", "instrument_type",
    "description_en", "commodity_type", "country_of_vendor", "reporting_period",
]
DEPT_CANDIDATES = ["owner_org_title", "owner_org", "buyer_name"]
CHUNK = 200_000


def _header(path: str, encoding: str) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding=encoding).columns)


def pick_dept_column(cfg: Config, columns: list[str]) -> str:
    if cfg.dept_column != "auto":
        if cfg.dept_column not in columns:
            raise ValueError(f"dept_column {cfg.dept_column!r} not in CSV header")
        return cfg.dept_column
    for c in DEPT_CANDIDATES:
        if c in columns:
            return c
    raise ValueError("No department column found; set dept_column in config.json")


def _clean_dept(s: pd.Series) -> pd.Series:
    # owner_org_title is bilingual, e.g. "Health Canada | Santé Canada"
    return s.fillna("").astype(str).str.split("|").str[0].str.strip()


def _year(df: pd.DataFrame) -> pd.Series:
    y = pd.to_datetime(df["contract_date"], errors="coerce").dt.year
    # fallback: first year of reporting_period "2023-2024-Q1"
    fb = pd.to_numeric(df.get("reporting_period", pd.Series(index=df.index, dtype=str))
                       .astype(str).str[:4], errors="coerce")
    return y.fillna(fb)


def load_slice(cfg: Config) -> pd.DataFrame:
    cols = _header(cfg.raw_csv, cfg.encoding)
    dept_col = pick_dept_column(cfg, cols)
    use = [c for c in KEEP if c in cols] + ([dept_col] if dept_col not in KEEP else [])
    missing = [c for c in ("reference_number", "procurement_id", "vendor_name",
                           "vendor_postal_code", "contract_date") if c not in cols]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    log.info("department column: %s", dept_col)

    depts = set(cfg.departments)
    if not depts:
        counts: dict[str, int] = {}
        for ch in pd.read_csv(cfg.raw_csv, usecols=[dept_col, "contract_date", "reporting_period"]
                              if "reporting_period" in cols else [dept_col, "contract_date"],
                              dtype=str, chunksize=CHUNK, encoding=cfg.encoding):
            yr = _year(ch)
            ch = ch[(yr >= cfg.year_min) & (yr <= cfg.year_max)]
            for k, v in _clean_dept(ch[dept_col]).value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[: cfg.top_n_departments]
        depts = {k for k, _ in top}
        log.info("top departments: %s", top)

    parts = []
    for ch in pd.read_csv(cfg.raw_csv, usecols=use, dtype=str, chunksize=CHUNK,
                          encoding=cfg.encoding):
        ch["department"] = _clean_dept(ch[dept_col])
        yr = _year(ch)
        keep = ch["department"].isin(depts) & (yr >= cfg.year_min) & (yr <= cfg.year_max)
        if keep.any():
            parts.append(ch[keep])
    if not parts:
        raise ValueError("Slice is empty: check departments and years")
    df = pd.concat(parts, ignore_index=True)
    return tidy(df)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vendor_name"] = df["vendor_name"].fillna("").astype(str).str.strip()
    df = df[df["vendor_name"] != ""]
    df["procurement_id"] = df["procurement_id"].fillna("").astype(str).str.strip()
    # rows without a procurement id get their own contract key
    no_pid = df["procurement_id"] == ""
    df.loc[no_pid, "procurement_id"] = "NOPID-" + df.loc[no_pid, "reference_number"].astype(str)
    for c in ("contract_value", "original_value", "amendment_value"):
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,\s]", "", regex=True),
                                  errors="coerce")
    df["instrument_type"] = df.get("instrument_type", "").fillna("").astype(str).str.upper()
    df["contract_key"] = df["department"] + "|" + df["procurement_id"]
    df = df.drop_duplicates("reference_number")
    return df.reset_index(drop=True)
