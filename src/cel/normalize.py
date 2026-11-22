"""Vendor name cleaning (English and French).

The goal is a canonical string that keeps what identifies a company and drops
what only describes its legal form or formatting.
"""
from __future__ import annotations

import re
import unicodedata

# Legal-form words, English and French, removed from the end of a name.
LEGAL = {
    "INC", "INCORPORATED", "INCORPOREE", "LTD", "LIMITED", "LTEE", "LIMITEE", "CORP",
    "CORPORATION", "CO", "COMPANY", "CIE", "LLC", "LLP", "LP", "ULC", "PLC", "SENCRL",
    "SRL", "SEC", "SENC", "GMBH", "AG", "SA", "SAS", "BV", "NV", "PTY", "INCORPORATION",
}
# Words skipped when building initials (acronyms).
STOP = {"AND", "OF", "THE", "DE", "DU", "DES", "LA", "LE", "LES", "ET", "D", "L", "FOR", "POUR"}

_punct = re.compile(r"[^A-Z0-9 ]+")
_space = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _join_single_letters(tokens: list[str]) -> list[str]:
    """'I B M' -> 'IBM', 'S E N C R L' -> 'SENCRL' (after punctuation removal)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def clean_name(name: str) -> str:
    s = strip_accents(str(name)).upper()
    s = s.replace("&", " AND ").replace("+", " AND ")
    s = s.replace("'", "")            # O'BRIEN -> OBRIEN, L'AGENCE -> LAGENCE
    s = _punct.sub(" ", s)
    toks = _join_single_letters(_space.sub(" ", s).strip().split(" "))
    toks = [t for t in toks if t]
    if toks and toks[0] == "THE":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in LEGAL:
        toks = toks[:-1]
    return " ".join(toks)


def initials(clean: str) -> str:
    toks = [t for t in clean.split() if t not in STOP]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def clean_postal(p) -> str:
    """First 3 characters of a Canadian postal code (the FSA). '' when missing or foreign."""
    s = re.sub(r"[^A-Z0-9]", "", str(p or "").upper())
    if s in ("", "NA", "NAN", "NONE") or len(s) < 3:
        return ""
    s = s[:3]
    return s if re.fullmatch(r"[A-Z][0-9][A-Z]", s) else ""
