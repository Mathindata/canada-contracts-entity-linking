"""Run settings. Edit config.json (or pass --config) instead of editing code."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Input: the open.canada.ca "Contracts over $10,000" CSV (contracts.csv, ~655 MB).
    raw_csv: str = "data/raw/contracts.csv"
    encoding: str = "utf-8-sig"
    work_dir: str = "data/work"
    out_dir: str = "results"

    # Slice. Empty list = take the top_n_departments with the most rows in the year range.
    departments: list[str] = field(default_factory=list)
    top_n_departments: int = 3
    year_min: int = 2022
    year_max: int = 2024
    # Column holding the department. "auto" tries owner_org_title, owner_org, then buyer_name.
    dept_column: str = "auto"

    # Linking
    seed: int = 20261121
    match_threshold: float = 0.5        # accept a link when P(match) >= this
    jw_baseline_threshold: float = 0.92  # fixed in advance for the string-similarity baseline
    u_random_pairs: int = 200_000       # random pairs used to estimate u (non-match) probabilities
    max_block_size: int = 3000          # skip blocks larger than this (and report them)

    # Labeling sheet
    label_per_stratum: int = 25
    n_strata: int = 10
    near_miss_window: int = 5           # sorted-neighbourhood window for the near-miss stratum
    near_miss_jw: float = 0.85
    near_miss_label: int = 40

    # Evaluation
    n_bootstrap: int = 1000

    @classmethod
    def load(cls, path: str | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def work(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
