"""End-to-end run on synthetic data with a known answer key."""
import json

import numpy as np
import pandas as pd
import pytest

from cel import cli, graph, synthetic
from cel.config import Config


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    raw, truth = d / "contracts.csv", d / "truth.csv"
    df, t = synthetic.generate(seed=7)
    df.to_csv(raw, index=False)
    t.to_csv(truth, index=False)
    cfg = Config(raw_csv=str(raw), work_dir=str(d / "work"), out_dir=str(d / "out"),
                 n_bootstrap=100)
    labels = str(d / "labels")
    cli.step_prepare(cfg)
    cli.step_link(cfg)
    cli.step_label_sheet(cfg, labels, force=False)
    cli.step_oracle_label(cfg, labels, str(truth))
    cli.step_evaluate(cfg, labels)
    cli.step_cluster(cfg)
    cli.step_export_graph(cfg)
    return cfg, labels, truth


def test_slice_filters_years_and_departments(run):
    cfg, *_ = run
    s = json.load(open(cfg.out / "slice_summary.json"))
    assert len(s["departments"]) == 3
    assert "National Research Council Canada" not in s["departments"]


def test_linking_quality_against_truth(run):
    cfg, labels, truth = run
    rows = cli._read_rows(cfg)
    t = synthetic.vendor_string_truth(rows, pd.read_csv(truth))
    X = pd.read_csv(cfg.work / "candidates_scored.csv")
    y = np.array([t[a] == t[b] for a, b in zip(X["a"], X["b"])])
    pred = X["prob"].to_numpy() >= cfg.match_threshold
    assert (pred & y).sum() / pred.sum() > 0.8      # precision
    assert (pred & y).sum() / y.sum() > 0.85        # recall within candidates
    s = json.load(open(cfg.out / "linking_summary.json"))
    assert s["reduction_ratio"] > 0.95


def test_label_sheet_hides_scores_and_is_protected(run):
    cfg, labels, _ = run
    sheet = pd.read_csv(f"{labels}/labeling_sheet.csv")
    assert not {"prob", "jw", "stratum", "weight"} & set(sheet.columns)
    with pytest.raises(SystemExit):
        cli.step_label_sheet(cfg, labels, force=False)


def test_weights_reproduce_population(run):
    cfg, labels, _ = run
    key = pd.read_csv(f"{labels}/label_key.csv")
    cand = len(pd.read_csv(cfg.work / "candidates_scored.csv"))
    near = len(pd.read_csv(cfg.work / "near_miss_scored.csv"))
    assert np.isclose(key["weight"].sum(), cand + near)


def test_evaluation_outputs(run):
    cfg, *_ = run
    m = pd.read_csv(cfg.out / "metrics.csv")
    assert len(m) == 3 and m["precision"].between(0, 1).all()
    for f in ("pr_curve.png", "reliability.png", "evaluation_summary.json"):
        assert (cfg.out / f).exists()


def test_graph_tables_consistent(run):
    cfg, *_ = run
    t = graph.read_tables(cfg.work / "graph")
    assert set(t["records"]["contract_key"]) <= set(t["contracts"]["contract_key"])
    assert set(t["records"]["vkey"]) <= set(t["vendor_names"]["vkey"])
    assert set(t["resolves_to"]["vkey"]) == set(t["vendor_names"]["vkey"])
    assert set(t["resolves_to"]["entity_id"]) == set(t["vendors"]["entity_id"])
    assert t["resolves_to"]["prob"].between(0, 1).all()
    assert (t["contracts"]["n_amendments"] <= t["contracts"]["n_records"] - 0).all()
