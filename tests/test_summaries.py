"""Linguistic summaries: the fuzzy pieces, the truth formulas, and an end-to-end check."""
import numpy as np
import pandas as pd
import pytest

from cel import cli, graph, summaries as S, synthetic
from cel.config import Config


def q(word):
    return next(x for x in S.QUANTIFIERS if x.word == word)


def test_membership_shapes():
    f = S.falling(2, 4)
    assert list(f([1, 2, 3, 4, 5])) == [1, 1, 0.5, 0, 0]
    g = S.rising(2, 4)
    assert list(g([1, 3, 9])) == [0, 0.5, 1]
    assert S.trapezoid(0, 1, 2, 3)(np.nan) == 0          # missing value: no membership


def test_most_follows_standard_definition():
    most = q("most")
    assert most(0.3) == 0 and most(0.8) == 1 and abs(most(0.55) - 0.5) < 1e-12


def test_type_one_and_two_truth_by_hand():
    # four objects; S memberships 1, 1, 0.5, 0
    mu_s = np.array([1.0, 1.0, 0.5, 0.0])
    assert S.proportion(None, mu_s) == pytest.approx(2.5 / 4)
    # R picks the first three, the third only half: sum min = 1 + 0 + 0.5, sum R = 2.5
    mu_r = np.array([1.0, 0.0, 1.0, 0.5])
    assert S.proportion(mu_r, mu_s) == pytest.approx((1 + 0 + 0.5 + 0) / 2.5)
    assert S.truth(q("most"), mu_r, mu_s) == pytest.approx(q("most")(0.6))


def test_best_quantifier_prefers_specific_word_on_ties():
    word, t = S.best_quantifier(0.97)
    assert word.word == "almost all" and t == 1.0     # "most" is also 1, but vaguer
    assert S.best_quantifier(0.02)[0].word == "almost none of the"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("sum")
    df, truth = synthetic.generate(seed=11)
    df.to_csv(d / "c.csv", index=False)
    cfg = Config(raw_csv=str(d / "c.csv"), work_dir=str(d / "w"), out_dir=str(d / "o"))
    for step in (cli.step_prepare, cli.step_link, cli.step_cluster, cli.step_export_graph):
        step(cfg)
    return cfg, graph.read_tables(cfg.work / "graph"), truth


def test_summaries_are_well_formed(synth):
    _, t, truth = synth
    runs, params = S.run_definitions(t, truth)
    ent = next(x for x in runs if x.name == "entities")
    c = ent.contract_summaries
    assert c["truth"].between(0, 1).all() and c["r"].between(0, 1).all()
    assert (c["type"] == "I").sum() == 11             # one per summarizer label
    assert c["sentence"].str.endswith(".").all()
    # a sentence never pairs a qualifier and summarizer from the same attribute
    assert not (c["qualifier"].str.split("=").str[0] == c["summarizer"].str.split("=").str[0]).any()
    # shortlist leaves out pairs that are linked by definition
    assert not S.top_summaries(c)["related_by_definition"].any()
    assert "most" in params["quantifiers"]


def test_linking_brings_vendor_summaries_closer_to_truth(synth):
    """On synthetic data with a known answer key: vendor-level sentences built on
    linked entities are closer to the truth than sentences built on raw strings."""
    _, t, truth = synth
    runs, _ = S.run_definitions(t, truth)
    comp = S.compare_definitions(runs, "truth").set_index(["definition", "level"])
    raw = comp.loc[("raw", "vendor"), "mean_abs_diff_r"]
    ent = comp.loc[("entities", "vendor"), "mean_abs_diff_r"]
    assert ent < raw
