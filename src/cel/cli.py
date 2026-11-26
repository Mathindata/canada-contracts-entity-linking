"""Command line: python -m cel <step> [--config config.json]

Steps, in order:
  prepare       read the slice of contracts.csv, build unique vendor strings
  link          blocking, comparisons, Fellegi-Sunter (EM), score candidate pairs
  label-sheet   write labels/labeling_sheet.csv for hand labeling (never overwrites labels)
  label         label the sheet interactively in the terminal (resume-safe)
  evaluate      metrics with bootstrap CIs, PR curve, calibration plot
  cluster       vendor entities from accepted links, plus a chaining audit
  export-graph  node and edge CSVs for Neo4j
  load-neo4j    push the graph to Neo4j (reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
  run-queries   run queries/*.cypher against Neo4j, save CSVs
  summarize     linguistic summaries ("Most X are Y") with truth degrees, under several
                vendor definitions (raw strings, cleaned names, linked entities)

Synthetic data (tests and demo only):
  make-synthetic   write a fake contracts.csv with a known answer key
  oracle-label     fill the label sheet from the answer key
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, cluster, compare, evaluate, graph, labeling, summaries, synthetic
from .config import Config
from .data import load_slice
from .fellegi_sunter import FellegiSunter
from .vendors import build_vendor_strings

log = logging.getLogger("cel")


# ---------- helpers ----------
def _read_v(cfg: Config) -> pd.DataFrame:
    v = pd.read_csv(cfg.work / "vendor_strings.csv", dtype=str, keep_default_na=False)
    v["n_rows"] = v["n_rows"].astype(int)
    return v.set_index("vkey", drop=False)


def _read_rows(cfg: Config) -> pd.DataFrame:
    r = pd.read_csv(cfg.work / "rows.csv.gz", dtype=str, keep_default_na=False)
    for c in ("contract_value", "original_value", "amendment_value"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    return r


def _save_model(fs: FellegiSunter, path: Path) -> None:
    json.dump({"lam": fs.lam, "m": {k: v.tolist() for k, v in fs.m.items()},
               "u": {k: v.tolist() for k, v in fs.u.items()}, "L": fs.L},
              open(path, "w"), indent=2)


def _load_model(path: Path) -> FellegiSunter:
    d = json.load(open(path))
    fs = FellegiSunter(d["L"])
    fs.lam = d["lam"]
    fs.m = {k: np.array(v) for k, v in d["m"].items()}
    fs.u = {k: np.array(v) for k, v in d["u"].items()}
    return fs


def train_model(v: pd.DataFrame, X_cand: pd.DataFrame, idf: dict, cfg: Config) -> FellegiSunter:
    """u from random pairs; m from two EM sessions; lambda over all pairs. No labels."""
    fs = FellegiSunter(compare.N_LEVELS)
    fs.fit_u(compare.compare(compare.random_pairs(v, cfg.u_random_pairs, cfg.seed), v, idf))
    # Session 1: pairs sharing a postal prefix -> m for the name-based fields
    s1 = compare.compare(blocking.training_pairs(v, "fsa", cfg.u_random_pairs, cfg.seed + 1),
                         v, idf)
    fs.fit_m_session(s1, ["name", "tokens", "acronym"])
    # Session 2: pairs sharing a first word, any region -> m for postal, anchored by names
    s2 = compare.compare(blocking.training_pairs(v, "tok0", cfg.u_random_pairs, cfg.seed + 2),
                         v, idf)
    fs.fit_m_session(s2, ["postal"], use_fields=compare.FIELDS)
    n_all = len(v) * (len(v) - 1) / 2
    fs.fit_lambda(X_cand, n_all)
    return fs


# ---------- steps ----------
def step_prepare(cfg: Config) -> None:
    rows = load_slice(cfg)
    rows, v = build_vendor_strings(rows)
    rows.to_csv(cfg.work / "rows.csv.gz", index=False)
    v.to_csv(cfg.work / "vendor_strings.csv", index=False)
    info = {"rows": len(rows), "contracts": int(rows["contract_key"].nunique()),
            "vendor_strings": len(v), "departments": sorted(rows["department"].unique()),
            "years": [cfg.year_min, cfg.year_max]}
    json.dump(info, open(cfg.out / "slice_summary.json", "w"), indent=2)
    log.info("prepare: %s", info)


def step_link(cfg: Config) -> None:
    v = _read_v(cfg)
    cand, skipped = blocking.candidate_pairs(v, cfg.max_block_size)
    idf = compare.token_idf(v["clean"])
    X = compare.compare(cand, v, idf)
    fs = train_model(v, X, idf, cfg)
    X["prob"] = fs.predict_proba(X)
    X["log2_bf"] = fs.log2_bf(X)
    X = X.merge(cand, on=["a", "b"])
    X.to_csv(cfg.work / "candidates_scored.csv", index=False)

    near = blocking.near_miss_pairs(v, cand, cfg.near_miss_window, cfg.near_miss_jw)
    Xn = compare.compare(near, v, idf) if len(near) else near.assign(prob=[])
    if len(near):
        Xn["prob"] = fs.predict_proba(Xn)
    Xn.to_csv(cfg.work / "near_miss_scored.csv", index=False)

    _save_model(fs, cfg.work / "fs_model.json")
    fs.weights_table(compare.LEVEL_LABELS).to_csv(cfg.out / "match_weights.csv", index=False)
    summary = {
        "vendor_strings": len(v), "candidate_pairs": len(X),
        "reduction_ratio": blocking.reduction_ratio(len(v), len(X)),
        "pairs_by_rule": {c: int(X[c].sum()) for c in X.columns if c.startswith("rule_")},
        "near_miss_pairs": len(Xn), "skipped_blocks": len(skipped),
        "prior_lambda_all_pairs": fs.lam,
        "expected_matches_in_candidates": float(X["prob"].sum()),
        "training_sessions": fs.sessions,
        "accepted_links": int((X["prob"] >= cfg.match_threshold).sum()),
    }
    json.dump(summary, open(cfg.out / "linking_summary.json", "w"), indent=2)
    log.info("link: %s", summary)


def step_label_sheet(cfg: Config, labels_dir: str, force: bool) -> None:
    out = Path(labels_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet_path = out / "labeling_sheet.csv"
    if sheet_path.exists() and not force:
        existing = pd.read_csv(sheet_path, dtype=str).fillna("")
        if (existing["label"].str.strip() != "").any():
            sys.exit(f"{sheet_path} already has labels. Refusing to overwrite (use --force).")
    v = _read_v(cfg)
    cand = pd.read_csv(cfg.work / "candidates_scored.csv")
    near = pd.read_csv(cfg.work / "near_miss_scored.csv")
    sheet, key = labeling.make_sheet(cand, near, v, cfg.label_per_stratum, cfg.near_miss_label,
                                     cfg.seed, cfg.n_strata)
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(out / "label_key.csv", index=False)
    log.info("label sheet: %d pairs -> %s", len(sheet), sheet_path)


def step_label_interactive(labels_dir: str) -> None:
    """Label pairs one by one in the terminal. Saves after every answer, so you can stop
    and resume any time. Pairs that already have a label are skipped."""
    path = Path(labels_dir) / "labeling_sheet.csv"
    sheet = pd.read_csv(path, dtype=str).fillna("")
    todo = [i for i in sheet.index if sheet.at[i, "label"].strip() == ""]
    print(f"{len(todo)} pairs left of {len(sheet)}. Answers: y = same vendor, n = different, "
          "u = unsure, q = quit. See labels/README.md for the rules.")
    for k, i in enumerate(todo, 1):
        r = sheet.loc[i]
        print(f"\n[{k}/{len(todo)}] {r['pair_id']}")
        print(f"  A: {r['name_a']}  ({r['postal_a'] or 'no postal'})  | {r['departments_a']}"
              f"  | {r['example_work_a']}")
        print(f"  B: {r['name_b']}  ({r['postal_b'] or 'no postal'})  | {r['departments_b']}"
              f"  | {r['example_work_b']}")
        while True:
            ans = input("  same vendor? [y/n/u/q]: ").strip().lower()
            if ans in {"y", "n", "u", "q"}:
                break
        if ans == "q":
            break
        sheet.at[i, "label"] = {"y": "1", "n": "0", "u": "?"}[ans]
        sheet.to_csv(path, index=False)
    done = (sheet["label"].str.strip() != "").sum()
    print(f"\nSaved. {done}/{len(sheet)} pairs answered.")


def step_oracle_label(cfg: Config, labels_dir: str, truth_csv: str) -> None:
    rows = _read_rows(cfg)
    truth = synthetic.vendor_string_truth(rows, pd.read_csv(truth_csv))
    out = Path(labels_dir)
    key = pd.read_csv(out / "label_key.csv")
    lab = labeling.oracle_labels(key, truth)
    sheet = pd.read_csv(out / "labeling_sheet.csv", dtype=str).fillna("")
    sheet["label"] = sheet["pair_id"].map(dict(zip(lab["pair_id"], lab["label"])))
    sheet["note"] = "oracle (synthetic)"
    sheet.to_csv(out / "labeling_sheet.csv", index=False)


def _labeled_frame(cfg: Config, labels_dir: str) -> pd.DataFrame:
    out = Path(labels_dir)
    lab = labeling.read_labels(out / "labeling_sheet.csv", out / "label_key.csv")
    cand = pd.read_csv(cfg.work / "candidates_scored.csv")
    near = pd.read_csv(cfg.work / "near_miss_scored.csv")
    cols = ["a", "b", "jw", "exact_clean"]
    feats = pd.concat([cand[cols], near[cols]]).drop_duplicates(["a", "b"])
    lab = lab.merge(feats, on=["a", "b"], how="left")
    return lab


def step_evaluate(cfg: Config, labels_dir: str) -> None:
    lab = _labeled_frame(cfg, labels_dir)
    n_all = len(lab)
    unsure = int(lab["y"].isna().sum())
    lab = lab[lab["y"].notna()].reset_index(drop=True)
    if lab.empty:
        sys.exit("No labels yet: fill the 'label' column in labels/labeling_sheet.csv")
    table = evaluate.metrics_table(lab, cfg)
    table.to_csv(cfg.out / "metrics.csv", index=False)
    extra = evaluate.plot_all(lab, cfg, cfg.out)
    summary = {"pairs_in_sheet": n_all, "labeled": len(lab), "unsure_or_blank": unsure,
               "labeled_matches": int(lab["y"].sum()),
               "estimated_true_matches": float((lab["weight"] * lab["y"]).sum()),
               "blocking_loss_share": evaluate.blocking_loss(lab), **extra}
    json.dump(summary, open(cfg.out / "evaluation_summary.json", "w"), indent=2)
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


def step_cluster(cfg: Config) -> None:
    v = _read_v(cfg)
    cand = pd.read_csv(cfg.work / "candidates_scored.csv")
    fs = _load_model(cfg.work / "fs_model.json")
    ent, acc = cluster.build_entities(v, cand, cfg.match_threshold)
    ent.to_csv(cfg.work / "entities.csv", index=False)
    acc[["a", "b", "prob"]].to_csv(cfg.work / "accepted_links.csv", index=False)
    audit = cluster.audit_entities(ent, v, fs, compare.token_idf(v["clean"]))
    audit.to_csv(cfg.out / "cluster_audit.csv", index=False)
    sizes = ent.groupby("entity_id").size()
    summary = {"vendor_strings": len(v), "entities": int(sizes.size),
               "multi_name_entities": int((sizes > 1).sum()), "largest_entity": int(sizes.max()),
               "weak_pairs_inside_entities": len(audit)}
    json.dump(summary, open(cfg.out / "cluster_summary.json", "w"), indent=2)
    log.info("cluster: %s", summary)


def step_export_graph(cfg: Config) -> None:
    rows, v = _read_rows(cfg), _read_v(cfg)
    ent = pd.read_csv(cfg.work / "entities.csv")
    acc = pd.read_csv(cfg.work / "accepted_links.csv")
    t = graph.build_tables(rows, v, ent, acc)
    graph.write_tables(t, cfg.work / "graph")
    log.info("graph: %s", {k: len(x) for k, x in t.items()})


def step_load_neo4j(cfg: Config, wipe: bool) -> None:
    from .neo4j_io import connect_from_env, load_graph
    t = graph.read_tables(cfg.work / "graph")
    with connect_from_env() as drv:
        counts = load_graph(drv, t, wipe=wipe)
    log.info("neo4j: %s", counts)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x)
                                        for x in r) + " |")
    return "\n".join(lines)


def step_summarize(cfg: Config, truth_csv: str | None) -> None:
    t = graph.read_tables(cfg.work / "graph")
    truth = pd.read_csv(truth_csv) if truth_csv else None
    runs, params = summaries.run_definitions(t, truth)
    by = {x.name: x for x in runs}
    ent = by["entities"]
    ent.contract_summaries.to_csv(cfg.out / "summaries_contracts.csv", index=False)
    ent.vendor_summaries.to_csv(cfg.out / "summaries_vendors.csv", index=False)
    target = "truth" if truth is not None else "entities"
    comp = summaries.compare_definitions(runs, target)
    comp.to_csv(cfg.out / "summaries_definitions.csv", index=False)
    json.dump(params, open(cfg.out / "summaries_params.json", "w"), indent=2)

    show = ["sentence", "truth", "r", "base_rate", "group_size"]
    other = "raw"
    ch = summaries.biggest_changes(runs, other, target, k=5)
    md = ["# Linguistic summaries", "",
          "Truth T in [0, 1]; r = share of the group with the property (sigma-counts); "
          "base_rate = the same share over the whole table. Vendors = linked entities.", "",
          "## Contracts: whole table (Type I)", "",
          _md_table(ent.contract_summaries.query("type == 'I'")[show]), "",
          "## Contracts: groups that differ from the whole (Type II, T >= 0.8)", "",
          _md_table(summaries.top_summaries(ent.contract_summaries)[show]), "",
          "## Vendors: whole table (Type I)", "",
          _md_table(ent.vendor_summaries.query("type == 'I'")[show]), "",
          "## Vendors: groups that differ from the whole (Type II, T >= 0.8)", "",
          _md_table(summaries.top_summaries(ent.vendor_summaries, k=10)[show]), "",
          f"## Does the vendor definition change the sentences? (compared to {target})", "",
          _md_table(comp), "",
          f"## Largest changes, {other} strings vs {target}", "",
          _md_table(ch), ""]
    (cfg.out / "summaries_report.md").write_text("\n".join(md))
    with pd.option_context("display.width", 200, "display.max_colwidth", 100):
        print(comp.to_string(index=False))
    log.info("summaries -> %s", cfg.out / "summaries_report.md")


def query_params(cfg: Config) -> dict:
    """Break points for queries/05, taken from the same fuzzy sets as `summarize`."""
    t = graph.read_tables(cfg.work / "graph")
    var = {v.attribute: v for v in summaries.vendor_variables(summaries.contract_table(t))}
    lo, hi = var["total_value"].labels[1].mf.xs
    return {"total_lo": float(lo), "total_hi": float(hi)}


def step_run_queries(cfg: Config, params: dict) -> None:
    from .neo4j_io import connect_from_env, run_query_files
    params = {**params, **query_params(cfg)}
    with connect_from_env() as drv:
        run_query_files(drv, Path("queries"), cfg.out / "queries", params)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--truth", default="data/raw/synthetic_truth.csv")
    ap.add_argument("--out", default="data/raw/synthetic_contracts.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wipe", action="store_true", help="delete all nodes before loading")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--vendor-id", default="")
    ap.add_argument("--use-truth", action="store_true",
                    help="summarize: also use the synthetic answer key (--truth)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config.load(a.config)

    if a.step == "make-synthetic":
        df, truth = synthetic.generate(seed=cfg.seed % 10_000)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        truth.to_csv(a.truth, index=False)
        log.info("synthetic: %d rows -> %s", len(df), a.out)
    elif a.step == "prepare":
        step_prepare(cfg)
    elif a.step == "link":
        step_link(cfg)
    elif a.step == "label-sheet":
        step_label_sheet(cfg, a.labels, a.force)
    elif a.step == "label":
        step_label_interactive(a.labels)
    elif a.step == "oracle-label":
        step_oracle_label(cfg, a.labels, a.truth)
    elif a.step == "evaluate":
        step_evaluate(cfg, a.labels)
    elif a.step == "cluster":
        step_cluster(cfg)
    elif a.step == "export-graph":
        step_export_graph(cfg)
    elif a.step == "load-neo4j":
        step_load_neo4j(cfg, a.wipe)
    elif a.step == "run-queries":
        step_run_queries(cfg, {"min_conf": a.min_conf, "vendor_id": a.vendor_id,
                               "min_amendments": 2, "min_growth": 1.5,
                               "low": cfg.match_threshold, "high": 0.99})
    elif a.step == "summarize":
        step_summarize(cfg, a.truth if a.use_truth else None)
    else:
        ap.error(f"unknown step {a.step!r}")


if __name__ == "__main__":
    main()
