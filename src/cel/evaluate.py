"""Weighted evaluation from a stratified label sample.

Every labeled pair carries a weight w = N_stratum / n_sampled. Weighted counts
estimate counts over all pairs:
  TP = sum w * [pred=1, y=1],  FP = sum w * [pred=1, y=0],  FN = sum w * [pred=0, y=1]
  precision = TP / (TP + FP),  recall = TP / (TP + FN)
Recall is measured against true matches in the candidate set plus the NEAR
stratum, so it includes what blocking loses (but not matches that look nothing
alike, which no string method would find).
Confidence intervals: bootstrap that resamples within each stratum.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _prf(pred, y, w):
    tp = np.sum(w * (pred & (y == 1)))
    fp = np.sum(w * (pred & (y == 0)))
    fn = np.sum(w * (~pred & (y == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else np.nan
    r = tp / (tp + fn) if tp + fn > 0 else np.nan
    f = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else np.nan
    return p, r, f


def _boot(lab: pd.DataFrame, pred: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(lab["stratum"].to_numpy() == s) for s in lab["stratum"].unique()]
    y, w = lab["y"].to_numpy(), lab["weight"].to_numpy()
    res = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        res.append(_prf(pred[idx], y[idx], w[idx]))
    res = np.array(res, dtype=float)
    lo = np.nanpercentile(res, 2.5, axis=0)
    hi = np.nanpercentile(res, 97.5, axis=0)
    return lo, hi


def predictions(lab: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    return {
        "Exact cleaned name": lab["exact_clean"].to_numpy() == 1,
        f"Jaro-Winkler >= {cfg.jw_baseline_threshold}": lab["jw"].to_numpy()
        >= cfg.jw_baseline_threshold,
        f"Fellegi-Sunter, P >= {cfg.match_threshold}": lab["in_candidates"].to_numpy().astype(bool)
        & (lab["prob"].to_numpy() >= cfg.match_threshold),
    }


def metrics_table(lab: pd.DataFrame, cfg) -> pd.DataFrame:
    y, w = lab["y"].to_numpy(), lab["weight"].to_numpy()
    rows = []
    for name, pred in predictions(lab, cfg).items():
        p, r, f = _prf(pred, y, w)
        lo, hi = _boot(lab, pred, cfg.n_bootstrap, cfg.seed)
        rows.append({"method": name, "precision": p, "precision_lo": lo[0], "precision_hi": hi[0],
                     "recall": r, "recall_lo": lo[1], "recall_hi": hi[1],
                     "f1": f, "f1_lo": lo[2], "f1_hi": hi[2]})
    return pd.DataFrame(rows)


def pr_curve(score: np.ndarray, y: np.ndarray, w: np.ndarray):
    order = np.argsort(-score, kind="stable")
    s, yy, ww = score[order], y[order], w[order]
    tp = np.cumsum(ww * (yy == 1))
    fp = np.cumsum(ww * (yy == 0))
    total_pos = np.sum(ww * (yy == 1))
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # end of each tied block
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / total_pos
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return rec, prec, ap


def reliability(prob: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        rows.append({"bin": k, "mean_prob": np.average(prob[m], weights=w[m]),
                     "frac_match": np.average(y[m], weights=w[m]),
                     "weight": w[m].sum(), "n_labeled": int(m.sum())})
    t = pd.DataFrame(rows)
    ece = float(np.sum(t["weight"] * (t["mean_prob"] - t["frac_match"]).abs()) / t["weight"].sum())
    return t, ece


def plot_all(lab: pd.DataFrame, cfg, out_dir) -> dict:
    y, w = lab["y"].to_numpy(), lab["weight"].to_numpy()
    fs_score = np.where(lab["in_candidates"].to_numpy().astype(bool), lab["prob"].to_numpy(), -1.0)
    r_fs, p_fs, ap_fs = pr_curve(fs_score, y, w)
    r_jw, p_jw, ap_jw = pr_curve(lab["jw"].to_numpy(), y, w)
    ex = lab["exact_clean"].to_numpy() == 1
    p_ex, r_ex, _ = _prf(ex, y, w)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(r_fs, p_fs, label=f"Fellegi-Sunter (AP {ap_fs:.2f})")
    ax.plot(r_jw, p_jw, label=f"Jaro-Winkler only (AP {ap_jw:.2f})", linestyle="--")
    ax.scatter([r_ex], [p_ex], marker="s", color="k", zorder=3, label="Exact cleaned name")
    ax.set_xlabel("Recall (weighted)")
    ax.set_ylabel("Precision (weighted)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Vendor-name linking: precision vs recall")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    cand = lab["in_candidates"].to_numpy().astype(bool)
    rel, ece = reliability(lab["prob"].to_numpy()[cand], y[cand], w[cand])
    rel.to_csv(out_dir / "reliability_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=1, label="perfect calibration")
    size = 30 + 270 * rel["weight"] / rel["weight"].max()
    ax.scatter(rel["mean_prob"], rel["frac_match"], s=size, alpha=0.8, zorder=3,
               label="bin (size = pairs it stands for)")
    ax.legend(loc="upper left", fontsize=7)
    for _, r in rel.iterrows():
        ax.annotate(str(int(r["n_labeled"])), (r["mean_prob"], r["frac_match"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=7, color="0.35")
    ax.set_xlabel("Predicted P(match)")
    ax.set_ylabel("Observed share of true matches")
    ax.set_title(f"Calibration of P(match), ECE {ece:.3f}\n(numbers = labeled pairs per bin)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)
    return {"ap_fellegi_sunter": ap_fs, "ap_jaro_winkler": ap_jw, "ece": ece}


def blocking_loss(lab: pd.DataFrame) -> float:
    """Weighted share of true matches that fall outside the candidate set."""
    pos = lab["y"] == 1
    tot = (lab["weight"] * pos).sum()
    out = (lab["weight"] * (pos & ~lab["in_candidates"].astype(bool))).sum()
    return float(out / tot) if tot else float("nan")
