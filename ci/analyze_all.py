"""All experiments in one table + two-panel Figure 1 (main grid | scale).

  python -m ci.analyze_all
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final"


def load(name):
    p = ROOT / "results" / name / "results.jsonl"
    if not p.exists():
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines()])
    df = df[df["violation"].notna()].drop_duplicates("key", keep="last").copy()
    df["flag"] = df["violation"].astype(bool)
    df["ep"] = df.get("scenario_seed", 0).astype(str) + "/" + df["scenario"] if "scenario_seed" in df else df["scenario"]
    return df


def ci(g, n=2000, seed=0):
    g = g[g.violating]
    if g.empty:
        return np.nan, np.nan, np.nan, 0
    by = g.groupby("ep").flag.agg(["sum", "count"])
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(by), (n, len(by)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return g.flag.mean(), np.percentile(est, 2.5), np.percentile(est, 97.5), len(g)


def row(exp, setup, g):
    m, lo, hi, n = ci(g)
    clean = g[~g.violating]
    return dict(experiment=exp, setup=setup, detection=m, lo=lo, hi=hi, n=n,
                false_alarm=clean.flag.mean() if len(clean) else np.nan, n_clean=len(clean))


def sel(df, mode, sp=None, k=None):
    g = df[df["mode"] == mode]
    if sp is not None:
        g = g[g["sum_policy"].astype(bool) == sp]
    if k is not None:
        g = g[g["chunks"].fillna(4) == k]
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    E = [("Main (DeepSeek logs, 40-60 actions)", load("raw_ds"), load("main")),
         ("Replication (fresh episodes, Qwen logs)", load("raw_seed1"), load("ext_seed1")),
         ("Scale (150-200 actions, Qwen logs)", load("raw_big"), load("big"))]
    for exp, raw, pipe in E:
        if len(raw):
            rows.append(row(exp, "Single analyst, raw log", raw))
        if len(pipe):
            for mode, sp, lab in [("naive", False, "Plain summaries"), ("provenance", False, "Provenance summaries"),
                                  ("naive", True, "Plain + policy-aware"), ("provenance", True, "Provenance + policy-aware"),
                                  ("actions", False, "Plain + raw action records")]:
                g = sel(pipe, mode, sp, 4)
                if len(g):
                    rows.append(row(exp, lab, g))
    summ = load("summ")
    if len(summ):
        for mode, sp, lab in [("naive", False, "Plain summaries"), ("provenance", True, "Provenance + policy-aware")]:
            rows.append(row("Qwen 3.8 27B summariser (DeepSeek logs)", lab, sel(summ, mode, sp)))
    ch, main = load("chunks"), load("main")
    for k in (2, 4, 8):
        src = main if k == 4 else ch
        for mode, sp, lab in [("naive", False, "Plain summaries"), ("provenance", True, "Provenance + policy-aware")]:
            g = sel(src, mode, sp, k)
            if len(g):
                rows.append(row(f"Chunks k={k} (DeepSeek logs)", lab, g))
    big = load("big")
    if len(big):
        rows.append(row("Scale (150-200 actions, Qwen logs)", "Plain summaries, 8 chunks", sel(big, "naive", False, 8)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "all_experiments.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))

    # Figure: two panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"Single analyst, raw log": "#2b6cb0", "Plain summaries": "#c53030", "Provenance summaries": "#dd6b20",
           "Plain + policy-aware": "#d69e2e", "Provenance + policy-aware": "#38a169", "Plain + raw action records": "#276749"}
    order = list(pal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    for ax, exp, title in [(axes[0], E[0][0], "A. Main grid: 40-60-action logs"),
                           (axes[1], E[2][0], "B. Scale: 150-200-action logs")]:
        d = T[T.experiment == exp].set_index("setup").reindex(order).dropna(subset=["detection"])
        y = np.arange(len(order))[::-1]
        ypos = {s: yy for s, yy in zip(order, y)}
        for s, r in d.iterrows():
            ax.barh(ypos[s], r.detection * 100, color=pal[s], height=0.62)
            ax.errorbar(r.detection * 100, ypos[s], xerr=[[(r.detection - r.lo) * 100], [(r.hi - r.detection) * 100]],
                        fmt="none", ecolor="#1a202c", capsize=3, lw=1)
            ax.text(r.hi * 100 + 1.5, ypos[s], f"{r.detection*100:.0f}%", va="center", fontsize=9, fontweight="bold")
        for s in order:
            if s not in d.index:
                ax.text(2, ypos[s], "not run", va="center", fontsize=8, color="#718096")
        ax.set_yticks(y, order, fontsize=9)
        ax.set_xlim(0, 116)
        ax.axvline(100, color="#a0aec0", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("Violations detected (%), 95% CI clustered by episode")
    fig.suptitle("Same evidence, different pipeline: summarise-then-judge loses violations, and more as logs grow",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_two_panel.png", dpi=200)
    print("saved", OUT / "fig1_two_panel.png")


if __name__ == "__main__":
    main()
