"""Select the K most extreme metric-vs-human disagreements in TID2013.

Extremeness of an image = max over metrics m of |rank_m - rank_MOS|, where ranks are percentile
ranks (0..1) among the 120 distorted versions of the SAME reference image (TID2013 MOS was collected
by comparing versions of the same reference, so within-reference ranks are the reliable basis).

Two lists are written:
  extreme_top.csv      : pure top-K by extremeness
  extreme_diverse.csv  : top-K with at most one image per distortion type (avoids one type dominating)

With --metric NAME only that metric's gap is used (files get a _NAME suffix), e.g. to find where
LPIPS / DISTS themselves fail, independently of the larger failures of PSNR/SSIM.

Usage: /mnt/HDD5/sys115/.venv/bin/python src/select_extreme.py --k 10
       /mnt/HDD5/sys115/.venv/bin/python src/select_extreme.py --k 10 --metric LPIPS
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluate import DIST_NAMES, MAIN, render_cases, within_ref_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tid2013")
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures/extreme")
    ap.add_argument("--metric", choices=list(MAIN), default=None, help="rank by this metric's gap only")
    args = ap.parse_args()
    suffix = f"_{args.metric}" if args.metric else ""
    if args.metric:
        args.figdir = f"{args.figdir}_{args.metric}"

    df = pd.read_csv(args.scores)
    df["name"] = df.name.str.lower()
    ranks = within_ref_ranks(df, MAIN)

    gap = pd.DataFrame({m: ranks[m] - ranks["mos"] for m in MAIN})  # >0: metric overrates
    worst = gap.abs().idxmax(axis=1) if args.metric is None else pd.Series(args.metric, index=gap.index)
    t_gap = gap.abs().max(axis=1) if args.metric is None else gap[args.metric].abs()
    t = df[["name", "ref", "dist_type", "level", "mos", "mos_std"]].copy()
    t["dist_name"] = t.dist_type.map(DIST_NAMES)
    t["extremeness"] = t_gap
    t["worst_metric"] = worst
    t["direction"] = [("overrates" if gap.at[i, m] > 0 else "underrates") for i, m in worst.items()]
    t["n_metrics_gap>0.5"] = (gap.abs() > 0.5).sum(axis=1)  # how many metrics fail at once
    t["rank_MOS"] = ranks["mos"]
    for m, (col, _) in MAIN.items():
        t[f"rank_{m}"] = ranks[m]
        t[m] = df[col]
    t = t.sort_values("extremeness", ascending=False)

    top = t.head(args.k)
    diverse = t.drop_duplicates("dist_type").head(args.k)
    os.makedirs(args.outdir, exist_ok=True)
    top.to_csv(os.path.join(args.outdir, f"extreme_top{suffix}.csv"), index=False, float_format="%.4f")
    diverse.to_csv(os.path.join(args.outdir, f"extreme_diverse{suffix}.csv"), index=False, float_format="%.4f")

    cols = ["name", "dist_name", "level", "mos", "rank_MOS", "worst_metric", "direction",
            "extremeness", "n_metrics_gap>0.5"] + [f"rank_{m}" for m in MAIN]
    fmt = lambda v: f"{v:.2f}"
    print(f"=== Top {args.k} (pure) ===")
    print(top[cols].to_string(index=False, float_format=fmt))
    print(f"\n=== Top {args.k} (one per distortion type) ===")
    print(diverse[cols].to_string(index=False, float_format=fmt))

    # figures: reuse the single-case renderer
    for tag, sel in [("top", top), ("diverse", diverse)]:
        s = sel.rename(columns={"extremeness": "disagreement"}).copy()
        s["kind"] = s.worst_metric + " vs MOS"
        s["direction"] = s.worst_metric + " " + s.direction
        render_cases(df, MAIN, ranks, s, pd.DataFrame(columns=["strength", "A", "B"]), args.root,
                     os.path.join(args.figdir, tag), n_single=args.k, n_pairs=0)
    print(f"\nfigures -> {args.figdir}/top/cases, {args.figdir}/diverse/cases")


if __name__ == "__main__":
    main()
