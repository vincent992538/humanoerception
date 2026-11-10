"""Evaluate metrics against TID2013 MOS and mine failure / disagreement cases.

Usage (inside the venv):
    /mnt/HDD5/sys115/.venv/bin/python src/evaluate.py

Input : results/scores.csv  (from compute_deep_metrics.py + compute_trad_metrics.py)
Output: results/summary_overall.csv        SRCC / KRCC / PLCC(raw) / PLCC(logistic) / RMSE + 95% bootstrap CI
        results/per_distortion_srcc.csv    24 distortion types x metrics
        results/failure_single.csv         images where a metric disagrees with MOS / with another metric
        results/failure_pairs.csv          same-reference pairs whose order is inverted vs. MOS
        figures/heatmap_srcc.png, figures/scatter.png, figures/cases/*.png
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from compute_deep_metrics import find_ci

# ----------------------------------------------------------------------------- config
# name -> (column, sign). sign=-1 for distances (lower=better) so every score is "higher = better".
MAIN = {
    "PSNR": ("psnr", +1),
    "SSIM": ("ssim", +1),
    "MS-SSIM": ("ms_ssim", +1),
    "LPIPS": ("lpips", -1),
    "DISTS": ("dists_r256", -1),
}
VARIANTS = {
    "PSNR-Y": ("psnr_y", +1),
    "SSIM (downsampled)": ("ssim_ds", +1),
    "LPIPS-VGG": ("lpips_vgg", -1),
    "DISTS (native res.)": ("dists", -1),
    "FSIMc (official)": ("fsimc_official", +1),
}
DIST_NAMES = {
    1: "Additive Gaussian noise", 2: "Color-component noise", 3: "Spatially correlated noise",
    4: "Masked noise", 5: "High-frequency noise", 6: "Impulse noise", 7: "Quantization noise",
    8: "Gaussian blur", 9: "Image denoising", 10: "JPEG", 11: "JPEG2000",
    12: "JPEG transm. errors", 13: "JPEG2000 transm. errors", 14: "Non-ecc. pattern noise",
    15: "Local block-wise", 16: "Mean shift", 17: "Contrast change", 18: "Color saturation",
    19: "Multiplicative noise", 20: "Comfort noise", 21: "Lossy compr. noisy img",
    22: "Color quant. + dither", 23: "Chromatic aberration", 24: "Sparse sampling + recon.",
}
GROUP = {**{t: "noise" for t in (1, 2, 3, 4, 5, 6, 7, 19, 20)},
         **{t: "blur/restoration" for t in (8, 9, 24)},
         **{t: "compression" for t in (10, 11, 21)},
         **{t: "transmission" for t in (12, 13)},
         **{t: "exotic" for t in (14, 15, 16, 17)},
         **{t: "color" for t in (18, 22, 23)}}


# ----------------------------------------------------------------------------- helpers
def score(df, col, sign):
    x = sign * df[col].to_numpy(dtype=float)
    finite = np.isfinite(x)
    if not finite.all():  # +inf PSNR (identical luminance) -> just above the max finite value
        x = x.copy()
        x[~finite] = np.nanmax(x[finite]) + 1.0
    return x


def logistic5(x, b1, b2, b3, b4, b5):
    """VQEG 5-parameter monotonic logistic mapping (Sheikh et al., 2006)."""
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_logistic(x, y):
    """b1, b2, b4 >= 0 keeps the mapping monotonically increasing (all scores are higher-better),
    so PLCC is not inflated by a non-monotonic fit."""
    p0 = [np.ptp(y), 1.0 / (np.std(x) + 1e-12), np.mean(x), 1e-3, np.mean(y)]
    lb = [0, 0, -np.inf, 0, -np.inf]
    popt, _ = curve_fit(logistic5, x, y, p0=p0, bounds=(lb, np.inf), maxfev=200000)
    return popt


def correlations(x, y):
    yhat = logistic5(x, *fit_logistic(x, y))
    return dict(SRCC=spearmanr(x, y)[0], KRCC=kendalltau(x, y)[0], PLCC_raw=pearsonr(x, y)[0],
                PLCC=pearsonr(yhat, y)[0], RMSE=float(np.sqrt(np.mean((yhat - y) ** 2))))


def bootstrap_ci(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    s = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


# ----------------------------------------------------------------------------- main parts
def overall(df, metrics, outdir):
    rows = []
    for name, (col, sign) in metrics.items():
        x, y = score(df, col, sign), df.mos.to_numpy()
        r = correlations(x, y)
        lo, hi = bootstrap_ci(x, y)
        rows.append(dict(metric=name, column=col, direction="higher" if sign > 0 else "lower (negated)",
                         **r, SRCC_CI_low=lo, SRCC_CI_high=hi))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(outdir, "summary_overall.csv"), index=False, float_format="%.4f")
    print(t.drop(columns=["column"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return t


def per_distortion(df, metrics, outdir, figdir):
    tab = pd.DataFrame({name: [spearmanr(score(g, col, sign), g.mos)[0] for _, g in df.groupby("dist_type")]
                        for name, (col, sign) in metrics.items()},
                       index=[f"{t:02d} {DIST_NAMES[t]}" for t in sorted(df.dist_type.unique())])
    tab.to_csv(os.path.join(outdir, "per_distortion_srcc.csv"), float_format="%.4f")

    fig, ax = plt.subplots(figsize=(1.1 * len(metrics) + 3, 9))
    im = ax.imshow(tab.values, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tab.columns)), tab.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(tab.index)), tab.index, fontsize=8)
    for i, j in itertools.product(range(tab.shape[0]), range(tab.shape[1])):
        ax.text(j, i, f"{tab.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="SRCC with MOS")
    ax.set_title("TID2013: SRCC per distortion type")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "heatmap_srcc.png"), dpi=200)
    plt.close(fig)
    return tab


def scatter(df, metrics, figdir):
    groups = sorted(set(GROUP.values()))
    colors = dict(zip(groups, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8))
    for ax, (name, (col, sign)) in zip(axes, metrics.items()):
        x, y = score(df, col, sign), df.mos.to_numpy()
        for g in groups:
            m = df.dist_type.map(GROUP).to_numpy() == g
            ax.scatter(x[m], y[m], s=3, alpha=0.5, color=colors[g], label=g)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, logistic5(xs, *fit_logistic(x, y)), "k-", lw=1.5)
        ax.set_xlabel(name + (" (negated)" if sign < 0 else ""))
        ax.set_title(f"{name}: SRCC={spearmanr(x, y)[0]:.3f}")
    axes[0].set_ylabel("MOS")
    axes[-1].legend(markerscale=4, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "scatter.png"), dpi=200)
    plt.close(fig)


def within_ref_ranks(df, metrics):
    """Percentile rank (0..1) of MOS and each metric among the 120 versions of the same reference.
    TID2013 MOS comes from comparisons among versions of the SAME reference, so within-reference
    ranks are the most trustworthy basis for picking failure cases."""
    r = pd.DataFrame(index=df.index)
    r["mos"] = df.groupby("ref").mos.rank(pct=True)
    for name, (col, sign) in metrics.items():
        s = pd.Series(score(df, col, sign), index=df.index)
        r[name] = s.groupby(df.ref).rank(pct=True)
    return r


def diversify(cands, key, k, per_type=2, per_ref=2):
    """Greedy top-k that avoids picking many cases of the same distortion type or reference."""
    out, nt, nr = [], {}, {}
    for _, c in cands.sort_values(key, ascending=False).iterrows():
        t, ref = c["dist_type"], c["ref"]
        if nt.get(t, 0) >= per_type or nr.get(ref, 0) >= per_ref:
            continue
        out.append(c)
        nt[t] = nt.get(t, 0) + 1
        nr[ref] = nr.get(ref, 0) + 1
        if len(out) == k:
            break
    return pd.DataFrame(out)


def mine_single(df, metrics, ranks, outdir, k=8):
    base = df[["name", "ref", "dist_type", "level", "mos", "mos_std"]].copy()
    base["dist_name"] = base.dist_type.map(DIST_NAMES)
    for n in metrics:
        base[f"rank_{n}"] = ranks[n].round(3)
    base["rank_MOS"] = ranks["mos"].round(3)
    for n, (col, _) in metrics.items():
        base[n] = df[col]
    res = []
    # (a) metric vs human
    for n in metrics:
        c = base.copy()
        c["kind"] = f"{n} vs MOS"
        c["disagreement"] = (ranks[n] - ranks["mos"]).abs()
        c["direction"] = np.where(ranks[n] > ranks["mos"], f"{n} overrates", f"{n} underrates")
        res.append(diversify(c, "disagreement", k))
    # (b) traditional vs learned
    for a, b in [("PSNR", "LPIPS"), ("SSIM", "DISTS"), ("PSNR", "DISTS"), ("LPIPS", "DISTS")]:
        c = base.copy()
        c["kind"] = f"{a} vs {b}"
        c["disagreement"] = (ranks[a] - ranks[b]).abs()
        # who is closer to humans?
        c["direction"] = np.where((ranks[a] - ranks["mos"]).abs() < (ranks[b] - ranks["mos"]).abs(),
                                  f"{a} closer to MOS", f"{b} closer to MOS")
        res.append(diversify(c, "disagreement", k))
    out = pd.concat(res, ignore_index=True)
    out.to_csv(os.path.join(outdir, "failure_single.csv"), index=False, float_format="%.4f")
    return out


def mine_pairs(df, metrics, outdir, k=8, min_dmos=1.0):
    """Same-reference pairs (A, B) where humans clearly prefer A (MOS_A - MOS_B >= min_dmos and
    larger than the pooled MOS std) but the metric clearly prefers B. Strength = how far the metric's
    within-reference rank is inverted."""
    rows = []
    for ref, g in df.groupby("ref"):
        idx = g.index.to_numpy()
        mos, sd = g.mos.to_numpy(), g.mos_std.to_numpy()
        ia, ib = np.triu_indices(len(idx), 1)
        dm = mos[ia] - mos[ib]
        flip = dm < 0
        ia[flip], ib[flip] = ib[flip], ia[flip]  # make A the human-preferred one
        dm = np.abs(dm)
        sig = (dm >= min_dmos) & (dm > np.sqrt(sd[ia] ** 2 + sd[ib] ** 2))
        for n, (col, sign) in metrics.items():
            s = score(g, col, sign)
            r = pd.Series(s).rank(pct=True).to_numpy()
            inv = r[ib] - r[ia]  # >0: metric prefers B
            m = sig & (inv > 0)
            for a, b, d, v in zip(ia[m], ib[m], dm[m], inv[m]):
                rows.append(dict(metric=n, ref=ref, A=df.at[idx[a], "name"], B=df.at[idx[b], "name"],
                                 type_A=df.at[idx[a], "dist_type"], type_B=df.at[idx[b], "dist_type"],
                                 mos_A=mos[a], mos_B=mos[b], dMOS=d, metric_rank_inversion=v,
                                 strength=min(d / 3.0, v)))
    p = pd.DataFrame(rows)
    p["dist_type"] = p.type_A * 100 + p.type_B  # used by diversify(): one entry per type combination
    picked = pd.concat([diversify(p[p.metric == n], "strength", k, per_type=1, per_ref=2)
                        for n in metrics], ignore_index=True)
    for n, (col, _) in metrics.items():
        lut = df.set_index("name")[col]
        picked[f"{n}_A"] = picked.A.map(lut)
        picked[f"{n}_B"] = picked.B.map(lut)
    picked = picked.drop(columns="dist_type")
    picked.to_csv(os.path.join(outdir, "failure_pairs.csv"), index=False, float_format="%.4f")
    # how often each metric inverts a clearly-significant human preference
    inv_rate = p.groupby("metric").size()
    return picked, inv_rate


def fmt_metrics(df_row_lookup, metrics, name):
    r = df_row_lookup.loc[name]
    return "  ".join(f"{n}={r[col]:.3f}" if abs(r[col]) < 100 else f"{n}=inf"
                     for n, (col, _) in metrics.items())


def render_cases(df, metrics, ranks, single, pairs, root, figdir, n_single=12, n_pairs=8):
    cdir = os.path.join(figdir, "cases")
    os.makedirs(cdir, exist_ok=True)
    ref_dir, dist_dir = os.path.join(root, "reference_images"), os.path.join(root, "distorted_images")
    lut = df.set_index("name")
    rk = ranks.set_index(df.name)
    img = lambda d, n: np.asarray(Image.open(find_ci(d, n)).convert("RGB"))

    def rank_line(n):
        return "within-ref rank  MOS={:.2f}  ".format(rk.at[n, "mos"]) + \
            "  ".join(f"{m}={rk.at[n, m]:.2f}" for m in metrics)

    top = single.sort_values("disagreement", ascending=False).drop_duplicates("name").head(n_single)
    for i, c in enumerate(top.itertuples()):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        ax[0].imshow(img(ref_dir, f"i{c.ref:02d}.bmp")); ax[0].set_title(f"reference i{c.ref:02d}")
        ax[1].imshow(img(dist_dir, c.name))
        ax[1].set_title(f"{c.name}  #{c.dist_type} {DIST_NAMES[c.dist_type]}  level {c.level}\n"
                        f"MOS={c.mos:.2f} (±{c.mos_std:.2f})")
        for a in ax: a.axis("off")
        fig.suptitle(f"[{c.kind}] {c.direction}, |rank gap|={c.disagreement:.2f}", fontsize=11)
        fig.text(0.5, 0.06, fmt_metrics(lut, metrics, c.name), ha="center", fontsize=9)
        fig.text(0.5, 0.02, rank_line(c.name), ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
        fig.savefig(os.path.join(cdir, f"single_{i:02d}_{c.name[:-4]}.png"), dpi=130)
        plt.close(fig)

    top = pairs.sort_values("strength", ascending=False).drop_duplicates(["A", "B"]).head(n_pairs)
    for i, c in enumerate(top.itertuples()):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        ax[0].imshow(img(ref_dir, f"i{c.ref:02d}.bmp")); ax[0].set_title(f"reference i{c.ref:02d}")
        for a, n, t, m, tag in [(ax[1], c.A, c.type_A, c.mos_A, "A (humans prefer)"),
                                (ax[2], c.B, c.type_B, c.mos_B, "B")]:
            a.imshow(img(dist_dir, n))
            a.set_title(f"{tag}: {n}  #{t} {DIST_NAMES[t]}\nMOS={m:.2f}")
        for a in ax: a.axis("off")
        fig.suptitle(f"[{c.metric} inverts human preference]  dMOS={c.dMOS:.2f}, "
                     f"metric rank inversion={c.metric_rank_inversion:.2f}", fontsize=11)
        fig.text(0.5, 0.06, "A: " + fmt_metrics(lut, metrics, c.A), ha="center", fontsize=9)
        fig.text(0.5, 0.02, "B: " + fmt_metrics(lut, metrics, c.B), ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
        fig.savefig(os.path.join(cdir, f"pair_{i:02d}_{c.A[:-4]}_vs_{c.B[:-4]}.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tid2013")
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    df = pd.read_csv(args.scores)
    df["name"] = df.name.str.lower()

    print("=== Overall (main metrics + implementation variants) ===")
    overall(df, {**MAIN, **VARIANTS}, args.outdir)

    print("\n=== Per-distortion SRCC (main metrics) ===")
    tab = per_distortion(df, MAIN, args.outdir, args.figdir)
    print(tab.to_string(float_format=lambda v: f"{v:+.2f}"))
    scatter(df, MAIN, args.figdir)

    ranks = within_ref_ranks(df, MAIN)
    single = mine_single(df, MAIN, ranks, args.outdir)
    pairs, inv = mine_pairs(df, MAIN, args.outdir)
    print("\n=== # of same-reference pairs where humans clearly prefer A but the metric prefers B ===")
    print(inv.reindex(list(MAIN)).to_string())
    print("\n=== Top single-image disagreements (see results/failure_single.csv) ===")
    print(single.sort_values("disagreement", ascending=False)
          [["kind", "name", "dist_name", "level", "mos", "disagreement", "direction"]]
          .drop_duplicates("name").head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    render_cases(df, MAIN, ranks, single, pairs, args.root, args.figdir)
    print(f"\nfigures -> {args.figdir}/ (heatmap_srcc.png, scatter.png, cases/)")


if __name__ == "__main__":
    main()
