"""Compute PSNR / SSIM / MS-SSIM on TID2013 with pyiqa and check them against the
official values shipped in data/tid2013/metrics_values/.

Usage (inside the venv):
    /mnt/HDD5/sys115/.venv/bin/python src/compute_trad_metrics.py

Variants computed (all HIGHER = better):
    psnr       : RGB, data range [0,1] (== 255 for 8-bit)
    psnr_y     : luminance (BT.601 Y) only  -> comparable to official PSNR.txt
    ssim       : Y channel, Gaussian 11x11 sigma 1.5, NO downsampling (pyiqa default)
    ssim_ds    : same, but with Wang et al.'s MATLAB downsampling
                 f = max(1, round(min(H,W)/256)) = 2 for 512x384  -> comparable to SSIM.txt
    ms_ssim    : Y channel, 5 scales (Wang et al. 2003)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

import pyiqa
from compute_deep_metrics import find_ci, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tid2013")
    ap.add_argument("--scores", default="results/scores_deep.csv", help="table produced by compute_deep_metrics.py")
    ap.add_argument("--out", default="results/scores.csv")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="default: cuda:0 if available, else cpu (~25 min on CPU)")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    dev = args.device
    metrics = {
        "psnr": pyiqa.create_metric("psnr", device=dev),
        "psnr_y": pyiqa.create_metric("psnry", device=dev),
        "ssim": pyiqa.create_metric("ssim", device=dev),
        "ssim_ds": pyiqa.create_metric("ssim", device=dev, downsample=True),
        "ms_ssim": pyiqa.create_metric("ms_ssim", device=dev),
    }
    out = {k: np.full(len(df), np.nan) for k in metrics}
    ref_dir = os.path.join(args.root, "reference_images")
    dist_dir = os.path.join(args.root, "distorted_images")

    with torch.no_grad():
        for ref_id, g in df.groupby("ref"):
            ref = load(find_ci(ref_dir, f"i{ref_id:02d}.bmp")).to(dev)
            idx = g.index.to_numpy()
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch]
                x = torch.stack([load(find_ci(dist_dir, df.at[i, "name"])) for i in b]).to(dev)
                for k, m in metrics.items():
                    out[k][b] = m(x, ref.expand_as(x)).flatten().cpu().numpy()
    for k, v in out.items():
        df[k] = np.where(np.isfinite(v), v, np.inf)  # identical-luminance images -> inf PSNR

    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"saved {args.out}")

    # ---- verification against official values
    print("\nours vs official (finite entries only):")
    for ours, off in [("psnr_y", "psnr_y_official"), ("psnr", "psnr_rgb_official"),
                      ("ssim", "ssim_official"), ("ssim_ds", "ssim_official"),
                      ("ms_ssim", "msssim_official")]:
        m = np.isfinite(df[ours]) & np.isfinite(df[off])
        a, b = df.loc[m, ours], df.loc[m, off]
        print(f"  {ours:8s} vs {off:18s} SRCC(ours,off)={spearmanr(a, b)[0]:.4f}  "
              f"median|diff|={np.median(np.abs(a - b)):.4f}  | SRCC vs MOS: ours={spearmanr(df[ours], df.mos)[0]:.3f} "
              f"official={spearmanr(df[off], df.mos)[0]:.3f}")


if __name__ == "__main__":
    main()
