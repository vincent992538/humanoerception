"""Compute LPIPS and DISTS for all 3000 TID2013 distorted images.

Usage (inside the venv):
    /mnt/HDD5/sys115/.venv/bin/python src/compute_deep_metrics.py \
        --root data/tid2013 --out results/scores_deep.csv --device cuda:0

Implementation notes
--------------------
* Metrics come from pyiqa (IQA-PyTorch) 0.1.16:
    - 'lpips'     : LPIPS v0.1, AlexNet backbone (Zhang et al., CVPR 2018)
    - 'lpips-vgg' : LPIPS v0.1, VGG16 backbone (optional, --extra-vgg)
    - 'dists'     : DISTS (Ding et al., TPAMI 2020)
* Input: RGB, float in [0, 1], native 512x384 resolution, no resizing/cropping.
  pyiqa's LPIPS rescales [0,1] -> [-1,1] internally (normalize=True).
* DISTS is computed twice:
    - 'dists'      : native resolution (pyiqa default)            -> SRCC ~0.708
    - 'dists_r256' : official DISTS preprocessing, i.e. PIL bilinear
                     resize so the short side = 256 (512x384 -> 341x256),
                     as in the authors' prepare_image()           -> SRCC ~0.830 (= paper)
* All three are distances: LOWER = more similar = better quality.
  MOS is HIGHER = better, so correlations must use the negated score.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

import pyiqa


def find_ci(directory, name):
    """Case-insensitive file lookup (TID2013 mixes 'I01.BMP' and 'i01.bmp')."""
    lut = {f.lower(): f for f in os.listdir(directory)}
    return os.path.join(directory, lut[name.lower()])


def load(path, short_side=None):
    im = Image.open(path).convert("RGB")
    if short_side is not None:  # official DISTS: transforms.Resize(256)
        w, h = im.size
        s = short_side / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # 3xHxW in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tid2013")
    ap.add_argument("--out", default="results/scores_deep.csv")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="default: cuda:0 if available, else cpu (CPU works, ~1.5 h with --extra-vgg)")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--extra-vgg", action="store_true", help="also compute LPIPS-VGG")
    args = ap.parse_args()

    ref_dir = os.path.join(args.root, "reference_images")
    dist_dir = os.path.join(args.root, "distorted_images")

    # --- MOS table (row order = official order of mos.txt / metrics_values/*.txt)
    rows = []
    with open(os.path.join(args.root, "mos_with_names.txt")) as f:
        for line in f:
            mos, name = line.split()
            ref_id, dtype, level = map(int, re.match(r"i(\d+)_(\d+)_(\d)", name, re.I).groups())
            rows.append(dict(name=name.lower(), ref=ref_id, dist_type=dtype, level=level, mos=float(mos)))
    df = pd.DataFrame(rows)
    df["mos_std"] = np.loadtxt(os.path.join(args.root, "mos_std.txt"))

    # --- metrics
    names = ["lpips", "dists"] + (["lpips-vgg"] if args.extra_vgg else [])
    metrics = {n: pyiqa.create_metric(n, device=args.device, as_loss=False) for n in names}
    for n, m in metrics.items():
        assert m.lower_better, n
    scores = {n: np.full(len(df), np.nan) for n in names + ["dists_r256"]}

    t0 = time.time()
    with torch.no_grad():
        for ref_id, g in df.groupby("ref"):
            ref_path = find_ci(ref_dir, f"i{ref_id:02d}.bmp")
            ref = load(ref_path).to(args.device)
            ref256 = load(ref_path, 256).to(args.device)
            idx = g.index.to_numpy()
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch]
                paths = [find_ci(dist_dir, df.at[i, "name"]) for i in b]
                dist = torch.stack([load(p) for p in paths]).to(args.device)
                for n, m in metrics.items():
                    scores[n][b] = m(dist, ref.expand_as(dist)).flatten().cpu().numpy()
                dist256 = torch.stack([load(p, 256) for p in paths]).to(args.device)
                scores["dists_r256"][b] = metrics["dists"](dist256, ref256.expand_as(dist256)).flatten().cpu().numpy()
            print(f"ref {ref_id:02d} done  ({time.time() - t0:.0f}s)", flush=True)

    names = names + ["dists_r256"]
    for n in names:
        df[n.replace("-", "_")] = scores[n]

    # --- attach official precomputed metrics for convenience (same row order)
    mv = os.path.join(args.root, "metrics_values")
    for col, fn in [("psnr_y_official", "PSNR.txt"), ("psnr_rgb_official", "PSNRc.txt"),
                    ("ssim_official", "SSIM.txt"), ("msssim_official", "MSSIM.txt"),
                    ("fsimc_official", "FSIMc.txt")]:
        v = np.loadtxt(os.path.join(mv, fn))
        df[col] = np.where(v >= 1e4, np.inf, v)  # 100000 = 'infinite' sentinel

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"saved {args.out}  ({len(df)} rows, NaN: {int(df[[n.replace('-', '_') for n in names]].isna().sum().sum())})")


if __name__ == "__main__":
    main()
