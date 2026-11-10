# HW3: Perceptual Metric Reproduction & Failure Analysis (TID2013)

This repo evaluates five full-reference image quality metrics on TID2013 and compares each one with human
Mean Opinion Scores (MOS). The metrics are **PSNR, SSIM, MS-SSIM, LPIPS and DISTS**. For each metric it
reports SRCC and PLCC, both over the whole dataset and per distortion type. It also selects failure and
disagreement cases from quantitative criteria rather than by eye.

## 1. Environment

- Python 3.12, PyTorch 2.6 (CUDA 12.4). Tested on one NVIDIA RTX 2080 Ti.
- Pinned packages are listed in [`requirements.txt`](requirements.txt).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs pyiqa 0.1.16 (IQA-PyTorch), which provides all metrics
```

Pretrained weights for LPIPS (AlexNet / VGG) and DISTS are downloaded by pyiqa automatically on first
use, into `~/.cache/torch/hub/`.

## 2. Dataset

The dataset is **not included** in this repo. Download TID2013 (Ponomarenko et al., 2015) and extract it so
that the layout looks like this:

```
data/tid2013/
├── reference_images/      25 reference images (512x384 BMP)
├── distorted_images/      3000 = 25 refs x 24 distortion types x 5 levels, named iXX_YY_Z.bmp
├── mos_with_names.txt     MOS (0-9, higher = better) + file name; row order = all other .txt files
├── mos_std.txt
└── metrics_values/        official PSNR / SSIM / MS-SSIM / FSIMc ... values (used for verification)
```

There are two sources:
- Official: <https://www.ponomarenko.info/tid2013/tid2013.rar>. You need a full RAR decoder such as
  `unrar` or official 7-Zip. **Do not use `bsdtar`/libarchive**: it silently produced 58 unreadable images
  and about 140 corrupted ones in our tests.
- Mirror used for this project (IQA-PyTorch):
  <https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/tid2013.tgz>.
  We validated it against the official archive. PSNR recomputed from the mirror files agrees with the
  official `metrics_values/PSNR.txt` for every image.

Note: file-name case is inconsistent (`I01.BMP`, `i25.bmp`, `I01_11_1.BMP`). The code looks up files
case-insensitively.

## 3. Reproduce

Run from the repo root, in this order. Total time is about 6 minutes on one 2080 Ti.

```bash
# (1) LPIPS (Alex, VGG) + DISTS (native res. and official 256 resize); attaches official metric values
python src/compute_deep_metrics.py --root data/tid2013 --out results/scores_deep.csv --extra-vgg

# (2) PSNR / SSIM / MS-SSIM, checked against the official values in metrics_values/
python src/compute_trad_metrics.py --root data/tid2013 --scores results/scores_deep.csv --out results/scores.csv

# (3) correlations, per-distortion analysis, figures, failure-case mining
python src/evaluate.py

# (4) the K most extreme metric-vs-human disagreements (overall, and for LPIPS / DISTS separately)
python src/select_extreme.py --k 10
python src/select_extreme.py --k 10 --metric LPIPS
python src/select_extreme.py --k 10 --metric DISTS
```

Use `--device cuda:N` (or `CUDA_VISIBLE_DEVICES`) to pick a GPU. All scripts accept `--help`.

## 4. Implementation details

| Metric | Implementation (pyiqa) | Input | Direction |
|---|---|---|---|
| PSNR | `psnr` | RGB, [0,1] | higher = better |
| SSIM | `ssim` (Gaussian 11x11, σ=1.5) | Y channel, **no** downsampling | higher = better |
| MS-SSIM | `ms_ssim` (5 scales) | Y channel | higher = better |
| LPIPS | `lpips` (v0.1, AlexNet) | RGB, [0,1], rescaled internally to [-1,1], native 512x384 | **lower = better** |
| DISTS | `dists` | RGB, [0,1], short side resized to 256 (bilinear, as in the authors' `prepare_image`) | **lower = better** |

Variants are also reported, to check how sensitive each metric is to implementation choices: PSNR-Y,
SSIM with Wang et al.'s MATLAB downsampling, LPIPS-VGG, and DISTS at native resolution. The official
FSIMc values are included as a reference.

**Score direction.** LPIPS and DISTS are distances, so they are multiplied by −1 before any correlation is
computed. After that, every metric is "higher = better", the same direction as MOS.

**Correlations** ([`src/evaluate.py`](src/evaluate.py)):
- **SRCC** (Spearman) and **KRCC** (Kendall) are computed on the raw scores.
- **PLCC** is computed after a monotonic 5-parameter logistic mapping (VQEG):
  `q = b1·(0.5 − 1/(1+exp(b2·(x−b3)))) + b4·x + b5`, with b1, b2, b4 ≥ 0 so the mapping is
  increasing. PLCC on raw scores (`PLCC_raw`) is also saved.
- 95% CIs for SRCC come from 1000 bootstrap resamples.
- Infinite PSNR values (luminance identical to the reference) are set to max finite value + 1. Ranks
  are unchanged.

**Failure-case selection.** TID2013 MOS was collected by comparing distorted versions of the *same*
reference image. For this reason, cases are selected using **within-reference percentile ranks**: the
rank of an image among the 120 distorted versions of its reference, scaled to 0..1.
- *Single images*: |rank(metric) − rank(MOS)| for one metric vs. MOS, and |rank(A) − rank(B)| for two
  metrics.
- *Pairs* (A, B) from the same reference: kept when humans clearly prefer A (ΔMOS ≥ 1 and larger than the
  pooled MOS std) while the metric prefers B.
- Selection is greedy over the ranked list, with caps on repeats of the same distortion type or
  reference so that no single type dominates. Only after selection were the images inspected visually.

**Verification against official values** (printed by `compute_trad_metrics.py`):
- PSNR (RGB) matches `PSNRc.txt` exactly.
- PSNR-Y has the same ranking as `PSNR.txt`, with a constant offset of 1.31 dB = 20·log10(255/219).
  The cause is a different Y range: pyiqa uses limited range 16–235, the official values use full range.
- SSIM: SRCC between ours and `SSIM.txt` is 0.995.
- MS-SSIM: SRCC between ours and `MSSIM.txt` is 0.997.
- LPIPS-VGG (SRCC 0.670) and DISTS (SRCC 0.830) reproduce the values reported in the literature.

## 5. Main results (TID2013, all 3000 images)

| Metric | SRCC [95% CI] | PLCC (logistic) |
|---|---|---|
| PSNR | 0.687 [0.665, 0.708] | 0.677 |
| SSIM | 0.627 [0.600, 0.652] | 0.685 |
| MS-SSIM | 0.786 [0.766, 0.804] | 0.834 |
| LPIPS | 0.744 [0.727, 0.761] | 0.771 |
| DISTS | 0.830 [0.817, 0.843] | 0.856 |

Sensitivity to scale:
- SSIM: 0.627 without downsampling, 0.741 with 2x downsampling.
- DISTS: 0.708 at native resolution, 0.830 with the short side resized to 256.

Per-distortion results are in `results/per_distortion_srcc.csv` and `figures/heatmap_srcc.png`.

## 6. Repository layout

```
src/
  compute_deep_metrics.py   LPIPS / LPIPS-VGG / DISTS (+ official metric values) -> results/scores_deep.csv
  compute_trad_metrics.py   PSNR / PSNR-Y / SSIM / SSIM-ds / MS-SSIM + verification -> results/scores.csv
  evaluate.py               correlations, per-distortion SRCC, scatter/heatmap, failure-case mining
  select_extreme.py         top-K most extreme metric-vs-MOS disagreements
results/
  scores.csv                one row per distorted image: ref, type, level, MOS, MOS std, all metric values
  summary_overall.csv       SRCC / KRCC / PLCC_raw / PLCC / RMSE / SRCC 95% CI per metric (+ variants)
  per_distortion_srcc.csv   24 distortion types x 5 metrics
  failure_single.csv        single-image disagreements (metric vs MOS; traditional vs learned)
  failure_pairs.csv         same-reference pairs where a metric inverts a clear human preference
  extreme_{top,diverse}[_LPIPS|_DISTS].csv   top-10 most extreme cases (pure / one per distortion type)
figures/
  heatmap_srcc.png          SRCC per distortion type x metric
  scatter.png               metric vs MOS with fitted logistic curve, colored by distortion family
  cases/                    single_*.png and pair_*.png from evaluate.py
  extreme*/{top,diverse}/cases/   figures for the select_extreme.py lists
```

## 7. References

- N. Ponomarenko et al., "Image database TID2013: Peculiarities, results and perspectives," *Signal
  Processing: Image Communication*, 30:57–77, 2015.
- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: From error visibility
  to structural similarity," *IEEE TIP*, 13(4):600–612, 2004.
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for image quality
  assessment," *Asilomar Conf. Signals, Systems and Computers*, 2003.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness of deep
  features as a perceptual metric," *CVPR*, 2018.
- K. Ding, K. Ma, S. Wang, E. P. Simoncelli, "Image quality assessment: Unifying structure and texture
  similarity," *IEEE TPAMI*, 44(5):2567–2581, 2022.
- C. Chen, J. Mo, "IQA-PyTorch: PyTorch toolbox for image quality assessment,"
  <https://github.com/chaofengc/IQA-PyTorch>, 2022.
- VQEG, "Final report from the Video Quality Experts Group on the validation of objective models of video
  quality assessment," 2000.
