"""Qualitative comparison figures. They contain challenge-dataset pixels, so they are written to
docs/qualitative/, which .gitignore excludes by default (see README, "Dataset licence").

Usage: python docs/make_qualitative.py <val_dump_dir> <final_render_dir>
  val_dump_dir      fixtest dump of validation scene 001_1: <model>/<view>_<frame>_{gt,render}.png
  final_render_dir  unpacked submission: sparseViewTrack/renders/<scene>/<view>/<frame>.jpg
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "qualitative"
MODELS = [("m_FGW20_s0", "FG weight 2, stride 10, 30k"),
          ("m_S5_FGW40_L60K", "FG weight 4, stride 5, 60k"),
          ("m_DFX_s0", "Difix-distilled, full-res")]
# person crop per view as fractions of the frame (x0, x1, y0, y1), frame 000200
CROPS = {"02": (0.40, 0.62, 0.00, 0.82), "18": (0.33, 0.57, 0.00, 0.82), "38": (0.36, 0.60, 0.05, 0.85)}


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def psnr(a, b):
    return -10 * np.log10(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean() / 255 ** 2)


def crop(im, box):
    h, w = im.shape[:2]
    x0, x1, y0, y1 = box
    return im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def val_comparison(dump, view, frame="000200"):
    gt = rgb(dump / MODELS[0][0] / f"{view}_{frame}_gt.png")
    cols = [("Ground truth", gt, None)]
    for m, label in MODELS:
        r = rgb(dump / m / f"{view}_{frame}_render.png")
        cols.append((label, r, psnr(gt, r)))
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 5.6), dpi=150,
                             gridspec_kw=dict(height_ratios=[1, 1.35]))
    for j, (label, im, p) in enumerate(cols):
        axes[0, j].imshow(cv2.resize(im, (1008, 550), interpolation=cv2.INTER_AREA))
        axes[0, j].set_title(label + (f"\nPSNR {p:.2f} dB" if p is not None else "\n"), fontsize=10)
        axes[1, j].imshow(crop(im, CROPS[view]))
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Validation scene 001_1, hidden view {view}, frame {frame} (full frame / person crop)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"val001_view{view}.jpg", pil_kwargs=dict(quality=90))
    plt.close(fig)


def final_grid(renders, frame="000200"):
    scenes = sorted(p.name for p in renders.iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(scenes), 4, figsize=(16, 2.35 * len(scenes)), dpi=150)
    for i, s in enumerate(scenes):
        views = sorted(p.name for p in (renders / s).iterdir())
        for j, v in enumerate(views[::2][:4]):
            im = rgb(renders / s / v / f"{frame}.jpg")
            axes[i, j].imshow(cv2.resize(im, (960, 524), interpolation=cv2.INTER_AREA))
            axes[i, j].set_title(f"{s}  view {v}", fontsize=8)
            axes[i, j].axis("off")
    fig.suptitle(f"Final submission: hidden test views, frame {frame} (no public ground truth)", fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "final_test_views.jpg", pil_kwargs=dict(quality=88))
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dump, final = Path(sys.argv[1]), Path(sys.argv[2])
    for v in CROPS:
        val_comparison(dump, v)
    final_grid(final / "sparseViewTrack" / "renders")
    print("written to", OUT)
