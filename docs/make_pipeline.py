"""Draw docs/figures/pipeline_{light,dark}.png. Usage: python docs/make_pipeline.py (needs matplotlib)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent / "figures"
THEMES = {
    "light": dict(surface="#fcfcfb", box="#f0efec", edge="#d6d5ce", text="#0b0b0b", text2="#52514e",
                  arrow="#898781", bg="#2a78d6", fg="#eb6834"),
    "dark": dict(surface="#1a1a19", box="#262624", edge="#383835", text="#ffffff", text2="#c3c2b7",
                 arrow="#898781", bg="#3987e5", fg="#d95926"),
}

# (key, x, y, w, h, title, body, accent)
BOXES = [
    ("in", 0.2, 2.2, 2.0, 1.6, "Input", "6 training views\nper test scene\n+ calibration", None),
    ("p1", 2.8, 3.55, 3.9, 1.75, "Person models (6-15)",
     "FreeTimeGS++, stride-5 keyframes\nperson-weighted loss x4, 60k iters\nSEVA pseudo-views on pixels\nno training camera sees", "fg"),
    ("p2", 7.3, 3.55, 3.4, 1.75, "Person ensemble",
     "pixel mean of\n1/2 person models +\n1/2 diverse scene mix", "fg"),
    ("p3", 11.3, 3.55, 3.0, 1.75, "Difix3D+ TTA",
     "reference = nearest\ntraining view, t = 199\nmean of 3 shifted\ntile grids", "fg"),
    ("b1", 2.8, 0.7, 3.9, 1.75, "Scene models (7)",
     "LPIPS 0.3, dense VGGT depth,\nSEVA pseudo-views, gating,\n4M Gaussians, full-res; plus a\nmodel distilled from Difix output", "bg"),
    ("b2", 7.3, 0.7, 3.4, 1.75, "Perceptual projection",
     "min LPIPS(x, Difix-TTA)\n+ 15 (1 - SSIM(x, raw))\n200 steps per image", "bg"),
    ("b3", 11.3, 0.7, 3.0, 1.75, "Static plate",
     "temporal mean per view\nonly where the person\nnever appears", "bg"),
    ("c1", 14.9, 2.2, 3.0, 1.6, "Composite",
     "DeepLabV3 person mask\nalpha 0.85, feather 101", None),
    ("c2", 18.5, 2.2, 3.2, 1.6, "Colour correction",
     "per-camera prior on the\ntwo rig-matched scenes\n(007 gain, 011 taper)", None),
    ("c3", 22.3, 2.2, 2.6, 1.6, "Package",
     "8 hidden views x\n5 scenes, JPEG q95", None),
]
ARROWS = [("in", "p1"), ("in", "b1"), ("p1", "p2"), ("p2", "p3"), ("b1", "b2"), ("b2", "b3"),
          ("b1", "p2"), ("p3", "c1"), ("b3", "c1"), ("c1", "c2"), ("c2", "c3")]


def draw(mode, t):
    fig, ax = plt.subplots(figsize=(18, 4.8), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 25.1)
    ax.set_ylim(0.1, 5.9)
    ax.axis("off")
    geo = {}
    for key, x, y, w, h, title, body, accent in BOXES:
        geo[key] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=t["box"], ec=t["edge"], lw=1))
        if accent:
            ax.add_patch(FancyBboxPatch((x, y + h - 0.07), w, 0.07, boxstyle="square,pad=0",
                                        fc=t[accent], ec="none"))
        ax.text(x + 0.15, y + h - 0.28, title, color=t["text"], fontsize=10.5, fontweight="bold", va="top")
        ax.text(x + 0.15, y + h - 0.62, body, color=t["text2"], fontsize=8.8, va="top", linespacing=1.35)
    for a, b in ARROWS:
        xa, ya, wa, ha = geo[a]
        xb, yb, wb, hb = geo[b]
        if (a, b) == ("b1", "p2"):
            start, end = (xa + wa * 0.8, ya + ha), (xb + 0.4, yb)
        else:
            start, end = (xa + wa, ya + ha / 2), (xb, yb + hb / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=t["arrow"],
                                     lw=1.2, connectionstyle="arc3,rad=0", shrinkA=2, shrinkB=2))
    ax.text(2.8, 5.55, "PERSON BRANCH (drives the foreground metrics)", color=t["fg"], fontsize=9,
            fontweight="bold")
    ax.text(2.8, 0.25, "BACKGROUND BRANCH (94% of the pixels, drives the full-image metrics)", color=t["bg"],
            fontsize=9, fontweight="bold")
    fig.savefig(FIG / f"pipeline_{mode}.png", facecolor=t["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for m, t in THEMES.items():
        draw(m, t)
    print("pipeline figures written")
