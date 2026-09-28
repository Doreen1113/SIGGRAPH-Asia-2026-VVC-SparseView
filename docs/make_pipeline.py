"""Pipeline overview (light + dark): one card per stage with a schematic drawing and the gain measured for it.
No dataset pixels. Usage: python docs/make_pipeline.py   (needs matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

from make_figures import FIG
from make_analysis import theme, person, camera

FW, FH = 12.0, 8.0                      # figure size in inches; GitHub shows it about 900 px wide
COLS = [                                # header, x0, x1 (figure fraction)
    ("input", 0.005, 0.150),
    ("4D Gaussians", 0.160, 0.390),
    ("combine", 0.400, 0.610),
    ("restore / clean", 0.620, 0.830),
    ("output", 0.840, 0.995),
]
ROW_Y = {"top": 0.665, "bot": 0.285, "mid": 0.47}    # bottom of the drawing box


def frame(ax, t, fill=None):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 16, 9, fc=fill or t["surface"], ec=t["edge"], lw=1, zorder=0))


def gaussian_cloud(ax, rng, pts, color, n, spread, alpha=0.5):
    for _ in range(n):
        x, y = pts[rng.integers(len(pts))]
        ax.add_patch(Ellipse((x + rng.normal(0, spread), y + rng.normal(0, spread)), rng.uniform(0.3, 1.1),
                             rng.uniform(0.2, 0.6), angle=rng.uniform(0, 180), fc=color, ec="none", alpha=alpha,
                             zorder=2))


def skeleton_points(cx, base, s):
    pts = []
    for y in np.linspace(0, 0.8, 10):
        pts += [(cx - 0.13 * s, base + y * s), (cx + 0.13 * s, base + y * s)]
    for y in np.linspace(0.85, 1.45, 10):
        pts += [(cx - 0.08 * s, base + y * s), (cx + 0.08 * s, base + y * s)]
    for k in np.linspace(0, 1, 8):
        pts += [(cx - (0.15 + 0.35 * k) * s, base + (1.4 - 0.45 * k) * s),
                (cx + (0.15 + 0.35 * k) * s, base + (1.4 - 0.45 * k) * s)]
    pts += [(cx, base + 1.65 * s)] * 4
    return pts


# --- drawings ---------------------------------------------------------------------------------------------------
def d_input(ax, t):
    frame(ax, t)
    ax.add_patch(Ellipse((8, 5.2), 13.5, 7.2, fc="none", ec=t["grid"], lw=1, ls="--", zorder=1))
    person(ax, 8, 3.2, 2.1, t["ink"])
    tgt = np.array([8.0, 4.8])
    for a in np.deg2rad(np.linspace(-77.5, 77.5, 6)):
        p = np.array([8 + 6.6 * np.sin(a), 5.2 - 3.5 * np.cos(a)])
        d = tgt - p
        camera(ax, *p, np.arctan2(d[1], d[0]), 0.8, t["ink"], t["ink"])
    for a in np.deg2rad(np.linspace(-66, 72, 8)):
        p = np.array([8 + 5.6 * np.sin(a), 5.4 - 2.9 * np.cos(a) + 0.5])
        d = tgt - p
        camera(ax, *p, np.arctan2(d[1], d[0]), 0.62, t["surface"], t["s2"])


def d_person_models(ax, t):
    frame(ax, t)
    rng = np.random.default_rng(1)
    gaussian_cloud(ax, rng, skeleton_points(5.5, 1.0, 4.0), t["s2"], 240, 0.16, 0.45)
    for x in np.linspace(10.6, 15.0, 10):
        ax.plot([x, x], [1.3, 2.2], color=t["s2"], lw=2, solid_capstyle="round")
    ax.plot([10.2, 15.4], [1.3, 1.3], color=t["muted"], lw=1)
    ax.text(12.8, 2.8, "keyframe every\n5 frames", color=t["text2"], fontsize=9, ha="center")
    ax.text(12.8, 6.0, "6 to 15\nmodels", color=t["text2"], fontsize=9, ha="center")


def d_scene_models(ax, t):
    frame(ax, t)
    rng = np.random.default_rng(2)
    floor = [(x, y) for x in np.linspace(0.8, 15.2, 30) for y in np.linspace(0.6, 2.6, 5)]
    wall = [(x, y) for x in np.linspace(0.8, 15.2, 30) for y in np.linspace(3.4, 8.3, 8)]
    gaussian_cloud(ax, rng, wall, t["s1"], 260, 0.25, 0.28)
    gaussian_cloud(ax, rng, floor, t["s1"], 160, 0.2, 0.45)
    for x in (3.0, 5.4, 7.8):
        ax.add_patch(Rectangle((x, 2.8), 1.9, 2.2, fc=t["s1"], alpha=0.6, ec="none", zorder=3))


def d_person_ens(ax, t):
    frame(ax, t)
    for k in range(3):
        dx, dy = 1.0 + 1.0 * k, 0.5 + 0.8 * k
        ax.add_patch(Rectangle((dx, dy), 10, 6.2, fc=t["box"], ec=t["edge"], lw=1, zorder=2 + k))
        person(ax, dx + 5.0, dy + 0.5, 2.4, t["s2"], alpha=0.35 if k < 2 else 1.0, pose=k % 2, z=3 + k)
    ax.text(15.3, 1.0, "pixel\nmean", color=t["text2"], fontsize=9, ha="right")


def d_projection(ax, t):
    frame(ax, t)
    for x, lab in ((0.6, "raw\nrender"), (5.4, "Difix\ntarget")):
        ax.add_patch(Rectangle((x, 5.0), 4.2, 3.3, fc=t["box"], ec=t["edge"], lw=1, zorder=2))
        ax.text(x + 2.1, 6.65, lab, color=t["text2"], fontsize=9, ha="center", va="center")
    ax.add_patch(Rectangle((11.0, 2.2), 4.4, 3.6, fc=t["s1"], alpha=0.35, ec=t["edge"], lw=1, zorder=2))
    ax.text(13.2, 4.0, "x", color=t["text"], fontsize=12, ha="center", va="center", style="italic")
    for x0, lab in ((2.7, "SSIM"), (7.5, "LPIPS")):
        ax.add_patch(FancyArrowPatch((x0, 4.9), (10.9, 4.0), arrowstyle="-|>", mutation_scale=8,
                                     color=t["muted"], lw=1, connectionstyle="arc3,rad=0.25", zorder=3))
        ax.text(x0 + 0.6, 2.4, lab, color=t["text2"], fontsize=9, ha="center")


def d_difix(ax, t):
    frame(ax, t, fill=t["box"])
    person(ax, 8, 0.9, 3.8, t["s2"])
    for c, (ox, oy) in zip((t["ink"], t["s2"], t["s1"]), ((0, 0), (2.0, 1.8), (4.0, 1.0))):
        for x in np.arange(-6 + ox, 16, 6):
            ax.plot([x, x], [0, 9], color=c, lw=0.9, ls=(0, (3, 2)), alpha=0.65, zorder=3)
        for y in np.arange(-4.5 + oy, 9, 4.5):
            ax.plot([0, 16], [y, y], color=c, lw=0.9, ls=(0, (3, 2)), alpha=0.65, zorder=3)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)


def d_plate(ax, t):
    frame(ax, t)
    fr = Rectangle((0, 0), 16, 9, fc=t["s1"], alpha=0.25, ec="none", zorder=1)
    ax.add_patch(fr)
    for k, (x, y) in enumerate(((3.6, 0.7), (6.5, 1.1), (9.4, 0.6), (12.3, 1.0))):
        e = Ellipse((x, y + 3.0), 2.8, 6.9, fc=t["surface"], ec="none", zorder=2)
        ax.add_patch(e)
        e.set_clip_path(fr)
        person(ax, x, y, 2.9, t["s2"] if k == 3 else t["muted"], alpha=1.0 if k == 3 else 0.35, pose=k % 2, z=3)


def d_composite(ax, t):
    frame(ax, t)
    yy, xx = np.mgrid[0:9:180j, 0:16:320j]
    m = np.exp(-(((xx - 8) / 2.4) ** 2 + ((yy - 4.4) / 3.8) ** 2) ** 3)
    img = np.zeros(xx.shape + (4,))
    c2, c1 = matplotlib.colors.to_rgba(t["s2"]), matplotlib.colors.to_rgba(t["s1"])
    for i in range(3):
        img[..., i] = m * c2[i] + (1 - m) * c1[i]
    img[..., 3] = 0.22 + 0.4 * m
    ax.imshow(img, extent=(0, 16, 0, 9), origin="lower", zorder=1)
    person(ax, 8, 1.0, 3.6, t["s2"], z=3)


def d_colour(ax, t):
    frame(ax, t)
    x = np.linspace(0, 0.82, 100)
    ax.plot(1 + 14 * x, 1 + 7 * x, color=t["grid"], lw=2)
    ax.plot(1 + 14 * x, 1 + 7 * np.clip(x + 0.1 * x / (x + 0.1), 0, 1), color=t["s2"], lw=2)
    ax.plot(1 + 14 * x, 1 + 7 * np.clip(x * 1.15, 0, 1), color=t["s1"], lw=2)
    ax.text(15, 1.4, "taper: 011\ngain: 007", color=t["text2"], fontsize=9, ha="right")


def d_submission(ax, t):
    frame(ax, t)
    for r in range(5):
        for c in range(8):
            ax.add_patch(Rectangle((0.8 + c * 1.8, 0.6 + r * 1.45), 1.55, 1.2, fc=t["s1"] if r % 2 else t["s2"],
                                   alpha=0.22 + 0.12 * ((r + c) % 3), ec="none", zorder=2))
    ax.text(8, 8.3, "5 scenes x 8 views", color=t["text2"], fontsize=9, ha="center", va="center")


# --- layout -----------------------------------------------------------------------------------------------------
CARDS = [  # key, column, row, drawing, title, subtitle, measured gain
    ("in", 0, "mid", d_input, "6 cameras", "train on 6,\nrender 8 hidden\nviews at 4K", None),
    ("pm", 1, "top", d_person_models, "person models", "stride-5 keyframes,\nperson-weighted loss", "+0.23 dB FG-PSNR"),
    ("sm", 1, "bot", d_scene_models, "scene models", "dense depth, masked\npseudo-views", "+0.63 dB (val)*"),
    ("pe", 2, "top", d_person_ens, "person ensemble", "specialists +\ndiverse mix", "FG-LPIPS -0.027"),
    ("pr", 2, "bot", d_projection, "projection", "LPIPS to Difix,\nSSIM to raw", "+0.20 dB"),
    ("dx", 3, "top", d_difix, "Difix3D+ TTA", "3 shifted\ntile grids", "LPIPS -0.049"),
    ("pl", 3, "bot", d_plate, "static plate", "time average where\nthe person never is", "SSIM +0.0057"),
    ("cp", 4, "top", d_composite, "composite", "soft mask +\ncolour prior", "+0.68 dB (colour)"),
    ("sb", 4, "bot", d_submission, "2,056 images", "27.04 dB full\n25.78 dB FG", "3rd place"),
]
ARROWS = [("in", "pm"), ("in", "sm"), ("pm", "pe"), ("sm", "pe"), ("sm", "pr"), ("pe", "dx"), ("pr", "pl"),
          ("dx", "cp"), ("pl", "cp"), ("cp", "sb")]


def draw(mode):
    t = theme(mode)
    fig = plt.figure(figsize=(FW, FH), dpi=250)
    fig.patch.set_facecolor(t["surface"])
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    for name, x0, x1 in COLS:
        bg.add_patch(FancyBboxPatch((x0, 0.06), x1 - x0, 0.875, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=t["box"], ec="none", zorder=0))
        bg.text((x0 + x1) / 2, 0.965, name, color=t["muted"], fontsize=14, fontweight="bold", ha="center",
                va="center")
    bg.text(0.170, 0.905, "PERSON BRANCH", color=t["s2"], fontsize=10.5, fontweight="bold")
    bg.text(0.170, 0.478, "BACKGROUND BRANCH", color=t["s1"], fontsize=10.5, fontweight="bold")
    bg.text(0.005, 0.022, "Gains: official test scores; * = validation scene 001_1. Schematic drawings, no dataset "
            "images.", color=t["muted"], fontsize=10)
    boxes = {}
    for key, col, row, fn, title, sub, gain in CARDS:
        _, x0, x1 = COLS[col]
        pad = 0.012
        h = min(0.18, (x1 - x0 - 2 * pad) * (FW / FH) * 9 / 16)
        w = h * (FH / FW) * 16 / 9
        x = (x0 + x1) / 2 - w / 2
        y = ROW_Y[row]
        ax = fig.add_axes([x, y, w, h])
        fn(ax, t)
        cx = x + w / 2
        bg.text(cx, y - 0.032, title, color=t["text"], fontsize=13, fontweight="bold", ha="center", va="center")
        bg.text(cx, y - 0.058, sub, color=t["text2"], fontsize=11, ha="center", va="top", linespacing=1.2)
        if gain:
            nl = sub.count("\n") + 1
            bg.text(cx, y - 0.068 - 0.030 * nl, gain, color=t["good"], fontsize=11.5, fontweight="bold",
                    ha="center", va="top")
        boxes[key] = (x, y, w, h)
    for a, b in ARROWS:
        xa, ya, wa, ha = boxes[a]
        xb, yb, wb, hb = boxes[b]
        if xb > xa + wa:
            p0, p1 = (xa + wa + 0.005, ya + ha / 2), (xb - 0.005, yb + hb / 2)
        else:  # same column: top card feeds the card below
            p0, p1 = (xa + wa / 2, ya - 0.175), (xb + wb / 2, yb + hb + 0.008)
        bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=t["muted"], lw=1.4,
                                     connectionstyle="arc3,rad=0", zorder=5))
    fig.savefig(FIG / f"pipeline_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    for m in ("light", "dark"):
        draw(m)
    print("pipeline figures written")
