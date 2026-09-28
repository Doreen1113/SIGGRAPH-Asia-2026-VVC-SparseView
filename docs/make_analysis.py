"""Analysis figures for the README (light + dark). No dataset pixels: numbers are official evaluator scores or
validation metrics from docs/EXPERIMENT_LOG.md; drawings are schematic.
Usage: python docs/make_analysis.py   (needs matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, Polygon, Ellipse

from make_figures import FIG, THEMES, SUBMISSIONS, style

EXTRA = {
    "light": dict(good="#006300", box="#f0efec", edge="#d6d5ce", ink="#0b0b0b"),
    "dark": dict(good="#0ca30c", box="#262624", edge="#383835", ink="#ffffff"),
}
S = {r[0]: r for r in SUBMISSIONS}


def theme(mode):
    return {**THEMES[mode], **EXTRA[mode]}


def legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    return leg


def save(fig, name, mode, t):
    fig.savefig(FIG / f"{name}_{mode}.png", facecolor=t["surface"], bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# 1. What each change did to the official score (isolated pairs of scored submissions)
STAGES = [
    ("Recipe F: depth, LPIPS loss,\ncolour corr., opacity reg., cull", "baseline_submission", "F_s0"),
    ("3-seed render ensemble", "F_s0", "F_ens3"),
    ("Difix3D+ blend (a~0.5)", "s03ppj", "difixA_all"),
    ("Difix 3-shift TTA", "six_p50_b45", "SIXTTA_p50_b45"),
    ("Person = FG-specialist ensemble", "SIX_p070_bpAVt15", "FGE6_p085"),
    ("Stride-5 person models", "FGE8_p085", "S5E_p085"),
    ("Person = half specialists,\nhalf diverse mix", "S6E_p085", "HB3_p085_six"),
    ("Cross-target background\nprojection", "HB3_p085_six", "HB3_p085_nbx15s"),
    ("Per-camera colour correction", "S1_splice", "C8_cc2"),
    ("Stride-5 x pseudo-view\nperson models", "C13", "C16"),
    ("Static background plate", "C22", "C35_p22_both"),
    ("Scene-011 person ensemble\n(final)", "C35_p22_both", "C43_011b"),
]


def stage_gains(mode):
    t = theme(mode)
    cols = [("PSNR (dB)", 0, 1, 1), ("SSIM (x100)", 1, 1, 100), ("LPIPS decrease (x100)", 2, -1, 100)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    y = np.arange(len(STAGES))[::-1]
    h = 0.36
    for ax, (label, k, sign, mult) in zip(axes, cols):
        style(ax, t)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=t["grid"], lw=1)
        ax.axvline(0, color=t["muted"], lw=1)
        full, fg = [], []
        for _, a, b in STAGES:
            ra, rb = S[a], S[b]
            full.append(mult * sign * (rb[3 + k] - ra[3 + k]))
            fg.append(mult * sign * (rb[6 + k] - ra[6 + k]) if ra[6] is not None else np.nan)
        ax.barh(y + h / 2 + 0.02, full, height=h, color=t["s1"], label="Full image", zorder=3)
        ax.barh(y - h / 2 - 0.02, fg, height=h, color=t["s2"], label="Foreground", zorder=3)
        ax.set_title(label, color=t["text"], fontsize=11, loc="left")
        ax.tick_params(axis="x", labelsize=8.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[0].set_yticks(y, [s[0] for s in STAGES], fontsize=9, color=t["text2"])
    axes[0].tick_params(axis="y", colors=t["text2"])
    legend(axes[2], t, loc="lower right", ncol=1)
    fig.suptitle("What each change did to the official score (right = better)", color=t["text"], fontsize=13,
                 x=0.01, ha="left", y=0.99)
    fig.text(0.01, -0.02, "Each row compares two scored submissions that differ only in that change. "
             "Foreground metrics were not reported before 6 Sept.", color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    save(fig, "stage_gains", mode, t)


# ---------------------------------------------------------------------------------------------------------------
# 2. Difix strength: SSIM vs LPIPS frontier on validation scene 001_1 (same raw render for every point)
RAW = (0.2180, 0.9326)
SINGLE = [(0.30, 0.1840, 0.9311), (0.45, 0.1681, 0.9272), (0.50, 0.1653, 0.9270), (0.60, 0.1588, 0.9232),
          (0.70, 0.1564, 0.9209), (1.00, 0.1603, 0.9089)]
TTA = [(0.45, 0.1696, 0.9284), (0.60, 0.1585, 0.9252), (1.00, 0.1522, 0.9136)]
FINETUNED = [(0.30, 0.2043, 0.9325), (0.45, 0.1999, 0.9318), (0.60, 0.1968, 0.9304), (1.00, 0.1960, 0.9242)]
PROJ = [("toward single-pass Difix", 0.1558, 0.9277), ("toward TTA, two LPIPS nets", 0.1502, 0.9269)]


def difix_frontier(mode):
    t = theme(mode)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.xaxis.grid(True, color=t["grid"], lw=1)
    for pts, c, lab in ((SINGLE, t["s1"], "Difix3D+, single pass"), (TTA, t["s2"], "Difix3D+, 3-shift TTA"),
                        (FINETUNED, t["s3"], "Difix fine-tuned on our renders")):
        xs = [RAW[0]] + [p[1] for p in pts]
        ys = [RAW[1]] + [p[2] for p in pts]
        ax.plot(xs, ys, color=c, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter(xs[1:], ys[1:], s=64, color=c, edgecolors=t["ring"], linewidths=2, zorder=3, label=lab)
    for a, x, y in SINGLE:
        if a in (0.30, 0.70, 1.00):
            ax.text(x + 0.0012, y + 0.0006, f"a={a:.2f}", color=t["text2"], fontsize=8)
    ax.scatter([RAW[0]], [RAW[1]], s=90, facecolors=t["surface"], edgecolors=t["ink"], linewidths=2, zorder=4)
    ax.text(RAW[0] - 0.0015, RAW[1] + 0.0009, "raw ensemble render", color=t["text2"], fontsize=8.5, ha="right")
    for i, (lab, x, y) in enumerate(PROJ):
        ax.scatter([x], [y], s=80, marker="D", color=t["ink"], edgecolors=t["ring"], linewidths=1.5, zorder=4,
                   label="perceptual projection" if i == 0 else None)
    x0, y0 = PROJ[0][1], PROJ[0][2]
    ax.annotate("projection toward\nsingle-pass Difix", (x0, y0), xytext=(0.1470, 0.9322), fontsize=8.5,
                color=t["text2"], arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8))
    x, y = PROJ[1][1], PROJ[1][2]
    ax.scatter([x], [y], s=380, facecolors="none", edgecolors=t["ink"], linewidths=1.5, zorder=5)
    ax.annotate("used in the submission:\nprojection toward the TTA output,\nAlexNet + VGG LPIPS", (x, y),
                xytext=(0.170, 0.9150), fontsize=9, color=t["text"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8))
    ax.set_xlabel("LPIPS (AlexNet), lower is better", color=t["text2"], fontsize=10)
    ax.set_ylabel("SSIM, higher is better", color=t["text2"], fontsize=10)
    ax.set_xlim(0.145, 0.222)
    ax.set_ylim(0.9060, 0.9345)
    ax.set_title("Difix strength trades SSIM for LPIPS; TTA and projection move the frontier",
                 color=t["text"], fontsize=12, loc="left", pad=12)
    legend(ax, t, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.text(0.01, -0.01, "Validation scene 001_1, native 4K, full image. Lines start at the raw render and follow "
             "the blend weight a toward the restored image.", color=t["muted"], fontsize=8.5)
    save(fig, "difix_frontier", mode, t)


# ---------------------------------------------------------------------------------------------------------------
# 3. Person-source composition: foreground SSIM vs LPIPS, official scores
ENSEMBLE_PTS = [
    ("SIX_p070_bpAVt15", "6 diverse scene models", (6, -12)),
    ("FGE_p085", "5 FG specialists", (8, -2)),
    ("FGE6_p085", "6 FG specialists", (8, -2)),
    ("FGEP_p085", "5 seeds, one config", (8, -3)),
    ("FGE8_p085", "8 FG specialists", (8, 2)),
    ("S5E_p085", "stride-5 seeds", (-8, -12)),
    ("S6E_p085", "6 stride-5 seeds", (8, -4)),
    ("HB3_p085_six", "half stride-5, half diverse", (8, -3)),
    ("C16", "stride-5 x pseudo-view models", (8, -6)),
    ("C43_011b", "final", (8, 2)),
]


def ensemble_plane(mode):
    t = theme(mode)
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.xaxis.grid(True, color=t["grid"], lw=1)
    xs = [S[n][8] for n, _, _ in ENSEMBLE_PTS]
    ys = [S[n][7] for n, _, _ in ENSEMBLE_PTS]
    ax.plot(xs, ys, color=t["grid"], lw=2, zorder=1)
    ax.scatter(xs, ys, s=70, color=t["s2"], edgecolors=t["ring"], linewidths=2, zorder=3)
    for (n, lab, off), x, y in zip(ENSEMBLE_PTS, xs, ys):
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=t["text2"],
                    ha="left" if off[0] > 0 else "right")
    ax.annotate("", xy=(0.2445, 0.8412), xytext=(0.2445, 0.8285),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.2))
    ax.text(0.2452, 0.8348, "more diverse\nmembers:\nSSIM up", color=t["muted"], fontsize=8.5, va="center")
    ax.annotate("", xy=(0.2440, 0.8222), xytext=(0.2540, 0.8222),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.2))
    ax.text(0.2490, 0.8229, "more seeds of one config:\nLPIPS down", color=t["muted"], fontsize=8.5,
            ha="center", va="bottom")
    ax.set_ylim(0.8195, 0.8540)
    ax.set_xlabel("foreground LPIPS, lower is better", color=t["text2"], fontsize=10)
    ax.set_ylabel("foreground SSIM, higher is better", color=t["text2"], fontsize=10)
    ax.invert_xaxis()
    ax.set_title("Choosing the person source: official foreground scores", color=t["text"], fontsize=12,
                 loc="left", pad=12)
    fig.text(0.01, -0.01, "Each point is a scored submission; the line follows upload order. Better is up and "
             "to the right.", color=t["muted"], fontsize=8.5)
    save(fig, "ensemble_plane", mode, t)


# ---------------------------------------------------------------------------------------------------------------
# schematic helpers
def person(ax, cx, cy, s, color, alpha=1.0, pose=0.0, z=3):
    """Stick figure built from ellipses (a stand-in for a cloud of Gaussians)."""
    parts = [(0, 1.62, 0.26, 0.30, 0), (0, 1.15, 0.42, 0.72, 0),
             (-0.30 - 0.1 * pose, 1.12, 0.14, 0.62, 25 + 20 * pose), (0.30 + 0.1 * pose, 1.12, 0.14, 0.62, -25 - 20 * pose),
             (-0.13, 0.45, 0.17, 0.80, 8 + 12 * pose), (0.13, 0.45, 0.17, 0.80, -8 - 12 * pose)]
    for dx, dy, w, h, ang in parts:
        ax.add_patch(Ellipse((cx + dx * s, cy + dy * s), w * s, h * s, angle=ang, fc=color, ec="none",
                             alpha=alpha, zorder=z))


def camera(ax, x, y, ang, size, fc, ec, z=4):
    d = np.array([np.cos(ang), np.sin(ang)])
    n = np.array([-d[1], d[0]])
    tip = np.array([x, y]) + d * size
    base = np.array([x, y])
    pts = [base + n * size * 0.55, base - n * size * 0.55, tip]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.2, zorder=z))


def blank(ax, t):
    ax.set_facecolor(t["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------------------------------------------
# 4. Coverage-masked pseudo-views
def pseudo_views(mode):
    t = theme(mode)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=200, gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.patch.set_facecolor(t["surface"])
    blank(a1, t)
    a1.set_aspect("equal")
    W, H = 16.0, 9.0
    yy, xx = np.mgrid[0:H:360j, 0:W:640j]
    # uncovered = seen by no training camera: the far-right wall strip, a ceiling corner, floor behind the person
    unc = (xx > 12.6 + 0.9 * np.sin(yy * 0.9)) | ((xx < 3.2) & (yy > 6.4 + 0.6 * np.cos(xx))) | \
          (((xx - 9.6) / 1.5) ** 2 + ((yy - 2.2) / 0.8) ** 2 < 1)
    img = np.zeros(xx.shape + (4,))
    img[~unc] = matplotlib.colors.to_rgba(t["grid"])
    img[unc] = matplotlib.colors.to_rgba(t["s2"], 0.8)
    a1.imshow(img, extent=(0, W, 0, H), origin="lower", zorder=1, interpolation="bilinear")
    a1.add_patch(Rectangle((0, 0), W, H, fc="none", ec=t["edge"], lw=1, zorder=2))
    person(a1, 8.2, 1.0, 3.4, t["ink"], z=3)
    a1.text(3.4, 3.2, "covered by at least one\ntraining camera:\npseudo-view loss off", color=t["text"],
            fontsize=9, ha="center", va="center", zorder=4)
    a1.text(14.4, 4.5, "seen by no\ntraining\ncamera:\nloss on", color=t["surface"], fontsize=9, ha="center",
            va="center", zorder=4, fontweight="bold")
    a1.set_xlim(-0.2, W + 0.2)
    a1.set_ylim(-0.2, H + 0.2)
    a1.set_title("Coverage mask of one hidden view (schematic)", color=t["text"], fontsize=11, loc="left")
    # right: validation results
    style(a2, t)
    rows = [("baseline recipe F", 24.565, "ink"), ("F + pseudo-views, all pixels", 23.614, "ink"),
            ("F + pseudo-views, uncovered only (P8)", 25.059, "s2"), ("recipe J (dense depth)", 25.156, "ink"),
            ("J + P8 ensemble", 25.794, "s2")]
    yy = np.arange(len(rows))[::-1]
    a2.yaxis.grid(False)
    a2.xaxis.grid(True, color=t["grid"], lw=1)
    for (lab, v, c), y in zip(rows, yy):
        a2.plot([23.3, v], [y, y], color=t["grid"], lw=2, zorder=1)
        a2.scatter([v], [y], s=80, color=t[c], edgecolors=t["ring"], linewidths=2, zorder=3)
        a2.text(v + 0.05, y + 0.18, f"{v:.2f}", color=t["text"], fontsize=9)
    a2.set_yticks(yy, [r[0] for r in rows], color=t["text2"], fontsize=9)
    a2.tick_params(axis="y", colors=t["text2"])
    a2.set_xlim(23.3, 26.2)
    a2.set_ylim(-0.6, len(rows) - 0.3)
    a2.set_xlabel("PSNR on validation scene 001_1 (dB)", color=t["text2"], fontsize=10)
    a2.set_title("Masking turns a loss into a gain", color=t["text"], fontsize=11, loc="left")
    fig.tight_layout(w_pad=3)
    save(fig, "pseudo_views", mode, t)


# ---------------------------------------------------------------------------------------------------------------
# 5. Static background plate
def static_plate(mode):
    t = theme(mode)
    fig = plt.figure(figsize=(12.5, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.25)
    titles = ["Unsafe (C27): plate everywhere\nexcept this frame's mask",
              "Safe (C35): plate only outside the\nunion of every frame's mask"]
    frames = [(-1.6, 0.0), (-0.7, 0.6), (0.4, -0.2), (1.3, 0.4)]
    for k in range(2):
        ax = fig.add_subplot(gs[0, k])
        blank(ax, t)
        ax.set_aspect("equal")
        frame = Rectangle((-3.2, -1.8), 6.4, 3.6, fc=t["s1"], alpha=0.18, ec=t["edge"], lw=1, zorder=0)
        ax.add_patch(frame)
        if k == 1:
            for fx, fy in frames:
                e = Ellipse((fx, fy - 0.05), 1.25, 2.6, fc=t["surface"], ec="none", zorder=1)
                ax.add_patch(e)
                e.set_clip_path(frame)
        for i, (fx, fy) in enumerate(frames):
            last = i == len(frames) - 1
            person(ax, fx, fy - 1.0, 1.05, t["s2"] if last else t["muted"], alpha=1.0 if last else 0.35, pose=i % 2,
                   z=3 if last else 2)
        if k == 0:
            ax.add_patch(Ellipse((frames[-1][0], frames[-1][1] - 0.05), 1.25, 2.6, fc="none", ec=t["ink"], lw=1.2,
                                 ls="--", zorder=4))
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-1.9, 1.9)
        ax.set_title(titles[k], color=t["text"], fontsize=10, loc="left")
    fig.text(0.125, 0.08, "tinted: temporal mean of the view   untinted: per-frame render kept   "
             "grey: the person in other frames", color=t["muted"], fontsize=8.5)
    ax = fig.add_subplot(gs[0, 2])
    style(ax, t)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=t["grid"], lw=1)
    ax.axvline(0, color=t["muted"], lw=1)
    unsafe = [S["C27_b100"][i] - S["C13"][i] for i in (4, 6, 7)]
    safe = [S["C35_p22_both"][i] - S["C22"][i] for i in (4, 6, 7)]
    labels = ["full SSIM (x100)", "FG PSNR (dB)", "FG SSIM (x100)"]
    scale = [100, 1, 100]
    y = np.arange(3)[::-1]
    ax.barh(y + 0.19, [u * s for u, s in zip(unsafe, scale)], height=0.34, color=t["muted"], label="unsafe (C27)")
    ax.barh(y - 0.19, [u * s for u, s in zip(safe, scale)], height=0.34, color=t["s1"], label="safe (C35)")
    ax.set_yticks(y, labels, color=t["text2"], fontsize=9)
    ax.tick_params(axis="y", colors=t["text2"])
    ax.set_title("Official change vs. the same package without it", color=t["text"], fontsize=10, loc="left")
    legend(ax, t, loc="lower left")
    save(fig, "static_plate", mode, t)


# ---------------------------------------------------------------------------------------------------------------
# 6. Colour correction for hidden cameras
def colour(mode):
    t = theme(mode)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(a1, t)
    a1.xaxis.grid(True, color=t["grid"], lw=1)
    x = np.linspace(0, 1, 200)
    d, tau, g = 0.06, 0.10, 1.10
    a1.plot(x, x, color=t["grid"], lw=2, label="uncorrected")
    a1.plot(x, np.clip(x + d, 0, 1), color=t["muted"], lw=2, label="additive (x + d)")
    a1.plot(x, np.clip(x + d * x / (x + tau), 0, 1), color=t["s2"], lw=2, label="taper, used on 011")
    a1.plot(x, np.clip(x * g, 0, 1), color=t["s1"], lw=2, label="gain, used on 007")
    a1.set_xlim(0, 0.4)
    a1.set_ylim(0, 0.46)
    a1.set_xlabel("input intensity (dark end)", color=t["text2"], fontsize=10)
    a1.set_ylabel("corrected intensity", color=t["text2"], fontsize=10)
    a1.set_title("Three ways to apply the same camera offset (illustrative parameters)", color=t["text"],
                 fontsize=10.5, loc="left")
    legend(a1, t, loc="upper left")
    style(a2, t)
    bins = ["0-0.10", "0.10-0.25", "0.25-0.50"]
    vals = [-0.042, 0.0, 0.005]
    a2.axhline(0, color=t["muted"], lw=1)
    a2.bar(range(3), vals, width=0.5, color=t["muted"], zorder=3)
    for i, v in enumerate(vals):
        a2.text(i, v + (0.002 if v >= 0 else -0.002), f"{v:+.3f}" if v else "0.000", ha="center", va="bottom" if v >= 0 else "top",
                color=t["text"], fontsize=9)
    a2.set_xticks(range(3), bins)
    a2.set_ylim(-0.05, 0.012)
    a2.set_xlabel("mean intensity of the SSIM window", color=t["text2"], fontsize=10)
    a2.set_ylabel("SSIM change", color=t["text2"], fontsize=10)
    a2.set_title("Additive offset: SSIM loss sits in dark windows (validation)", color=t["text"], fontsize=10.5,
                 loc="left")
    fig.tight_layout(w_pad=3)
    save(fig, "colour", mode, t)


if __name__ == "__main__":
    for m in THEMES:
        stage_gains(m)
        difix_frontier(m)
        ensemble_plane(m)
        pseudo_views(m)
        static_plate(m)
        colour(m)
    print("analysis figures written to", FIG)
