"""Build the data-only README figures (light + dark) and docs/data/*.csv.

Every number here was returned by the official evaluator (submissions) or measured on the public
validation scene 001_1 (per-view chart). No dataset pixels are drawn.
Usage: python docs/make_figures.py   (needs matplotlib)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parent
FIG = DOCS / "figures"
DATA = DOCS / "data"

# name, date (UTC), what changed, FULL psnr/ssim/lpips, FG psnr/ssim/lpips (None = FG metric not yet reported)
SUBMISSIONS = [
    ("baseline_submission", "09-03", "stock FreeTimeGS++, 30k iterations", 22.9251, 0.90516, 0.27468, None, None, None),
    ("F_s0", "09-03", "+ colour correction, LPIPS loss, sparse depth, opacity reg, near-camera cull", 23.4307, 0.90894, 0.26021, None, None, None),
    ("F_ens3", "09-03", "3-seed render ensemble", 24.5169, 0.92075, 0.24353, None, None, None),
    ("slotcheck", "09-04", "+ dense VGGT depth (recipe J), 2 seeds", 24.8510, 0.92143, 0.23921, None, None, None),
    ("submit_now", "09-04", "LPIPS weight 0.15 + SEVA pseudo-view members (P8/P9)", 25.0302, 0.92408, 0.23815, None, None, None),
    ("bestmix_sharp", "09-04", "per-scene best source + unsharp mask", 25.1986, 0.92080, 0.23373, None, None, None),
    ("s03ppj", "09-05", "4-member ensemble S03+P8+P9+J (LPIPS weight 0.3)", 25.5210, 0.92642, 0.23570, None, None, None),
    ("difixA_all", "09-06", "+ Difix3D+ (reference-guided, t=199), blend a~0.5", 26.0200, 0.91813, 0.18705, None, None, None),
    ("difixD_all", "09-06", "lighter per-view Difix blend (a~0.3)", 25.8871, 0.92297, 0.20040, 24.3456, 0.83804, 0.31322),
    ("difixD_t100_all", "09-06", "Difix at t=100", 25.8983, 0.92538, 0.21875, 24.4727, 0.84133, 0.35435),
    ("bbox_in30_out55", "09-07", "bbox-protected Difix blend (failed)", 25.9921, 0.91760, 0.19215, 24.4510, 0.84104, 0.35146),
    ("pb_p50_b45", "09-08", "person/background-decoupled Difix strength", 25.9436, 0.91855, 0.18649, 24.2482, 0.83247, 0.27716),
    ("six_p50_b45", "09-09", "6-member ensemble incl. full-res, gating, 4M-Gaussian members", 26.1365, 0.91977, 0.18907, 24.1259, 0.83049, 0.28070),
    ("SIXTTA_p50_b45", "09-11", "+ Difix 3-shift test-time augmentation", 26.1726, 0.92203, 0.19157, 24.2726, 0.83555, 0.28581),
    ("NINE_p50_b45", "09-11", "9-member ensemble (weaker members diluted it)", 25.9757, 0.92129, 0.19568, 24.1478, 0.83358, 0.29450),
    ("SIX_p070_bpAVt15", "09-12", "+ perceptual projection of the background", 26.0309, 0.92066, 0.18312, 24.2483, 0.83203, 0.26493),
    ("FGP8L20_p050", "09-12", "person region from two FG-specialist models", 26.0361, 0.92178, 0.18524, 24.5557, 0.83790, 0.28055),
    ("FGE_p085", "09-13", "person = 5-model FG-specialist ensemble", 26.0720, 0.92038, 0.18170, 24.4169, 0.82711, 0.24382),
    ("FGE6_p085", "09-13", "person = 6-model FG ensemble", 26.0861, 0.92053, 0.18125, 24.4921, 0.82914, 0.23779),
    ("FGE6P_p085", "09-13", "+ projection on the person (failed)", 26.0478, 0.91997, 0.18348, 24.2435, 0.82064, 0.26877),
    ("FGEP_p085", "09-13", "person = 5 seeds of one config", 26.0833, 0.92009, 0.18091, 24.4180, 0.82250, 0.23295),
    ("FGE8_p085", "09-14", "person = 8-model FG ensemble", 26.0911, 0.92056, 0.18108, 24.5471, 0.82992, 0.23541),
    ("S5E_p085", "09-14", "person models with stride-5 keyframes", 26.1111, 0.92083, 0.18088, 24.7808, 0.83414, 0.23225),
    ("S6E_p085", "09-14", "6 stride-5 seeds", 26.1137, 0.92085, 0.18077, 24.8088, 0.83496, 0.23061),
    ("HB3_p070_six", "09-15", "person = half stride-5, half diverse mix (alpha 0.70)", 26.1246, 0.92133, 0.18193, 24.9408, 0.84204, 0.24771),
    ("HB3_p085_six", "09-15", "same, alpha 0.85", 26.1219, 0.92123, 0.18123, 24.8984, 0.84055, 0.23739),
    ("HB3_p085_nbx15s", "09-15", "+ cross-target projected background", 26.3235, 0.92184, 0.18150, 25.0286, 0.84144, 0.23441),
    ("HB3_p085_nbx15_f101s4", "09-15", "+ wider composite feather", 26.3254, 0.92175, 0.18143, 25.0480, 0.84138, 0.23492),
    ("C0_hb3_nbx15_f101", "09-15", "feather 101 on the projected background", 26.3519, 0.92234, 0.18097, 25.1525, 0.84345, 0.23424),
    ("C1_hb3_tp15_f101", "09-15", "background projected toward the 6-member Difix-TTA target", 26.3500, 0.92183, 0.18140, 25.2347, 0.84387, 0.23188),
    ("C2_hb3_bp4_f101", "09-15", "background mix variant bp4", 26.3329, 0.92208, 0.18113, 25.1923, 0.84370, 0.23349),
    ("C3_hb3_tp15_f131", "09-15", "feather 131", 26.3514, 0.92186, 0.18152, 25.2412, 0.84429, 0.23346),
    ("C4_hb3_bp4_f81", "09-15", "feather 81", 26.3312, 0.92205, 0.18106, 25.1858, 0.84335, 0.23246),
    ("C1_011a095", "09-15", "stronger Difix on scene 011 only", 26.3509, 0.92185, 0.18130, 25.2405, 0.84406, 0.23052),
    ("C6_colorfix", "09-16", "additive per-camera colour prior on 007/011 (PSNR up, SSIM down)", 27.2079, 0.92107, 0.18170, 25.4277, 0.84029, 0.23208),
    ("C5_hb3v2_tp15_f101", "09-16", "+ full-res 4M-Gaussian SEVA member", 26.3500, 0.92180, 0.18140, 25.2366, 0.84390, 0.23220),
    ("C8_cc2", "09-16", "colour correction in fixed forms: taper (011), gain (007)", 27.1073, 0.92280, 0.18130, 25.4550, 0.84680, 0.23060),
    ("S1_splice", "09-16", "exact per-scene splice of scored packages", 26.4315, 0.92230, 0.18120, 25.2570, 0.84430, 0.23230),
    ("C13", "09-16", "9-seed person ensemble + colour fix", 27.1016, 0.92270, 0.18130, 25.5605, 0.84870, 0.22990),
    ("C12_patcam", "09-16", "colour pattern on shared cameras of 004/006/009", 27.1024, 0.92190, 0.18240, 25.3167, 0.84350, 0.23130),
    ("C9_hb3v4_tp15_f101", "09-16", "9-seed person ensemble, no colour fix", 26.3583, 0.92200, 0.18140, 25.3306, 0.84600, 0.23180),
    ("C25_a090", "09-16", "person Difix alpha 0.90", 27.0904, 0.92250, 0.18110, 25.4468, 0.84620, 0.22700),
    ("C24_meta", "09-16", "pixel average of scored packages", 27.0314, 0.92320, 0.18570, 25.5076, 0.84990, 0.23910),
    ("C29_q92", "09-16", "JPEG quality 92 (failed)", 27.0694, 0.91830, 0.19710, 25.4543, 0.84470, 0.23340),
    ("C28_ba090", "09-16", "background alpha 0.90", 27.0712, 0.92280, 0.18430, 25.4797, 0.84680, 0.23010),
    ("C25_a095", "09-16", "person Difix alpha 0.95", 27.0891, 0.92250, 0.18090, 25.4254, 0.84560, 0.22430),
    ("C16", "09-16", "person = 6 new stride-5 x pseudo-view models", 27.1159, 0.92290, 0.18110, 25.6965, 0.85100, 0.22670),
    ("C27_b100", "09-16", "pure temporal background plate (failed: mask mismatch)", 26.6234, 0.92840, 0.22290, 22.5538, 0.82080, 0.29100),
    ("C19_a095", "09-16", "C16 with alpha 0.95", 27.1128, 0.92270, 0.18070, 25.6457, 0.84950, 0.22120),
    ("C23_6shift", "09-16", "6-shift Difix TTA", 27.1176, 0.92290, 0.18110, 25.7076, 0.85170, 0.22680),
    ("C22", "09-16", "new models also in the diverse branch", 27.1218, 0.92290, 0.18100, 25.7361, 0.85120, 0.22520),
    ("C18", "09-16", "50/50 old vs new person models", 27.1057, 0.92270, 0.18120, 25.6005, 0.84970, 0.22840),
    ("C14_hb3v5_tp15_f101", "09-16", "wider person ensemble, no colour fix", 26.3587, 0.92200, 0.18140, 25.3371, 0.84610, 0.23150),
    ("C26_a085", "09-16", "second Difix pass", 27.1036, 0.92220, 0.18090, 25.4999, 0.84230, 0.22460),
    ("C35_p22_both", "09-17", "C22 person + new projected background + safe temporal plate", 27.0430, 0.92860, 0.20900, 25.7777, 0.85110, 0.22280),
    ("C38_silod", "09-17", "silhouette loss + opacity decay models (failed)", 26.7698, 0.92100, 0.18220, 23.6517, 0.82140, 0.24380),
    ("S3_recovery", "09-17", "exact per-scene splice (recovery after C38)", 27.1297, 0.92390, 0.18690, 25.5637, 0.84680, 0.22250),
    ("C43_011b", "09-17", "C35 + 15-member person ensemble for scene 011 (final)", 27.0411, 0.92870, 0.20910, 25.7787, 0.85170, 0.22390),
]

# Full-frame PSNR per hidden view, validation scene 001_1, native 4K, 3 frames per view (000000/000200/000400).
PER_VIEW_MODELS = [
    ("m_FGW20_s0", "FG weight 2, stride-10, 30k"),
    ("m_S5_FGW40_L60K", "FG weight 4, stride-5, 60k"),
    ("m_DFX_s0", "Difix-distilled, full-res"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", ring="#fcfcfb"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 s1="#3987e5", s2="#d95926", s3="#199e70", ring="#1a1a19"),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def write_csv():
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "submissions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "file", "date_utc", "change", "psnr", "ssim", "lpips", "fg_psnr", "fg_ssim", "fg_lpips"])
        for i, r in enumerate(SUBMISSIONS, 1):
            w.writerow([i, *r])


def progress(t, mode):
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    x = list(range(1, len(SUBMISSIONS) + 1))
    full = [r[3] for r in SUBMISSIONS]
    fgx = [i for i, r in zip(x, SUBMISSIONS) if r[6] is not None]
    fg = [r[6] for r in SUBMISSIONS if r[6] is not None]
    ax.scatter(x, full, s=64, color=t["s1"], edgecolors=t["ring"], linewidths=2, label="Full image", zorder=3)
    ax.scatter(fgx, fg, s=64, color=t["s2"], edgecolors=t["ring"], linewidths=2,
               label="Foreground (person crop)", zorder=3)
    ax.text(x[-1] + 0.9, full[-1], f"{full[-1]:.2f}  final", color=t["text"], va="center", fontsize=9)
    ax.text(fgx[-1] + 0.9, fg[-1], f"{fg[-1]:.2f}  final", color=t["text"], va="center", fontsize=9)

    names = [r[0] for r in SUBMISSIONS]
    notes = [
        ("baseline_submission", "full", "stock baseline", (1.5, -0.6)),
        ("F_ens3", "full", "seed ensemble", (1.5, -0.8)),
        ("s03ppj", "full", "SEVA pseudo-views\n+ LPIPS weight", (-5.5, 1.2)),
        ("difixA_all", "full", "Difix3D+", (3.0, 1.35)),
        ("S5E_p085", "fg", "stride-5 person models", (-3.0, -1.0)),
        ("C8_cc2", "full", "per-camera colour correction", (-14.0, 0.75)),
        ("C27_b100", "fg", "failed: temporal plate", (-12.0, -0.6)),
        ("C38_silod", "fg", "failed: silhouette +\nopacity decay", (-3.0, -1.3)),
    ]
    for name, series, label, (dx, dy) in notes:
        i = names.index(name)
        y = SUBMISSIONS[i][3] if series == "full" else SUBMISSIONS[i][6]
        ax.annotate(label, xy=(i + 1, y), xytext=(i + 1 + dx, y + dy), fontsize=8.5, color=t["text2"],
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8), ha="left", va="center")
    ax.set_xlim(0, len(x) + 7)
    ax.set_ylim(21.5, 28.5)
    ax.set_ylabel("PSNR (dB)", color=t["text2"], fontsize=10)
    ax.set_xlabel("scored submission, in upload order (09-03 to 09-17)", color=t["text2"], fontsize=10)
    ax.set_title("Official PSNR of every scored submission", color=t["text"], fontsize=13, loc="left", pad=26)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    fig.tight_layout()
    fig.savefig(FIG / f"progress_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def per_view(t, mode, pv):
    views = ["02", "10", "17", "18", "19", "33", "38", "46"]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    for i, v in enumerate(views):
        vals = [pv[m][v] for m, _ in PER_VIEW_MODELS]
        ax.plot([i, i], [min(vals), max(vals)], color=t["grid"], lw=2, zorder=2, solid_capstyle="round")
        ax.text(i + 0.12, max(vals), f"{max(vals):.1f}", va="center", color=t["text2"], fontsize=8.5)
    for (m, label), c in zip(PER_VIEW_MODELS, (t["s1"], t["s2"], t["s3"])):
        ax.scatter(range(len(views)), [pv[m][v] for v in views], s=64, color=c, edgecolors=t["ring"],
                   linewidths=2, label=label, zorder=3)
    ax.set_xticks(range(len(views)), [f"view {v}" for v in views])
    ax.set_xlim(-0.5, len(views) - 0.2)
    ax.set_ylim(17, 33)
    ax.set_ylabel("full-frame PSNR (dB)", color=t["text2"], fontsize=10)
    ax.set_title("Per-view PSNR on validation scene 001_1 (4K, 3 frames per view)", color=t["text"],
                 fontsize=13, loc="left", pad=26)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=3, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    fig.tight_layout()
    fig.savefig(FIG / f"per_view_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def write_per_view_csv(pv):
    with open(DATA / "val_001_1_per_view_psnr.csv", "w", newline="") as f:
        w = csv.writer(f)
        views = sorted(next(iter(pv.values())))
        w.writerow(["model"] + [f"view_{v}" for v in views])
        for m, vals in pv.items():
            w.writerow([m] + [f"{vals[v]:.3f}" for v in views])


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv()
    pv = json.load(open(DATA / "val_001_1_per_view_psnr.json"))
    write_per_view_csv(pv)
    for mode, t in THEMES.items():
        progress(t, mode)
        per_view(t, mode, pv)
    print("figures written to", FIG)
