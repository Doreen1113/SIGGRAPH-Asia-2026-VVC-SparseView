"""Trimmed-mean render ensemble: per pixel, drop the k lowest and k highest member values and average the rest.

Every ensemble so far is a plain pixel mean. A mean of renders that disagree on geometry blurs the disagreement
(lower SSIM) and lets one member's floaters leak into the average. Trimming the extremes per pixel keeps the
noise-averaging of a mean while discarding outlier members, which is where floaters and mis-placed limbs live.
Usage: weighted_ensemble_trim.py <dst> <root1> <root2> ... [--trim 1] [--procs 64]
"""
import sys, os, argparse
from multiprocessing import Pool
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument('dst'); ap.add_argument('srcs', nargs='+')
ap.add_argument('--trim', type=int, default=1); ap.add_argument('--procs', type=int, default=64)
a = ap.parse_args()
def work(rel):
    out = f'{a.dst}/{rel}'
    if os.path.exists(out): return 0
    ims = []
    for root in a.srcs:
        im = cv2.imread(f'{root}/{rel}')
        if im is None: return -1
        ims.append(im.astype(np.float32))
    st = np.sort(np.stack(ims, 0), axis=0)          # (M,H,W,3) sorted per pixel/channel
    k = a.trim
    core = st[k:len(ims) - k] if len(ims) > 2 * k else st
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.clip(core.mean(0), 0, 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
base = a.srcs[0]
rels = [f'{c}/{v}/{f}' for c in sorted(os.listdir(base)) for v in sorted(os.listdir(f'{base}/{c}')) for f in sorted(os.listdir(f'{base}/{c}/{v}'))]
cv2.setNumThreads(1)
with Pool(a.procs) as p: r = p.map(work, rels, chunksize=8)
print(f'TRIMMED ENSEMBLE DONE (trim {a.trim} of {len(a.srcs)}): written {r.count(1)}, skipped {r.count(0)}, missing {r.count(-1)} of {len(rels)}')
