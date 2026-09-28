"""Weight-average render trees into a combined ensemble render tree.
An ensemble render is just the mean of its members' renders, so an existing k-member average can be folded in
with weight k alongside individually-rendered new members.
Usage: weighted_ensemble.py <dst> <root1>:<w1> <root2>:<w2> ... [--procs 64]
Example (6-member average + 3 new members = 9-member ensemble):
  weighted_ensemble.py out/renders SIX_ens/renders:6 M_G3M_FR_s0/renders:1 M_GATE_FR_s0/renders:1 M_FULLRES_s1/renders:1
"""
import sys, os, argparse
from multiprocessing import Pool
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument('dst'); ap.add_argument('srcs', nargs='+'); ap.add_argument('--procs', type=int, default=64)
a = ap.parse_args()
PAIRS = []
for s in a.srcs:
    root, w = s.rsplit(':', 1); PAIRS.append((root, float(w)))
TOT = sum(w for _, w in PAIRS)
def work(rel):
    out = f'{a.dst}/{rel}'
    if os.path.exists(out): return 0
    acc = None
    for root, w in PAIRS:
        im = cv2.imread(f'{root}/{rel}')
        if im is None: return -1
        acc = im.astype(np.float32)*w if acc is None else acc + im.astype(np.float32)*w
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.clip(acc/TOT, 0, 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
base = PAIRS[0][0]
rels = [f'{c}/{v}/{f}' for c in sorted(os.listdir(base)) for v in sorted(os.listdir(f'{base}/{c}')) for f in sorted(os.listdir(f'{base}/{c}/{v}'))]
cv2.setNumThreads(1)
with Pool(a.procs) as p: r = p.map(work, rels, chunksize=8)
print(f'WEIGHTED ENSEMBLE DONE ({" + ".join(f"{os.path.basename(os.path.dirname(k))}x{w:g}" for k, w in PAIRS)}): '
      f'written {r.count(1)}, skipped {r.count(0)}, missing {r.count(-1)} of {len(rels)}')
