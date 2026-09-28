"""Average N Difix render trees (shifted tile grids) -> one tree. CPU multiprocess, resumable, jpg q95.
Usage: tta_average.py <dst_root> <src_root1> <src_root2> [...]  [--procs 64]"""
import sys, os, argparse
from multiprocessing import Pool
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument('dst'); ap.add_argument('srcs', nargs='+'); ap.add_argument('--procs', type=int, default=64)
a = ap.parse_args()
def work(rel):
    out = f'{a.dst}/{rel}'
    if os.path.exists(out): return 0
    raw = [cv2.imread(f'{s}/{rel}') for s in a.srcs]
    if any(i is None for i in raw): return -1   # a source that lacks this image -> skip, don't crash
    ims = [i.astype(np.float32) for i in raw]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.clip(np.mean(ims, 0), 0, 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
rels = []
for c in sorted(os.listdir(a.srcs[0])):
    for v in sorted(os.listdir(f'{a.srcs[0]}/{c}')):
        for f in sorted(os.listdir(f'{a.srcs[0]}/{c}/{v}')): rels.append(f'{c}/{v}/{f}')
cv2.setNumThreads(1)
with Pool(a.procs) as p: r = p.map(work, rels, chunksize=8)
print(f'TTA AVERAGE DONE: written {r.count(1)}, skipped {r.count(0)}, missing {r.count(-1)} of {len(rels)}')
