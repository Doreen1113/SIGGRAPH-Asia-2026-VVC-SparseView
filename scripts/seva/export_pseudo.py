"""Collect seva outputs for a case into a pseudo-view supervision dir with index.json.
Usage: ftg.sh export_pseudo.py <scenes_root> <seva_out_root> <case_dir> <case_prefix e.g. 001_1> <out_dir> [--fps 60]
Scene dirs are <prefix>_f<frame>_s with transforms.json + train_test_split_6.json; seva outputs in <seva_out_root>/<scene>/samples-rgb/<j:03d>.png"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('scenes'); ap.add_argument('seva_out'); ap.add_argument('case'); ap.add_argument('prefix'); ap.add_argument('out')
ap.add_argument('--fps', type=float, default=60.0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
offsets = json.loads((Path(a.case) / 't_offsets.json').read_text())
items = []; missing = []
for sd in sorted(Path(a.scenes).glob(f'{a.prefix}_f*_s')):
    frame = int(sd.name[len(a.prefix) + 2:-2])
    so = Path(a.seva_out) / sd.name / 'samples-rgb'
    if not so.exists(): missing.append(frame); continue
    meta = json.load(open(sd / 'transforms.json')); split = json.load(open(next(sd.glob('train_test_split_*.json'))))
    for j, i in enumerate(split['test_ids']):
        f = meta['frames'][i]; name = f['name']
        src = so / f'{j:03d}.png'
        if not src.exists(): missing.append((frame, name)); continue
        c2w = np.array(f['transform_matrix']); c2w[:, 1:3] *= -1; w2c = np.linalg.inv(c2w)
        K = [[f['fl_x'], 0, f['cx']], [0, f['fl_y'], f['cy']], [0, 0, 1]]
        (out / f'{frame:06d}').mkdir(exist_ok=True); dst = out / f'{frame:06d}' / f'{name}.png'; shutil.copyfile(src, dst)
        items.append(dict(frame=frame, view=name, t=frame / a.fps - float(offsets.get(name, 0.0)), K=K, w2c=w2c.tolist(), w=f['w'], h=f['h'], path=f'{frame:06d}/{name}.png'))
json.dump(items, open(out / 'index.json', 'w'))
print(f'{len(items)} pseudo views written to {out}; missing: {missing[:10]}{"..." if len(missing) > 10 else ""}')
