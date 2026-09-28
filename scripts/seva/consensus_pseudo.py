"""Build multi-seed-consensus pseudo views: average N seva samples of the same scene, keeping only
pixels where the samples AGREE (low cross-seed std) as supervision targets. Where samples disagree,
seva is guessing -> exclude from supervision (same spirit as the coverage mask).
Usage: ftg.sh consensus_pseudo.py <scenes_root> <seva_out_dirs comma-separated, in seed order>
       <case_dir> <case_prefix> <out_dir> [--std_thresh 0.06 --fps 60]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
ap = argparse.ArgumentParser()
ap.add_argument('scenes'); ap.add_argument('seva_outs'); ap.add_argument('case'); ap.add_argument('prefix'); ap.add_argument('out')
ap.add_argument('--std_thresh', type=float, default=0.06); ap.add_argument('--fps', type=float, default=60.0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
seva_outs = [Path(p) for p in a.seva_outs.split(',')]
offsets = json.loads((Path(a.case) / 't_offsets.json').read_text())
items = []; n_agree_frac = []
for sd in sorted(Path(a.scenes).glob(f'{a.prefix}_f*_s')):
    frame = int(sd.name[len(a.prefix) + 2:-2])
    meta = json.load(open(sd / 'transforms.json')); split = json.load(open(next(sd.glob('train_test_split_*.json'))))
    for j, i in enumerate(split['test_ids']):
        f = meta['frames'][i]; name = f['name']
        srcs = [so / sd.name / 'samples-rgb' / f'{j:03d}.png' for so in seva_outs]
        if not all(s.exists() for s in srcs): continue
        imgs = [cv2.imread(str(s)).astype(np.float32) / 255 for s in srcs]  # BGR, list of HxWx3
        stack = np.stack(imgs, 0)  # N,H,W,3
        mean = stack.mean(0); std = stack.std(0).mean(-1)  # H,W (avg over channels)
        agree = (std < a.std_thresh).astype(np.uint8) * 255
        n_agree_frac.append((agree > 0).mean())
        c2w = np.array(f['transform_matrix']); c2w[:, 1:3] *= -1; w2c = np.linalg.inv(c2w)
        K = [[f['fl_x'], 0, f['cx']], [0, f['fl_y'], f['cy']], [0, 0, 1]]
        (out / f'{frame:06d}').mkdir(exist_ok=True)
        cv2.imwrite(str(out / f'{frame:06d}' / f'{name}.png'), (mean * 255).round().astype(np.uint8))
        cv2.imwrite(str(out / f'{frame:06d}' / f'{name}_agree.png'), agree)
        items.append(dict(frame=frame, view=name, t=frame / a.fps - float(offsets.get(name, 0.0)), K=K, w2c=w2c.tolist(),
                           w=f['w'], h=f['h'], path=f'{frame:06d}/{name}.png'))
json.dump(items, open(out / 'index.json', 'w'))
print(f'{len(items)} consensus views written to {out}; mean agree fraction {np.mean(n_agree_frac):.3f} (n_seeds={len(seva_outs)})')
