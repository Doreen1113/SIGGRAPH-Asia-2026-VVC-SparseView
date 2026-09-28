"""Post-hoc floater removal using VGGT depth: a Gaussian is a floater if, in >= k training views where it projects,
its depth is in front of the VGGT surface by more than `margin` (relative). Uses the nearest keyframe depth to the Gaussian's time.
Usage: ftg.sh prune_vggt.py <run_dir> <case_dir> <depth_dir> <out.pt> [--k 2 --margin 0.15 --stride 10 --fps 60]"""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('depth'); ap.add_argument('out')
ap.add_argument('--filename', default='gaussians.pt'); ap.add_argument('--k', type=int, default=2); ap.add_argument('--margin', type=float, default=0.15)
ap.add_argument('--stride', type=int, default=10); ap.add_argument('--fps', type=float, default=60.0); ap.add_argument('--conf', type=float, default=2.0)
a = ap.parse_args(); case = Path(a.case)
cams = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); offsets = json.loads((case / 't_offsets.json').read_text())
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
keys = sorted(int(p.stem[1:]) for p in Path(a.depth).glob('f*.npz')); keys = [k for k in keys if k % a.stride == 0]
depths = {}
for k in keys:
    z = np.load(Path(a.depth) / f'f{k:06d}.npz'); views = [str(v) for v in z['views']]
    depths[k] = {v: (torch.from_numpy(z['depth'][i].astype(np.float32)).cuda(), torch.from_numpy(z['conf'][i].astype(np.float32)).cuda()) for i, v in enumerate(views)}
with torch.no_grad():
    t = gs.times.squeeze(-1); frame_of = (t * a.fps).round().long()
    key_t = torch.tensor(keys, device=t.device); nearest = key_t[torch.argmin(torch.abs(frame_of[:, None] - key_t[None]), dim=1)]
    m = gs.means_t(nearest.float() / a.fps) if False else gs.means  # use own-time position
    N = len(gs); front = torch.zeros(N, device=m.device, dtype=torch.int32); seen = torch.zeros(N, device=m.device, dtype=torch.int32)
    for name, cam in cams.items():
        K = torch.tensor(cam.K, dtype=torch.float32, device=m.device); W = 2 * K[0, 2]; H = 2 * K[1, 2]
        w2c = torch.tensor(cam.w2c, dtype=torch.float32, device=m.device); c = w2c[:3, :3] @ m.T + w2c[:3, 3:4]
        z = c[2]; u = K[0, 0] * c[0] / z + K[0, 2]; v = K[1, 1] * c[1] / z + K[1, 2]
        inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for k in keys:
            sel = inside & (nearest == k)
            if not sel.any(): continue
            d, cf = depths[k][name]; h, w = d.shape
            ui = (u[sel] * w / W).long().clamp(0, w - 1); vi = (v[sel] * h / H).long().clamp(0, h - 1)
            dz = d[vi, ui]; ok = cf[vi, ui] > a.conf
            idx = sel.nonzero(as_tuple=True)[0]
            seen[idx[ok]] += 1
            front[idx[ok & (z[sel] < (1 - a.margin) * dz)]] += 1
    drop = front >= a.k
    print(f'N={N} drop={int(drop.sum())} ({drop.float().mean()*100:.2f}%) seen-median={int(seen.float().median())}')
    gs.mask(~drop).save(Path(a.run) / a.out); print('saved')
