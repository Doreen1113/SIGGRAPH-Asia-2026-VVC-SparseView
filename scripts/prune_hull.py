"""Compliant floater removal: dynamic Gaussians (temporal_scale < dyn_thr) must lie inside the visual hull of the person
(from the 6 train views) at their own time; those outside (beyond `margin` m) are removed.
Usage: ftg.sh prune_hull.py <run_dir> <case_dir> <hull_dir> <out.pt> [--dyn_thr 5 --margin 0.15 --fps 60]"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.models.gaussians import Gaussians
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('hull'); ap.add_argument('out')
ap.add_argument('--filename', default='gaussians.pt'); ap.add_argument('--dyn_thr', type=float, default=5.0); ap.add_argument('--margin', type=float, default=0.15); ap.add_argument('--fps', type=float, default=60.0)
a = ap.parse_args()
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
hulls = {int(p.stem.split('_')[1]): np.load(p) for p in sorted(Path(a.hull).glob('hull_*.npz'))}
keys = np.array(sorted(hulls)); voxel = float(hulls[keys[0]]['voxel'])
with torch.no_grad():
    ts = gs.temporal_scale().squeeze(-1); dyn = ts < a.dyn_thr
    frame = (gs.times.squeeze(-1) * a.fps).round().long().cpu().numpy()
    kidx = np.abs(frame[:, None] - keys[None]).argmin(1); kf = keys[kidx]
    keep = torch.ones(len(gs), dtype=torch.bool, device='cuda'); means = gs.means
    n_dyn = int(dyn.sum()); removed = 0
    for k in keys:
        sel = dyn & torch.from_numpy(kf == k).cuda()
        if not sel.any(): continue
        pts = torch.from_numpy(hulls[k]['pts']).cuda()
        if len(pts) == 0: keep[sel] = False; removed += int(sel.sum()); continue
        # occupancy grid
        lo = pts.min(0).values - a.margin - voxel; res = voxel
        idx = ((pts - lo) / res).long(); dims = idx.max(0).values + 1
        grid = torch.zeros(tuple(dims.tolist()), dtype=torch.bool, device='cuda'); grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        # dilate by margin (in voxels) via max-pool
        r = int(np.ceil(a.margin / res))
        if r > 0:
            g = torch.nn.functional.max_pool3d(grid[None, None].float(), kernel_size=2 * r + 1, stride=1, padding=r)[0, 0] > 0
        else: g = grid
        m = means[sel]; gi = ((m - lo) / res).long(); inside = ((gi >= 0) & (gi < dims)).all(1)
        ok = torch.zeros(len(m), dtype=torch.bool, device='cuda'); gi_in = gi[inside]; ok[inside] = g[gi_in[:, 0], gi_in[:, 1], gi_in[:, 2]]
        sel_idx = sel.nonzero(as_tuple=True)[0]; keep[sel_idx[~ok]] = False; removed += int((~ok).sum())
    print(f'N={len(gs)} dynamic={n_dyn} removed outside hull={removed} ({removed/len(gs)*100:.1f}%)')
    gs.mask(keep).save(Path(a.run) / a.out); print('saved')
