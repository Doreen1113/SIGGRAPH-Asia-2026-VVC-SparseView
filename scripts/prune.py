"""Post-hoc pruning of floaters for sparse-view 4DGS.
Usage: ftg.sh prune.py <run_dir> <case_dir> <out_name.pt> [--min_views 2] [--max_scale S] [--near N]
Removes Gaussians (at their own time) that fall inside fewer than --min_views training frusta,
that have max scale > max_scale (meters), or that are closer than --near to any training camera."""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('out')
ap.add_argument('--filename', default='gaussians.pt'); ap.add_argument('--min_views', type=int, default=2)
ap.add_argument('--max_scale', type=float, default=1e9); ap.add_argument('--near', type=float, default=0.0)
ap.add_argument('--margin', type=float, default=0.0, help='frustum margin as fraction of image size (negative shrinks)')
a = ap.parse_args()
case = Path(a.case)
cams = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
with torch.no_grad():
    m = gs.means  # positions at their own time
    N = len(gs); count = torch.zeros(N, device=m.device, dtype=torch.int32); mind = torch.full((N,), 1e9, device=m.device)
    for name, cam in cams.items():
        K = torch.tensor(cam.K, dtype=torch.float32, device=m.device); W = 2 * K[0, 2]; H = 2 * K[1, 2]
        w2c = torch.tensor(cam.w2c, dtype=torch.float32, device=m.device)
        c = (w2c[:3, :3] @ m.T + w2c[:3, 3:4])  # 3xN
        z = c[2]; u = K[0, 0] * c[0] / z + K[0, 2]; v = K[1, 1] * c[1] / z + K[1, 2]
        mx, my = a.margin * W, a.margin * H
        inside = (z > 0.05) & (u >= -mx) & (u < W + mx) & (v >= -my) & (v < H + my)
        count += inside.int()
        cc = torch.tensor(-cam.w2c[:3, :3].T @ cam.w2c[:3, 3], dtype=torch.float32, device=m.device)
        mind = torch.minimum(mind, torch.norm(m - cc, dim=1))
    smax = gs.scales.exp().max(dim=1).values
    keep = (count >= a.min_views) & (smax <= a.max_scale) & (mind >= a.near)
    print(f'N={N} keep={int(keep.sum())} ({keep.float().mean()*100:.1f}%)  removed: few_views={int((count<a.min_views).sum())} big={int((smax>a.max_scale).sum())} near={int((mind<a.near).sum())}')
    print('scale percentiles (m):', np.percentile(smax.cpu().numpy(), [50, 90, 99, 99.9]).round(3))
    gs.mask(keep).save(Path(a.run) / a.out)
print('saved', Path(a.run) / a.out)
