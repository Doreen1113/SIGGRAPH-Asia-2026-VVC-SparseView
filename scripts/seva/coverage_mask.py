"""Precompute a per-pixel 'covered by train views' mask for each pseudo view (compliant: uses only the
6 train views' VGGT depth + a compliant reference model's rendered depth for unprojection).
Adapted from work/visfill_eval.py's `valid` computation. Writes <path>_covered.png (255=covered, do NOT
supervise there; 0=uncovered, safe to supervise with the pseudo-view loss) next to each pseudo png.
Usage: ftg.sh coverage_mask.py <ref_run_dir> <case_dir> <vggt_depth_dir> <pseudo_dir> [--tol 0.15 --conf 2.0 --dilate 3]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
ap = argparse.ArgumentParser()
ap.add_argument('ref_run'); ap.add_argument('case'); ap.add_argument('depth'); ap.add_argument('pseudo')
ap.add_argument('--tol', type=float, default=0.15); ap.add_argument('--conf', type=float, default=2.0)
ap.add_argument('--dilate', type=int, default=3); ap.add_argument('--filename', default='gaussians.pt')
ap.add_argument('--overwrite', action='store_true')
a = ap.parse_args(); case = Path(a.case)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
gs = Gaussians.load(Path(a.ref_run) / a.filename).cuda().eval()
depth_cache = {}
def vggt(frame):
    p = Path(a.depth) / f'f{frame:06d}.npz'
    if not p.exists():
        cands = sorted(Path(a.depth).glob('f*.npz'))
        p = min(cands, key=lambda q: abs(int(q.stem[1:]) - frame))
    if p not in depth_cache:
        z = np.load(p)
        depth_cache[p] = {str(v): (torch.from_numpy(z['depth'][i].astype(np.float32)).cuda(), torch.from_numpy(z['conf'][i].astype(np.float32)).cuda()) for i, v in enumerate(z['views'])}
    return depth_cache[p]
items = json.load(open(Path(a.pseudo) / 'index.json'))
n = 0
with torch.inference_mode():
    for e in items:
        dst = Path(a.pseudo) / e['path'].replace('.png', '_covered.png')
        if dst.exists() and not a.overwrite: continue
        Ws, Hs = int(e['w']), int(e['h'])
        w2c = torch.tensor(np.array(e['w2c'], dtype=np.float32)).cuda()
        Kt = torch.tensor(np.array(e['K'], dtype=np.float32)).cuda()
        img, alpha, meta = gs(t=torch.tensor(float(e['t'])).cuda(), w2c=w2c[None], intrinsic=Kt[None], shape=(Hs, Ws), render_depth=True)
        D = meta['depth'][0, ..., 0]
        u, v = torch.meshgrid(torch.arange(Ws, device='cuda').float(), torch.arange(Hs, device='cuda').float(), indexing='xy')
        rays = torch.stack([(u - Kt[0, 2]) / Kt[0, 0], (v - Kt[1, 2]) / Kt[1, 1], torch.ones_like(u)], -1)
        c2w = torch.inverse(w2c)
        P = rays * D[..., None]; Pw = P @ c2w[:3, :3].T + c2w[:3, 3]
        valid = torch.zeros((Hs, Ws), dtype=torch.bool, device='cuda')
        dd = vggt(int(e['frame']))
        for tn, tcam in tr.items():
            if tn not in dd: continue
            dz, dc = dd[tn]; h, w = dz.shape
            Kt_ = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt_[0, 2]; Ht = 2 * Kt_[1, 2]
            w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
            c3 = Pw @ w2[:3, :3].T + w2[:3, 3]; z = c3[..., 2]
            uu = (Kt_[0, 0] * c3[..., 0] / z + Kt_[0, 2]) * (w / Wt); vv = (Kt_[1, 1] * c3[..., 1] / z + Kt_[1, 2]) * (h / Ht)
            ok = (z > 0.05) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            ui = uu.clamp(0, w - 1).long(); vi = vv.clamp(0, h - 1).long()
            zs = dz[vi, ui]; cf = dc[vi, ui]
            consistent = ok & (cf > a.conf) & (torch.abs(z - zs) < a.tol * zs)
            valid |= consistent
        cov = (valid.cpu().numpy().astype(np.uint8)) * 255
        if a.dilate > 0: cov = cv2.dilate(cov, np.ones((a.dilate * 2 + 1,) * 2, np.uint8))
        cv2.imwrite(str(dst), cov); n += 1
print(f'wrote {n} coverage masks to {a.pseudo}')
