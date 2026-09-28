"""Per-view coverage + per-view model comparison (compliant: coverage uses only the 6 training cameras).
For each hidden view: (a) compute the fraction of rendered-surface pixels seen consistently by >=1 training camera,
(b) score each candidate model. Then report whether coverage predicts which model wins, and the oracle/rule-based mean.
Usage: ftg.sh coverage_select.py <case_dir> <vggt_depth_dir> <run_A> <run_B> [...] [--scale 0.5 --every 20]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('depth'); ap.add_argument('runs', nargs='+')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--every', type=int, default=20); ap.add_argument('--fps', type=float, default=60.0)
ap.add_argument('--tol', type=float, default=0.15); ap.add_argument('--conf', type=float, default=2.0); ap.add_argument('--stride', type=int, default=10)
ap.add_argument('--cull_near_frac', type=float, default=1.1); ap.add_argument('--cull_min_views', type=int, default=4)
ap.add_argument('--out_json', default=None)
a = ap.parse_args(); case = Path(a.case)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); te = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
offsets = json.loads((case / 't_offsets.json').read_text())
centers = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tr.values()]); ctr = centers.mean(0)
cam_radius = float(np.linalg.norm(centers - ctr, axis=1).mean())
models = {}
for r in a.runs:
    g = Gaussians.load(Path(r) / 'gaussians.pt').cuda().eval()
    cnt = torch.zeros(len(g), device='cuda', dtype=torch.int32)
    for tn, tcam in tr.items():
        K = torch.tensor(tcam.K, dtype=torch.float32).cuda(); W = 2 * K[0, 2]; H = 2 * K[1, 2]; w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
        c3 = w2[:3, :3] @ g.means.T + w2[:3, 3:4]; z = c3[2]; u = K[0, 0] * c3[0] / z + K[0, 2]; v = K[1, 1] * c3[1] / z + K[1, 2]
        cnt += ((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).int()
    models[Path(r).name] = (g, cnt)
depth_cache = {}
def vggt(frame):
    k = (frame // a.stride) * a.stride
    if k not in depth_cache:
        p = Path(a.depth) / f'f{k:06d}.npz'
        if not p.exists(): p = sorted(Path(a.depth).glob('f*.npz'))[-1]
        z = np.load(p); depth_cache[k] = {str(v): (torch.from_numpy(z['depth'][i].astype(np.float32)).cuda(), torch.from_numpy(z['conf'][i].astype(np.float32)).cuda()) for i, v in enumerate(z['views'])}
    return depth_cache[k]
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
def metr(p, gt):
    return (psnr_f(p[None].permute(0,3,1,2).clamp(0,1), gt[None].permute(0,3,1,2)).item(),
            ssim_f(p[None].permute(0,3,1,2).clamp(0,1), gt[None].permute(0,3,1,2)).item(),
            lpips_f(p[None].permute(0,3,1,2).clamp(0,1), gt[None].permute(0,3,1,2)).item())
rows = {}
with torch.inference_mode():
    for name, cam in te.items():
        frames = sorted((case / 'images' / name).glob('*.jpg'))
        if not frames: continue
        K = cam.K.copy(); W = int(round(2*K[0,2])); H = int(round(2*K[1,2])); Ws, Hs = int(round(W*a.scale)), int(round(H*a.scale)); K[0,:] *= Ws/W; K[1,:] *= Hs/H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda(); Kt = torch.tensor(K, dtype=torch.float32).cuda(); c2w = torch.inverse(w2c)
        cc = c2w[:3, 3]; off = float(offsets.get(name, 0.0))
        uu, vv = torch.meshgrid(torch.arange(Ws, device='cuda').float(), torch.arange(Hs, device='cuda').float(), indexing='xy')
        rays = torch.stack([(uu-Kt[0,2])/Kt[0,0], (vv-Kt[1,2])/Kt[1,1], torch.ones_like(uu)], -1)
        # geometric coverage proxy: angle to the nearest training camera and distance outside the camera hull
        d_hidden = np.linalg.norm(np.asarray(-cam.w2c[:3,:3].T @ cam.w2c[:3,3]) - ctr)
        angs = []
        for tcam in tr.values():
            v1 = np.asarray(-cam.w2c[:3,:3].T @ cam.w2c[:3,3]) - ctr; v2 = np.asarray(-tcam.w2c[:3,:3].T @ tcam.w2c[:3,3]) - ctr
            angs.append(np.degrees(np.arccos(np.clip(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2), -1, 1))))
        min_ang = float(np.min(angs))
        scores = {k: [] for k in models}; covs = []
        for i in range(0, len(frames), a.every):
            gt = cv2.imread(str(frames[i]))[..., ::-1]; gt = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_AREA)
            gt = torch.from_numpy(np.ascontiguousarray(gt)).cuda().float()/255
            t = torch.tensor(i/a.fps - off).cuda()
            for k, (g, cnt) in models.items():
                near = (torch.norm(g.means - cc, dim=1) < a.cull_near_frac*cam_radius) & (cnt < a.cull_min_views); gv = g.mask(~near)
                img, _, meta = gv(t=t, w2c=w2c[None], intrinsic=Kt[None], shape=(Hs, Ws), render_depth=True)
                R = img[0].clamp(0,1); scores[k].append(metr(R, gt))
                if k == list(models)[0]:
                    Pw = (rays * meta['depth'][0,...,0][...,None]) @ c2w[:3,:3].T + c2w[:3,3]
                    valid = torch.zeros((Hs, Ws), dtype=torch.bool, device='cuda'); dd = vggt(i)
                    for tn, tcam in tr.items():
                        if tn not in dd: continue
                        dz, dc = dd[tn]; h, w = dz.shape
                        Ktr = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2*Ktr[0,2]; Ht = 2*Ktr[1,2]; w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
                        c3 = Pw @ w2[:3,:3].T + w2[:3,3]; z = c3[...,2]
                        pu = (Ktr[0,0]*c3[...,0]/z + Ktr[0,2])*(w/Wt); pv = (Ktr[1,1]*c3[...,1]/z + Ktr[1,2])*(h/Ht)
                        ok = (z>0.05)&(pu>=0)&(pu<w)&(pv>=0)&(pv<h)
                        zs = dz[pv.clamp(0,h-1).long(), pu.clamp(0,w-1).long()]; cf = dc[pv.clamp(0,h-1).long(), pu.clamp(0,w-1).long()]
                        valid |= ok & (cf > a.conf) & (torch.abs(z-zs) < a.tol*zs)
                    covs.append(float(valid.float().mean()))
        rows[name] = dict(coverage=float(np.mean(covs)), min_angle_deg=min_ang, **{k: float(np.mean([s[0] for s in v])) for k, v in scores.items()},
                          **{f'{k}_ssim': float(np.mean([s[1] for s in v])) for k, v in scores.items()},
                          **{f'{k}_lpips': float(np.mean([s[2] for s in v])) for k, v in scores.items()})
        keys = list(models)
        print(f"view {name}: coverage={rows[name]['coverage']:.3f} min_ang={min_ang:5.1f}deg  " + "  ".join(f'{k}={rows[name][k]:.2f}' for k in keys), flush=True)
keys = list(models)
print()
for k in keys: print(f'MEAN {k}: psnr={np.mean([r[k] for r in rows.values()]):.3f} ssim={np.mean([r[k+"_ssim"] for r in rows.values()]):.4f} lpips={np.mean([r[k+"_lpips"] for r in rows.values()]):.4f}')
best = np.mean([max(r[k] for k in keys) for r in rows.values()]); print(f'ORACLE per-view best: psnr={best:.3f}')
for thr in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
    sel = [ (keys[-1] if r['coverage'] < thr else keys[0]) for r in rows.values() ]
    v = np.mean([rows[n][s] for n, s in zip(rows, sel)])
    print(f'  rule coverage<{thr}: use {keys[-1]} else {keys[0]} -> psnr={v:.3f}  ({sum(1 for s in sel if s==keys[-1])} views switched)')
if a.out_json: json.dump(rows, open(a.out_json, 'w'), indent=1)
