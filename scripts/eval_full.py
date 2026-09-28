"""Local full-resolution evaluation on val cases (test views with GT).
Usage: ftg.sh work/eval_full.py <run_dir> <case_dir> [--every 10] [--scale 1.0] [--filename gaussians.pt] [--save_dir DIR]
Metrics: PSNR/SSIM/LPIPS(alex) averaged over (test view, frame) like the challenge."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('case')
ap.add_argument('--every', type=int, default=10); ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--filename', default='gaussians.pt'); ap.add_argument('--fps', type=float, default=60.0)
ap.add_argument('--save_dir', default=None); ap.add_argument('--views', default=None)
ap.add_argument('--extra_runs', default=None, help='comma list of extra run dirs to ensemble (average renders)')
ap.add_argument('--cull_near', type=float, default=0.0, help='drop Gaussians within this distance (m) of the render camera')
ap.add_argument('--unsharp', default='', help='amount,radius post-process sharpening, e.g. 0.4,4')
ap.add_argument('--ens_median', action='store_true', help='pixel-wise median across ensemble members instead of mean')
ap.add_argument('--temporal_avg', default='', help='k,dt : also render at t±j*dt for j=1..k and average (temporal smoothing)')
ap.add_argument('--supersample', type=int, default=1, help='render at NxN the target resolution then box-downsample (anti-aliasing)')
ap.add_argument('--weights', default=None, help='comma-separated ensemble weights for [main]+extra_runs')
ap.add_argument('--cull_global_views', type=int, default=0, help='drop Gaussians seen by fewer than N training views REGARDLESS of distance (targets far oblique haze that survives the near cull)')
ap.add_argument('--cull_max_opacity', type=float, default=0.0, help='with cull_global_views: only drop those whose opacity is below this (semi-transparent haze)')
ap.add_argument('--cull_near_frac', type=float, default=0.0)
ap.add_argument('--big_scale', type=float, default=0.0, help='also drop Gaussians with max scale > this (m) within big_near of the render camera')
ap.add_argument('--big_near_frac', type=float, default=1.5)
ap.add_argument('--cull_min_views', type=int, default=0, help='with cull_near: only drop near Gaussians seen by fewer than this many training views')
a = ap.parse_args()
case = Path(a.case)
cams = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
if a.views: cams = {k: v for k, v in cams.items() if k in a.views.split(',')}
offsets = json.loads((case / 't_offsets.json').read_text())
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
extra = [Gaussians.load(Path(r) if Path(r).suffix == '.pt' else Path(r) / a.filename).cuda().eval() for r in a.extra_runs.split(',')] if a.extra_runs else []
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
res = {}
tc_ = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
centers_ = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tc_.values()])
cam_radius = float(np.linalg.norm(centers_ - centers_.mean(0), axis=1).mean())
if a.cull_near_frac > 0: a.cull_near = a.cull_near_frac * float(cam_radius)
print(f'camera radius {cam_radius:.2f} m; cull radius {a.cull_near:.2f} m')
with torch.inference_mode():
    for name, cam in cams.items():
        frames = sorted((case / 'images' / name).glob('*.jpg'))
        K = cam.K.copy(); W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2]))
        Ws, Hs = int(round(W * a.scale)), int(round(H * a.scale)); K[0, :] *= Ws / W; K[1, :] *= Hs / H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda()[None]; Kt = torch.tensor(K, dtype=torch.float32).cuda()[None]
        off = float(offsets.get(name, 0.0)); acc = []
        gs_v = gs
        if a.cull_global_views > 0:
            if 'train_count' not in globals():
                tc = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); cnt = torch.zeros(len(gs), device='cuda', dtype=torch.int32)
                for tn, tcam in tc.items():
                    Kt_ = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt_[0, 2]; Ht = 2 * Kt_[1, 2]
                    w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda(); c3 = w2[:3, :3] @ gs.means.T + w2[:3, 3:4]
                    z = c3[2]; u = Kt_[0, 0] * c3[0] / z + Kt_[0, 2]; v = Kt_[1, 1] * c3[1] / z + Kt_[1, 2]
                    cnt += ((z > 0.05) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)).int()
                globals()['train_count'] = cnt
            drop = globals()['train_count'] < a.cull_global_views
            if a.cull_max_opacity > 0:
                op = gs.opacities.squeeze(-1) if gs.opacities.dim() > 1 else gs.opacities
                drop = drop & (torch.sigmoid(op) < a.cull_max_opacity)
            gs_v = gs.mask(~drop)
            gs = gs_v
            globals()['train_count'] = globals()['train_count'][~drop]
        if a.cull_near > 0:
            cc = torch.tensor(-cam.w2c[:3, :3].T @ cam.w2c[:3, 3], dtype=torch.float32).cuda()
            near = torch.norm(gs.means - cc, dim=1) < a.cull_near
            if a.cull_min_views > 0:
                if 'train_count' not in globals():
                    tc = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); cnt = torch.zeros(len(gs), device='cuda', dtype=torch.int32)
                    for tn, tcam in tc.items():
                        Kt_ = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt_[0, 2]; Ht = 2 * Kt_[1, 2]
                        w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda(); c3 = w2[:3, :3] @ gs.means.T + w2[:3, 3:4]
                        z = c3[2]; u = Kt_[0, 0] * c3[0] / z + Kt_[0, 2]; v = Kt_[1, 1] * c3[1] / z + Kt_[1, 2]
                        cnt += ((z > 0.05) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)).int()
                    globals()['train_count'] = cnt
                near = near & (globals()['train_count'] < a.cull_min_views)
            if a.big_scale > 0:
                big = (gs.scales.exp().max(dim=1).values > a.big_scale) & (torch.norm(gs.means - cc, dim=1) < a.big_near_frac * float(cam_radius))
                near = near | big
            gs_v = gs.mask(~near)
        for i in range(0, len(frames), a.every):
            gt = cv2.imread(str(frames[i]))[..., ::-1]
            assert gt.shape[1] == W and gt.shape[0] == H, (gt.shape, W, H)
            if a.scale != 1.0: gt = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_AREA)
            gt = torch.from_numpy(np.ascontiguousarray(gt)).float().cuda().permute(2, 0, 1)[None] / 255
            ss = max(1, a.supersample)
            Kss = Kt.clone(); Kss[:, 0, :] *= ss; Kss[:, 1, :] *= ss
            models_all = [gs_v] + list(extra)
            if a.weights:
                wts = [float(x) for x in a.weights.split(',')]
                assert len(wts) == len(models_all), f'{len(wts)} weights for {len(models_all)} models'
            else:
                wts = [1.0] * len(models_all)
            wsum = sum(wts); pred = None; stack = []
            tlist = [0.0]
            if a.temporal_avg:
                k, dt = a.temporal_avg.split(','); k = int(k); dt = float(dt)
                tlist = [j * dt for j in range(-k, k + 1)]
            for g2, wt in zip(models_all, wts):
                tacc = None
                for tj in tlist:
                    im2, _, _ = g2(t=torch.tensor(i / a.fps - off + tj).cuda(), w2c=w2c, intrinsic=Kss, shape=(Hs * ss, Ws * ss))
                    c = im2.permute(0, 3, 1, 2).clamp(0, 1)
                    tacc = c if tacc is None else tacc + c
                c = tacc / len(tlist)
                if a.ens_median:
                    stack.append(c)
                else:
                    c = c * (wt / wsum)
                    pred = c if pred is None else pred + c
            if a.ens_median:
                pred = torch.stack(stack, 0).median(0).values
            if ss > 1:
                pred = torch.nn.functional.avg_pool2d(pred, ss)
            if a.unsharp:
                _am, _r = a.unsharp.split(','); _am = float(_am); _r = int(_r); _k = _r * 2 + 1
                _b = F.avg_pool2d(F.pad(pred, (_k // 2,) * 4, mode='reflect'), _k, stride=1)
                pred = (pred + _am * (pred - _b)).clamp(0, 1)
            acc.append((psnr_f(pred, gt).item(), ssim_f(pred, gt).item(), lpips_f(pred, gt).item()))
            if a.save_dir:
                d = Path(a.save_dir) / name; d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f'{i:06d}.jpg'), (pred[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = np.mean(acc, 0); res[name] = m.tolist()
        print(f'view {name} {Ws}x{Hs}: psnr={m[0]:.3f} ssim={m[1]:.4f} lpips={m[2]:.4f}  ({len(acc)} frames)')
avg = np.mean(list(res.values()), 0)
print(f'MEAN psnr={avg[0]:.3f} ssim={avg[1]:.4f} lpips={avg[2]:.4f}')
json.dump({'per_view': res, 'mean': avg.tolist(), 'scale': a.scale}, open(Path(a.run) / f'eval_full_s{a.scale}{"_ens" if extra else ""}.json', 'w'), indent=1)
