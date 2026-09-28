"""Local experiment: composite model render (dynamic person) over a real static background per view.
Usage: ftg.sh composite_eval.py <run_dir> <case_dir> <bg_dir> [--scale 0.5 --every 20 --dyn_thr 5.0 --views ...]
bg_dir/<view>.png = background image at full res. Person alpha = alpha of Gaussians with temporal_scale < dyn_thr (dynamic set),
thresholded/dilated/feathered. Also color-matches the background to the render on non-person pixels (per-channel affine)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('bg')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--every', type=int, default=20); ap.add_argument('--fps', type=float, default=60.0)
ap.add_argument('--dyn_thr', type=float, default=5.0); ap.add_argument('--alpha_thr', type=float, default=0.3); ap.add_argument('--dilate', type=int, default=15); ap.add_argument('--feather', type=int, default=21)
ap.add_argument('--views', default=None); ap.add_argument('--mask_dir', default=None, help='use precomputed person masks <mask_dir>/<view>/<frame>.png (255=person) instead of dynamic alpha'); ap.add_argument('--no_colormatch', action='store_true'); ap.add_argument('--save_dir', default=None)
a = ap.parse_args(); case = Path(a.case)
cams = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
if a.views: cams = {k: v for k, v in cams.items() if k in a.views.split(',')}
offsets = json.loads((case / 't_offsets.json').read_text())
gs = Gaussians.load(Path(a.run) / 'gaussians.pt').cuda().eval()
dyn = gs.mask((gs.temporal_scale().squeeze(-1) < a.dyn_thr))
print(f'dynamic subset: {len(dyn)}/{len(gs)}')
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
res = {}; res_raw = {}
with torch.inference_mode():
    for name, cam in cams.items():
        frames = sorted((case / 'images' / name).glob('*.jpg'))
        K = cam.K.copy(); W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2])); Ws, Hs = int(round(W * a.scale)), int(round(H * a.scale)); K[0, :] *= Ws / W; K[1, :] *= Hs / H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda()[None]; Kt = torch.tensor(K, dtype=torch.float32).cuda()[None]; off = float(offsets.get(name, 0.0))
        bgp = Path(a.bg) / f'{name}.png'
        if not bgp.exists(): print('no bg for', name); continue
        bg = cv2.imread(str(bgp))[..., ::-1]; bg = cv2.resize(bg, (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        acc = []; acc_raw = []
        for i in range(0, len(frames), a.every):
            gt = cv2.imread(str(frames[i]))[..., ::-1]; gt = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            t = torch.tensor(i / a.fps - off).cuda()
            img, _, _ = gs(t=t, w2c=w2c, intrinsic=Kt, shape=(Hs, Ws)); R = img[0].clamp(0, 1).cpu().numpy()
            if a.mask_dir:
                mp = Path(a.mask_dir) / name / f'{i:06d}.png'
                if not mp.exists(): continue
                m = (cv2.resize(cv2.imread(str(mp), 0), (Ws, Hs), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
            else:
                _, alpha, _ = dyn(t=t, w2c=w2c, intrinsic=Kt, shape=(Hs, Ws)); al = alpha[0, ..., 0].cpu().numpy()
                m = (al > a.alpha_thr).astype(np.uint8)
            if a.dilate > 0: m = cv2.dilate(m, np.ones((a.dilate, a.dilate), np.uint8))
            mf = cv2.GaussianBlur(m.astype(np.float32), (a.feather | 1, a.feather | 1), 0)[..., None]
            B = bg
            if not a.no_colormatch:
                sel = (mf[..., 0] < 0.05)
                if sel.sum() > 1000:
                    B = bg.copy()
                    for c in range(3):
                        x = bg[..., c][sel]; y = R[..., c][sel]; A_ = np.vstack([x, np.ones_like(x)]).T; k, b = np.linalg.lstsq(A_, y, rcond=None)[0]
                        B[..., c] = np.clip(bg[..., c] * k + b, 0, 1)
            out = mf * R + (1 - mf) * B
            def metr(p):
                pt = torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1)[None].cuda(); g = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].cuda()
                return (psnr_f(pt, g).item(), ssim_f(pt, g).item(), lpips_f(pt.clamp(0, 1), g).item())
            acc.append(metr(out)); acc_raw.append(metr(R))
            if a.save_dir and i == 0:
                d = Path(a.save_dir); d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f'{name}_cmp.jpg'), (np.hstack([gt, R, out, np.repeat(mf, 3, 2)]) * 255).astype(np.uint8)[..., ::-1])
        if not acc: continue
        res[name] = np.mean(acc, 0); res_raw[name] = np.mean(acc_raw, 0)
        print(f'view {name}: raw psnr={res_raw[name][0]:.2f} -> composite psnr={res[name][0]:.2f} ssim={res[name][1]:.4f} lpips={res[name][2]:.4f}')
m = np.mean(list(res.values()), 0); mr = np.mean(list(res_raw.values()), 0)
print(f'MEAN raw psnr={mr[0]:.3f} ssim={mr[1]:.4f} lpips={mr[2]:.4f} | composite psnr={m[0]:.3f} ssim={m[1]:.4f} lpips={m[2]:.4f}')
