"""Visibility-aware inpainting post-process (compliant: uses only the 6 train views + VGGT depth of them).
For each hidden-view render: unproject rendered depth -> 3D; a pixel is VALID if at least one train camera sees that point
consistently (inside frustum and |z - vggt_depth| within tolerance). Invalid pixels (unobserved regions / floaters) are
replaced by inpainting from valid neighbours. Evaluates raw vs filled on val.
Usage: ftg.sh visfill_eval.py <run_dir> <case_dir> <vggt_depth_dir> [--scale 0.5 --every 20 --tol 0.15 --dilate 5 --method telea|blur]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('depth')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--every', type=int, default=20); ap.add_argument('--fps', type=float, default=60.0)
ap.add_argument('--tol', type=float, default=0.15); ap.add_argument('--dilate', type=int, default=5); ap.add_argument('--method', default='telea')
ap.add_argument('--conf', type=float, default=2.0); ap.add_argument('--cull_near_frac', type=float, default=1.1); ap.add_argument('--cull_min_views', type=int, default=4)
ap.add_argument('--views', default=None); ap.add_argument('--save_dir', default=None); ap.add_argument('--stride', type=int, default=10)
a = ap.parse_args(); case = Path(a.case)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); te = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
if a.views: te = {k: v for k, v in te.items() if k in a.views.split(',')}
offsets = json.loads((case / 't_offsets.json').read_text())
gs = Gaussians.load(Path(a.run) / 'gaussians.pt').cuda().eval()
centers = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tr.values()]); cam_radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())
# train-view visibility counts for culling
def train_count(g):
    cnt = torch.zeros(len(g), device='cuda', dtype=torch.int32)
    for tn, tcam in tr.items():
        K = torch.tensor(tcam.K, dtype=torch.float32).cuda(); W = 2 * K[0, 2]; H = 2 * K[1, 2]; w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
        c3 = w2[:3, :3] @ g.means.T + w2[:3, 3:4]; z = c3[2]; u = K[0, 0] * c3[0] / z + K[0, 2]; v = K[1, 1] * c3[1] / z + K[1, 2]
        cnt += ((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).int()
    return cnt
tcount = train_count(gs)
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
    pt = torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1)[None].cuda().clamp(0, 1); g = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].cuda()
    return (psnr_f(pt, g).item(), ssim_f(pt, g).item(), lpips_f(pt, g).item())
res_raw, res_fill = {}, {}
with torch.inference_mode():
    for name, cam in te.items():
        frames = sorted((case / 'images' / name).glob('*.jpg'))
        K = cam.K.copy(); W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2])); Ws, Hs = int(round(W * a.scale)), int(round(H * a.scale)); K[0, :] *= Ws / W; K[1, :] *= Hs / H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda(); Kt = torch.tensor(K, dtype=torch.float32).cuda()
        cc = torch.tensor(-cam.w2c[:3, :3].T @ cam.w2c[:3, 3], dtype=torch.float32).cuda()
        near = (torch.norm(gs.means - cc, dim=1) < a.cull_near_frac * cam_radius) & (tcount < a.cull_min_views); gv = gs.mask(~near)
        off = float(offsets.get(name, 0.0)); acc_raw, acc_fill = [], []
        # pixel rays
        u, v = torch.meshgrid(torch.arange(Ws, device='cuda').float(), torch.arange(Hs, device='cuda').float(), indexing='xy')
        rays = torch.stack([(u - Kt[0, 2]) / Kt[0, 0], (v - Kt[1, 2]) / Kt[1, 1], torch.ones_like(u)], -1)  # Hs,Ws,3
        c2w = torch.inverse(w2c)
        for i in range(0, len(frames), a.every):
            gt = cv2.imread(str(frames[i]))[..., ::-1]; gt = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            t = torch.tensor(i / a.fps - off).cuda()
            img, alpha, meta = gv(t=t, w2c=w2c[None], intrinsic=Kt[None], shape=(Hs, Ws), render_depth=True)
            R = img[0].clamp(0, 1); D = meta['depth'][0, ..., 0]
            P = rays * D[..., None]; Pw = P @ c2w[:3, :3].T + c2w[:3, 3]  # Hs,Ws,3
            valid = torch.zeros((Hs, Ws), dtype=torch.bool, device='cuda'); dd = vggt(i)
            for tn, tcam in tr.items():
                if tn not in dd: continue
                dz, dc = dd[tn]; h, w = dz.shape
                Kt_ = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt_[0, 2]; Ht = 2 * Kt_[1, 2]; w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
                c3 = Pw @ w2[:3, :3].T + w2[:3, 3]; z = c3[..., 2]
                uu = (Kt_[0, 0] * c3[..., 0] / z + Kt_[0, 2]) * (w / Wt); vv = (Kt_[1, 1] * c3[..., 1] / z + Kt_[1, 2]) * (h / Ht)
                ok = (z > 0.05) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                ui = uu.clamp(0, w - 1).long(); vi = vv.clamp(0, h - 1).long()
                zs = dz[vi, ui]; cf = dc[vi, ui]
                consistent = ok & (cf > a.conf) & (torch.abs(z - zs) < a.tol * zs)
                valid |= consistent
            inv = (~valid).cpu().numpy().astype(np.uint8)
            if a.dilate > 0: inv = cv2.dilate(inv, np.ones((a.dilate * 2 + 1,) * 2, np.uint8))
            Rn = R.cpu().numpy()
            if inv.mean() > 0.98: filled = Rn
            elif a.method == 'telea':
                filled = cv2.inpaint((Rn * 255).astype(np.uint8), inv * 255, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
            else:
                m = inv.astype(bool); filled = Rn.copy(); blur = cv2.blur(Rn * (~m)[..., None], (61, 61)); wsum = cv2.blur((~m).astype(np.float32), (61, 61))[..., None] + 1e-6; filled[m] = (blur / wsum)[m]
            acc_raw.append(metr(Rn, gt)); acc_fill.append(metr(filled, gt))
            if a.save_dir and i == 0:
                d = Path(a.save_dir); d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f'{name}_cmp.jpg'), (np.hstack([gt, Rn, filled, np.repeat(inv[..., None].astype(np.float32), 3, 2)]) * 255).astype(np.uint8)[..., ::-1])
        res_raw[name] = np.mean(acc_raw, 0); res_fill[name] = np.mean(acc_fill, 0)
        print(f'view {name}: raw {res_raw[name][0]:.2f} -> filled {res_fill[name][0]:.2f}  ssim {res_raw[name][1]:.4f}->{res_fill[name][1]:.4f} lpips {res_raw[name][2]:.4f}->{res_fill[name][2]:.4f}  invalid frac {inv.mean():.2f}', flush=True)
mr = np.mean(list(res_raw.values()), 0); mf = np.mean(list(res_fill.values()), 0)
print(f'MEAN raw psnr={mr[0]:.3f} ssim={mr[1]:.4f} lpips={mr[2]:.4f} | filled psnr={mf[0]:.3f} ssim={mf[1]:.4f} lpips={mf[2]:.4f}')
