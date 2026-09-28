"""Image-based rendering refinement (COMPLIANT: uses only the case's 6 training views + model depth).
For each hidden-view pixel: unproject with the Gaussian-rendered depth, project into each training view,
check occlusion against that train view's model-rendered depth, sample the REAL training image,
blend top-k views by angular weight, and composite over the Gaussian render where nothing is visible.
Usage: ftg.sh ibr_eval.py <run_dir> <case_dir> [--scale 0.5 --every 20 --tol 0.05 --k 3 --blend 0.8]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('case')
ap.add_argument('--filename', default='gaussians.pt')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--every', type=int, default=20); ap.add_argument('--fps', type=float, default=60.0)
ap.add_argument('--tol', type=float, default=0.05, help='relative depth tolerance for occlusion test')
ap.add_argument('--k', type=int, default=3, help='blend the k best training views per pixel')
ap.add_argument('--blend', type=float, default=1.0, help='weight of IBR colour vs model render where IBR is valid')
ap.add_argument('--cull_near_frac', type=float, default=1.1); ap.add_argument('--cull_min_views', type=int, default=4)
ap.add_argument('--views', default=None); ap.add_argument('--save_dir', default=None); ap.add_argument('--out_json', default=None)
a = ap.parse_args(); case = Path(a.case)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); te = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
if a.views: te = {k: v for k, v in te.items() if k in a.views.split(',')}
offsets = json.loads((case / 't_offsets.json').read_text())
ho = case / 'heldout_t_offsets.json'
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
centers = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tr.values()]); cam_radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())
def train_count(g):
    cnt = torch.zeros(len(g), device='cuda', dtype=torch.int32)
    for tn, tcam in tr.items():
        K = torch.tensor(tcam.K, dtype=torch.float32).cuda(); W = 2 * K[0, 2]; H = 2 * K[1, 2]; w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
        c3 = w2[:3, :3] @ g.means.T + w2[:3, 3:4]; z = c3[2]; u = K[0, 0] * c3[0] / z + K[0, 2]; v = K[1, 1] * c3[1] / z + K[1, 2]
        cnt += ((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).int()
    return cnt
tcount = train_count(gs)
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
def metr(p, gt):
    pt = p[None].permute(0, 3, 1, 2).clamp(0, 1); g = gt[None].permute(0, 3, 1, 2)
    return (psnr_f(pt, g).item(), ssim_f(pt, g).item(), lpips_f(pt, g).item())
# pre-load train camera tensors
trt = {}
for tn, tcam in tr.items():
    K = torch.tensor(tcam.K, dtype=torch.float32).cuda(); w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda()
    ctr = torch.tensor(-tcam.w2c[:3, :3].T @ tcam.w2c[:3, 3], dtype=torch.float32).cuda()
    trt[tn] = (K, w2, ctr, int(round(2 * float(K[0, 2]))), int(round(2 * float(K[1, 2]))))
res_raw, res_ibr = {}, {}
with torch.inference_mode():
    for name, cam in te.items():
        frames = sorted((case / 'images' / name).glob('*.jpg'))
        K = cam.K.copy(); W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2])); Ws, Hs = int(round(W * a.scale)), int(round(H * a.scale)); K[0, :] *= Ws / W; K[1, :] *= Hs / H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda(); Kt = torch.tensor(K, dtype=torch.float32).cuda(); c2w = torch.inverse(w2c)
        ccen = c2w[:3, 3]
        cc = torch.tensor(-cam.w2c[:3, :3].T @ cam.w2c[:3, 3], dtype=torch.float32).cuda()
        near = (torch.norm(gs.means - cc, dim=1) < a.cull_near_frac * cam_radius) & (tcount < a.cull_min_views); gv = gs.mask(~near)
        off = float(offsets.get(name, 0.0)); acc_raw, acc_ibr = [], []
        uu, vv = torch.meshgrid(torch.arange(Ws, device='cuda').float(), torch.arange(Hs, device='cuda').float(), indexing='xy')
        rays = torch.stack([(uu - Kt[0, 2]) / Kt[0, 0], (vv - Kt[1, 2]) / Kt[1, 1], torch.ones_like(uu)], -1)
        for i in range(0, len(frames), a.every):
            gt = cv2.imread(str(frames[i]))[..., ::-1]; gt = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_AREA)
            gt = torch.from_numpy(np.ascontiguousarray(gt)).cuda().float() / 255
            t = torch.tensor(i / a.fps - off).cuda()
            img, _, meta = gv(t=t, w2c=w2c[None], intrinsic=Kt[None], shape=(Hs, Ws), render_depth=True)
            R = img[0].clamp(0, 1); D = meta['depth'][0, ..., 0]
            Pw = (rays * D[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
            cols = []; wts = []
            for tn, tcam in tr.items():
                Ktr, w2, tctr, Wt, Ht = trt[tn]
                toff = float(offsets.get(tn, 0.0)); tt = torch.tensor(i / a.fps - toff).cuda()
                # model depth as seen from this training camera (occlusion reference)
                sc = a.scale; Kts = Ktr.clone(); Wts, Hts = int(round(Wt * sc)), int(round(Ht * sc)); Kts[0, :] *= Wts / Wt; Kts[1, :] *= Hts / Ht
                _, _, m2 = gv(t=tt, w2c=w2[None], intrinsic=Kts[None], shape=(Hts, Wts), render_depth=True)
                Dtr = m2['depth'][0, ..., 0]
                real = cv2.imread(str(case / 'images' / tn / f'{i:06d}.jpg'))[..., ::-1]
                real = cv2.resize(real, (Wts, Hts), interpolation=cv2.INTER_AREA)
                real = torch.from_numpy(np.ascontiguousarray(real)).cuda().float() / 255
                c3 = Pw @ w2[:3, :3].T + w2[:3, 3]; z = c3[..., 2]
                px = Kts[0, 0] * c3[..., 0] / z + Kts[0, 2]; py = Kts[1, 1] * c3[..., 1] / z + Kts[1, 2]
                inb = (z > 0.05) & (px >= 0) & (px < Wts - 1) & (py >= 0) & (py < Hts - 1)
                gx = (px / (Wts - 1) * 2 - 1).clamp(-1, 1); gy = (py / (Hts - 1) * 2 - 1).clamp(-1, 1)
                grid = torch.stack([gx, gy], -1)[None]
                samp = torch.nn.functional.grid_sample(real.permute(2, 0, 1)[None], grid, mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                zs = torch.nn.functional.grid_sample(Dtr[None, None], grid, mode='nearest', align_corners=True)[0, 0]
                vis = inb & (torch.abs(z - zs) < a.tol * zs.clamp(min=0.1))
                # angular weight: how similar is the viewing direction
                d1 = torch.nn.functional.normalize(Pw - ccen, dim=-1); d2 = torch.nn.functional.normalize(Pw - tctr, dim=-1)
                cosang = (d1 * d2).sum(-1).clamp(-1, 1)
                w = torch.where(vis, (cosang.clamp(min=0)) ** 4 + 1e-6, torch.zeros_like(cosang))
                cols.append(samp); wts.append(w)
            Wt_ = torch.stack(wts)  # V,H,W
            Cs = torch.stack(cols)  # V,H,W,3
            if a.k < len(wts):
                topw, topi = Wt_.topk(a.k, dim=0)
                Cs = torch.gather(Cs, 0, topi[..., None].expand(-1, -1, -1, 3)); Wt_ = topw
            s = Wt_.sum(0)
            ibr = (Cs * Wt_[..., None]).sum(0) / s.clamp(min=1e-6)[..., None]
            valid = (s > 1e-5)[..., None].float()
            out = valid * (a.blend * ibr + (1 - a.blend) * R) + (1 - valid) * R
            acc_raw.append(metr(R, gt)); acc_ibr.append(metr(out, gt))
            if a.save_dir and i == 0:
                d = Path(a.save_dir); d.mkdir(parents=True, exist_ok=True)
                vis3 = valid.repeat(1, 1, 3)
                cv2.imwrite(str(d / f'{name}_cmp.jpg'), (torch.cat([gt, R, out, vis3], 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
        res_raw[name] = np.mean(acc_raw, 0); res_ibr[name] = np.mean(acc_ibr, 0)
        print(f'view {name}: raw {res_raw[name][0]:.2f} -> ibr {res_ibr[name][0]:.2f} | ssim {res_raw[name][1]:.4f}->{res_ibr[name][1]:.4f} | lpips {res_raw[name][2]:.4f}->{res_ibr[name][2]:.4f} | valid {float(valid.mean()):.2f}', flush=True)
mr = np.mean(list(res_raw.values()), 0); mi = np.mean(list(res_ibr.values()), 0)
print(f'MEAN raw psnr={mr[0]:.3f} ssim={mr[1]:.4f} lpips={mr[2]:.4f} | IBR psnr={mi[0]:.3f} ssim={mi[1]:.4f} lpips={mi[2]:.4f}')
if a.out_json: json.dump({'raw': {k: v.tolist() for k, v in res_raw.items()}, 'ibr': {k: v.tolist() for k, v in res_ibr.items()}, 'mean_raw': mr.tolist(), 'mean_ibr': mi.tolist()}, open(a.out_json, 'w'), indent=1)
