"""Compare seva pseudo-views vs GT vs our 4DGS render at the seva crop resolution.
Usage: ftg.sh compare.py <scene_dir> <seva_out_dir> <run_dir> <case_dir> <frame> [--cull_near_frac 1.1 --cull_min_views 4]"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('seva_out'); ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('frame', type=int)
ap.add_argument('--fps', type=float, default=60.0); ap.add_argument('--cull_near_frac', type=float, default=1.1); ap.add_argument('--cull_min_views', type=int, default=4)
ap.add_argument('--save', default=None)
a = ap.parse_args()
scene = Path(a.scene); meta = json.load(open(scene / 'transforms.json')); split = json.load(open(next(scene.glob('train_test_split_*.json'))))
case = Path(a.case); offsets = json.loads((case / 't_offsets.json').read_text())
gs = Gaussians.load(Path(a.run) / 'gaussians.pt').cuda().eval()
tc = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
centers = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tc.values()]); cam_radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())
cull_r = a.cull_near_frac * cam_radius
cnt = torch.zeros(len(gs), device='cuda', dtype=torch.int32)
for tcam in tc.values():
    Kt = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt[0, 2]; Ht = 2 * Kt[1, 2]
    w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda(); c3 = w2[:3, :3] @ gs.means.T + w2[:3, 3:4]
    z = c3[2]; u = Kt[0, 0] * c3[0] / z + Kt[0, 2]; v = Kt[1, 1] * c3[1] / z + Kt[1, 2]
    cnt += ((z > 0.05) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)).int()
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
def T(im): return torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).float().cuda().permute(2, 0, 1)[None] / 255
def metrics(p, g): return psnr_f(p, g).item(), ssim_f(p, g).item(), lpips_f(p.clamp(0, 1), g).item()
seva_imgs = sorted(glob.glob(str(Path(a.seva_out) / '**' / '*.png'), recursive=True))
print('seva outputs:', len(seva_imgs), seva_imgs[:3])
rows = []
with torch.inference_mode():
    for j, i in enumerate(split['test_ids']):
        f = meta['frames'][i]; name = f['name']
        gt = cv2.imread(str(scene / f['file_path'])) if f['file_path'] else None
        if gt is None: continue
        c2w = np.array(f['transform_matrix']); c2w[:, 1:3] *= -1; w2c = np.linalg.inv(c2w)
        K = np.array([[f['fl_x'], 0, f['cx']], [0, f['fl_y'], f['cy']], [0, 0, 1]], dtype=np.float32)
        cc = torch.tensor(c2w[:3, 3], dtype=torch.float32).cuda()
        gv = gs.mask(~((torch.norm(gs.means - cc, dim=1) < cull_r) & (cnt < a.cull_min_views)))
        t = a.frame / a.fps - float(offsets.get(name, 0.0))
        img, _, _ = gv(t=torch.tensor(t).cuda(), w2c=torch.tensor(w2c, dtype=torch.float32).cuda()[None], intrinsic=torch.tensor(K).cuda()[None], shape=(f['h'], f['w']))
        ours = img.permute(0, 3, 1, 2).clamp(0, 1); gtt = T(gt)
        m_ours = metrics(ours, gtt)
        # seva output for target j: try to find file by index among test outputs
        cand = [p for p in seva_imgs if Path(p).stem.endswith(f'{j:03d}') or Path(p).stem == f'{i:03d}' or Path(p).stem == name]
        m_seva = None
        if cand:
            sv = cv2.imread(cand[0]); sv = cv2.resize(sv, (f['w'], f['h']), interpolation=cv2.INTER_AREA); m_seva = metrics(T(sv), gtt)
            if a.save:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{a.save}/{name}.jpg', np.concatenate([gt, sv, (ours[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1]], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append((name, m_ours, m_seva))
        print(f'view {name}: ours psnr={m_ours[0]:.2f} ssim={m_ours[1]:.3f} lpips={m_ours[2]:.3f} | seva ' + (f'psnr={m_seva[0]:.2f} ssim={m_seva[1]:.3f} lpips={m_seva[2]:.3f}' if m_seva else 'n/a'))
mo = np.mean([r[1] for r in rows], 0); print(f'MEAN ours psnr={mo[0]:.2f} ssim={mo[1]:.3f} lpips={mo[2]:.3f}')
if all(r[2] for r in rows): ms = np.mean([r[2] for r in rows], 0); print(f'MEAN seva psnr={ms[0]:.2f} ssim={ms[1]:.3f} lpips={ms[2]:.3f}')
