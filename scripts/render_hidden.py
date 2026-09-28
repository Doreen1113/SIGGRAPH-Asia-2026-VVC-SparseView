"""Render hidden test views for VVC sparse track.
Usage: .venv/bin/python render_hidden.py <run_dir> <case_dir> <out_dir> [--every 10] [--filename gaussians.pt] [--fps 60]
Reads <case_dir>/test_intri.yml + test_extri.yml (held-out cams), infers W=round(2cx), H=round(2cy),
t = frame/fps - t_offset[view]; writes <out_dir>/<view>/<frame:06d>.jpg
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('case'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=10); ap.add_argument('--filename', default='gaussians.pt')
ap.add_argument('--fps', type=float, default=60.0); ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--intri', default='test_intri.yml'); ap.add_argument('--extri', default='test_extri.yml')
ap.add_argument('--frames', default=None, help='comma list of frame indices to render (default: every N over the case)')
ap.add_argument('--jpg_quality', type=int, default=95)
ap.add_argument('--cull_near', type=float, default=0.0)
ap.add_argument('--cull_near_frac', type=float, default=0.0, help='cull radius as fraction of mean train-camera distance to camera centroid')
ap.add_argument('--cull_min_views', type=int, default=4)
ap.add_argument('--vggt_depth_dir', default=None, help='if set, prune floaters in front of VGGT depth (k=2, margin 0.15) before rendering')
ap.add_argument('--extra_runs', default=None, help='comma list of extra run dirs to ensemble (average renders)')
a = ap.parse_args()
case = Path(a.case)
cams = read_camera(case / a.intri, case / a.extri)
offsets = json.loads((case / 't_offsets.json').read_text()) if (case / 't_offsets.json').exists() else {}
import glob as _glob
for hp in _glob.glob(f'/home/intern_2603055/vvc/heldout/*/*/sparse/test/{case.name}/t_offsets.json'):
    ho = json.loads(open(hp).read()); offsets.update({k: v for k, v in ho.items() if k not in offsets}); print(f'merged {len(ho)} held-out time offsets from {hp}')
missing_off = [n for n in cams if n not in offsets]
if missing_off: print('WARNING: no time offset for hidden views', missing_off)
# number of frames from any train view folder
views_with_imgs = sorted(p for p in (case / 'images').iterdir() if p.is_dir())
frame_files = sorted(views_with_imgs[0].glob('*.jpg')) or sorted(views_with_imgs[0].glob('*.png'))
frame_ids = [int(f.stem) for f in frame_files]
first = frame_ids[0]; n = len(frame_ids)
sel = [int(x) for x in a.frames.split(',')] if a.frames else list(range(0, n, a.every))
gs = Gaussians.load(Path(a.run) / a.filename).cuda().eval()
extra = [Gaussians.load(Path(r) / a.filename).cuda().eval() for r in a.extra_runs.split(',')] if a.extra_runs else []
print(f'{len(gs)} gaussians; {len(cams)} hidden views; {len(sel)} frames each; frames start at {first}')
out = Path(a.out)
def train_view_count(g):
    tc = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); cnt = torch.zeros(len(g), device='cuda', dtype=torch.int32)
    for tn, tcam in tc.items():
        Kt_ = torch.tensor(tcam.K, dtype=torch.float32).cuda(); Wt = 2 * Kt_[0, 2]; Ht = 2 * Kt_[1, 2]
        w2 = torch.tensor(tcam.w2c, dtype=torch.float32).cuda(); c3 = w2[:3, :3] @ g.means.T + w2[:3, 3:4]
        z = c3[2]; u = Kt_[0, 0] * c3[0] / z + Kt_[0, 2]; v = Kt_[1, 1] * c3[1] / z + Kt_[1, 2]
        cnt += ((z > 0.05) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)).int()
    return cnt
tc_ = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
centers_ = np.stack([-c.w2c[:3, :3].T @ c.w2c[:3, 3] for c in tc_.values()])
cam_radius = float(np.linalg.norm(centers_ - centers_.mean(0), axis=1).mean())
cull_r = a.cull_near if a.cull_near > 0 else (a.cull_near_frac * cam_radius if a.cull_near_frac > 0 else 0.0)
models = [gs] + extra
if a.vggt_depth_dir:
    import subprocess
    pruned = []
    for i, (g, rdir) in enumerate(zip(models, [a.run] + (a.extra_runs.split(',') if a.extra_runs else []))):
        outp = Path(rdir) / f'pv_render_{a.filename}'
        if not outp.exists():
            r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'prune_vggt.py'), str(rdir), str(case), a.vggt_depth_dir, outp.name, '--filename', a.filename, '--k', '2', '--margin', '0.15'], capture_output=True, text=True)
            if r.returncode != 0:
                print('prune_vggt failed, using unpruned model:', r.stderr[-800:])
                outp = Path(rdir) / a.filename
        pruned.append(Gaussians.load(outp).cuda().eval())
    models = pruned; gs = models[0]; extra = models[1:]
    print('applied VGGT-depth floater pruning to', len(models), 'models')
counts = [train_view_count(g) for g in models] if cull_r > 0 else None
print(f'camera radius {cam_radius:.2f} m; cull radius {cull_r:.2f} m; min_views {a.cull_min_views}')
with torch.inference_mode():
    for name, cam in cams.items():
        K = cam.K.copy(); W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2]))
        D = np.asarray(getattr(cam, 'dist', getattr(cam, 'D', np.zeros(5))), dtype=np.float64).reshape(-1)
        if hasattr(cam, 'W') and hasattr(cam, 'H') and (cam.W or 0) > 0 and (cam.H or 0) > 0: W, H = int(cam.W), int(cam.H)  # yml stores -1 when unknown
        distort_map = None
        if np.abs(D).sum() > 0:  # GT is the raw distorted frame: render a pinhole view that covers it, then re-distort
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha=1.0, newImgSize=(W, H))
            up = 1.25; Wu, Hu = int(round(W * up)), int(round(H * up)); newK = newK.copy(); newK[0, :] *= Wu / W; newK[1, :] *= Hu / H
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
            und = cv2.undistortPoints(pts, K, D, P=newK).reshape(H, W, 2).astype(np.float32)
            distort_map = (und[..., 0], und[..., 1]); K = newK; W, H = Wu, Hu
            print(f'view {name}: distortion {D[:2].round(4)} -> render {Wu}x{Hu} pinhole and remap to {uu.shape[1]}x{uu.shape[0]}')
        Ws, Hs = int(round(W * a.scale)), int(round(H * a.scale))
        K[0, :] *= Ws / W; K[1, :] *= Hs / H
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda()[None]
        Kt = torch.tensor(K, dtype=torch.float32).cuda()[None]
        off = float(offsets.get(name, 0.0))
        (out / name).mkdir(parents=True, exist_ok=True)
        mv = models
        if cull_r > 0:
            cc = torch.tensor(-cam.w2c[:3, :3].T @ cam.w2c[:3, 3], dtype=torch.float32).cuda()
            mv = [g.mask(~((torch.norm(g.means - cc, dim=1) < cull_r) & (cnt < a.cull_min_views))) for g, cnt in zip(models, counts)]
        for i in sel:
            if (out / name / f'{first + i:06d}.jpg').exists(): continue  # resumable
            t = i / a.fps - off  # logical frame index (mvdataset: time = logical_frame / fps - offset)
            img = None
            for g2 in mv:
                im2, _, _ = g2(t=torch.tensor(t).cuda(), w2c=w2c, intrinsic=Kt, shape=(Hs, Ws))
                img = im2.clamp(0, 1) if img is None else img + im2.clamp(0, 1)
            img = img / len(mv)
            im = (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            if distort_map is not None:
                im = cv2.remap(im, distort_map[0] * a.scale, distort_map[1] * a.scale, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(out / name / f'{first + i:06d}.jpg'), im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, a.jpg_quality])
        print('view', name, f'{Ws}x{Hs}', 'done')
