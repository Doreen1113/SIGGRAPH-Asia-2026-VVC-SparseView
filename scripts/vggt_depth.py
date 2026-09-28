"""Dense metric depth for every training frame via VGGT (6 views jointly), scale-aligned to GT camera centers.
Usage: python vggt_depth.py <case_dir> <out_dir> [--start 0 --stop N --every 1 --check_points_dir DIR]
Writes <out_dir>/f{frame:06d}.npz with depth (V,h,w) fp16 [meters], conf (V,h,w) fp16, views list, scale, size of VGGT input."""
import argparse, json, sys, re, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/intern_2603055/projects/vggt')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('out')
ap.add_argument('--start', type=int, default=0); ap.add_argument('--stop', type=int, default=None); ap.add_argument('--every', type=int, default=1)
ap.add_argument('--check_points_dir', default=None)
a = ap.parse_args()
case = Path(a.case); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def read_yml_cams(intri, extri):
    s_i = open(intri).read(); s_e = open(extri).read()
    names = re.findall(r'^  - "(\w+)"', s_i, re.M)
    cams = {}
    for n in names:
        K = np.array(re.search(r'K_%s:.*?data: \[(.*?)\]' % n, s_i, re.S).group(1).split(','), float).reshape(3, 3)
        R = np.array(re.search(r'Rot_%s:.*?data: \[(.*?)\]' % n, s_e, re.S).group(1).split(','), float).reshape(3, 3)
        T = np.array(re.search(r'T_%s:.*?data: \[(.*?)\]' % n, s_e, re.S).group(1).split(','), float)
        cams[n] = (K, R, T)
    return cams
cams = read_yml_cams(case / 'train_intri.yml', case / 'train_extri.yml')
views = list(cams.keys()); gt_centers = np.stack([-R.T @ T for (_, R, T) in cams.values()])
frames = sorted(int(p.stem) for p in (case / 'images' / views[0]).glob('*.jpg'))
stop = a.stop if a.stop is not None else len(frames)
device = torch.device('cuda'); dtype = torch.bfloat16
model = VGGT.from_pretrained('facebook/VGGT-1B').to(device).eval()
def umeyama_scale(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    return np.sqrt((d0 ** 2).sum() / (s0 ** 2).sum())
for fi in range(a.start, stop, a.every):
    fr = frames[fi]; op = out / f'f{fr:06d}.npz'
    if op.exists(): continue
    paths = [str(case / 'images' / v / f'{fr:06d}.jpg') for v in views]
    images = load_and_preprocess_images(paths).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        tok, ps = model.aggregator(images[None])
        pose_enc = model.camera_head(tok)[-1]
        ext, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, conf = model.depth_head(tok, images[None], ps)
    ext = ext[0].float().cpu().numpy(); depth = depth[0, ..., 0].float().cpu().numpy(); conf = conf[0].float().cpu().numpy()
    pred_centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in ext])
    s = umeyama_scale(pred_centers, gt_centers)
    np.savez_compressed(op, depth=(depth * s).astype(np.float16), conf=conf.astype(np.float16), views=np.array(views), scale=s, vggt_hw=np.array(images.shape[-2:]))
    msg = f'frame {fr}: scale {s:.3f} depth med {np.median(depth*s):.2f} conf med {np.median(conf):.2f}'
    if a.check_points_dir:
        pp = Path(a.check_points_dir) / f'f{fr:06d}.ply'
        if pp.exists():
            from plyfile import PlyData
            v = PlyData.read(str(pp))['vertex']; P = np.stack([v['x'], v['y'], v['z']], 1)
            errs = []
            h, w = images.shape[-2:]
            for vi, name in enumerate(views):
                K, R, T = cams[name]; c = (R @ P.T + T[:, None]); z = c[2]; ok = z > 0.1
                Wf = 2 * K[0, 2]; Hf = 2 * K[1, 2]; sx = w / Wf; sy = h / Hf
                u = (K[0, 0] * c[0] / z + K[0, 2]) * sx; vv = (K[1, 1] * c[1] / z + K[1, 2]) * sy
                ok &= (u >= 0) & (u < w - 1) & (vv >= 0) & (vv < h - 1)
                ui = u[ok].astype(int); vi_ = vv[ok].astype(int); d = depth[vi, vi_, ui] * s
                rel = np.abs(d - z[ok]) / z[ok]; errs.append(np.median(rel))
            msg += f' | median rel err vs RoMa pts per view: {np.round(errs,3).tolist()}'
    print(msg, flush=True)
print('DONE')
