"""Build keyframe point clouds (same format as RoMa points_stride10) by back-projecting VGGT dense depth from all train views.
Usage: python vggt_points.py <case_dir> <depth_dir> <out_dir> [--stride 10 --per_frame 108000 --conf 2.0]"""
import argparse, re, sys
from pathlib import Path
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('depth'); ap.add_argument('out')
ap.add_argument('--stride', type=int, default=10); ap.add_argument('--per_frame', type=int, default=108000); ap.add_argument('--conf', type=float, default=2.0)
a = ap.parse_args(); case = Path(a.case); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
s_i = open(case / 'train_intri.yml').read(); s_e = open(case / 'train_extri.yml').read()
names = re.findall(r'^  - "(\w+)"', s_i, re.M); cams = {}
for n in names:
    K = np.array(re.search(r'K_%s:.*?data: \[(.*?)\]' % n, s_i, re.S).group(1).split(','), float).reshape(3, 3)
    R = np.array(re.search(r'Rot_%s:.*?data: \[(.*?)\]' % n, s_e, re.S).group(1).split(','), float).reshape(3, 3)
    T = np.array(re.search(r'T_%s:.*?data: \[(.*?)\]' % n, s_e, re.S).group(1).split(','), float); cams[n] = (K, R, T)
frames = sorted(int(p.stem) for p in (case / 'images' / names[0]).glob('*.jpg'))
keys = list(range(0, len(frames), a.stride))
if keys[-1] != len(frames) - 1: keys.append(len(frames) - 1)
def write_ply(path, P, C):
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        arr = np.empty(len(P), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
        arr['x'], arr['y'], arr['z'] = P[:, 0], P[:, 1], P[:, 2]; arr['r'], arr['g'], arr['b'] = C[:, 0], C[:, 1], C[:, 2]; f.write(arr.tobytes())
rng = np.random.RandomState(0)
for ki in keys:
    fr = frames[ki]; z = np.load(Path(a.depth) / f'f{fr:06d}.npz'); views = [str(v) for v in z['views']]
    h, w = z['depth'].shape[1:]; per_view = a.per_frame // len(views); P_all, C_all = [], []
    for vi, n in enumerate(views):
        K, R, T = cams[n]; d = z['depth'][vi].astype(np.float32); c = z['conf'][vi].astype(np.float32)
        img = cv2.imread(str(case / 'images' / n / f'{fr:06d}.jpg'))[..., ::-1]; Wf, Hf = img.shape[1], img.shape[0]
        img_s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        ok = (c > a.conf) & (d > 0.1); ys, xs = np.nonzero(ok)
        if len(ys) == 0: continue
        sel = rng.choice(len(ys), min(per_view, len(ys)), replace=False); ys, xs = ys[sel], xs[sel]
        # pixel in vggt res -> full-res pixel -> ray
        u = (xs + 0.5) * Wf / w; v = (ys + 0.5) * Hf / h; zz = d[ys, xs]
        x = (u - K[0, 2]) / K[0, 0] * zz; y = (v - K[1, 2]) / K[1, 1] * zz
        Pc = np.stack([x, y, zz], 1); Pw = (R.T @ (Pc.T - T[:, None])).T
        P_all.append(Pw); C_all.append(img_s[ys, xs])
    P = np.concatenate(P_all); C = np.concatenate(C_all)
    write_ply(out / f'f{fr:06d}.ply', P.astype(np.float32), C.astype(np.uint8)); print(f'f{fr:06d}: {len(P)} pts', flush=True)
print('DONE')
