"""Person masks for hidden views via visual hull from DeepLabV3 silhouettes in the 6 train views.
Usage: ftg.sh hull_mask.py <case_dir> <out_dir> [--every 10 --voxel 0.04 --dilate 8 --hidden_intri test_intri.yml --hidden_extri test_extri.yml]
Writes <out_dir>/<hidden_view>/<frame:06d>.png (255 = person) at full hidden-view resolution."""
import argparse, sys, json
from pathlib import Path
import numpy as np, cv2, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo' / 'baseline_code'))
from ftgspp.data.utils.easy_utils import read_camera
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('out'); ap.add_argument('--every', type=int, default=10)
ap.add_argument('--voxel', type=float, default=0.04); ap.add_argument('--dilate', type=int, default=8); ap.add_argument('--sil_dilate', type=int, default=6)
ap.add_argument('--seg_scale', type=float, default=0.25); ap.add_argument('--min_views', type=int, default=6)
ap.add_argument('--debug_dir', default=None); ap.add_argument('--save_hull', action='store_true', help='save hull voxel centers (m) per frame to <out>/hull_<frame>.npz'); ap.add_argument('--no_masks', action='store_true')
a = ap.parse_args(); case = Path(a.case); out = Path(a.out)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml'); te = read_camera(case / 'test_intri.yml', case / 'test_extri.yml')
model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def silhouette(img):
    x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1)[None].float() / 255; x = ((x - mean) / std).half()
    with torch.inference_mode(): p = model(x)['out'].float().softmax(1)[0, 15]
    m = (p > 0.3).cpu().numpy().astype(np.uint8)
    if a.sil_dilate > 0: m = cv2.dilate(m, np.ones((a.sil_dilate * 2 + 1,) * 2, np.uint8))
    return m
# capture volume is placed per frame around the triangulated person position (silhouette centroids)
ups = np.stack([-c.w2c[:3, :3][1] for c in tr.values()]); up = ups.mean(0); up /= np.linalg.norm(up)
e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
r = 1.6; n1 = int(2 * r / a.voxel); nh = int(3.0 / a.voxel)
g1, g2, gh = np.meshgrid(np.linspace(-r, r, n1), np.linspace(-r, r, n1), np.linspace(-1.5, 1.5, nh), indexing='ij')
Poff = g1.reshape(-1, 1) * e1[None] + g2.reshape(-1, 1) * e2[None] + gh.reshape(-1, 1) * up[None]
N = len(Poff)
def triangulate(rays):  # rays: list of (origin, dir) -> least squares point
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        d = d / np.linalg.norm(d); M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
    return np.linalg.solve(A, b)
frames = sorted(int(p.stem) for p in (case / 'images' / list(tr.keys())[0]).glob('*.jpg'))
def proj(cam, K, Pw):
    w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda(); c = Pw @ w2c[:3, :3].T + w2c[:3, 3]; z = c[:, 2]
    u = K[0, 0] * c[:, 0] / z + K[0, 2]; v = K[1, 1] * c[:, 1] / z + K[1, 2]; return u, v, z
for fi in range(0, len(frames), a.every):
    fr = frames[fi]; sils = {}; rays = []
    for name, cam in tr.items():
        img = cv2.imread(str(case / 'images' / name / f'{fr:06d}.jpg'))[..., ::-1]; H, W = img.shape[:2]
        small = cv2.resize(img, (int(W * a.seg_scale), int(H * a.seg_scale)), interpolation=cv2.INTER_AREA); sil = silhouette(small)
        if a.debug_dir and fi == 0:
            Path(a.debug_dir).mkdir(parents=True, exist_ok=True); cv2.imwrite(f'{a.debug_dir}/sil_{name}.jpg', np.hstack([small[..., ::-1], cv2.cvtColor(sil * 255, cv2.COLOR_GRAY2BGR)]))
        sils[name] = sil
        # largest component centroid -> ray
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(sil)
        if nlab > 1:
            k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]); cx, cy = cents[k] / a.seg_scale
            Kc = cam.K; d_cam = np.array([(cx - Kc[0, 2]) / Kc[0, 0], (cy - Kc[1, 2]) / Kc[1, 1], 1.0])
            R = cam.w2c[:3, :3]; o = -R.T @ cam.w2c[:3, 3]; rays.append((o, R.T @ d_cam))
    if len(rays) < 3: print(f'frame {fr}: person not found'); continue
    ctr = triangulate(rays); P = torch.from_numpy((ctr[None] + Poff).astype(np.float32)).cuda()
    inside = torch.zeros(N, dtype=torch.int32, device='cuda')
    for name, cam in tr.items():
        sil = sils[name]; K = cam.K.copy() * a.seg_scale; K[2, 2] = 1; u, v, z = proj(cam, torch.tensor(K, dtype=torch.float32).cuda(), P)
        h, w = sil.shape; ui = u.round().long(); vi = v.round().long(); ok = (z > 0.1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        silt = torch.from_numpy(sil).cuda(); hit = torch.zeros(N, dtype=torch.bool, device='cuda'); hit[ok] = silt[vi[ok], ui[ok]] > 0
        inside += hit.int()
    hull = P[inside >= a.min_views]
    if a.save_hull:
        out.mkdir(parents=True, exist_ok=True); np.savez_compressed(out / f'hull_{fr:06d}.npz', pts=hull.cpu().numpy().astype(np.float32), voxel=a.voxel)
    if a.no_masks:
        print(f'frame {fr}: hull voxels {len(hull)}', flush=True); continue
    for name, cam in te.items():
        K = cam.K; W = int(round(2 * K[0, 2])); H = int(round(2 * K[1, 2])); sc = 0.25
        Ks = torch.tensor(K * sc, dtype=torch.float32).cuda(); Ks[2, 2] = 1
        m = np.zeros((int(H * sc), int(W * sc)), np.uint8)
        if len(hull):
            u, v, z = proj(cam, Ks, hull); ok = (z > 0.1) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
            m[v[ok].long().cpu().numpy(), u[ok].long().cpu().numpy()] = 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            if a.dilate > 0: m = cv2.dilate(m, np.ones((a.dilate * 2 + 1,) * 2, np.uint8))
        (out / name).mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out / name / f'{fr:06d}.png'), cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR))
    print(f'frame {fr}: hull voxels {len(hull)}', flush=True)
print('DONE')
