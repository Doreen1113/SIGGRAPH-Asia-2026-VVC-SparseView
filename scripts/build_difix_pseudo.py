"""Difix3D's *3D* step, which we never ran: render the six-member ensemble at this case's TEST camera poses,
fix each frame with Difix (ref = nearest TRAINING camera, same frame), and export in the FTGSPP_PSEUDO_DIR
format so the corrections are distilled into one 3D model instead of being blended in 2D.
Compliant: inputs are only this case's 6 training views (their renders + reference frames) and the Difix prior.
No test-view images are read (test cases have none; for val they are deliberately not touched).
Usage: ftg.sh build_difix_pseudo.py <scene> <run1,run2,...> <out_dir> [every=20] [scale=0.5]"""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
scene = sys.argv[1]; runs = sys.argv[2].split(','); out = Path(sys.argv[3])
EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 20; SC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
case = Path(f'/home/intern_2603055/vvc/data/{scene}'); R = Path(f'/home/intern_2603055/vvc/runs/siga_{scene}')
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
out.mkdir(parents=True, exist_ok=True); dump = out/'dump'; dump.mkdir(exist_ok=True)
gs = [Gaussians.load(R/r/'gaussians.pt').cuda().eval() for r in runs]
trc = {k: -c.w2c[:3, :3].T @ c.w2c[:3, 3] for k, c in tr.items()}
centers = np.stack(list(trc.values())); rad = float(np.linalg.norm(centers-centers.mean(0), axis=1).mean())
def tcount(g):
    cnt = torch.zeros(len(g), device='cuda', dtype=torch.int32)
    for tc in tr.values():
        K = torch.tensor(tc.K, dtype=torch.float32).cuda(); W = 2*K[0, 2]; H = 2*K[1, 2]; w2 = torch.tensor(tc.w2c, dtype=torch.float32).cuda()
        c3 = w2[:3, :3] @ g.means.T + w2[:3, 3:4]; z = c3[2]; u = K[0, 0]*c3[0]/z + K[0, 2]; v = K[1, 1]*c3[1]/z + K[1, 2]
        cnt += ((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).int()
    return cnt
cnts = [tcount(g) for g in gs]
nframes = len(sorted((case/'images'/next(iter(tr))).glob('*.jpg')))   # frame count from a TRAIN view only
man = []; items = []
with torch.inference_mode():
    for vn, cam in te.items():
        K = cam.K.copy(); W = int(round(2*K[0, 2])); H = int(round(2*K[1, 2]))
        Ks = K.copy(); Ks[:2] *= SC; w = int(round(W*SC)); h = int(round(H*SC))
        w2c = torch.tensor(cam.w2c, dtype=torch.float32).cuda()[None]; Kt = torch.tensor(Ks, dtype=torch.float32).cuda()[None]
        w2c4 = np.eye(4, dtype=np.float64); w2c4[:3, :4] = np.asarray(cam.w2c)[:3, :4]
        ccn = -cam.w2c[:3, :3].T @ cam.w2c[:3, 3]; cc = torch.tensor(ccn, dtype=torch.float32).cuda(); off = float(offs.get(vn, 0.0))
        ref_cam = min(trc, key=lambda k: np.linalg.norm(trc[k]-ccn)); refs = sorted((case/'images'/ref_cam).glob('*.jpg'))
        mv = [g.mask(~((torch.norm(g.means-cc, dim=1) < 1.1*rad) & (c < 4))) for g, c in zip(gs, cnts)]
        for i in range(0, nframes, EVERY):
            img = None
            for g2 in mv:
                im2, _, _ = g2(t=torch.tensor(i/60.0-off).cuda(), w2c=w2c, intrinsic=Kt, shape=(h, w)); img = im2.clamp(0, 1) if img is None else img+im2.clamp(0, 1)
            img = (img/len(mv))[0].cpu().numpy()
            tag = f'{vn}_{i:06d}'
            cv2.imwrite(str(dump/f'{tag}_render.png'), (img[..., ::-1]*255).round().astype(np.uint8))
            ref = cv2.resize(cv2.imread(str(refs[i])), (w, h), interpolation=cv2.INTER_AREA); cv2.imwrite(str(dump/f'{tag}_ref.png'), ref)
            man.append(dict(tag=tag, view=vn, frame=i))
            items.append(dict(frame=i, view=vn, t=i/60.0-off, K=Ks.tolist(), w2c=w2c4.tolist(), w=w, h=h, path=f'{i:06d}/{vn}.png'))
        print('rendered', vn, flush=True)
json.dump(man, open(dump/'manifest.json', 'w')); json.dump(items, open(out/'index.json', 'w'))
print('RENDER DONE', len(man), 'images at', w, 'x', h)
