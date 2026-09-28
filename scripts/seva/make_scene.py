"""Build a Stable-Virtual-Camera (ReconfusionParser) scene folder from a VVC case + frame.
Usage: ftg.sh make_scene.py <case_dir> <frame> <out_dir> [--W 1024 --H 576] [--heldout_dir DIR]
Train views: train_intri/extri (images from case); target views: test_intri/extri (images optional, GT if present).
Images are resized so the short side = H then center-cropped to W; intrinsics adjusted accordingly."""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
ap = argparse.ArgumentParser()
ap.add_argument('case'); ap.add_argument('frame', type=int); ap.add_argument('out')
ap.add_argument('--W', type=int, default=1024); ap.add_argument('--H', type=int, default=576)
ap.add_argument('--test_intri', default=None); ap.add_argument('--test_extri', default=None)
a = ap.parse_args()
case = Path(a.case); out = Path(a.out); (out / 'images').mkdir(parents=True, exist_ok=True)
tr = read_camera(case / 'train_intri.yml', case / 'train_extri.yml')
te = read_camera(a.test_intri or case / 'test_intri.yml', a.test_extri or case / 'test_extri.yml')
frames = []; train_ids = []; test_ids = []
def add(name, cam, is_train):
    K = cam.K.copy(); W0 = int(round(2 * K[0, 2])); H0 = int(round(2 * K[1, 2]))
    s = a.H / H0; Ws = int(round(W0 * s)); assert Ws >= a.W, (Ws, a.W)
    x0 = (Ws - a.W) // 2
    img_path = case / 'images' / name / f'{a.frame:06d}.jpg'
    fp = None
    if img_path.exists():
        im = cv2.imread(str(img_path)); assert im.shape[1] == W0 and im.shape[0] == H0, (im.shape, W0, H0)
        im = cv2.resize(im, (Ws, a.H), interpolation=cv2.INTER_AREA)[:, x0:x0 + a.W]
        fp = f'images/{name}.png'; cv2.imwrite(str(out / fp), im)
    c2w = np.linalg.inv(cam.w2c); c2w[:, 1:3] *= -1  # OpenCV -> OpenGL
    frames.append({'file_path': fp, 'transform_matrix': c2w.tolist(), 'fl_x': K[0, 0] * s, 'fl_y': K[1, 1] * s,
                   'cx': K[0, 2] * s - x0, 'cy': K[1, 2] * s, 'w': a.W, 'h': a.H, 'name': name})
    (train_ids if is_train else test_ids).append(len(frames) - 1)
for n, c in tr.items(): add(n, c, True)
for n, c in te.items(): add(n, c, False)
json.dump({'frames': frames}, open(out / 'transforms.json', 'w'), indent=1)
json.dump({'train_ids': train_ids, 'test_ids': test_ids}, open(out / f'train_test_split_{len(train_ids)}.json', 'w'))
print(f'wrote {out}: {len(train_ids)} inputs, {len(test_ids)} targets; test names {[frames[i]["name"] for i in test_ids]}')
