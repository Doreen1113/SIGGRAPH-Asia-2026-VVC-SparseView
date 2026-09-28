"""Convert an existing Difix-corrected render tree into FTGSPP_PSEUDO_DIR format, so the 2D corrections can be
distilled into a single 3D model (Difix3D's "3D" step) instead of only being blended in 2D.

Compliant: the render tree came from models trained on this case's 6 training views only, and Difix (an external
general prior) was applied with a reference frame from the case's own training cameras. No test-view image is read.

Usage: tta_to_pseudo.py <scene> <fix_root> <out_dir> [--every 2] [--scale 0.5]
  <fix_root>/<scene>/<view>/<frame:06d>.jpg   (frames are every 10th, as in the submission layout)
  --every N   keep every Nth available frame (2 => every 20th original frame, matching the historical recipe)
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/work/doreen071/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera

ap = argparse.ArgumentParser(); ap.add_argument('scene'); ap.add_argument('fix_root'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=2); ap.add_argument('--scale', type=float, default=0.5)
ap.add_argument('--fps', type=float, default=60.0)
a = ap.parse_args()

case = Path(f'/work/doreen071/vvc/data/{a.scene}')
te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text()) if (case/'t_offsets.json').exists() else {}
out = Path(a.out); (out/'img').mkdir(parents=True, exist_ok=True)

index = []
src = Path(a.fix_root)/a.scene
for vn in sorted(p.name for p in src.iterdir() if p.is_dir()):
    if vn not in te:
        print('skip view without test calibration:', vn); continue
    cam = te[vn]
    K = cam.K.copy(); W = int(round(2*K[0, 2])); H = int(round(2*K[1, 2]))
    Ks = K.copy(); Ks[:2] *= a.scale; w = int(round(W*a.scale)); h = int(round(H*a.scale))
    off = float(offs.get(vn, 0.0))
    files = sorted((src/vn).glob('*.jpg'))[::a.every]
    for f in files:
        frame = int(f.stem)
        im = cv2.imread(str(f))
        if im is None: continue
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        rel = f'img/{vn}_{frame:06d}.png'
        cv2.imwrite(str(out/rel), im)
        index.append(dict(path=rel, view=vn, frame=frame, t=frame/a.fps - off,
                          w2c=cam.w2c.tolist(), K=Ks.tolist(), h=h, w=w))
json.dump(index, open(out/'index.json', 'w'))
print(f'{a.scene}: wrote {len(index)} pseudo views ({w}x{h}) -> {out}')
