"""Read the per-training-camera affine colour correctors out of the trained checkpoints.

Training uses photo = render @ (mat + I) + vec per training camera, so the corrector IS that camera's colour
offset relative to the model's canonical space; render_hidden.py never applies it, which is why hidden views
carry a per-camera colour error. Cameras that are TRAINING cameras in one scene are HIDDEN cameras in another
(15 is training in 007/011 and hidden in 004; 45 is training in 004/009 and hidden in 006), so a corrector
measured in one scene can correct that same physical camera where it is hidden.

First it checks the premise: the same camera's corrector, learned independently in different scenes, should
agree. Prints the equivalent colour shift at a mid-grey and at the scene's own mean colour.
"""
import sys, json
from pathlib import Path
import numpy as np, torch

V = Path('/work/doreen071/vvc')
SCENES = {'004_1_seq0': ['11', '13', '26', '27', '45', '49'],
          '006_1_seq0': ['11', '14', '27', '28', '46', '50'],
          '007_0_seq0': ['12', '15', '28', '29', '47', '51'],
          '009_0_seq0': ['11', '14', '27', '28', '45', '49'],
          '011_0_seq0': ['12', '15', '28', '29', '47', '51']}
MODELS = sys.argv[1:] or ['S5_FGW40_L60K', 'S5_FGW40_L60K_s1', 'FGW40_L60K', 'S5PS']
GREY = np.array([0.45, 0.45, 0.45])          # representative mid-tone in [0,1] RGB

def load(scene, model):
    p = V / 'runs' / f'siga_{scene}' / f'run_{model}' / 'color_correctors.pt'
    if not p.exists(): return None
    cc = torch.load(p, map_location='cpu', weights_only=False)
    # the module is indexed by CAMERA NUMBER (size = max id + 1); only training cameras are ever non-zero
    out = {}
    for i, c in enumerate(cc._correctors):
        mat = c.mat.detach().numpy().astype(np.float64); vec = c.vec.detach().numpy().astype(np.float64)
        if abs(mat).sum() + abs(vec).sum() > 1e-8: out[f'{i:02d}'] = (mat, vec)
    return out

def shift(mat, vec, c=GREY):                 # RGB delta at colour c, in 0-255 levels
    return ((c @ (mat + np.eye(3)) + vec) - c) * 255.0

per_cam = {}
for scene, cams in SCENES.items():
    for model in MODELS:
        cc = load(scene, model)
        if cc is None: continue
        for cam, (mat, vec) in cc.items():
            per_cam.setdefault(cam, []).append((scene, model, mat, vec))
        break   # one model per scene is enough for the consistency check

print('同一支相機在不同場景學到的色彩校正 (中灰處的 RGB 位移, 0-255 色階):')
shared = 0; devs = []
for cam in sorted(per_cam):
    rows = per_cam[cam]
    s = [shift(m, v) for _, _, m, v in rows]
    tag = ' '.join(f"{sc.split('_')[0]}:{np.round(x,1)}" for (sc, _, _, _), x in zip(rows, s))
    if len(rows) > 1:
        shared += 1; d = np.abs(np.array(s) - np.mean(s, 0)).mean(); devs.append(d)
        print(f'  cam {cam}: {tag}   跨場景平均偏差 {d:.2f}')
    else:
        print(f'  cam {cam}: {tag}')
if devs:
    mag = np.mean([np.abs(np.mean([shift(m, v) for _, _, m, v in per_cam[c]], 0)).mean() for c in per_cam])
    print(f'\n出現在多個場景的相機: {shared} 支;跨場景平均偏差 {np.mean(devs):.2f} 色階,'
          f' 相對於位移本身的平均大小 {mag:.2f} 色階')

out = {}
for cam, rows in per_cam.items():
    mat = np.mean([m for _, _, m, _ in rows], 0); vec = np.mean([v for _, _, _, v in rows], 0)
    out[cam] = {'mat': mat.tolist(), 'vec': vec.tolist(), 'scenes': [sc for sc, _, _, _ in rows],
                'shift_at_grey': list(map(float, shift(mat, vec)))}
json.dump(out, open(V / 'work/priors/correctors_by_cam.json', 'w'), indent=1)
print(f'\n寫出 work/priors/correctors_by_cam.json ({len(out)} 支相機)')

HIDDEN = {'004_1_seq0': ['02', '09', '15', '16', '17', '31', '36', '44'],
          '006_1_seq0': ['01', '09', '16', '17', '18', '32', '37', '45'],
          '009_0_seq0': ['01', '09', '16', '17', '18', '31', '36', '44']}
val = set(json.load(open(V / 'work/priors/cc2_012_0.json')))
print('\n各場景隱藏相機的可修正覆蓋率:')
for scene, cams in HIDDEN.items():
    have = [c for c in cams if c in val or c in out]
    src = ', '.join(f"{c}{'(驗證量測)' if c in val else '(校正器)'}" for c in have)
    print(f'  {scene}: {len(have)}/8  -> {src}')
