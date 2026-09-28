"""Can a hidden camera's colour offset be predicted from OTHER cameras on the same physical rig?

All five test scenes draw their cameras from one rig numbered 01-51, and the two released validation scenes
share 007/011's eight hidden cameras exactly. Measured offsets are consistent per CAMERA across two different
validation scenes (cam 17 strongly negative in both, cams 02/10/18/19/46 positive in both), which says the
offset is a camera property, not a scene property -> it should transfer to 004/006/009 for the cameras they
share, and interpolate for the rest.

This script (a) leave-one-out tests spatial interpolation of the offset over camera position on the rig, using
the 8 cameras we have ground truth for, and (b) if that holds up, writes predicted priors for the cameras of
004/006/009 that we have never measured. Pure numpy over calibration files and already-fitted priors: no GT
pixels, no rendering.
"""
import json, re, sys
import numpy as np

V = '/work/doreen071/vvc'

def read_centres(path):
    """Camera centres C = -R^T t from an extri.yml (OpenCV matrices, flat 'data:' lists)."""
    txt = open(path).read()
    out = {}
    for name in re.findall(r'^\s+- "?(\w+)"?\s*$', txt, re.M):
        rot = re.search(rf'Rot_{name}: !!opencv-matrix\n\s+rows: 3\n\s+cols: 3\n\s+dt: d\n\s+data: \[([^\]]+)\]', txt)
        t = re.search(rf'T_{name}: !!opencv-matrix\n\s+rows: 3\n\s+cols: 1\n\s+dt: d\n\s+data: \[([^\]]+)\]', txt)
        if not rot or not t: continue
        R = np.array([float(x) for x in rot.group(1).split(',')]).reshape(3, 3)
        T = np.array([float(x) for x in t.group(1).split(',')]).reshape(3)
        out[name] = -R.T @ T
    return out

cent = {}
for scn in ['012_0_seq0', '001_1_seq0', '004_1_seq0', '006_1_seq0', '007_0_seq0', '009_0_seq0', '011_0_seq0']:
    for f in ['test_extri.yml', 'train_extri.yml']:
        try: cent.setdefault(scn, {}).update(read_centres(f'{V}/data/{scn}/{f}'))
        except FileNotFoundError: pass

# measured offsets (BGR) for the eight shared hidden cameras, from each validation scene
pri = {s: json.load(open(f'{V}/work/priors/cc2_{s}.json')) for s in ['012_0', '001_1']}
cams = sorted(pri['012_0'])
print('量測到的色偏 (BGR, 兩個驗證場景各自) 與相機在環上的位置:')
for c in cams:
    a = np.round(pri['012_0'][c]['off'], 1); b = np.round(pri['001_1'][c]['off'], 1)
    print(f"  cam {c}: 012_0 {a}   001_1 {b}   位置 {np.round(cent['012_0_seq0'][c], 2)}")

# camera-level pattern = mean of the two scenes, with each scene's global mean removed first
pat = {}
for c in cams:
    v = []
    for s in ['012_0', '001_1']:
        o = np.array(pri[s][c]['off'])
        gm = np.mean([pri[s][x]['off'] for x in cams], 0)
        v.append(o - gm)
    pat[c] = np.mean(v, 0)
print('\n去掉每個場景整體平均後的「相機個別特性」(BGR):')
for c in cams: print(f'  cam {c}: {np.round(pat[c],1)}  (兩場景一致性檢查: '
                     f"{np.round(np.array(pri['012_0'][c]['off'])-np.mean([pri['012_0'][x]['off'] for x in cams],0),1)} vs "
                     f"{np.round(np.array(pri['001_1'][c]['off'])-np.mean([pri['001_1'][x]['off'] for x in cams],0),1)})")

P = np.stack([cent['012_0_seq0'][c] for c in cams])
Y = np.stack([pat[c] for c in cams])

def predict(p, Ptr, Ytr, mode, k=3, power=2.0):
    if mode == 'idw':
        d = np.linalg.norm(Ptr - p, axis=1); idx = np.argsort(d)[:k]
        w = 1.0 / np.maximum(d[idx], 1e-6) ** power; w /= w.sum()
        return (Ytr[idx] * w[:, None]).sum(0)
    if mode == 'nearest':
        return Ytr[np.argmin(np.linalg.norm(Ptr - p, axis=1))]
    if mode == 'linear':                      # least squares on [x,y,z,1]
        A = np.hstack([Ptr, np.ones((len(Ptr), 1))])
        coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
        return np.append(p, 1.0) @ coef
    if mode == 'mean':
        return Ytr.mean(0)
    raise ValueError(mode)

print('\n留一法測試:用其他 7 支相機預測第 8 支 (誤差越小越好, 對照「全部猜 0」的基準)')
base = np.abs(Y).mean()
for mode in ['nearest', 'idw', 'linear', 'mean']:
    err = []
    for i, c in enumerate(cams):
        m = np.ones(len(cams), bool); m[i] = False
        err.append(np.abs(predict(P[i], P[m], Y[m], mode) - Y[i]).mean())
    print(f'  {mode:8s} 平均絕對誤差 {np.mean(err):5.2f} 色階   (不修正的話誤差就是 {base:.2f})')

best = sys.argv[1] if len(sys.argv) > 1 else 'idw'
print(f'\n用 {best} 為 004/006/009 沒量過的相機推估色偏:')
out = {}
for scn, key in [('004_1_seq0', '004'), ('006_1_seq0', '006'), ('009_0_seq0', '009')]:
    d = {}
    for c in sorted(cent[scn]):
        if c not in [x for x in cent[scn]]: continue
        p = cent[scn][c]
        if c in pat: v = pat[c]; tag = '(已量測)'
        else: v = predict(p, P, Y, best); tag = '(推估)'
        d[c] = {'off': list(map(float, v)), 'gain': [1.0, 1.0, 1.0]}
        print(f'  {key} cam {c}: {np.round(v,1)} {tag}')
    out[scn] = d
json.dump(out, open(f'{V}/work/priors/interp_pattern.json', 'w'), indent=1)
print('\n寫出 work/priors/interp_pattern.json')
