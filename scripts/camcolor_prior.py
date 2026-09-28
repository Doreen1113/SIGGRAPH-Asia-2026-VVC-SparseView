"""Per-camera colour-offset prior learned from a released validation scene that shares the physical camera rig
with a test scene (012_0 -> 011_0, 001_1 -> 007_0; identical hidden-camera IDs). The prior is 3 numbers per camera
(mean GT-minus-render offset over that validation scene's frames); no validation pixels enter the output.
Usage:
  camcolor_prior.py fit <val_dump_dir> <render_variant_subpath_fmt> <out.json>
  camcolor_prior.py apply <src.zip> <dst_root> <case> <prior.json> <scale> [--shard i/n]"""
import sys, os, json, zipfile
import numpy as np, cv2
mode = sys.argv[1]
if mode == 'fit':
    D, fmt, out = sys.argv[2:5]
    man = json.load(open(f'{D}/manifest.json')); acc = {}
    for e in man:
        gt = cv2.imread(f"{D}/{e['tag']}_gt.png").astype(np.float64); r = cv2.imread(fmt.format(D=D, tag=e['tag'])).astype(np.float64)
        acc.setdefault(e['view'], []).append((gt - r).reshape(-1, 3).mean(0))
    prior = {v: np.mean(x, 0).tolist() for v, x in acc.items()}
    json.dump(prior, open(out, 'w'), indent=1); print({v: np.round(p, 2).tolist() for v, p in prior.items()})
else:
    zp, dst, case, pj, scale = sys.argv[2:7]; scale = float(scale)
    si, sn = (map(int, sys.argv[8].split('/')) if len(sys.argv) > 8 else (0, 1))
    prior = {v: np.array(p) for v, p in json.load(open(pj)).items()}
    z = zipfile.ZipFile(zp); n = 0
    names = sorted(x for x in z.namelist() if x.endswith('.jpg') and f'/renders/{case}/' in x)
    for k, name in enumerate(names):
        if k % sn != si: continue
        rel = name.split('/renders/', 1)[1]; view = rel.split('/')[1]; out = f'{dst}/{rel}'
        if os.path.exists(out): continue
        im = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
        im = np.clip(im + scale * prior[view].astype(np.float32), 0, 255)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, im.round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
    print(f'APPLY DONE {case} scale {scale}: {n}')
