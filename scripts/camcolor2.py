"""Per-camera colour correction, in the form each rig's own validation scene says is safest.

C6 used a plain additive offset and lost SSIM on the real test set. Measured cause (val, production-faithful
base): SSIM's luminance term is RELATIVE, so a +8 offset on a window whose mean is 10 is an 80% error while on
a mean of 150 it is 5%. Per-window SSIM delta by brightness for add_full: [0-0.1] -0.042, [0.1-0.25] -0.000,
[0.25-0.5] +0.005. Two forms avoid that damage:
  taper  x + d * lum/(lum+TAU)   dark pixels get proportionally less of the offset  (best on 012_0 = 011's rig)
  gain   x * g                   multiplicative, leaves black at black              (best on 001_1 = 007's rig)
Both improve all six metrics on their own rig's validation scene; the additive form does not.

The prior is fitted leave-nothing-out on the released validation scene sharing the rig (012_0 -> 011_0,
001_1 -> 007_0); it is 3 offsets + 3 gains per camera. No validation pixels enter the output.

  camcolor2.py fit <val_dump> <render_fmt> <out.json>
  camcolor2.py apply <src.zip> <dst_root> <case> <prior.json> <form: add|taper|gain> <scale> [--shard i/n] [--tau 0.10]
"""
import sys, os, json, zipfile
import numpy as np, cv2

mode = sys.argv[1]
if mode == 'fit':
    D, fmt, out = sys.argv[2:5]
    man = json.load(open(f'{D}/manifest.json')); off = {}; gain = {}
    for e in man:
        gt = cv2.imread(f"{D}/{e['tag']}_gt.png").astype(np.float64)
        r = cv2.imread(fmt.format(D=D, tag=e['tag'])).astype(np.float64)
        off.setdefault(e['view'], []).append((gt - r).reshape(-1, 3).mean(0))
        gain.setdefault(e['view'], []).append(gt.reshape(-1, 3).mean(0) / np.clip(r.reshape(-1, 3).mean(0), 1e-3, None))
    prior = {v: {'off': np.mean(off[v], 0).tolist(), 'gain': np.mean(gain[v], 0).tolist()} for v in off}
    json.dump(prior, open(out, 'w'), indent=1)
    for v, p in prior.items():
        print(v, 'off', np.round(p['off'], 2).tolist(), 'gain', np.round(p['gain'], 4).tolist())
else:
    zp, dst, case, pj, form, scale = sys.argv[2:8]; scale = float(scale)
    args = sys.argv[8:]; si, sn = 0, 1; tau = 0.10
    for i, a in enumerate(args):
        if a == '--shard': si, sn = map(int, args[i + 1].split('/'))
        if a == '--tau': tau = float(args[i + 1])
    prior = json.load(open(pj))
    z = zipfile.ZipFile(zp); n = 0
    names = sorted(x for x in z.namelist() if x.endswith('.jpg') and f'/renders/{case}/' in x)
    for k, name in enumerate(names):
        if k % sn != si: continue
        rel = name.split('/renders/', 1)[1]; view = rel.split('/')[1]; out = f'{dst}/{rel}'
        if os.path.exists(out) or view not in prior: continue
        im = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
        if form == 'gain':
            g = np.array(prior[view]['gain'], np.float32).reshape(1, 1, 3)
            y = im * (1.0 + scale * (g - 1.0))
        else:
            d = scale * np.array(prior[view]['off'], np.float32).reshape(1, 1, 3)
            if form == 'taper':
                lum = im.mean(2, keepdims=True) / 255.0
                d = d * (lum / (lum + tau))
            y = im + d
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, np.clip(y, 0, 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
    print(f'APPLY DONE {case} form={form} scale={scale} tau={tau}: {n}')
