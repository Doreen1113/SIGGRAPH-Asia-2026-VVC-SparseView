"""Pixel-average the FINAL composited output of several already-scored submission packages.
Different post-processing chains (different model ensembles feeding into different Difix TTA runs) applied to
the same camera views are, from a noise-reduction standpoint, just more TTA members: averaging their outputs
should suppress uncorrelated residual noise the way multi-shift Difix TTA already does, without any retraining.
Usage: meta_ensemble.py out.zip a.zip:w a.zip:w ...
"""
import sys, zipfile, numpy as np, cv2
out = sys.argv[1]
srcs = [(a.rsplit(':', 1)[0], float(a.rsplit(':', 1)[1])) for a in sys.argv[2:]]
zips = [(zipfile.ZipFile(p), w) for p, w in srcs]
names = sorted(n for n in zips[0][0].namelist() if n.endswith('.jpg'))
print(f'{len(names)} frames, {len(zips)} sources, weights {[w for _, w in zips]}')
with zipfile.ZipFile(out, 'w') as zo:
    zo.writestr('team_name.txt', zips[0][0].read('team_name.txt'))
    tot = sum(w for _, w in zips)
    for i, n in enumerate(names):
        acc = None
        for z, w in zips:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            acc = im * w if acc is None else acc + im * w
        res = np.clip(acc / tot, 0, 255).round().astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 95])
        zo.writestr(n, buf.tobytes())
        if i % 400 == 0: print(f'{i}/{len(names)}', flush=True)
print('META_ENSEMBLE_DONE', out)
