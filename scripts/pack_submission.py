"""Package a render tree into the portal zip. Usage: pack_submission.py <out_zip> <renders_root>
Layout: team_name.txt + sparseViewTrack/renders/<scene>/<view>/<frame:06d>.jpg (all 5 scenes, 8 views each, every 10th frame)."""
import sys, zipfile, hashlib
from pathlib import Path
out, root = sys.argv[1], Path(sys.argv[2])
scenes = ['004_1_seq0', '006_1_seq0', '007_0_seq0', '009_0_seq0', '011_0_seq0']
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
    z.writestr('team_name.txt', 'Doreen071')
    for s in scenes:
        d = root / s
        first = sorted(p for p in (Path('/work/doreen071/vvc/data') / s / 'images').iterdir() if p.is_dir())[0]
        need = (len(list(first.glob('*.jpg'))) + 9) // 10
        views = sorted(v for v in d.iterdir() if v.is_dir())
        assert len(views) == 8, f'{s}: {len(views)} views'
        for v in views:
            fs = sorted(v.glob('*.jpg')); assert len(fs) >= need, f'{s}/{v.name}: {len(fs)} < {need}'
            for f in fs: z.write(f, f'sparseViewTrack/renders/{s}/{v.name}/{f.name}'); n += 1
h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
print(f'wrote {out}: {n} images, sha256 {h[:32]}')
