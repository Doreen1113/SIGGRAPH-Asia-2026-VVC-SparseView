"""Build a submission from the best available renders per scene: python mix_submission.py <out_zip> <renders_root_pref1> [<renders_root_pref2> ...]
For each scene, the first root that contains a complete render set (8 views) is used."""
import sys, zipfile
from pathlib import Path
args = sys.argv[1:]; selfcap = None
if '--selfcap' in args:
    i = args.index('--selfcap'); selfcap = Path(args[i + 1]); args = args[:i] + args[i + 2:]
out = args[0]; roots = [Path(r) for r in args[1:]]
scenes = ['004_1_seq0', '006_1_seq0', '007_0_seq0', '009_0_seq0', '011_0_seq0']
with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
    z.writestr('team_name.txt', 'Doreen071'); n = 0
    for s in scenes:
        src = None
        for r in roots:
            d = r / s
            if not d.is_dir(): continue
            imgdir = Path('/home/intern_2603055/vvc/data') / s / 'images'; first = sorted(p for p in imgdir.iterdir() if p.is_dir())[0]
            need = (len(list(first.glob('*.jpg'))) + 9) // 10
            views = [v for v in d.iterdir() if v.is_dir()]
            if len(views) >= 8 and all(len(list(v.glob('*.jpg'))) >= need for v in views): src = d; break
            print('  incomplete:', d)
        if src is None: print('!! no renders for', s); continue
        for v in sorted(p for p in src.iterdir() if p.is_dir()):
            for f in sorted(v.glob('*.jpg')): z.write(f, f'sparseViewTrack/renders/{s}/{v.name}/{f.name}'); n += 1
        print(s, '<-', src)
    if selfcap is not None:
        for s in ['0512_bike', '0525_corgi', '0811_yoga']:
            d = selfcap / s
            if not d.is_dir(): print('!! no selfcap renders for', s); continue
            for v in sorted(p for p in d.iterdir() if p.is_dir()):
                for f in sorted(v.glob('*.jpg')): z.write(f, f'sparseViewTrack/renders/{s}/{v.name}/{f.name}'); n += 1
            print(s, '<-', d)
print('wrote', out, n, 'images')
