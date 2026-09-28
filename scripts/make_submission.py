"""Assemble submission.zip: team_name.txt + sparseViewTrack/renders/<scene>/<view>/<frame:06d>.jpg
Usage: python make_submission.py <team_name> <renders_root> <out_zip>
<renders_root>/<scene>/<view>/*.jpg (scene = case name e.g. 004_1_seq0)"""
import sys, zipfile, os
from pathlib import Path
team, root, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
scenes = sorted(p for p in root.iterdir() if p.is_dir())
with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
    z.writestr('team_name.txt', team)
    n = 0
    for s in scenes:
        for v in sorted(p for p in s.iterdir() if p.is_dir()):
            for f in sorted(v.glob('*.jpg')):
                z.write(f, f'sparseViewTrack/renders/{s.name}/{v.name}/{f.name}'); n += 1
print(f'wrote {out}: {len(scenes)} scenes, {n} images, team={team}')
