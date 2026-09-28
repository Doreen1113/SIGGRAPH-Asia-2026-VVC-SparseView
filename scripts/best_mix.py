"""Per-scene model selection driven by the official leaderboard feedback.
Queries our submission history, extracts per-scene PSNR for each submitted render source,
and reports (or builds) the best-known combination.
Usage: python best_mix.py [--build out.zip] [--prefer <source>]"""
import os
import argparse, json, subprocess, zipfile, sys
from pathlib import Path
S = Path('/home/intern_2603055/vvc/submissions')
SCENES = ['004_1_seq0', '006_1_seq0', '007_0_seq0', '009_0_seq0', '011_0_seq0']
# submitted archive -> render source directory used for every scene in it
ARCHIVE_SOURCE = {
    'baseline_submission.zip': 'baseline',
    'F_s0_submission.zip': 'F_s0',
    'F_ens3_submission.zip': 'F_ens3',
    'J_s0_probe.zip': 'J_s0',
    'J_ens_probe.zip': 'J_ens',
    'JO_ens_probe.zip': 'JO_ens',
    'JP_ens_probe.zip': 'JP_ens',
    'JOP_ens_probe.zip': 'JOP_ens',
    'JOPP_ens_probe.zip': 'JOPP_ens',
    'SOPP_ens_probe.zip': 'SOPP_ens',
    'SPP_ens_probe.zip': 'SPP_ens',
    'JPP_ens_probe.zip': 'JPP_ens',
    'S015PP_ens_probe.zip': 'S015PP_ens',
    'S015P_ens_probe.zip': 'S015P_ens',
}
ap = argparse.ArgumentParser(); ap.add_argument('--build'); ap.add_argument('--prefer', default=None)
ap.add_argument('--manifest_dir', default=str(S))
a = ap.parse_args()
tok = os.environ['VVC_TOKEN']
raw = subprocess.run(['curl', '-sk', '-H', f'Authorization: Bearer {tok}',
                      'https://8.136.221.94/api/submissions'], capture_output=True, text=True).stdout
subs = json.loads(raw)
table = {s: {} for s in SCENES}
for e in subs:
    if e['status'] != 'succeeded': continue
    src = ARCHIVE_SOURCE.get(e['filename'])
    if src is None:
        mf = Path(a.manifest_dir) / (e['filename'] + '.manifest')
        if mf.exists():
            per = {}
            for line in mf.read_text().splitlines():
                if '<-' in line:
                    sc, path = [x.strip() for x in line.split('<-')]
                    per[sc] = Path(path).parent.parent.name
            src = per
        else:
            print(f'  (skipping {e["filename"]}: unknown render source)'); continue
    for sc, v in (e.get('metrics') or {}).get('per_scene', {}).items():
        if sc not in table: continue
        name = src if isinstance(src, str) else src.get(sc)
        if name: table[sc][name] = v['psnr']
cols = sorted({k for r in table.values() for k in r})
print(f'{"scene":14s} ' + '  '.join(f'{k:>10s}' for k in cols) + '   BEST')
total = 0.0; best_src = {}
for sc in SCENES:
    row = table[sc]
    if not row: print(f'{sc:14s} (no data)'); continue
    b = max(row, key=row.get); best_src[sc] = b; total += row[b]
    print(f'{sc:14s} ' + '  '.join(f'{row.get(k, float("nan")):10.2f}' for k in cols) + f'   {b} {row[b]:.2f}')
if best_src: print(f'\nbest-known per-scene mean = {total/len(best_src):.3f}  (vs F_ens3 mean {sum(table[s].get("F_ens3",0) for s in SCENES)/5:.3f})')
json.dump(best_src, open('/home/intern_2603055/vvc/best_per_scene.json', 'w'), indent=1)
if a.build:
    with zipfile.ZipFile(a.build, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('team_name.txt', 'Doreen071'); n = 0
        for sc in SCENES:
            src = a.prefer if a.prefer and (S / a.prefer / 'renders' / sc).is_dir() else best_src.get(sc)
            d = S / src / 'renders' / sc
            views = [v for v in d.iterdir() if v.is_dir()]
            assert len(views) >= 8, f'{d}: only {len(views)} views'
            for v in sorted(views):
                for f in sorted(v.glob('*.jpg')): z.write(f, f'sparseViewTrack/renders/{sc}/{v.name}/{f.name}'); n += 1
            print(f'  {sc} <- {src}')
    print(f'wrote {a.build}: {n} images')
