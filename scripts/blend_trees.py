"""Blend raw and Difix render trees per camera id: out = (1-a)*raw + a*difix.  Resumable; JPEG q95.
Usage: python blend_trees.py <raw_root> <difix_root> <out_root> --map '{"02":0.7,...}' [--cases a,b]"""
import sys, os, json, argparse
import numpy as np, cv2
ap=argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('fix'); ap.add_argument('out'); ap.add_argument('--map', required=True); ap.add_argument('--cases', default=''); ap.add_argument('--default', type=float, default=0.5, help='alpha for camera ids not in --map (uniform 0.5 validated on both val scenes)')
a=ap.parse_args(); amap=json.loads(a.map); cases=a.cases.split(',') if a.cases else sorted(os.listdir(a.fix)); n=0; miss=0
for c in cases:
    for v in sorted(os.listdir(f'{a.fix}/{c}')):
        al=amap.get(v, a.default); os.makedirs(f'{a.out}/{c}/{v}', exist_ok=True)
        for f in sorted(os.listdir(f'{a.raw}/{c}/{v}')):
            o=f'{a.out}/{c}/{v}/{f}'
            if os.path.exists(o): continue
            pf=f'{a.fix}/{c}/{v}/{f}'
            if not os.path.exists(pf): miss+=1; continue
            r=cv2.imread(f'{a.raw}/{c}/{v}/{f}').astype(np.float32); x=cv2.imread(pf).astype(np.float32)
            cv2.imwrite(o, np.clip((1-al)*r+al*x,0,255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY,95]); n+=1
print(f'blended {n} images, {miss} missing difix outputs')
