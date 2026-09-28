"""Per-view static background via temporal median of another sequence's images of the same physical camera.
Usage: python make_bg.py <src_case_dir> <view> <out.png> [--n 60]"""
import argparse, numpy as np, cv2
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('view'); ap.add_argument('out'); ap.add_argument('--n',type=int,default=60)
a=ap.parse_args(); files=sorted((Path(a.src)/'images'/a.view).glob('*.jpg')); idx=np.linspace(0,len(files)-1,a.n).astype(int)
stack=np.stack([cv2.imread(str(files[i])) for i in idx]); med=np.median(stack,axis=0).astype(np.uint8)
cv2.imwrite(a.out, med, [cv2.IMWRITE_JPEG_QUALITY,98] if a.out.endswith('.jpg') else []); print(a.out, med.shape, 'from', len(idx), 'frames')
