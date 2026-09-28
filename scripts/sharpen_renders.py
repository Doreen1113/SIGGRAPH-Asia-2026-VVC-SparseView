"""Apply the two-scene-validated unsharp mask (a0.4 r4) to a rendered submission tree."""
import sys, os
from pathlib import Path
import numpy as np, cv2
src, dst, amt, rad = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
k = rad*2+1
n = 0
for f in sorted(src.rglob('*.jpg')):
    rel = f.relative_to(src); out = dst/rel; out.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(f)).astype(np.float32)/255.0
    blur = cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT)
    sharp = np.clip(img + amt*(img-blur), 0, 1)
    cv2.imwrite(str(out), (sharp*255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1
print(f'sharpened {n} images -> {dst}')
