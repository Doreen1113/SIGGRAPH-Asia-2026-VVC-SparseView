"""Dense blend sweep: blending is pure CPU once the full-strength outputs exist, so sample the whole
LPIPS-vs-SSIM frontier for several Difix variants and compare them fairly at matched SSIM."""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
d = Path(sys.argv[1]); variants = sys.argv[2].split(','); alphas = [float(x) for x in sys.argv[3].split(',')]
man = json.load(open(d/'manifest.json'))
for v in variants:
    for a in alphas:
        tagname = f"{v}a{int(round(a*100)):03d}"
        for e in man:
            out = d/f"{e['tag']}_{tagname}.png"
            if out.exists(): continue
            raw = np.asarray(Image.open(d/f"{e['tag']}_render.png").convert('RGB'), np.float32)
            fix = np.asarray(Image.open(d/f"{e['tag']}_{v}.png").convert('RGB'), np.float32)
            Image.fromarray(np.clip(a*fix + (1-a)*raw, 0, 255).astype(np.uint8)).save(out)
        print(tagname, flush=True)
