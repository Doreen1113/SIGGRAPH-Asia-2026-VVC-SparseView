"""Uncertainty-guided blending: per-pixel alpha proportional to how much Difix changed the render (smoothed),
scaled so the MEAN alpha hits a target -> directly comparable with the uniform blend at the same mean alpha.
Rationale: spend SSIM only where the raw render is already bad."""
import sys, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
d = Path(sys.argv[1]); v = sys.argv[2]; targets = [float(t) for t in sys.argv[3].split(',')]; rad = int(sys.argv[4]) if len(sys.argv) > 4 else 15
man = json.load(open(d/'manifest.json'))
for e in man:
    R = np.asarray(Image.open(d/f"{e['tag']}_render.png").convert('RGB'), np.float32)
    D = np.asarray(Image.open(d/f"{e['tag']}_{v}.png").convert('RGB'), np.float32)
    diff = cv2.blur(np.abs(D-R).mean(2), (2*rad+1, 2*rad+1))
    for t in targets:
        lo, hi = 0.0, 1.0
        while np.clip(hi*diff, 0, 1).mean() < t and hi < 1e6: hi *= 2
        for _ in range(40):
            mid = (lo+hi)/2
            if np.clip(mid*diff, 0, 1).mean() < t: lo = mid
            else: hi = mid
        a = np.clip(hi*diff, 0, 1)[..., None]
        Image.fromarray(np.clip(a*D+(1-a)*R, 0, 255).astype(np.uint8)).save(d/f"{e['tag']}_{v}adp{int(round(t*100)):03d}.png")
    print(e['tag'], flush=True)
