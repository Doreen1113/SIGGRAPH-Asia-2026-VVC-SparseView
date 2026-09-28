"""Measure the metric cost of lower JPEG quality (to fit under the evaluator's disk quota)."""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
case = Path('/home/intern_2603055/vvc/data/001_1_seq0')
src = Path('/home/intern_2603055/vvc/submissions/J_ens/renders/004_1_seq0')  # our real renders
psnr_f = PeakSignalNoiseRatio(data_range=1.0).cuda(); ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
views = sorted(p for p in src.iterdir() if p.is_dir())[:3]
files = [sorted(v.glob('*.jpg'))[0] for v in views] + [sorted(v.glob('*.jpg'))[10] for v in views]
tot = {}
for q in [95, 92, 90, 85, 80]:
    ds, sz = [], 0
    for f in files:
        ref = cv2.imread(str(f))                       # our q95 render = the reference here
        ok, buf = cv2.imencode('.jpg', ref, [cv2.IMWRITE_JPEG_QUALITY, q]); sz += len(buf)
        deg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        a = torch.from_numpy(ref[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
        b = torch.from_numpy(deg[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
        ds.append((psnr_f(b,a).item(), ssim_f(b,a).item(), lpips_f(b,a).item()))
    m = np.mean(ds,0); tot[q] = (m, sz/len(files)/1024)
    print(f'q={q}: vs our q95 render  psnr={m[0]:.2f} ssim={m[1]:.5f} lpips={m[2]:.5f}   avg {sz/len(files)/1024:.0f} KB/img  -> est zip {sz/len(files)*2056/1e9:.2f} GB')
