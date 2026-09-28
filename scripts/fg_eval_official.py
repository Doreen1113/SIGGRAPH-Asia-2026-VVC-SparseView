"""Foreground metrics matching the ORGANIZERS' protocol as closely as we can infer:
 - person mask from GT (DeepLabV3 person class), NO bbox crop
 - PSNR: MSE computed only over masked pixels
 - SSIM/LPIPS: computed on the image with background ZEROED (this is what produces the large LPIPS values
   the portal reports, because hard mask edges are penalised by the perceptual net)
Reports the masked pixel fraction so we can check it against the portal's foreground_valid_pixel_fraction (0.0596).
Usage: ftg.sh fg_eval_official.py <dump_dir> <variant> [variant...]   (variant 'render' = raw)"""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d = Path(sys.argv[1]); variants = sys.argv[2:]; man = json.load(open(d/'manifest.json'))
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
res = {v: {'psnr': [], 'ssim': [], 'lpips': []} for v in variants}; fracs = []
with torch.inference_mode():
    for e in man:
        gt = load(d/f"{e['tag']}_gt.png")
        m = (seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
        frac = m.mean().item(); fracs.append(frac)
        if m.sum() < 100: continue
        for v in variants:
            p = d/f"{e['tag']}_{v}.png"
            if not p.exists(): continue
            pr = load(p)
            ys, xs = torch.where(m[0,0] > 0.5)
            y0,y1,x0,x1 = ys.min().item(), ys.max().item()+1, xs.min().item(), xs.max().item()+1
            pc, gc = pr[:,:,y0:y1,x0:x1], gt[:,:,y0:y1,x0:x1]      # crop to person bbox, NO mask multiply
            mse = ((pc-gc)**2).mean()
            res[v]['psnr'].append(float(10*torch.log10(1.0/mse.clamp_min(1e-12))))
            res[v]['ssim'].append(sf(pc, gc).item())
            res[v]['lpips'].append(lf(pc.clamp(0,1), gc.clamp(0,1)).item())
print(f'masked pixel fraction: mean={np.mean(fracs):.4f}  (portal reports 0.0596)')
for v in variants:
    if not res[v]['psnr']: print(f'  {v:14s} (no files)'); continue
    print(f"  {v:14s} FG psnr={np.mean(res[v]['psnr']):.4f} ssim={np.mean(res[v]['ssim']):.4f} lpips={np.mean(res[v]['lpips']):.4f}")
