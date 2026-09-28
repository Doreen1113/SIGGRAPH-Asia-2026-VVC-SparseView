"""FULL + FG (organizer-style bbox crop, mask from GT) metrics with BOTH LPIPS backbones (alex, vgg).
Usage: metrics_both.py <dump_dir> <variant1> [variant2 ...]   ('render' = raw)"""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d = Path(sys.argv[1]); variants = sys.argv[2:]; man = json.load(open(d/'manifest.json'))
pf = PeakSignalNoiseRatio(data_range=1.0).cuda(); sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
la = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda(); lv = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
res = {v: {'full': [], 'fg': []} for v in variants}
with torch.inference_mode():
    for e in man:
        gt = load(d/f"{e['tag']}_gt.png")
        m = (seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)
        box = None
        if m.sum() >= 100:
            ys, xs = torch.where(m); box = (ys.min().item(), ys.max().item()+1, xs.min().item(), xs.max().item()+1)
        for v in variants:
            p = d/f"{e['tag']}_{v}.png"
            if not p.exists(): continue
            pr = load(p).clamp(0, 1)
            res[v]['full'].append((pf(pr, gt).item(), sf(pr, gt).item(), la(pr, gt).item(), lv(pr, gt).item()))
            if box:
                y0,y1,x0,x1 = box; pc, gc = pr[:,:,y0:y1,x0:x1], gt[:,:,y0:y1,x0:x1]
                mse = ((pc-gc)**2).mean()
                res[v]['fg'].append((float(10*torch.log10(1.0/mse.clamp_min(1e-12))), sf(pc, gc).item(), la(pc, gc).item(), lv(pc, gc).item()))
print(f"{'variant':16s} {'FULL psnr/ssim/alex/vgg':>34s}   {'FG psnr/ssim/alex/vgg':>34s}")
for v in variants:
    if not res[v]['full']: print(f'{v:16s} (no files)'); continue
    f = np.mean(res[v]['full'], 0); g = np.mean(res[v]['fg'], 0) if res[v]['fg'] else [float('nan')]*4
    print(f'{v:16s} {f[0]:7.3f}/{f[1]:.4f}/{f[2]:.4f}/{f[3]:.4f}   {g[0]:7.3f}/{g[1]:.4f}/{g[2]:.4f}/{g[3]:.4f}')
