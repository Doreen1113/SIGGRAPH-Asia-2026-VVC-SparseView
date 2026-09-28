"""Protect the whole PERSON BBOX from Difix (FG metrics only measure inside it), apply strong Difix outside.
alpha field: 0 inside dilated bbox -> a_bg outside, smooth transition. Reports FULL and FG (calibrated protocol).
Usage: ftg.sh bbox_protect_search.py <dump_dir>"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d = Path(sys.argv[1]); man = json.load(open(d/'manifest.json'))
pf = PeakSignalNoiseRatio(data_range=1.0).cuda(); sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
# (label, bbox margin px, alpha inside bbox, alpha outside, source outside)
CFGS = [
    ('in.25 out.50', 0, 0.25, 0.50, 't199'),
    ('in.25 out.55', 0, 0.25, 0.55, 't199'),
    ('in.30 out.50', 0, 0.30, 0.50, 't199'),
    ('in.30 out.55', 0, 0.30, 0.55, 't199'),
    ('in.30 out.60', 0, 0.30, 0.60, 't199'),
    ('in.35 out.55', 0, 0.35, 0.55, 't199'),
    ('in.35 out.60', 0, 0.35, 0.60, 't199'),
    ('in.40 out.55', 0, 0.40, 0.55, 't199'),
    ('in.20 out.55', 0, 0.20, 0.55, 't199'),
]
res = {c[0]: {'full': [], 'fg': [], 'prot': []} for c in CFGS}
with torch.inference_mode():
    for e in man:
        gt = load(d/f"{e['tag']}_gt.png"); raw = load(d/f"{e['tag']}_render.png")
        t199 = load(d/f"{e['tag']}_difix.png"); t100 = load(d/f"{e['tag']}_difix_t100.png")
        src = {'t199': t199, 't100': t100}
        # bbox from the RENDER's person mask (legal at test time)
        pmr = (seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)
        gm = (seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
        if gm.sum() < 100 or pmr.sum() < 100: continue
        ys, xs = torch.where(pmr); H, W = raw.shape[-2:]
        gy, gx = torch.where(gm[0,0] > 0.5); y0g,y1g,x0g,x1g = gy.min().item(), gy.max().item()+1, gx.min().item(), gx.max().item()+1
        for lbl, marg, a_in, a_out, so in CFGS:
            if marg < 0:
                A = torch.full((1,1,H,W), a_in, device='cuda')
                out = (A*t100 + (1-A)*raw).clamp(0,1)
            else:
                by0 = max(0, ys.min().item()-marg); by1 = min(H, ys.max().item()+1+marg)
                bx0 = max(0, xs.min().item()-marg); bx1 = min(W, xs.max().item()+1+marg)
                A = torch.full((1,1,H,W), a_out, device='cuda'); A[:,:,by0:by1,bx0:bx1] = a_in
                A = F.avg_pool2d(F.pad(A,(32,)*4,mode='replicate'), 65, stride=1)
                inner = a_in*t100 + (1-a_in)*raw
                outer = a_out*src[so] + (1-a_out)*raw
                w = ((A - a_in)/(a_out - a_in + 1e-8)).clamp(0,1)
                out = ((1-w)*inner + w*outer).clamp(0,1)
                res[lbl]['prot'].append(float(((by1-by0)*(bx1-bx0))/(H*W)))
            res[lbl]['full'].append((pf(out,gt).item(), sf(out,gt).item(), lf(out,gt).item()))
            oc, gc = out[:,:,y0g:y1g,x0g:x1g], gt[:,:,y0g:y1g,x0g:x1g]
            res[lbl]['fg'].append((pf(oc,gc).item(), sf(oc,gc).item(), lf(oc,gc).item()))
print(f'--- {d.name}')
for lbl, *_ in CFGS:
    f = np.mean(res[lbl]['full'], 0); g = np.mean(res[lbl]['fg'], 0)
    pr = f" prot={np.mean(res[lbl]['prot']):.3f}" if res[lbl]['prot'] else ""
    print(f'  {lbl:30s} FULL {f[0]:.3f}/{f[1]:.4f}/{f[2]:.4f}   FG {g[0]:.3f}/{g[1]:.4f}/{g[2]:.4f}{pr}')
