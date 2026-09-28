"""Search person/background decoupled Difix compositing.
For each (person_src, person_alpha, bg_src, bg_alpha): out = mask*(blend on person) + (1-mask)*(blend on bg)
Mask comes from DeepLabV3 on the RENDER (legal at test time), softened. Reports FULL and FG (bbox-crop protocol).
Usage: ftg.sh decoupled_search.py <dump_dir>"""
import sys, json, itertools
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
def soften(m, r=12):
    k = 2*r+1; return F.avg_pool2d(F.pad(m, (r,)*4, mode='replicate'), k, stride=1).clamp(0,1)
# (label, person_src, person_alpha, bg_src, bg_alpha)
CFGS = [
    ('CUR t100 a.30 uniform',        't100', 0.30, 't100', 0.30),
    ('P t100 1.0 | bg t199 .40',     't100', 1.00, 't199', 0.40),
    ('P t100 1.0 | bg t199 .45',     't100', 1.00, 't199', 0.45),
    ('P t100 1.0 | bg t199 .50',     't100', 1.00, 't199', 0.50),
    ('P t100 1.0 | bg t199 .55',     't100', 1.00, 't199', 0.55),
    ('P t100 .60 | bg t199 .45',     't100', 0.60, 't199', 0.45),
    ('P t100 .60 | bg t199 .50',     't100', 0.60, 't199', 0.50),
    ('P t100 1.0 | bg mix45',        't100', 1.00, 'mix45', 1.00),
]
res = {c[0]: {'full': [], 'fg': []} for c in CFGS}
with torch.inference_mode():
    for e in man:
        gt = load(d/f"{e['tag']}_gt.png"); raw = load(d/f"{e['tag']}_render.png")
        t199i = load(d/f"{e['tag']}_difix.png"); t100i = load(d/f"{e['tag']}_difix_t100.png")
        src = {'raw': raw, 't199': t199i, 't100': t100i, 'mix45': 0.45*t199i+0.55*raw}
        pm = (seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
        sm = soften(pm)
        gm = (seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
        if gm.sum() < 100: continue
        ys, xs = torch.where(gm[0,0] > 0.5); y0,y1,x0,x1 = ys.min().item(), ys.max().item()+1, xs.min().item(), xs.max().item()+1
        for lbl, ps, pa, bs, ba in CFGS:
            person = pa*src[ps] + (1-pa)*raw
            bg     = ba*src[bs] + (1-ba)*raw
            out = (sm*person + (1-sm)*bg).clamp(0,1)
            res[lbl]['full'].append((pf(out,gt).item(), sf(out,gt).item(), lf(out,gt).item()))
            oc, gc = out[:,:,y0:y1,x0:x1], gt[:,:,y0:y1,x0:x1]
            res[lbl]['fg'].append((pf(oc,gc).item(), sf(oc,gc).item(), lf(oc,gc).item()))
print(f'--- {d.name}')
for lbl, *_ in CFGS:
    f = np.mean(res[lbl]['full'], 0); g = np.mean(res[lbl]['fg'], 0)
    print(f'  {lbl:30s} FULL {f[0]:.3f}/{f[1]:.4f}/{f[2]:.4f}   FG {g[0]:.3f}/{g[1]:.4f}/{g[2]:.4f}')
