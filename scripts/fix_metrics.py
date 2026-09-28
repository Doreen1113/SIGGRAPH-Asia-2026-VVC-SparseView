"""FULL + FOREGROUND metrics for a dumped val set: compares <tag>_<variant>.png against <tag>_gt.png.
Usage: ftg.sh fix_metrics.py <dump_dir> <variant1> [variant2 ...]   (variant 'render' = raw)"""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d=Path(sys.argv[1]); variants=sys.argv[2:]; man=json.load(open(d/'manifest.json'))
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[...,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
res={v:{'full':[],'fg':[],'perview':{}} for v in variants}
with torch.inference_mode():
    for e in man:
        gt=load(d/f"{e['tag']}_gt.png")
        person=(seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.3)
        ys,xs=torch.where(person); box=(ys.min().item(),ys.max().item()+1,xs.min().item(),xs.max().item()+1) if person.sum()>200 else None
        m=person[None,None].float()
        for v in variants:
            p=d/f"{e['tag']}_{v}.png"
            if not p.exists(): continue
            pr=load(p)
            if pr.shape!=gt.shape: pr=torch.nn.functional.interpolate(pr,size=gt.shape[-2:],mode='bilinear',align_corners=False)
            full=(pf(pr,gt).item(),sf(pr,gt).item(),lf(pr,gt).item()); res[v]['full'].append(full)
            res[v]['perview'].setdefault(e['view'],[]).append(full[0])
            if box:
                y0,y1,x0,x1=box; pm=(pr*m)[:,:,y0:y1,x0:x1]; gm=(gt*m)[:,:,y0:y1,x0:x1]
                res[v]['fg'].append((pf(pm,gm).item(),sf(pm,gm).item(),lf(pm,gm).item()))
for v in variants:
    if not res[v]['full']: print(f'{v:14s} (no files)'); continue
    f=np.mean(res[v]['full'],0); g=np.mean(res[v]['fg'],0) if res[v]['fg'] else [float('nan')]*3
    pv=' '.join(f"{k}:{np.mean(x):.1f}" for k,x in sorted(res[v]['perview'].items()))
    print(f'{v:14s} FULL {f[0]:.3f}/{f[1]:.4f}/{f[2]:.4f}   FG {g[0]:.3f}/{g[1]:.4f}/{g[2]:.4f}   [{pv}]')
