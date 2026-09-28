"""Metrics for derived variants that need no new Difix runs: blends raw/difix and person-mask composites.
Usage: ftg.sh fix_variants_metrics.py <dump_dir>"""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d=Path(sys.argv[1]); man=json.load(open(d/'manifest.json'))
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[...,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
def soft(mask, r=24):
    k=2*r+1; m=torch.nn.functional.avg_pool2d(torch.nn.functional.pad(mask,(r,)*4,mode='replicate'),k,stride=1); return m.clamp(0,1)
variants={'raw':None,'difix':None,'blend0.3':0.3,'blend0.5':0.5,'blend0.7':0.7,'bg_only':'bg','fg_only':'fg','bg_only+blend0.5fg':'bgb'}
res={v:{'full':[],'fg':[]} for v in variants}
with torch.inference_mode():
    for e in man:
        gt=load(d/f"{e['tag']}_gt.png"); raw=load(d/f"{e['tag']}_render.png"); fx=load(d/f"{e['tag']}_difix.png")
        person=(seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.3)
        # for compositing use a mask computed on the RENDER (no GT at test time)
        pm_r=(seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.3)[None,None].float(); sm=soft(pm_r)
        ys,xs=torch.where(person); box=(ys.min().item(),ys.max().item()+1,xs.min().item(),xs.max().item()+1) if person.sum()>200 else None
        m=person[None,None].float()
        for v,spec in variants.items():
            if v=='raw': pr=raw
            elif v=='difix': pr=fx
            elif spec=='bg': pr=sm*raw+(1-sm)*fx
            elif spec=='fg': pr=sm*fx+(1-sm)*raw
            elif spec=='bgb': pr=sm*(0.5*raw+0.5*fx)+(1-sm)*fx
            else: pr=spec*fx+(1-spec)*raw
            res[v]['full'].append((pf(pr,gt).item(),sf(pr,gt).item(),lf(pr,gt).item()))
            if box:
                y0,y1,x0,x1=box; res[v]['fg'].append((pf((pr*m)[:,:,y0:y1,x0:x1],(gt*m)[:,:,y0:y1,x0:x1]).item(),sf((pr*m)[:,:,y0:y1,x0:x1],(gt*m)[:,:,y0:y1,x0:x1]).item(),lf((pr*m)[:,:,y0:y1,x0:x1],(gt*m)[:,:,y0:y1,x0:x1]).item()))
for v in variants:
    f=np.mean(res[v]['full'],0); g=np.mean(res[v]['fg'],0)
    print(f'{v:20s} FULL {f[0]:.3f}/{f[1]:.4f}/{f[2]:.4f}   FG {g[0]:.3f}/{g[1]:.4f}/{g[2]:.4f}')
