"""Directly optimise the OFFICIAL objective: Final Rank = (Rank_Full + Rank_FG)/2, with all live opponents.
Sweeps person-region and background-region Difix strength on the val dumps, converts val->test with the
calibrated offsets, and reports the rank each config would actually achieve."""
import json, itertools
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
OPP = {
 'shengqi': dict(fp=27.682, fs=0.9181, fl=0.1819, gp=25.551, gs=0.8394, gl=0.2950),
 'mmm':     dict(fp=24.979, fs=0.9111, fl=0.2789, gp=24.192, gs=0.8292, gl=0.2658),
 'mingzai': dict(fp=23.732, fs=0.9114, fl=0.2575, gp=23.035, gs=0.8248, gl=0.4194),
}
# val -> test calibration (validated to 3 decimals on restore_t100)
OFF = dict(fp=-0.070, fs=-0.0017, fl=-0.0025, gp=+0.110, gs=-0.034, gl=+0.066)
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[...,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
def soften(m,r=12):
    k=2*r+1; return F.avg_pool2d(F.pad(m,(r,)*4,mode='replicate'),k,stride=1).clamp(0,1)
PA=[0.0,0.2,0.3,0.5,0.7,1.0]      # person-region difix strength
BA=[0.3,0.45,0.55,0.7]            # background-region difix strength
res={(pa,ba):{'full':[],'fg':[]} for pa in PA for ba in BA}
for scene in ['001_1','012_0']:
    d=Path(f'/home/intern_2603055/vvc/fixtest/{scene}'); man=json.load(open(d/'manifest.json'))
    with torch.inference_mode():
        for e in man:
            gt=load(d/f"{e['tag']}_gt.png"); raw=load(d/f"{e['tag']}_render.png")
            t199=load(d/f"{e['tag']}_difix.png"); t100=load(d/f"{e['tag']}_difix_t100.png")
            pmr=(seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.5)[None,None].float(); sm=soften(pmr)
            gm=(seg(((gt-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.5)
            if gm.sum()<100: continue
            ys,xs=torch.where(gm); y0,y1,x0,x1=ys.min().item(),ys.max().item()+1,xs.min().item(),xs.max().item()+1
            for pa in PA:
                person = pa*t199 + (1-pa)*raw          # person uses the stronger fixer (best LPIPS reach)
                for ba in BA:
                    bg = ba*t199 + (1-ba)*raw
                    out=(sm*person+(1-sm)*bg).clamp(0,1)
                    res[(pa,ba)]['full'].append((pf(out,gt).item(),sf(out,gt).item(),lf(out,gt).item()))
                    oc,gc=out[:,:,y0:y1,x0:x1],gt[:,:,y0:y1,x0:x1]
                    res[(pa,ba)]['fg'].append((pf(oc,gc).item(),sf(oc,gc).item(),lf(oc,gc).item()))
def rank_of(v,others,lower):  return sorted(others+[v],reverse=not lower).index(v)+1
o=list(OPP.values()); rows=[]
for (pa,ba),r in res.items():
    f=np.mean(r['full'],0); g=np.mean(r['fg'],0)
    us=dict(fp=f[0]+OFF['fp'], fs=f[1]+OFF['fs'], fl=f[2]+OFF['fl'],
            gp=g[0]+OFF['gp'], gs=g[1]+OFF['gs'], gl=g[2]+OFF['gl'])
    rf=[rank_of(us['fp'],[x['fp'] for x in o],False),rank_of(us['fs'],[x['fs'] for x in o],False),rank_of(us['fl'],[x['fl'] for x in o],True)]
    rg=[rank_of(us['gp'],[x['gp'] for x in o],False),rank_of(us['gs'],[x['gs'] for x in o],False),rank_of(us['gl'],[x['gl'] for x in o],True)]
    ours=(sum(rf)/3+sum(rg)/3)/2
    s=OPP['shengqi']; oth=[us,OPP['mmm'],OPP['mingzai']]
    srf=[rank_of(s['fp'],[x['fp'] for x in oth],False),rank_of(s['fs'],[x['fs'] for x in oth],False),rank_of(s['fl'],[x['fl'] for x in oth],True)]
    srg=[rank_of(s['gp'],[x['gp'] for x in oth],False),rank_of(s['gs'],[x['gs'] for x in oth],False),rank_of(s['gl'],[x['gl'] for x in oth],True)]
    sh=(sum(srf)/3+sum(srg)/3)/2
    rows.append((ours,sh,pa,ba,us,rf,rg))
rows.sort(key=lambda x:(x[0],-x[1]))
print(f'{"person":>7s}{"bg":>6s}  FULL p/s/l (test)          FG p/s/l (test)           FULLrk    FGrk    OURS   SHENGQI')
for ours,sh,pa,ba,us,rf,rg in rows[:14]:
    tag = ' <-- WIN' if ours<sh-1e-9 else (' <-- tie' if abs(ours-sh)<1e-9 else '')
    print(f'{pa:>7.1f}{ba:>6.2f}  {us["fp"]:.3f}/{us["fs"]:.4f}/{us["fl"]:.4f}   {us["gp"]:.3f}/{us["gs"]:.4f}/{us["gl"]:.4f}   {rf} {rg}  {ours:.3f}  {sh:.3f}{tag}')
