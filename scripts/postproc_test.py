"""Render once at 4K, then evaluate several post-processing variants for LPIPS gain.
LPIPS is 1/3 of the official rank and our weakest metric (0.2205 vs shengqi 0.1819)."""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2, torch.nn.functional as F
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
scene, main_run, extra = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
case = Path(f'/home/intern_2603055/vvc/data/{scene}')
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in [main_run]+extra]
centers = np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad=float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt += ((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
def unsharp(x, amt, rad_px):
    k = int(rad_px)*2+1
    blur = F.avg_pool2d(F.pad(x, (k//2,)*4, mode='reflect'), k, stride=1)
    return (x + amt*(x-blur)).clamp(0,1)
variants = {'raw': lambda x: x}
for amt,r in [(0.4,4),(0.4,8),(0.6,4),(0.6,8),(0.8,4)]:
    variants[f'unsharp a{amt} r{r}'] = (lambda a,rr: (lambda x: unsharp(x,a,rr)))(amt,r)
res={k:[] for k in variants}
with torch.inference_mode():
    for name,cam in te.items():
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        for i in range(0,len(frames),200):
            gt=cv2.imread(str(frames[i]))[...,::-1]
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W)); img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            base=(img/len(mv)).permute(0,3,1,2)
            for k,fn in variants.items():
                p=fn(base)
                res[k].append((pf(p,gtt).item(), sf(p,gtt).item(), lf(p,gtt).item()))
for k in variants:
    m=np.mean(res[k],0); print(f'{k:18s} psnr={m[0]:.3f} ssim={m[1]:.4f} lpips={m[2]:.4f}')
