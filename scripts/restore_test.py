"""Learned detail restoration for LPIPS. Our 4K renders lack high-frequency detail (LPIPS 0.22 vs rival 0.18).
Test: downscale render -> learned 4x SR back to 4K, replacing our blurry detail with a natural-image prior.
Compliant: external general-purpose pretrained model (same class as VGGT/SEVA/DeepLabV3 already in use)."""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2, torch.nn.functional as F
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from transformers import Swin2SRForImageSuperResolution
scene, main_run, extra = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
case = Path(f'/home/intern_2603055/vvc/data/{scene}')
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in [main_run]+extra]
centers=np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad=float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
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
sr = Swin2SRForImageSuperResolution.from_pretrained('/home/intern_2603055/vvc/models/swin2sr').cuda().eval().half()
def sr_tile(x, tile=384, ov=32):
    """x: (1,3,h,w) in [0,1] at 1/4 target size -> 4x SR, tiled to fit memory."""
    _,_,h,w = x.shape; out = torch.zeros((1,3,h*4,w*4), device='cuda')
    wsum = torch.zeros_like(out)
    for y0 in range(0, h, tile-ov):
        for x0 in range(0, w, tile-ov):
            y1,x1 = min(y0+tile,h), min(x0+tile,w)
            patch = x[:,:,y0:y1,x0:x1]
            ph, pw = patch.shape[-2:]
            pad_h = (-ph) % 8; pad_w = (-pw) % 8
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
            with torch.inference_mode(): o = sr(patch.half()).reconstruction.float().clamp(0,1)
            o = o[:, :, :ph*4, :pw*4]
            out[:,:,y0*4:y1*4, x0*4:x1*4] += o; wsum[:,:,y0*4:y1*4, x0*4:x1*4] += 1
    return out/wsum.clamp(min=1)
def unsharp(x, amt, r):
    k=r*2+1; blur=F.avg_pool2d(F.pad(x,(k//2,)*4,mode='reflect'),k,stride=1); return (x+amt*(x-blur)).clamp(0,1)
res={k:[] for k in ['raw','unsharp a0.4 r4','SR restore','SR blend 0.5','SR blend 0.3']}
with torch.inference_mode():
    for name,cam in list(te.items())[:4]:
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        for i in range(0,len(frames),400):
            gt=cv2.imread(str(frames[i]))[...,::-1]
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W)); img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            base=(img/len(mv)).permute(0,3,1,2)
            small = F.interpolate(base, size=(H//4, W//4), mode='area')
            srout = F.interpolate(sr_tile(small), size=(H,W), mode='bilinear', align_corners=False)
            P=lambda p: (pf(p,gtt).item(), sf(p,gtt).item(), lf(p,gtt).item())
            res['raw'].append(P(base))
            res['unsharp a0.4 r4'].append(P(unsharp(base,0.4,4)))
            res['SR restore'].append(P(srout))
            res['SR blend 0.5'].append(P((0.5*base+0.5*srout).clamp(0,1)))
            res['SR blend 0.3'].append(P((0.7*base+0.3*srout).clamp(0,1)))
for k,v in res.items():
    m=np.mean(v,0); print(f'{k:18s} psnr={m[0]:.3f} ssim={m[1]:.4f} lpips={m[2]:.4f}')
