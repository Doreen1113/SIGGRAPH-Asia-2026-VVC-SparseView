"""Test-time idea: the studio background is static, so the per-pixel TEMPORAL MEDIAN of the background across
many frames (person masked out) is a denoised, floater-free background. Composite the per-frame person onto it.
Unlike Difix this hallucinates nothing, so it may raise SSIM and LPIPS together. Val 001_1, 6-member ensemble."""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0,'/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
scene='001_1_seq0'; case=Path(f'/home/intern_2603055/vvc/data/{scene}'); R=Path(f'/home/intern_2603055/vvc/runs/siga_{scene}')
runs=['run_FULLRES','run_P8','run_P9','run_J_dense01','run_GATEONLY','run_G4M']
tr=read_camera(case/'train_intri.yml',case/'train_extri.yml'); te=read_camera(case/'test_intri.yml',case/'test_extri.yml')
offs=json.loads((case/'t_offsets.json').read_text())
gs=[Gaussians.load(R/r/'gaussians.pt').cuda().eval() for r in runs]
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
centers=np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad=float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt+=((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]
def person_mask(img):  # (1,3,H,W) in [0,1]
    return (seg(((img-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.5)[None,None].float()
def soft(m,r=16): k=2*r+1; return F.avg_pool2d(F.pad(m,(r,)*4,mode='replicate'),k,stride=1).clamp(0,1)
res={'plain':{'f':[],'g':[]},'median_bg':{'f':[],'g':[]},'median_bg_blend50':{'f':[],'g':[]}}
NMED=24  # frames used for the temporal median
with torch.inference_mode():
    for vn,cam in te.items():
        frames=sorted((case/'images'/vn).glob('*.jpg')); K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(vn,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        def render(i):
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W)); img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            return (img/len(mv)).permute(0,3,1,2)
        # temporal median background from NMED frames spread over the sequence, person masked out
        stack=[]; masks=[]
        for i in np.linspace(0,len(frames)-1,NMED).astype(int):
            r=render(int(i)); m=person_mask(r); stack.append(r.half().cpu()); masks.append(m.half().cpu())
        S=torch.cat(stack,0); M=torch.cat(masks,0)                     # (N,3,H,W),(N,1,H,W)
        Sm=S.clone(); Sm[M.expand_as(S)>0.5]=float('nan')             # exclude person pixels
        med=torch.nanmedian(Sm,dim=0).values                            # (3,H,W); nan where always occluded
        med=torch.where(torch.isnan(med), torch.nanmean(S,dim=0), med)  # fallback
        med=med.float().cuda()[None]
        for i in range(0,len(frames),200):
            gt=cv2.imread(str(frames[i]))[...,::-1]; gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            pr=render(i); m=soft(person_mask(pr))
            outs={'plain':pr, 'median_bg': m*pr+(1-m)*med, 'median_bg_blend50': m*pr+(1-m)*(0.5*pr+0.5*med)}
            gm=person_mask(gtt)[0,0]>0.5
            if gm.sum()<100: continue
            ys,xs=torch.where(gm); y0,y1,x0,x1=ys.min().item(),ys.max().item()+1,xs.min().item(),xs.max().item()+1
            for k,o in outs.items():
                o=o.clamp(0,1); res[k]['f'].append((pf(o,gtt).item(),sf(o,gtt).item(),lf(o,gtt).item()))
                oc,gc=o[:,:,y0:y1,x0:x1],gtt[:,:,y0:y1,x0:x1]; res[k]['g'].append((pf(oc,gc).item(),sf(oc,gc).item(),lf(oc,gc).item()))
        print(f'view {vn} done', flush=True)
for k,r in res.items():
    f=np.mean(r['f'],0); g=np.mean(r['g'],0)
    print(f'{k:20s} FULL {f[0]:.3f}/{f[1]:.4f}/{f[2]:.4f}   FG {g[0]:.3f}/{g[1]:.4f}/{g[2]:.4f}')
