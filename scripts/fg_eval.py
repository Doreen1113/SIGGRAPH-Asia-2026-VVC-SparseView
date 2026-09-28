"""Foreground-only PSNR/SSIM/LPIPS on val, using DeepLabV3 person segmentation as the foreground mask.
Organizers announced (2026-09-05) that foreground metrics will be weighted equally with full-image metrics."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser()
ap.add_argument('main_run'); ap.add_argument('scene'); ap.add_argument('--extra_runs', default='')
ap.add_argument('--weights', default=''); ap.add_argument('--every', type=int, default=200); ap.add_argument('--cull_near_frac', type=float, default=1.1); ap.add_argument('--cull_min_views', type=int, default=4)
a = ap.parse_args()
case = Path(f'/home/intern_2603055/vvc/data/{a.scene}')
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
runs = [a.main_run] + ([r for r in a.extra_runs.split(',') if r] if a.extra_runs else [])
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in runs]
centers = np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad = float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
cull_near = a.cull_near_frac * rad
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
w = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1; seg = deeplabv3_resnet101(weights=w).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
res_full=[]; res_fg=[]
with torch.inference_mode():
    for name,cam in te.items():
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<cull_near)&(c<a.cull_min_views))) for g,c in zip(gs,cnts)]
        fr,fg_=[],[]
        wts=[float(x) for x in a.weights.split(',')] if a.weights else [1.0]*len(mv)
        wsum=sum(wts)
        for i in range(0,len(frames),a.every):
            gt=cv2.imread(str(frames[i]))[...,::-1]
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2,wt in zip(mv,wts):
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W))
                c=im2.clamp(0,1)*(wt/wsum); img=c if img is None else img+c
            pred=img.permute(0,3,1,2)
            fr.append((pf(pred,gtt).item(), sf(pred,gtt).item(), lf(pred,gtt).item()))
            xg = ((gtt - mean_) / std_).half()
            person = (seg(xg)['out'].float().softmax(1)[0,15] > 0.3)
            if person.sum() < 200:
                continue
            ys,xs = torch.where(person); y0,y1,x0,x1 = ys.min().item(),ys.max().item()+1,xs.min().item(),xs.max().item()+1
            m = person[None,None].float()
            pm = pred*m; gm = gtt*m
            fg_.append((pf(pm[:,:,y0:y1,x0:x1],gm[:,:,y0:y1,x0:x1]).item(),
                        sf(pm[:,:,y0:y1,x0:x1],gm[:,:,y0:y1,x0:x1]).item(),
                        lf(pred[:,:,y0:y1,x0:x1]*m[:,:,y0:y1,x0:x1], gtt[:,:,y0:y1,x0:x1]*m[:,:,y0:y1,x0:x1]).item()))
        res_full.append(np.mean(fr,0)); res_fg.append(np.mean(fg_,0) if fg_ else np.array([np.nan]*3))
mf=np.nanmean(res_full,0); mg=np.nanmean(res_fg,0)
print(f'FULL: psnr={mf[0]:.3f} ssim={mf[1]:.4f} lpips={mf[2]:.4f}')
print(f'FG  : psnr={mg[0]:.3f} ssim={mg[1]:.4f} lpips={mg[2]:.4f}')
