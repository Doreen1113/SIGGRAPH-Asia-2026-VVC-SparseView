"""Measure the REAL metric cost of JPEG quality: render -> encode at q -> decode -> compare against ground truth."""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
case = Path('/home/intern_2603055/vvc/data/001_1_seq0')
runs = ['/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_S_lpips0.15',
        '/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_P8',
        '/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_P9']
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in runs]
centers = np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad = float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt += ((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]
pf=PeakSignalNoiseRatio(data_range=1.0).cuda(); sf=StructuralSimilarityIndexMeasure(data_range=1.0).cuda(); lf=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
res={q:[] for q in [95,92,90,85]}
with torch.inference_mode():
    for name,cam in list(te.items()):
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2])); sc=0.5
        Ws,Hs=int(round(W*sc)),int(round(H*sc)); K[0,:]*=Ws/W; K[1,:]*=Hs/H
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        for i in range(0,len(frames),200):
            gt=cv2.imread(str(frames[i]))[...,::-1]; gt=cv2.resize(gt,(Ws,Hs),interpolation=cv2.INTER_AREA)
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(Hs,Ws))
                img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            arr=((img/len(mv))[0].cpu().numpy()*255).round().astype(np.uint8)
            for q in res:
                ok,buf=cv2.imencode('.jpg',arr[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,q])
                dec=cv2.imdecode(buf,cv2.IMREAD_COLOR)[...,::-1]
                p=torch.from_numpy(np.ascontiguousarray(dec)).permute(2,0,1)[None].cuda().float()/255
                res[q].append((pf(p,gtt).item(),sf(p,gtt).item(),lf(p,gtt).item(),len(buf)))
for q in sorted(res,reverse=True):
    m=np.mean(res[q],0)
    print(f'q={q}: vs GROUND TRUTH psnr={m[0]:.4f} ssim={m[2-1]:.5f} lpips={m[2]:.5f}   {m[3]/1024:.0f} KB/img -> zip {m[3]*2056/1e9:.2f} GB')
