"""Does applying the MEAN train-camera color corrector to hidden views help? Measured vs GT on val.
Also computes the per-view ORACLE affine fit as an upper bound."""
import sys, json
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
from torchmetrics.image import PeakSignalNoiseRatio
scene, main_run, extra = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
case = Path(f'/home/intern_2603055/vvc/data/{scene}')
tr = read_camera(case/'train_intri.yml', case/'train_extri.yml'); te = read_camera(case/'test_intri.yml', case/'test_extri.yml')
offs = json.loads((case/'t_offsets.json').read_text())
runs = [main_run] + extra
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in runs]
# mean corrector from the MAIN run (train cams only)
ids = [int(k) for k in tr.keys()]
sd = torch.load(Path(main_run)/'color_correctors.pt', map_location='cpu', weights_only=False).state_dict()
M = torch.stack([sd[f'_correctors.{i}.mat'] for i in ids]).mean(0).cuda() + torch.eye(3).cuda()
V = torch.stack([sd[f'_correctors.{i}.vec'] for i in ids]).mean(0).cuda()
print('mean corrector diag', M.diag().cpu().numpy().round(4), 'vec', V.cpu().numpy().round(4))
centers = np.stack([-c.w2c[:3,:3].T@c.w2c[:3,3] for c in tr.values()]); rad = float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt += ((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]
pf=PeakSignalNoiseRatio(data_range=1.0).cuda()
raw, mean_c, oracle = [], [], []
with torch.inference_mode():
    for name,cam in te.items():
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2])); sc=0.5
        Ws,Hs=int(round(W*sc)),int(round(H*sc)); K[0,:]*=Ws/W; K[1,:]*=Hs/H
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        cc=torch.tensor(-cam.w2c[:3,:3].T@cam.w2c[:3,3],dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        pr, pm, po = [], [], []
        for i in range(0,len(frames),100):
            gt=cv2.imread(str(frames[i]))[...,::-1]; gt=cv2.resize(gt,(Ws,Hs),interpolation=cv2.INTER_AREA)
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(Hs,Ws))
                img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            img=(img/len(mv))[0]                                   # (H,W,3)
            pr.append(pf(img.permute(2,0,1)[None].clamp(0,1),gtt).item())
            corr=(img@M+V).clamp(0,1); pm.append(pf(corr.permute(2,0,1)[None],gtt).item())
            # oracle affine fit (upper bound, uses GT — diagnostic only)
            X=torch.cat([img.reshape(-1,3),torch.ones(img.numel()//3,1,device='cuda')],1); Y=gtt[0].permute(1,2,0).reshape(-1,3)
            sol=torch.linalg.lstsq(X,Y).solution; orc=(X@sol).reshape(img.shape).clamp(0,1)
            po.append(pf(orc.permute(2,0,1)[None],gtt).item())
        raw.append(np.mean(pr)); mean_c.append(np.mean(pm)); oracle.append(np.mean(po))
        print(f'  view {name}: raw {np.mean(pr):.3f}  +mean-corrector {np.mean(pm):.3f}  oracle-affine {np.mean(po):.3f}')
print(f'{scene} MEAN: raw {np.mean(raw):.3f}  +mean-corrector {np.mean(mean_c):.3f}  oracle-affine {np.mean(oracle):.3f}')
