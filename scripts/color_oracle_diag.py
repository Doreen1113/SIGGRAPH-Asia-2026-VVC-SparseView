"""Is the oracle per-view color transform (a) stable over time -> camera property, (b) predictable from the
nearest TRAIN camera's learned corrector (compliant: uses only this case's train views + calibration)?"""
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
gs = [Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in [main_run]+extra]
ids = [int(k) for k in tr.keys()]
sd = torch.load(Path(main_run)/'color_correctors.pt', map_location='cpu', weights_only=False).state_dict()
trM = {i:(sd[f'_correctors.{i}.mat']+torch.eye(3)).cuda() for i in ids}; trV = {i:sd[f'_correctors.{i}.vec'].cuda() for i in ids}
trC = {i:-c.w2c[:3,:3].T@c.w2c[:3,3] for i,c in zip(ids,tr.values())}; trD = {i:c.w2c[2,:3] for i,c in zip(ids,tr.values())}  # cam center, view dir
centers = np.stack(list(trC.values())); rad = float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt += ((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]; pf=PeakSignalNoiseRatio(data_range=1.0).cuda()
tot={'raw':[], 'nearest_pos':[], 'nearest_dir':[], 'idw_dir':[], 'oracle':[]}
with torch.inference_mode():
    for name,cam in te.items():
        frames=sorted((case/'images'/name).glob('*.jpg'))
        K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2])); sc=0.5
        Ws,Hs=int(round(W*sc)),int(round(H*sc)); K[0,:]*=Ws/W; K[1,:]*=Hs/H
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        ccn=-cam.w2c[:3,:3].T@cam.w2c[:3,3]; dirn=cam.w2c[2,:3]; cc=torch.tensor(ccn,dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        # predictors
        npos=min(ids,key=lambda i:np.linalg.norm(trC[i]-ccn)); ndir=max(ids,key=lambda i:float(np.dot(trD[i],dirn)))
        wts={i:max(float(np.dot(trD[i],dirn)),0)**4 for i in ids}; ws=sum(wts.values())+1e-9
        Midw=sum(trM[i]*wts[i] for i in ids)/ws; Vidw=sum(trV[i]*wts[i] for i in ids)/ws
        res={k:[] for k in tot}; sols=[]
        for i in range(0,len(frames),100):
            gt=cv2.imread(str(frames[i]))[...,::-1]; gt=cv2.resize(gt,(Ws,Hs),interpolation=cv2.INTER_AREA)
            gtt=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].cuda().float()/255
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(Hs,Ws)); img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            img=(img/len(mv))[0]
            P=lambda x: pf(x.clamp(0,1).permute(2,0,1)[None],gtt).item()
            res['raw'].append(P(img)); res['nearest_pos'].append(P(img@trM[npos]+trV[npos])); res['nearest_dir'].append(P(img@trM[ndir]+trV[ndir])); res['idw_dir'].append(P(img@Midw+Vidw))
            X=torch.cat([img.reshape(-1,3),torch.ones(img.numel()//3,1,device='cuda')],1); Y=gtt[0].permute(1,2,0).reshape(-1,3)
            sol=torch.linalg.lstsq(X,Y).solution; sols.append(sol.cpu()); res['oracle'].append(P((X@sol).reshape(img.shape)))
        S=torch.stack(sols); stab=(S-S.mean(0)).abs().max().item()
        print(f"  view {name}: raw {np.mean(res['raw']):.2f} | nearest-pos(cam{npos}) {np.mean(res['nearest_pos']):.2f} | nearest-dir(cam{ndir}) {np.mean(res['nearest_dir']):.2f} | idw-dir {np.mean(res['idw_dir']):.2f} | oracle {np.mean(res['oracle']):.2f}   oracle diag={S.mean(0)[:3].diag().numpy().round(3)} bias={S.mean(0)[3].numpy().round(3)} temporal-var={stab:.3f}")
        for k in tot: tot[k].append(np.mean(res[k]))
print(f"{scene} MEAN: " + ' | '.join(f'{k} {np.mean(v):.3f}' for k,v in tot.items()))
