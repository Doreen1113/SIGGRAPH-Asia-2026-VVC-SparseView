"""Dump 4K (render, GT, nearest-train reference) triplets for val test views, for post-fix experiments (Difix etc.).
Usage: ftg.sh dump_val_renders.py <scene> <main_run> <extra_runs,comma> <out_dir> [--every 200]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/work/doreen071/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
ap=argparse.ArgumentParser(); ap.add_argument('scene'); ap.add_argument('main'); ap.add_argument('extra'); ap.add_argument('out'); ap.add_argument('--every',type=int,default=200)
a=ap.parse_args(); case=Path('/work/doreen071/vvc/data')/a.scene; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
tr=read_camera(case/'train_intri.yml',case/'train_extri.yml'); te=read_camera(case/'test_intri.yml',case/'test_extri.yml'); offs=json.loads((case/'t_offsets.json').read_text())
gs=[Gaussians.load(Path(r)/'gaussians.pt').cuda().eval() for r in [a.main]+[x for x in a.extra.split(',') if x.strip()]]
trc={k:-c.w2c[:3,:3].T@c.w2c[:3,3] for k,c in tr.items()}; centers=np.stack(list(trc.values())); rad=float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
def tcount(g):
    cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
    for tc in tr.values():
        K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
        c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
        cnt+=((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
    return cnt
cnts=[tcount(g) for g in gs]; manifest=[]
with torch.inference_mode():
    for name,cam in te.items():
        frames=sorted((case/'images'/name).glob('*.jpg')); K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
        w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
        ccn=-cam.w2c[:3,:3].T@cam.w2c[:3,3]; cc=torch.tensor(ccn,dtype=torch.float32).cuda(); off=float(offs.get(name,0.0))
        ref_cam=min(trc,key=lambda k:np.linalg.norm(trc[k]-ccn))
        mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
        for i in range(0,len(frames),a.every):
            img=None
            for g2 in mv:
                im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W)); img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
            arr=((img/len(mv))[0].cpu().numpy()*255).round().astype(np.uint8)
            tag=f'{name}_{i:06d}'
            cv2.imwrite(str(out/f'{tag}_render.png'),arr[...,::-1]); 
            cv2.imwrite(str(out/f'{tag}_gt.png'),cv2.imread(str(frames[i])))
            reff=sorted((case/'images'/ref_cam).glob('*.jpg'))[min(i,len(frames)-1)]
            cv2.imwrite(str(out/f'{tag}_ref.png'),cv2.imread(str(reff)))
            manifest.append({'tag':tag,'view':name,'frame':i,'ref_cam':ref_cam,'W':W,'H':H})
json.dump(manifest,open(out/'manifest.json','w'),indent=1); print('dumped',len(manifest),'triplets ->',out)
