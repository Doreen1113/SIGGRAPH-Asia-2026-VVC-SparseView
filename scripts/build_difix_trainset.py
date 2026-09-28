"""Build a paired dataset to FINE-TUNE Difix on OUR OWN artifacts.
Organizers explicitly permit using this track's training views to fine-tune a generative prior.
Pairs: our 6-member ensemble render (input) -> real ground truth (target), with the nearest
training-camera frame as the reference image (the same conditioning used at inference).
Uses the two val scenes, whose held-out views have ground truth."""
import sys, json, os
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0,'/home/intern_2603055/vvc/repo/baseline_code')
from ftgspp.data.utils.easy_utils import read_camera
from ftgspp.models.gaussians import Gaussians
OUT=Path('/home/intern_2603055/vvc/difix_ft'); OUT.mkdir(parents=True, exist_ok=True)
(OUT/'input').mkdir(exist_ok=True); (OUT/'target').mkdir(exist_ok=True); (OUT/'ref').mkdir(exist_ok=True)
SCENES={'001_1_seq0':['run_FULLRES','run_P8','run_P9','run_J_dense01','run_GATEONLY','run_G4M'],
        '012_0_seq0':['run_S03','run_P8','run_P9','run_J','run_GATEONLY','run_G3M']}
EVERY=int(os.environ.get('FT_EVERY','20'))   # sample density
data={'train':{},'test':{}}
for scene,runs in SCENES.items():
    case=Path(f'/home/intern_2603055/vvc/data/{scene}'); R=Path(f'/home/intern_2603055/vvc/runs/siga_{scene}')
    tr=read_camera(case/'train_intri.yml',case/'train_extri.yml'); te=read_camera(case/'test_intri.yml',case/'test_extri.yml')
    offs=json.loads((case/'t_offsets.json').read_text())
    gs=[Gaussians.load(R/r/'gaussians.pt').cuda().eval() for r in runs]
    trc={k:-c.w2c[:3,:3].T@c.w2c[:3,3] for k,c in tr.items()}
    centers=np.stack(list(trc.values())); rad=float(np.linalg.norm(centers-centers.mean(0),axis=1).mean())
    def tcount(g):
        cnt=torch.zeros(len(g),device='cuda',dtype=torch.int32)
        for tc in tr.values():
            K=torch.tensor(tc.K,dtype=torch.float32).cuda(); W=2*K[0,2]; H=2*K[1,2]; w2=torch.tensor(tc.w2c,dtype=torch.float32).cuda()
            c3=w2[:3,:3]@g.means.T+w2[:3,3:4]; z=c3[2]; u=K[0,0]*c3[0]/z+K[0,2]; v=K[1,1]*c3[1]/z+K[1,2]
            cnt+=((z>0.05)&(u>=0)&(u<W)&(v>=0)&(v<H)).int()
        return cnt
    cnts=[tcount(g) for g in gs]
    with torch.inference_mode():
        for vn,cam in te.items():
            frames=sorted((case/'images'/vn).glob('*.jpg'))
            K=cam.K.copy(); W=int(round(2*K[0,2])); H=int(round(2*K[1,2]))
            w2c=torch.tensor(cam.w2c,dtype=torch.float32).cuda()[None]; Kt=torch.tensor(K,dtype=torch.float32).cuda()[None]
            ccn=-cam.w2c[:3,:3].T@cam.w2c[:3,3]; cc=torch.tensor(ccn,dtype=torch.float32).cuda(); off=float(offs.get(vn,0.0))
            ref_cam=min(trc,key=lambda k:np.linalg.norm(trc[k]-ccn)); refs=sorted((case/'images'/ref_cam).glob('*.jpg'))
            mv=[g.mask(~((torch.norm(g.means-cc,dim=1)<1.1*rad)&(c<4))) for g,c in zip(gs,cnts)]
            for i in range(0,len(frames),EVERY):
                img=None
                for g2 in mv:
                    im2,_,_=g2(t=torch.tensor(i/60.0-off).cuda(),w2c=w2c,intrinsic=Kt,shape=(H,W))
                    img=im2.clamp(0,1) if img is None else img+im2.clamp(0,1)
                arr=((img/len(mv))[0].cpu().numpy()*255).round().astype(np.uint8)
                tag=f'{scene}_{vn}_{i:06d}'
                cv2.imwrite(str(OUT/'input'/f'{tag}.png'), arr[...,::-1])
                cv2.imwrite(str(OUT/'target'/f'{tag}.png'), cv2.imread(str(frames[i])))
                cv2.imwrite(str(OUT/'ref'/f'{tag}.png'), cv2.imread(str(refs[min(i,len(refs)-1)])))
                split='test' if (i//EVERY)%12==0 else 'train'
                data[split][tag]={'image':str(OUT/'input'/f'{tag}.png'),
                                  'target_image':str(OUT/'target'/f'{tag}.png'),
                                  'ref_image':str(OUT/'ref'/f'{tag}.png'),
                                  'prompt':'remove degradation'}
    print(f'{scene} done', flush=True)
json.dump(data, open(OUT/'dataset.json','w'), indent=1)
print(f'train={len(data["train"])} test={len(data["test"])} -> {OUT}/dataset.json')
