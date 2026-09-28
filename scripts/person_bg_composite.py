"""Composite with SEPARATE Difix strength for the person mask and the background.
  out = soft_mask*(pa*difix + (1-pa)*raw) + (1-soft_mask)*(ba*difix + (1-ba)*raw)
Person mask from DeepLabV3 on the RAW render (test-time legal). Resumable, JPEG q95.
Usage: ftg.sh person_bg_composite.py <raw_root> <difix_root> <out_root> --pa 0.5 --ba 0.45 [--cases a,b]"""
import sys, os, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap=argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('fix'); ap.add_argument('out')
ap.add_argument('--pa',type=float,required=True); ap.add_argument('--ba',type=float,required=True); ap.add_argument('--cases',default='')
a=ap.parse_args()
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(p)[...,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
cases=a.cases.split(',') if a.cases else sorted(os.listdir(a.raw)); n=0
with torch.inference_mode():
    for c in cases:
        for v in sorted(os.listdir(f'{a.raw}/{c}')):
            os.makedirs(f'{a.out}/{c}/{v}',exist_ok=True)
            for f in sorted(os.listdir(f'{a.raw}/{c}/{v}')):
                op=f'{a.out}/{c}/{v}/{f}'
                if os.path.exists(op): continue
                raw=load(f'{a.raw}/{c}/{v}/{f}'); fx=load(f'{a.fix}/{c}/{v}/{f}')
                pm=(seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.5)[None,None].float()
                sm=F.avg_pool2d(F.pad(pm,(12,)*4,mode='replicate'),25,stride=1).clamp(0,1)
                person=a.pa*fx+(1-a.pa)*raw; bg=a.ba*fx+(1-a.ba)*raw
                out=(sm*person+(1-sm)*bg).clamp(0,1)
                arr=(out[0].cpu().numpy()*255).round().astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(op,arr[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95]); n+=1
                if n%200==0: print(f'{n} done',flush=True)
print(f'composited {n} images (person={a.pa} bg={a.ba})')
