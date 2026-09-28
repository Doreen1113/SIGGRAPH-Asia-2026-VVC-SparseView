"""Production compositor: protect the person bbox (light t100 fix) and strongly fix outside it (t199).
  out = (1-w)*[a_in*t100 + (1-a_in)*raw]  +  w*[a_out*t199 + (1-a_out)*raw],  w = smoothed 0-inside-bbox / 1-outside
Person bbox comes from DeepLabV3 on the RAW render (test-time legal). Resumable. JPEG q95.
Usage: ftg.sh bbox_composite.py <raw_root> <t100_root> <t199_root> <out_root> [--a_in 0.30 --a_out 0.55 --cases a,b]"""
import sys, os, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('t100'); ap.add_argument('t199'); ap.add_argument('out')
ap.add_argument('--a_in', type=float, default=0.30); ap.add_argument('--a_out', type=float, default=0.55); ap.add_argument('--cases', default='')
a = ap.parse_args()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
cases = a.cases.split(',') if a.cases else sorted(os.listdir(a.raw))
n = 0; nofg = 0
with torch.inference_mode():
    for c in cases:
        for v in sorted(os.listdir(f'{a.raw}/{c}')):
            os.makedirs(f'{a.out}/{c}/{v}', exist_ok=True)
            for f in sorted(os.listdir(f'{a.raw}/{c}/{v}')):
                op = f'{a.out}/{c}/{v}/{f}'
                if os.path.exists(op): continue
                raw = load(f'{a.raw}/{c}/{v}/{f}'); t100 = load(f'{a.t100}/{c}/{v}/{f}'); t199 = load(f'{a.t199}/{c}/{v}/{f}')
                H, W = raw.shape[-2:]
                pm = (seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)
                inner = a.a_in*t100 + (1-a.a_in)*raw
                outer = a.a_out*t199 + (1-a.a_out)*raw
                if pm.sum() < 100:
                    out = outer; nofg += 1
                else:
                    ys, xs = torch.where(pm)
                    y0,y1,x0,x1 = ys.min().item(), ys.max().item()+1, xs.min().item(), xs.max().item()+1
                    w = torch.ones((1,1,H,W), device='cuda'); w[:,:,y0:y1,x0:x1] = 0.0
                    w = F.avg_pool2d(F.pad(w,(32,)*4,mode='replicate'), 65, stride=1).clamp(0,1)
                    out = (1-w)*inner + w*outer
                arr = (out[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(op, arr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
print(f'composited {n} images ({nofg} with no person detected -> outer only)')
