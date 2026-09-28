"""Person/background composite from two different sources.
  out = sm*(pa*A + (1-pa)*raw) + (1-sm)*(ba*B + (1-ba)*raw), sm = soft DeepLabV3 person mask on the RAW render.
Dump mode : composite2.py dump <dump_dir> <A_variant> <pa> <B_variant> <ba> <out_variant>
Tree mode : composite2.py tree <raw_root> <A_root> <pa> <B_root> <ba> <dst_root> [--cases a,b]  (resumable, jpg q95)"""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['dump', 'tree']); ap.add_argument('args', nargs=6); ap.add_argument('--cases', default='')
ap.add_argument('--dilate', type=int, default=0, help='max-pool the person mask by this radius before feathering; our DeepLab person mask covers 0.0526 of the frame while the evaluator foreground is 0.0596, so the outer ring of its FG region never receives the person source')
ap.add_argument('--q', type=int, default=95, help='final JPEG quality; measured on val (single-compression): lowering q RAISES SSIM and lowers LPIPS quality, and our FULL-LPIPS has far more slack than our FULL-SSIM does')
ap.add_argument('--feather', type=int, default=25, help='avg_pool2d kernel width for the soft mask blur; default 25 was never swept')
o = ap.parse_args()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
@torch.inference_mode()
def comp(raw, A, pa, B, ba):
    pm = (seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
    if o.dilate:
        k = 2*o.dilate + 1
        pm = F.max_pool2d(pm, k, stride=1, padding=o.dilate)
    pad = o.feather // 2
    sm = F.avg_pool2d(F.pad(pm, (pad,)*4, mode='replicate'), o.feather, stride=1).clamp(0, 1)
    out = (sm*(pa*A+(1-pa)*raw) + (1-sm)*(ba*B+(1-ba)*raw)).clamp(0, 1)
    return (out[0].cpu().numpy()*255).round().astype(np.uint8).transpose(1,2,0)[..., ::-1]
if o.mode == 'dump':
    d, va, pa, vb, ba, vo = o.args; pa, ba = float(pa), float(ba)
    for e in json.load(open(f'{d}/manifest.json')):
        t = e['tag']; out = comp(load(f'{d}/{t}_render.png'), load(f'{d}/{t}_{va}.png'), pa, load(f'{d}/{t}_{vb}.png'), ba)
        cv2.imwrite(f'{d}/{t}_{vo}.png', out)
    print('COMPOSITE DONE', vo)
else:
    raw, A, pa, B, ba, dst = o.args; pa, ba = float(pa), float(ba); n = 0
    cases = o.cases.split(',') if o.cases else sorted(os.listdir(raw))
    for c in cases:
        for v in sorted(os.listdir(f'{raw}/{c}')):
            os.makedirs(f'{dst}/{c}/{v}', exist_ok=True)
            for f in sorted(os.listdir(f'{raw}/{c}/{v}')):
                op = f'{dst}/{c}/{v}/{f}'
                if os.path.exists(op): continue
                cv2.imwrite(op, comp(load(f'{raw}/{c}/{v}/{f}'), load(f'{A}/{c}/{v}/{f}'), pa, load(f'{B}/{c}/{v}/{f}'), ba), [cv2.IMWRITE_JPEG_QUALITY, o.q]); n += 1
                if n % 100 == 0: print(f'{n} done', flush=True)
    print(f'COMPOSITE TREE DONE {n}')
