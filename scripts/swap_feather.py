"""Change the person-mask feather width of an already-composited image without the person source tree.
Composite (ba=1): C = s*P + (1-s)*B, s = blur_k(pm). Target: C' = s'*P + (1-s')*B = C + (s'-s)*(P-B).
P is recovered by inversion P = B + (C-B)/s where s is large enough; where s is small the inversion amplifies JPEG
noise, so P blends toward a fallback F (weight w = clamp((s-t0)/(t1-t0), 0, 1)). pm is re-derived from C.
Dump mode: swap_feather.py dump <dir> <C_var> <B_var> <F_var|B> <out_var> --k0 25 --k1 81
Tree mode: swap_feather.py tree <C.zip> <B_root> <F_root|B> <dst_root> --k0 25 --k1 81 --shard i/n"""
import os, json, zipfile, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['dump', 'tree']); ap.add_argument('args', nargs='+')
ap.add_argument('--k0', type=int, default=25); ap.add_argument('--k1', type=int, default=81)
ap.add_argument('--t0', type=float, default=0.15); ap.add_argument('--t1', type=float, default=0.5)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--q', type=int, default=95)
ap.add_argument('--sigma', type=float, default=0.0, help='normalized-convolution smoothing of the inversion: (P-B) ~ G*((C-B)*s) / G*(s*s)')
o = ap.parse_args()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); std_ = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def t(a): return torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def u(x): return (x[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)[..., ::-1]
def blur(pm, k): return F.avg_pool2d(F.pad(pm, (k // 2,) * 4, mode='replicate'), k, stride=1).clamp(0, 1)
@torch.inference_mode()
def run(C, B, Fb):
    pm = (seg(((C - mean_) / std_).half())['out'].float().softmax(1)[0, 15] > 0.5)[None, None].float()
    s0, s1 = blur(pm, o.k0), blur(pm, o.k1)
    if o.sigma > 0:
        r = int(3 * o.sigma) | 1; ks = 2 * r + 1
        g1 = torch.exp(-torch.arange(-r, r + 1, device='cuda').float() ** 2 / (2 * o.sigma ** 2)); g1 = g1 / g1.sum()
        def G(x):
            c = x.shape[1]
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), g1.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
            return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), g1.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
        Pinv = B + G((C - B) * s0) / G(s0 * s0).clamp_min(1e-4)
    else:
        Pinv = B + (C - B) / s0.clamp_min(1e-3)
    w = ((s0 - o.t0) / (o.t1 - o.t0)).clamp(0, 1)
    P = (w * Pinv + (1 - w) * Fb).clamp(0, 1)
    return C + (s1 - s0) * (P - B)
if o.mode == 'dump':
    d, cv, bv, fv, ov = o.args
    for e in json.load(open(f'{d}/manifest.json')):
        g = e['tag']; C = t(cv2.imread(f'{d}/{g}_{cv}.png')); B = t(cv2.imread(f'{d}/{g}_{bv}.png'))
        Fb = B if fv == 'B' else t(cv2.imread(f'{d}/{g}_{fv}.png'))
        cv2.imwrite(f'{d}/{g}_{ov}.png', u(run(C, B, Fb)))
    print('SWAP_FEATHER DUMP DONE', ov)
else:
    zp, broot, froot, dst = o.args; si, sn = map(int, o.shard.split('/'))
    z = zipfile.ZipFile(zp); names = sorted(n for n in z.namelist() if n.endswith('.jpg') and '/renders/' in n); n = 0
    for k, name in enumerate(names):
        if k % sn != si: continue
        rel = name.split('/renders/', 1)[1]; out = f'{dst}/{rel}'
        if os.path.exists(out): continue
        C = t(cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)); B = t(cv2.imread(f'{broot}/{rel}'))
        Fb = B if froot == 'B' else t(cv2.imread(f'{froot}/{rel}'))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, u(run(C, B, Fb)), [cv2.IMWRITE_JPEG_QUALITY, o.q]); n += 1
    print(f'SWAP_FEATHER TREE DONE shard {si}/{sn}: {n}')
