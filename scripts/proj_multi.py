"""Perceptual projection with a configurable LPIPS backbone mix (robust to which LPIPS the organizers use).
loss = sum_i w_i * LPIPS_i(x, difix) + lam * (1 - SSIM(x, raw)), x initialised at init_alpha*difix + (1-init_alpha)*raw.
Dump mode : proj_multi.py dump <dump_dir> <difix_variant> <out_variant> [opts]   (writes <tag>_<out_variant>.png)
Tree mode : proj_multi.py tree <raw_root> <fix_root> <dst_root> [opts]           (resumable, jpg q95)
opts: --nets alex:1,vgg:1  --lam 10 --steps 200 --batch 4 --init_alpha 0.5 --cases a,b --shard i/n --limit N
"""
import sys, os, json, argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['dump', 'tree']); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('c')
ap.add_argument('--nets', default='alex:1'); ap.add_argument('--lam', type=float, default=10.0); ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--batch', type=int, default=4); ap.add_argument('--init_alpha', type=float, default=0.5); ap.add_argument('--cases', default='')
ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0)
o = ap.parse_args()
nets = []
for tok in o.nets.split(','):
    name, w = tok.split(':'); n = _NoTrainLpips(net=name).cuda().eval()
    for p in n.parameters(): p.requires_grad_(False)
    nets.append((n, float(w)))
g = cv2.getGaussianKernel(11, 1.5); w = torch.from_numpy((g @ g.T).astype(np.float32)).cuda()[None, None].repeat(3, 1, 1, 1)
def ssim(x, y):
    mu_x = F.conv2d(x, w, groups=3); mu_y = F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x*x, w, groups=3) - mu_x**2; syy = F.conv2d(y*y, w, groups=3) - mu_y**2; sxy = F.conv2d(x*y, w, groups=3) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (((2*mu_x*mu_y + c1)*(2*sxy + c2)) / ((mu_x**2 + mu_y**2 + c1)*(sxx + syy + c2))).mean(dim=(1, 2, 3))
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2, 0, 1).float()/255
def project(R, D):
    x = (o.init_alpha*D + (1-o.init_alpha)*R).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.005)
    for _ in range(o.steps):
        opt.zero_grad(set_to_none=True); xc = x.clamp(0, 1)
        lp = sum(wt*n(xc*2-1, D*2-1).view(-1) for n, wt in nets)
        (lp + o.lam*(1-ssim(xc, R))).sum().backward(); opt.step()
    return (x.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255).round().astype(np.uint8)
si, sn = [int(t) for t in o.shard.split('/')]
if o.mode == 'dump':
    d = o.a; man = json.load(open(f'{d}/manifest.json'))[si::sn]
    if o.limit: man = man[:o.limit]
    from itertools import groupby
    groups = [list(g) for _, g in groupby(sorted(man, key=lambda e: (e['H'], e['W'])), key=lambda e: (e['H'], e['W']))]
    batches = [g[i:i+o.batch] for g in groups for i in range(0, len(g), o.batch)]
    for es in batches:
        R = torch.stack([load(f"{d}/{e['tag']}_render.png") for e in es]).cuda(); D = torch.stack([load(f"{d}/{e['tag']}_{o.b}.png") for e in es]).cuda()
        for e, out in zip(es, project(R, D)): cv2.imwrite(f"{d}/{e['tag']}_{o.c}.png", out[..., ::-1]); print(e['tag'], flush=True)
    print('PROJ DONE', o.c)
else:
    raw, fix, dst = o.a, o.b, o.c; cases = o.cases.split(',') if o.cases else sorted(os.listdir(raw))
    jobs = [(c, v) for c in cases for v in sorted(os.listdir(f'{raw}/{c}'))][si::sn]; n = 0
    for c, v in jobs:
        os.makedirs(f'{dst}/{c}/{v}', exist_ok=True)
        files = [f for f in sorted(os.listdir(f'{raw}/{c}/{v}')) if not os.path.exists(f'{dst}/{c}/{v}/{f}')]
        for i in range(0, len(files), o.batch):
            b = files[i:i+o.batch]
            R = torch.stack([load(f'{raw}/{c}/{v}/{f}') for f in b]).cuda(); D = torch.stack([load(f'{fix}/{c}/{v}/{f}') for f in b]).cuda()
            for f, out in zip(b, project(R, D)): cv2.imwrite(f'{dst}/{c}/{v}/{f}', out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
            print(f'{n} done', flush=True)
    print(f'PROJ TREE DONE shard {si}/{sn}: {n}')
