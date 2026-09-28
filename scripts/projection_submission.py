"""Production perceptual-projection pass: per image, x is optimised to be LPIPS-close to an EXISTING Difix output
and lambda*(1-SSIM)-close to the raw render, starting from a light uniform blend. Batched across images per view
for speed (small LPIPS/SSIM nets, unlike the heavy diffusion Difix pass this reuses rather than recomputes).
Validated: beats the single-shift Difix frontier at matched SSIM by 0.014-0.018 LPIPS on both val gate scenes.
  in : <raw>/<case>/<view>/<frame:06d>.jpg  <fix>/<case>/<view>/<frame:06d>.jpg (existing Difix output)
  out: <dst>/<case>/<view>/<frame:06d>.jpg
Usage: python projection_submission.py <raw_root> <fix_root> <dst_root> [--lam 10] [--steps 200] [--batch 8]"""
import sys, os, argparse, glob
import numpy as np, cv2, torch, torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('fix'); ap.add_argument('dst')
ap.add_argument('--lam', type=float, default=10.0); ap.add_argument('--steps', type=int, default=200); ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--cases', default=''); ap.add_argument('--init_alpha', type=float, default=0.5)
a = ap.parse_args()
net = _NoTrainLpips(net='alex').cuda().eval()
for p in net.parameters(): p.requires_grad_(False)
g = cv2.getGaussianKernel(11, 1.5); w = torch.from_numpy((g @ g.T).astype(np.float32)).cuda()[None, None].repeat(3, 1, 1, 1)
def ssim(x, y):  # per-sample mean SSIM, x,y: (B,3,H,W)
    mu_x = F.conv2d(x, w, groups=3, padding=0); mu_y = F.conv2d(y, w, groups=3, padding=0)
    sxx = F.conv2d(x*x, w, groups=3, padding=0) - mu_x**2; syy = F.conv2d(y*y, w, groups=3, padding=0) - mu_y**2
    sxy = F.conv2d(x*y, w, groups=3, padding=0) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    m = ((2*mu_x*mu_y + c1)*(2*sxy + c2)) / ((mu_x**2 + mu_y**2 + c1)*(sxx + syy + c2))
    return m.mean(dim=(1, 2, 3))
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2, 0, 1).float()/255
cases = a.cases.split(',') if a.cases else sorted(os.listdir(a.raw))
n = 0
for c in cases:
    for v in sorted(os.listdir(f'{a.raw}/{c}')):
        os.makedirs(f'{a.dst}/{c}/{v}', exist_ok=True)
        files = [f for f in sorted(os.listdir(f'{a.raw}/{c}/{v}')) if not os.path.exists(f'{a.dst}/{c}/{v}/{f}')]
        for i in range(0, len(files), a.batch):
            batch = files[i:i+a.batch]
            R = torch.stack([load(f'{a.raw}/{c}/{v}/{f}') for f in batch]).cuda()
            D = torch.stack([load(f'{a.fix}/{c}/{v}/{f}') for f in batch]).cuda()
            x = (a.init_alpha*D + (1-a.init_alpha)*R).clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=0.005)
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                xc = x.clamp(0, 1)
                loss = net(xc*2-1, D*2-1).squeeze(-1).squeeze(-1).squeeze(-1) + a.lam*(1-ssim(xc, R))
                loss.sum().backward(); opt.step()
            out = (x.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
            for f, o in zip(batch, out):
                cv2.imwrite(f'{a.dst}/{c}/{v}/{f}', o[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
            print(f'{n} done', flush=True)
print(f'PROJECTION SUBMISSION DONE {n} images')
