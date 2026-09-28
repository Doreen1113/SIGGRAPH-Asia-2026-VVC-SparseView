"""Perceptual projection: per image, optimise x to be LPIPS-close to the Difix output and SSIM-close to the raw
render. LPIPS and SSIM measure different things, so the optimum is not a linear blend of the two.
Usage: ftg.sh perceptual_projection.py <dump> <difix_variant> <init_variant> <lambda> <steps> <out_name>"""
import sys, json
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
d = Path(sys.argv[1]); v = sys.argv[2]; init = sys.argv[3]; lam = float(sys.argv[4]); steps = int(sys.argv[5]); name = sys.argv[6]
man = json.load(open(d/'manifest.json'))
net = _NoTrainLpips(net='alex').cuda().eval()
for p in net.parameters(): p.requires_grad_(False)
g = cv2.getGaussianKernel(11, 1.5); w = torch.from_numpy((g @ g.T).astype(np.float32)).cuda()[None, None].repeat(3, 1, 1, 1)
def ssim(x, y):
    mu_x = F.conv2d(x, w, groups=3); mu_y = F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x*x, w, groups=3) - mu_x**2; syy = F.conv2d(y*y, w, groups=3) - mu_y**2; sxy = F.conv2d(x*y, w, groups=3) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (((2*mu_x*mu_y + c1)*(2*sxy + c2)) / ((mu_x**2 + mu_y**2 + c1)*(sxx + syy + c2))).mean()
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda()/255
for e in man:
    R = load(d/f"{e['tag']}_render.png"); D = load(d/f"{e['tag']}_{v}.png"); x = load(d/f"{e['tag']}_{init}.png").clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.005)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        xc = x.clamp(0, 1)
        loss = net(xc*2-1, D*2-1).mean() + lam*(1-ssim(xc, R))
        loss.backward(); opt.step()
    out = (x.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    cv2.imwrite(str(d/f"{e['tag']}_{name}.png"), out[..., ::-1]); print(e['tag'], round(float(loss), 4), flush=True)
print("PROJ DONE", name)
