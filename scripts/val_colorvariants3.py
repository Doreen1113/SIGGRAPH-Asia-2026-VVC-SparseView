"""FG-ONLY forms on the production-faithful base. Only FG-PSNR (+0.294) and FG-SSIM (+0.0070) can still win us a
rank; a whole-frame correction cannot (FULL-PSNR needs +1.25) and it threatens our rank-1 FULL-LPIPS. So correct
the person region only, and leave the background — which drives FULL-LPIPS — untouched.
Production-faithful colour-correction comparison. Base variants now match what we actually ship
(012_0: feather-101 composite; 001_1: composite over the tp15 projection background), which the first
run did not. Adds brightness-tapered and multiplicative forms, because an additive offset penalises the
SSIM luminance term hardest in dark windows (mean 10 + 8 is a 80% change; mean 150 + 8 is 5%).
Offsets fitted leave-one-frame-out per camera; every variant goes through a q95 JPEG round-trip.
Also reports the SSIM delta split by window brightness, to confirm the mechanism."""
import json, numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
V = '/work/doreen071/vvc'
SCN = {'012_0': '{D}/ps_lev/{tag}_f101.png', '001_1': '{D}/ps_cur/{tag}_c_x_tp15.png'}
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mu = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); sd = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def pmask(x): return (seg(((x - mu) / sd).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()
def jpeg(x):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
    b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)
    return torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def ssim_map(a, b):                                  # per-pixel SSIM, 11x11 gaussian, matching the usual definition
    a = a.mean(1, keepdim=True); b = b.mean(1, keepdim=True)
    k = cv2.getGaussianKernel(11, 1.5); k = torch.tensor((k @ k.T), dtype=torch.float32).cuda()[None, None]
    f = lambda x: F.conv2d(x, k, padding=5)
    ma, mb = f(a), f(b); saa = f(a * a) - ma * ma; sbb = f(b * b) - mb * mb; sab = f(a * b) - ma * mb
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma * ma + mb * mb + C1) * (saa + sbb + C2)), mb
def metrics(pr, gt, gm):
    out = {'psnr': float(10 * torch.log10(1 / ((pr - gt) ** 2).mean())), 'ssim': sf(pr, gt).item(), 'lpips': lf(pr, gt).item()}
    ys, xs = torch.where(gm[0, 0] > 0.5)
    if len(ys) < 100: return out
    y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
    pc, gc = pr[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1]
    out.update(fpsnr=float(10 * torch.log10(1 / ((pc - gc) ** 2).mean())), fssim=sf(pc, gc).item(), flpips=lf(pc, gc).item())
    return out
BINS = [(0, .1), (.1, .25), (.25, .5), (.5, 1.01)]
with torch.inference_mode():
    for scn, fmt in SCN.items():
        D = f'{V}/fixtest/{scn}'; man = json.load(open(f'{D}/manifest.json')); cache = {}
        for e in man:
            gt = load(f"{D}/{e['tag']}_gt.png"); r = load(fmt.format(D=D, tag=e['tag'])); rm = pmask(r)
            cache[e['tag']] = dict(view=e['view'], d_all=(gt - r).mean((2, 3))[0],
                                   d_fg=((gt - r) * rm).sum((2, 3))[0] / rm.sum().clamp_min(1),
                                   d_bg=((gt - r) * (1 - rm)).sum((2, 3))[0] / (1 - rm).sum().clamp_min(1),
                                   g_all=gt.mean((2, 3))[0] / r.mean((2, 3))[0].clamp_min(1e-3))
        res = {}; binacc = {}
        for e in man:
            t = e['tag']; v = e['view']; others = [c for k, c in cache.items() if c['view'] == v and k != t]
            if not others: continue
            avg = lambda key: torch.stack([o[key] for o in others]).mean(0).view(1, 3, 1, 1)
            gt = load(f'{D}/{t}_gt.png'); r = load(fmt.format(D=D, tag=t)); gm = pmask(gt)
            dA, gA = avg('d_all'), avg('g_all')
            lum = r.mean(1, keepdim=True)
            dF, dB = avg('d_fg'), avg('d_bg')
            sm = F.avg_pool2d(F.pad(pmask(r), (50,) * 4, mode='replicate'), 101, stride=1)
            VAR = {'base': r, 'add_full': r + dA,
                   'fgonly_dall': r + sm * dA, 'fgonly_dfg': r + sm * dF,
                   'fgonly_dfg_taper': r + sm * dF * (lum / (lum + 0.10)),
                   'fgonly_dfg_half': r + 0.5 * sm * dF,
                   'fgonly_gain': r * (1 + sm * (gA - 1)),
                   'fg_dfg_bg_dbg': r + sm * dF + (1 - sm) * dB}
            for name, x in VAR.items():
                y = jpeg(x.clamp(0, 1)); m = metrics(y, gt, gm)
                for k, val in m.items(): res.setdefault(name, {}).setdefault(k, []).append(val)
                if name != 'base':
                    sm_new, mb = ssim_map(y, gt); sm_old, _ = ssim_map(jpeg(r.clamp(0, 1)), gt)
                    d = sm_new - sm_old
                    for lo, hi in BINS:
                        sel = (mb >= lo) & (mb < hi)
                        if sel.sum() > 0: binacc.setdefault(name, {}).setdefault((lo, hi), []).append(float(d[sel].mean()))
        print(f'== {scn} (production-faithful base)')
        b = {k: np.mean(vs) for k, vs in res['base'].items()}
        for name, mm in res.items():
            a = {k: np.mean(vs) for k, vs in mm.items()}
            print(f"  {name:22s} " + ' '.join(f"{k}={a[k]:.4f}({a[k]-b[k]:+.4f})" for k in ['psnr', 'ssim', 'lpips', 'fpsnr', 'fssim', 'flpips'] if k in a))
        print('  SSIM delta by window brightness (dark -> bright):')
        for name, bb in binacc.items():
            print(f"    {name:22s} " + '  '.join(f"[{lo:.2f}-{hi:.2f}] {np.mean(v):+.4f}" for (lo, hi), v in sorted(bb.items())))
print('VAL_COLORVARIANTS3_DONE')
