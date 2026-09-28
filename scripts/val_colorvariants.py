"""Which colour-correction form keeps SSIM/LPIPS while getting the PSNR gain?  Leave-one-frame-out per camera on
the two released validation scenes (012_0 = 011 rig, 001_1 = 007 rig). Offsets are fitted on the OTHER frames of the
same camera (test-time analogue: same rig, unseen frames). Every variant goes through a q95 JPEG round-trip.
Metrics: full PSNR/SSIM/LPIPS(alex) and FG (person bbox crop from GT DeepLab mask, as fg_eval_official.py)."""
import json, sys, numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
V = '/work/doreen071/vvc'
SCN = {'012_0': '{D}/ps_lev/{tag}_f25.png', '001_1': '{D}/ps_cur/{tag}_c_xl15.png'}
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mu = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); sd = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def pmask(x): return (seg(((x - mu) / sd).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()
def soft(m, k=101): p = k // 2; return F.avg_pool2d(F.pad(m, (p,) * 4, mode='replicate'), k, stride=1)
def jpeg(x):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
    b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)
    return torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def metrics(pr, gt, gm):
    out = {'psnr': float(10 * torch.log10(1 / ((pr - gt) ** 2).mean())), 'ssim': sf(pr, gt).item(), 'lpips': lf(pr, gt).item()}
    ys, xs = torch.where(gm[0, 0] > 0.5)
    if len(ys) < 100: return out
    y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
    pc, gc = pr[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1]
    out.update(fpsnr=float(10 * torch.log10(1 / ((pc - gc) ** 2).mean())), fssim=sf(pc, gc).item(), flpips=lf(pc, gc).item())
    return out
allres = {}
with torch.inference_mode():
    for scn, fmt in SCN.items():
        D = f'{V}/fixtest/{scn}'; man = json.load(open(f'{D}/manifest.json'))
        cache = {}
        for e in man:   # per-frame statistics (image-level mean diff, person-region mean diff, ratio)
            gt = load(f"{D}/{e['tag']}_gt.png"); r = load(fmt.format(D=D, tag=e['tag']))
            gm = pmask(gt); rm = pmask(r)
            d_all = (gt - r).mean((2, 3))[0]
            d_fg = ((gt - r) * rm).sum((2, 3))[0] / rm.sum().clamp_min(1)
            d_bg = ((gt - r) * (1 - rm)).sum((2, 3))[0] / (1 - rm).sum().clamp_min(1)
            g_all = gt.mean((2, 3))[0] / r.mean((2, 3))[0].clamp_min(1e-3)
            cache[e['tag']] = dict(view=e['view'], d_all=d_all, d_fg=d_fg, d_bg=d_bg, g_all=g_all, fgfrac=rm.mean().item())
        res = {}
        for e in man:
            t = e['tag']; v = e['view']; others = [c for k, c in cache.items() if c['view'] == v and k != t]
            if not others: continue
            avg = lambda key: torch.stack([o[key] for o in others]).mean(0).view(1, 3, 1, 1)
            gt = load(f'{D}/{t}_gt.png'); r = load(fmt.format(D=D, tag=t)); gm = pmask(gt)
            sm = soft(pmask(r))
            dA, dF, dB, gA = avg('d_all'), avg('d_fg'), avg('d_bg'), avg('g_all')
            VAR = {'base': r, 'add_full': r + dA, 'add_half': r + 0.5 * dA, 'add_fgonly': r + sm * dA, 'add_bgonly': r + (1 - sm) * dA,
                   'fgprior_fg+bgprior_bg': r + sm * dF + (1 - sm) * dB, 'fgprior_fgonly': r + sm * dF, 'gain_full': r * gA,
                   'add_full_lowpass': r + dA}   # identical to add_full; control for jpeg noise
            for name, x in VAR.items():
                m = metrics(jpeg(x.clamp(0, 1)), gt, gm)
                for k, val in m.items(): res.setdefault(name, {}).setdefault(k, []).append(val)
        print(f'== {scn}  (n={len(man)}, render person-mask frac ~{np.mean([c["fgfrac"] for c in cache.values()]):.3f})')
        b = {k: np.mean(vs) for k, vs in res['base'].items()}
        for name, mm in res.items():
            a = {k: np.mean(vs) for k, vs in mm.items()}
            print(f"  {name:24s} " + ' '.join(f"{k}={a[k]:.4f}({a[k]-b[k]:+.4f})" for k in ['psnr', 'ssim', 'lpips', 'fpsnr', 'fssim', 'flpips'] if k in a))
        print('  per-view offsets (d_all vs d_fg, 0-255):')
        for v in sorted({c['view'] for c in cache.values()}):
            cs = [c for c in cache.values() if c['view'] == v]
            print(f"    {v}: all {np.round(torch.stack([c['d_all'] for c in cs]).mean(0).cpu().numpy()*255,1)}  fg {np.round(torch.stack([c['d_fg'] for c in cs]).mean(0).cpu().numpy()*255,1)}  bg {np.round(torch.stack([c['d_bg'] for c in cs]).mean(0).cpu().numpy()*255,1)}")
print('VAL_COLORVARIANTS_DONE')
