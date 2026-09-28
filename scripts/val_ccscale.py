"""Is scale=1.0 leaving headroom? Sweep the colour-fix scale (0.5..1.5) for both forms, leave-one-out on the
rig-matched validation scenes, production-faithful base."""
import json, numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
V = '/work/doreen071/vvc'
SCN = {'012_0': ('{D}/ps_lev/{tag}_f101.png', 'taper', 0.10), '001_1': ('{D}/ps_cur/{tag}_c_x_tp15.png', 'gain', 0)}
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def jpeg(x):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
    b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)
    return torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def metrics(pr, gt):
    return {'psnr': float(10 * torch.log10(1 / ((pr - gt) ** 2).mean())), 'ssim': sf(pr, gt).item(), 'lpips': lf(pr, gt).item()}
with torch.inference_mode():
    for scn, (fmt, form, tau) in SCN.items():
        D = f'{V}/fixtest/{scn}'; man = json.load(open(f'{D}/manifest.json')); cache = {}
        for e in man:
            gt = load(f"{D}/{e['tag']}_gt.png"); r = load(fmt.format(D=D, tag=e['tag']))
            cache[e['tag']] = dict(view=e['view'], d_all=(gt - r).mean((2, 3))[0], g_all=gt.mean((2, 3))[0] / r.mean((2, 3))[0].clamp_min(1e-3))
        res = {}
        for e in man:
            t = e['tag']; v = e['view']; others = [c for k, c in cache.items() if c['view'] == v and k != t]
            if not others: continue
            avg = lambda key: torch.stack([o[key] for o in others]).mean(0).view(1, 3, 1, 1)
            gt = load(f'{D}/{t}_gt.png'); r = load(fmt.format(D=D, tag=t))
            dA, gA = avg('d_all'), avg('g_all'); lum = r.mean(1, keepdim=True)
            for s in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                if form == 'taper': x = r + s * dA * (lum / (lum + tau))
                else: x = r * (1 + s * (gA - 1))
                m = metrics(jpeg(x.clamp(0, 1)), gt)
                for k, val in m.items(): res.setdefault(s, {}).setdefault(k, []).append(val)
        print(f'== {scn} form={form}')
        for s, mm in sorted(res.items()):
            a = {k: np.mean(v) for k, v in mm.items()}
            print(f"  scale={s:.1f}  psnr={a['psnr']:.4f} ssim={a['ssim']:.4f} lpips={a['lpips']:.4f}")
print('VAL_CCSCALE_DONE')
