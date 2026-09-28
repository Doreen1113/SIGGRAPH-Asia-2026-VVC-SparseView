"""Buy FG-SSIM with FG-LPIPS: unsharp masking inside the person mask only.

FG-SSIM is the other rank we can still win (needs +0.0046) and colour correction does nothing for it. SSIM
rewards local structure, and Difix TTA leaves the person slightly soft. Sharpening was dropped from the
pipeline long ago because it cost LPIPS — but FG-LPIPS is now rank 1 with 0.0045 of headroom over the next
team, so a small trade is affordable if the SSIM gain is real. This measures the exchange rate on both
rig-matched validation scenes, production-faithful bases, person mask only, q95 JPEG round-trip.
"""
import json, numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
V = '/work/doreen071/vvc'
SCN = {'012_0': '{D}/ps_lev/{tag}_f101.png', '001_1': '{D}/ps_cur/{tag}_c_x_tp15.png'}
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
MU = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); SD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def pmask(x):
    with torch.inference_mode(): return (seg(((x - MU) / SD).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()
def soft(m, k=101):
    p = k // 2; return F.avg_pool2d(F.pad(m, (p,) * 4, mode='replicate'), k, stride=1)
def gauss(x, sigma):
    r = max(1, int(3 * sigma)); k = torch.arange(-r, r + 1, device='cuda', dtype=torch.float32)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2)); k = (k / k.sum())
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
def jpeg(x):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
    b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)
    return torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def fg_metrics(pr, gt, gm):
    ys, xs = torch.where(gm[0, 0] > 0.5)
    if len(ys) < 100: return None
    y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
    pc, gc = pr[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1]
    return {'fpsnr': float(10 * torch.log10(1 / ((pc - gc) ** 2).mean())), 'fssim': sf(pc, gc).item(),
            'flpips': lf(pc.clamp(0, 1), gc.clamp(0, 1)).item(),
            'psnr': float(10 * torch.log10(1 / ((pr - gt) ** 2).mean())), 'ssim': sf(pr, gt).item(),
            'lpips': lf(pr.clamp(0, 1), gt.clamp(0, 1)).item()}
AMTS = [0.0, -0.15, -0.3, -0.5, -0.8]
SIGS = [1.0, 2.0]
with torch.inference_mode():
    for scn, fmt in SCN.items():
        D = f'{V}/fixtest/{scn}'; man = json.load(open(f'{D}/manifest.json')); res = {}
        for e in man:
            gt = load(f"{D}/{e['tag']}_gt.png"); r = load(fmt.format(D=D, tag=e['tag']))
            gm = pmask(gt); sm = soft(pmask(r))
            for sig in SIGS:
                blur = gauss(r, sig)
                for a in AMTS:
                    x = r + a * sm * (r - blur)
                    m = fg_metrics(jpeg(x.clamp(0, 1)), gt, gm)
                    if m is None: continue
                    for k, val in m.items(): res.setdefault((sig, a), {}).setdefault(k, []).append(val)
        print(f'== {scn}')
        b = {k: np.mean(v) for k, v in res[(SIGS[0], 0.0)].items()}
        for (sig, a), mm in sorted(res.items()):
            m = {k: np.mean(v) for k, v in mm.items()}
            print(f"  sigma={sig:.1f} amount={a:.2f}  FG ssim={m['fssim']:.4f}({m['fssim']-b['fssim']:+.4f}) "
                  f"lpips={m['flpips']:.4f}({m['flpips']-b['flpips']:+.4f}) psnr={m['fpsnr']:.3f}({m['fpsnr']-b['fpsnr']:+.3f}) | "
                  f"FULL ssim={m['ssim']:.4f}({m['ssim']-b['ssim']:+.4f}) lpips={m['lpips']:.4f}({m['lpips']-b['lpips']:+.4f})")
print('VAL_FGSHARPEN_DONE')
