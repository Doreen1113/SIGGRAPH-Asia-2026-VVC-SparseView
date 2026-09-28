"""Does a better person mask buy FG metrics?

The evaluator's foreground is derived from the GROUND-TRUTH frame; ours is one DeepLab pass over the render.
Wherever the two disagree, the evaluator scores person pixels that we filled from the background source (and
vice versa) — that hurts FG-PSNR, FG-SSIM and FG-LPIPS at once, and dilation alone was already shown to trade
one against another. So instead of growing the mask, make it more ACCURATE: average the segmentation
probability over flips, scales and over both the raw person render and the Difix output.

Measures, on both validation scenes: (a) IoU of each mask against the GT-derived mask, and (b) the FG metrics
after recompositing person-over-background with that mask.
"""
import json, numpy as np, torch, torch.nn.functional as F, cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
V = '/work/doreen071/vvc'
SCN = {'012_0': ('{D}/ps_lev/{tag}_tta.png', '{D}/ps_lev/{tag}_bg.png'),
       '001_1': ('{D}/ps_cur/{tag}_tta.png', '{D}/ps_cur/{tag}_bg_x_tp15.png')}
sf = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lf = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
MU = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); SD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def load(p):
    a = cv2.imread(p)
    return None if a is None else torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def prob(x):                                    # person-class probability map
    with torch.inference_mode():
        return seg(((x - MU) / SD).half())['out'].float().softmax(1)[:, 15:16]
def prob_tta(x, scales=(0.75, 1.0, 1.25), flip=True):
    H, W = x.shape[-2:]; acc = 0.0; n = 0
    for s in scales:
        xs = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False) if s != 1.0 else x
        p = F.interpolate(prob(xs), size=(H, W), mode='bilinear', align_corners=False); acc = acc + p; n += 1
        if flip:
            p = F.interpolate(prob(torch.flip(xs, [-1])), size=(H, W), mode='bilinear', align_corners=False)
            acc = acc + torch.flip(p, [-1]); n += 1
    return acc / n
def soft(m, k=101):
    p = k // 2; return F.avg_pool2d(F.pad(m, (p,) * 4, mode='replicate'), k, stride=1)
def jpeg(x):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
    b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], 1)
    return torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
def fg(pr, gt, gm):
    ys, xs = torch.where(gm[0, 0] > 0.5)
    if len(ys) < 100: return None
    y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
    pc, gc = pr[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1]
    return {'fpsnr': float(10 * torch.log10(1 / ((pc - gc) ** 2).mean())), 'fssim': sf(pc, gc).item(),
            'flpips': lf(pc.clamp(0, 1), gc.clamp(0, 1)).item(),
            'psnr': float(10 * torch.log10(1 / ((pr - gt) ** 2).mean())), 'ssim': sf(pr, gt).item()}
with torch.inference_mode():
    for scn, (pfmt, bfmt) in SCN.items():
        D = f'{V}/fixtest/{scn}'; man = json.load(open(f'{D}/manifest.json'))
        res = {}; iou = {}
        for e in man:
            P = load(pfmt.format(D=D, tag=e['tag'])); B = load(bfmt.format(D=D, tag=e['tag']))
            gt = load(f"{D}/{e['tag']}_gt.png")
            if P is None or B is None or gt is None: continue
            gm = (prob(gt) > 0.5).float()
            C0 = 0.85 * P + 0.15 * P        # placeholder to keep person source explicit
            MASKS = {'current(single pass on person)': (prob(P) > 0.5).float(),
                     'tta(flip+scales on person)': (prob_tta(P) > 0.5).float(),
                     'tta+union with bg-composite': ((prob_tta(P) + prob_tta(0.85 * P + 0.15 * B)) / 2 > 0.5).float(),
                     'tta thresh 0.4 (slightly inclusive)': (prob_tta(P) > 0.4).float()}
            for name, m in MASKS.items():
                inter = (m * gm).sum().item(); union = ((m + gm) > 0).float().sum().item()
                iou.setdefault(name, []).append(inter / max(union, 1))
                s = soft(m)
                out = jpeg((s * P + (1 - s) * B).clamp(0, 1))
                r = fg(out, gt, gm)
                if r:
                    for k, v in r.items(): res.setdefault(name, {}).setdefault(k, []).append(v)
        print(f'== {scn}')
        base = {k: np.mean(v) for k, v in res['current(single pass on person)'].items()}
        for name in res:
            a = {k: np.mean(v) for k, v in res[name].items()}
            print(f"  {name:36s} IoU={np.mean(iou[name]):.4f}  " +
                  ' '.join(f"{k}={a[k]:.4f}({a[k]-base[k]:+.4f})" for k in ['fpsnr', 'fssim', 'flpips', 'psnr', 'ssim']))
print('VAL_MASKQUALITY_DONE')
