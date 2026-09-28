"""Temporally averaged background plate, restricted to pixels the performer NEVER occupies.

C27 tried the same idea and gained FULL-SSIM +0.0057 (ten times the threshold we need) while losing 3 dB of
FG-PSNR. The loss came from the plate being built with one person mask and then composited under a different
one: frames where the two disagreed let averaged-in person residue show through around the silhouette.

Here the plate is only allowed to replace background where the UNION of every frame's person mask (generously
dilated) says the performer never was. Inside that union the original per-frame background is kept untouched,
so nothing the evaluator crops as foreground can be affected, while the far background — which is what
FULL-SSIM and FULL-LPIPS are actually made of — still gets the noise-free plate.

Usage: temporal_bg_safe.py <bg_root> <person_root> <out_root> --case C [--views v1,v2] [--guard 201]
"""
import os, argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

ap = argparse.ArgumentParser()
ap.add_argument('bg'); ap.add_argument('person'); ap.add_argument('out')
ap.add_argument('--case', required=True); ap.add_argument('--views', default='')
ap.add_argument('--guard', type=int, default=201, help='dilation of the never-visited union; must comfortably exceed the composite feather (101) so the blend band never touches plate pixels')
ap.add_argument('--q', type=int, default=95)
o = ap.parse_args()

seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
MU = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
SD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

def person(bgr):
    x = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
    with torch.inference_mode():
        return (seg(((x - MU) / SD).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()

views = [v for v in o.views.split(',') if v] or sorted(os.listdir(f'{o.bg}/{o.case}'))
for v in views:
    frames = sorted(os.listdir(f'{o.bg}/{o.case}/{v}'))
    acc = None; cnt = None; union = None
    for f in frames:
        b = cv2.imread(f'{o.bg}/{o.case}/{v}/{f}')
        p = cv2.imread(f'{o.person}/{o.case}/{v}/{f}')
        if b is None or p is None: continue
        m = person(p)
        union = m if union is None else torch.maximum(union, m)
        keep = (1.0 - m)[0, 0].cpu().numpy()[..., None]
        if acc is None:
            acc = np.zeros(b.shape, np.float32); cnt = np.zeros(b.shape[:2] + (1,), np.float32)
        acc += b.astype(np.float32) * keep; cnt += keep
    if acc is None:
        print(f'{o.case}/{v}: no frames'); continue
    plate = acc / np.maximum(cnt, 1e-6)
    g = o.guard
    guard = F.max_pool2d(union, g, stride=1, padding=g // 2)[0, 0].cpu().numpy() > 0.5   # anywhere the person ever was, plus a margin
    safe = (~guard) & (cnt[..., 0] > 0.5)
    os.makedirs(f'{o.out}/{o.case}/{v}', exist_ok=True)
    n = 0
    for f in frames:
        b = cv2.imread(f'{o.bg}/{o.case}/{v}/{f}')
        if b is None: continue
        out = b.astype(np.float32).copy()
        out[safe] = plate[safe]
        cv2.imwrite(f'{o.out}/{o.case}/{v}/{f}', np.clip(out, 0, 255).round().astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, o.q]); n += 1
    print(f'{o.case}/{v}: {n} frames, plate applied to {safe.mean()*100:.1f}% of pixels', flush=True)
print('TEMPORAL_BG_SAFE_DONE', o.case)
