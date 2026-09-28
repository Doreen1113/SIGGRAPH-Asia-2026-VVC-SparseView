"""Collapse the background of each hidden view to a single temporally-averaged plate.

The cameras are fixed and the scene is static apart from the performer, so every frame of a given view should
show the SAME background — yet we render it independently per frame, so each frame carries its own render
noise. Averaging over the frames of a view (excluding, per frame, the pixels the performer covers) removes
that noise almost entirely. Background is ~94% of the pixels and drives FULL-SSIM and FULL-LPIPS, which are
0.0005 and 0.0050 away from a rank each.

An earlier pass over this idea was dropped because PSNR stayed flat (the residual error is systematic floaters,
not per-frame noise) and SSIM only moved +0.005 — but +0.005 SSIM is ten times the current FULL-SSIM threshold,
so it is worth exactly what it was not worth then.

Usage: temporal_bg.py <bg_root> <person_root> <out_root> --case C --views v1,v2 [--blend 1.0]
  bg_root/person_root/out_root are <root>/<case>/<view>/<frame>.jpg trees.
  --blend w : out = w*plate + (1-w)*per-frame background (1.0 = pure plate)
"""
import os, sys, argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

ap = argparse.ArgumentParser()
ap.add_argument('bg'); ap.add_argument('person'); ap.add_argument('out')
ap.add_argument('--case', required=True); ap.add_argument('--views', default='')
ap.add_argument('--blend', type=float, default=1.0)
ap.add_argument('--dilate', type=int, default=61, help='grow the excluded person region before averaging, so its soft edge never leaks into the plate')
ap.add_argument('--q', type=int, default=95)
o = ap.parse_args()

seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
MU = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
SD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

def person_mask(bgr):
    x = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
    with torch.inference_mode():
        m = (seg(((x - MU) / SD).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()
    if o.dilate:
        k = o.dilate; m = F.max_pool2d(m, k, stride=1, padding=k // 2)
    return m[0, 0].cpu().numpy()

views = [v for v in o.views.split(',') if v] or sorted(os.listdir(f'{o.bg}/{o.case}'))
for v in views:
    frames = sorted(os.listdir(f'{o.bg}/{o.case}/{v}'))
    acc = None; cnt = None
    for f in frames:
        b = cv2.imread(f'{o.bg}/{o.case}/{v}/{f}')
        p = cv2.imread(f'{o.person}/{o.case}/{v}/{f}')
        if b is None or p is None: continue
        keep = (1.0 - person_mask(p))[..., None].astype(np.float32)   # 1 where background is visible
        if acc is None:
            acc = np.zeros(b.shape, np.float32); cnt = np.zeros(b.shape[:2] + (1,), np.float32)
        acc += b.astype(np.float32) * keep; cnt += keep
    if acc is None:
        print(f'{o.case}/{v}: no frames'); continue
    plate = acc / np.maximum(cnt, 1e-6)
    thin = (cnt[..., 0] < 0.5)          # pixels the performer covered in every frame: keep the per-frame render there
    n = 0
    os.makedirs(f'{o.out}/{o.case}/{v}', exist_ok=True)
    for f in frames:
        b = cv2.imread(f'{o.bg}/{o.case}/{v}/{f}')
        if b is None: continue
        out = o.blend * plate + (1 - o.blend) * b.astype(np.float32)
        out[thin] = b.astype(np.float32)[thin]
        cv2.imwrite(f'{o.out}/{o.case}/{v}/{f}', np.clip(out, 0, 255).round().astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, o.q]); n += 1
    print(f'{o.case}/{v}: {n} frames, never-visible pixels {thin.mean()*100:.2f}%', flush=True)
print('TEMPORAL_BG_DONE', o.case)
