"""Person-region-only per-camera colour correction.

Why FG-only: only FG-PSNR (needs +0.294) and FG-SSIM (+0.0070) can still win us a rank. A whole-frame
correction cannot (FULL-PSNR needs +1.25, and the two rig-matched scenes can supply at most +0.875) while it
does threaten our rank-1 FULL-LPIPS, which is 0.0007 ahead of shengqi and is driven by the background. So we
correct only inside the soft person mask and leave the background — and therefore FULL-LPIPS — untouched.

The offset is 3 numbers per camera: mean(GT - render) over the person mask, averaged over the frames of a
released validation scene that shares the physical rig with a test scene (012_0 -> 011_0, 001_1 -> 007_0).
No validation pixels enter the output.

  camcolor_fg.py fit <val_dump> <render_fmt> <out.json>
  camcolor_fg.py apply <src.zip> <dst_root> <case> <prior.json> <scale> [--shard i/n] [--taper TAU]

--taper scales the offset by lum/(lum+TAU) so dark pixels get less of it: the SSIM luminance term is relative,
so a +8 offset on a window whose mean is 10 is an 80% error while on a mean of 150 it is 5%. TAU=0 disables.
"""
import sys, os, json, zipfile
import numpy as np, cv2, torch, torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
MU = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
SD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

def to_t(bgr):   # HxWx3 uint8/float BGR -> 1x3xHxW RGB float in [0,1]
    return torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255

def soft_person(x, feather=101):
    with torch.inference_mode():
        m = (seg(((x - MU) / SD).half())['out'].float().softmax(1)[:, 15:16] > 0.5).float()
    p = feather // 2
    return F.avg_pool2d(F.pad(m, (p,) * 4, mode='replicate'), feather, stride=1)

mode = sys.argv[1]
if mode == 'fit':
    D, fmt, out = sys.argv[2:5]
    man = json.load(open(f'{D}/manifest.json')); acc = {}
    for e in man:
        gt = cv2.imread(f"{D}/{e['tag']}_gt.png").astype(np.float64)
        r = cv2.imread(fmt.format(D=D, tag=e['tag'])).astype(np.float64)
        m = (soft_person(to_t(r))[0, 0].cpu().numpy() > 0.5)
        if m.sum() < 1000: continue
        acc.setdefault(e['view'], []).append((gt - r)[m].mean(0))
    prior = {v: np.mean(x, 0).tolist() for v, x in acc.items()}
    json.dump(prior, open(out, 'w'), indent=1)
    print({v: np.round(p, 2).tolist() for v, p in prior.items()})
else:
    zp, dst, case, pj, scale = sys.argv[2:7]; scale = float(scale)
    args = sys.argv[7:]
    si, sn = (0, 1); tau = 0.0
    for i, a in enumerate(args):
        if a == '--shard': si, sn = map(int, args[i + 1].split('/'))
        if a == '--taper': tau = float(args[i + 1])
    prior = {v: np.array(p, np.float32) for v, p in json.load(open(pj)).items()}
    z = zipfile.ZipFile(zp); n = 0
    names = sorted(x for x in z.namelist() if x.endswith('.jpg') and f'/renders/{case}/' in x)
    for k, name in enumerate(names):
        if k % sn != si: continue
        rel = name.split('/renders/', 1)[1]; view = rel.split('/')[1]; out = f'{dst}/{rel}'
        if os.path.exists(out) or view not in prior: continue
        im = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        x = to_t(im)
        sm = soft_person(x)[0, 0].cpu().numpy()[..., None]
        d = scale * prior[view].reshape(1, 1, 3)
        if tau > 0:
            lum = im.astype(np.float32).mean(2, keepdims=True) / 255.0
            d = d * (lum / (lum + tau))
        outim = np.clip(im.astype(np.float32) + sm * d, 0, 255)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, outim.round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
    print(f'FG APPLY DONE {case} scale {scale} taper {tau}: {n}')
