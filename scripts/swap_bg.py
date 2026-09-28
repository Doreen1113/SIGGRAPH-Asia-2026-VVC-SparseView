"""Swap the background of an already-packed submission without rebuilding its person source.
A packed composite is C = sm*P + (1-sm)*B_old (background alpha 1.0), so
    C + (1-sm)*(B_new - B_old) = sm*P + (1-sm)*B_new
exactly, provided sm is the same mask. sm is re-derived from C with the same DeepLabV3 recipe composite2.py uses
(person class > 0.5, 25px feathering); C and the raw render share the person silhouette, and any mask mismatch
only scales the small difference B_new - B_old.
Usage: swap_bg.py <packed.zip> <old_bg_root> <new_bg_root> <dst_root> [--shard i/n]"""
import os, sys, zipfile, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser()
ap.add_argument('zip'); ap.add_argument('old'); ap.add_argument('new'); ap.add_argument('dst'); ap.add_argument('--shard', default='0/1'); ap.add_argument('--q', type=int, default=95)
o = ap.parse_args(); si, sn = map(int, o.shard.split('/'))
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); std_ = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def to_t(a): return torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
z = zipfile.ZipFile(o.zip)
names = sorted(n for n in z.namelist() if n.endswith('.jpg') and '/renders/' in n)
n_done = 0
with torch.inference_mode():
    for k, name in enumerate(names):
        if k % sn != si: continue
        rel = name.split('/renders/', 1)[1]                    # <scene>/<view>/<frame>.jpg
        out = f'{o.dst}/{rel}'
        if os.path.exists(out): continue
        C = to_t(cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR))
        Bo = to_t(cv2.imread(f'{o.old}/{rel}')); Bn = to_t(cv2.imread(f'{o.new}/{rel}'))
        pm = (seg(((C - mean_) / std_).half())['out'].float().softmax(1)[0, 15] > 0.5)[None, None].float()
        sm = F.avg_pool2d(F.pad(pm, (12,) * 4, mode='replicate'), 25, stride=1).clamp(0, 1)
        R = (C + (1 - sm) * (Bn - Bo)).clamp(0, 1)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, (R[0].cpu().numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, o.q])
        n_done += 1
print(f'SWAP DONE shard {si}/{sn}: {n_done}')
