"""Temporal background average for fixed-camera renders.
For each view, frames are sorted; output frame f = mean of neighbour frames f-K..f+K over pixels that are
background in BOTH the current and the neighbour frame (DeepLabV3 person mask, dilated); person pixels of the
current frame are kept as-is. Static cameras + static background => this averages away per-frame floaters and
keyframe-segment flicker without touching the person.
Dump mode : temporal_avg.py dump <dump_dir> <K> <out_variant> [--dilate 24]
Tree mode : temporal_avg.py tree <src_root> <dst_root> <K> [--dilate 24] [--cases a,b] [--shard i/n]"""
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['dump', 'tree']); ap.add_argument('args', nargs='+')
ap.add_argument('--dilate', type=int, default=24); ap.add_argument('--cases', default=''); ap.add_argument('--shard', default='0/1')
o = ap.parse_args()
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_ = torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2,0,1)[None].cuda().float()/255
@torch.inference_mode()
def person_mask(img):
    pm = (seg(((img-mean_)/std_).half())['out'].float().softmax(1)[0,15] > 0.5)[None,None].float()
    if o.dilate: pm = F.max_pool2d(pm, 2*o.dilate+1, stride=1, padding=o.dilate)
    return pm  # 1 = person (dilated)
@torch.inference_mode()
def process_sequence(paths, K, write):
    """paths: ordered list of (in_path, out_path) for one view."""
    imgs = {}; masks = {}
    def get(i):
        if i not in imgs:
            imgs[i] = load(paths[i][0]); masks[i] = person_mask(imgs[i])
        return imgs[i], masks[i]
    for i in range(len(paths)):
        if os.path.exists(paths[i][1]): continue
        cur, mc = get(i); bg_cur = 1 - mc
        acc = torch.zeros_like(cur); cnt = torch.zeros_like(mc)
        for j in range(max(0, i-K), min(len(paths), i+K+1)):
            im, mj = get(j); w = bg_cur * (1 - mj)
            acc += im * w; cnt += w
        avg = acc / cnt.clamp(min=1)
        out = torch.where(cnt > 0, avg, cur)             # background pixels: temporal mean; person: original
        write(paths[i][1], out)
        for j in list(imgs):                              # free frames that are out of the window
            if j < i - K: imgs.pop(j); masks.pop(j)
def write_png(p, t): cv2.imwrite(p, (t[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8).transpose(1,2,0)[..., ::-1])
def write_jpg(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, (t[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8).transpose(1,2,0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
if o.mode == 'dump':
    d, K, vo = o.args[0], int(o.args[1]), o.args[2]
    man = json.load(open(f'{d}/manifest.json')); views = {}
    for e in man: views.setdefault(e['view'], []).append(e)
    for v, es in views.items():
        es.sort(key=lambda e: e['frame'])
        process_sequence([(f"{d}/{e['tag']}_render.png", f"{d}/{e['tag']}_{vo}.png") for e in es], K, write_png)
    print('temporal avg done', vo)
else:
    src, dst, K = o.args[0], o.args[1], int(o.args[2]); si, sn = map(int, o.shard.split('/'))
    cases = [c for c in sorted(os.listdir(src)) if not o.cases or c in o.cases.split(',')]
    seqs = [(c, v) for c in cases for v in sorted(os.listdir(f'{src}/{c}'))]
    for k, (c, v) in enumerate(seqs):
        if k % sn != si: continue
        fs = sorted(os.listdir(f'{src}/{c}/{v}'))
        process_sequence([(f'{src}/{c}/{v}/{f}', f'{dst}/{c}/{v}/{f}') for f in fs], K, write_jpg)
        print('done', c, v, flush=True)
    print(f'TEMPORAL AVG TREE DONE shard {si}/{sn}')
