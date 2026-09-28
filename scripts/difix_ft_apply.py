"""Apply a Difix model (released difix_ref, or a fine-tuned checkpoint on top of it) to a dumped val set with the
SAME tiling as difix_submission.py, writing <tag>_<variant>.png (and optional raw/fixed blends) for fix_metrics.py.
Usage: envs/difix/bin/python difix_ft_apply.py <dump_dir> <variant> [--ckpt model_N.pkl] [--t 199] [--blend 0.45,0.5]"""
import sys, os, json, argparse
sys.path.insert(0, '/home/intern_2603055/projects/Difix3D/src')
import numpy as np, torch
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('variant'); ap.add_argument('--ckpt', default=None)
ap.add_argument('--t', type=int, default=199); ap.add_argument('--blend', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shift', default='')
a = ap.parse_args()
os.environ.setdefault('HF_HOME', '/home/intern_2603055/.cache/huggingface')
from model import Difix
from load_difix_ref import load_difix_ref
SNAP = sorted(__import__('glob').glob('/home/intern_2603055/.cache/huggingface/hub/models--nvidia--difix_ref/snapshots/*/'))[0]
net = Difix(lora_rank_vae=4, timestep=a.t, mv_unet=True)
load_difix_ref(net, SNAP)
if a.ckpt:
    sd = torch.load(a.ckpt, map_location='cpu')
    su = net.unet.state_dict(); su.update(sd['state_dict_unet']); net.unet.load_state_dict(su)
    sv = net.vae.state_dict(); n = 0
    for k, v in sd['state_dict_vae'].items():
        if k in sv: sv[k] = v; n += 1
    net.vae.load_state_dict(sv); print(f'loaded ckpt {a.ckpt}: unet {len(sd["state_dict_unet"])} keys, vae {n} keys')
net.set_eval(); net.unet.half(); net.vae.half(); net.text_encoder.half()
TH, TW, OV = 576, 1024, 64
def feather(h, w):
    wy = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1); wx = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
FW = feather(TH, TW)
from torchvision import transforms
T = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
@torch.inference_mode()
def fix(img, ref):
    x = torch.stack([T(img), T(ref)], 0)[None].cuda().half()
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(x, timesteps=None, prompt='remove degradation')[:, 0]
    return ((out[0].float().cpu().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255)
def tiled(img, ref):
    W, H = img.size; acc = np.zeros((H, W, 3), np.float32); ws = np.zeros((H, W, 1), np.float32)
    ys = list(range(0, max(H-TH, 0)+1, TH-OV)); xs = list(range(0, max(W-TW, 0)+1, TW-OV))
    if ys[-1] != H-TH: ys.append(H-TH)
    if xs[-1] != W-TW: xs.append(W-TW)
    for y in ys:
        for x in xs:
            b = (x, y, x+TW, y+TH); o = fix(img.crop(b), ref.crop(b)); acc[y:y+TH, x:x+TW] += o*FW; ws[y:y+TH, x:x+TW] += FW
    return np.clip(acc/np.maximum(ws, 1e-6), 0, 255)
man = json.load(open(os.path.join(a.dump, 'manifest.json')))
blends = [float(b) for b in a.blend.split(',') if b]
for i, e in enumerate(man):
    if a.limit and i >= a.limit: break
    tag = e['tag']; img = Image.open(f'{a.dump}/{tag}_render.png').convert('RGB'); ref = Image.open(f'{a.dump}/{tag}_ref.png').convert('RGB')
    raw = np.asarray(img, np.float32)
    if a.shift:  # test-time augmentation: shift the tile grid by reflect-padding top/left, then crop back
        dx, dy = [int(t) for t in a.shift.split(',')]
        pad = lambda im: Image.fromarray(np.pad(np.asarray(im), ((dy, 0), (dx, 0), (0, 0)), mode='reflect'))
        o = tiled(pad(img), pad(ref))[dy:, dx:]
    else:
        o = tiled(img, ref)
    Image.fromarray(o.astype(np.uint8)).save(f'{a.dump}/{tag}_{a.variant}.png')
    for b in blends:
        Image.fromarray(np.clip(b*o + (1-b)*raw, 0, 255).astype(np.uint8)).save(f'{a.dump}/{tag}_{a.variant}b{int(round(b*100))}.png')
    print(f'{i+1}/{len(man)} {tag}', flush=True)
print('DONE', a.variant)
