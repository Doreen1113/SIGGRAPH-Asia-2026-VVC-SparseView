"""Difix over a val dump dir (<tag>_render.png + <tag>_ref.png -> <tag>_<variant>.png), H200 paths.
Usage: difix_dump_h200.py <dump_dir> <variant> [--shift dx,dy] [--shard i/n]"""
import sys, os, glob, json, argparse, time
sys.path.insert(0, '/work/doreen071/vvc/difix_src/src')
os.environ.setdefault('HF_HOME', '/work/doreen071/vvc/hf_home')
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np, torch, cv2, torch.nn.functional as F
torch.set_num_threads(6); cv2.setNumThreads(4)
from model import Difix
from load_difix_ref import load_difix_ref
ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('variant')
ap.add_argument('--shift', default=''); ap.add_argument('--shard', default='0/1'); ap.add_argument('--ref', default='ref')
a = ap.parse_args()
SNAP = sorted(glob.glob('/work/doreen071/vvc/hf_home/hub/models--nvidia--difix_ref/snapshots/*/'))[0]
net = Difix(lora_rank_vae=4, timestep=int(os.environ.get('DIFIX_T', '199')), mv_unet=True)
load_difix_ref(net, SNAP); net.set_eval(); net.unet.half(); net.vae.half(); net.text_encoder.half()
TH, TW, OV = 576, 1024, 64
def feather(h, w):
    wy = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1); wx = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
FW = torch.from_numpy(feather(TH, TW)).cuda()
@torch.inference_mode()
def fix(img, ref):
    x = torch.stack([img, ref], 0)[None].half()*2-1
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(x, timesteps=None, prompt='remove degradation')[:, 0]
    return (out[0].float().clamp(-1, 1)*0.5+0.5).permute(1, 2, 0)*255
def tiled(img, ref):
    H, W = img.shape[-2:]; acc = torch.zeros((H, W, 3), device='cuda'); ws = torch.zeros((H, W, 1), device='cuda')
    ys = list(range(0, max(H-TH, 0)+1, TH-OV)); xs = list(range(0, max(W-TW, 0)+1, TW-OV))
    if ys[-1] != H-TH: ys.append(H-TH)
    if xs[-1] != W-TW: xs.append(W-TW)
    for y in ys:
        for x in xs:
            o = fix(img[:, y:y+TH, x:x+TW], ref[:, y:y+TH, x:x+TW]); acc[y:y+TH, x:x+TW] += o*FW; ws[y:y+TH, x:x+TW] += FW
    return (acc/ws.clamp_min(1e-6)).clamp(0, 255)
def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1).cuda().float()/255
si, sn = [int(t) for t in a.shard.split('/')]
man = json.load(open(f'{a.dump}/manifest.json'))[si::sn]
t0 = time.time()
for i, e in enumerate(man):
    t = e['tag']; out = f"{a.dump}/{t}_{a.variant}.png"
    if os.path.exists(out): continue
    img = load(f'{a.dump}/{t}_render.png'); ref = load(f'{a.dump}/{t}_{a.ref}.png')
    if ref.shape != img.shape: ref = F.interpolate(ref[None], size=img.shape[-2:], mode='bilinear', align_corners=False, antialias=True)[0]
    if a.shift:
        dx, dy = [int(x) for x in a.shift.split(',')]
        pad = lambda im: F.pad(im[None], (dx, 0, dy, 0), mode='reflect')[0]
        o = tiled(pad(img), pad(ref))[dy:, dx:]
    else:
        o = tiled(img, ref)
    cv2.imwrite(out, o.round().byte().cpu().numpy()[..., ::-1])
    print(f'{i+1}/{len(man)} {t} {(time.time()-t0)/(i+1):.1f}s', flush=True)
print('DIFIX DUMP DONE', a.variant)
