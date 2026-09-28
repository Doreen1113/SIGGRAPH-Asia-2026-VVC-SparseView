"""Production Difix pass over a submission render tree (resumable), H200-native paths.
  in : <src>/<case>/<view>/<frame:06d>.jpg   out: <dst>/<case>/<view>/<frame:06d>.jpg
  ref: the same frame index from the case's NEAREST TRAINING camera (compliant: the case's own train views).
Usage: python difix_submission_h200.py <src_root> <dst_root> [--cases a,b] [--views 17,18] [--shift dx,dy]
Env: DIFIX_T (timestep, default 199), DIFIX_SHARD i/n (process every n-th case-view for multi-GPU sharding)
"""
import sys, os, glob, json, argparse, time
sys.path.insert(0, '/work/doreen071/vvc/difix_src/src')
os.environ.setdefault('HF_HOME', '/work/doreen071/vvc/hf_home')
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np, torch, cv2, torch.nn.functional as F
torch.set_num_threads(6); cv2.setNumThreads(4)
from PIL import Image
from torchvision import transforms
from model import Difix
from load_difix_ref import load_difix_ref

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--cases', default=''); ap.add_argument('--views', default='')
ap.add_argument('--shift', default='')
a = ap.parse_args()

REF_MAP = json.load(open('/work/doreen071/vvc/work/ref_map.json'))
SNAP = sorted(glob.glob('/work/doreen071/vvc/hf_home/hub/models--nvidia--difix_ref/snapshots/*/'))[0]
DIFIX_T = int(os.environ.get('DIFIX_T', '199'))
net = Difix(lora_rank_vae=4, timestep=DIFIX_T, mv_unet=True)
load_difix_ref(net, SNAP)
net.set_eval(); net.unet.half(); net.vae.half(); net.text_encoder.half()

TH, TW, OV = 576, 1024, 64
def feather(h, w):
    wy = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1); wx = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
FW = torch.from_numpy(feather(TH, TW)).cuda()

@torch.inference_mode()
def fix(img, ref):  # img, ref: (3,TH,TW) cuda float in [0,1]
    x = torch.stack([img, ref], 0)[None].half()*2-1
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(x, timesteps=None, prompt='remove degradation')[:, 0]
    return (out[0].float().clamp(-1, 1)*0.5+0.5).permute(1, 2, 0)*255

def tiled(img, ref):  # (3,H,W) cuda float
    H, W = img.shape[-2:]; acc = torch.zeros((H, W, 3), device='cuda'); ws = torch.zeros((H, W, 1), device='cuda')
    ys = list(range(0, max(H-TH, 0)+1, TH-OV)); xs = list(range(0, max(W-TW, 0)+1, TW-OV))
    if ys[-1] != H-TH: ys.append(H-TH)
    if xs[-1] != W-TW: xs.append(W-TW)
    for y in ys:
        for x in xs:
            o = fix(img[:, y:y+TH, x:x+TW], ref[:, y:y+TH, x:x+TW]); acc[y:y+TH, x:x+TW] += o*FW; ws[y:y+TH, x:x+TW] += FW
    return (acc/ws.clamp_min(1e-6)).clamp(0, 255)

def load_cuda(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1).cuda().float()/255

cases = a.cases.split(',') if a.cases else sorted(os.listdir(a.src))
views_f = set(a.views.split(',')) if a.views else None
shard_i, shard_n = [int(x) for x in os.environ['DIFIX_SHARD'].split('/')] if 'DIFIX_SHARD' in os.environ else (0, 1)

jobs = []
for c in cases:
    for v in sorted(os.listdir(f'{a.src}/{c}')):
        if views_f and v not in views_f: continue
        jobs.append((c, v))
jobs = jobs[shard_i::shard_n]

n = 0; t0 = time.time()
for c, v in jobs:
    ref_cam = REF_MAP[c][v]
    data = f'/work/doreen071/vvc/data/{c}'
    refs = sorted(glob.glob(f'{data}/images/{ref_cam}/*.jpg'))
    os.makedirs(f'{a.dst}/{c}/{v}', exist_ok=True)
    for f in sorted(os.listdir(f'{a.src}/{c}/{v}')):
        out = f'{a.dst}/{c}/{v}/{f}'
        if os.path.exists(out): continue
        k = int(os.path.splitext(f)[0])
        img = load_cuda(f'{a.src}/{c}/{v}/{f}'); ref = load_cuda(refs[min(k, len(refs)-1)])
        if ref.shape != img.shape: ref = F.interpolate(ref[None], size=img.shape[-2:], mode='bilinear', align_corners=False, antialias=True)[0]
        if a.shift:
            dx, dy = [int(t) for t in a.shift.split(',')]
            pad = lambda im: F.pad(im[None], (dx, 0, dy, 0), mode='reflect')[0]
            o = tiled(pad(img), pad(ref))[dy:, dx:]
        else:
            o = tiled(img, ref)
        o = o.round().byte().cpu().numpy()
        cv2.imwrite(out, o[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
        if n % 20 == 0: print(f'{n}/{sum(len(os.listdir(f"{a.src}/{cc}/{vv}")) for cc,vv in jobs)} done, {(time.time()-t0)/n:.1f}s/img', flush=True)
print(f'DONE shard {shard_i}/{shard_n}: {n} images in {(time.time()-t0)/60:.1f} min')
