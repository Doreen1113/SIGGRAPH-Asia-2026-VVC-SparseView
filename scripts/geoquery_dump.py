"""GeoQuery (SIGGRAPH 2026, single-step SD-Turbo refiner with geometry-guided cross-view attention) as a drop-in for
difix_dump_h200.py on a val dump dir (<tag>_render.png + <tag>_ref.png -> <tag>_<variant>.png).
Adaptations vs the upstream sample(): separate intrinsics for reference/target (upstream uses one K), tiled 4K with
matching reference crops and per-tile principal-point shifts, reference depth = VGGT metric depth of the reference
train camera at the same frame (depth_vggt/<case>/f<frame>.npz, scale-aligned to GT camera centres).
Usage: geoquery_dump.py <dump_dir> <variant> --case <case_dir> --depth <depth_dir> [--shift dx,dy] [--shard i/n]"""
import sys, os, re, json, argparse, time, types
GQ = '/work/doreen071/vvc/ext/GeoQuery/src'
sys.path.insert(0, GQ); sys.path.insert(0, '/work/doreen071/vvc/ext/pylib')
os.environ.setdefault('HF_HOME', '/work/doreen071/vvc/hf_home')
import numpy as np, torch, torch.nn.functional as F, cv2
torch.set_num_threads(6); cv2.setNumThreads(4)
ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('variant')
ap.add_argument('--case', required=True); ap.add_argument('--depth', required=True)
ap.add_argument('--intri', default='intri.yml'); ap.add_argument('--extri', default='extri.yml')
ap.add_argument('--shift', default=''); ap.add_argument('--shard', default='0/1')
ap.add_argument('--ckpt', default='/work/doreen071/vvc/ext/ckpt/geoquery_window_size3_down1.pkl')
ap.add_argument('--t', type=int, default=199); ap.add_argument('--nogeo', action='store_true', help='disable geometry (ablation)')
a = ap.parse_args()
from model import GeoQuery
from softsplat import softsplat

def read_cams(intri, extri):
    si, se = open(intri).read(), open(extri).read(); cams = {}
    for n in re.findall(r'^  - "(\w+)"', si, re.M):
        K = np.array(re.search(r'K_%s:.*?data: \[(.*?)\]' % n, si, re.S).group(1).split(','), float).reshape(3, 3)
        R = np.array(re.search(r'Rot_%s:.*?data: \[(.*?)\]' % n, se, re.S).group(1).split(','), float).reshape(3, 3)
        T = np.array(re.search(r'T_%s:.*?data: \[(.*?)\]' % n, se, re.S).group(1).split(','), float)
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ T
        cams[n] = (K, c2w)
    return cams
cams = read_cams(f'{a.case}/{a.intri}', f'{a.case}/{a.extri}')

def corr_two_k(ref_coord_map, ref_depth, K2, ref_pose, target_pose, alpha=0.5):
    """build_geometric_correspondence with K2 = (B,2,3,3): [:,0] reference intrinsics, [:,1] target intrinsics."""
    Kr, Kt = K2[:, 0], K2[:, 1]
    B, _, H, W = ref_coord_map.shape; dev = ref_coord_map.device
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    pix = torch.stack([x, y, torch.ones_like(x)], 0).reshape(1, 3, -1).expand(B, -1, -1)
    pts = (torch.inverse(Kr) @ pix) * ref_depth.reshape(B, 1, -1)
    world = ref_pose @ torch.cat([pts, torch.ones_like(pts[:, :1])], 1)
    tc = (torch.inverse(target_pose) @ world)[:, :3]
    invalid = (tc[:, 2:3] <= 0) | (ref_depth.reshape(B, 1, -1) <= 0)
    proj = Kt @ tc; proj = proj[:, :2] / proj[:, 2:3].clamp(min=1e-8)
    proj[invalid.expand(-1, 2, -1)] = -1e6
    flow = torch.zeros(B, 2, H, W, device=dev)
    flow[:, 0] = (proj[:, 0] - x.reshape(1, -1)).reshape(B, H, W); flow[:, 1] = (proj[:, 1] - y.reshape(1, -1)).reshape(B, H, W)
    imp = alpha / tc[:, 2:3].clamp(min=1e-6); imp[invalid] = 0.0
    imp = imp - imp.amin(dim=2, keepdim=True); imp = imp / (imp.amax(dim=2, keepdim=True) + 1e-6)
    imp = (imp * 10 - 10).reshape(B, 1, H, W)
    corr = softsplat(ref_coord_map, flow, imp, 'soft')
    inv = (corr == 0.0).all(dim=1, keepdim=True).to(ref_coord_map.dtype)
    return {'correspondence': corr, 'validity_mask': 1.0 - inv}

net = GeoQuery(pretrained_path=a.ckpt, lora_rank_vae=4, timestep=a.t, neighborhood_size=3, low_res_only=True)
net.set_eval()
def prep(self, ref_depth, K, ref_pose, target_pose, H, W):
    B = ref_depth.shape[0]; dev = ref_depth.device
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    grid = torch.stack([x, y], 0).unsqueeze(0).repeat(B, 1, 1, 1)
    r = corr_two_k(grid, ref_depth.float(), K.float(), ref_pose.float(), target_pose.float())
    prep.valid.append(r['validity_mask'].mean().item())
    return {'correspondence': r['correspondence'].float(), 'validity_mask': r['validity_mask'].float()}
prep.valid = []
net.prepare_correspondence = types.MethodType(prep, net)
caption = 'remove degradation'

TH, TW, OV = 576, 1024, 64
def feather(h, w):
    wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1); wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
FW = torch.from_numpy(feather(TH, TW)).cuda()

@torch.inference_mode()
def fix(img, ref, dep, Kr, Kt, pr, pt):
    x = torch.stack([img, ref], 0)[None] * 2 - 1
    geo = None if a.nogeo else {'ref_depth': dep[None, None], 'K': torch.stack([Kr, Kt])[None], 'ref_pose': pr[None], 'target_pose': pt[None]}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net.forward(x, prompt=caption, geometry_inputs=geo)[:, 0]
    return (out[0].float().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0) * 255

def tiled(img, ref, dep, Kr, Kt, pr, pt):
    H, W = img.shape[-2:]; acc = torch.zeros((H, W, 3), device='cuda'); ws = torch.zeros((H, W, 1), device='cuda')
    ys = list(range(0, max(H - TH, 0) + 1, TH - OV)); xs = list(range(0, max(W - TW, 0) + 1, TW - OV))
    if ys[-1] != H - TH: ys.append(H - TH)
    if xs[-1] != W - TW: xs.append(W - TW)
    for y0 in ys:
        for x0 in xs:
            sh = torch.tensor([[0, 0, x0], [0, 0, y0], [0, 0, 0]], device='cuda', dtype=torch.float32)
            o = fix(img[:, y0:y0 + TH, x0:x0 + TW].contiguous(), ref[:, y0:y0 + TH, x0:x0 + TW].contiguous(), dep[y0:y0 + TH, x0:x0 + TW].contiguous(), Kr - sh, Kt - sh, pr, pt)
            acc[y0:y0 + TH, x0:x0 + TW] += o * FW; ws[y0:y0 + TH, x0:x0 + TW] += FW
    return (acc / ws.clamp_min(1e-6)).clamp(0, 255)

def load(p): return torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1).cuda().float() / 255
si, sn = [int(t) for t in a.shard.split('/')]
man = json.load(open(f'{a.dump}/manifest.json'))[si::sn]
t0 = time.time()
for i, e in enumerate(man):
    t = e['tag']; out = f'{a.dump}/{t}_{a.variant}.png'
    if os.path.exists(out): continue
    img = load(f'{a.dump}/{t}_render.png'); ref = load(f'{a.dump}/{t}_ref.png')
    H, W = img.shape[-2:]; Hr, Wr = ref.shape[-2:]
    if (Hr, Wr) != (H, W): ref = F.interpolate(ref[None], size=(H, W), mode='bilinear', align_corners=False, antialias=True)[0]
    Kr, cr = cams[e['ref_cam']]; Kt, ct = cams[e['view']]
    Kr = Kr.copy(); Kr[0] *= W / Wr; Kr[1] *= H / Hr
    z = np.load(f"{a.depth}/f{int(e['frame']):06d}.npz"); vi = list(z['views']).index(e['ref_cam'])
    dep = torch.from_numpy(z['depth'][vi].astype(np.float32)).cuda()[None, None]
    dep = F.interpolate(dep, size=(H, W), mode='nearest')[0, 0]
    Kr = torch.from_numpy(Kr).float().cuda(); Kt = torch.from_numpy(Kt).float().cuda()
    pr = torch.from_numpy(cr).float().cuda(); pt = torch.from_numpy(ct).float().cuda()
    if a.shift:
        dx, dy = [int(v) for v in a.shift.split(',')]
        pad = lambda im: F.pad(im[None], (dx, 0, dy, 0), mode='reflect')[0]
        sh = torch.tensor([[0, 0, -dx], [0, 0, -dy], [0, 0, 0]], device='cuda', dtype=torch.float32)
        o = tiled(pad(img), pad(ref), F.pad(dep[None, None], (dx, 0, dy, 0))[0, 0], Kr - sh, Kt - sh, pr, pt)[dy:, dx:]
    else:
        o = tiled(img, ref, dep, Kr, Kt, pr, pt)
    cv2.imwrite(out, o.round().byte().cpu().numpy()[..., ::-1])
    v = np.mean(prep.valid) if prep.valid else float('nan'); prep.valid.clear()
    print(f'{i+1}/{len(man)} {t} {(time.time()-t0)/(i+1):.1f}s/img  mean tile validity {v:.3f}', flush=True)
print('GEOQUERY DUMP DONE', a.variant)
