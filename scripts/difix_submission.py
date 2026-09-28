"""Production Difix pass over a submission render tree (resumable).
  in : <src>/<case>/<view>/<frame:06d>.jpg   out: <dst>/<case>/<view>/<frame:06d>.jpg
  ref: the same frame index from the case's NEAREST TRAINING camera (compliant: the case's own train views).
Usage: python difix_submission.py <src_root> <dst_root> --mode tiled|ds [--cases a,b] [--views 17,18]"""
import sys, os, glob, json, argparse, time
import numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, '/home/intern_2603055/projects/Difix3D/src')
from pipeline_difix import DifixPipeline
REF_MAP = json.load(open('/home/intern_2603055/vvc/work/ref_map.json'))
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--mode', default='tiled'); ap.add_argument('--cases', default=''); ap.add_argument('--views', default='')
a = ap.parse_args()
P = glob.glob('/home/intern_2603055/.cache/huggingface/hub/models--nvidia--difix_ref/snapshots/*/')[0]
pipe = DifixPipeline.from_pretrained(P, torch_dtype=torch.float16).to('cuda'); pipe.set_progress_bar_config(disable=True)
TH, TW, OV = 576, 1024, 64
DIFIX_T = int(os.environ.get('DIFIX_T', '199'))
def fix(img, ref): return pipe('remove degradation', image=img, ref_image=ref, num_inference_steps=1, timesteps=[DIFIX_T], guidance_scale=0.0, height=TH, width=TW).images[0]
def feather(h, w):
    wy = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1); wx = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
FW = feather(TH, TW)
BATCH = int(os.environ.get('DIFIX_BATCH', '8'))
def tiled(img, ref):
    W, H = img.size; acc = np.zeros((H, W, 3), np.float32); ws = np.zeros((H, W, 1), np.float32)
    ys = list(range(0, max(H-TH, 0)+1, TH-OV)); xs = list(range(0, max(W-TW, 0)+1, TW-OV))
    if ys[-1] != H-TH: ys.append(H-TH)
    if xs[-1] != W-TW: xs.append(W-TW)
    boxes = [(x, y, x+TW, y+TH) for y in ys for x in xs]
    for i in range(0, len(boxes), BATCH):
        bb = boxes[i:i+BATCH]
        outs = pipe(['remove degradation']*len(bb), image=[img.crop(b) for b in bb], ref_image=[ref.crop(b) for b in bb], num_inference_steps=1, timesteps=[DIFIX_T], guidance_scale=0.0, height=TH, width=TW).images
        for b, o in zip(bb, outs):
            x, y = b[0], b[1]; acc[y:y+TH, x:x+TW] += np.asarray(o, np.float32)*FW; ws[y:y+TH, x:x+TW] += FW
    return Image.fromarray(np.clip(acc/np.maximum(ws, 1e-6), 0, 255).astype(np.uint8))
def ds(img, ref):
    W, H = img.size; return fix(img.resize((TW, TH), Image.LANCZOS), ref.resize((TW, TH), Image.LANCZOS)).resize((W, H), Image.LANCZOS)
cases = a.cases.split(',') if a.cases else sorted(os.listdir(a.src)); views_f = set(a.views.split(',')) if a.views else None
n = 0; t0 = time.time()
with torch.inference_mode():
    for c in cases:
        data = f'/home/intern_2603055/vvc/data/{c}'
        for v in sorted(os.listdir(f'{a.src}/{c}')):
            if views_f and v not in views_f: continue
            ref_cam = REF_MAP[c][v]
            refs = sorted(glob.glob(f'{data}/images/{ref_cam}/*.jpg')); os.makedirs(f'{a.dst}/{c}/{v}', exist_ok=True)
            for f in sorted(os.listdir(f'{a.src}/{c}/{v}')):
                out = f'{a.dst}/{c}/{v}/{f}'
                if os.path.exists(out): continue
                k = int(os.path.splitext(f)[0]); img = Image.open(f'{a.src}/{c}/{v}/{f}').convert('RGB'); ref = Image.open(refs[min(k, len(refs)-1)]).convert('RGB')
                if ref.size != img.size: ref = ref.resize(img.size, Image.LANCZOS)
                o = tiled(img, ref) if a.mode == 'tiled' else ds(img, ref)
                cv2.imwrite(out, np.asarray(o)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
                if n % 50 == 0: print(f'{n} done, {(time.time()-t0)/n:.1f}s/img', flush=True)
print(f'DONE {n} images in {(time.time()-t0)/60:.1f} min')
