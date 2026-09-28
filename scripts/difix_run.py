"""Apply Difix (reference-guided) to dumped val renders. Two variants:
  difix   : native-4K tiling (576x1024 tiles, 64px overlap, feathered blend), ref tile = same crop of nearest train view
  difixds : downscale to 1024 wide -> fix -> upscale back
Usage: python difix_run.py <dump_dir>"""
import sys, glob, json, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/intern_2603055/projects/Difix3D/src')
from pipeline_difix import DifixPipeline
d = sys.argv[1]; TS = int(sys.argv[2]) if len(sys.argv) > 2 else 199; SUF = sys.argv[3] if len(sys.argv) > 3 else 'difix'; man = json.load(open(f'{d}/manifest.json'))
P = glob.glob('/home/intern_2603055/.cache/huggingface/hub/models--nvidia--difix_ref/snapshots/*/')[0]
pipe = DifixPipeline.from_pretrained(P, torch_dtype=torch.float16).to('cuda'); pipe.set_progress_bar_config(disable=True)
TH, TW, OV = 576, 1024, 64
def fix(img, ref):
    return pipe('remove degradation', image=img, ref_image=ref, num_inference_steps=1, timesteps=[TS], guidance_scale=0.0, height=TH, width=TW).images[0]
def feather(h, w):
    wy = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1); wx = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.minimum(np.outer(wy, wx), OV).astype(np.float32)[..., None]
def tiled(img, ref):
    W, H = img.size; acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W, 1), np.float32); fw = feather(TH, TW)
    ys = list(range(0, max(H-TH, 0)+1, TH-OV)); xs = list(range(0, max(W-TW, 0)+1, TW-OV))
    if ys[-1] != H-TH: ys.append(H-TH)
    if xs[-1] != W-TW: xs.append(W-TW)
    for y in ys:
        for x in xs:
            box = (x, y, x+TW, y+TH); o = np.asarray(fix(img.crop(box), ref.crop(box)), np.float32)
            acc[y:y+TH, x:x+TW] += o*fw; wsum[y:y+TH, x:x+TW] += fw
    return Image.fromarray(np.clip(acc/np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8))
def downscaled(img, ref):
    W, H = img.size; s = TW/W; h = int(round(H*s/8))*8
    o = fix(img.resize((TW, h), Image.LANCZOS), ref.resize((TW, h), Image.LANCZOS)) if h == TH else None
    if o is None:  # keep aspect: pad/crop height to 576 by resizing the whole frame to 1024x576 then back
        o = fix(img.resize((TW, TH), Image.LANCZOS), ref.resize((TW, TH), Image.LANCZOS))
    return o.resize((W, H), Image.LANCZOS)
t0 = time.time()
with torch.inference_mode():
    for e in man:
        img = Image.open(f"{d}/{e['tag']}_render.png").convert('RGB'); ref = Image.open(f"{d}/{e['tag']}_ref.png").convert('RGB')
        tiled(img, ref).save(f"{d}/{e['tag']}_{SUF}.png")
print(f'fixed {len(man)} images in {time.time()-t0:.0f}s')
