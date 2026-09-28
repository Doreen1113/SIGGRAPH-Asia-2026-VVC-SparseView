"""Re-derive the person/background alpha frontier on TTA-Difix outputs (mirrors person_bg_composite.py's blend,
applied directly to a val dump so it can be scored with the SAME bbox-crop FG protocol as the real submission).
Usage: ftg.sh person_bg_sweep.py <dump> <fix_variant> <pa1:ba1,pa2:ba2,...>"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
d = Path(sys.argv[1]); v = sys.argv[2]; pairs = [tuple(float(x) for x in p.split(':')) for p in sys.argv[3].split(',')]
man = json.load(open(d/'manifest.json'))
seg = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_ = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); std_ = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
def load(p): return torch.from_numpy(cv2.imread(str(p))[..., ::-1].copy()).permute(2, 0, 1)[None].cuda().float()/255
with torch.inference_mode():
    for e in man:
        raw = load(d/f"{e['tag']}_render.png"); fx = load(d/f"{e['tag']}_{v}.png")
        pm = (seg(((raw-mean_)/std_).half())['out'].float().softmax(1)[0, 15] > 0.5)[None, None].float()
        sm = F.avg_pool2d(F.pad(pm, (12,)*4, mode='replicate'), 25, stride=1).clamp(0, 1)
        for pa, ba in pairs:
            person = pa*fx + (1-pa)*raw; bg = ba*fx + (1-ba)*raw
            out = (sm*person + (1-sm)*bg).clamp(0, 1)
            arr = (out[0].cpu().numpy()*255).round().astype(np.uint8).transpose(1, 2, 0)
            cv2.imwrite(str(d/f"{e['tag']}_pb_p{int(round(pa*100)):03d}_b{int(round(ba*100)):03d}.png"), arr[..., ::-1])
        print(e['tag'], flush=True)
print('SWEEP DONE', v)
