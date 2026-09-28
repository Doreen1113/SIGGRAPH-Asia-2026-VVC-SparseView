"""Write <img>_mask.png (255 = background / supervise, 0 = person, dilated) for every png listed in <pseudo_dir>/index.json.
Usage: ftg.sh person_mask.py <pseudo_dir> [--dilate 12]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2, torch
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
ap = argparse.ArgumentParser(); ap.add_argument('pdir'); ap.add_argument('--dilate', type=int, default=12); ap.add_argument('--overwrite', action='store_true')
a = ap.parse_args(); pdir = Path(a.pdir)
w = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1; model = deeplabv3_resnet101(weights=w).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
items = json.load(open(pdir / 'index.json')); n = 0; fg = []
with torch.inference_mode():
    for e in items:
        src = pdir / e['path']; dst = pdir / e['path'].replace('.png', '_mask.png')
        if dst.exists() and not a.overwrite: continue
        im = cv2.imread(str(src))[..., ::-1].copy()
        x = torch.from_numpy(im).cuda().permute(2, 0, 1)[None].float() / 255; x = ((x - mean) / std).half()
        out = model(x)['out'].float().softmax(1)[0]
        person = (out[15] > 0.3).cpu().numpy().astype(np.uint8)
        if a.dilate > 0: person = cv2.dilate(person, np.ones((a.dilate * 2 + 1, a.dilate * 2 + 1), np.uint8))
        cv2.imwrite(str(dst), (1 - person) * 255); n += 1; fg.append(person.mean())
print(f'wrote {n} masks; mean person fraction {np.mean(fg) if fg else 0:.3f}')
