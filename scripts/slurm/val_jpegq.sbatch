#!/bin/bash
#SBATCH --job-name=val_jpegq
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_jpegq_%j.log
V=/work/doreen071/vvc; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY - <<'PYEOF'
import json, cv2, numpy as np
D='/work/doreen071/vvc/fixtest/001_1'
for e in json.load(open(D+'/manifest.json')):
    t=e['tag']; im=cv2.imread(f'{D}/{t}_sub070.png')
    for q in (90,95,98,100):
        ok,buf=cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY,q]); cv2.imwrite(f'{D}/{t}_jq{q}.png', cv2.imdecode(buf,1))
print('built')
PYEOF
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
$PY $V/work/metrics_both.py $D sub070 jq90 jq95 jq98 jq100 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_JPEGQ_DONE
