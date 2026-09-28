#!/bin/bash
#SBATCH --job-name=val_hb3alpha
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_hb3alpha_%j.log
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
S=$D/ps_bg_p4d2s3_h
CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py dump $S tta 0.80 bg 1.0 p080 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $W/composite2.py dump $S tta 0.75 bg 1.0 p075 > /dev/null 2>&1 &
wait
$PY - "$S" "$D/dil_s5" <<'PYEOF'
# 5-decimal metrics for the close calls (FULL-LPIPS ties at 4 decimals)
import sys, json, numpy as np, cv2, torch, lpips
S, S0 = sys.argv[1], sys.argv[2]
man = json.load(open(S + '/manifest.json'))
net = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
def lp(a, b):
    ta = torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    tb = torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    with torch.no_grad(): return net(ta, tb).item()
for name, d in [('d00(S5,現行)', (S0, 'd00')), ('hb3 p085', (S, 'p085')), ('hb3 p080', (S, 'p080')), ('hb3 p075', (S, 'p075')), ('hb3 p070', (S, 'p070'))]:
    vals = []
    for e in man:
        t = e['tag']; gt = cv2.imread(f"{S}/{t}_gt.png"); im = cv2.imread(f"{d[0]}/{t}_{d[1]}.png")
        if gt.shape != im.shape: im = cv2.resize(im, (gt.shape[1], gt.shape[0]))
        vals.append(lp(im, gt))
    print(f"{name:<14} FULL-LPIPS(alex) = {np.mean(vals):.5f}")
PYEOF
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
$PY $W/metrics_both.py $S p085 p080 p075 p070 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_HB3ALPHA_DONE $(date +%H:%M)
