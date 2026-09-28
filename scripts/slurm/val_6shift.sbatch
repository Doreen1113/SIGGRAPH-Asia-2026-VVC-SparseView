#!/bin/bash
#SBATCH --job-name=val_6shift
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_6shift_%j.log
# Does going from 3 to 6 Difix tile-grid shifts actually lower LPIPS at unchanged SSIM? TTA is variance
# reduction rather than an SSIM/LPIPS trade, which is the only shape of gain that can close our FULL-LPIPS
# gap (0.0012) without spending the FULL-SSIM margin (0.0004).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== 3 extra shifts on the val dump, $(date -u +%H:%M) =="
for sp in "160,96:repo199s3" "480,384:repo199s4" "800,480:repo199s5"; do
  SH=${sp%%:*}; VN=${sp##*:}
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $D $VN --shift $SH --shard $i/8 > /dev/null 2>&1 & done
  wait; echo "   $VN done"
done
$PY - <<'PYEOF'
import json, numpy as np, cv2
d='/work/doreen071/vvc/fixtest/001_1'
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']
    ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2','repo199s3','repo199s4','repo199s5')]
    cv2.imwrite(f'{d}/{t}_repo199tta6.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
print('6-shift average done')
PYEOF
echo "== projection lam15 on the 6-shift target, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $D repo199tta6 pAV6_l15 --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 > /dev/null 2>&1 & done
wait
i=0
for PA in 0.70 0.80; do
  TAG=$(printf "t6_%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $D repo199tta6 $PA pAV6_l15 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "=== 6-shift (t6_*) vs 3-shift (sub*) — 同 lambda15、同 alpha ==="
$PY $W/metrics_both.py $D sub070 t6_070 sub080 t6_080 repo199tta repo199tta6 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_6SHIFT_DONE
