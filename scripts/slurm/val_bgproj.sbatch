#!/bin/bash
#SBATCH --job-name=val_bgproj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgproj_%j.log
# The new background (diverse mix) gives FULL-PSNR +0.60 but FULL-LPIPS +0.0079 after the lam15 projection, which
# would cost our FULL-LPIPS rank. Objective is LPIPS(x, D) + lam*(1-SSIM(x, R)). Sweep:
#   own target  : R = mix raw, D = mix Difix TTA, lam 5/8/10/12        (pull harder toward Difix features)
#   cross target: R = mix raw, D = PEER Difix TTA, lam 10/15/20        (structure from the mix, texture from peer)
#   mean target : R = mix raw, D = mean(mix TTA, peer TTA), lam 15
# Each background is composited with the same S5 person source (alpha 0.85) and compared to d00 (current recipe).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$D/ps_bg_p4x2d2s; S=$D/dil_s5
for f in $D/*_repo199tta.png; do t=$(basename $f _repo199tta.png); ln -sf $f $P/${t}_ptta.png; done
$PY - "$P" <<'PYEOF'
import json, sys, numpy as np, cv2
d=sys.argv[1]
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; a=cv2.imread(f'{d}/{t}_tta.png').astype(np.float32); b=cv2.imread(f'{d}/{t}_ptta.png').astype(np.float32)
    cv2.imwrite(f'{d}/{t}_mtta.png', np.clip((a+b)/2,0,255).round().astype(np.uint8))
print('mean target built')
PYEOF
g=0
for spec in "tta 5 nbl05" "tta 8 nbl08" "tta 10 nbl10" "tta 12 nbl12" "ptta 10 xl10" "ptta 15 xl15" "ptta 20 xl20" "mtta 15 ml15"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py dump $P $1 $3 --nets alex:1,vgg:0.5 --lam $2 --steps 200 --batch 2 > $V/jobs/valbgproj_$3.log 2>&1 &
  g=$((g+1))
done
wait
echo "projections: $(ls $P/*_nbl05.png $P/*_xl15.png $P/*_ml15.png 2>/dev/null | wc -l)/72"
g=0
for v in nbl05 nbl08 nbl10 nbl12 xl10 xl15 xl20 ml15; do
  for f in $P/*_$v.png; do t=$(basename $f _$v.png); ln -sf $f $S/${t}_bg_$v.png; done
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $S tta 0.85 bg_$v 1.0 c_$v > /dev/null 2>&1 &
  g=$((g+1))
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(d00 = 現行 peer 背景; nbp = 新背景 lam15 own target)"
$PY $W/metrics_both.py $S d00 nbp c_nbl05 c_nbl08 c_nbl10 c_nbl12 c_xl10 c_xl15 c_xl20 c_ml15 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGPROJ_DONE $(date +%H:%M)
