#!/bin/bash
#SBATCH --job-name=val_pswap40
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_pswap40_%j.log
# Person-region swap using FGW40_L60K (our strongest FG model) on the PEER base, swept to high person-Difix
# alpha. The previous person swap (FGW80+LONG60K, alpha 0.50) really did deliver FG-SSIM on the test set
# (+0.0059, landing at 0.83790) but blew FG-LPIPS (+0.0156); a stronger FG model plus a higher alpha should
# keep the SSIM gain while pushing FG-LPIPS back under mmm's 0.26582.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
for name,w in [('p40a',2),('p40b',4)]:
    for e in man:
        t=e['tag']
        a=cv2.imread(f'{D}/{t}_render.png').astype(np.float64)*4
        f=cv2.imread(f'{H}/m_FGW40_L60K/{t}_render.png')
        if f is None: print('missing'); break
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip((a+f.astype(np.float64)*w)/(4+w),0,255).round().astype(np.uint8))
    print('built', name)
PYEOF
for MIX in p40a p40b; do
  S=$D/ps40_$MIX; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png $D/*_pAVt_l15.png; do ln -sf $f $S/; done
  for f in $D/*_$MIX.png; do t=$(basename $f _$MIX.png); ln -sf $f $S/${t}_render.png; done
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
  done
  $PY - <<PYEOF
import json, numpy as np, cv2
d="$S"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
  i=0
  for PA in 0.50 0.70 0.85 0.95; do
    TAG=$(printf "w%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== 人物區 = peer4 + FGW40_L60K x${MIX: -1}（a=2 或 b=4）, 背景 = peer 投影 lam15 ==="
  $PY $W/metrics_both.py $S w050 w070 w085 w095 2>&1 | grep -vE "Warning|warn|load_state"
done
echo VAL_PSWAP40_DONE
