#!/bin/bash
#SBATCH --job-name=val_pswap2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_pswap2_%j.log
# Mirror the production person-swap composite exactly on val:
#   person = pa*DifixTTA(mix) + (1-pa)*mix,  background = dual-backbone projection of the peer render.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
for MIX in ps_f2 ps_fl2 ps_fl4; do
  S=$D/prod_$MIX; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
  for f in $D/*_$MIX.png; do t=$(basename $f _$MIX.png); ln -sf $f $S/${t}_render.png; done
  for f in $D/*_pAVt_l15.png; do ln -sf $f $S/; done
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
  for PA in 0.30 0.50 0.70; do
    TAG=$(printf "pp%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== $MIX (person=pa*Difix(mix)+(1-pa)*mix, bg=projection) ==="
  $PY $W/metrics_both.py $S pp030 pp050 pp070 2>&1 | grep -vE "Warning|warn|load_state"
done
echo "=== reference: peer base, same composite (pbt_p050/p070_bpAVt15 from earlier sweep) ==="
grep -E "^pbt_p050_bpAVt15|^pbt_p070_bpAVt15" $V/jobs/sweepmetricsB_001_1.log 2>/dev/null
echo VAL_PSWAP2_DONE
