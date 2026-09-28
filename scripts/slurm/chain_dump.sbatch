#!/bin/bash
#SBATCH --job-name=chain_dump
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/chain_%x_%j.log
# Full post-processing chain on a val dump dir: Difix x3 shifts -> TTA average -> dual-backbone projection
# -> person/bg composites -> metrics (both LPIPS backbones).
# Usage: sbatch --job-name=chain_e6 --export=ALL,DUMP=/work/doreen071/vvc/fixtest_h200/001_1_seq0/e6 chain_dump.sbatch
PY=/home/doreen071/.conda/envs/vvc/bin/python
V=/work/doreen071/vvc; W=$V/work
export OMP_NUM_THREADS=10
echo "== difix 3 shifts, sharded 8-way each"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $DUMP $vn ${s:+--shift $s} --shard $i/8 > $V/jobs/chain_${SLURM_JOB_NAME}_${vn}_$i.log 2>&1 &
  done
  wait
  echo "   $vn done"
done
echo "== TTA average"
$PY - <<PYEOF
import json, numpy as np, cv2
d = "$DUMP"
for e in json.load(open(d+'/manifest.json')):
    t = e['tag']
    ims = [cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
print('tta done')
PYEOF
echo "== dual-backbone projection"
CUDA_VISIBLE_DEVICES=0 $PY $W/proj_multi.py dump $DUMP repo199tta pAVt_l10 --nets alex:1,vgg:0.5 --lam 10 --steps 200 --batch 2 > $V/jobs/chain_${SLURM_JOB_NAME}_pAVt10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $W/proj_multi.py dump $DUMP repo199tta pAVt_l15 --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 > $V/jobs/chain_${SLURM_JOB_NAME}_pAVt15.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $W/proj_multi.py dump $DUMP repo199    pAV_l10  --nets alex:1,vgg:0.5 --lam 10 --steps 200 --batch 2 > $V/jobs/chain_${SLURM_JOB_NAME}_pAV10.log 2>&1 &
wait
echo "== composites"
i=0
for spec in "repo199:0.50:repo199:0.45:pb_p050_b045" \
            "repo199tta:0.50:pAVt_l10:1.0:pbt_p050_bpAVt10" \
            "repo199tta:0.50:pAVt_l15:1.0:pbt_p050_bpAVt15" \
            "repo199tta:0.60:pAVt_l10:1.0:pbt_p060_bpAVt10" \
            "repo199tta:0.70:pAVt_l10:1.0:pbt_p070_bpAVt10" \
            "repo199tta:0.60:repo199tta:0.55:pbt_p060_b055"; do
  IFS=: read -r PS PA BS BA NAME <<< "$spec"
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $DUMP $PS $PA $BS $BA $NAME > $V/jobs/chain_${SLURM_JOB_NAME}_$NAME.log 2>&1 &
  i=$((i+1))
done
wait
echo "== metrics"
$PY $W/metrics_both.py $DUMP render repo199 repo199tta pAV_l10 pAVt_l10 pAVt_l15 pb_p050_b045 \
  pbt_p050_bpAVt10 pbt_p050_bpAVt15 pbt_p060_bpAVt10 pbt_p070_bpAVt10 pbt_p060_b055 2>&1 | grep -vE "Warning|warn|load_state"
echo CHAIN_${SLURM_JOB_NAME}_DONE
