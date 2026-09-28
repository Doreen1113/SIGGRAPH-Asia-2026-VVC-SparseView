#!/bin/bash
#SBATCH --job-name=sweep_pa
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/sweep_pb_%j.log
PY=/home/doreen071/.conda/envs/vvc/bin/python
V=/work/doreen071/vvc; W=$V/work; F=$V/fixtest
export OMP_NUM_THREADS=10
# person source / person alpha / background source / background alpha / name
SPEC="repo199:0.50:repo199:0.45:pb_p050_b045
repo199tta:0.60:pAVt_l10:1.0:pbt_p060_bpAVt10
repo199tta:0.70:pAVt_l10:1.0:pbt_p070_bpAVt10
repo199tta:0.80:pAVt_l10:1.0:pbt_p080_bpAVt10
repo199tta:1.00:pAVt_l10:1.0:pbt_p100_bpAVt10
repo199tta:0.60:pAVt_l15:1.0:pbt_p060_bpAVt15
repo199tta:0.70:pAVt_l15:1.0:pbt_p070_bpAVt15
repo199tta:0.80:pAVt_l15:1.0:pbt_p080_bpAVt15
repo199tta:0.70:repo199tta:0.55:pbt_p070_b055
repo199tta:0.80:repo199tta:0.60:pbt_p080_b060
repo199:0.70:pAVt_l10:1.0:pbs_p070_bpAVt10
repo199tta:0.65:pAVt_l10:1.0:pbt_p065_bpAVt10
repo199tta:0.75:pAVt_l10:1.0:pbt_p075_bpAVt10"
for S in 001_1 012_0; do
  i=0
  while IFS=: read -r PS PA BS BA NAME; do
    [ -z "$NAME" ] && continue
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $F/$S $PS $PA $BS $BA $NAME > $V/jobs/sweep_${S}_${NAME}.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done <<< "$SPEC"
  wait
done
NAMES=$(echo "$SPEC" | cut -d: -f5 | tr '\n' ' ')
for S in 001_1 012_0; do
  $PY $W/metrics_both.py $F/$S pb_p060_b055 pAVt_l10 pbt_p050_bpAVt10 pbt_p050_bpAVt15 $NAMES 2>&1 | grep -vE "Warning|warn|load_state" > $V/jobs/sweepmetricsB_$S.log
done
$PY $W/rank_predict.py $V/jobs/sweepmetricsB_001_1.log $V/jobs/sweepmetricsB_012_0.log --lpips alex > $V/jobs/rankB_alex.log 2>&1
$PY $W/rank_predict.py $V/jobs/sweepmetricsB_001_1.log $V/jobs/sweepmetricsB_012_0.log --lpips vgg  > $V/jobs/rankB_vgg.log 2>&1
echo SWEEP_PB_DONE
