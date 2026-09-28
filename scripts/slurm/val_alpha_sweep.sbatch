#!/bin/bash
#SBATCH --job-name=val_alpha
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_alpha_%j.log
# Which person alpha is best on the LARGEST val ensembles we have (e7/e8), using the production recipe
# (Difix 3-shift TTA for the person, dual-backbone projection for the background).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
for D in $V/fixtest_h200/001_1_seq0/e7 $V/fixtest_h200/001_1_seq0/e8; do
  i=0
  for PA in 0.40 0.70 0.80 0.90; do
    TAG=$(printf "pbt_p%03d_bpAVt15" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%4)) $PY $W/composite2.py dump $D repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done
  wait
  echo "=== $(basename $D) ==="
  $PY $W/metrics_both.py $D render pbt_p040_bpAVt15 pbt_p050_bpAVt15 pbt_p060_bpAVt15 pbt_p070_bpAVt15 pbt_p080_bpAVt15 pbt_p090_bpAVt15 2>&1 | grep -vE "Warning|warn|load_state"
done
echo VAL_ALPHA_DONE
