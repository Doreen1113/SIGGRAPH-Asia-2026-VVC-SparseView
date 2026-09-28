#!/bin/bash
#SBATCH --job-name=six_difix
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/six_difix_%j.log
# SHIFT env: "" / "320,288" / "640,192" -> SIX_difix / SIX_difix_s1 / SIX_difix_s2
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHIFT="${SHIFT:-}"; SHIFT="${SHIFT//x/,}"
case "$SHIFT" in "") VN=SIX_difix;; "320,288") VN=SIX_difix_s1;; *) VN=SIX_difix_s2;; esac
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/SIX_ens/renders $V/submissions/$VN/renders ${SHIFT:+--shift $SHIFT} > $V/jobs/sixdifix_${VN}_$i.log 2>&1 &
done; wait
echo "$VN: $(find $V/submissions/$VN -name '*.jpg' | wc -l)/2056"
echo SIX_DIFIX_DONE $VN $(date +%H:%M)
