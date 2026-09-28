#!/bin/bash
#SBATCH --job-name=six_comp
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/six_comp_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
SRC=$V/submissions/SIX_ens/renders
echo "== waiting for projection tree, $(date -u +%H:%M) =="
until [ "$(find $V/submissions/SIX_pAVt15 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== composites at 4 person alphas, $(date -u +%H:%M) =="
i=0
for PA in 0.60 0.70 0.80; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/SIX_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/SIX_${TAG}_bpAVt15/renders --cases $c \
      > $V/jobs/sixcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 060 070 080; do
  OUT=SIX_p${PA}_bpAVt15
  n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo SIX_COMP_DONE $(date -u +%H:%M)
