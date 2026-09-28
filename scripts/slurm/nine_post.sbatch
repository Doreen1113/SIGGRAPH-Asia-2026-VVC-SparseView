#!/bin/bash
#SBATCH --job-name=nine_post
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nine_post_%j.log
# Waits for the three Difix shift trees, then: TTA average -> projection shard -> composites at 4 person alphas -> zips.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
SRC=$V/submissions/NINE_ens/renders
echo "== waiting for 3 Difix shift trees, $(date -u +%H:%M) =="
until (for d in NINE_difix NINE_difix_s1 NINE_difix_s2; do
         [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
echo "== TTA average, $(date -u +%H:%M) =="
$PY $W/tta_average.py $V/submissions/NINE_TTA/renders $V/submissions/NINE_difix/renders \
  $V/submissions/NINE_difix_s1/renders $V/submissions/NINE_difix_s2/renders --procs 90
echo "== projection lam15, 8-way sharded, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $SRC $V/submissions/NINE_TTA/renders $V/submissions/NINE_pAVt15/renders \
    --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 4 --shard $i/8 > $V/jobs/ninepost_proj_$i.log 2>&1 &
done
wait
echo "== composites at 4 person alphas, $(date -u +%H:%M) =="
i=0
for PA in 0.50 0.60 0.70 0.80; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/NINE_TTA/renders $PA \
      $V/submissions/NINE_pAVt15/renders 1.0 $V/submissions/NINE_${TAG}_bpAVt15/renders --cases $c \
      > $V/jobs/ninepost_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 050 060 070 080; do
  OUT=NINE_p${PA}_bpAVt15
  n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo NINE_POST_DONE $(date -u +%H:%M)
