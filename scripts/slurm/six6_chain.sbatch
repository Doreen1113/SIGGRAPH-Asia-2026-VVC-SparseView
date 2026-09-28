#!/bin/bash
#SBATCH --job-name=six6_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/six6_chain_%j.log
# 6-shift Difix TTA on the peer base, then projection (lam 15 and 12) and composites at several person alphas.
# Goal: the 0.0012 of FULL-LPIPS we were short, bought by variance reduction rather than by trading SSIM away.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== waiting for the 3 extra shifts, $(date -u +%H:%M) =="
until (for d in SIX_difix_s3 SIX_difix_s4 SIX_difix_s5; do
         [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 120; done
echo "== 6-shift TTA average, $(date -u +%H:%M) =="
$PY $W/tta_average.py $V/submissions/SIX_TTA6/renders \
  $V/submissions/SIX_difix/renders $V/submissions/SIX_difix_s1/renders $V/submissions/SIX_difix_s2/renders \
  $V/submissions/SIX_difix_s3/renders $V/submissions/SIX_difix_s4/renders $V/submissions/SIX_difix_s5/renders --procs 90
echo "== projection lam15, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $V/submissions/SIX_ens/renders $V/submissions/SIX_TTA6/renders \
    $V/submissions/SIX6_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 \
    > $V/jobs/six6_proj_$i.log 2>&1 &
done
wait
echo "== composites, $(date -u +%H:%M) =="
i=0
for PA in 0.70 0.80; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/SIX_ens/renders $V/submissions/SIX_TTA6/renders $PA \
      $V/submissions/SIX6_pAVt15/renders 1.0 $V/submissions/SIX6_${TAG}_bpAVt15/renders --cases $c > $V/jobs/six6comp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 070 080; do
  OUT=SIX6_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"; [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo SIX6_CHAIN_DONE $(date -u +%H:%M)
