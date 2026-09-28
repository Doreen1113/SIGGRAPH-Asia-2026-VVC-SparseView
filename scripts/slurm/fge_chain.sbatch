#!/bin/bash
#SBATCH --job-name=fge_chain
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge_chain_%j.log
# person <- Difix 3-shift TTA of the FG-specialist ensemble (alpha 0.70 / 0.85 over the raw ensemble),
# background <- SIX_pAVt15 (unchanged from the current best package).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until (for d in FGE_difix FGE_difix_s1 FGE_difix_s2; do [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
echo "== TTA average $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/FGE_TTA/renders $V/submissions/FGE_difix/renders $V/submissions/FGE_difix_s1/renders $V/submissions/FGE_difix_s2/renders --procs 90
SRC=$V/submissions/FGE_ens/renders
echo "== composites $(date +%H:%M) =="
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/FGE_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGE_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fgecomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=FGE_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"; [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo FGE_CHAIN_DONE $(date +%H:%M)
