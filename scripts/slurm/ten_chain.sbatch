#!/bin/bash
#SBATCH --job-name=ten_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/ten_chain_%j.log
# Waits for TEN_ens, then Difix x3 (sequential on this node) -> TTA -> projection -> composites -> zips.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=$V/submissions/TEN_ens/renders
echo "== waiting for TEN_ens, $(date -u +%H:%M) =="
until [ "$(find $SRC -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== difix 3 shifts, $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=TEN_difix;; "320,288") vn=TEN_difix_s1;; *) vn=TEN_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} \
      > $V/jobs/ten_${vn}_$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/TEN_TTA/renders $V/submissions/TEN_difix/renders \
  $V/submissions/TEN_difix_s1/renders $V/submissions/TEN_difix_s2/renders --procs 90
echo "== projection, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $SRC $V/submissions/TEN_TTA/renders $V/submissions/TEN_pAVt15/renders \
    --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 > $V/jobs/ten_proj_$i.log 2>&1 &
done
wait
echo "== composites, $(date -u +%H:%M) =="
i=0
for PA in 0.50 0.60 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/TEN_TTA/renders $PA \
      $V/submissions/TEN_pAVt15/renders 1.0 $V/submissions/TEN_${TAG}_bpAVt15/renders --cases $c \
      > $V/jobs/tencomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 050 060 070; do
  OUT=TEN_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo TEN_CHAIN_DONE $(date -u +%H:%M)
