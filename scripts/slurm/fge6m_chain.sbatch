#!/bin/bash
#SBATCH --job-name=fge6m_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge6m_chain_%j.log
# FGE_p085 measured on the real board: FULL 1.333 (LPIPS rank 1) but FG-SSIM fell 0.0049 to rank 6 because the
# person region contained ONLY our FG models. Hedge: person source = 50/50 mean of the 6-member FG ensemble and
# the peer SIX_ens, which should restore FG-SSIM while keeping most of the -0.021 FG-LPIPS gain.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== waiting for FGE6_ens, $(date +%H:%M) =="
until [ "$(find $V/submissions/FGE6_ens -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== person source = FGE6_ens:1 + SIX_ens:1, $(date +%H:%M) =="
$PY $W/weighted_ensemble.py $V/submissions/FGE6M_ens/renders $V/submissions/FGE6_ens/renders:1 $V/submissions/SIX_ens/renders:1 --procs 90
SRC=$V/submissions/FGE6M_ens/renders
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGE6M_difix;; "320,288") vn=FGE6M_difix_s1;; *) vn=FGE6M_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fge6m_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/FGE6M_TTA/renders $V/submissions/FGE6M_difix/renders $V/submissions/FGE6M_difix_s1/renders $V/submissions/FGE6M_difix_s2/renders --procs 90
echo "== composites, $(date +%H:%M) =="
i=0
for PA in 0.85; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/FGE6M_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGE6M_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fge6mcomp_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
OUT=FGE6M_p085_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
echo "-- $OUT: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
echo FGE6M_CHAIN_DONE $(date +%H:%M)
