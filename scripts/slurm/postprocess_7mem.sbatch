#!/bin/bash
#SBATCH --job-name=post7mem
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/post7mem_%j.log
# Full production chain on the 7-member ensemble: Difix x3 shifts (TTA) -> dual-backbone projection (lam15)
# -> person/bg composite (pa=0.60/ba=1.0-on-projection per the validated ens_ladder2 winner) -> pack.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
SRC=$V/submissions/SEVEN_ens/renders
export OMP_NUM_THREADS=10
echo "== difix 3 shifts (8-way sharded), $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=SEVEN_difix;; "320,288") vn=SEVEN_difix_s1;; *) vn=SEVEN_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} \
      > $V/jobs/post7_${vn}_shard$i.log 2>&1 &
  done
  wait
  echo "   $vn done $(date -u +%H:%M)"
done
echo "== TTA average, $(date -u +%H:%M) =="
$PY $W/tta_average.py $V/submissions/SEVEN_TTA/renders $V/submissions/SEVEN_difix/renders $V/submissions/SEVEN_difix_s1/renders $V/submissions/SEVEN_difix_s2/renders --procs 90
echo "== dual-backbone projection lam15 (8-way sharded), $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $SRC $V/submissions/SEVEN_TTA/renders $V/submissions/SEVEN_pAVt15/renders \
    --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 4 --shard $i/8 > $V/jobs/post7_proj_shard$i.log 2>&1 &
done
wait
echo "== person/bg composite pa=0.60 (TTA) / bg=1.0 (projection), $(date -u +%H:%M) =="
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $SRC $V/submissions/SEVEN_TTA/renders 0.60 $V/submissions/SEVEN_pAVt15/renders 1.0 \
    $V/submissions/SEVEN_p060_bpAVt15/renders --cases $c > $V/jobs/post7_composite_$c.log 2>&1 &
  i=$((i+1))
done
wait
for c in $(ls $SRC); do echo "$c: $(find $V/submissions/SEVEN_p060_bpAVt15/renders/$c -name '*.jpg' | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
$PY $W/pack_submission.py $V/submissions/SEVEN_p060_bpAVt15.zip $V/submissions/SEVEN_p060_bpAVt15/renders
echo POST7MEM_DONE $(date -u +%H:%M)
