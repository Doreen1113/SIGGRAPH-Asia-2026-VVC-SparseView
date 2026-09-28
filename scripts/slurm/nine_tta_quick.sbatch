#!/bin/bash
#SBATCH --job-name=nine_quick
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:5
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/nine_quick_%j.log
# Calibration submission: identical recipe to the already-scored SIXTTA (TTA + person 0.50 / bg 0.45),
# only the base ensemble changes 6 -> 9 members. Isolates the ensemble effect against real test scores.
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
SRC=$V/submissions/NINE_ens/renders; FIX=$V/submissions/NINE_TTA/renders
OUT=$V/submissions/NINE_p50_b45/renders
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$i $PY $V/work/person_bg_composite.py $SRC $FIX $OUT --pa 0.50 --ba 0.45 --cases $c \
    > $V/jobs/ninequick_$c.log 2>&1 &
  i=$((i+1))
done
wait
n=$(find $OUT -name '*.jpg' | wc -l); echo "composited $n/2056"
[ "$n" -eq 2056 ] && $PY $V/work/pack_submission.py $V/submissions/NINE_p50_b45.zip $OUT
echo NINE_QUICK_DONE $(date -u +%H:%M)
