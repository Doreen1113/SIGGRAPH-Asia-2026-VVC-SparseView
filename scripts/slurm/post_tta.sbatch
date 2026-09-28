#!/bin/bash
#SBATCH --job-name=post_tta
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/post_tta_%j.log
PY=/home/doreen071/.conda/envs/vvc/bin/python; V=/work/doreen071/vvc
$PY $V/work/tta_average.py $V/submissions/SIX_TTA/renders $V/submissions/SIX_difix/renders $V/submissions/SIX_difix_s1/renders $V/submissions/SIX_difix_s2/renders --procs 90
SRC=$V/submissions/SIX_ens/renders; FIX=$V/submissions/SIX_TTA/renders; OUT=SIXTTA_p50_b45; DST=$V/submissions/$OUT/renders
export OMP_NUM_THREADS=8; i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$i $PY $V/work/person_bg_composite.py $SRC $FIX $DST --pa 0.50 --ba 0.45 --cases $c > $V/jobs/composite_${OUT}_$c.log 2>&1 &
  i=$((i+1))
done
wait
for c in $(ls $SRC); do echo "$c: $(find $DST/$c -name '*.jpg' | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
$PY $V/work/pack_submission.py $V/submissions/$OUT.zip $DST && echo POST_TTA_DONE
